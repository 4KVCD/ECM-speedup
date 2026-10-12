/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2023, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     GpuIntraRec.cpp
    \brief    intra reconstruction of a picture on the GPU (verification of the recorded CPU reconstruction)
*/

#include "GpuIntraRec.h"
#include "VkCompute.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "CommonLib/CodingStructure.h"
#include "CommonLib/BilateralFilter.h"
#include "CommonLib/DepQuant.h"
#include "CommonLib/InterpolationFilter.h"
#include "CommonLib/MipData.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Quant.h"
#include "CommonLib/Reshape.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Slice.h"
#include "CommonLib/TrQuant.h"
#include "CommonLib/UnitTools.h"

#include "intra_rec.spv.h"
#include "intra_resi.spv.h"
#include "intra_deps.spv.h"
#include "scan.spv.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace
{
// layout shared with intra_rec.comp (struct Tu): 96 int32
struct Tu
{
  int32_t comp, x, y, w, h;
  int32_t dirMode;
  int32_t isModeVer, intraPredAngle, absInvAngle, useCubic;
  int32_t unitW, unitH;
  int32_t bitDepth;
  int32_t cbf, coeffOffset;
  int32_t dqScale, dqRightShift, dqInMin, dqInMax;
  int32_t trMin, trMax, shift1, shift2;
  int32_t recMin, recMax;
  int32_t dbgOffset;
  int32_t plIdx;
  int32_t refFilter, applyPDPC, gradPDPC, angularScale, fusion, disableRefFilter;
  int32_t avail[2][7];
  int32_t readMax[2][2];
  int32_t nzRows, nzCols;
  int32_t bf;
  int32_t qp;
  int32_t lmcs;
  int32_t crsX, crsY, crsW, crsH;
  int32_t crsAvail;
  int32_t quant;
  int32_t jccr, jccrOffset;
  int32_t mtxV, mtxH;
  int32_t transformSkip;
  int32_t mipMatrix, mipTransposed, mipSizeId;
  int32_t secKind, secMatrix, secRowLength, secNumIn, secScan, secLayout, secSize, secTranspose;
  int32_t mrl;
  int32_t bfAvail[3];
  int32_t isp;
  int32_t cuX, cuY, cuW, cuH;
  int32_t regX, regW;
  int32_t ispSide;
  int32_t ispPrev;
  int32_t spCoded, spOffset;
  int32_t resiFrom;
  int32_t cu;
  int32_t derive;
};
static_assert( sizeof( Tu ) == 96 * 4, "Tu layout" );

// Tu::derive (intra_common.glsl's DERIVE_*)
constexpr int DERIVE_NODE         = 32;   // not a block: the derivation of the mode of CU Tu::cu, before its blocks
constexpr int DERIVE_MODE         = 1;    // intra_rec takes the mode from the CU's node and sets its parameters
constexpr int DERIVE_FETCH2ND     = 2;    // IntraPrediction's fetchRef2nd: intra fusion may read the second line
constexpr int DERIVE_BF           = 4;    // the intra bilateral filter applies unless the mode excludes it
constexpr int DERIVE_NO_PDPC      = 8;    // the SPS disables PDPC
constexpr int DERIVE_NO_SMOOTHING = 16;   // the SPS disables intra smoothing

// the parameters intra_rec derives per DERIVE_MODE block, downloaded for the comparison (intra_common.glsl's
// DBG_PARAMS): dirMode, isModeVer, intraPredAngle, absInvAngle, useCubic, refFilter, applyPDPC, gradPDPC, angularScale,
// fusion, bf, plIdx
constexpr int DBG_PARAMS = 12;

// getSpatialIpm's positions: left, above, above-left, above-right, below-left, then the non-adjacent ones
constexpr int MPM_NEIGHBOURS = 23;

// a coding unit of one channel type (intra_common.glsl's Cu): 40 int32
struct Cu
{
  int32_t kind;
  int32_t node;
  int32_t x, y, w, h;
  int32_t mpmFlag, secondMpmFlag, ipredIdx;
  int32_t nbFirst, nbLast;
  int32_t plIdx;
  int32_t candId;
  int32_t lumaCu;
  int32_t nbr[MPM_NEIGHBOURS];
  int32_t modes[2];
  int32_t pad;
};
static_assert( sizeof( Cu ) == 40 * 4, "Cu layout" );

// Cu::kind
constexpr int CU_KNOWN       = 0;   // the host knows the mode
constexpr int CU_MPM         = 1;   // luma: from the MPM lists (intra_modes.glsl)
constexpr int CU_CHROMA_DM   = 2;   // chroma: the co-located luma mode
constexpr int CU_CHROMA_CAND = 3;   // chroma: one of the first four candidates

// intra_modes.glsl's applyModeParams (the kernels' copy): a block's prediction parameters for a mode as
// initPredIntraParams sets them without TIMD / TMRL, with intra fusion, the directional planar index and the intra
// bilateral filter's mode conditions; here for the read ranges of a derived mode's possible values (endPicture)
void applyModeParams( Tu& t, const int mode, const int plIdx )
{
  static const int angTable[32]    = { 0,  1,  2,  3,  4,  6,  8,  10,  12,  14,  16,  18,  20,  23,  26,  29,
                                       32, 35, 39, 45, 51, 57, 64, 73, 86, 102, 128, 171, 256, 341, 512, 1024 };
  static const int invAngTable[32] = { 0,   16384, 8192, 5461, 4096, 2731, 2048, 1638, 1365, 1170, 1024, 910, 819, 712, 630, 565,
                                       512, 468,   420,  364,  321,  287,  256,  224,  191,  161,  128,  96,  64,  48,  32,  16 };
  const bool luma = t.comp == COMPONENT_Y;
  const bool isp  = t.isp && luma;
  const int  puW = t.regW, puH = t.h;
  const int  predMode = IntraPrediction::wideAngle( isp ? t.cuW : puW, isp ? t.cuH : puH, mode );
  t.dirMode        = mode;
  t.isModeVer      = predMode >= DIA_IDX ? 1 : 0;
  t.intraPredAngle = 0;
  t.absInvAngle    = 0;
  t.angularScale   = 0;
  t.gradPDPC       = 0;
  bool pdpc = puW >= 4 && puH >= 4 && t.mrl == 0;
  const int angMode = t.isModeVer ? predMode - VER_IDX : -( predMode - HOR_IDX );
  int absAng = 0;
  if( mode > DC_IDX && mode < NUM_LUMA_MODE )
  {
    absAng           = angTable[std::abs( angMode )];
    t.absInvAngle    = invAngTable[std::abs( angMode )];
    t.intraPredAngle = angMode < 0 ? -absAng : absAng;
    if( angMode < 0 )
    {
      pdpc = false;
    }
    else if( angMode > 0 )
    {
      t.angularScale = std::min( 2, floorLog2( t.isModeVer ? puH : puW ) - ( floorLog2( 3 * t.absInvAngle - 2 ) - 8 ) );
      if( t.angularScale < 0 && luma )
      {
        t.angularScale = ( floorLog2( puW ) + floorLog2( puH ) - 2 ) >> 2;
        t.gradPDPC     = 1;
      }
      pdpc = pdpc && t.angularScale >= 0;
    }
  }
  if( t.derive & DERIVE_NO_PDPC )
  {
    pdpc       = false;
    t.gradPDPC = 0;
  }
  int refFilter = 0, interpolation = 0;
  if( ( t.derive & DERIVE_NO_SMOOTHING ) || !luma || isp || t.mrl || mode == DC_IDX )
  {
  }
  else if( mode == PLANAR_IDX )
  {
    refFilter = puW * puH > 32 ? 1 : 0;
  }
  else
  {
    const int diff     = std::min( std::abs( predMode - HOR_IDX ), std::abs( predMode - VER_IDX ) );
    const int log2Size = ( floorLog2( puW ) + floorLog2( puH ) ) >> 1;
    if( diff > IntraPrediction::intraFilterThreshold( log2Size ) && puW * puH > 32 )
    {
      const bool intSlope = ( absAng & 31 ) == 0;
      refFilter     = intSlope ? 1 : 0;
      interpolation = intSlope ? 0 : 1;
    }
  }
  t.refFilter = t.disableRefFilter ? 0 : refFilter;
  t.useCubic  = interpolation ? 0 : 1;
  t.applyPDPC = pdpc ? 1 : 0;
  t.fusion    = ( t.derive & DERIVE_FETCH2ND ) && mode > DC_IDX && ( absAng & 31 ) != 0 && !isp ? 1 : 0;
  t.plIdx     = luma && mode == PLANAR_IDX ? plIdx : 0;
  t.bf        = 0;
  if( ( t.derive & DERIVE_BF ) && !( ( t.mrl || isp ) && mode > DC_IDX ) )
  {
    const bool excluded = mode > DC_IDX && ( ( mode >= 2 && mode <= 3 ) || ( mode >= 17 && mode <= 19 ) || ( mode >= 33 && mode <= 35 )
                                             || ( mode >= 49 && mode <= 51 ) || ( mode >= 65 && mode <= 66 ) );
    t.bf = excluded ? 0 : mode <= DC_IDX ? 1 : 2;
  }
}

// ECM_GPU_MODES=1: the GPU derives the intra modes of the CUs whose modes the MPM lists give (and of the chroma CUs
// with DM or a substituted candidate) also where the host knows them; tests the derivation on every stream
bool gpuDerivesModes()
{
  static const bool on = getenv( "ECM_GPU_MODES" ) && atoi( getenv( "ECM_GPU_MODES" ) ) != 0;
  return on;
}

// the signPred data of a block (intra_common.glsl's SP_*): the number of predicted signs, the coded signs, the
// positions, then the templates intra_resi computes
constexpr int SP_TEMPLATES = 10;
static_assert( SP_TEMPLATES == 2 + SIGN_PRED_MAX_NUM, "signPred layout" );

// Tu::lmcs
constexpr int LMCS_BF_INVERSE  = 1;   // the intra bilateral filter works on inverse-mapped samples
constexpr int LMCS_CHROMA_SCALE = 2;   // the chroma residual is scaled by the CU's neighbouring luma

// Tu::quant
constexpr int QUANT_DEP = 1;   // dependent quantization: the coefficients are quantization indices (DepQuant)
constexpr int QUANT_QCS = 2;   // quantization center shift (JVET-AN0095 / JVET-AE0125)

// Tu::jccr (joint coding of the chroma residuals): the Cb block of a TU carries the ICT mode (TU::getICTMode, +-1..+-3)
// and computes both residuals; its Cr block, recorded right after it, is marked JCCR_DERIVED
constexpr int JCCR_DERIVED = 4;

// Tu::secKind (intra_common.glsl's SEC_*): the inverse LFNST before the primary transform, or NSPT instead of it
constexpr int SEC_NONE  = 0;
constexpr int SEC_LFNST = 1;
constexpr int SEC_NSPT  = 2;

// the inverse LFNST or NSPT the CPU applies to a block, with its choices (TrQuant::xIT's xGetInvTrParams and
// CU::nsptApplyCond, xDeQuantLfnst / xInvLfnst / invLfnstNxN with the extended kernels, xInvNspt / getNsptMatrix): the
// matrix takes numIn coefficients in the order of scan and gives rowLength outputs per input (output j of input i at
// matrix[i * rowLength + j])
struct Secondary
{
  int                kind      = SEC_NONE;
  const int8_t*      matrix    = nullptr;
  int                rowLength = 0;
  int                numIn     = 0;
  const ScanElement* scan      = nullptr;
  int                sbSize    = 0;       // LFNST: its outputs lie in the top-left sbSize x sbSize
  bool               transpose = false;
};

// false where the block's LFNST is one the GPU path does not have (the VVC kernels without the extension)
bool invSecondary( const TransformUnit& tu, const ComponentID compID, Secondary& s )
{
  s = Secondary();
  const CodingUnit& cu   = *tu.cu;
  const SPS&        sps  = *tu.cs->sps;
  const CompArea&   area = tu.blocksResidual[compID];
  const int         W = area.width, H = area.height;
  if( !cu.lfnstIdx || W < 2 || H < 2 )   // NSPT is in xIT's 2-D path only; LFNST is not signalled there
  {
    return true;
  }
  const bool spsIntraLfnstEnabled = ( cu.slice->getSliceType() == I_SLICE && sps.getUseIntraLFNSTISlice() )
                                    || ( cu.slice->getSliceType() != I_SLICE && sps.getUseIntraLFNSTPBSlice() );
  const bool allowNSPT = CU::isNSPTAllowed( tu, compID, W, H, spsIntraLfnstEnabled && CU::isIntra( cu ) );
  bool       secondBucket = false;
  if( CU::nsptApplyCond( tu, compID, allowNSPT ) )
  {
    const uint32_t intraMode = PU::getFinalIntraModeForTransform( secondBucket, tu, compID ).first;
    s.transpose  = PU::getTransposeFlag( intraMode );
    const int tw = s.transpose ? H : W, th = s.transpose ? W : H;   // the kernel's block
    s.kind       = SEC_NSPT;
    s.numIn      = PU::getNSPTMatrixDim( tw, th );
    s.matrix     = TrQuant::getNsptMatrix( g_nsptLut[intraMode], tw, th, cu.lfnstIdx - 1, PU::getNSPTBucket( tu, secondBucket, compID ) );
    s.rowLength  = tw * th;
    s.scan       = g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][gp_sizeIdxInfo->idxFrom( W )][gp_sizeIdxInfo->idxFrom( H )];
    return true;
  }
  const bool lfnstOn = ( spsIntraLfnstEnabled && CU::isIntra( cu ) ) || ( sps.getUseInterLFNST() && CU::isInter( cu ) );
  if( !lfnstOn || tu.mtsIdx[compID] == MTS_SKIP || !( cu.separateTree || isLuma( compID ) ) || allowNSPT )
  {
    return true;
  }
  const bool ext = sps.getUseLFNSTExt();
  if( cu.lfnstIdx >= ( ext ? 4 : 3 ) )
  {
    return true;
  }
  if( !ext )
  {
    return false;
  }
  const uint32_t intraMode = PU::getFinalIntraModeForTransform( secondBucket, tu, compID ).first;
  const bool     whge4 = PU::getUseLFNST16( W, H ), whge3 = PU::getUseLFNST8( W, H );
  const int      widthIdx = gp_sizeIdxInfo->idxFrom( W ), heightIdx = gp_sizeIdxInfo->idxFrom( H );
  const int      mode = g_lfnstLut[intraMode], index = cu.lfnstIdx - 1;
  s.kind      = SEC_LFNST;
  s.transpose = PU::getTransposeFlag( intraMode );
  s.sbSize    = whge4 ? 16 : whge3 ? 8 : 4;
  s.scan      = whge4 ? g_coefTopLeftDiagScan16x16[widthIdx]
                      : whge3 ? g_coefTopLeftDiagScan8x8[widthIdx] : g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][widthIdx][heightIdx];
  s.numIn     = PU::getLFNSTMatrixDim( W, H, ext );   // zeroOutSize: the inputs invLfnstNxN uses
  s.matrix    = s.sbSize > 8 ? &g_lfnst16x16[mode][index][0][0] : s.sbSize > 4 ? &g_lfnst8x8[mode][index][0][0] : &g_lfnst4x4[mode][index][0][0];
  s.rowLength = s.sbSize > 8 ? L16W_ZO : s.sbSize > 4 ? L8W_ZO : 16;
  return true;
}

// xInvLfnst's arrangement of the extended LFNST's outputs in the top-left sbSize x sbSize of the coefficients: the
// position x | y << 16 of each output, found by replaying its copies (with the CPU's row stride sbSize) on the outputs'
// numbers
std::vector<int32_t> lfnstOutputLayout( const int sbSize, const bool transpose )
{
  const int            numOut = sbSize > 8 ? L16W_ZO : sbSize > 4 ? L8W_ZO : 16, width = sbSize;
  std::vector<int32_t> out( numOut ), region( sbSize * sbSize, -1 );
  for( int j = 0; j < numOut; j++ )
  {
    out[j] = j;
  }
  const int32_t* lfnstTemp = out.data();
  int32_t*       coeffTemp = region.data();
  if( transpose )
  {
    if( sbSize == 4 )
    {
      for( int y = 0; y < 4; y++, lfnstTemp++, coeffTemp += width )
      {
        coeffTemp[0] = lfnstTemp[0]; coeffTemp[1] = lfnstTemp[4]; coeffTemp[2] = lfnstTemp[8]; coeffTemp[3] = lfnstTemp[12];
      }
    }
    else if( sbSize == 8 )
    {
      for( int y = 0; y < 8; y++, lfnstTemp++, coeffTemp += width )
      {
        for( int x = 0; x < 8; x++ ) coeffTemp[x] = lfnstTemp[8 * x];
      }
    }
    else
    {
      for( int y = 0; y < 12; y++, lfnstTemp++, coeffTemp += width )
      {
        coeffTemp[0] = lfnstTemp[0]; coeffTemp[1] = lfnstTemp[12]; coeffTemp[2] = lfnstTemp[24]; coeffTemp[3] = lfnstTemp[36];
        if( y < 8 )
        {
          coeffTemp[4] = lfnstTemp[48]; coeffTemp[5] = lfnstTemp[56]; coeffTemp[6] = lfnstTemp[64]; coeffTemp[7] = lfnstTemp[72];
        }
        if( y < 4 )
        {
          coeffTemp[8] = lfnstTemp[80]; coeffTemp[9] = lfnstTemp[84]; coeffTemp[10] = lfnstTemp[88]; coeffTemp[11] = lfnstTemp[92];
        }
      }
    }
  }
  else
  {
    for( int y = 0; y < ( sbSize == 16 ? 12 : sbSize ); y++, coeffTemp += width )
    {
      const int stride = sbSize == 16 ? ( y < 4 ? 12 : y < 8 ? 8 : 4 ) : sbSize;
      std::copy_n( lfnstTemp, stride, coeffTemp );
      lfnstTemp += stride;
    }
  }
  std::vector<int32_t> pos( numOut, -1 );
  for( int y = 0; y < sbSize; y++ )
  {
    for( int x = 0; x < sbSize; x++ )
    {
      if( region[y * sbSize + x] >= 0 )
      {
        pos[region[y * sbSize + x]] = x | ( y << 16 );
      }
    }
  }
  CHECK( std::find( pos.begin(), pos.end(), -1 ) != pos.end(), "an LFNST output without a position" );
  return pos;
}

// xFillReferenceSamples' rule (the kernels' copy is intra_common.glsl's fillSource): the reconstructed sample (sx, sy)
// whose value entry idx of the top or left arm of a reference line takes, for the area (fx, fy, fw, fh) filled with
// arms of predSize / predHSize samples from line mrl (multiRefIdx), given the availability av: the corner flag and the
// available prefix of the above, above-right, left, below-left, left MRL and above MRL units. False where no unit is
// available (the mid-level value). The units in the CPU's order: below-left and left (bottom-up), left MRL, the
// corner C, above MRL, above, above-right; a line beyond the first unit has lM MRL units left of and aM above the
// corner unit, which keeps entries 0 .. cLen of both arms (JVET-Y0116). An unavailable unit takes the last sample of
// the unit before it in that order, units before the first available one take its first sample.
bool fillSource( int fx, int fy, int fw, int fh, int predSize, int predHSize, int mrl, int uW, int uH, const int32_t av[7],
                 bool top, int idx, int& sx, int& sy )
{
  const int tAU = predSize / uW, tLU = predHSize / uH;
  const int lM = mrl / uH, aM = mrl / uW, cLen = mrl - lM * uH, C = tLU + lM;
  const int totalUnits = tAU + tLU + 1 + lM + aM;
  const int numAbove = std::max( fw / uW, 1 ), numLeft = std::max( fh / uH, 1 );
  const int x0 = fx - 1 - mrl, y0 = fy - 1 - mrl;
  const int corner = av[0], pA = av[1], pAR = av[2], pL = av[3], pBL = av[4], pLM = av[5], pAM = av[6];
  // the available intervals [start, end] in unit order (empty when the prefix is 0); the MRL groups are scanned from
  // the corner on
  const int blS = tLU - numLeft - pBL, blE = tLU - 1 - numLeft, lS = tLU - pL, lE = tLU - 1;
  const int lmS = C - pLM, lmE = C - 1, amS = C + 1, amE = C + pAM;
  const int aS = C + 1 + aM, aE = C + aM + pA, arS = C + 1 + aM + numAbove, arE = C + aM + numAbove + pAR;
  int firstAvail = totalUnits;
  if( pAR > 0 ) firstAvail = arS;
  if( pA > 0 ) firstAvail = aS;
  if( pAM > 0 ) firstAvail = amS;
  if( corner ) firstAvail = C;
  if( pLM > 0 ) firstAvail = lmS;
  if( pL > 0 ) firstAvail = lS;
  if( pBL > 0 ) firstAvail = blS;
  if( firstAvail >= totalUnits )
  {
    return false;
  }
  const int u = idx <= cLen ? C : top ? C + 1 + ( idx - 1 - cLen ) / uW : C - 1 - ( idx - 1 - cLen ) / uH;
  int pm = -1;   // the nearest available unit at or before u
  if( pBL > 0 && blS <= u ) pm = std::max( pm, std::min( u, blE ) );
  if( pL > 0 && lS <= u ) pm = std::max( pm, std::min( u, lE ) );
  if( pLM > 0 && lmS <= u ) pm = std::max( pm, std::min( u, lmE ) );
  if( corner && C <= u ) pm = std::max( pm, C );
  if( pAM > 0 && amS <= u ) pm = std::max( pm, std::min( u, amE ) );
  if( pA > 0 && aS <= u ) pm = std::max( pm, std::min( u, aE ) );
  if( pAR > 0 && arS <= u ) pm = std::max( pm, std::min( u, arE ) );
  if( pm == u )   // available: its own sample
  {
    sx = top ? x0 + idx : x0;
    sy = top ? y0 : y0 + idx;
    return true;
  }
  // the last sample of the unit before (in unit order), or the first sample of the first available unit
  const bool before = pm >= 0;
  const int  s      = before ? pm : firstAvail;
  if( s == C )
  {
    sx = before ? x0 + cLen : x0;
    sy = before ? y0 : y0 + cLen;
  }
  else if( s > C )
  {
    sx = x0 + ( before ? ( s - C ) * uW + cLen : ( s - C - 1 ) * uW + 1 + cLen );
    sy = y0;
  }
  else
  {
    sx = x0;
    sy = y0 + ( before ? ( C - s - 1 ) * uH + 1 + cLen : ( C - s ) * uH + cLen );
  }
  return true;
}

// the sample entry idx of the top or left arm of reference line ln of block t takes (false: the mid-level value; the
// kernels' copy is intra_common.glsl's refSource): xFillReferenceSamples at the block with arms of 8x its size, or for
// an ISP sub-partition initIntraPatternChTypeISP: the first one fills the CU's references with arms of twice the CU's
// size; a later one takes the CU's arm along the split shifted by its offset in the CU (the CPU shifts its buffer by
// each sub-partition) where that side of it is available (Tu::ispSide; uniform along the CU, whose left column / above
// row is decoded before it), else the first sample of the row / column it borders, and the other arm from the row above
// / column left of it (the previous sub-partition), its last sample repeated; the corner is the shifted arm's entry 0
bool refSource( const Tu& t, int ln, bool top, int idx, int& sx, int& sy )
{
  if( !t.isp )
  {
    return fillSource( t.x, t.y, t.w, t.h, t.w << 3, t.h << 3, t.mrl + ln, t.unitW, t.unitH, t.avail[ln], top, idx, sx, sy );
  }
  const bool hor = t.isp == HOR_INTRA_SUBPARTITIONS;
  const int  off = hor ? t.y - t.cuY : t.regX - t.cuX;
  auto cuFill = [&]( bool cuTop, int cuIdx )
  {
    return fillSource( t.cuX, t.cuY, t.cuW, t.cuH, 2 * t.cuW, 2 * t.cuH, 0, t.unitW, t.unitH, t.avail[0], cuTop, cuIdx, sx, sy );
  };
  if( off == 0 )
  {
    return cuFill( top, idx );
  }
  const bool shiftedArm = hor ? !top : top;
  if( shiftedArm || idx == 0 )
  {
    if( t.ispSide )
    {
      return cuFill( !hor, ( shiftedArm ? idx : 0 ) + off );
    }
    sx = hor ? t.regX : t.regX - 1;
    sy = hor ? t.y - 1 : t.y;
    return true;
  }
  const int j = std::min( idx, hor ? t.regW : t.h ) - 1;
  sx = hor ? t.regX + j : t.regX - 1;
  sy = hor ? t.y - 1 : t.y + j;
  return true;
}

// layout shared with intra_rec.comp (struct Params, std430)
struct Params
{
  uint64_t tus;
  uint64_t coeffs;
  uint64_t plane[3];
  uint64_t owner[3];
  uint64_t filters;
  uint64_t matrices;
  uint64_t dbgPred;
  uint64_t dbgResi;
  uint64_t depCount;
  uint64_t depOffset;
  uint64_t depList;
  uint64_t queue;
  uint64_t counter;
  uint64_t depStart;
  uint64_t depTmp;
  uint64_t numDep;
  uint64_t lmcs;
  uint64_t signPred;
  uint64_t secKernels;
  uint64_t cus;
  uint64_t cuMode;
  uint64_t dbgParams;
  int32_t  origin[3];
  int32_t  stride[3];
  int32_t  width[3];
  int32_t  height[3];
  int32_t  ownerW[2];
  int32_t  ownerH[2];
  int32_t  unitLog2X[2];
  int32_t  unitLog2Y[2];
  int32_t  edgeCap;
  int32_t  lmcsLutSize;
  int32_t  lumaClipMin, lumaClipMax;
  int32_t  qcsOffset;
  int32_t  pad;
};
static_assert( offsetof( Params, origin ) == 208 && offsetof( Params, edgeCap ) == 288 && sizeof( Params ) == 312,
               "Params layout" );

struct Push
{
  uint64_t params;
  int32_t  numTus;
  int32_t  verify;   // intra_deps: the pass
};

struct PushScan
{
  uint64_t src, dst;
  int32_t  n;
  int32_t  pad;
};

constexpr int FILTER_TAPS[5] = { 6, 8, 10, 12, 16 };
constexpr int RESI_GROUPS    = 256;   // persistent workgroups of intra_resi

// the kernels and the constant tables, created once and shared by every instance (the decoder contexts of the frame
// threads); destroyed by GpuIntraRec::releaseShared, before the device
struct Shared
{
  gpu::Kernel                      kernel, kernelResi, kernelDeps, kernelScan;
  gpu::Buffer                      filters, matrices;
  gpu::Buffer                      secKernels;                    // int8: the CPU's LFNST and NSPT tables, one after the other
  int32_t                          trOffset[NUM_TRANS_TYPE][7];   // of the inverse transform matrices of sizes 2..128 in matrices (-1: none)
  int32_t                          qcsOffset = 0;                 // of the QCS weights in matrices
  int32_t                          mipOffset[3] = {};             // of the MIP matrices of each size class in matrices
  int32_t                          lfnstLayout[3][2] = {};        // of the LFNST output positions in matrices (sbSize 4, 8, 16; transposed)
  std::vector<std::pair<const int8_t*, size_t>>      secTables;   // each table's start and bytes, in secKernels' order
  std::vector<std::pair<const ScanElement*, int32_t>> secScans;  // the scans LFNST / NSPT read, their positions' offset in matrices
  std::unique_ptr<BilateralFilter> bif;   // the intra prediction bilateral filter's tables

  // the offset in secKernels of a matrix of the CPU's LFNST / NSPT tables (-1: not in them)
  int32_t kernelOffset( const int8_t* matrix ) const
  {
    size_t offset = 0;
    for( const auto& table: secTables )
    {
      if( matrix >= table.first && matrix < table.first + table.second )
      {
        return int32_t( offset + ( matrix - table.first ) );
      }
      offset += table.second;
    }
    return -1;
  }
  // the offset in matrices of the positions (x | y << 16) of a scan LFNST / NSPT read (-1: not one of them)
  int32_t scanOffset( const ScanElement* scan ) const
  {
    for( const auto& s: secScans )
    {
      if( s.first == scan )
      {
        return s.second;
      }
    }
    return -1;
  }
};

Shared* createShared( gpu::Context& ctx )
{
  Shared* sh = new Shared;
  sh->kernel.create( ctx, spv_intra_rec, sizeof( spv_intra_rec ), sizeof( Push ), "intra_rec" );
  sh->kernelResi.create( ctx, spv_intra_resi, sizeof( spv_intra_resi ), sizeof( Push ), "intra_resi" );
  sh->kernelDeps.create( ctx, spv_intra_deps, sizeof( spv_intra_deps ), sizeof( Push ), "intra_deps" );
  sh->kernelScan.create( ctx, spv_scan, sizeof( spv_scan ), sizeof( PushScan ), "scan" );

  // tables, from the C++ tables themselves
  sh->bif.reset( new BilateralFilter );   // BilateralFilter::create derives the LUTs
  sh->bif->create();
  std::vector<int32_t> filt( 5 * 32 * 16 + 32 * 4 + 26 * 48 + 64 + 16, 0 );
  for( int t = 0; t < 5; t++ )
  {
    for( int frac = 0; frac < 32; frac++ )
    {
      const TFilterCoeff* f = InterpolationFilter::getIntraNtapNonSmoothingFilter( false, frac, FILTER_TAPS[t] );
      for( int i = 0; i < FILTER_TAPS[t]; i++ )
      {
        filt[( t * 32 + frac ) * 16 + i] = f[i];
      }
    }
  }
  for( int frac = 0; frac < 32; frac++ )   // weak 4-tap filter of the negative-angle reference extension
  {
    const TFilterCoeff* f = InterpolationFilter::getWeak4TapFilterTable( frac );
    for( int i = 0; i < 4; i++ )
    {
      filt[5 * 32 * 16 + frac * 4 + i] = f[i];
    }
  }
  {
    int32_t* bt = filt.data() + 5 * 32 * 16 + 32 * 4;   // intra_rec's BF_LUT, BF_SIZE, BF_MAD
    for( int q = 0; q < 26; q++ ) for( int i = 0; i < 48; i++ ) bt[q * 48 + i] = sh->bif->m_lut[q][i];
    for( int i = 0; i < 64; i++ ) bt[26 * 48 + i] = sh->bif->m_tuSizeFactor[0][i];
    for( int i = 0; i < 16; i++ ) bt[26 * 48 + 64 + i] = sh->bif->m_tuMADFactor[0][i];
  }
  // the inverse transform matrices of every type and size the CPU has (TrQuant::xIT's transform functions)
  std::vector<int32_t> mat;
  {
    // Don't put the TrQuant on the stack: it holds coefficient buffers of the largest transform, beyond the stack of
    // the thread that creates the device
    std::unique_ptr<TrQuant> trQuant( new TrQuant );
    trQuant->init( nullptr, MAX_TB_SIZEY, false, false, false, false );
    std::vector<TCoeff> m;
    for( int type = 0; type < NUM_TRANS_TYPE; type++ )
    {
      for( int l = 0; l < 7; l++ )
      {
        sh->trOffset[type][l] = -1;
        if( trQuant->getInvTransformMatrix( type, l + 1, m ) )
        {
          sh->trOffset[type][l] = (int32_t) mat.size();
          mat.insert( mat.end(), m.begin(), m.end() );
        }
      }
    }
  }
  sh->qcsOffset = (int32_t) mat.size();   // read by intra_resi's dequantization with QCS
  mat.insert( mat.end(), m_coeffShift64, m_coeffShift64 + 64 );
  // MIP (MatrixIntraPrediction::getMatrixData): per size class, the modes' matrices, each a row of weights per
  // reduced prediction sample (the large class without the first input's weight)
  sh->mipOffset[0] = (int32_t) mat.size();
  mat.insert( mat.end(), &mipMatrix4x4[0][0][0], &mipMatrix4x4[0][0][0] + sizeof( mipMatrix4x4 ) );
  sh->mipOffset[1] = (int32_t) mat.size();
  mat.insert( mat.end(), &mipMatrix8x8[0][0][0], &mipMatrix8x8[0][0][0] + sizeof( mipMatrix8x8 ) );
  sh->mipOffset[2] = (int32_t) mat.size();
  mat.insert( mat.end(), &mipMatrix16x16[0][0][0], &mipMatrix16x16[0][0][0] + sizeof( mipMatrix16x16 ) );
  // LFNST / NSPT (invSecondary): the positions x | y << 16 of the first 64 entries of each scan they read (the grouped
  // 4x4 diagonal scans, the top-left 8x8 / 16x16 ones of LFNST; from each entry's raster index, as the 16x16 scan's x
  // and y lie within its 8x8 quarters), and the extended LFNST's output positions
  auto addScan = [&]( const ScanElement* scan, const int width, const int count )
  {
    sh->secScans.emplace_back( scan, (int32_t) mat.size() );
    for( int i = 0; i < count; i++ )
    {
      mat.push_back( int32_t( scan[i].idx % width ) | int32_t( scan[i].idx / width ) << 16 );
    }
  };
  for( int wIdx = 0; wIdx < (int) gp_sizeIdxInfo->numWidths() && wIdx <= MAX_CU_DEPTH; wIdx++ )
  {
    const int w = gp_sizeIdxInfo->sizeFrom( wIdx );
    if( w >= 8 && w <= MAX_CU_SIZE )
    {
      addScan( g_coefTopLeftDiagScan8x8[wIdx], w, 64 );
    }
    if( w >= 16 && w <= MAX_CU_SIZE )
    {
      addScan( g_coefTopLeftDiagScan16x16[wIdx], w, 64 );
    }
    for( int hIdx = 0; hIdx < (int) gp_sizeIdxInfo->numHeights(); hIdx++ )
    {
      const int          h    = gp_sizeIdxInfo->sizeFrom( hIdx );
      const ScanElement* scan = w >= 4 && h >= 4 && w <= MAX_CU_SIZE && h <= MAX_CU_SIZE ? g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][wIdx][hIdx] : nullptr;
      if( scan )
      {
        addScan( scan, w, std::min( 64, w * h ) );
      }
    }
  }
  for( int s = 0; s < 3; s++ )
  {
    for( int transpose = 0; transpose < 2; transpose++ )
    {
      const std::vector<int32_t> layout = lfnstOutputLayout( 4 << s, transpose != 0 );
      sh->lfnstLayout[s][transpose] = (int32_t) mat.size();
      mat.insert( mat.end(), layout.begin(), layout.end() );
    }
  }
  // the CPU's LFNST (extended) and NSPT matrices as they are (int8)
  sh->secTables = { { &g_lfnst4x4[0][0][0][0], sizeof( g_lfnst4x4 ) },   { &g_lfnst8x8[0][0][0][0], sizeof( g_lfnst8x8 ) },
                    { &g_lfnst16x16[0][0][0][0], sizeof( g_lfnst16x16 ) }, { &g_nspt4x4[0][0][0], sizeof( g_nspt4x4 ) },
                    { &g_nspt4x8[0][0][0], sizeof( g_nspt4x8 ) },          { &g_nspt8x4[0][0][0], sizeof( g_nspt8x4 ) },
                    { &g_nspt8x8[0][0][0], sizeof( g_nspt8x8 ) },          { &g_nspt4x16[0][0][0], sizeof( g_nspt4x16 ) },
                    { &g_nspt16x4[0][0][0], sizeof( g_nspt16x4 ) },        { &g_nspt8x16[0][0][0], sizeof( g_nspt8x16 ) },
                    { &g_nspt16x8[0][0][0], sizeof( g_nspt16x8 ) },        { &g_nspt4x32[0][0][0], sizeof( g_nspt4x32 ) },
                    { &g_nspt32x4[0][0][0], sizeof( g_nspt32x4 ) },        { &g_nspt8x32[0][0][0], sizeof( g_nspt8x32 ) },
                    { &g_nspt32x8[0][0][0], sizeof( g_nspt32x8 ) } };
  std::vector<int8_t> kernels;
  for( const auto& table: sh->secTables )
  {
    kernels.insert( kernels.end(), table.first, table.first + table.second );
  }
  sh->filters.create( ctx, filt.size() * 4, gpu::Buffer::DEVICE );
  sh->matrices.create( ctx, mat.size() * 4, gpu::Buffer::DEVICE );
  sh->secKernels.create( ctx, kernels.size(), gpu::Buffer::DEVICE );
  gpu::Buffer upload;
  upload.create( ctx, ( filt.size() + mat.size() ) * 4 + kernels.size(), gpu::Buffer::UPLOAD );
  memcpy( upload.mapped, filt.data(), filt.size() * 4 );
  memcpy( (char*) upload.mapped + filt.size() * 4, mat.data(), mat.size() * 4 );
  memcpy( (char*) upload.mapped + ( filt.size() + mat.size() ) * 4, kernels.data(), kernels.size() );
  {
    gpu::Stream stream( ctx );
    stream.begin();
    stream.copy( upload, 0, sh->filters, 0, filt.size() * 4 );
    stream.copy( upload, filt.size() * 4, sh->matrices, 0, mat.size() * 4 );
    stream.copy( upload, ( filt.size() + mat.size() ) * 4, sh->secKernels, 0, kernels.size() );
    stream.submitAndWait();
  }
  upload.destroy( ctx );
  return sh;
}

std::once_flag s_sharedOnce;
Shared*        s_shared = nullptr;

Shared* shared( gpu::Context& ctx )
{
  std::call_once( s_sharedOnce, [&] { s_shared = createShared( ctx ); } );
  return s_shared;
}
}   // namespace

void GpuIntraRec::releaseShared()
{
  if( !s_shared )   // never created (gpu::Context::get() would create the device)
  {
    return;
  }
  gpu::Context* ctx = gpu::Context::get();
  s_shared->filters.destroy( *ctx );
  s_shared->matrices.destroy( *ctx );
  s_shared->secKernels.destroy( *ctx );
  delete s_shared;   // the kernels
  s_shared = nullptr;
}

struct GpuIntraRec::Impl
{
  gpu::Context*               ctx = nullptr;
  const Shared*               sh  = nullptr;
  std::unique_ptr<gpu::Stream> stream;
  gpu::Buffer                 params, tus, coeffs, planes[3], owner[3], sched, dbgPred, dbgResi, lmcs, signPredBuf;
  gpu::Buffer                 cuBuf, dbgParams;   // the CU records and their modes; the derived parameters (verification)
  Reshape*                    reshape = nullptr;   // the decoder's reshaper (LMCS state of the picture being decoded)
  TrQuant*                    trQuant = nullptr;   // the decoder's TrQuant (transform types of the blocks)
  int                         numGroups = 0;
  size_t                      edgeCap   = 0;   // dependency edges per picture the scheduling buffer holds
  gpu::Buffer                 upload, download;

  // the picture being recorded
  std::vector<Tu>             rec;
  std::vector<int32_t>        coef;
  std::vector<int16_t>        cpuPred, cpuResi;   // verification only
  size_t                      numSamples = 0;     // samples of the recorded blocks
  bool                        lmcsUsed = false;   // a block of the picture reads the LMCS tables
  int                         lastIspLuma = -1;   // the record of the last ISP luma sub-partition (Tu::ispPrev)
  std::vector<int32_t>        signPred;           // the sign prediction data of the blocks (Tu::spOffset)
  bool                        deriveModes = false;   // the GPU derives intra modes in the picture (gpuDerivesModes())
  std::vector<Cu>             cus;                // the CUs of the recorded blocks (Tu::cu), where it does
  std::vector<int32_t>        cuMode;             // per CU the host's mode (CU_KNOWN), else -1 (Params::cuMode)
  std::unordered_map<const CodingUnit*, int32_t> cuIndex[2];   // per channel type: the record of a CU
  // the sign prediction prepareSignPred took from a TU, for the record() of its coded component
  struct PendingSignPred
  {
    int                 num = 0;   // predicted signs (0: none pending)
    ComponentID         comp = COMPONENT_Y;
    Position            pos;       // of the block
    int32_t             codedSigns = 0;
    int32_t             xy[SIGN_PRED_MAX_NUM] = {};
    std::vector<TCoeff> coef;      // the levels with the decoder's signs, the predicted ones positive
  }                           pendingSign;
  std::string                 unsupportedWhy;
  int                         poc = 0;
  bool                        on = false;
  int                         numOn = 0, numFallback = 0;

  // totals
  int                         numPictures = 0, numVerified = 0, numMismatch = 0, numSkipped = 0;
  std::string                 firstSkipReason;
  double                      gpuMs = 0;

  ~Impl()
  {
    if( !ctx )
    {
      return;
    }
    for( gpu::Buffer* b: { &params, &tus, &coeffs, &planes[0], &planes[1], &planes[2], &owner[0], &owner[1], &owner[2],
                           &sched, &dbgPred, &dbgResi, &lmcs, &signPredBuf, &cuBuf, &dbgParams, &upload, &download } )
    {
      b->destroy( *ctx );
    }
  }
};

GpuIntraRec* GpuIntraRec::create()
{
  gpu::Context* ctx = gpu::Context::get();
  if( !ctx || !ctx->vulkanMemoryModel() )   // the blocks of a picture synchronize through the Vulkan memory model
  {
    return nullptr;
  }
  GpuIntraRec* r = new GpuIntraRec;
  r->m_impl.reset( new Impl );
  Impl& m = *r->m_impl;
  m.ctx = ctx;
  m.sh  = shared( *ctx );
  m.stream.reset( new gpu::Stream( *ctx ) );
  // a picture's commands upload its records and fill every buffer they read before writing it (owner maps, planes,
  // scheduling counters): they may run twice (gpu::Stream::wait)
  m.stream->setRepeatable( true );
  // persistent workgroups of intra_rec (ECM_GPU_GROUPS overrides)
  m.numGroups = getenv( "ECM_GPU_GROUPS" ) ? std::max( 1, atoi( getenv( "ECM_GPU_GROUPS" ) ) ) : 32;
  m.params.create( *ctx, sizeof( Params ), gpu::Buffer::DEVICE );
  return r;
}

GpuIntraRec::~GpuIntraRec()
{
  if( m_impl )
  {
    const Impl& m = *m_impl;
    if( m.on )
    {
      printf( "\n[gpu-intra] %d pictures: %d reconstructed on the GPU, %d by the CPU%s%s; GPU %.1f ms\n", m.numPictures,
              m.numOn, m.numPictures - m.numOn, m.firstSkipReason.empty() ? "" : ", first reason: ", m.firstSkipReason.c_str(),
              m.gpuMs );
    }
    else
    {
      printf( "\n[gpu-intra] %d pictures: %d verified (%d with differences), %d skipped%s%s; GPU %.1f ms\n", m.numPictures,
              m.numVerified, m.numMismatch, m.numSkipped, m.firstSkipReason.empty() ? "" : ", first reason: ",
              m.firstSkipReason.c_str(), m.gpuMs );
    }
  }
}

void GpuIntraRec::setOn( bool on )
{
  m_impl->on = on;
}

void GpuIntraRec::setReshape( Reshape* reshape )
{
  m_impl->reshape = reshape;
}

void GpuIntraRec::setTrQuant( TrQuant* trQuant )
{
  m_impl->trQuant = trQuant;
}

bool GpuIntraRec::on() const
{
  return m_impl->on;
}

bool GpuIntraRec::failed() const
{
  return !m_impl->unsupportedWhy.empty();
}

bool GpuIntraRec::supported( const Slice& slice, std::string& why )
{
  // every tool whose mode derivation or prediction reads reconstructed samples, or that the kernels do not implement,
  // must be off: with the GPU reconstruction on, the CPU reconstructs nothing the derivations could read
  const SPS& sps = *slice.getSPS();
  why.clear();
  if( !slice.isIntra() )                                                                  why = "not an intra slice";
  else if( sps.getChromaFormatIdc() != CHROMA_420 && sps.getChromaFormatIdc() != CHROMA_400 ) why = "chroma format";
  else if( sps.getLog2MaxTbSize() > 7 )                                                   why = "transform blocks above 128";
  // the reference availability of the kernels is "decoded earlier"; the CPU's (CodingStructure::getCURestricted) also
  // excludes the CUs of other tiles, with wavefronts the CTUs right of the current one in the CTU row above, and with
  // GDR the dirty area (several slices are rejected by DecSlice)
  else if( slice.getPPS()->getNumTiles() > 1 )                                            why = "tiles";
  else if( sps.getEntropyCodingSyncEnabledFlag() )                                        why = "wavefronts";
  else if( sps.getGDREnabledFlag() )                                                      why = "GDR";
  // implicit MTS selects by modes that only DIMD derives (TrQuant::getTrTypes, candModeListForTransform): with DIMD
  // off the stock encoder stops at a CHECK, so it can be neither supported nor tested before DIMD
  else if( sps.getUseImplicitMTS() )                                                      why = "implicit MTS";
  else if( sps.getScalingListFlag() )                                                     why = "scaling lists";
  else if( sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag() )               why = "extended precision";
  else if( sps.getUseColorTrans() )                                                       why = "ACT";
  // TrQuant::predCoeffSigns keeps the templates of each block size and transform type of the first block that needs
  // them, whatever its component; intra_resi computes them per block, with the block's bit depth
  else if( sps.getNumPredSigns() > 0 && sps.getBitDepth( CHANNEL_TYPE_LUMA ) != sps.getBitDepth( CHANNEL_TYPE_CHROMA ) ) why = "sign prediction with unequal luma and chroma bit depths";
  else if( sps.getUseTmrl() )                                                             why = "TMRL";
  else if( sps.getUseLMChroma() || sps.getUseCcpMerge() || sps.getUseDdCcpFusion() || sps.getUseBvgCccm() ) why = "cross-component prediction";
  else if( sps.getUseChromaReordering() || sps.getUseIntraDBV() )                         why = "chroma reordering / DBV";
  else if( sps.getUseDimd() || sps.getUseBvgDimd() || sps.getUseTimd() || sps.getUseTimdMrg() ) why = "DIMD / TIMD";
  else if( sps.getUseMdip() || sps.getUseMpmSorting() )                                   why = "MDIP / MPM sorting";
  else if( sps.getUseIntraTMP() || sps.getIBCFlag() )                                     why = "IntraTMP / IBC";
  else if( sps.getUseEip() || sps.getUseSgpm() )                                          why = "EIP / SGPM";
  else if( sps.getUsePDP() )                                                              why = "PDP";
  else if( slice.getPnnMode() )                                                           why = "NN intra prediction";
  return why.empty();
}

void GpuIntraRec::beginPicture( const CodingStructure& cs )
{
  Impl& m = *m_impl;
  m.rec.clear();
  m.coef.clear();
  m.cpuPred.clear();
  m.cpuResi.clear();
  m.numSamples = 0;
  m.lmcsUsed   = false;
  m.lastIspLuma = -1;
  m.signPred.clear();
  m.pendingSign.num = 0;
  m.deriveModes = gpuDerivesModes();
  m.cus.clear();
  m.cuMode.clear();
  m.cuIndex[0].clear();
  m.cuIndex[1].clear();
  m.unsupportedWhy.clear();
  m.poc = cs.slice->getPOC();
  std::string why;
  if( !supported( *cs.slice, why ) )
  {
    unsupported( why.c_str() );
  }
}

void GpuIntraRec::unsupported( const char* why )
{
  if( m_impl->unsupportedWhy.empty() )
  {
    m_impl->unsupportedWhy = why;
  }
}

void GpuIntraRec::prepareSignPred( TransformUnit& tu, const ComponentID compID )
{
  Impl& m = *m_impl;
  m.pendingSign.num = 0;
  if( !m.unsupportedWhy.empty() )
  {
    return;
  }
  // TrQuant::predCoeffSigns' blocks: with joint CbCr at the Cb block, for the coded component
  const bool jccr = tu.jointCbCr && isChroma( compID );
  if( jccr && compID == COMPONENT_Cr )
  {
    return;
  }
  const ComponentID resiComp = jccr && !( tu.jointCbCr >> 1 ) ? COMPONENT_Cr : compID;
  if( !TU::getUseSignPred( tu, resiComp ) || !( TU::getCbf( tu, compID ) || jccr ) )
  {
    return;
  }
  // DepQuant::getPredictedSigns gives the coded signs to the levels in the decoder's order (in place) and lists the
  // predicted ones; the TU gets its levels back, as the CPU calls it again where it reconstructs the block
  CoeffBuf            levels = tu.getCoeffs( resiComp );
  const CompArea&     area   = tu.blocks[resiComp];
  std::vector<TCoeff> saved( area.area() );
  for( int y = 0; y < (int) area.height; y++ )
  {
    std::copy_n( levels.bufAt( 0, y ), area.width, saved.data() + y * area.width );
  }
  static_vector<Position, SIGN_PRED_MAX_NUM> predicted;
  uint8_t                                    signBuf[SIGN_PRED_FREQ_RANGE * SIGN_PRED_FREQ_RANGE];
  DepQuant::getPredictedSigns( tu, resiComp, predicted, signBuf, true );
  Impl::PendingSignPred& p = m.pendingSign;
  p.coef.resize( area.area() );
  for( int y = 0; y < (int) area.height; y++ )
  {
    std::copy_n( levels.bufAt( 0, y ), area.width, p.coef.data() + y * area.width );
    std::copy_n( saved.data() + y * area.width, area.width, levels.bufAt( 0, y ) );
  }
  // the coded signs of the predicted levels (whether each prediction is wrong), which become positive
  p.comp       = resiComp;
  p.pos        = area.pos();
  p.codedSigns = 0;
  for( int k = 0; k < (int) predicted.size(); k++ )
  {
    TCoeff& level = p.coef[predicted[k].y * area.width + predicted[k].x];
    if( level < 0 )
    {
      p.codedSigns |= 1 << k;
      level = -level;
    }
    p.xy[k] = predicted[k].x | ( predicted[k].y << 16 );
  }
  p.num = (int) predicted.size();
}

// the record of the CU of a block of the channel (made at the CU's first block): how its mode is found and the CUs that
// reads. With gpuDerivesModes() the GPU derives the luma modes the MPM lists give (from the CUs at getSpatialIpm's
// positions, getPURestricted's) and the chroma modes DM or the candidates' substitution give from those, in a node
// recorded before the CU's first block (intra_rec); the host keeps the others, for luma the mode the neighbours' MPM
// lists take (PU::getIntraDirLuma: planar for MIP). -1 with why set where a neighbour cannot be read.
static int32_t cuRecord( GpuIntraRec::Impl& m, const TransformUnit& tu, const ComponentID compID, const char*& why )
{
  const CodingUnit&     cu     = *tu.cu;
  const PredictionUnit& pu     = *cu.firstPU;
  const ChannelType     chType = toChannelType( compID );
  const auto            known  = m.cuIndex[chType].find( &cu );
  if( known != m.cuIndex[chType].end() )
  {
    return known->second;
  }
  Cu c{};
  c.kind    = CU_KNOWN;
  c.node    = -1;
  c.lumaCu  = -1;
  std::fill_n( c.nbr, MPM_NEIGHBOURS, -1 );
  c.modes[0] = c.modes[1] = -1;
  const CompArea& a = cu.blocks[chType == CHANNEL_TYPE_LUMA ? COMPONENT_Y : COMPONENT_Cb];
  c.x = a.x;
  c.y = a.y;
  c.w = a.width;
  c.h = a.height;
  int32_t mode = -1;
  if( isLuma( chType ) )
  {
    if( gpuDerivesModes() && pu.parseLumaMode && !cu.mipFlag && !cu.bdpcmMode )
    {
      c.kind          = CU_MPM;
      c.mpmFlag       = pu.mpmFlag ? 1 : 0;
      c.secondMpmFlag = pu.secondMpmFlag ? 1 : 0;
      c.ipredIdx      = pu.ipredIdx;
      c.plIdx         = cu.plIdx;
      if( pu.mpmFlag && pu.ipredIdx == 0 )   // the MPM list starts with planar (without PDP and NN, which supported() rejects)
      {
        c.modes[0] = PLANAR_IDX;
      }
      int nbFirst = 0, nbLast = 0;
      CU::getNbModesRemovedFirstLast( cu.areAboveRightUnavail, cu.areBelowLeftUnavail, a.height, a.width, nbFirst, nbLast );
      c.nbFirst = nbFirst;
      c.nbLast  = nbLast;
      // getSpatialIpm's positions: the adjacent ones, then per distance d the non-adjacent ones (idxMap: 0 below-left, 1
      // above-right, 2 above the middle, 3 left of the middle, 4 above-left)
      const int w = a.width, h = a.height;
      Position  pos[MPM_NEIGHBOURS];
      int       n = 0;
      pos[n++] = a.topLeft().offset( -1, h - 1 );
      pos[n++] = a.topLeft().offset( w - 1, -1 );
      pos[n++] = a.topLeft().offset( -1, -1 );
      pos[n++] = a.topLeft().offset( w, -1 );
      pos[n++] = a.topLeft().offset( -1, h );
      const int numNA[4]     = { 3, 5, 5, 5 };
      const int idxMap[4][5] = { { 0, 1, 4 }, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 } };
      for( int d = 0; d < NADISTANCE_LEVEL; d++ )
      {
        const int dh = w * ( d + 1 ), dv = h * ( d + 1 );
        for( int k = 0; k < numNA[d]; k++ )
        {
          const int i = idxMap[d][k];
          const int ox = i == 0 || i == 3 || i == 4 ? -dh - 1 : i == 1 ? w + dh - 1 : w >> 1;
          const int oy = i == 0 ? h + dv - 1 : i == 3 ? h >> 1 : -dv - 1;
          pos[n++] = a.topLeft().offset( ox, oy );
        }
      }
      CHECK( n != MPM_NEIGHBOURS, "getSpatialIpm's positions" );
      for( int k = 0; k < MPM_NEIGHBOURS; k++ )
      {
        const PredictionUnit* nb = cu.cs->getPURestricted( pos[k], pu, CHANNEL_TYPE_LUMA );
        if( !nb )
        {
          continue;
        }
        const auto rec = m.cuIndex[CHANNEL_TYPE_LUMA].find( nb->cu );
        if( !CU::isIntra( *nb->cu ) || rec == m.cuIndex[CHANNEL_TYPE_LUMA].end() )
        {
          why = "an MPM neighbour that is not a recorded intra CU";
          return -1;
        }
        c.nbr[k] = rec->second;
      }
    }
    else
    {
      mode = PU::getIntraDirLuma( pu );
    }
  }
  else
  {
    const auto luma = m.cuIndex[CHANNEL_TYPE_LUMA].find( PU::getCoLocatedLumaPU( pu ).cu );
    c.lumaCu = luma != m.cuIndex[CHANNEL_TYPE_LUMA].end() ? luma->second : -1;
    const bool lumaDerived = c.lumaCu >= 0 && m.cus[c.lumaCu].kind != CU_KNOWN && !cu.bdpcmModeChroma;
    if( lumaDerived && pu.intraDir[CHANNEL_TYPE_CHROMA] == DM_CHROMA_IDX )
    {
      c.kind     = CU_CHROMA_DM;
      c.modes[0] = m.cus[c.lumaCu].modes[0];
      c.modes[1] = m.cus[c.lumaCu].modes[1];
    }
    else if( lumaDerived && pu.parseChromaMode && pu.candId < 4 )
    {
      static const int32_t cand[4] = { PLANAR_IDX, VER_IDX, HOR_IDX, DC_IDX };
      c.kind     = CU_CHROMA_CAND;
      c.candId   = pu.candId;
      c.modes[0] = cand[pu.candId];
      c.modes[1] = VDIA_IDX;   // replaces it where the luma mode is the same
    }
    else
    {
      mode = PU::getFinalIntraMode( pu, CHANNEL_TYPE_CHROMA );
    }
  }
  const int32_t index = (int32_t) m.cus.size();
  if( c.kind != CU_KNOWN )   // its node, which derives the mode before the CU's blocks (no samples)
  {
    Tu node{};
    node.comp      = chType == CHANNEL_TYPE_LUMA ? COMPONENT_Y : COMPONENT_Cb;
    node.x         = c.x;
    node.y         = c.y;
    node.dbgOffset = (int32_t) m.numSamples;
    node.mipMatrix = -1;
    node.ispPrev = node.spCoded = node.spOffset = node.resiFrom = -1;
    node.cu        = index;
    node.derive    = DERIVE_NODE;
    c.node         = (int32_t) m.rec.size();
    m.rec.push_back( node );
  }
  m.cus.push_back( c );
  m.cuMode.push_back( mode );
  m.cuIndex[chType][&cu] = index;
  return index;
}

void GpuIntraRec::record( const TransformUnit& tu, const ComponentID compID, const IntraPrediction::IpaSnapshot& ipa,
                          const QpParam& qp, const CPelBuf* pred, const CPelBuf* resi )
{
  Impl& m = *m_impl;
  if( !m.unsupportedWhy.empty() )
  {
    return;
  }
  const CodingUnit&      cu  = *tu.cu;
  const PredictionUnit&  pu  = *cu.firstPU;
  const CodingStructure& cs  = *tu.cs;
  const SPS&             sps = *cs.sps;
  const Slice&           slice = *cs.slice;
  const ChannelType      chType = toChannelType( compID );
  const CompArea&        area   = tu.blocks[compID];
  // MIP blocks (luma; IntraPrediction::predIntraMip) are recorded with the DC mode: they read the references DC reads
  // (a superset), and their bilateral filter, like DC's, has no angular texture gate; Tu::mipMatrix selects MIP
  const bool             mip     = isLuma( compID ) && cu.mipFlag;
  const int              dirMode = mip ? DC_IDX : PU::getFinalIntraMode( pu, chType );

  // what the GPU path does not do yet
  const char* why = nullptr;
  if( !CU::isIntra( cu ) )                                          why = "not intra";
  else if( !isLuma( compID ) && PU::isMIP( pu, chType ) )           why = "chroma MIP (4:4:4)";
  else if( cu.ispMode && ipa.multiRefIndex )                        why = "ISP with MRL";
  // LFNST / NSPT of MIP blocks select their kernel by a DIMD mode the CPU derives from the MIP prediction
  else if( cu.mipFlag && cu.lfnstIdx )                              why = "LFNST of MIP";
  // transform skip with the regular residual coding is dequantized like a transform block (DepQuant::dequant)
  else if( ( tu.mtsIdx[compID] == MTS_SKIP || ( tu.jointCbCr && tu.mtsIdx[COMPONENT_Cr] == MTS_SKIP ) )
           && slice.getTSResidualCodingDisabledFlag() )             why = "transform skip with regular residual coding";
  else if( cu.tmpFlag )                                             why = "IntraTMP";
  else if( cu.dimd )                                                why = "DIMD";
  else if( cu.timd )                                                why = "TIMD";
  else if( cu.sgpm )                                                why = "SGPM";
  else if( cu.tmrlFlag )                                            why = "TMRL";
  else if( cu.eipFlag )                                             why = "EIP";
  else if( !isLuma( compID ) && ipa.multiRefIndex )                why = "chroma MRL";
  else if( dirMode >= NUM_LUMA_MODE )                               why = "intra mode beyond 66 (LM / NN / ...)";
  else if( sps.getUseLmcs() && !m.reshape )                         why = "LMCS without the decoder's reshaper";
  else if( sps.getScalingListFlag() )                               why = "scaling lists";
  else if( sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag() ) why = "extended precision";
  // ISP splits luma into sub-partitions down to 1 x 4 / 4 x 1
  else if( ( ( area.width < 2 || area.height < 2 ) && !( cu.ispMode && isLuma( compID ) ) ) || area.width > 128 || area.height > 128 ) why = "block size";
  else if( ( area.width & ( area.width - 1 ) ) || ( area.height & ( area.height - 1 ) ) ) why = "block size not a power of 2";
  if( why )
  {
    unsupported( why );
    return;
  }
  // the block's CU, where the GPU derives modes (of a CU_MPM / CU_CHROMA_* one: the mode-dependent fields below are
  // then the CPU's, for the comparison only)
  int32_t cuIdx = -1;
  if( m.deriveModes )
  {
    cuIdx = cuRecord( m, tu, compID, why );
    if( cuIdx < 0 )
    {
      unsupported( why );
      return;
    }
  }
  const bool derived = cuIdx >= 0 && m.cus[cuIdx].kind != CU_KNOWN;

  Tu t{};
  t.comp           = compID;
  t.x              = area.x;
  t.y              = area.y;
  t.w              = area.width;
  t.h              = area.height;
  t.dirMode        = dirMode;
  t.isModeVer      = ipa.isModeVer;
  t.intraPredAngle = ipa.intraPredAngle;
  t.absInvAngle    = ipa.absInvAngle;
  t.useCubic       = !ipa.interpolationFlag;
  t.plIdx          = isLuma( compID ) && dirMode == PLANAR_IDX ? cu.plIdx : 0;
  t.refFilter        = ipa.refFilterFlag;
  t.applyPDPC        = ipa.applyPDPC;
  t.gradPDPC         = ipa.useGradPDPC;
  t.angularScale     = ipa.angularScale;
  t.fusion           = ipa.applyFusion;
  t.disableRefFilter = sps.getDisableRefFilter();
  t.qp = cu.qp;
  // MRL (luma): the reference line, the fusion line is the next one; xPredIntraAng turns the bilateral filter off
  // for it, the DC prediction (MRL excludes planar) keeps it
  t.mrl = ipa.multiRefIndex;
  // ISP (luma; DecCu::xIntraRecBlk, IntraPrediction::initIntraPatternChTypeISP): the CU, whose references the first
  // sub-partition fills and the later ones shift; the prediction region, which for vertical sub-partitions 1 or 2 wide
  // is 4 wide and shared (CU::isPredRegDiffFromTB, adjustPredArea); and the previous sub-partition, which the
  // scheduling chains to, as the 4x4 owner units cannot tell sub-partitions 1 or 2 wide or high apart
  t.isp     = 0;
  t.regX    = area.x;
  t.regW    = area.width;
  t.ispPrev = -1;
  if( cu.ispMode && isLuma( compID ) )
  {
    const CompArea& cuArea = cu.blocks[COMPONENT_Y];
    t.isp = cu.ispMode;   // HOR_INTRA_SUBPARTITIONS 1, VER_INTRA_SUBPARTITIONS 2
    t.cuX = cuArea.x;
    t.cuY = cuArea.y;
    t.cuW = cuArea.width;
    t.cuH = cuArea.height;
    if( CU::isPredRegDiffFromTB( cu, compID ) )
    {
      t.regX = cuArea.x + ( area.x - cuArea.x ) / PRED_REG_MIN_WIDTH * PRED_REG_MIN_WIDTH;
      t.regW = std::max<int>( PRED_REG_MIN_WIDTH, area.width );
    }
    if( !CU::isISPFirst( cu, area, compID ) )
    {
      t.ispPrev = m.lastIspLuma;
    }
    m.lastIspLuma = (int) m.rec.size();
  }
  // the intra bilateral filter: checkBfApplication on the predicted block (an ISP prediction region) gives the
  // conditions of size and QP (bfBase); xPredIntraAng turns the filter off for angular modes with ISP or MRL, and
  // checkBfApplication for the modes next to the diagonals, horizontal and vertical (intra_modes.glsl's
  // applyModeParams has the same rule for a derived mode)
  bool bfBase = false;
  if( compID == COMPONENT_Y && sps.getUseIntraPredBf() && slice.isIntra() )
  {
    const int W = t.regW, H = area.height, qpY = cu.qp;
    bfBase = qpY > 17 && W <= 128 && H <= 128 && ( W & ( W - 1 ) ) == 0 && ( H & ( H - 1 ) ) == 0 && W >= 4 && H >= 4;
    bfBase = bfBase && W * H > 16 && W * H < 1024 && std::abs( floorLog2( W ) - floorLog2( H ) ) <= 2;
    bfBase = bfBase && m.sh->bif->m_tuSizeFactor[0][std::min( floorLog2( W ), 7 ) * 8 + std::min( floorLog2( H ), 7 )] != 0;
  }
  {
    const int  md      = dirMode;
    const bool modeOff = ( ( t.mrl || t.isp ) && md > DC_IDX )
                         || ( md > DC_IDX && ( ( md >= 2 && md <= 3 ) || ( md >= 17 && md <= 19 ) || ( md >= 33 && md <= 35 )
                                               || ( md >= 49 && md <= 51 ) || ( md >= 65 && md <= 66 ) ) );
    t.bf = bfBase && !modeOff ? ( md <= DC_IDX ? 1 : 2 ) : 0;
  }
  // bilateralFilterPredDiamond5x5Clip pads at the CU, which ISP sub-partitions reproduce (Tu::isp); other blocks smaller
  // than their CU are not supported (with a derived mode: wherever the filter may apply)
  if( ( t.bf || ( derived && bfBase ) ) && !t.isp
      && ( area.pos() != cu.blocks[COMPONENT_Y].pos() || area.size() != cu.blocks[COMPONENT_Y].size() ) )
  {
    unsupported( "intra bilateral filter of a split CU" );
    return;
  }
  // MIP (MatrixIntraPrediction::predBlock): the mode's matrix, transposition and size class; the prediction reads the
  // unfiltered references and uses none of the angular parameters
  t.mipMatrix = -1;
  if( mip )
  {
    const int    sizeId = getMipSizeId( area.size() );
    const int    rows[3] = { 16, 16, 64 }, rowLen[3] = { 4, 8, 7 };   // reduced samples, weights per sample
    const int    modeIdx = pu.intraDir[CHANNEL_TYPE_LUMA];
    if( modeIdx < 0 || modeIdx >= getNumModesMip( area.size() ) )
    {
      unsupported( "MIP mode index out of range" );
      return;
    }
    t.mipMatrix     = m.sh->mipOffset[sizeId] + modeIdx * rows[sizeId] * rowLen[sizeId];
    t.mipTransposed = pu.mipTransposedFlag ? 1 : 0;
    t.mipSizeId     = sizeId;
    t.refFilter = t.applyPDPC = t.gradPDPC = t.fusion = 0;
  }
  // BDPCM (IntraPrediction::xPredIntraBDPCM): the unfiltered left column (1, horizontal) or above row (2, vertical)
  // copied, i.e. the angular prediction of mode 18 or 50 without reference filter, PDPC, fusion or bilateral filter
  const int bdpcmPred = isLuma( compID ) ? cu.bdpcmMode : cu.bdpcmModeChroma;
  if( bdpcmPred )
  {
    if( dirMode != ( bdpcmPred == 1 ? HOR_IDX : VER_IDX ) )
    {
      unsupported( "BDPCM with an intra mode other than its direction" );
      return;
    }
    t.refFilter = t.applyPDPC = t.gradPDPC = t.fusion = t.bf = 0;
  }
  // a derived mode: what its parameters depend on besides the mode (initIntraPatternChType's fetchRef2nd, the
  // filter's conditions, the SPS's switches)
  t.cu     = cuIdx;
  t.derive = 0;
  if( derived )
  {
    t.derive = DERIVE_MODE | ( ipa.fetchRef2nd ? DERIVE_FETCH2ND : 0 ) | ( bfBase ? DERIVE_BF : 0 )
               | ( sps.getDisablePdpc() ? DERIVE_NO_PDPC : 0 )
               | ( sps.getSpsRangeExtension().getIntraSmoothingDisabledFlag() ? DERIVE_NO_SMOOTHING : 0 );
  }
  const PreCalcValues& pcv = *cs.pcv;
  const bool noShift = pcv.noChroma2x2 && area.width == 4;
  t.unitW    = pcv.minCUWidth >> ( noShift ? 0 : getComponentScaleX( compID, sps.getChromaFormatIdc() ) );
  t.unitH    = pcv.minCUHeight >> ( noShift ? 0 : getComponentScaleY( compID, sps.getChromaFormatIdc() ) );
  t.bitDepth = sps.getBitDepth( chType );
  if( t.mrl && t.unitW != t.unitH )   // xFillReferenceSamples counts the corner samples of both arms by the unit height
  {
    unsupported( "MRL with non-square availability units" );
    return;
  }

  // the component whose coefficients give the block's residual: with joint CbCr (DecCu::xIntraRecBlk at Cb), the
  // coded one of the pair (Cb for jointCbCr 2 and 3, else Cr with its own QP); the Cb block computes both residuals
  // (TrQuant::invTransformICT), the Cr block that follows it gets its residual from there
  ComponentID resiComp = compID;
  t.jccr       = 0;
  t.jccrOffset = 0;
  t.spCoded = t.spOffset = t.resiFrom = -1;
  if( tu.jointCbCr && compID == COMPONENT_Cb )
  {
    t.jccr   = TU::getICTMode( tu );
    resiComp = ( tu.jointCbCr >> 1 ) ? COMPONENT_Cb : COMPONENT_Cr;
  }
  else if( tu.jointCbCr && compID == COMPONENT_Cr )
  {
    // DecCu::xIntraRecQT records a TU's Cb and Cr blocks one after the other, so the Cr samples follow the Cb ones
    const Tu* cb = m.rec.empty() ? nullptr : &m.rec.back();
    if( !cb || cb->comp != COMPONENT_Cb || cb->x != area.x || cb->y != area.y || cb->jccr == 0
        || cb->jccrOffset != (int32_t) m.numSamples )
    {
      unsupported( "joint CbCr: a Cr block not right after its Cb block" );
      return;
    }
    t.jccr = JCCR_DERIVED;
    if( cb->spOffset >= 0 )   // sign prediction of the pair, decided by the coded component's block
    {
      const int32_t cbIndex = (int32_t) m.rec.size() - 1;
      if( std::abs( cb->jccr ) == 3 )   // Cr coded: this block decides and writes the Cb block's residual
      {
        t.spCoded               = cbIndex;
        m.rec[cbIndex].resiFrom = cbIndex + 1;
      }
      else
      {
        t.resiFrom = cbIndex;
      }
    }
  }
  // sign prediction (TrQuant::predCoeffSigns): prepareSignPred took the predicted levels of the coded component before
  // the CPU decided their signs
  Impl::PendingSignPred& sign     = m.pendingSign;
  const bool             signPred = sign.num > 0 && t.jccr != JCCR_DERIVED;
  if( signPred && ( sign.comp != resiComp || sign.pos != tu.blocks[resiComp].pos() ) )
  {
    unsupported( "sign prediction taken from another block" );
    return;
  }
  const QpParam resiQp = resiComp == compID ? qp : QpParam( tu, resiComp );
  t.cbf = t.jccr != JCCR_DERIVED && TU::getCbf( tu, resiComp ) ? 1 : 0;
  // transform skip (TrQuant::xITransformSkip): the dequantized levels are the residual
  const bool ts   = tu.mtsIdx[resiComp] == MTS_SKIP;
  t.transformSkip = ts ? 1 : 0;

  // TrQuant::xIT's transform types (DCT-II, or those of explicit MTS) and their matrices; with lfnstIdx the inverse
  // LFNST before them or NSPT instead of them (invSecondary: the CPU's own choices of matrix, scan and sizes)
  t.mtxV = t.mtxH = 0;
  t.secKind = SEC_NONE;
  t.secMatrix = t.secRowLength = t.secNumIn = t.secScan = t.secLayout = t.secSize = t.secTranspose = 0;
  if( t.cbf && !ts )
  {
    if( !m.trQuant )
    {
      unsupported( "transform types without the decoder's TrQuant" );
      return;
    }
    Secondary sec;
    if( !invSecondary( tu, resiComp, sec ) )
    {
      unsupported( "LFNST without its extension (the VVC kernels)" );
      return;
    }
    int        trTypeHor, trTypeVer;
    const bool separable = m.trQuant->getInvTransformTypes( tu, resiComp, trTypeHor, trTypeVer );
    if( area.width > 1 && area.height > 1 && separable == ( sec.kind == SEC_NSPT ) )   // xIT: NSPT or the separable one
    {
      unsupported( "NSPT where xIT does not apply it" );
      return;
    }
    if( sec.kind != SEC_NONE )
    {
      t.secKind      = sec.kind;
      t.secMatrix    = m.sh->kernelOffset( sec.matrix );
      t.secRowLength = sec.rowLength;
      t.secNumIn     = sec.numIn;
      t.secScan      = m.sh->scanOffset( sec.scan );
      t.secSize      = sec.sbSize;
      t.secTranspose = sec.transpose ? 1 : 0;
      t.secLayout    = sec.kind == SEC_LFNST ? m.sh->lfnstLayout[floorLog2( sec.sbSize ) - 2][t.secTranspose] : 0;
      if( t.secMatrix < 0 || t.secScan < 0 || sec.numIn > 64 || ( sec.kind == SEC_LFNST && sec.rowLength > 96 ) )
      {
        unsupported( "an LFNST / NSPT matrix or scan the GPU lacks" );
        return;
      }
    }
    if( sec.kind == SEC_NSPT )
    {
      // its matrix gives the residual
    }
    else if( !separable )
    {
      if( area.width == 1 || area.height == 1 )   // xIT's 1-D transform (ISP sub-partitions): one stage along the block
      {
        t.mtxV = area.width == 1 ? m.sh->trOffset[trTypeVer][floorLog2( area.height ) - 1] : 0;
        t.mtxH = area.height == 1 ? m.sh->trOffset[trTypeHor][floorLog2( area.width ) - 1] : 0;
        if( t.mtxV < 0 || t.mtxH < 0 )
        {
          unsupported( "a transform type without a matrix of this size" );
          return;
        }
      }
      else
      {
        unsupported( "a transform the GPU lacks" );
        return;
      }
    }
    else
    {
      t.mtxV = m.sh->trOffset[trTypeVer][floorLog2( area.height ) - 1];
      t.mtxH = m.sh->trOffset[trTypeHor][floorLog2( area.width ) - 1];
      if( t.mtxV < 0 || t.mtxH < 0 )
      {
        unsupported( "a transform type without a matrix of this size" );
        return;
      }
    }
  }
  if( signPred && ( !t.cbf || ts || area.width < 4 || area.height < 4 || area.width > 32 || area.height > 32 ) )
  {
    unsupported( "sign prediction of a block outside TU::getUseSignPred's conditions" );
    return;
  }

  // dequantization without scaling lists or extended precision: DepQuant::dequant (dequantBlock) with dependent
  // quantization (not for transform skip, which uses its own residual coding), else Quant::dequant; QCS blends either
  // with the dequantization of the next larger index
  const int maxLog2TrDynamicRange = sps.getMaxLog2TrDynamicRange( chType );
  t.quant = ( slice.getDepQuantEnabledIdc() && !ts ? QUANT_DEP : 0 ) | ( sps.getUseQcs() ? QUANT_QCS : 0 );
  if( t.quant & QUANT_DEP )
  {
    // the index of a level depends on the quantizer state (the indices are derived with the levels below); QP + 1,
    // and the scale is shifted left instead of a negative right shift, without rounding
    const bool sqrt2          = TU::needsSqrt2Scale( tu, resiComp );
    const int  qpDQ           = resiQp.Qp( false ) + 1;
    const int  qpPer          = qpDQ / 6;
    const int  qpRem          = qpDQ - 6 * qpPer;
    const int  transformShift = getTransformShift( t.bitDepth, area.size(), maxLog2TrDynamicRange ) + ( sqrt2 ? -1 : 0 );
    const int  shift          = IQUANT_SHIFT + 1 - qpPer - transformShift;
    const int  scale          = g_invQuantScales[sqrt2 ? 1 : 0][qpRem];
    t.dqScale      = shift < 0 ? scale << -shift : scale;
    t.dqRightShift = std::max( shift, 0 );
    t.dqInMin      = INT32_MIN;   // the indices are not clipped
    t.dqInMax      = INT32_MAX;
  }
  else
  {
    // transform skip: the QP of transform skip, and no transform shift
    const bool needSqrtAdjustment  = TU::needsBlockSizeTrafoScale( tu, resiComp );
    const int  iTransformShift     = getTransformShift( t.bitDepth, area.size(), maxLog2TrDynamicRange ) + ( needSqrtAdjustment ? -1 : 0 );
    const int  rightShift          = IQUANT_SHIFT - ( ( ts ? 0 : iTransformShift ) + resiQp.per( ts ) );
    const int  scaleBits           = IQUANT_SHIFT + 1;
    const int  targetInputBitDepth = std::min<int>( maxLog2TrDynamicRange + 1, 32 + rightShift - scaleBits );
    t.dqScale      = g_invQuantScales[needSqrtAdjustment ? 1 : 0][resiQp.rem( ts )];
    t.dqRightShift = rightShift;
    t.dqInMin      = -( 1 << ( targetInputBitDepth - 1 ) );
    t.dqInMax      = ( 1 << ( targetInputBitDepth - 1 ) ) - 1;
  }
  t.trMin        = -( 1 << maxLog2TrDynamicRange );
  t.trMax        = ( 1 << maxLog2TrDynamicRange ) - 1;
  if( t.trMin < INT16_MIN || t.trMax > INT16_MAX )   // intra_resi keeps the first transform stage in 16 bits
  {
    unsupported( "transform range above 16 bits" );
    return;
  }
  // TrQuant::xIT
  const int transformMatrixShift = g_transformMatrixShift[TRANSFORM_INVERSE];
  t.shift1 = transformMatrixShift + 1 + COM16_C806_TRANS_PREC;
  t.shift2 = ( transformMatrixShift + maxLog2TrDynamicRange - 1 ) - t.bitDepth + COM16_C806_TRANS_PREC;
  // DecCu::xIntraRecBlk reconstruction clipping (adaptive clipping for luma)
  if( compID == COMPONENT_Y )
  {
    t.recMin = slice.getLumaPelMin();
    t.recMax = slice.getLumaPelMax();
  }
  else
  {
    t.recMin = slice.clpRng( compID ).min;
    t.recMax = slice.clpRng( compID ).max;
  }
  // LMCS (the reconstruction stays in the mapped domain): the luma clipping range is mapped (DecCu::xIntraRecBlk),
  // the intra bilateral filter works on inverse-mapped samples (IntraPrediction::checkBfLmcsApplication), and the
  // chroma residual is scaled by the average luma next to the CU (DecCu::xIntraRecBlk, calculateChromaAdjVpduNei)
  t.lmcs = 0;
  if( sps.getUseLmcs() )
  {
    Reshape&                rsp   = *m.reshape;
    const SliceReshapeInfo& rInfo = rsp.getSliceReshaperInfo();
    if( compID == COMPONENT_Y && cs.picHeader->getLmcsEnabledFlag() )
    {
      t.recMin = rsp.getFwdLUT()[slice.getLumaPelMin()];
      t.recMax = rsp.getFwdLUT()[slice.getLumaPelMax()];
    }
    const bool bfLmcs = cs.picHeader->getLmcsEnabledFlag() && rInfo.getUseSliceReshaper()
                        && ( slice.getSliceType() == I_SLICE ? rInfo.getUseSliceReshaper() : rsp.getCTUFlag() );
    if( t.bf && bfLmcs )
    {
      t.lmcs |= LMCS_BF_INVERSE;
    }
    const bool lmcsSlice = slice.getLmcsEnabledFlag() && ( slice.isIntra() || rsp.getCTUFlag() );
    if( lmcsSlice && isChroma( compID ) && area.width * area.height > 4 && ( TU::getCbf( tu, compID ) || tu.jointCbCr )
        && cs.picHeader->getLmcsChromaResidualScaleFlag() )
    {
      t.lmcs |= LMCS_CHROMA_SCALE;
      // the CU's luma, or the chroma CU's area in luma samples (DecCu::xIntraRecBlk with LMCS_CHROMA_CALC_CU)
      const Area areaY = cu.Y().valid() ? Area( cu.Y() )
                                        : Area( recalcPosition( tu.chromaFormat, tu.chType, CHANNEL_TYPE_LUMA, cu.blocks[tu.chType].pos() ),
                                                recalcSize( tu.chromaFormat, tu.chType, CHANNEL_TYPE_LUMA, cu.blocks[tu.chType].size() ) );
      const CodingUnit *cuLeft, *cuAbove;
      Reshape::getChromaAdjNeighbours( tu, areaY.pos(), cuLeft, cuAbove );
      t.crsX     = areaY.x;
      t.crsY     = areaY.y;
      t.crsW     = areaY.width;
      t.crsH     = areaY.height;
      t.crsAvail = ( cuLeft ? 1 : 0 ) | ( cuAbove ? 2 : 0 );
    }
    m.lmcsUsed |= t.lmcs != 0;
  }

  // the levels: only the region holding the nonzero ones (rows < nzRows, columns < nzCols)
  t.coeffOffset = (int32_t) m.coef.size();
  t.dbgOffset   = (int32_t) m.numSamples;
  m.numSamples += size_t( area.width ) * area.height;
  if( t.jccr != 0 && t.jccr != JCCR_DERIVED )
  {
    t.jccrOffset = (int32_t) m.numSamples;   // the Cr block's samples
  }
  t.nzRows = t.nzCols = 0;
  if( t.cbf )
  {
    // BDPCM (Quant::dequant, with transform skip): the levels accumulated along the rows or columns
    const bool          bdpcm = isLuma( resiComp ) ? cu.bdpcmMode != 0 : cu.bdpcmModeChroma != 0;
    std::vector<TCoeff> accumulated;
    CCoeffBuf           c = signPred ? CCoeffBuf( sign.coef.data(), area.width, area.height ) : tu.getCoeffs( resiComp );
    if( bdpcm )
    {
      accumulated.resize( area.area() );
      CoeffBuf acc( accumulated.data(), area.width, area.height );
      invResDPCM( tu, resiComp, acc );
      c = CCoeffBuf( accumulated.data(), area.width, area.height );
    }
    for( int y = 0; y < (int) area.height; y++ )
    {
      const TCoeff* row = c.bufAt( 0, y );
      for( int x = (int) area.width - 1; x >= t.nzCols; x-- )
      {
        if( row[x] != 0 )
        {
          t.nzRows = y + 1;
          t.nzCols = x + 1;
          break;
        }
      }
      if( t.nzRows < y + 1 )
      {
        for( int x = 0; x < t.nzCols; x++ )
        {
          if( row[x] != 0 )
          {
            t.nzRows = y + 1;
            break;
          }
        }
      }
    }
    if( t.secKind == SEC_LFNST )   // intra_resi takes the whole region of its outputs, which holds every level
    {
      if( t.nzRows > t.secSize || t.nzCols > t.secSize )
      {
        unsupported( "LFNST with levels outside its region" );
        return;
      }
      t.nzRows = t.nzCols = t.secSize;
    }
    for( int y = 0; y < t.nzRows; y++ )
    {
      const TCoeff* row = c.bufAt( 0, y );
      m.coef.insert( m.coef.end(), row, row + t.nzCols );
    }
    if( t.quant & QUANT_DEP )
    {
      // DepQuant's dequantBlock: from the last nonzero level down the diagonal scan, the quantizer state (a function
      // of the levels' parities only) turns each level into its quantization index
      CHECK( c.stride != area.width, "coefficients of a TU are contiguous" );
      const ScanElement* scan = g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][gp_sizeIdxInfo->idxFrom( area.width )]
                                           [gp_sizeIdxInfo->idxFrom( area.height )];
      const uint64_t     stateTransTab = g_stateTransTab[slice.getDepQuantEnabledIdc()];
      int                lastScanIdx   = (int) area.area() - 1;
      while( lastScanIdx >= 0 && c.buf[scan[lastScanIdx].idx] == 0 )
      {
        lastScanIdx--;
      }
      for( int state = 0, scanIdx = lastScanIdx; scanIdx >= 0; scanIdx-- )
      {
        const unsigned rasterPos = scan[scanIdx].idx;
        const TCoeff   level     = c.buf[rasterPos];
        if( level )
        {
          m.coef[t.coeffOffset + ( rasterPos / area.width ) * t.nzCols + rasterPos % area.width] =
            ( level << 1 ) + ( level > 0 ? -( state & 1 ) : ( state & 1 ) );
        }
        state = int( ( stateTransTab >> ( ( state << 3 ) + ( ( level & 1 ) << 2 ) ) ) & 15 );
      }
    }
  }
  if( signPred )   // the signPred data; the block decides itself unless its pair's Cr is coded
  {
    t.spOffset = (int32_t) m.signPred.size();
    t.spCoded  = std::abs( t.jccr ) == 3 ? -1 : (int32_t) m.rec.size();
    m.signPred.push_back( sign.num );
    m.signPred.push_back( sign.codedSigns );
    m.signPred.insert( m.signPred.end(), sign.xy, sign.xy + SIGN_PRED_MAX_NUM );
    m.signPred.resize( m.signPred.size() + size_t( sign.num ) * ( area.width + area.height ), 0 );
    sign.num = 0;
  }
  if( pred && resi )   // the CPU's prediction and residual, for the comparison
  {
    for( int y = 0; y < (int) area.height; y++ )
    {
      for( int x = 0; x < (int) area.width; x++ )
      {
        m.cpuPred.push_back( pred->at( x, y ) );
        m.cpuResi.push_back( resi->at( x, y ) );
      }
    }
  }
  m.rec.push_back( t );
}

void GpuIntraRec::endPicture( const CodingStructure& cs )
{
  Impl& m = *m_impl;
  m.numPictures++;
  if( m.on && !m.unsupportedWhy.empty() )
  {
    m.numFallback++;
  }
  if( !m.unsupportedWhy.empty() || m.rec.empty() )
  {
    m.numSkipped++;
    if( m.firstSkipReason.empty() )
    {
      m.firstSkipReason = m.unsupportedWhy.empty() ? "no intra blocks" : m.unsupportedWhy;
    }
    return;
  }
  gpu::Context& ctx = *m.ctx;
  const int numComp = cs.pcv->chrFormat == CHROMA_400 ? 1 : 3;

  Params P{};
  size_t planeBytes[3] = {};
  for( int c = 0; c < numComp; c++ )
  {
    const CPelBuf reco = cs.picture->getRecoBuf( ComponentID( c ) );
    P.width[c]  = reco.width;
    P.height[c] = reco.height;
    P.stride[c] = reco.width;
    P.origin[c] = 0;
    planeBytes[c] = size_t( reco.width ) * reco.height * 2;
    m.planes[c].ensure( ctx, planeBytes[c], gpu::Buffer::DEVICE );
    P.plane[c] = m.planes[c].addr;
  }
  // owner maps: per unit and component the block covering it (CodingStructure unit scales, 4:2:0); -1 elsewhere
  std::vector<int32_t> ownerMap[3];
  size_t ownerBytes[3] = {};
  for( int ch = 0; ch < ( numComp > 1 ? 2 : 1 ); ch++ )
  {
    const int sx = ch == 0 ? 2 : 1, sy = ch == 0 ? 2 : 1;   // UnitScaleArray (4:2:0)
    P.unitLog2X[ch] = sx;
    P.unitLog2Y[ch] = sy;
    P.ownerW[ch]    = ( P.width[ch] + ( 1 << sx ) - 1 ) >> sx;
    P.ownerH[ch]    = ( P.height[ch] + ( 1 << sy ) - 1 ) >> sy;
  }
  // with the reconstruction on, intra_deps computes the owner maps and the scheduling data on the GPU; verification
  // computes them here too and compares
  const bool hostDeps = !m.on;
  for( int c = 0; c < numComp && hostDeps; c++ )
  {
    const int ch = c == 0 ? 0 : 1;
    ownerMap[c].assign( size_t( P.ownerW[ch] ) * P.ownerH[ch], -1 );
  }
  for( size_t r = 0; r < m.rec.size() && hostDeps; r++ )
  {
    const Tu& t  = m.rec[r];
    const int ch = t.comp == 0 ? 0 : 1;
    if( t.derive & DERIVE_NODE )   // no samples
    {
      continue;
    }
    for( int uy = t.y >> P.unitLog2Y[ch]; uy <= ( t.y + t.h - 1 ) >> P.unitLog2Y[ch]; uy++ )
    {
      for( int ux = t.x >> P.unitLog2X[ch]; ux <= ( t.x + t.w - 1 ) >> P.unitLog2X[ch]; ux++ )
      {
        ownerMap[t.comp][size_t( uy ) * P.ownerW[ch] + ux] = (int32_t) r;
      }
    }
  }
  for( int c = 0; c < numComp; c++ )
  {
    const int ch  = c == 0 ? 0 : 1;
    ownerBytes[c] = size_t( P.ownerW[ch] ) * P.ownerH[ch] * 4;
    m.owner[c].ensure( ctx, ownerBytes[c], gpu::Buffer::DEVICE );
    P.owner[c] = m.owner[c].addr;
  }
  // dependencies: the blocks owning a sample a block reads - geometry, decoding order and syntax only. Per reference
  // line: the availability the CPU sees (corner flag, available prefix of each unit group), the highest top / left
  // index the prediction can read (a superset), then for each sample in that range its owner, or for an unavailable
  // sample the owner of its padding source (refSource, the kernel's fill rule); an ISP sub-partition also depends on
  // the previous one, a block with sign prediction on the owners of its border, and with joint CbCr a block whose
  // residual the decision in the other block writes on that block
  const int numTus = (int) m.rec.size();
  std::vector<std::vector<int>> deps( numTus );
  {
    constexpr int PAD = 6;   // interpFilterNumPadded
    std::vector<int> mark( numTus, -1 );
    for( int i = 0; i < numTus && hostDeps; i++ )
    {
      Tu&  t = m.rec[i];
      auto addBlock = [&]( int b )
      {
        if( b >= 0 && b < i && mark[b] != i )
        {
          mark[b] = i;
          deps[i].push_back( b );
        }
      };
      if( t.derive & DERIVE_NODE )   // a CU's mode: the derived modes it reads, from those CUs' nodes
      {
        const Cu& c = m.cus[t.cu];
        if( c.kind == CU_MPM )
        {
          for( int k = 0; k < MPM_NEIGHBOURS; k++ )
          {
            if( c.nbr[k] >= 0 )
            {
              addBlock( m.cus[c.nbr[k]].node );
            }
          }
        }
        else
        {
          addBlock( m.cus[c.lumaCu].node );
        }
        continue;
      }
      const int ch = t.comp == 0 ? 0 : 1;
      auto own = [&]( int c, int x, int y ) -> int
      {
        if( x < 0 || y < 0 || ( x >> P.unitLog2X[ch] ) >= P.ownerW[ch] || ( y >> P.unitLog2Y[ch] ) >= P.ownerH[ch] )
        {
          return -1;
        }
        return ownerMap[c][size_t( y >> P.unitLog2Y[ch] ) * P.ownerW[ch] + ( x >> P.unitLog2X[ch] )];
      };
      const int availComp = t.comp == 0 ? 0 : 1;
      auto avail = [&]( int x, int y ) { const int a = own( availComp, x, y ); return a >= 0 && a < i; };
      auto addOwner = [&]( int sx, int sy )
      {
        const int o = own( t.comp, sx, sy );
        if( o >= 0 && o < i && mark[o] != i )
        {
          mark[o] = i;
          deps[i].push_back( o );
        }
      };
      memset( t.avail, 0, sizeof( t.avail ) );
      memset( t.readMax, 0, sizeof( t.readMax ) );
      memset( t.bfAvail, 0, sizeof( t.bfAvail ) );
      t.ispSide = 0;
      // the area the references are filled for: the block with arms of 8x its size, for ISP the CU with arms of twice
      // its size; the prediction's arm lengths (m_topRefLength / m_leftRefLength): 8x the block, for ISP the CU's
      // size plus the prediction region's
      const bool isp = t.isp != 0;
      const int  fx = isp ? t.cuX : t.x, fy = isp ? t.cuY : t.y, fw = isp ? t.cuW : t.w, fh = isp ? t.cuH : t.h;
      const int  predSize = isp ? 2 * t.cuW : t.w << 3, predHSize = isp ? 2 * t.cuH : t.h << 3;
      const int  numAbove = std::max( fw / t.unitW, 1 ), numLeft = std::max( fh / t.unitH, 1 );
      const int  tAU = predSize / t.unitW, tLU = predHSize / t.unitH;
      const int  W = t.regW, H = t.h, topLen = isp ? t.cuW + W : W << 3, leftLen = isp ? t.cuH + H : H << 3;
      if( isp )   // the side of the sub-partition along the split (initIntraPatternChTypeISP's isLeftAvail / isAboveAvail)
      {
        t.ispSide = ( t.isp == HOR_INTRA_SUBPARTITIONS ? avail( t.cuX - 1, t.y ) : avail( t.regX, t.cuY - 1 ) ) ? 1 : 0;
      }
      // the read range of line ln (0: multiRefIdx k, 1: intra fusion's k + 1, non-integer slopes) for the mode of tm
      auto readRange = [&]( const Tu& tm, const int ln, int& topMax, int& leftMax )
      {
        const int mrl = tm.mrl + ln;
        const int rf  = tm.refFilter ? 1 : 0;
        if( tm.dirMode == PLANAR_IDX )   // never with MRL; xPredIntraPlanar reads entries 1 .. w + 1 anyway
        {
          topMax  = W + 1 + rf;
          leftMax = H + 1 + rf;
        }
        else if( tm.dirMode == DC_IDX )   // xPredIntraDc: from entry multiRefIdx + 1 on
        {
          topMax  = W + 1 + mrl;
          leftMax = H + 1 + mrl;
        }
        else
        {
          const bool ver = tm.isModeVer != 0;
          const int  w = ver ? W : H, h = ver ? H : W, refLength = ver ? topLen : leftLen, sideLength = ver ? leftLen : topLen;
          const int  angle = tm.intraPredAngle;
          int mainMax, sideMax;
          if( ln == 0 )   // xPredIntraAng reads refMain + multiRefIdx from deltaPos = angle * (1 + multiRefIdx) on
          {
            if( angle == 0 )
            {
              mainMax = w + 1 + mrl;
              sideMax = tm.applyPDPC ? h + 1 : 1 + mrl;   // PDPC only without MRL
            }
            else if( angle < 0 )
            {
              mainMax = w + PAD + 2 + mrl;   // the copied part (the extension reads the side up to sizeSide + 2 + mrl)
              sideMax = h + PAD + 2 + mrl;
            }
            else
            {
              // beyond refLength + mrl: its replicated value
              mainMax = std::min( refLength + mrl, ( ( angle * ( h + mrl ) ) >> 5 ) + w + 10 + mrl );
              sideMax = 1 + mrl;
              if( tm.applyPDPC )
              {
                const int limit = std::min( 3 << tm.angularScale, w );
                sideMax = tm.gradPDPC ? h + 1 : std::min( sideLength, h + ( ( 256 + limit * tm.absInvAngle ) >> 9 ) + 1 );
              }
            }
            mainMax += rf;
            sideMax += rf;
          }
          else   // the fusion line: refMain2nd + multiRefIdx2nd, deltaPos one angle further
          {
            if( angle < 0 )
            {
              mainMax = w + PAD + 9 + mrl;
              sideMax = std::min( sideLength + mrl, ( ( ( h + 1 + PAD ) * tm.absInvAngle + 8 ) >> 4 >> 5 ) + 2 + mrl );
            }
            else
            {
              mainMax = std::min( refLength + mrl, ( ( angle * ( h + mrl ) ) >> 5 ) + w + 11 + mrl );
              sideMax = mrl;
            }
          }
          topMax  = ver ? mainMax : sideMax;
          leftMax = ver ? sideMax : mainMax;
        }
        if( tm.bf == 2 && ln == 0 )   // the filter's gate: the texture of the whole reference arrays (never with MRL / ISP)
        {
          topMax  = topLen;
          leftMax = leftLen;
        }
      };
      // a mode intra_rec derives is not known here: the union of the read ranges of the modes the syntax leaves
      // (Cu::modes, else all), the fusion line where one of them reads it, the bilateral filter's neighbours where it
      // applies to one
      const bool derived  = ( t.derive & DERIVE_MODE ) != 0;
      const Cu*  cuRec    = derived ? &m.cus[t.cu] : nullptr;
      const int  numModes = !derived ? 1 : cuRec->modes[0] < 0 ? NUM_LUMA_MODE : cuRec->modes[1] < 0 ? 1 : 2;
      bool       bfNeeded = !derived && t.bf != 0;
      for( int ln = 0; ln < 2; ln++ )
      {
        const int mrl = t.mrl + ln;   // the reference line (multiRefIdx; the fusion line is the next one)
        int topMax = -1, leftMax = -1;   // -1: the line is not read
        for( int k = 0; k < numModes; k++ )
        {
          Tu tm = t;
          if( derived )
          {
            applyModeParams( tm, cuRec->modes[0] < 0 ? k : cuRec->modes[k], cuRec->plIdx );
          }
          if( ln == 1 && !tm.fusion )
          {
            continue;
          }
          int a, b;
          readRange( tm, ln, a, b );
          topMax   = std::max( topMax, a );
          leftMax  = std::max( leftMax, b );
          bfNeeded = bfNeeded || tm.bf != 0;
        }
        if( topMax < 0 )
        {
          break;
        }
        // never beyond the arms (the ISP prediction reads only within them; the copies beyond are never read)
        t.readMax[ln][0] = std::max( mrl, std::min( topMax, topLen + mrl ) );
        t.readMax[ln][1] = std::max( mrl, std::min( leftMax, leftLen + mrl ) );

        // availability: the corner flag and the available prefix of each unit group (in the CPU's scan direction:
        // the MRL groups from the corner on). Without ISP the groups are scanned only over the units the read range
        // overlaps (at least one: a group's first unit decides whether any of it is available): units beyond the
        // range change neither a sample in range nor its padding source, which comes from the nearest available unit
        // before it in the CPU's unit order (an interval end, or a group's fixed start) or from the first available
        // unit, whose position only depends on whether its group's prefix is 0. ISP sub-partitions read the CU's
        // arms at their offset: those groups are scanned whole.
        const int x0 = fx - 1 - mrl, y0 = fy - 1 - mrl;
        const int lM = mrl / t.unitH, aM = mrl / t.unitW;
        int*      av = t.avail[ln];
        const int unitsTop  = isp ? tAU : t.readMax[ln][0] >= 1 + mrl ? ( t.readMax[ln][0] - 1 - mrl ) / t.unitW + 1 : 0;
        const int unitsLeft = isp ? tLU : t.readMax[ln][1] >= 1 + mrl ? ( t.readMax[ln][1] - 1 - mrl ) / t.unitH + 1 : 0;
        const int limA  = std::max( 1, std::min( numAbove, unitsTop ) );
        const int limAR = std::max( 1, std::min( tAU - numAbove, unitsTop - numAbove ) );
        const int limL  = std::max( 1, std::min( numLeft, unitsLeft ) );
        const int limBL = std::max( 1, std::min( tLU - numLeft, unitsLeft - numLeft ) );
        av[0] = avail( x0, y0 ) ? 1 : 0;
        int k;
        for( k = 0; k < limA && avail( fx + k * t.unitW, y0 ); k++ ) {}
        av[1] = k;
        for( k = 0; k < limAR && avail( fx + fw - 1 + t.unitW + k * t.unitW, y0 ); k++ ) {}
        av[2] = k;
        for( k = 0; k < limL && avail( x0, fy + k * t.unitH ); k++ ) {}
        av[3] = k;
        for( k = 0; k < limBL && avail( x0, fy + fh - 1 + t.unitH + k * t.unitH ); k++ ) {}
        av[4] = k;
        for( k = 0; k < lM && avail( x0, fy - lM * t.unitH + k * t.unitH ); k++ ) {}   // left MRL, downwards
        av[5] = k;
        for( k = 0; k < aM && avail( fx - aM * t.unitW + k * t.unitW, y0 ); k++ ) {}   // above MRL, rightwards
        av[6] = k;

        if( bfNeeded && ln == 0 )   // the bilateral filter's neighbours (rows y-2, y-1; columns x-2, x-1)
        {
          // its padding (bilateralFilterPredDiamond5x5Clip): line 0's corner and above / left units of the CU (the
          // block, for ISP the CU, whose origin it pads at), also with MRL; a superset of the samples it copies
          int bfA, bfL;
          for( bfA = 0; bfA < numAbove && avail( fx + bfA * t.unitW, fy - 1 ); bfA++ ) {}
          for( bfL = 0; bfL < numLeft && avail( fx - 1, fy + bfL * t.unitH ); bfL++ ) {}
          t.bfAvail[0] = avail( fx - 1, fy - 1 ) ? 1 : 0;
          t.bfAvail[1] = bfA;
          t.bfAvail[2] = bfL;
          if( t.bfAvail[0] )
          {
            addOwner( fx - 2, fy - 2 ); addOwner( fx - 1, fy - 2 ); addOwner( fx - 2, fy - 1 ); addOwner( fx - 1, fy - 1 );
          }
          for( int x = 0; x < bfA * t.unitW; x++ )
          {
            addOwner( fx + x, fy - 2 );
            addOwner( fx + x, fy - 1 );
          }
          for( int y = 0; y < bfL * t.unitH; y++ )
          {
            addOwner( fx - 2, fy + y );
            addOwner( fx - 1, fy + y );
          }
        }
        // the owners of the samples in range (the kernel's fill rule)
        for( int side = 0; side < 2; side++ )
        {
          for( int idx = 0; idx <= t.readMax[ln][side]; idx++ )
          {
            int sx, sy;
            if( refSource( t, ln, side == 0, idx, sx, sy ) )
            {
              addOwner( sx, sy );
            }
          }
        }
      }
      if( t.ispPrev >= 0 && mark[t.ispPrev] != i )   // the chain of an ISP CU's sub-partitions
      {
        mark[t.ispPrev] = i;
        deps[i].push_back( t.ispPrev );
      }
      if( derived )   // the mode, from the CU's node
      {
        addBlock( cuRec->node );
      }
      if( t.spCoded >= 0 )   // sign prediction's border (TU::predBorderResi): the two columns left of / rows above the block
      {
        for( int k = 0; k < t.h && t.x > 0; k++ )
        {
          addOwner( t.x - 1, t.y + k );
          addOwner( t.x - 2, t.y + k );
        }
        for( int k = 0; k < t.w && t.y > 0; k++ )
        {
          addOwner( t.x + k, t.y - 1 );
          addOwner( t.x + k, t.y - 2 );
        }
      }
      // joint CbCr with sign prediction: the residual comes from the decision in the other block of the pair, before
      // or after this one
      if( t.resiFrom >= 0 && mark[t.resiFrom] != i )
      {
        mark[t.resiFrom] = i;
        deps[i].push_back( t.resiFrom );
      }
      if( t.lmcs & LMCS_CHROMA_SCALE )   // the luma column left of and the row above the CU, clipped to the picture
      {
        auto addLuma = [&]( int sx, int sy )
        {
          const int o = ownerMap[COMPONENT_Y][size_t( sy >> P.unitLog2Y[0] ) * P.ownerW[0] + ( sx >> P.unitLog2X[0] )];
          if( o >= 0 && o < i && mark[o] != i )
          {
            mark[o] = i;
            deps[i].push_back( o );
          }
        };
        if( t.crsAvail & 1 )
        {
          for( int k = 0; k < t.crsH; k++ ) addLuma( t.crsX - 1, std::min( t.crsY + k, P.height[0] - 1 ) );
        }
        if( t.crsAvail & 2 )
        {
          for( int k = 0; k < t.crsW; k++ ) addLuma( std::min( t.crsX + k, P.width[0] - 1 ), t.crsY - 1 );
        }
      }
    }
  }
  double         ms      = 0;
  const int16_t* gPlanes = nullptr;   // the downloaded results
  const int16_t* gPredDl = nullptr;
  const Tu*      gTusDl  = nullptr;
  const int32_t* gOffDl  = nullptr;
  const int32_t* gListDl = nullptr;
  const int32_t* gParamsDl = nullptr;
  // the scheduling buffer (intra_deps fills it): pending counts, the dependents of each block (CSR, edgeCap entries),
  // the queue, the counters, then intra_deps' own arrays
  const size_t n = size_t( numTus );
  m.edgeCap      = std::max( m.edgeCap, n * 8 );
  const size_t tuBytes = m.rec.size() * sizeof( Tu ), coefBytes = m.coef.size() * 4, dbgBytes = m.numSamples * 2;
  const size_t spBytes = m.signPred.size() * 4;
  m.tus.ensure( ctx, tuBytes, gpu::Buffer::DEVICE );
  m.coeffs.ensure( ctx, std::max<size_t>( coefBytes, 4 ), gpu::Buffer::DEVICE );
  m.signPredBuf.ensure( ctx, std::max<size_t>( spBytes, 4 ), gpu::Buffer::DEVICE );
  m.dbgPred.ensure( ctx, dbgBytes, gpu::Buffer::DEVICE );
  m.dbgResi.ensure( ctx, dbgBytes, gpu::Buffer::DEVICE );
  P.tus      = m.tus.addr;
  P.coeffs   = m.coeffs.addr;
  P.signPred   = m.signPredBuf.addr;
  P.secKernels = m.sh->secKernels.addr;
  P.filters  = m.sh->filters.addr;
  P.matrices = m.sh->matrices.addr;
  P.dbgPred  = m.dbgPred.addr;
  P.dbgResi  = m.dbgResi.addr;
  P.lumaClipMin = cs.slice->clpRng( COMPONENT_Y ).min;
  P.lumaClipMax = cs.slice->clpRng( COMPONENT_Y ).max;
  P.qcsOffset   = m.sh->qcsOffset;
  // the CU records, then their modes (the derived ones -1 until the CU's node writes them); with verification
  // the parameters intra_rec derives
  const size_t cusBytes = m.cus.size() * sizeof( Cu ), cuBytes = cusBytes + m.cuMode.size() * 4;
  const size_t dbgParamsBytes = m.on ? 4 : n * DBG_PARAMS * 4;
  m.cuBuf.ensure( ctx, std::max<size_t>( cuBytes, 4 ), gpu::Buffer::DEVICE );
  m.dbgParams.ensure( ctx, dbgParamsBytes, gpu::Buffer::DEVICE );
  P.cus       = m.cuBuf.addr;
  P.cuMode    = m.cuBuf.addr + cusBytes;
  P.dbgParams = m.dbgParams.addr;
  std::vector<int32_t> cuData( cuBytes / 4 );
  if( cuBytes )
  {
    memcpy( cuData.data(), m.cus.data(), cusBytes );
    memcpy( cuData.data() + cusBytes / 4, m.cuMode.data(), cuBytes - cusBytes );
  }
  // the picture's LMCS tables: the forward and inverse LUTs and the chroma scale of every average luma value
  // (Reshape::calculateChromaAdj itself)
  std::vector<int32_t> lmcsTable;
  if( m.lmcsUsed )
  {
    Reshape&  rsp = *m.reshape;
    const int lutSize = (int) rsp.getFwdLUT().size();
    lmcsTable.resize( 3 * size_t( lutSize ) );
    for( int v = 0; v < lutSize; v++ )
    {
      lmcsTable[v]               = rsp.getFwdLUT()[v];
      lmcsTable[lutSize + v]     = rsp.getInvLUT()[v];
      lmcsTable[2 * lutSize + v] = rsp.calculateChromaAdj( Pel( v ) );
    }
    m.lmcs.ensure( ctx, lmcsTable.size() * 4, gpu::Buffer::DEVICE );
    P.lmcs        = m.lmcs.addr;
    P.lmcsLutSize = lutSize;
  }
  const size_t lmcsBytes = lmcsTable.size() * 4;

  for( ;; )   // once, again with more room when the dependencies do not fit
  {
    const size_t cap = m.edgeCap;
    size_t       schedInts = 0;
    auto take = [&]( size_t ints ) { const size_t o = schedInts; schedInts += ints; return o; };
    const size_t oCount = take( n ), oOffset = take( n + 1 ), oList = take( cap ), oQueue = take( n ), oCounter = take( 4 );
    const size_t oStart = take( n + 1 ), oTmp = take( cap ), oNumDep = take( n );
    m.sched.ensure( ctx, schedInts * 4, gpu::Buffer::DEVICE );
    P.depCount  = m.sched.addr + oCount * 4;
    P.depOffset = m.sched.addr + oOffset * 4;
    P.depList   = m.sched.addr + oList * 4;
    P.queue     = m.sched.addr + oQueue * 4;
    P.counter   = m.sched.addr + oCounter * 4;
    P.depStart  = m.sched.addr + oStart * 4;
    P.depTmp    = m.sched.addr + oTmp * 4;
    P.numDep    = m.sched.addr + oNumDep * 4;
    P.edgeCap   = (int32_t) cap;

    const size_t upBytes = sizeof( Params ) + tuBytes + coefBytes + lmcsBytes + spBytes + cuBytes;
    const size_t upSign  = sizeof( Params ) + tuBytes + coefBytes + lmcsBytes;   // where the signPred data goes
    const size_t upCu    = upSign + spBytes;                                      // the CU records and modes
    m.upload.ensure( ctx, upBytes, gpu::Buffer::UPLOAD );
    memcpy( m.upload.mapped, &P, sizeof( Params ) );
    memcpy( (char*) m.upload.mapped + sizeof( Params ), m.rec.data(), tuBytes );
    memcpy( (char*) m.upload.mapped + sizeof( Params ) + tuBytes, m.coef.data(), coefBytes );
    memcpy( (char*) m.upload.mapped + sizeof( Params ) + tuBytes + coefBytes, lmcsTable.data(), lmcsBytes );
    memcpy( (char*) m.upload.mapped + upSign, m.signPred.data(), spBytes );
    memcpy( (char*) m.upload.mapped + upCu, cuData.data(), cuBytes );
    // downloaded: the dependency count (overflow check), with verification the CPU comparison data, the GPU's
    // scheduling data and the derived parameters, then the planes
    size_t downBytes = 8;
    const size_t dPred = downBytes, dResi = dPred + dbgBytes, dTus = dResi + dbgBytes, dOffset = dTus + tuBytes;
    const size_t dList = dOffset + ( n + 1 ) * 4, dParams = dList + cap * 4;
    if( !m.on )
    {
      downBytes = dParams + dbgParamsBytes;
    }
    size_t dPlane[3] = {};
    for( int c = 0; c < numComp; c++ )
    {
      dPlane[c] = downBytes;
      downBytes += planeBytes[c];
    }
    m.download.ensure( ctx, downBytes, gpu::Buffer::DOWNLOAD );

    gpu::Stream& s = *m.stream;
    s.begin();
    s.copy( m.upload, 0, m.params, 0, sizeof( Params ) );
    s.copy( m.upload, sizeof( Params ), m.tus, 0, tuBytes );
    if( coefBytes )
    {
      s.copy( m.upload, sizeof( Params ) + tuBytes, m.coeffs, 0, coefBytes );
    }
    if( lmcsBytes )
    {
      s.copy( m.upload, sizeof( Params ) + tuBytes + coefBytes, m.lmcs, 0, lmcsBytes );
    }
    if( spBytes )
    {
      s.copy( m.upload, upSign, m.signPredBuf, 0, spBytes );
    }
    if( cuBytes )
    {
      s.copy( m.upload, upCu, m.cuBuf, 0, cuBytes );
    }
    for( int c = 0; c < numComp; c++ )
    {
      s.fill( m.owner[c], 0, ownerBytes[c], 0xFFFFFFFFu );
      s.fill( m.planes[c], 0, ( planeBytes[c] + 3 ) & ~size_t( 3 ), 0 );
    }
    s.fill( m.sched, oQueue * 4, n * 4, 0xFFFFFFFFu );
    s.fill( m.sched, oCounter * 4, 4 * 4, 0 );
    s.fill( m.sched, oNumDep * 4, n * 4, 0 );
    s.barrier();
    const uint32_t t0 = s.timestamp();
    const uint32_t depGroups = (uint32_t) ( ( n + 63 ) / 64 );
    Push push{ m.params.addr, (int32_t) n, m.on ? 0 : 1 };
    s.dispatch( m.sh->kernelResi, &push, (uint32_t) std::min<size_t>( RESI_GROUPS, n ) );   // the residuals
    Push pass{ m.params.addr, (int32_t) n, 0 };
    s.dispatch( m.sh->kernelDeps, &pass, depGroups );   // the owner maps
    s.barrier();
    pass.verify = 1;
    s.dispatch( m.sh->kernelDeps, &pass, depGroups );   // availability, read ranges, dependency counts
    s.barrier();
    PushScan scan{ P.depCount, P.depStart, (int32_t) n, 0 };
    s.dispatch( m.sh->kernelScan, &scan, 1 );
    s.barrier();
    pass.verify = 2;
    s.dispatch( m.sh->kernelDeps, &pass, depGroups );   // the dependencies, the dependents' counts
    s.barrier();
    scan = PushScan{ P.numDep, P.depOffset, (int32_t) n, 0 };
    s.dispatch( m.sh->kernelScan, &scan, 1 );
    s.barrier();
    s.fill( m.sched, oNumDep * 4, n * 4, 0 );
    s.barrier();
    pass.verify = 3;
    s.dispatch( m.sh->kernelDeps, &pass, depGroups );   // the dependents lists, the queue
    s.barrier();
    s.dispatch( m.sh->kernel, &push, (uint32_t) std::min<size_t>( m.numGroups, n ) );
    const uint32_t t1 = s.timestamp();
    s.barrier();
    s.copy( m.sched, ( oStart + n ) * 4, m.download, 0, 4 );
    if( !m.on )
    {
      s.copy( m.dbgPred, 0, m.download, dPred, dbgBytes );
      s.copy( m.dbgResi, 0, m.download, dResi, dbgBytes );
      s.copy( m.tus, 0, m.download, dTus, tuBytes );
      s.copy( m.sched, oOffset * 4, m.download, dOffset, ( n + 1 ) * 4 );
      s.copy( m.sched, oList * 4, m.download, dList, cap * 4 );
      s.copy( m.dbgParams, 0, m.download, dParams, dbgParamsBytes );
    }
    for( int c = 0; c < numComp; c++ )
    {
      s.copy( m.planes[c], 0, m.download, dPlane[c], planeBytes[c] );
    }
    s.submitAndWait();
    const size_t numEdges = (size_t) *(const int32_t*) m.download.mapped;
    if( numEdges > cap )   // nothing was reconstructed: again with room for them
    {
      m.edgeCap = numEdges + numEdges / 4;
      continue;
    }
    if( numEdges > cap - cap / 4 )
    {
      m.edgeCap = numEdges + numEdges / 4;   // the next pictures
    }
    ms = s.elapsedNs( t0, t1 ) * 1e-6;
    gPlanes  = (const int16_t*) ( (const char*) m.download.mapped + dPlane[0] );
    gPredDl  = (const int16_t*) ( (const char*) m.download.mapped + dPred );
    gTusDl   = (const Tu*) ( (const char*) m.download.mapped + dTus );
    gOffDl   = (const int32_t*) ( (const char*) m.download.mapped + dOffset );
    gListDl  = (const int32_t*) ( (const char*) m.download.mapped + dList );
    gParamsDl = (const int32_t*) ( (const char*) m.download.mapped + dParams );
    break;
  }
  m.gpuMs += ms;

  if( m.on )   // the GPU's picture is the reconstruction
  {
    const int16_t* gPlane = gPlanes;
    for( int c = 0; c < numComp; c++ )
    {
      PelBuf reco = cs.picture->getRecoBuf( ComponentID( c ) );
      for( int y = 0; y < P.height[c]; y++ )
      {
        memcpy( reco.bufAt( 0, y ), gPlane + size_t( y ) * P.width[c], P.width[c] * sizeof( Pel ) );
      }
      gPlane += size_t( P.width[c] ) * P.height[c];
    }
    m.numOn++;
    return;
  }
  // compare: per block prediction and residual, then the whole picture
  const int16_t* gPred = gPredDl;
  const int16_t* gResi = gPred + m.cpuPred.size();
  std::string first;
  int numBad = 0;
  // the GPU's scheduling data against the host's: availability, read ranges, the dependency graph
  std::vector<std::vector<int>> dependents( numTus );
  for( int i = 0; i < numTus; i++ ) for( int d: deps[i] ) dependents[d].push_back( i );
  {
    for( int i = 0; i < numTus && first.empty(); i++ )
    {
      const Tu& h = m.rec[i];
      const Tu& g = gTusDl[i];
      std::vector<int> gd( gListDl + gOffDl[i], gListDl + gOffDl[i + 1] );
      std::sort( gd.begin(), gd.end() );
      gd.erase( std::unique( gd.begin(), gd.end() ), gd.end() );
      std::sort( dependents[i].begin(), dependents[i].end() );
      if( memcmp( h.avail, g.avail, sizeof( h.avail ) ) || memcmp( h.readMax, g.readMax, sizeof( h.readMax ) )
          || memcmp( h.bfAvail, g.bfAvail, sizeof( h.bfAvail ) ) || h.ispSide != g.ispSide || gd != dependents[i] )
      {
        first = "scheduling data of block " + std::to_string( i ) + " differ (availability, read ranges or dependents)";
      }
    }
  }
  // the modes and parameters intra_rec derived against the CPU's: every field the kernels read for the mode (the
  // angular ones of angular modes, PDPC's scale for positive angles)
  for( int i = 0; i < numTus && first.empty(); i++ )
  {
    const Tu& h = m.rec[i];
    if( !( h.derive & DERIVE_MODE ) )
    {
      continue;
    }
    const int32_t* g = gParamsDl + size_t( i ) * DBG_PARAMS;
    const char*    field = nullptr;
    int            cpu = 0, gpu = 0;
    auto check = [&]( const char* name, int a, int b )
    {
      if( !field && a != b )
      {
        field = name;
        cpu   = a;
        gpu   = b;
      }
    };
    check( "mode", h.dirMode, g[0] );
    check( "isModeVer", h.isModeVer, g[1] );
    if( h.dirMode > DC_IDX )
    {
      check( "intraPredAngle", h.intraPredAngle, g[2] );
      check( "absInvAngle", h.absInvAngle, g[3] );
      if( h.intraPredAngle > 0 )
      {
        check( "gradPDPC", h.gradPDPC, g[7] );
        check( "angularScale", h.angularScale, g[8] );
      }
    }
    check( "useCubic", h.useCubic, g[4] );
    check( "refFilter", h.refFilter, g[5] );
    check( "applyPDPC", h.applyPDPC, g[6] );
    check( "fusion", h.fusion, g[9] );
    check( "bf", h.bf, g[10] );
    check( "plIdx", h.plIdx, g[11] );
    if( field )
    {
      char buf[200];
      snprintf( buf, sizeof( buf ), "derived %s of block %d (comp %d at %d,%d %dx%d): CPU %d GPU %d", field, i, h.comp, h.x,
                h.y, h.w, h.h, cpu, gpu );
      first = buf;
    }
  }
  for( size_t i = 0; i < m.rec.size() && first.empty(); i++ )
  {
    const Tu& t = m.rec[i];
    for( int k = 0; k < t.w * t.h; k++ )
    {
      const int o = t.dbgOffset + k;
      const bool predBad = gPred[o] != m.cpuPred[o], resiBad = gResi[o] != m.cpuResi[o];
      if( predBad || resiBad )
      {
        char buf[320];
        snprintf( buf, sizeof( buf ),
                  "block %d (comp %d at %d,%d %dx%d, mode %d, angle %d, ver %d, cubic %d, cbf %d): %s at (%d,%d) CPU %d GPU %d",
                  (int) i, t.comp, t.x, t.y, t.w, t.h, t.dirMode, t.intraPredAngle, t.isModeVer, t.useCubic, t.cbf,
                  predBad ? "prediction" : "residual", k % t.w, k / t.w, predBad ? m.cpuPred[o] : m.cpuResi[o],
                  predBad ? gPred[o] : gResi[o] );
        first = buf;
        // the blocks owning the reference row above (x - 1 ... x + 2 w) and the column left (... y + 2 h)
        const int ch = t.comp == 0 ? 0 : 1;
        auto own = [&]( int x, int y ) -> int
        {
          if( x < 0 || y < 0 || ( x >> P.unitLog2X[ch] ) >= P.ownerW[ch] || ( y >> P.unitLog2Y[ch] ) >= P.ownerH[ch] )
          {
            return -1;
          }
          return ownerMap[t.comp][size_t( y >> P.unitLog2Y[ch] ) * P.ownerW[ch] + ( x >> P.unitLog2X[ch] )];
        };
        std::string above = "; owners above:", left = "; left:";
        for( int x = t.x - 1; x < t.x + 2 * t.w; x += 1 << P.unitLog2X[ch] )
        {
          above += " " + std::to_string( own( x, t.y - 1 ) );
        }
        for( int y = t.y - 1; y < t.y + 2 * t.h; y += 1 << P.unitLog2Y[ch] )
        {
          left += " " + std::to_string( own( t.x - 1, y ) );
        }
        first += above + left;
        break;
      }
    }
  }
  const int16_t* gPlane = gPlanes;
  for( int c = 0; c < numComp; c++ )
  {
    const CPelBuf reco = cs.picture->getRecoBuf( ComponentID( c ) );
    for( int y = 0; y < P.height[c]; y++ )
    {
      for( int x = 0; x < P.width[c]; x++ )
      {
        if( gPlane[y * P.width[c] + x] != reco.at( x, y ) )
        {
          if( numBad == 0 && first.empty() )
          {
            char buf[160];
            snprintf( buf, sizeof( buf ), "picture comp %d at (%d,%d): CPU %d GPU %d", c, x, y, reco.at( x, y ), gPlane[y * P.width[c] + x] );
            first = buf;
          }
          numBad++;
        }
      }
    }
    gPlane += P.width[c] * P.height[c];
  }
  m.numVerified++;
  const bool bad = numBad > 0 || !first.empty();
  m.numMismatch += bad ? 1 : 0;
  // the longest chain of blocks that wait for each other, in a topological order (a residual dependency of joint CbCr
  // with sign prediction can point forward)
  std::vector<int> depth( numTus, 1 ), waiting( numTus ), ready;
  for( int i = 0; i < numTus; i++ )
  {
    waiting[i] = (int) deps[i].size();
    if( waiting[i] == 0 )
    {
      ready.push_back( i );
    }
  }
  int maxDepth = 0;
  while( !ready.empty() )
  {
    const int b = ready.back();
    ready.pop_back();
    maxDepth = std::max( maxDepth, depth[b] );
    for( int d: dependents[b] )
    {
      depth[d] = std::max( depth[d], depth[b] + 1 );
      if( --waiting[d] == 0 )
      {
        ready.push_back( d );
      }
    }
  }
  int numFilter = 0, numPdpc = 0, numGrad = 0, numFusion = 0, numBf = 0, numNodes = 0;
  for( const Tu& t: m.rec )
  {
    numFilter += t.refFilter;
    numPdpc += t.applyPDPC;
    numGrad += t.applyPDPC && t.gradPDPC;
    numFusion += t.fusion;
    numBf += t.bf != 0;
    numNodes += ( t.derive & DERIVE_NODE ) != 0;
  }
  printf( "\n[gpu-intra] POC %d: %d blocks, chain %d (ref filter %d, PDPC %d, gradient PDPC %d, fusion %d, bilateral filter candidates %d, mode nodes %d), GPU %.2f ms: %s%s%s", m.poc,
          (int) m.rec.size() - numNodes, maxDepth, numFilter, numPdpc, numGrad, numFusion, numBf, numNodes, ms, bad ? "MISMATCH, " : "identical",
          bad ? first.c_str() : "", numBad ? ( ", " + std::to_string( numBad ) + " picture samples differ" ).c_str() : "" );
}
