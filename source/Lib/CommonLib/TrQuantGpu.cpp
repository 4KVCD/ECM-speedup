/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
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

/** \file     TrQuantGpu.cpp
    \brief    The decisions of the inverse transform, recorded for the GPU decoder
*/

#include "TrQuant.h"
#include "DepQuant.h"
#include "GpuRecon.h"
#include "UnitTools.h"
#include "CodingStructure.h"
#include "Rom.h"

GpuReconHooks* g_gpuReconHooks = nullptr;

// The record functions follow the code that this configuration of TypeDef.h compiles; with other settings the
// inverse transform stays on the CPU.
#define GPU_RECON_TRANSFORM                                                                                             \
  ( JVET_AL0181_ASBT && JVET_AH0103_LOW_DELAY_LFNST_NSPT && JVET_AC0130_NSPT && JVET_W0119_LFNST_EXTENSION              \
    && AHG7_LN_TOOLOFF_CFG && JVET_AK0217_INTRA_MTSS && JVET_AK0187_IMPLICIT_MTS_LUT_EXTENSION && TCQ_8STATES           \
    && JVET_AI0136_ADAPTIVE_DUAL_TREE && INTRA_RM_SMALL_BLOCK_SIZE_CONSTRAINTS && TRANSFORM_SIMD_OPT                    \
    && JVET_AN0095_QUANTIZATION_CENTER_SHIFT && JVET_AJ0061_TIMD_MERGE && JVET_AM0074_INTRA_MERGE                       \
    && JVET_AJ0175_NSPT_FOR_NONREG_MODES && AHG7_MTS_TOOLOFF_CFG && TU_256 && JVET_R0351_HIGH_BIT_DEPTH_SUPPORT )

void Quant::gpuDequantParams( const TransformUnit &tu, const ComponentID &compID, const QpParam &cQP, GpuInvTransformJob &job )
{
  // Quant::dequant
  const SPS      *sps                   = tu.cs->sps;
  const CompArea &area                  = tu.blocksResidual[compID];
  const uint32_t  width                 = area.width;
  const uint32_t  height                = area.height;
  const int       maxLog2TrDynamicRange = sps->getMaxLog2TrDynamicRange( toChannelType( compID ) );
  const bool      isTransformSkip       = tu.mtsIdx[compID] == MTS_SKIP;
  const bool      disableSMForLFNST     = tu.cs->slice->getExplicitScalingListUsed() ? sps->getDisableScalingMatrixForLfnstBlks() : false;
  const bool      isLfnstApplied        = tu.cu->lfnstIdx > 0 && ( tu.cu->separateTree ? true : isLuma( compID ) );
  const bool      disableSMForACT       = sps->getScalingMatrixForAlternativeColourSpaceDisabledFlag()
                                         && ( sps->getScalingMatrixDesignatedColourSpaceFlag() == tu.cu->colorTransform );
  const bool      enableScalingLists    = getUseScalingList( width, height, isTransformSkip, isLfnstApplied, disableSMForLFNST, disableSMForACT );
  const int       scalingListType       = getScalingListType( tu.cu->predMode, compID );
  const int       channelBitDepth       = sps->getBitDepth( toChannelType( compID ) );

  job.depQuant = false;
  job.coefMin  = -( 1 << maxLog2TrDynamicRange );
  job.coefMax  = ( 1 << maxLog2TrDynamicRange ) - 1;
  job.bdpcm    = 0;
  if( ( tu.cu->bdpcmMode && isLuma( compID ) ) || ( tu.cu->bdpcmModeChroma && isChroma( compID ) ) )
  {
    job.bdpcm = ( isLuma( compID ) ? tu.cu->bdpcmMode == 1 : tu.cu->bdpcmModeChroma == 1 ) ? 1 : 2;
  }

  const bool bClipTransformShiftTo0 = !isTransformSkip && sps->getSpsRangeExtension().getExtendedPrecisionProcessingFlag();
  const int  originalTransformShift = getTransformShift( channelBitDepth, area.size(), maxLog2TrDynamicRange );
  const bool needSqrtAdjustment     = TU::needsBlockSizeTrafoScale( tu, compID );
  const int  iTransformShift        = ( bClipTransformShiftTo0 ? std::max<int>( 0, originalTransformShift ) : originalTransformShift ) + ( needSqrtAdjustment ? -1 : 0 );
  const int  qpPer                  = cQP.per( isTransformSkip );
  const int  qpRem                  = cQP.rem( isTransformSkip );
  const int  rightShift             = ( IQUANT_SHIFT - ( ( isTransformSkip ? 0 : iTransformShift ) + qpPer ) ) + ( enableScalingLists ? LOG2_SCALING_LIST_NEUTRAL_VALUE : 0 );

  uint32_t targetInputBitDepth;
  if( enableScalingLists )
  {
    const uint32_t dequantCoefBits = 1 + IQUANT_SHIFT + SCALING_LIST_BITS;
    targetInputBitDepth = std::min<uint32_t>( maxLog2TrDynamicRange + 1, ( ( sizeof( Intermediate_Int ) * 8 ) + rightShift ) - dequantCoefBits );
    job.dequantCoef     = getDequantCoeff( scalingListType, qpRem, floorLog2( width ), floorLog2( height ) );
  }
  else
  {
    job.scale           = g_invQuantScales[needSqrtAdjustment ? 1 : 0][qpRem];
    targetInputBitDepth = std::min<uint32_t>( maxLog2TrDynamicRange + 1, ( ( sizeof( Intermediate_Int ) * 8 ) + rightShift ) - ( IQUANT_SHIFT + 1 ) );
  }
  job.scalingList = enableScalingLists;
  job.shift       = rightShift;
  job.inputMin    = -( 1 << ( targetInputBitDepth - 1 ) );
  job.inputMax    = ( 1 << ( targetInputBitDepth - 1 ) ) - 1;
  job.qcs         = sps->getUseQcs();
  job.qcsTable    = qcsTable();
}

void DepQuant::gpuDequantParams( const TransformUnit &tu, const ComponentID &compID, const QpParam &cQP, GpuInvTransformJob &job )
{
  // DepQuant::dequant and DQIntern::Quantizer::dequantBlock
  const bool useRegularResidualCoding = tu.cu->slice->getTSResidualCodingDisabledFlag() || tu.mtsIdx[compID] != MTS_SKIP;
  if( !tu.cs->slice->getDepQuantEnabledIdc() || !useRegularResidualCoding )
  {
    Quant::gpuDequantParams( tu, compID, cQP, job );
    return;
  }
  const SPS      &sps             = *tu.cs->sps;
  const int       qpDQ            = cQP.Qp( tu.mtsIdx[compID] == MTS_SKIP ) + 1;
  const int       qpPer           = qpDQ / 6;
  const int       qpRem           = qpDQ - 6 * qpPer;
  const CompArea &rect            = tu.blocksResidual[compID];
  const int       width           = rect.width;
  const int       height          = rect.height;
  const uint32_t  scalingListType = getScalingListType( tu.cu->predMode, compID );
  CHECK( scalingListType >= SCALING_LIST_NUM, "Invalid scaling list" );
  const bool disableSMForLFNST  = tu.cs->slice->getExplicitScalingListUsed() ? sps.getDisableScalingMatrixForLfnstBlks() : false;
  const bool isLfnstApplied     = tu.cu->lfnstIdx > 0 && ( tu.cu->separateTree ? true : isLuma( compID ) );
  const bool disableSMForACT    = sps.getScalingMatrixForAlternativeColourSpaceDisabledFlag()
                                 && ( sps.getScalingMatrixDesignatedColourSpaceFlag() == tu.cu->colorTransform );
  const bool enableScalingLists = getUseScalingList( width, height, tu.mtsIdx[compID] == MTS_SKIP, isLfnstApplied, disableSMForLFNST, disableSMForACT );

  const ChannelType chType                = toChannelType( compID );
  const int         channelBitDepth       = sps.getBitDepth( chType );
  const int         maxLog2TrDynamicRange = sps.getMaxLog2TrDynamicRange( chType );
  const int         nomTransformShift     = getTransformShift( channelBitDepth, rect.size(), maxLog2TrDynamicRange );
  const bool        clipTransformShift    = tu.mtsIdx[compID] == MTS_SKIP && sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag();
  const bool        needsSqrt2            = TU::needsSqrt2Scale( tu, compID );
  const int         transformShift        = ( clipTransformShift ? std::max<int>( 0, nomTransformShift ) : nomTransformShift ) + ( needsSqrt2 ? -1 : 0 );
  const int         shift                 = IQUANT_SHIFT + 1 - qpPer - transformShift + ( enableScalingLists ? LOG2_SCALING_LIST_NEUTRAL_VALUE : 0 );

  job.depQuant    = true;
  job.bdpcm       = 0;
  job.scalingList = enableScalingLists;
  job.dequantCoef = enableScalingLists ? getDequantCoeff( scalingListType, qpRem, floorLog2( width ), floorLog2( height ) ) : nullptr;
  job.shift       = shift;
  job.scale       = g_invQuantScales[needsSqrt2 ? 1 : 0][qpRem];
  job.add         = shift < 0 ? 0 : ( ( 1 << shift ) >> 1 );
  job.coefMin     = -( 1 << maxLog2TrDynamicRange );
  job.coefMax     = ( 1 << maxLog2TrDynamicRange ) - 1;
  job.stateTrans  = g_stateTransTab[tu.cs->slice->getDepQuantEnabledIdc()];
  job.qcs         = sps.getUseQcs();
  job.qcsTable    = qcsTable();
  const ScanElement* scan = g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][gp_sizeIdxInfo->idxFrom( width )][gp_sizeIdxInfo->idxFrom( height )];
  job.dqScan.resize( width * height );
  for( int i = 0; i < width * height; i++ )
  {
    job.dqScan[i] = int( scan[i].idx );
  }
}

const int16_t* TrQuant::inverseTransformKernel( int trType, int log2Size )
{
  return m_inverseTransformKernels[trType][log2Size - 1];
}

bool TrQuant::recordInvTransform( TransformUnit &tu, const ComponentID compID, const QpParam &cQP, GpuInvTransformJob &job )
{
#if GPU_RECON_TRANSFORM
  const CompArea &area   = tu.blocksResidual[compID];
  const int       width  = area.width;
  const int       height = area.height;
  CHECK( width > tu.cs->sps->getMaxTbSize() || height > tu.cs->sps->getMaxTbSize(), "Maximal allowed transformation size exceeded!" );
  job.width  = width;
  job.height = height;
  job.coeffs = tu.getCoeffs( compID ).buf;
  job.lfnst  = false;
  job.lfnstScan.clear();
  job.nsptScan.clear();
  job.dqScan.clear();
  job.dequantCoef = nullptr;
  m_quant->gpuDequantParams( tu, compID, cQP, job );

  const SPS& sps                  = *tu.cs->sps;
  const int  maxLog2TrDynamicRange = sps.getMaxLog2TrDynamicRange( toChannelType( compID ) );
  const bool spsIntraLfnstEnabled = ( tu.cu->slice->getSliceType() == I_SLICE && sps.getUseIntraLFNSTISlice() )
                                    || ( tu.cu->slice->getSliceType() != I_SLICE && sps.getUseIntraLFNSTPBSlice() );
  const bool lfnstStage           = ( spsIntraLfnstEnabled && CU::isIntra( *tu.cu ) ) || ( sps.getUseInterLFNST() && CU::isInter( *tu.cu ) );
  const bool allowNSPT            = CU::isNSPTAllowed( tu, compID, width, height, spsIntraLfnstEnabled && CU::isIntra( *tu.cu ) );
  const uint32_t lfnstIdx         = tu.cu->lfnstIdx;

  // xInvLfnst
  if( lfnstStage && lfnstIdx && tu.mtsIdx[compID] != MTS_SKIP && ( tu.cu->separateTree ? true : isLuma( compID ) ) && !allowNSPT )
  {
    const bool ext       = sps.getUseLFNSTExt();
    const bool whge4     = ext ? PU::getUseLFNST16( width, height ) : false;
    const bool whge3     = PU::getUseLFNST8( width, height );
    const int  widthIdx  = gp_sizeIdxInfo->idxFrom( width );
    const int  heightIdx = gp_sizeIdxInfo->idxFrom( height );
    const ScanElement* scan = whge4 ? g_coefTopLeftDiagScan16x16[widthIdx]
                                    : ( whge3 ? g_coefTopLeftDiagScan8x8[widthIdx] : g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][widthIdx][heightIdx] );
    const uint32_t intraMode = PU::getFinalIntraModeForTransform( secondBucket, tu, compID ).first;
    if( lfnstIdx < uint32_t( ext ? 4 : 3 ) )
    {
      const int sbSize     = whge4 ? 16 : ( whge3 ? 8 : 4 );
      const int numIn      = ext ? ( whge4 ? L16H : ( whge3 ? L8H : 16 ) ) : 16;
      const int lutMode    = ext ? g_lfnstLut[intraMode] : g_vvcLfnstLut[intraMode];
      const int index      = lfnstIdx - 1;
      CHECK( index >= 4, "Wrong index" );
      job.lfnst          = true;
      job.lfnstExt       = ext;
      job.lfnstSbSize    = sbSize;
      job.lfnstNumIn     = numIn;
      job.lfnstZeroOut   = PU::getLFNSTMatrixDim( width, height, ext );
      job.lfnstTrSize    = ext ? ( ( sbSize > 8 ) ? L16W_ZO : ( ( sbSize > 4 ) ? L8W_ZO : 16 ) ) : ( ( sbSize > 4 ) ? 48 : 16 );
      job.lfnstTranspose = getTransposeFlag( intraMode );
      job.lfnstMat       = ext ? ( ( sbSize > 8 ) ? g_lfnst16x16[lutMode][index][0] : ( ( sbSize > 4 ) ? g_lfnst8x8[lutMode][index][0] : g_lfnst4x4[lutMode][index][0] ) )
                               : ( ( sbSize > 4 ) ? g_vvcLfnst8x8[lutMode][index][0] : g_vvcLfnst4x4[lutMode][index][0] );
      job.lfnstScan.resize( numIn );
      for( int i = 0; i < numIn; i++ )
      {
        job.lfnstScan[i] = int( scan[i].idx );
      }
    }
  }

  if( tu.mtsIdx[compID] == MTS_SKIP )
  {
    job.mode = GpuInvTransformJob::MODE_SKIP;
    return true;
  }

  // xIT
  const int bitDepth          = sps.getBitDepth( toChannelType( compID ) );
  const int matrixShift       = g_transformMatrixShift[TRANSFORM_INVERSE];
  int       trTypeHor         = DCT2;
  int       trTypeVer         = DCT2;
  if( tu.cu->timdMrg && !tu.cu->lfnstIdx && !sps.getUseImplicitMTS() )
  {
    const int implicitDst7 = PU::canTimdMergeImplicitDst7( tu );
    trTypeHor = ( implicitDst7 & 2 ) ? DST7 : tu.cu->timdmTrType[tu.cu->timdMrg - 1][0];
    trTypeVer = ( implicitDst7 & 1 ) ? DST7 : tu.cu->timdmTrType[tu.cu->timdMrg - 1][1];
    trTypeHor = ( width > sps.getIntraMTSMaxSize() ) ? DCT2 : trTypeHor;
    trTypeVer = ( height > sps.getIntraMTSMaxSize() ) ? DCT2 : trTypeVer;
  }
  else if( tu.cu->intraMergeMode && !tu.cu->lfnstIdx )
  {
    const int implicitDst7 = PU::canIntraMergeImplicitDst7( tu );
    trTypeHor = ( implicitDst7 & 2 ) ? DST7 : DCT2;
    trTypeVer = ( implicitDst7 & 1 ) ? DST7 : DCT2;
    trTypeHor = ( width > sps.getIntraMTSMaxSize() ) ? DCT2 : trTypeHor;
    trTypeVer = ( height > sps.getIntraMTSMaxSize() ) ? DCT2 : trTypeVer;
  }
  else
  {
    getTrTypes( tu, compID, trTypeHor, trTypeVer );
  }
  int skipWidth  = width > JVET_C0024_ZERO_OUT_TH ? width - JVET_C0024_ZERO_OUT_TH : 0;
  int skipHeight = height > JVET_C0024_ZERO_OUT_TH ? height - JVET_C0024_ZERO_OUT_TH : 0;
  if( lfnstStage && tu.cu->lfnstIdx && !allowNSPT )
  {
    if( ( width == 4 && height > 16 ) || ( width > 16 && height == 4 ) )
    {
      skipWidth  = width - 4;
      skipHeight = height - 4;
    }
    else if( width >= 16 && height >= 16 )
    {
      skipWidth  = width - 16;
      skipHeight = height - 16;
    }
    else if( ( width == 8 && height > 16 ) || ( width > 16 && height == 8 ) )
    {
      skipWidth  = width - 8;
      skipHeight = height - 8;
    }
  }
  job.skipWidth  = skipWidth;
  job.skipHeight = skipHeight;
  job.matHor     = width > 1 ? inverseTransformKernel( trTypeHor, floorLog2( width ) ) : nullptr;
  job.matVer     = height > 1 ? inverseTransformKernel( trTypeVer, floorLog2( height ) ) : nullptr;
  if( width > 1 && height > 1 )
  {
    const int shift1 = matrixShift + 1 + COM16_C806_TRANS_PREC;
    const int shift2 = ( matrixShift + maxLog2TrDynamicRange - 1 ) - bitDepth + COM16_C806_TRANS_PREC;
    CHECK( shift1 < 0 || shift2 < 0, "Negative shift" );
    job.shift1 = shift1;
    job.shift2 = shift2;
    if( CU::nsptApplyCond( tu, compID, allowNSPT ) )
    {
      // xInvNspt
      const ScanElement* scan      = g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][gp_sizeIdxInfo->idxFrom( width )][gp_sizeIdxInfo->idxFrom( height )];
      const uint32_t     intraMode = PU::getFinalIntraModeForTransform( secondBucket, tu, compID ).first;
      const bool         transpose = getTransposeFlag( intraMode );
      const int          w         = transpose ? height : width;
      const int          h         = transpose ? width : height;
      const int          numIn     = PU::getNSPTMatrixDim( w, h );
      job.mode          = GpuInvTransformJob::MODE_NSPT;
      job.nsptW         = w;
      job.nsptH         = h;
      job.nsptTranspose = transpose;
      job.nsptNumIn     = numIn;
      job.nsptShift     = shift1 + shift2 - 7;
      job.nsptMat       = getNsptMatrix( g_nsptLut[intraMode], w, h, tu.cu->lfnstIdx - 1, PU::getNSPTBucket( tu, secondBucket, compID ) );
      job.nsptScan.resize( numIn );
      for( int i = 0; i < numIn; i++ )
      {
        job.nsptScan[i] = int( scan[i].idx );
      }
    }
    else
    {
      job.mode = GpuInvTransformJob::MODE_2D;
    }
  }
  else
  {
    const int shift = ( matrixShift + maxLog2TrDynamicRange - 1 ) - bitDepth + COM16_C806_TRANS_PREC;
    CHECK( shift < 0, "Negative shift" );
    job.mode   = width == 1 ? GpuInvTransformJob::MODE_1D_VER : GpuInvTransformJob::MODE_1D_HOR;
    job.shift1 = shift + 1;
  }
  return true;
#else
  return false;
#endif
}
