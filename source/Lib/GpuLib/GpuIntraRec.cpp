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
#include "GpuPnn.h"
#include "VkCompute.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <unordered_map>

#include "CommonLib/CodingStructure.h"
#include "CommonLib/BilateralFilter.h"
#include "CommonLib/DepQuant.h"
#include "CommonLib/InterpolationFilter.h"
#if JVET_AJ0249_NEURAL_NETWORK_BASED
#include "CommonLib/IntraPredictionNN.h"
#endif
#include "CommonLib/MipData.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Quant.h"
#include "CommonLib/Reshape.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Slice.h"
#include "CommonLib/TrQuant.h"
#include "CommonLib/UnitTools.h"

#include "intra_rec.spv.h"
#include "intra_rec_lfnst.spv.h"
#include "intra_resi.spv.h"
#include "intra_deps.spv.h"
#include "scan.spv.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace
{
// layout shared with intra_rec.comp (struct Tu): 103 int32
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
  int32_t mtsSet, secIdx, secBucket;
  int32_t lm;     // a CCLM block's mark, slope adjustments and LBCCP filter (intra_common.glsl's LM_*)
  int32_t cflm;   // a CFLM block's template record in Params::timdData (TemplateUnits, its units' bits; -1: none)
  int32_t ccp;    // an LM / LM chroma fusion / CCP merge block's CU's CCP model record in Params::ccpOut (CCP_OUT;
                  // -1: none)
  int32_t ccpMerge;   // a CCP merge block's entry in Params::ccpData (CCP_MERGE_*; -1: not a merge block)
  int32_t pdp;        // a block PDP may predict: its size, its references' padding (intra_common.glsl; -1: none)
  int32_t pnn;        // a block of NN intra prediction: its record in Params::timdData (PNN_REC_*; -1: none)
};
static_assert( sizeof( Tu ) == 105 * 4, "Tu layout" );

// a block of NN intra prediction's record in Params::timdData (intra_common.glsl's PNN_REC_*)
constexpr int PNN_REC_AVAIL = 12, PNN_REC_TARGET = 16, PNN_REC_TOP_BOUNDARY = 19;
constexpr int PNN_REC = 20;

// the PDP sizes with filters (g_size's first 16) and MIP's filter sets (per MIP mode, then transposed; intra_pdp.glsl's
// PDP_SIZES, PDP_MIP_GROUPS)
constexpr int PDP_SIZES      = 16;
constexpr int PDP_MIP_GROUPS = 32;

// Tu::derive (intra_common.glsl's DERIVE_*)
constexpr int DERIVE_NODE         = 32;   // not a block: the derivation of the mode of CU Tu::cu, before its blocks
constexpr int DERIVE_MODE         = 1;    // intra_rec takes the mode from the CU's node and sets its parameters
constexpr int DERIVE_FETCH2ND     = 2;    // IntraPrediction's fetchRef2nd: intra fusion may read the second line
constexpr int DERIVE_BF           = 4;    // the intra bilateral filter applies unless the mode excludes it
constexpr int DERIVE_NO_PDPC      = 8;    // the SPS disables PDPC
constexpr int DERIVE_NO_SMOOTHING = 16;   // the SPS disables intra smoothing
constexpr int DERIVE_TRANSFORM    = 64;   // the transforms depend on the mode: intra_rec chooses them, computes the residual
constexpr int DERIVE_DIMD         = 128;  // a block of a DIMD CU: intra_rec blends the CU's DIMD modes where it blends
constexpr int DERIVE_OBIC         = 256;  // a block of an OBIC CU: intra_rec blends the CU's OBIC modes
constexpr int DERIVE_MERGE        = 512;  // a block of an intra merge CU: intra_rec blends the chosen candidate's modes
constexpr int DERIVE_MTSS         = 1024; // MTSS (mdirIdx): the LFNST / NSPT mode and bucket from the CU's DimdOut (Cu::mtss)
constexpr int DERIVE_MIP_DIMD     = 2048; // a MIP block with LFNST / NSPT: their mode from a DIMD of its prediction
constexpr int DERIVE_TIMD         = 4096; // a block of a TIMD CU: intra_rec predicts the CU's TIMD modes (TimdOut)
constexpr int DERIVE_NO_FUSION    = 8192; // the SPS disables intra fusion (also the fusion of blended TIMD modes)
constexpr int DERIVE_TIMD_MERGE   = 16384;   // a block of a TIMD merge CU: its blend fuses as TIMD merge's
constexpr int DERIVE_INHERIT_HOR  = 32768;   // a TIMD merge CU's block whose rows / columns take the transform type its
constexpr int DERIVE_INHERIT_VER  = 65536;   // chosen candidate gives (TimdOut's timdmTrType; with DERIVE_TRANSFORM)
constexpr int DERIVE_TMRL         = 131072;  // a block of a TMRL CU: its mode and line from its node, its line's
                                             // availability found by intra_rec
constexpr int DERIVE_CCLM_DIMD    = 262144;  // a CCLM block (also a CCP merge block) with LFNST / NSPT: their mode a
                                             // DIMD of the CU's chroma predictions (Tu::lm's LM_HIST_*), found by
                                             // intra_rec
constexpr int DERIVE_CCP_DIMD     = 524288;  // a block of a CCP merge CU of fusion with DIMD chroma: intra_rec blends
                                             // the merge prediction and that of its node's DIMD chroma mode
constexpr int DERIVE_CPU_FILL     = 1048576; // with DERIVE_NODE: a block partial verification takes from the CPU, which
                                             // owns its area (the blocks after it read it as decoded) and does nothing
constexpr int DERIVE_PNN_DIMD     = 2097152; // an NN block whose transforms' mode is a DIMD of its prediction
constexpr int DERIVE_PNN_LFNST_SEC = 4194304; // ... whose LFNST / NSPT take the lfnstSecFlag rule's mode

// Tu::lm (intra_common.glsl's LM_*): whether the block is a CCLM block (the modes from 67 on are also those of TIMD /
// TMRL luma blocks), its slope adjustments (+ 8, in 4 bits each) and LBCCP's inside filter; with DERIVE_CCLM_DIMD the
// predictions its transforms' DIMD reads and the other chroma component's slope adjustments; a CCCM block's template
// selection (cccmFlag) and kind (GL-CCCM, several downsampling filters, no subsampling, GLM with luma: then the
// template selection is its LM mode's, its glmIdc in the GLM field); a block of LM chroma fusion (not an LM block):
// the template selection of the CCCM models it blends in, the weight of its own prediction
constexpr int LM_DELTA0_SHIFT = 0, LM_DELTA1_SHIFT = 4, LM_INSIDE = 256, LM_ACTIVE = 512;
constexpr int LM_HIST_CB = 1024, LM_HIST_CR = 2048, LM_OTHER_SHIFT = 12;
constexpr int LM_GLM_SHIFT = 20, LM_OTHER_GLM_SHIFT = 24;   // GLM's gradient pattern (glmIdc), the other component's
constexpr int LM_CCCM_SHIFT = 28, LM_CCCM_GL = 4096, LM_CCCM_MDF_SHIFT = 13, LM_CCCM_NOSUB = 32768;
constexpr int LM_CCCM_GLM_LUMA = 65536;
// (LM chroma fusion's mark beyond every LM field: a CCLM block's other-component field reaches bit 19; its type
// (isChromaFusion) in the second slope adjustment's field and the first type's weight in the first's, which a fusion
// block does not have)
constexpr int LM_FUSION = 1 << 30, LM_FUSION_W_SHIFT = 0, LM_FUSION_TYPE_SHIFT = 4;

// the CCP model record of an LM CU or an LM chroma fusion CU (intra_common.glsl's CCP_OUT_*, int32): the
// CCPModelCandidate fields predIntraChromaLM / predIntraCCCM / geneChromaFusionPred set (CCP merge's curCand) and the
// inside filter flags, written by the CU's blocks, compared in verification
constexpr int CCP_OUT_TYPE = 0, CCP_OUT_GLM_IDC = 1, CCP_OUT_Y_THRES = 2, CCP_OUT_LUMA_OFFSET = 3, CCP_OUT_COR_OFF = 4;
constexpr int CCP_OUT_MDF = 6, CCP_OUT_NUM_PARAMS = 7, CCP_OUT_SHIFT = 8, CCP_OUT_SHIFT2 = 10, CCP_OUT_INSIDE = 12;
constexpr int CCP_OUT_INSIDE_HIST = 13, CCP_OUT_RANGE = 14;
constexpr int CCP_MAX_PARAMS = 11, CCP_OUT_PARAMS = 26, CCP_OUT_PARAMS2 = CCP_OUT_PARAMS + 4 * CCP_MAX_PARAMS;
constexpr int CCP_OUT = CCP_OUT_PARAMS2 + 4 * CCP_MAX_PARAMS;
// Params::ccpData (intra_common.glsl's CCP_WRITERS, CCP_MERGE_*; int32): per record the Tu indices of its Cb and Cr
// blocks, the CTU rows' histories, per CCP merge CU its entry (the history's position and number, idxNonLocalCCP - 1
// or decoder-derived merge fusion's pair, the first of its blocks' default records, CCP_DEFAULTS each - with
// adjustment its on-the-fly models', for decoder-derived CCP fusion its models' -, the kind: ccpMergeAdjustFlag,
// CCP_KIND_DERIVED or CCP_KIND_DD_FUSION, the spatial sources)
constexpr int CCP_WRITERS = 2, CCP_DEFAULTS = 12;
constexpr int CCP_MERGE_HIST = 0, CCP_MERGE_HIST_NUM = 1, CCP_MERGE_INDEX = 2, CCP_MERGE_DEFAULTS = 3;
constexpr int CCP_MERGE_KIND = 4, CCP_MERGE_NUM_SPATIAL = 5, CCP_MERGE_SPATIAL = 6;
// the kinds beyond ccpMergeAdjustFlag's values: a block of decoder-derived CCP fusion, one of decoder-derived CCP merge
// fusion (its entry's index the pair's, ddNonLocalCCPFusion - 1), one of merge fusion with CCCM by a blend model
constexpr int CCP_KIND_DERIVED = 3, CCP_KIND_DD_FUSION = 4, CCP_KIND_FUSION_BLEND = 5;
// decoder-derived CCP merge fusion's pairs of the first 9 candidates and the words of each pair's cost and blend model
// (intra_ccp.glsl's CCP_PAIRS_MAX, CCP_PAIR_WORDS)
constexpr int CCP_PAIRS_MAX = 36, CCP_PAIR_WORDS = 7;

// the CCCM scratch a block of decoder-derived CCP fusion, of decoder-derived CCP merge fusion or of merge fusion with
// CCCM by a blend model needs beyond its luma buffers (intra_ccp.glsl's ccpProbeBase): its prediction, list[1]'s (or
// the CCCM one) at the scratch's end and before that - while no prediction is there yet - the probe or template samples
// of up to CCP_DEFAULTS modes or candidates and of the CCCM prediction, and the pairs' models
static int ccpDerivedScratch( const CompArea& area )
{
  const int samples = ( CCP_DEFAULTS + 1 ) * 2 * int( area.width + area.height ) + CCP_PAIRS_MAX * CCP_PAIR_WORDS;
  return (int) area.area() + std::max( (int) area.area(), samples );
}
#if JVET_AD0188_CCP_MERGE
static_assert( NUM_CCP_PARAMS == CCP_MAX_PARAMS && CCP_TYPE_CCLM == 1 && CCP_TYPE_MMLM == 2 && CCP_TYPE_GLM0123 == 4
                 && CCP_TYPE_GLM4567 == 8 && CCP_TYPE_CCCM == 16 && CCP_TYPE_GLCCCM == 32 && CCP_TYPE_NSCCCM == 64
                 && CCP_TYPE_MDFCCCM == 128,
               "intra_common.glsl's CCP record" );
#endif

// the type of a block of LM chroma fusion (isChromaFusion: 1 the blend with CCCM, 2 / 3 CFLM), 0 for other blocks
int lmFusionType( const Tu& t )
{
  return ( t.lm & LM_FUSION ) ? ( t.lm >> LM_FUSION_TYPE_SHIFT ) & 3 : 0;
}

// whether PU pu is a CCP merge PU of fusion with DIMD chroma (JVET-AG0059's ccpMergeFusionType 1)
static bool ccpFusionDimd( const PredictionUnit& pu )
{
#if JVET_AG0059_CCP_MERGE_ENHANCEMENT
  return pu.idxNonLocalCCP && pu.ccpMergeFusionFlag && pu.ccpMergeFusionType == 1;
#else
  return false;
#endif
}

// whether PU pu is a CCP merge PU of fusion with CCCM by a blend model (JVET-AN0168: predCCPCandidate's
// useBlendModelCcpm, the SPS's ccpFusionMode; without it the average)
static bool ccpFusionBlend( const PredictionUnit& pu )
{
#if JVET_AG0059_CCP_MERGE_ENHANCEMENT && JVET_AN0168_REGRESSION_CCP_FUSION
  return pu.idxNonLocalCCP && pu.ccpMergeFusionFlag && pu.ccpMergeFusionType == 0 && pu.cs->sps->getCcpFusionMode();
#else
  return false;
#endif
}

// the parameters intra_rec derives per DERIVE_MODE block, downloaded for the comparison (intra_common.glsl's
// DBG_PARAMS): dirMode, isModeVer, intraPredAngle, absInvAngle, useCubic, refFilter, applyPDPC, gradPDPC, angularScale,
// fusion, bf, plIdx, then the transforms: mtxV, mtxH, secMatrix, secTranspose, secLayout
constexpr int DBG_PARAMS = 17;

// the directory of the transform tables of derived modes (intra_common.glsl's TR_TAB_*, createShared)
constexpr int TR_TAB_MATRIX       = 0;
constexpr int TR_TAB_IPM_TO_SET   = 1;
constexpr int TR_TAB_SET          = 2;
constexpr int TR_TAB_IDX_TO_TR    = 3;
constexpr int TR_TAB_ID_LUT       = 4;
constexpr int TR_TAB_LFNST_LUT    = 5;
constexpr int TR_TAB_NSPT_LUT     = 6;
constexpr int TR_TAB_LFNST_MATRIX = 7;
constexpr int TR_TAB_NSPT_MATRIX  = 8;
constexpr int TR_TAB_LFNST_LAYOUT = 9;
constexpr int TR_TAB_EXT_MODES    = 10;
constexpr int TR_TAB_SET_ROW      = 11;
constexpr int TR_TAB_IDTR         = 12;
constexpr int TR_TAB_NUM          = 13;
// the NSPT shapes of TR_TAB_NSPT_MATRIX (intra_common.glsl's nsptShape)
constexpr int NSPT_SHAPES                 = 12;
constexpr int NSPT_SHAPE_W[NSPT_SHAPES] = { 4, 8, 4, 8, 4, 16, 8, 16, 4, 32, 8, 32 };
constexpr int NSPT_SHAPE_H[NSPT_SHAPES] = { 4, 8, 8, 4, 16, 4, 16, 8, 32, 4, 32, 8 };

// getSpatialIpm's positions: left, above, above-left, above-right, below-left, then the non-adjacent ones
constexpr int MPM_NEIGHBOURS = 23;

// a coding unit of one channel type (intra_common.glsl's Cu): 47 int32
struct Cu
{
  int32_t kind;
  int32_t node;
  int32_t x, y, w, h;
  int32_t mpmFlag, secondMpmFlag, ipredIdx;   // (a TMRL CU's tmrlListIdx in ipredIdx)
  int32_t nbFirst, nbLast;
  int32_t plIdx;
  int32_t candId;
  int32_t lumaCu;
  int32_t nbr[MPM_NEIGHBOURS];
  int32_t modes[2];
  int32_t dimd;      // the first of the CU's DIMD histograms (-1: no DIMD)
  int32_t dimdNum;   // their number
  int32_t obic;      // an OBIC CU's index among the picture's OBIC CUs (its neighbours, its OBIC; -1: not OBIC)
  int32_t merge;     // an intra merge CU's record in Params::mergeData (MergeHead; -1: not intra merge)
  int32_t mtss;      // with MTSS (mdirIdx) the first of the MTSS_CALLS histograms of its mode list (-1: no MTSS)
  int32_t mtssNspt;  // its luma block may use NSPT (CU::isNSPTAllowed), which MTSS's choice depends on
  int32_t timd;      // a TIMD CU's record in Params::timdData (TimdHead; -1: not TIMD)
  int32_t tmrl;      // a TMRL CU's index among the picture's TMRL CUs (Params::tmrlOut; -1: not TMRL)
};
static_assert( sizeof( Cu ) == 47 * 4, "Cu layout" );

// Cu::kind
constexpr int CU_KNOWN       = 0;   // the host knows the mode
constexpr int CU_MPM         = 1;   // luma: from the MPM lists (intra_modes.glsl)
constexpr int CU_CHROMA_DM   = 2;   // chroma: the co-located luma mode
constexpr int CU_CHROMA_CAND = 3;   // chroma: one of the first four candidates
constexpr int CU_DIMD        = 4;   // luma: a DIMD CU (intra_dimd.glsl)
constexpr int CU_CHROMA_DIMD = 5;   // chroma: DIMD chroma
constexpr int CU_OBIC        = 6;   // luma: an OBIC CU (intra_obic.glsl), with its DIMD
constexpr int CU_MERGE       = 7;   // luma: an intra merge CU (intra_merge.glsl), without DIMD
constexpr int CU_TIMD        = 8;   // luma: a TIMD CU (intra_timd.glsl), with its DIMD (its MPM list's first modes)
constexpr int CU_TMRL        = 9;   // luma: a TMRL CU (intra_tmrl.glsl), with its DIMD (its mode list's first modes)
constexpr int CU_CHROMA_TMRL = 10;  // chroma: a chroma TMRL CU (intra_tmrl.glsl), with its DIMD chroma; Cu::nbr from
                                    // CHROMA_TMRL_NBR_LUMA: the luma CUs at its five positions, then the chroma CUs
constexpr int CHROMA_TMRL_NBR_LUMA = 0, CHROMA_TMRL_NBR_CHROMA = 5;

// a CU kind's name, for the verification's reports
static const char* cuKindName( int kind )
{
  static const char* const NAMES[] = { "known mode", "MPM", "chroma DM", "chroma candidate", "DIMD", "DIMD chroma",
                                       "OBIC", "intra merge", "TIMD", "TMRL", "chroma TMRL" };
  return kind >= 0 && kind <= CU_CHROMA_TMRL ? NAMES[kind] : "?";
}

// int32 per TMRL CU (intra_common.glsl's TMRL_OUT_*): the mode (131-mode set) and multiRefIdx of its m_tmrlList entry
constexpr int TMRL_OUT = 2, TMRL_OUT_MODE = 0, TMRL_OUT_MRL = 1;
// TMRL's reference lines (EXT_REF_LINE_IDX; intra_common.glsl's TMRL_LINES), of a CU at picture row y those inside the
// picture, line L < y (getTmrlSearchRange)
constexpr int TMRL_LINES[5] = { 1, 3, 5, 7, 12 };
static int tmrlNumLines( int y )
{
  int n = 0;
  while( n < 5 && TMRL_LINES[n] < y )
  {
    n++;
  }
  return n;
}
// chroma TMRL's chroma lines (CHROMA_MULTI_REF_LINE_IDX; intra_common.glsl's CHROMA_TMRL_LINES), of a CU at chroma
// row y those inside the picture (getChromaTmrlSearchRange)
constexpr int CHROMA_TMRL_LINES[3] = { 1, 3, 5 };
static int chromaTmrlNumLines( int y )
{
  int n = 0;
  while( n < 3 && CHROMA_TMRL_LINES[n] < y )
  {
    n++;
  }
  return n;
}

// the DIMD the kernels have (IntraPrediction::dimdHistogramCalls, dimdChromaHistogramCalls, obicNeighbours)
#if ENABLE_DIMD && JVET_AJ0267_ADAPTIVE_HOG && JVET_AJ0203_DIMD_2X2_EDGE_OP && JVET_AC0098_LOC_DEP_DIMD && JVET_Z0050_DIMD_CHROMA_FUSION \
  && JVET_AC0094_REF_SAMPLES_OPT && JVET_AH0076_OBIC && JVET_AK0056_WEIGHTED_OBIC
#define GPU_DIMD 1
#else
#define GPU_DIMD 0
#endif

// one histogram of a CU's DIMD (intra_common.glsl's DimdCall, from IntraPrediction::DimdHistogramCall, pReco in its
// plane): 16 int32
struct DimdCall
{
  int32_t kind, histogram, x, y, height, width, direction, filterSizeIdx;
  int32_t sizeExt[3], extra[3];
  int32_t maxAmp;
  int32_t comp;
};
static_assert( sizeof( DimdCall ) == 16 * 4, "DimdCall layout" );
// int32 per CU (intra_common.glsl's DIMD_OUT_*): dimdMode, dimdBlending, dimdBlendMode[7], dimdRelWeight[8],
// dimdLocDep[7] (DIMD_OUT_PARAMS), then candModeListForTransform[1], with MTSS candModeListForTransformMtss[3] and
// getFinalIntraModeForTransform's result: the LFNST / NSPT mode (NSPT's numbering) and secondBucket. A MIP CU's: its
// block's mipDimdMode as the first, dimdDerivedIntraDir2nd as candModeListForTransform[1], with MTSS the list and result
constexpr int DIMD_OUT_PARAMS      = 24;
constexpr int DIMD_OUT_TR_SECOND   = 24;
constexpr int DIMD_OUT_MTSS_LIST   = 25;
constexpr int DIMD_OUT_MTSS_MODE   = 28;
constexpr int DIMD_OUT_MTSS_BUCKET = 29;
constexpr int DIMD_OUT             = 30;
constexpr int MTSS_CALLS           = 3;      // the histograms of deriveDimdModeList (left, above, above-left)
// the transforms of DIMD CUs the kernels have: getFinalIntraModeForTransform's DIMD branch (JVET-AM0138; without
// BVG-DIMD and intra merge it would take intra merge CUs too), MTSS (JVET-AK0217) with JVET-AK0187's mode pairs
#if GPU_DIMD && JVET_AM0138_ENHANCED_TMP_MERGE_LIST_TIMD_BV && JVET_AL0108_BVG_DIMD && JVET_AM0074_INTRA_MERGE \
  && JVET_AK0217_INTRA_MTSS && JVET_AK0187_IMPLICIT_MTS_LUT_EXTENSION
#define GPU_DIMD_TRANSFORMS 1
static_assert( MTSS_LIST_SIZE == 3, "MTSS's mode list" );
#else
#define GPU_DIMD_TRANSFORMS 0
#endif
// the LFNST / NSPT of MIP blocks the kernels have: their mode from deriveIpmForTransform over the MIP prediction
// (JVET-AB0067; with OBIC for every MIP block, with MTSS over the whole prediction, subsampled, and its second mode)
#if GPU_DIMD_TRANSFORMS && JVET_AB0067_MIP_DIMD_LFNST && JVET_AH0076_OBIC && JVET_AI0050_INTER_MTSS
#define GPU_MIP_DIMD 1
#else
#define GPU_MIP_DIMD 0
#endif
// the TIMD the kernels have: JVET-W0123 with JVET-AC0094's wide angles beyond the regular ones, JVET-AG0092's fusion
// with a non-angular mode, JVET-X0149's weights, JVET-AG0128's fusion of close blended modes, PDPC of a blend's planar /
// DC (JVET-X0148), and in the blocks no reference filter (JVET-AM0104) and no Gaussian interpolation (JVET-AM0163) of
// angular modes, JVET-AN0158's filter lengths; TIMD-SAD (JVET-AJ0146) and TIMD merge (JVET-AJ0061, with its transform
// types inherited in dual tree)
#if GPU_DIMD_TRANSFORMS && JVET_W0123_TIMD_FUSION && JVET_AG0092_ENHANCED_TIMD_FUSION && JVET_X0149_TIMD_DIMD_LUT \
  && JVET_AG0128_REF_LINE_OPT_TIMD_FUSION && JVET_X0148_TIMD_PDPC && JVET_AM0104_DISABLE_REFFILTER_FOR_TIMD \
  && JVET_AM0163_CUBIC_FILTER_FOR_TIMD && JVET_AN0158_INTRA_LONGTAP && SECONDARY_MPM && JVET_AJ0146_TIMDSAD \
  && JVET_AJ0061_TIMD_MERGE && JVET_AG0146_DIMD_ITMP_IBC   // (templateUnits, IntraPrediction::timdReferenceUnits)
#define GPU_TIMD 1
#else
#define GPU_TIMD 0
#endif
// the TMRL the kernels have: JVET-AB0157 with JVET-AD0085's 131-mode set and candidate list (the DIMD modes first:
// JVET-AL0125), JVET-AD0082's SPS switch, the lines up to the picture's top (JVET-AH0065), the template's SATD with the
// kernels of arms one sample thick (JVET-AJ0096)
#if GPU_TIMD && JVET_AB0157_TMRL && JVET_AD0085_TMRL_EXTENSION && JVET_AD0082_TMRL_CONFIG && JVET_AH0065_RELAX_LINE_BUFFER \
  && JVET_AL0125_IMPROVEMENT_ON_MPM && JVET_AJ0096_SATD_REORDER_INTRA
#define GPU_TMRL 1
#else
#define GPU_TMRL 0
#endif
// and chroma TMRL (JVET-AJ0081): its cost predicts the co-located luma block as chroma reordering does (JVET-AH0136's
// predCoLuma and xFillReferenceSamplesForCoLuma, with JVET-AI0136's dual-tree rules), the SATD's DC term a quarter
// (JVET-R0164)
#if GPU_TMRL && JVET_AJ0081_CHROMA_TMRL && JVET_AH0136_CHROMA_REORDERING && JVET_AI0136_ADAPTIVE_DUAL_TREE \
  && JVET_R0164_MEAN_SCALED_SATD
#define GPU_CHROMA_TMRL 1
#else
#define GPU_CHROMA_TMRL 0
#endif
// the CCLM the kernels have (intra_cclm.glsl): ECM's least-squares model (LMS_LINEAR_MODEL, 64-bit sums: JVET-AJ0237;
// the means without division: JVET-AD0184) of LM_CHROMA, MDLM and MMLM, the luma downsampling filters by the SPS's
// collocated chroma flags (JVET-AN0090), the slope adjustment (JVET-Z0050) and LBCCP's inside filter (JVET-AD0120);
// GLM, CCCM and LM chroma fusion are rejected
#if LMS_LINEAR_MODEL && MMLM && JVET_AJ0237_INTERNAL_12BIT && JVET_AD0184_REMOVAL_OF_DIVISION_OPERATIONS \
  && JVET_AN0090_ADAPTIVE_SUBSAMPLING_FILTER_SELECTION && JVET_Z0050_CCLM_SLOPE && JVET_AD0120_LBCCP
#define GPU_CCLM 1
#else
#define GPU_CCLM 0
#endif
// GLM's gradient patterns the kernels have (intra_cclm.glsl's GLM_PATTERN: four with CCCM, of nine taps with
// JVET-AP0105's location types); GLM with luma (glmIdc beyond them) is a CCCM model (GPU_CCCM)
#if GPU_CCLM && JVET_AA0126_GLM && JVET_AB0092_GLM_WITH_LUMA && JVET_AA0057_CCCM && JVET_AP0105_SUBSAMPLING_FILTER_CCLM_CCCM_FIX
#define GPU_GLM 1
static_assert( NUM_GLM_PATTERN == 4, "intra_cclm.glsl's GLM_PATTERN" );
#else
#define GPU_GLM 0
#endif
// the CCCM the kernels have (intra_cccm.glsl): the plain 7-tap models, GL-CCCM's (JVET-AC0054), those of several
// downsampling filters (JVET-AD0202), without subsampling (JVET-AC0147) and GLM with luma's (JVET-AB0092; the
// division-free solver: JVET-AB0174, the Gaussian elimination: JVET-AC0053, JVET-AJ0237's precision up to 10 bits),
// the template selection (JVET-AB0143), the clipping (JVET-AP0168), LBCCP's two sets of models and inside filter
// (JVET-AD0120, chosen by the template's SATD with JVET-AJ0096's kernels), the downsampling and the taps of CCLM's
// location types (JVET-AN0090, JVET-AP0105); BVG-CCCM, CCP merge and decoder-derived CCP fusion are rejected
#if GPU_CCLM && JVET_AA0057_CCCM && JVET_AB0174_CCCM_DIV_FREE && JVET_AC0053_GAUSSIAN_SOLVER && JVET_AJ0237_INTERNAL_12BIT \
  && JVET_AB0143_CCCM_TS && JVET_AP0168_CCCM_CLIP && JVET_AD0202_CCCM_MDF && JVET_AJ0096_SATD_REORDER_INTRA \
  && JVET_AE0174_NONINTER_TM_TOOLS_CONTROL && JVET_AP0105_SUBSAMPLING_FILTER_CCLM_CCCM_FIX && JVET_AC0054_GLCCCM \
  && JVET_AC0147_CCCM_NO_SUBSAMPLING
#define GPU_CCCM 1
static_assert( CCCM_NUM_PARAMS == 7 && CCCM_WINDOW_SIZE == 6 && CCCM_DECIM_BITS == 16 && CCCM_MATRIX_BITS == 22
                 && CCCM_FILTER_PADDING == 1 && CCCM_REF_LINES_ABOVE_CTU == 0 && CCCM_LOC_OFFSET == 8
                 && CCCM_LOC_SHIFT == 3 && CCCM_NUM_PRED_FILTER == 4 && CCCM_MULTI_PRED_FILTER_NUM_PARAMS == 10
                 && CCCM_MULTI_PRED_FILTER_NUM_PARAMS2 == 11 && CCCM_NO_SUB_NUM_PARAMS == 11
                 && CCCM_NO_SUB_NUM_PARAMS2 == 9 && GLM_NUM_PARAMS == 3,
               "intra_cccm.glsl's constants" );
#else
#define GPU_CCCM 0
#endif
// the CFLM the kernels have (LM chroma fusion's types 2 and 3, JVET-AC0119; intra_cccm.glsl's cflmPrediction): solve1's
// model of three parameters over the template's prediction by xPredTimdIntraAng's chroma path (intra_timd.glsl), whose
// reference line's units templateUnits gives
#if GPU_CCCM && GPU_TIMD && JVET_AC0119_LM_CHROMA_FUSION
#define GPU_CFLM 1
static_assert( CFLM_NUM_PARAMS == 3, "intra_cccm.glsl's CFLM" );
#else
#define GPU_CFLM 0
#endif
constexpr int CFLM_TEMPLATE = 2;   // CFLM's template: lines above / columns left of the block (xCflmGetRefBuf)
// the capacity of intra_cccm.glsl's luma buffer (the four angular arrays): (areaW + 2) x (areaH + 2) samples (without
// subsampling twice as many per direction); of its prediction in s_tmp (larger blocks, and several filters' buffers,
// are in the workgroup's CCCM scratch)
constexpr int CCCM_LUMA_MAX = 4 * 1792, CCCM_PRED_MAX = 2048;
constexpr int BLEND_ACC  = 3 * 128 * 128;    // int32 per intra_rec workgroup: the sums of DIMD / OBIC blending

// a neighbour of an OBIC CU whose mode counts (intra_common.glsl's ObicNbr, IntraPrediction::obicNeighbours): its
// record (-1: none) and the samples it counts
struct ObicNbr
{
  int32_t cu, numSamples;
};
constexpr int OBIC_NBRS  = 20;               // ObicNbr per OBIC CU (deriveObicMode's limitMaxNeigh)
constexpr int OBIC_MODES = 6;                // the modes OBIC blends (OBIC_FUSION_NUM)
constexpr int OBIC_OUT   = 3 * OBIC_MODES;   // int32 per OBIC CU: obicMode, obicFusionWeight, obicLocDep
#if GPU_DIMD
static_assert( OBIC_FUSION_NUM == OBIC_MODES, "OBIC's modes" );
#endif

// a template of TIMD's kind (intra_common.glsl's TemplateUnits, layout shared): its arms, tw columns left of the CU and th
// rows above it (0: no such arm), and the units of the reference line of its block, the CU extended by the arms (8 x
// its sides, initTimdIntraPatternLuma; IntraPrediction::TimdReferenceUnits), whose availability bits follow its record
// (32 per word)
struct TemplateUnits
{
  int32_t tw, th;
  int32_t unitW, unitH;
  int32_t leftTempUnitNum, aboveTempUnitNum, totalLeftUnits, totalAboveUnits, totalUnits, numIntraNeighbor;
};
static_assert( sizeof( TemplateUnits ) == 10 * 4, "TemplateUnits layout" );

// an intra merge CU's record in Params::mergeData (intra_common.glsl's MergeHead; layout shared): the header, then
// nbrNum pairs (neighbour record, MERGE_NBR_*) in deriveIntraMergeCand's order, then the template's availability bits
struct MergeHead
{
  int32_t       index;   // its index among the picture's intra merge CUs (Params::mergeCost)
  int32_t       nbrNum;
  TemplateUnits t;       // reorderIntraMergeCand's template (none: tw 0, x or y at the picture's edge)
};
static_assert( sizeof( MergeHead ) == 12 * 4, "MergeHead layout" );

// a TIMD CU's record in Params::timdData (intra_common.glsl's TimdHead; layout shared): the header, then its
// template's availability bits, then its TIMD-SAD template's
constexpr int TIMD_MERGE_CANDS = 5;   // the neighbours searchTimdMrgCus keeps (its limitMaxNeigh)
struct TimdHead
{
  int32_t       index;   // its index among the picture's TIMD CUs (Params::timdOut)
  int32_t       type;    // its template (CU::deriveTimdRefType): TIMD_NONE (the picture's top-left CU), TIMD_ABOVE,
                         // TIMD_LEFT, TIMD_BOTH
  int32_t       waip;    // where deriveTimdMode's wide angles beyond the regular ones may reach: TIMD_WAIP_* bits
  TemplateUnits t;       // its template, an arm it lacks 0 (a TIMD merge CU's calcTimdMrgCandCosts', one sample thick)
  int32_t       sad;     // a TIMD-SAD CU: its node also does deriveTimdModeSad
  TemplateUnits ts;      // deriveTimdModeSad's template (arms of up to 8, the same type), where sad
  // the transform types of the CU's luma block as TrQuant::getTrTypes gives them (CodingUnit::timdmTrType, which a TIMD
  // merge CU may take): explicit MTS's pair index where the mode chooses its kernels (-1: none), else the types
  int32_t       mtsTrIdx;
  int32_t       trHor, trVer;
  // a TIMD merge CU: its candidates' number (0: not TIMD merge) and records, searchTimdMrgCus' neighbours in its order;
  // inheritTr: the candidates give their transform types (genTimdMrgList, dual tree), else DCT-II
  int32_t       merge;
  int32_t       mergeNbr[TIMD_MERGE_CANDS];
  int32_t       inheritTr;
};
static_assert( sizeof( TimdHead ) == 34 * 4, "TimdHead layout" );
constexpr int TIMD_NONE  = 0;
constexpr int TIMD_ABOVE = 1;
constexpr int TIMD_LEFT  = 2;
constexpr int TIMD_BOTH  = 3;
constexpr int TIMD_WAIP_MAX_CLOSE = 1;   // increaseMaxClose: a PU above the CU's last column
constexpr int TIMD_WAIP_MAX_FAR   = 2;   // increaseMaxFar: a PU above, a CU width further right
constexpr int TIMD_WAIP_MIN_CLOSE = 4;   // decreaseMinClose: a PU left of the CU's last row
constexpr int TIMD_WAIP_MIN_FAR   = 8;   // decreaseMinFar: a PU left, a CU height further down
// int32 per TIMD CU (intra_common.glsl's TIMD_OUT_*): the CodingUnit fields deriveTimdMode sets, those it does not
// at their initial values: timdMode, timdModeSecondary, timdModeNonAng (131-mode set), timdIsBlended,
// timdFusionWeight[3], timdLocDep[3], timdModeCheckWA, timdModeSecondaryCheckWA; from TIMD_OUT_SAD, for a TIMD-SAD
// CU, deriveTimdModeSad's (the same fields' *Sad counterparts); a TIMD merge CU's chosen candidate in the first fields
// (timdMrgList[0], ... timdMrgModeCheckWA[0]). Then, in the pipeline with TIMD merge, timdmTrType
constexpr int TIMD_OUT_FIELDS = 12;   // the fields of one derivation
constexpr int TIMD_OUT_SAD    = TIMD_OUT_FIELDS;       // the offset of deriveTimdModeSad's
constexpr int TIMD_OUT_TR_HOR = 2 * TIMD_OUT_FIELDS;   // timdmTrType's horizontal type
constexpr int TIMD_OUT_TR_VER = TIMD_OUT_TR_HOR + 1;   // and vertical one
constexpr int TIMD_OUT        = TIMD_OUT_TR_VER + 1;
constexpr int TIMD_OUT_MODE = 0, TIMD_OUT_SECONDARY = 1, TIMD_OUT_NON_ANG = 2, TIMD_OUT_BLENDED = 3, TIMD_OUT_WEIGHT = 4;
constexpr int TIMD_OUT_LOC_DEP = 7, TIMD_OUT_CHECK_WA = 10, TIMD_OUT_SEC_CHECK_WA = 11;
static_assert( DCT2 == 0, "intra_common.glsl's TR_DCT2" );
// the reference units a TIMD template's record may have (intra_timd.glsl's s_timdUnits: 17 words of 32)
constexpr int TIMD_UNITS_MAX  = 17 * 32;
// how a neighbour may be a candidate (CU::isAValidNeighborForIntraMerge), and whose parameters it gives
constexpr int MERGE_NBR_MERGE = 1;   // an intra merge CU: always, its merge parameters (its DimdOut slot)
constexpr int MERGE_NBR_OBIC  = 2;   // an OBIC CU: always, its OBIC converted
constexpr int MERGE_NBR_DIMD  = 3;   // a DIMD CU or an MPM CU: where its DIMD blends, its DIMD
constexpr int MERGE_NBR_TIMD  = 4;   // a TIMD CU: where its modes blend or its mode is angular, its TIMD converted (or its DIMD)
constexpr int MERGE_NBR_TIMD_MERGE = 5;   // a TIMD merge CU: as a TIMD CU, but no candidate where its TIMD does not convert
constexpr int MERGE_COST      = NUM_LUMA_MODE;   // uint64 per intra merge CU: the template cost of each mode

// the luma samples the template predictions of CU c read (inclusive; intra_deps.comp has a copy): the reference row
// and column of the template's block (8 x its sides, initTimdIntraPatternLuma), the template above the CU with the
// corner and the template left of it (empty where the arm is 0)
void templateReadAreas( const Cu& c, const TemplateUnits& t, int area[4][4] )
{
  const int x0 = c.x - t.tw - 1, y0 = c.y - t.th - 1;
  const int a[4][4] = { { x0, y0, x0 + ( ( c.w + t.tw ) << 3 ), y0 },
                        { x0, y0, x0, y0 + ( ( c.h + t.th ) << 3 ) },
                        { c.x - t.tw, c.y - t.th, c.x + c.w - 1, c.y - 1 },
                        { c.x - t.tw, c.y, c.x - 1, c.y + c.h - 1 } };
  memcpy( area, a, sizeof( a ) );
}

// the samples of component k.comp the gradients of DIMD histogram k read (inclusive; false: none): the histogram's
// region and one sample around it (intra_deps.comp has a copy; the gradients of its samples, also those one line
// further out of buildHistogramAdaptive's first line, lie inside it)
bool dimdCallArea( const DimdCall& k, int& x0, int& y0, int& x1, int& y1 )
{
  x0 = k.x - 1;
  y0 = k.y - 1;
  x1 = k.x + k.width;
  y1 = k.y + k.height;
  return k.kind != 0;
}

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

#if GPU_TIMD
// intra_rec.comp's timdModeParams (the kernels' copy): a TIMD luma block's prediction parameters for a mode of the
// 131-mode set as initPredIntraParams sets them (bExtIntraDir): its wide angle getWideAngleExt's, or with checkWA false
// getTimdWideAngleExt's, angles in 1/64 samples, planar's reference filter, none for angular modes (JVET-AM0104) and
// always the cubic filter (JVET-AM0163); fusion and the bilateral filter apart. Here for the comparison of blended
// TIMD blocks, whose CPU parameters after the prediction are the last blended mode's
void applyTimdModeParams( Tu& t, const int mode, const bool checkWA )
{
  // TIMD's 1/64-sample angles and their inverses (IntraPrediction's g_extAngTable is private, the inverses local)
  static const int extAngTable[64]    = { 0,   1,   2,   3,   4,   5,   6,   7,   8,   10,  12,  14,  16,   18,   20,   22,
                                          24,  26,  28,  30,  32,  34,  36,  38,  40,  43,  46,  49,  52,   55,   58,   61,
                                          64,  67,  70,  74,  78,  84,  90,  96,  102, 108, 114, 121, 128,  137,  146,  159,
                                          172, 188, 204, 230, 256, 299, 342, 427, 512, 597, 682, 853, 1024, 1536, 2048, 3072 };
  static const int extInvAngTable[64] = { 0,    32768, 16384, 10923, 8192, 6554, 5461, 4681, 4096, 3277, 2731, 2341, 2048,
                                          1820, 1638,  1489,  1365,  1260, 1170, 1092, 1024, 964,  910,  862,  819,  762,
                                          712,  669,   630,   596,   565,  537,  512,  489,  468,  443,  420,  390,  364,
                                          341,  321,   303,   287,   271,  256,  239,  224,  206,  191,  174,  161,  142,
                                          128,  110,   96,    77,    64,   55,   48,   38,   32,   21,   16,   11 };
  const int W = t.regW, H = t.h;
  const int predMode = IntraPrediction::timdWideAngle( W, H, mode, checkWA );
  t.dirMode        = mode;
  t.isModeVer      = predMode >= EXT_DIA_IDX ? 1 : 0;
  t.intraPredAngle = 0;
  t.absInvAngle    = 0;
  t.angularScale   = 0;
  t.gradPDPC       = 0;
  bool pdpc = W >= 4 && H >= 4 && t.mrl == 0;
  const int angMode = t.isModeVer ? predMode - EXT_VER_IDX : -( predMode - EXT_HOR_IDX );
  if( mode > DC_IDX && mode <= EXT_VDIA_IDX )
  {
    const int absAng = extAngTable[std::abs( angMode )];
    t.absInvAngle    = extInvAngTable[std::abs( angMode )];
    t.intraPredAngle = angMode < 0 ? -absAng : absAng;
    if( angMode < 0 )
    {
      pdpc = false;
    }
    else if( angMode > 0 )
    {
      t.angularScale = std::min( 2, floorLog2( t.isModeVer ? H : W ) - ( floorLog2( 3 * t.absInvAngle - 2 ) - 8 ) );
      if( t.angularScale < 0 )
      {
        t.angularScale = ( floorLog2( W ) + floorLog2( H ) - 2 ) >> 2;
        t.gradPDPC     = 1;
      }
    }
  }
  if( t.derive & DERIVE_NO_PDPC )
  {
    pdpc       = false;
    t.gradPDPC = 0;
  }
  const bool smoothing = !( t.derive & DERIVE_NO_SMOOTHING ) && !t.mrl && !t.disableRefFilter;
  t.refFilter = mode == PLANAR_IDX && smoothing && W * H > 32 ? 1 : 0;
  t.useCubic  = 1;
  t.applyPDPC = pdpc ? 1 : 0;
  t.plIdx     = 0;
}
#endif

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
  uint64_t dimdCalls;
  uint64_t dimdOut;
  uint64_t obicNbrs;
  uint64_t obicOut;
  uint64_t scratch;
  uint64_t mergeData;
  uint64_t mergeCost;
  uint64_t timdData;
  uint64_t timdOut;
  uint64_t tmrlOut;
  uint64_t ccpOut;
  uint64_t ccpData;
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
  int32_t  trTables;
  int32_t  ctuSize;
  int32_t  lumaUnitW, lumaUnitH;
  int32_t  cclmFilter;
  int32_t  dualTree;
  int32_t  cccmLbccp;
  int32_t  cccmScratchOffset;
  int32_t  cccmScratchWords;
  int32_t  ccpFusionPairs;
  int32_t  pdpTable;
};
static_assert( offsetof( Params, origin ) == 304 && offsetof( Params, edgeCap ) == 384 && sizeof( Params ) == 448,
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

// intra_rec's pipelines (its specialization constants WITH_DIMD, WITH_DERIVED_TRANSFORMS, WITH_MERGE, WITH_TIMD,
// WITH_TIMD_SAD, WITH_TIMD_MERGE, WITH_TMRL): code that never runs still slows every block, so a picture runs the
// pipeline without the features it does not use. The one with DIMD has the derived transforms too: the compiler made the
// one with DIMD alone slower in every case measured (16.8 instead of 14.2 ms per picture without either feature, 46.3
// instead of 42.4 with DIMD); intra merge's code slowed the pictures with DIMD but without intra merge by ~11 %,
// TIMD-SAD's the pictures with TIMD but without it by ~5 %
enum IntraRecVariant
{
  VARIANT_PLAIN,                // neither DIMD (Cu::dimd, DERIVE_DIMD) nor blocks with DERIVE_TRANSFORM
  VARIANT_DERIVED_TRANSFORMS,   // blocks with DERIVE_TRANSFORM, no DIMD
  VARIANT_FULL,                 // DIMD (and OBIC), no intra merge
  VARIANT_MERGE,                // DIMD with intra merge CUs
  VARIANT_TIMD,                 // TIMD CUs (with DIMD and intra merge)
  VARIANT_TIMD_SAD,             // TIMD-SAD CUs (with TIMD)
  VARIANT_TIMD_MERGE,           // TIMD merge CUs (with TIMD-SAD)
  VARIANT_TMRL,                 // TMRL CUs (with TIMD merge)
  NUM_VARIANTS
};

// and the LM modes it has (WITH_CCLM, WITH_CCCM, WITH_CFLM, WITH_CCP_MERGE, WITH_CCP_DERIVED; CFLM's code slowed the
// CCCM pictures by ~2 %; the CCP merge pictures without decoder-derived CCP fusion take a pipeline without its
// branches). LEVEL_CCP_LFNST's pipelines are those of intra_rec's second module (spv_intra_rec_lfnst, compiled with
// CCP_LFNST_MODULE): CCP merge or decoder-derived CCP fusion blocks with LFNST / NSPT
enum LmLevel
{
  LEVEL_NO_LM,
  LEVEL_CCLM,
  LEVEL_CCCM,          // with CCLM
  LEVEL_CFLM,          // with CCCM
  LEVEL_CCP_MERGE,     // with CFLM
  LEVEL_CCP_DERIVED,   // with CCP merge
  LEVEL_CCP_LFNST,     // with decoder-derived CCP fusion
  NUM_LM_LEVELS
};

// the kernels and the constant tables, created once and shared by every instance (the decoder contexts of the frame
// threads); destroyed by GpuIntraRec::releaseShared, before the device
struct Shared
{
  // intra_rec, per IntraRecVariant and LmLevel (intraRecKernel: on first use)
  mutable gpu::Kernel              kernel[NUM_VARIANTS][NUM_LM_LEVELS];
  mutable std::once_flag           kernelOnce[NUM_VARIANTS][NUM_LM_LEVELS];
  gpu::Kernel                      kernelResi, kernelDeps, kernelScan;
  gpu::Buffer                      filters, matrices;
  gpu::Buffer                      secKernels;                    // int8: the CPU's LFNST and NSPT tables, one after the other
  int32_t                          trOffset[NUM_TRANS_TYPE][7];   // of the inverse transform matrices of sizes 2..128 in matrices (-1: none)
  int32_t                          qcsOffset = 0;                 // of the QCS weights in matrices
  int32_t                          mipOffset[3] = {};             // of the MIP matrices of each size class in matrices
  int32_t                          lfnstLayout[3][2] = {};        // of the LFNST output positions in matrices (sbSize 4, 8, 16; transposed)
  int32_t                          trTables = 0;                  // of the transform tables' directory in matrices (TR_TAB_*)
  int32_t                          pdpTable = 0;                  // of PDP's filters' offsets in matrices
  std::vector<int32_t>             pnnModel;                      // of each NN intra prediction model in matrices
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

// intra_rec's pipeline of a variant, compiled the first time a picture needs it (by any thread: the decoder contexts
// share it). Compiling all of them up front cost ~11 s each where the driver's shader cache misses (a new build), and a
// stream rarely needs more than two. Not while the caller's stream is recording: creating a pipeline waits until no
// submission is in flight (gpu::Context::inFlight), which includes the caller's own
gpu::Kernel& intraRecKernel( const Shared& sh, gpu::Context& ctx, IntraRecVariant variant, LmLevel lm )
{
  // the names (the kernel keeps the pointer) and the specialization constants (WITH_DIMD, WITH_DERIVED_TRANSFORMS,
  // WITH_MERGE, WITH_TIMD, WITH_TIMD_SAD, WITH_TIMD_MERGE, WITH_TMRL; then WITH_CCLM, WITH_CCCM, WITH_CFLM,
  // WITH_CCP_MERGE, WITH_CCP_DERIVED) per IntraRecVariant and LmLevel, LEVEL_CCP_LFNST's of the second module
  static const char* const NAMES[NUM_VARIANTS][NUM_LM_LEVELS] = {
    { "intra_rec", "intra_rec (CCLM)", "intra_rec (CCLM, CCCM)", "intra_rec (CCLM, CCCM, CFLM)",
      "intra_rec (CCLM, CCCM, CFLM, CCP merge)", "intra_rec (CCLM, CCCM, CFLM, CCP merge, derived CCP)",
      "intra_rec (CCLM, CCCM, CFLM, CCP merge, derived CCP, CCP LFNST)" },
    { "intra_rec (derived transforms)", "intra_rec (derived transforms, CCLM)",
      "intra_rec (derived transforms, CCLM, CCCM)", "intra_rec (derived transforms, CCLM, CCCM, CFLM)",
      "intra_rec (derived transforms, CCLM, CCCM, CFLM, CCP merge)",
      "intra_rec (derived transforms, CCLM, CCCM, CFLM, CCP merge, derived CCP)",
      "intra_rec (derived transforms, CCLM, CCCM, CFLM, CCP merge, derived CCP, CCP LFNST)" },
    { "intra_rec (DIMD)", "intra_rec (DIMD, CCLM)", "intra_rec (DIMD, CCLM, CCCM)",
      "intra_rec (DIMD, CCLM, CCCM, CFLM)", "intra_rec (DIMD, CCLM, CCCM, CFLM, CCP merge)",
      "intra_rec (DIMD, CCLM, CCCM, CFLM, CCP merge, derived CCP)",
      "intra_rec (DIMD, CCLM, CCCM, CFLM, CCP merge, derived CCP, CCP LFNST)" },
    { "intra_rec (intra merge)", "intra_rec (intra merge, CCLM)", "intra_rec (intra merge, CCLM, CCCM)",
      "intra_rec (intra merge, CCLM, CCCM, CFLM)", "intra_rec (intra merge, CCLM, CCCM, CFLM, CCP merge)",
      "intra_rec (intra merge, CCLM, CCCM, CFLM, CCP merge, derived CCP)",
      "intra_rec (intra merge, CCLM, CCCM, CFLM, CCP merge, derived CCP, CCP LFNST)" },
    { "intra_rec (TIMD)", "intra_rec (TIMD, CCLM)", "intra_rec (TIMD, CCLM, CCCM)",
      "intra_rec (TIMD, CCLM, CCCM, CFLM)", "intra_rec (TIMD, CCLM, CCCM, CFLM, CCP merge)",
      "intra_rec (TIMD, CCLM, CCCM, CFLM, CCP merge, derived CCP)",
      "intra_rec (TIMD, CCLM, CCCM, CFLM, CCP merge, derived CCP, CCP LFNST)" },
    { "intra_rec (TIMD-SAD)", "intra_rec (TIMD-SAD, CCLM)", "intra_rec (TIMD-SAD, CCLM, CCCM)",
      "intra_rec (TIMD-SAD, CCLM, CCCM, CFLM)", "intra_rec (TIMD-SAD, CCLM, CCCM, CFLM, CCP merge)",
      "intra_rec (TIMD-SAD, CCLM, CCCM, CFLM, CCP merge, derived CCP)",
      "intra_rec (TIMD-SAD, CCLM, CCCM, CFLM, CCP merge, derived CCP, CCP LFNST)" },
    { "intra_rec (TIMD merge)", "intra_rec (TIMD merge, CCLM)", "intra_rec (TIMD merge, CCLM, CCCM)",
      "intra_rec (TIMD merge, CCLM, CCCM, CFLM)", "intra_rec (TIMD merge, CCLM, CCCM, CFLM, CCP merge)",
      "intra_rec (TIMD merge, CCLM, CCCM, CFLM, CCP merge, derived CCP)",
      "intra_rec (TIMD merge, CCLM, CCCM, CFLM, CCP merge, derived CCP, CCP LFNST)" },
    { "intra_rec (TMRL)", "intra_rec (TMRL, CCLM)", "intra_rec (TMRL, CCLM, CCCM)",
      "intra_rec (TMRL, CCLM, CCCM, CFLM)", "intra_rec (TMRL, CCLM, CCCM, CFLM, CCP merge)",
      "intra_rec (TMRL, CCLM, CCCM, CFLM, CCP merge, derived CCP)",
      "intra_rec (TMRL, CCLM, CCCM, CFLM, CCP merge, derived CCP, CCP LFNST)" } };
  static const uint32_t SPECIALIZATION[NUM_VARIANTS][7] = {
    { 0, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0, 0 }, { 1, 1, 1, 0, 0, 0, 0 },
    { 1, 1, 1, 1, 0, 0, 0 }, { 1, 1, 1, 1, 1, 0, 0 }, { 1, 1, 1, 1, 1, 1, 0 }, { 1, 1, 1, 1, 1, 1, 1 } };
  std::call_once( sh.kernelOnce[variant][lm], [&] {
    std::vector<uint32_t> constants( SPECIALIZATION[variant], SPECIALIZATION[variant] + 7 );
    constants.push_back( lm != LEVEL_NO_LM ? 1 : 0 );
    constants.push_back( lm >= LEVEL_CCCM ? 1 : 0 );
    constants.push_back( lm >= LEVEL_CFLM ? 1 : 0 );
    constants.push_back( lm >= LEVEL_CCP_MERGE ? 1 : 0 );
    constants.push_back( lm >= LEVEL_CCP_DERIVED ? 1 : 0 );
    const bool lfnst = lm == LEVEL_CCP_LFNST;
    sh.kernel[variant][lm].create( ctx, lfnst ? spv_intra_rec_lfnst : spv_intra_rec,
                                   lfnst ? sizeof( spv_intra_rec_lfnst ) : sizeof( spv_intra_rec ), sizeof( Push ),
                                   NAMES[variant][lm], constants );
  } );
  return sh.kernel[variant][lm];
}

Shared* createShared( gpu::Context& ctx )
{
  Shared* sh = new Shared;
  sh->kernelResi.create( ctx, spv_intra_resi, sizeof( spv_intra_resi ), sizeof( Push ), "intra_resi" );
  sh->kernelDeps.create( ctx, spv_intra_deps, sizeof( spv_intra_deps ), sizeof( Push ), "intra_deps" );
  sh->kernelScan.create( ctx, spv_scan, sizeof( spv_scan ), sizeof( PushScan ), "scan" );

  // tables, from the C++ tables themselves
  sh->bif.reset( new BilateralFilter );   // BilateralFilter::create derives the LUTs
  sh->bif->create();
  std::vector<int32_t> filt( 5 * 32 * 16 + 32 * 4 + 26 * 48 + 64 + 16 + 64 * 4 + 5 * 64 * 16, 0 );
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
    // intra_rec's EXT_CUBIC: the 1/64 cubic filter of template predictions (xIntraPredTimdAngLuma)
    int32_t* ec = bt + 26 * 48 + 64 + 16;
    for( int frac = 0; frac < 64; frac++ ) for( int i = 0; i < 4; i++ ) ec[frac * 4 + i] = InterpolationFilter::getExtIntraCubicFilter( frac )[i];
    // intra_rec's EXT_NTAP: the 1/64 N-tap tables of TIMD blocks (getIntraNtapNonSmoothingFilter with bExtIntraDir), as
    // the 1/32 ones above
    int32_t* en = ec + 64 * 4;
    for( int t = 0; t < 5; t++ )
    {
      for( int frac = 0; frac < 64; frac++ )
      {
        const TFilterCoeff* f = InterpolationFilter::getIntraNtapNonSmoothingFilter( true, frac, FILTER_TAPS[t] );
        for( int i = 0; i < FILTER_TAPS[t]; i++ )
        {
          en[( t * 64 + frac ) * 16 + i] = f[i];
        }
      }
    }
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
  // PDP (intra_pdp.glsl): per mode, then per MIP filter set (JVET-AI0208), and per size (g_size's first PDP_SIZES) the
  // offset in int16 entries of its filters (-1: none, or a mode xPredIntraOpt rejects), then the filters of every mode
  // and size (createPdpFilters' and createMipFilters' layout: per row of the block and 4 columns len4 weights; DecApp
  // creates them before the decoder), two int16 per int32
  sh->pdpTable = (int32_t) mat.size();
  mat.resize( mat.size() + ( NUM_LUMA_MODE + PDP_MIP_GROUPS ) * PDP_SIZES, -1 );
#if JVET_AH0209_PDP
  {
    const int32_t        weights = (int32_t) mat.size();
    std::vector<int16_t> pdp;
    for( int set = 0; set < NUM_LUMA_MODE + PDP_MIP_GROUPS; set++ )
    {
      for( int size = 0; size < PDP_SIZES; size++ )
      {
        const int  mode = set;   // (of a regular set)
        const bool mip  = set >= NUM_LUMA_MODE;
#if JVET_AI0208_PDP_MIP
        int16_t*** f = mip ? g_pdpFiltersMip[set - NUM_LUMA_MODE][size] : g_pdpFilters[mode][size];
#else
        int16_t*** f = mip ? nullptr : g_pdpFilters[mode][size];
#endif
        if( !f || ( !mip && mode > 1 && mode % 2 ) || ( !mip && size > 12 && mode > 1 && mode % 4 != 2 ) )
        {
          continue;
        }
        const bool shortLen = !mip
                              && ( mode < PDP_SHORT_TH[0] || ( mode >= PDP_SHORT_TH[1] && mode <= PDP_SHORT_TH[2] ) );
        const int  len4     = ( ( shortLen ? g_sizeData[size][10] : g_sizeData[size][7] ) + 7 ) / 8 * 32;
        mat[sh->pdpTable + set * PDP_SIZES + size] = 2 * weights + (int32_t) pdp.size();
        for( int y = 0; y < g_sizeData[size][4]; y++ )
        {
          for( int q = 0; q < g_sizeData[size][3] / 4; q++ )
          {
            pdp.insert( pdp.end(), f[y][q], f[y][q] + len4 );
          }
        }
      }
    }
    for( size_t i = 0; i < pdp.size(); i += 2 )   // (each mode's filters are a multiple of 32 entries)
    {
      mat.push_back( int32_t( uint16_t( pdp[i] ) ) | int32_t( uint32_t( uint16_t( pdp[i + 1] ) ) << 16 ) );
    }
  }
#endif
  // the NN intra prediction's models (intra_nn.glsl, GpuPnn.h; none where one has what the kernels lack)
  if( !appendPnnModels( mat, sh->pnnModel ) )
  {
    sh->pnnModel.clear();
  }
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
  // the transforms of a mode intra_rec derives (intra_modes.glsl's selectTransform, TR_TAB_*): a directory of the
  // tables' offsets, then the inverse transform matrices' offsets per type and size, ECM's explicit MTS kernel sets
  // (getTrTypes), the LFNST / NSPT sets of the modes and the offsets in secKernels of their matrices per size class,
  // set, index (and NSPT bucket), found with the CPU's own lookups (TrQuant::getNsptMatrix), and the LFNST output
  // arrangements
  {
    std::vector<int32_t> tab( TR_TAB_NUM, 0 );
    auto add = [&]( const int which, const std::vector<int32_t>& values )
    {
      tab[which] = (int32_t) ( mat.size() + tab.size() );
      tab.insert( tab.end(), values.begin(), values.end() );
    };
    std::vector<int32_t> v;
    for( int type = 0; type < NUM_TRANS_TYPE; type++ ) for( int l = 0; l < 7; l++ ) v.push_back( sh->trOffset[type][l] );
    add( TR_TAB_MATRIX, v );
    add( TR_TAB_IPM_TO_SET, std::vector<int32_t>( &g_aucIpmToTrSet[0][0], &g_aucIpmToTrSet[0][0] + sizeof( g_aucIpmToTrSet ) ) );
    add( TR_TAB_SET, std::vector<int32_t>( &g_aucTrSet[0][0], &g_aucTrSet[0][0] + sizeof( g_aucTrSet ) ) );
    add( TR_TAB_IDX_TO_TR, std::vector<int32_t>( &g_aucTrIdxToTr[0][0], &g_aucTrIdxToTr[0][0] + sizeof( g_aucTrIdxToTr ) ) );
    add( TR_TAB_ID_LUT, std::vector<int32_t>( &g_aiIdLut[0][0], &g_aiIdLut[0][0] + sizeof( g_aiIdLut ) ) );
    add( TR_TAB_LFNST_LUT, std::vector<int32_t>( g_lfnstLut, g_lfnstLut + sizeof( g_lfnstLut ) ) );
    add( TR_TAB_NSPT_LUT, std::vector<int32_t>( g_nsptLut, g_nsptLut + sizeof( g_nsptLut ) ) );
    v.clear();
    for( int s = 0; s < 3; s++ )   // sbSize 4, 8, 16
    {
      for( int set = 0; set < 35; set++ )
      {
        for( int idx = 0; idx < 3; idx++ )
        {
          const int8_t* m = s == 0 ? &g_lfnst4x4[set][idx][0][0] : s == 1 ? &g_lfnst8x8[set][idx][0][0] : &g_lfnst16x16[set][idx][0][0];
          v.push_back( sh->kernelOffset( m ) );
          CHECK( v.back() < 0, "an LFNST matrix outside the kernels" );
        }
      }
    }
    add( TR_TAB_LFNST_MATRIX, v );
    v.clear();
    for( int shape = 0; shape < NSPT_SHAPES; shape++ )
    {
      for( int set = 0; set < 35; set++ )
      {
        for( int idx = 0; idx < 3; idx++ )
        {
          for( int bucket = 0; bucket < 3; bucket++ )
          {
            v.push_back( sh->kernelOffset( TrQuant::getNsptMatrix( set, NSPT_SHAPE_W[shape], NSPT_SHAPE_H[shape], idx, bucket ) ) );
            CHECK( v.back() < 0, "an NSPT matrix outside the kernels" );
          }
        }
      }
    }
    add( TR_TAB_NSPT_MATRIX, v );
    v.assign( &sh->lfnstLayout[0][0], &sh->lfnstLayout[0][0] + 6 );
    add( TR_TAB_LFNST_LAYOUT, v );
    tab[TR_TAB_EXT_MODES] = NUM_EXT_LUMA_MODE;
    tab[TR_TAB_SET_ROW]   = (int32_t) ( sizeof( g_aucTrSet[0] ) );
    tab[TR_TAB_IDTR]      = IDTR;
    sh->trTables = (int32_t) mat.size();
    mat.insert( mat.end(), tab.begin(), tab.end() );
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
  gpu::Buffer                 scratch;            // intra_rec's: DIMD / OBIC blending's sums, then CCCM's data
  Reshape*                    reshape = nullptr;   // the decoder's reshaper (LMCS state of the picture being decoded)
  TrQuant*                    trQuant = nullptr;   // the decoder's TrQuant (transform types of the blocks)
  const IntraPrediction*      intraPred = nullptr; // the decoder's intra prediction (its intra merge template costs)
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
  // the GPU derives intra modes in the picture: with DIMD the MPM lists depend on the reconstruction, with TIMD or TMRL
  // their CUs' modes, else where gpuDerivesModes() tests it
  bool                        deriveModes = false;
  std::vector<Cu>             cus;                // the CUs of the recorded blocks (Tu::cu), where it does
  std::vector<int32_t>        cuMode;             // per CU the host's mode (CU_KNOWN), else -1 (Params::cuMode)
  std::vector<DimdCall>       dimdCalls;          // the CUs' DIMD histograms (Cu::dimd)
  std::vector<int32_t>        cpuDimd;            // per CU the CPU's DIMD (DIMD_OUT), compared in verification
  std::vector<ObicNbr>        obicNbrs;           // per OBIC CU its neighbours (OBIC_NBRS, Cu::obic)
  std::vector<int32_t>        cpuObic;            // per OBIC CU the CPU's OBIC (OBIC_OUT), compared in verification
  // the DIMD histograms of an MPM CU whose list starts with planar: the CPU derives its DIMD all the same, which
  // only an intra merge CU reads (its dimdBlending); kept aside per CU (first, number; -1: none) until one does
  std::vector<std::pair<int32_t, int32_t>> pendingDimd;
  std::vector<int32_t>        mergeData;          // the intra merge CUs' records (Cu::merge, MergeHead)
  int                         numMerge = 0;
  std::vector<uint64_t>       cpuMergeCost;       // per intra merge CU the CPU's template costs (MAX_UINT64 also where none)
  std::vector<uint8_t>        cpuMergeCostValid;  // per intra merge CU: the CPU ranked by template costs
  std::vector<int32_t>        timdData;           // the TIMD CUs' records (Cu::timd, TimdHead)
  int                         numTimd = 0;
  int                         numTimdSad = 0;     // of which TIMD-SAD CUs
  int                         numTimdMerge = 0;   // and TIMD merge CUs
  std::vector<int32_t>        cpuTimd;            // per TIMD CU the CPU's TIMD (TIMD_OUT), compared in verification
  std::vector<int32_t>        cpuTmrl;            // per TMRL CU the CPU's mode and line (TMRL_OUT), compared in verification
  std::vector<uint8_t>        cuMipDimd;          // per CU: a MIP CU whose block derives mipDimdMode (DERIVE_MIP_DIMD)
  bool                        blendBlocks = false;   // a block of a DIMD / OBIC / merge CU was recorded (the
                                                     // workgroups' blending sums, BLEND_ACC, are needed)
  int32_t                     cccmScratchWords = 0;  // the CCCM scratch each intra_rec workgroup needs (int32): the
                                                     // largest CCCM block's data beyond shared memory
  std::unordered_map<const CodingUnit*, int32_t> cuIndex[2];   // per channel type: the record of a CU
  // the CCP models (CCP merge's curCand) of the LM CUs and LM chroma fusion CUs the GPU computes (Tu::ccp: per PU,
  // both components' blocks write it), the PU of each record (verification compares its curCand after the picture)
  int                                                  numCcp = 0;
  std::unordered_map<const PredictionUnit*, int32_t>  ccpIndex;
  std::vector<const PredictionUnit*>                   ccpPu;
  std::vector<const PredictionUnit*>                   untypedCcpPu;   // the chroma PUs without a record
  // CCP merge (Params::ccpData, assembled at endPicture): per record its blocks (CCP_WRITERS), the records
  // CU::saveModelsInHCCP pushes into the CTU rows' histories (cs.ccpLut) in decoding order, the current row and where
  // its history begins, the merge CUs' entries (their history's position relative to ccpHist, their ordinal for their
  // default records until endPicture places them), per merge PU its entry, the merge CUs
  std::vector<int32_t>                                 ccpWriters;
  std::vector<int32_t>                                 ccpHist;
  int                                                  ccpHistRow = -1;
  size_t                                               ccpHistRowStart = 0;
  std::vector<int32_t>                                 ccpMerge;
  std::unordered_map<const PredictionUnit*, int32_t>  ccpMergeIndex;
  int                                                  numCcpMerge = 0;   // the entries (also decoder-derived fusion's)
  int                                                  numCcpAdjust = 0;   // the merge blocks with adjustment
  int                                                  numDerivedCcp = 0;  // the blocks of decoder-derived CCP fusion
  int                                                  numDdMergeFusion = 0;   // and those of its merge fusion
  int                                                  numFusionBlend = 0;  // merge fusion blocks by a blend model
  int                                                  numCcpLfnst = 0;     // merge / derived blocks with LFNST
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
  // partial verification (ECM_GPU_PARTIAL=1 with --GpuRecon=2): a CU with a block the GPU cannot reconstruct yet is
  // taken from the CPU (its samples uploaded with the picture) instead of sending the picture to the CPU, and a CU
  // whose mode derivation the GPU lacks takes the CPU's mode (CU_KNOWN); the GPU's other blocks are verified as usual,
  // so the regular streams test every tool ported so far
  bool                                                         partial = false;
  std::unordered_map<const CodingUnit*, std::vector<int32_t>> cuTus;    // each CU's blocks and node (rec indices)
  std::unordered_map<const CodingUnit*, int>                   cpuCus;   // the CUs taken from the CPU
  int32_t                     cpuDummyCu = -1;   // the record of the nodes that replace blocks recorded before
  // CCP records of such CUs, from the CPU's curCand: the record, the model, whether a merge CU's (its inside filter
  // flag also the history's); a non-intra CU's per TU (JVET-AF0073: its TUs' models are candidates and history)
  struct CpuCcp
  {
    int32_t                  record;
    const CCPModelCandidate* cand;
    bool                     merge;
  };
  std::vector<CpuCcp>                                          cpuCcp;
  std::unordered_map<const TransformUnit*, int32_t>            ccpTuIndex;
  std::map<std::string, int>  cpuReasons;        // totals: why CUs were taken from the CPU

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
                           &sched, &dbgPred, &dbgResi, &lmcs, &signPredBuf, &cuBuf, &dbgParams, &scratch,
                           &upload, &download } )
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
      if( !m.cpuReasons.empty() )
      {
        std::vector<std::pair<int, std::string>> byCount;
        for( const auto& r: m.cpuReasons )
        {
          byCount.emplace_back( r.second, r.first );
        }
        std::sort( byCount.rbegin(), byCount.rend() );
        printf( "[gpu-intra] CUs taken from the CPU:" );
        for( const auto& r: byCount )
        {
          printf( " %s %d;", r.second.c_str(), r.first );
        }
        printf( "\n" );
      }
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

void GpuIntraRec::setIntraPrediction( const IntraPrediction* intraPred )
{
  m_impl->intraPred = intraPred;
}

bool GpuIntraRec::on() const
{
  return m_impl->on;
}

bool GpuIntraRec::failed() const
{
  return !m_impl->unsupportedWhy.empty();
}

bool GpuIntraRec::supported( const Slice& slice, std::string& why, bool partial )
{
  // every tool whose mode derivation or prediction reads reconstructed samples, or that the kernels do not implement,
  // must be off: with the GPU reconstruction on, the CPU reconstructs nothing the derivations could read. Partial
  // verification takes the CUs of such tools from the CPU (record) and checks the tool families below per CU
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
  else if( sps.getUseTmrl() && !GPU_TMRL )                                                why = "TMRL of another configuration";
  else if( sps.getUseLMChroma() && !GPU_CCLM )                                           why = "CCLM of another configuration";
  else if( sps.getUseCccm() && !GPU_CCCM )                                                why = "CCCM of another configuration";
  else if( sps.getUseDimd() && !GPU_DIMD )                                                why = "DIMD of another configuration";
  else if( sps.getUseTimd() && !GPU_TIMD )                                                why = "TIMD of another configuration";
  else if( partial )                                                                      return true;
  else if( sps.getUseChromaReordering() || sps.getUseIntraDBV() )                         why = "chroma reordering / DBV";
  else if( sps.getUseBvgDimd() )                                                          why = "BVG-DIMD";
  else if( sps.getUseMdip() || sps.getUseMpmSorting() )                                   why = "MDIP / MPM sorting";
  // (also what leaves TIMD's block vector candidates empty, JVET-AM0138's: they come from IntraTMP / IBC CUs and the
  // CUs that took a vector from one; EIP and SGPM below)
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
  m.deriveModes = gpuDerivesModes() || cs.sps->getUseDimd() || cs.sps->getUseTimd() || cs.sps->getUseTmrl();
  m.cus.clear();
  m.cuMode.clear();
  m.dimdCalls.clear();
  m.cpuDimd.clear();
  m.obicNbrs.clear();
  m.cpuObic.clear();
  m.pendingDimd.clear();
  m.mergeData.clear();
  m.numMerge = 0;
  m.cpuMergeCost.clear();
  m.cpuMergeCostValid.clear();
  m.timdData.clear();
  m.numTimd = 0;
  m.numTimdSad = 0;
  m.numTimdMerge = 0;
  m.cpuTimd.clear();
  m.cpuTmrl.clear();
  m.cuMipDimd.clear();
  m.blendBlocks = false;
  m.cccmScratchWords = 0;
  m.numCcp = 0;
  m.ccpIndex.clear();
  m.ccpPu.clear();
  m.untypedCcpPu.clear();
  m.ccpWriters.clear();
  m.ccpHist.clear();
  m.ccpHistRow      = -1;
  m.ccpHistRowStart = 0;
  m.ccpMerge.clear();
  m.ccpMergeIndex.clear();
  m.numCcpMerge = 0;
  m.numCcpAdjust = 0;
  m.numDerivedCcp = 0;
  m.numDdMergeFusion = 0;
  m.numFusionBlend = 0;
  m.numCcpLfnst = 0;
  m.cuIndex[0].clear();
  m.cuIndex[1].clear();
  m.unsupportedWhy.clear();
  m.poc = cs.slice->getPOC();
  m.partial = !m.on && getenv( "ECM_GPU_PARTIAL" );
  m.cuTus.clear();
  m.cpuCus.clear();
  m.cpuDummyCu = -1;
  m.cpuCcp.clear();
  m.ccpTuIndex.clear();
  std::string why;
  if( !supported( *cs.slice, why, m.partial ) )
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

#if JVET_AH0209_PDP
// PDP (JVET-AH0209): the size index (g_size) of a luma block whose prediction is PDP's where its mode has filters
// (predIntraAng, intra_rec's pdpFilters; MIP's own filters: predIntraMip), -1 for others: a size of g_size, its
// references available (the snapshot's), its CU without the tools that exclude PDP
static int pdpSize( const CodingUnit& cu, const PredictionUnit& pu, const ComponentID compID,
                    const IntraPrediction::IpaSnapshot& ipa )
{
  if( !isLuma( compID ) || !cu.cs->sps->getUsePDP() || !ipa.pdpRefAvailable || cu.ispMode || cu.bdpcmMode || cu.plIdx
      || cu.sgpm || cu.timd || cu.tmrlFlag || pu.multiRefIdx || pu.intraDir[CHANNEL_TYPE_LUMA] == PNN_IDX )
  {
    return -1;
  }
  const auto size = g_size.find( ( int( pu.lwidth() ) << 8 ) + int( pu.lheight() ) );
  return size != g_size.end() ? size->second : -1;
}
#endif

// a CU record that derives nothing (CU_KNOWN, mode `mode`), appended with what each record has beside it
static int32_t knownCuRecord( GpuIntraRec::Impl& m, const CompArea& a, int32_t mode )
{
  Cu c{};
  c.kind = CU_KNOWN;
  c.node = c.lumaCu = c.dimd = c.obic = c.merge = c.mtss = c.timd = c.tmrl = -1;
  std::fill_n( c.nbr, MPM_NEIGHBOURS, -1 );
  c.modes[0] = c.modes[1] = -1;
  c.x = a.x;
  c.y = a.y;
  c.w = a.width;
  c.h = a.height;
  const int32_t index = (int32_t) m.cus.size();
  m.cus.push_back( c );
  m.cuMode.push_back( mode );
  m.pendingDimd.push_back( { -1, 0 } );
  m.cuMipDimd.push_back( 0 );
#if GPU_DIMD
  m.cpuDimd.insert( m.cpuDimd.end(), DIMD_OUT, 0 );
#endif
  return index;
}

// the luma mode of PU pu as the chroma CUs read a co-located one (getCoLocatedIntraLumaMode): an EIP CU's DIMD mode, a
// TIMD or TMRL CU's mapped to the 67 modes, NN intra's planar, else its mode. (getSpatialIpm reads an SGPM CU's
// partition angle instead, getChromaTmrlSearchRange an EIP CU's own mode: the derivations partial verification
// leaves on the GPU rarely meet such CPU-filled CUs)
static int32_t neighbourLumaMode( const PredictionUnit& pu )
{
#if JVET_AG0058_EIP
  if( pu.cu->eipFlag )
  {
    return pu.cu->eipModel.eipDimdMode;
  }
#endif
  int32_t mode = PU::getIntraDirLuma( pu );
  if( pu.cu->timd || pu.cu->tmrlFlag )
  {
    mode = MAP131TO67( mode );
  }
#if JVET_AJ0249_NEURAL_NETWORK_BASED
  else if( mode == PNN_IDX )
  {
    mode = PLANAR_IDX;
  }
#endif
  return mode;
}

static int32_t ccpRecord( GpuIntraRec::Impl& m, const PredictionUnit& pu, bool* isNew );
static void    cpuCcpHistory( GpuIntraRec::Impl& m, const CodingUnit& cu, int32_t r );

#if JVET_AD0188_CCP_MERGE
// partial verification: the CCP record of a CU taken from the CPU, its model h in the words the GPU writes for its
// type (cclmStoreCcp, cccmStoreCcp; a merge CU's inside filter flag also as the history's)
static void cpuCcpWords( const CCPModelCandidate& h, bool merge, int32_t* o )
{
  std::fill_n( o, CCP_OUT, 0 );
  auto put = [&]( bool second, int c, int p, int64_t v )
  {
    const int i = ( second ? CCP_OUT_PARAMS2 : CCP_OUT_PARAMS ) + 2 * ( c * CCP_MAX_PARAMS + p );
    o[i]     = int32_t( uint32_t( uint64_t( v ) ) );
    o[i + 1] = int32_t( uint32_t( uint64_t( v ) >> 32 ) );
  };
  const bool two = ( h.type & CCP_TYPE_MMLM ) != 0;
  o[CCP_OUT_TYPE]        = h.type;
  o[CCP_OUT_INSIDE]      = h.ccInsideFilter;
  o[CCP_OUT_INSIDE_HIST] = merge ? h.ccInsideFilter : 0;
  o[CCP_OUT_Y_THRES]     = two ? h.yThres : 0;
  if( h.type & ( CCP_TYPE_CCLM | CCP_TYPE_GLM0123 ) )
  {
    o[CCP_OUT_GLM_IDC]    = h.glmIdc;
    o[CCP_OUT_NUM_PARAMS] = 2;
    for( int c = 0; c < 2; c++ )
    {
      o[CCP_OUT_SHIFT + c] = h.shift[c];
      put( false, c, 0, h.params[c][0] );
      put( false, c, 1, h.params[c][1] );
      if( two )
      {
        o[CCP_OUT_SHIFT2 + c] = h.shift2[c];
        put( true, c, 0, h.params2[c][0] );
        put( true, c, 1, h.params2[c][1] );
      }
    }
    return;
  }
  // (CCCM's kinds and GLM with luma's: every field, the second model's where there are two)
  o[CCP_OUT_GLM_IDC]     = h.glmIdc;
  o[CCP_OUT_LUMA_OFFSET] = h.lumaOffset;
  o[CCP_OUT_COR_OFF]     = h.corOffX;
  o[CCP_OUT_COR_OFF + 1] = h.corOffY;
  o[CCP_OUT_MDF]         = h.cccmMultiFilterIdx;
  o[CCP_OUT_NUM_PARAMS]  = h.numParams;
  const int ranges[2][2][3] = { { { h.rangeMinCb, h.rangeMaxCb, h.rangeChangeCb }, { h.rangeMinCr, h.rangeMaxCr, h.rangeChangeCr } },
                                { { h.rangeMinCb2, h.rangeMaxCb2, h.rangeChangeCb2 },
                                  { h.rangeMinCr2, h.rangeMaxCr2, h.rangeChangeCr2 } } };
  for( int k = 0; k < ( two ? 2 : 1 ); k++ )
  {
    for( int c = 0; c < 2; c++ )
    {
      std::copy_n( ranges[k][c], 3, o + CCP_OUT_RANGE + 3 * ( 2 * k + c ) );
      for( int p = 0; p < std::min( h.numParams, CCP_MAX_PARAMS ); p++ )
      {
        put( k != 0, c, p, k ? h.params2[c][p] : h.params[c][p] );
      }
    }
  }
}
#endif

// partial verification: block area a of component comp taken from the CPU, at its place in decoding order (a node of
// the record that derives nothing, DERIVE_CPU_FILL: it owns its area, its samples come with the picture)
static void cpuFillBlock( GpuIntraRec::Impl& m, const ComponentID comp, const CompArea& a )
{
  Tu t{};
  t.comp      = comp;
  t.x         = a.x;
  t.y         = a.y;
  t.w         = a.width;
  t.h         = a.height;
  t.dbgOffset = (int32_t) m.numSamples;
  t.mipMatrix = -1;
  t.ispPrev = t.spCoded = t.spOffset = t.resiFrom = t.cflm = t.ccp = t.ccpMerge = t.pdp = t.pnn = -1;
  t.derive    = DERIVE_NODE | DERIVE_CPU_FILL;
  t.cu        = m.cpuDummyCu;
  m.rec.push_back( t );
}

// partial verification: CU cu from the CPU (why: the first reason). Its samples come with the picture (endPicture);
// its node and blocks recorded before become nodes of a record that derives nothing, the blocks owning their areas
// (DERIVE_CPU_FILL); an intra CU keeps a record of its mode as the CPU decoded it (CU_KNOWN) for the CUs reading it
static void cpuFillCu( GpuIntraRec::Impl& m, const CodingUnit& cu, const char* why )
{
  if( !m.cpuCus.emplace( &cu, 0 ).second )
  {
    return;
  }
  m.cpuReasons[why]++;
  if( m.cpuDummyCu < 0 )
  {
    m.cpuDummyCu = knownCuRecord( m, CompArea(), -1 );
  }
  const auto recorded = m.cuTus.find( &cu );
  if( recorded != m.cuTus.end() )
  {
    for( const int32_t i: recorded->second )
    {
      Tu& t = m.rec[i];
      t.derive    = DERIVE_NODE | DERIVE_CPU_FILL;
      t.cu        = m.cpuDummyCu;
      t.mipMatrix = -1;
      t.ispPrev = t.spCoded = t.spOffset = t.resiFrom = t.cflm = t.ccp = t.ccpMerge = t.pdp = t.pnn = -1;
    }
  }
  if( !CU::isIntra( cu ) )
  {
    return;
  }
  for( int ch = 0; ch < 2; ch++ )
  {
    const ComponentID first = ch == 0 ? COMPONENT_Y : COMPONENT_Cb;
    if( !cu.blocks[first].valid() || ( ch == 1 && cu.chromaFormat == CHROMA_400 ) )
    {
      continue;
    }
    const PredictionUnit& pu   = *cu.firstPU;
    const int32_t         mode = ch == 0 ? neighbourLumaMode( pu ) : (int32_t) PU::getFinalIntraMode( pu, CHANNEL_TYPE_CHROMA );
#if JVET_AD0188_CCP_MERGE
    // its CCP model, the CPU's, for the merge CUs (getCCPModelCandidateList: any typed curCand of an intra PU) and,
    // where it is an LM CU, the CTU row's history (CU::saveModelsInHCCP) that read it
    if( ch == 1 && pu.curCand.type != CCP_TYPE_NONE )
    {
      bool          newRecord = false;
      const int32_t r         = ccpRecord( m, pu, &newRecord );
      m.cpuCcp.push_back( { r, &pu.curCand, pu.idxNonLocalCCP != 0 } );
      if( newRecord && PU::isLMCMode( mode ) )
      {
        cpuCcpHistory( m, cu, r );
      }
    }
#endif
    const auto            rec  = m.cuIndex[ch].find( &cu );
    if( rec == m.cuIndex[ch].end() )
    {
      m.cuIndex[ch][&cu] = knownCuRecord( m, cu.blocks[first], mode );
      continue;
    }
    Cu& c = m.cus[rec->second];
    c.kind = CU_KNOWN;
    c.node = c.dimd = c.obic = c.merge = c.mtss = c.timd = c.tmrl = -1;
    m.cuMode[rec->second] = mode;
  }
}

// partial verification: the CTU row's history of CCP models takes record r (CU::saveModelsInHCCP at CU cu, reset at
// each CTU row's start; a single tree's luma ISP CU adds none)
static void cpuCcpHistory( GpuIntraRec::Impl& m, const CodingUnit& cu, int32_t r )
{
  const int row = ( cu.blocks[COMPONENT_Cb].y << getComponentScaleY( COMPONENT_Cb, cu.chromaFormat ) )
                  / (int) cu.cs->sps->getMaxCUHeight();
  if( row != m.ccpHistRow )
  {
    m.ccpHistRow      = row;
    m.ccpHistRowStart = m.ccpHist.size();
  }
  if( CS::isDualITree( *cu.cs ) || !cu.ispMode )
  {
    m.ccpHist.push_back( r );
  }
}

void GpuIntraRec::cpuCuModels( const CodingUnit& cu )
{
#if JVET_AD0188_CCP_MERGE && JVET_AF0073_INTER_CCP_MERGE
  Impl& m = *m_impl;
  if( !m.partial || cu.chromaFormat == CHROMA_400 || !cu.blocks[COMPONENT_Cb].valid() || CU::isIntra( cu ) )
  {
    return;
  }
  for( const TransformUnit& tu: CU::traverseTUs( cu ) )
  {
    if( tu.curCand.type != CCP_TYPE_NONE )
    {
      const auto r = m.ccpTuIndex.emplace( &tu, m.numCcp );
      if( r.second )   // (a record of no PU: verification compares the GPU's records only)
      {
        m.numCcp++;
        m.ccpPu.push_back( nullptr );
        m.ccpWriters.insert( m.ccpWriters.end(), CCP_WRITERS, -1 );
      }
      m.cpuCcp.push_back( { r.first->second, &tu.curCand, false } );
      cpuCcpHistory( m, cu, r.first->second );
    }
  }
#endif
}

void GpuIntraRec::cpuCu( const CodingUnit& cu, const char* why )
{
  Impl& m = *m_impl;
  if( !m.partial )
  {
    unsupported( why );
    return;
  }
  cpuFillCu( m, cu, why );
  for( int c = 0; c < ( cu.chromaFormat == CHROMA_400 ? 1 : 3 ); c++ )   // (its blocks are never recorded)
  {
    if( cu.blocks[c].valid() )
    {
      cpuFillBlock( m, ComponentID( c ), cu.blocks[c] );
    }
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

// JVET-AK0217's MTSS: the LFNST / NSPT set of the block's luma comes from a DIMD mode list of its own
static bool usesMtss( const TransformUnit& tu )
{
#if JVET_AK0217_INTRA_MTSS
  return tu.mdirIdx[COMPONENT_Y] != 0;
#else
  return false;
#endif
}

// a CCCM CU of another kind than intra_cccm.glsl's (the plain models, GL-CCCM's, those of several downsampling
// filters and without subsampling): BVG-CCCM (JVET-AE0100); its kind, else nullptr
static const char* otherCccm( const PredictionUnit& pu )
{
#if JVET_AE0100_BVGCCCM
  if( pu.bvgCccmFlag )
  {
    return "BVG-CCCM";
  }
#endif
  return nullptr;
}

// the samples of intra_cccm.glsl's luma buffer for a CCCM block's largest reference area: up to 6 lines above / left,
// arms of twice the block, a sample of padding all around; without subsampling (noSub) the luma itself, twice as many
// per direction
static int cccmLumaArea( const CompArea& area, bool noSub )
{
  const int s = noSub ? 1 : 0;
  return ( ( std::min<int>( area.x, CCCM_WINDOW_SIZE ) + 2 * area.width + 2 ) << s )
         * ( ( std::min<int>( area.y, CCCM_WINDOW_SIZE ) + 2 * area.height + 2 ) << s );
}

#if GPU_DIMD
// one histogram call of a CU, its pReco relative to origin (the top-left sample of the area of its component), placed
// in its plane
static DimdCall placedDimdCall( const IntraPrediction::DimdHistogramCall& k, const Position origin )
{
  CHECK( k.maxAmp > INT32_MAX, "DIMD's amplitude limit beyond 32 bits" );
  DimdCall d{};
  d.kind          = k.kind;
  d.histogram     = k.histogram;
  d.x             = origin.x + k.x;
  d.y             = origin.y + k.y;
  d.height        = k.height;
  d.width         = k.width;
  d.direction     = k.direction;
  d.filterSizeIdx = k.filterSizeIdx;
  std::copy_n( k.sizeExt, 3, d.sizeExt );
  std::copy_n( k.extra, 3, d.extra );
  d.maxAmp = (int32_t) k.maxAmp;
  d.comp   = k.comp;
  return d;
}
#endif

// the histograms of a CU's DIMD, added to the picture's (Cu::dimd, dimdNum): of a luma CU (IntraPrediction::
// dimdHistogramCalls) or of DIMD chroma (dimdChromaHistogramCalls), placed in their planes
static void addDimdCalls( GpuIntraRec::Impl& m, const CodingUnit& cu, const ChannelType chType, Cu& c )
{
#if GPU_DIMD
  IntraPrediction::DimdHistogramCall calls[8];
  int                                num = 3;
  if( isLuma( chType ) )
  {
    IntraPrediction::dimdHistogramCalls( cu, cu.Y(), calls );
  }
  else
  {
    num = IntraPrediction::dimdChromaHistogramCalls( cu, cu.blocks[COMPONENT_Cb], calls );
  }
  c.dimd    = (int32_t) m.dimdCalls.size();
  c.dimdNum = num;
  for( int n = 0; n < num; n++ )
  {
    // the area of its component: the CU's, or for the luma of DIMD chroma the co-located luma area
    const ComponentID comp   = ComponentID( calls[n].comp );
    const Position    origin = isLuma( chType ) || comp != COMPONENT_Y ? cu.blocks[comp].pos() : cu.blocks[COMPONENT_Cb].lumaPos();
    m.dimdCalls.push_back( placedDimdCall( calls[n], origin ) );
  }
#endif
}

// MTSS (JVET-AK0217: mdirIdx of a DIMD CU with LFNST) chooses the LFNST / NSPT mode from the CU's DIMD and a mode list
// of its own (deriveDimdModeList, which DecCu derives for such a CU): that list's histograms (dimdHistogramCalls with
// modeList), added to the picture's (Cu::mtss, MTSS_CALLS of them), and whether the luma block may use NSPT
static void addMtssCalls( GpuIntraRec::Impl& m, const CodingUnit& cu, Cu& c )
{
#if GPU_DIMD_TRANSFORMS
  IntraPrediction::DimdHistogramCall calls[MTSS_CALLS];
  IntraPrediction::dimdHistogramCalls( cu, cu.Y(), calls, true );
  c.mtss = (int32_t) m.dimdCalls.size();
  for( int n = 0; n < MTSS_CALLS; n++ )
  {
    m.dimdCalls.push_back( placedDimdCall( calls[n], cu.Y().pos() ) );
  }
  const TransformUnit& tu   = *cu.firstTU;
  const CompArea&      area = tu.blocksResidual[COMPONENT_Y];
  c.mtssNspt = CU::isNSPTAllowed( tu, COMPONENT_Y, area.width, area.height, CU::isIntra( cu ) ) ? 1 : 0;
#endif
}

// the neighbours of an OBIC CU whose modes its histogram counts (IntraPrediction::obicNeighbours), added to the
// picture's (Cu::obic): their records and the samples each counts. MIP CUs count nothing in an intra slice (deriveObicMode
// takes their DIMD mode in inter slices only); the GPU tells the others apart by their records' kind. False with why
// set where one cannot be read
static bool addObicNeighbours( GpuIntraRec::Impl& m, const CodingUnit& cu, Cu& c, const char*& why )
{
#if GPU_DIMD
  const CodingUnit* nbrs[NUM_OBIC_CUS];
  int               numSamples[NUM_OBIC_CUS];
  const int         num = IntraPrediction::obicNeighbours( cu, nbrs, numSamples );
  if( num < 0 )
  {
    why = "an inter OBIC neighbour";
    return false;
  }
  CHECK( num > OBIC_NBRS, "OBIC counts at most 20 neighbours" );
  c.obic = (int32_t) ( m.obicNbrs.size() / OBIC_NBRS );
  ObicNbr* o = &*m.obicNbrs.insert( m.obicNbrs.end(), OBIC_NBRS, ObicNbr{ -1, 0 } );
  for( int n = 0; n < num; n++ )
  {
    if( nbrs[n]->mipFlag )
    {
      continue;
    }
    // partial verification: a neighbour taken from the CPU counts its mode once (a known CU's), which deriveObicMode
    // does only for a CU of no tool it counts otherwise (TIMD's, DIMD's, SGPM's, NN's modes; nothing of IntraTMP, EIP)
    const CodingUnit& nb = *nbrs[n];
    if( m.partial && m.cpuCus.count( &nb )
        && ( nb.dimd || nb.timd || nb.sgpm || nb.tmrlFlag || nb.tmpFlag || nb.eipFlag || !CU::isIntra( nb )
             || nb.firstPU->intraDir[CHANNEL_TYPE_LUMA] >= NUM_LUMA_MODE ) )
    {
      why = "OBIC with a neighbour from the CPU that counts otherwise";
      return false;
    }
    const auto rec = m.cuIndex[CHANNEL_TYPE_LUMA].find( nbrs[n] );
    if( rec == m.cuIndex[CHANNEL_TYPE_LUMA].end() )
    {
      why = "an OBIC neighbour that is not a recorded CU";
      return false;
    }
    *o++ = ObicNbr{ rec->second, numSamples[n] };
  }
  return true;
#else
  why = "OBIC";
  return false;
#endif
}

#if GPU_DIMD && JVET_AJ0146_TIMDSAD && JVET_AG0146_DIMD_ITMP_IBC && JVET_AM0074_INTRA_MERGE   // IntraPrediction's helpers
// the template of block area of CU cu with arms tw, th (0: none) as the GPU reads it (TemplateUnits), the availability
// of the units of its block's reference line appended to bits (32 per word); the line's arms are 2^refShift x the
// template block's sides
constexpr int TIMD_REF_SHIFT = 3;   // TIMD's luma templates (initTimdIntraPatternLuma)
constexpr int CFLM_REF_SHIFT = 1;   // CFLM's chroma template (xCflmCreateChromaPred)
static TemplateUnits templateUnits( const CodingUnit& cu, const CompArea& area, const int tw, const int th,
                                    const int refShift, std::vector<int32_t>& bits )
{
  IntraPrediction::TimdReferenceUnits u{};
  IntraPrediction::timdReferenceUnits( cu, area, tw, th, ( area.width + tw ) << refShift,
                                       ( area.height + th ) << refShift, u );
  TemplateUnits t{};
  t.tw               = tw;
  t.th               = th;
  t.unitW            = u.unitWidth;
  t.unitH            = u.unitHeight;
  t.leftTempUnitNum  = u.leftTempUnitNum;
  t.aboveTempUnitNum = u.aboveTempUnitNum;
  t.totalLeftUnits   = u.totalLeftUnits;
  t.totalAboveUnits  = u.totalAboveUnits;
  t.totalUnits       = u.totalUnits;
  t.numIntraNeighbor = u.numIntraNeighbor;
  for( int w = 0; w < ( t.totalUnits + 31 ) / 32; w++ )
  {
    uint32_t word = 0;
    for( int b = 0; b < 32 && w * 32 + b < t.totalUnits; b++ )
    {
      word |= u.flags[w * 32 + b] ? 1u << b : 0u;
    }
    bits.push_back( (int32_t) word );
  }
  return t;
}
#endif

// an intra merge CU's record (Cu::merge, MergeHead): deriveIntraMergeCand's neighbours in its order with how each may
// be a candidate (CU::isAValidNeighborForIntraMerge: an intra merge or an OBIC CU always, a DIMD CU or an MPM CU where
// its DIMD blends, a TIMD or TIMD merge CU where its modes blend or its mode is angular, which the GPU derives; the
// others never, left out), the DIMD of MPM CUs that have it aside put to use; reorderIntraMergeCand's template and the
// availability of its reference units; the CPU's template costs. False with why set where one cannot be read
static bool addMergeRecord( GpuIntraRec::Impl& m, const CodingUnit& cu, Cu& c, const char*& why )
{
#if GPU_DIMD && JVET_AJ0146_TIMDSAD && JVET_AG0146_DIMD_ITMP_IBC && JVET_AM0074_INTRA_MERGE   // IntraPrediction's helpers
  // the CPU's SATD (xGetHADs_SIMD) adds in 16-bit lanes, exactly only up to 10 bits
  if( cu.cs->sps->getBitDepth( CHANNEL_TYPE_LUMA ) > 10 )
  {
    why = "intra merge above 10 bits";
    return false;
  }
  // the node's shared arrays hold the neighbour list and the reference units of a CU up to 128 x 128 (intra_merge.glsl's
  // MT_ENTRIES, intra_timd.glsl's s_timdUnits)
  if( cu.lwidth() > 128 || cu.lheight() > 128 )
  {
    why = "intra merge CU above 128";
    return false;
  }
  std::vector<const CodingUnit*> nbrs;
  IntraPrediction::intraMergeNeighbours( cu, nbrs );
  std::vector<int32_t> pairs;
  for( const CodingUnit* n: nbrs )
  {
    if( n->bvgDimdFlag )
    {
      why = "a BVG-DIMD neighbour of intra merge";
      return false;
    }
    // CU::isAValidNeighborForIntraMerge's "MPM" CU
    const bool mpm = !n->timd && !n->bdpcmMode && !n->tmpFlag && !n->mipFlag && !n->eipFlag && !n->mdip && !n->plIdx && n->firstPU->intraDir[0] != PNN_IDX;
    const int  cls = CU::isIBC( *n ) ? 0 : n->intraMergeMode ? MERGE_NBR_MERGE : n->obicFlag ? MERGE_NBR_OBIC
                     : n->timd ? ( n->timdMrg ? MERGE_NBR_TIMD_MERGE : MERGE_NBR_TIMD ) : n->dimd || mpm ? MERGE_NBR_DIMD : 0;
    if( !cls )
    {
      continue;
    }
    const auto rec = m.cuIndex[CHANNEL_TYPE_LUMA].find( n );
    if( rec == m.cuIndex[CHANNEL_TYPE_LUMA].end() )
    {
      why = "an intra merge neighbour that is not a recorded CU";
      return false;
    }
    Cu& nc = m.cus[rec->second];
    const int kind = cls == MERGE_NBR_MERGE ? CU_MERGE : cls == MERGE_NBR_OBIC ? CU_OBIC
                     : cls == MERGE_NBR_TIMD || cls == MERGE_NBR_TIMD_MERGE ? CU_TIMD : n->dimd ? CU_DIMD
                     : n->tmrlFlag ? CU_TMRL : CU_MPM;
    // (a TIMD CU's DIMD: where its TIMD does not convert; a TIMD merge CU, which has none, is then no candidate)
    if( nc.kind != kind || ( cls == MERGE_NBR_TIMD && nc.dimd < 0 ) )
    {
      why = "an intra merge neighbour of a kind the GPU lacks";
      return false;
    }
    if( kind == CU_MPM && nc.dimd < 0 )   // an MPM CU whose list starts with planar: its DIMD, kept aside
    {
      nc.dimd    = m.pendingDimd[rec->second].first;
      nc.dimdNum = m.pendingDimd[rec->second].second;
      if( nc.dimd < 0 )
      {
        why = "an intra merge neighbour without DIMD";
        return false;
      }
    }
    pairs.push_back( rec->second );
    pairs.push_back( cls );
  }
  MergeHead h{};
  h.index  = m.numMerge++;
  h.nbrNum = (int32_t) pairs.size() / 2;
  std::vector<int32_t> bits;
  if( cu.lx() > 0 && cu.ly() > 0 )   // CU::deriveTimdRefType's LEFT_ABOVE_NEIGHBOR (else candidate 0, no costs)
  {
    h.t = templateUnits( cu, cu.Y(), cu.lwidth() <= 8 ? 2 : 4, cu.lheight() <= 8 ? 2 : 4, TIMD_REF_SHIFT, bits );
  }
  c.merge = (int32_t) m.mergeData.size();
  const int32_t* hp = reinterpret_cast<const int32_t*>( &h );
  m.mergeData.insert( m.mergeData.end(), hp, hp + sizeof( h ) / 4 );
  m.mergeData.insert( m.mergeData.end(), pairs.begin(), pairs.end() );
  m.mergeData.insert( m.mergeData.end(), bits.begin(), bits.end() );
  // the CPU's template costs (reorderIntraMergeCand ranked this CU last)
  const bool valid = m.intraPred && m.intraPred->m_mergeTmCostValid;
  m.cpuMergeCostValid.push_back( valid ? 1 : 0 );
  for( int i = 0; i < MERGE_COST; i++ )
  {
    m.cpuMergeCost.push_back( valid ? m.intraPred->m_mergeTmCost[i] : MAX_UINT64 );
  }
  return true;
#else
  why = "intra merge";
  return false;
#endif
}

// the CUs at getSpatialIpm's positions around luma CU cu (getPURestricted's; Cu::nbr), whose modes its MPM list reads:
// the adjacent ones, then per distance d the non-adjacent ones (idxMap: 0 below-left, 1 above-right, 2 above the
// middle, 3 left of the middle, 4 above-left). False with why set where one cannot be read
static bool addMpmNeighbours( GpuIntraRec::Impl& m, const CodingUnit& cu, Cu& c, const char*& why )
{
  const CompArea&       a  = cu.blocks[COMPONENT_Y];
  const PredictionUnit& pu = *cu.firstPU;
  const int             w = a.width, h = a.height;
  Position              pos[MPM_NEIGHBOURS];
  int                   n = 0;
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
      return false;
    }
    c.nbr[k] = rec->second;
  }
  return true;
}

// the CUs whose modes chroma TMRL PU pu's list reads (getChromaTmrlSearchRange; Cu::nbr from CHROMA_TMRL_NBR_LUMA): the
// luma CUs at the centre and the four corners of the co-located luma area (clipped to the picture), then the chroma CUs
// at its five neighbour positions where the list takes their final mode - intra, not LM nor DBV (DM and DIMD chroma in
// an intra slice), else -1. False with why set where one cannot be read
static bool addChromaTmrlNeighbours( GpuIntraRec::Impl& m, const PredictionUnit& pu, Cu& c, const char*& why )
{
#if GPU_CHROMA_TMRL
  const CompArea& cb = pu.Cb();
  CompArea lumaArea( COMPONENT_Y, pu.chromaFormat, cb.lumaPos(), recalcSize( pu.chromaFormat, CHANNEL_TYPE_CHROMA, CHANNEL_TYPE_LUMA, cb.size() ) );
  lumaArea = clipArea( lumaArea, pu.cs->picture->block( COMPONENT_Y ) );
  const Position lumaPos[5] = { lumaArea.center(), lumaArea.topLeft(), lumaArea.topRight(), lumaArea.bottomLeft(), lumaArea.bottomRight() };
  for( int k = 0; k < 5; k++ )
  {
    const PredictionUnit* lumaPU = pu.cs->getLumaPU( lumaPos[k], CHANNEL_TYPE_LUMA );
    const auto rec = lumaPU ? m.cuIndex[CHANNEL_TYPE_LUMA].find( lumaPU->cu ) : m.cuIndex[CHANNEL_TYPE_LUMA].end();
    if( rec == m.cuIndex[CHANNEL_TYPE_LUMA].end() )
    {
      why = "a chroma TMRL luma CU that is not recorded";
      return false;
    }
    c.nbr[CHROMA_TMRL_NBR_LUMA + k] = rec->second;
  }
  const Position chromaPos[5] = { pu.chromaPos().offset( -1, cb.height - 1 ), pu.chromaPos().offset( cb.width - 1, -1 ),
                                  pu.chromaPos().offset( -1, cb.height ), pu.chromaPos().offset( cb.width, -1 ),
                                  pu.chromaPos().offset( -1, -1 ) };
  for( int k = 0; k < 5; k++ )
  {
    const PredictionUnit* nb = PU::getPUFromPos( pu, CHANNEL_TYPE_CHROMA, chromaPos[k] );
    if( !nb || !CU::isIntra( *nb->cu ) || PU::isLMCMode( nb->intraDir[1] ) || PU::isDbvMode( nb->intraDir[1] )
        || ( ( nb->intraDir[1] == DM_CHROMA_IDX || nb->intraDir[1] == DIMD_CHROMA_IDX ) && !nb->cu->slice->isIntra() ) )
    {
      continue;
    }
    const auto rec = m.cuIndex[CHANNEL_TYPE_CHROMA].find( nb->cu );
    if( rec == m.cuIndex[CHANNEL_TYPE_CHROMA].end() )
    {
      why = "a chroma TMRL neighbour that is not a recorded CU";
      return false;
    }
    c.nbr[CHROMA_TMRL_NBR_CHROMA + k] = rec->second;
  }
  return true;
#else
  why = "chroma TMRL";
  return false;
#endif
}

#if GPU_TIMD
// one derivation's CodingUnit fields in TimdOut's layout (TIMD_OUT_*), for the comparison with the GPU's (the weights
// int8_t, a TIMD merge CU's int)
template<typename Weight>
static void cpuTimdFields( int32_t* fields, int mode, int secondary, int nonAng, bool blended, const Weight* weights,
                           const int8_t* locDep, bool checkWA, bool secondaryCheckWA )
{
  fields[TIMD_OUT_MODE]      = mode;
  fields[TIMD_OUT_SECONDARY] = secondary;
  fields[TIMD_OUT_NON_ANG]   = nonAng;
  fields[TIMD_OUT_BLENDED]   = blended ? 1 : 0;
  std::copy_n( weights, TIMD_FUSION_NUM, fields + TIMD_OUT_WEIGHT );
  std::copy_n( locDep, TIMD_FUSION_NUM, fields + TIMD_OUT_LOC_DEP );
  fields[TIMD_OUT_CHECK_WA]     = checkWA ? 1 : 0;
  fields[TIMD_OUT_SEC_CHECK_WA] = secondaryCheckWA ? 1 : 0;
}

// deriveTimdModeSad's template arms of CU cu (columns left, rows above): TIMD's doubled, at most the CU's distance from
// the picture's edge (2 at least); an arm of 8 only where the samples 8 out beside both ends of the CU's side are
// decoded, of CUs of its slice and tile (CodingStructure::isDecomp, getCURestricted), else 4
static std::pair<int, int> timdSadTemplateArms( const CodingUnit& cu )
{
  const CodingStructure& cs = *cu.cs;
  const int              W = cu.lwidth(), H = cu.lheight();
  const Position         pos = cu.lumaPos();
  auto decoded = [&]( const Position& p ) { return cs.isDecomp( p, CHANNEL_TYPE_LUMA ) && cs.getCURestricted( p, cu, CHANNEL_TYPE_LUMA ); };
  int left  = std::min( std::max( cu.lx(), 2 ), 2 * ( W <= 8 ? 2 : 4 ) );
  int above = std::min( std::max( cu.ly(), 2 ), 2 * ( H <= 8 ? 2 : 4 ) );
  if( above > 4 && !( decoded( pos.offset( 0, -above ) ) && decoded( pos.offset( W - 1, -above ) ) ) )
  {
    above = 4;
  }
  if( left > 4 && !( decoded( pos.offset( -left, 0 ) ) && decoded( pos.offset( -left, H - 1 ) ) ) )
  {
    left = 4;
  }
  return { left, above };
}
#endif

// a TIMD CU's record (Cu::timd, TimdHead): deriveTimdMode's template (CU::deriveTimdRefType: both arms, at the
// picture's left / top edge the above / left one only, at its top-left corner none) and the availability of the units
// of its block's reference line, the PUs whose presence lets its wide angles beyond the regular ones reach further
// (getPURestricted), a TIMD-SAD CU's deriveTimdModeSad template alike; what its luma block's transform types depend
// on; a TIMD merge CU's neighbours (searchTimdMrgCus); the CPU's TIMD (and TIMD-SAD, a TIMD merge CU's candidate),
// compared in verification. False with why set where it cannot be read
static bool addTimdRecord( GpuIntraRec::Impl& m, const CodingUnit& cu, Cu& c, const char*& why )
{
#if GPU_TIMD
  // the CPU's SATD (xGetHADs_SIMD) adds in 16-bit lanes, exactly only up to 10 bits
  if( cu.cs->sps->getBitDepth( CHANNEL_TYPE_LUMA ) > 10 )
  {
    why = "TIMD above 10 bits";
    return false;
  }
  if( !m.trQuant )
  {
    why = "TIMD without the decoder's TrQuant";
    return false;
  }
#if JVET_AM0138_ENHANCED_TMP_MERGE_LIST_TIMD_BV
  // TIMD's candidates from block vectors (of IntraTMP / IBC CUs and the CUs that took one): the GPU has none
  for( int k = 0; k < TIMD_FUSION_NUM; k++ )
  {
    if( cu.isBvTimd[k] || cu.isBvTimdSad[k] )
    {
      why = "TIMD with a block vector";
      return false;
    }
  }
#endif
  const int W = cu.lwidth(), H = cu.lheight();
  TimdHead  head{};
  head.index = m.numTimd++;
  head.type  = cu.lx() > 0 && cu.ly() > 0 ? TIMD_BOTH : cu.lx() > 0 ? TIMD_LEFT : cu.ly() > 0 ? TIMD_ABOVE : TIMD_NONE;
  std::vector<int32_t> unitBits, unitBitsSad;
  if( head.type != TIMD_NONE )   // (calcTimdMrgCandCosts' template of a TIMD merge CU has arms one sample thick)
  {
    const int left = cu.timdMrg ? 1 : W <= 8 ? 2 : 4, above = cu.timdMrg ? 1 : H <= 8 ? 2 : 4;
    head.t = templateUnits( cu, cu.Y(), head.type != TIMD_ABOVE ? left : 0, head.type != TIMD_LEFT ? above : 0,
                            TIMD_REF_SHIFT, unitBits );
  }
  const CodingStructure& cs    = *cu.cs;
  const PredictionUnit&  pu    = *cu.firstPU;
  const Position         posRT = pu.Y().topRight(), posLB = pu.Y().bottomLeft();
  auto present = [&]( const Position& pos ) { return cs.getPURestricted( pos, pu, CHANNEL_TYPE_LUMA ) != nullptr; };
  head.waip = ( present( posRT.offset( 0, -1 ) ) ? TIMD_WAIP_MAX_CLOSE : 0 ) | ( present( posRT.offset( W, -1 ) ) ? TIMD_WAIP_MAX_FAR : 0 )
              | ( present( posLB.offset( -1, 0 ) ) ? TIMD_WAIP_MIN_CLOSE : 0 ) | ( present( posLB.offset( -1, H ) ) ? TIMD_WAIP_MIN_FAR : 0 );
  head.sad = cu.timdSad ? 1 : 0;
  m.numTimdSad += head.sad;
  if( cu.timdSad && head.type != TIMD_NONE )
  {
    const auto arms = timdSadTemplateArms( cu );
    head.ts = templateUnits( cu, cu.Y(), head.type != TIMD_ABOVE ? arms.first : 0,
                             head.type != TIMD_LEFT ? arms.second : 0, TIMD_REF_SHIFT, unitBitsSad );
  }
  // (intra slices keep TIMD-SAD CUs at 1024 samples: at most ~300 units with the arms of 8)
  if( head.t.totalUnits > TIMD_UNITS_MAX || head.ts.totalUnits > TIMD_UNITS_MAX )
  {
    why = "a TIMD template of more reference units than the GPU holds";
    return false;
  }
  // the transform types of its luma block (getTrTypes after DecCu's xReconIntraQT: timdmTrType, which a TIMD merge CU
  // may take; with LFNST the initial DCT-II): explicit MTS's of MTSExt by the mode the node derives, else the types
  // getTrTypes gives without the mode (the SPS has no implicit MTS: supported() rejects it)
  const TransformUnit& tu  = *cu.firstTU;
  const SPS&           sps = *cu.cs->sps;
  head.mtsTrIdx = -1;
  head.trHor = head.trVer = DCT2;
  if( !cu.lfnstIdx && sps.getUseMTS() && sps.getUseIntraMTS() && sps.getUseMTSExt() && tu.mtsIdx[COMPONENT_Y] > MTS_SKIP )
  {
    head.mtsTrIdx = tu.mtsIdx[COMPONENT_Y] - MTS_DST7_DST7;
  }
  else if( !cu.lfnstIdx )
  {
    m.trQuant->getTrTypes( tu, COMPONENT_Y, head.trHor, head.trVer );
  }
  // a TIMD merge CU: the neighbours genTimdMrgList takes the candidates from (TIMD CUs of the picture, recorded before
  // it), which give their transform types in dual tree only
  if( cu.timdMrg )
  {
    static_vector<const CodingUnit*, NUM_TIMD_MERGE_CUS> nbrs;
    IntraPrediction::searchTimdMrgCus( cu, nbrs );
    CHECK( nbrs.size() > TIMD_MERGE_CANDS, "searchTimdMrgCus keeps at most five neighbours" );
    for( const CodingUnit* n: nbrs )
    {
      const auto rec = m.cuIndex[CHANNEL_TYPE_LUMA].find( n );
      if( rec == m.cuIndex[CHANNEL_TYPE_LUMA].end() || m.cus[rec->second].kind != CU_TIMD )
      {
        why = "a TIMD merge neighbour that is not a recorded TIMD CU";
        return false;
      }
      head.mergeNbr[head.merge++] = rec->second;
    }
    if( head.merge == 0 )   // (the CPU cannot decode such a CU either)
    {
      why = "TIMD merge without a neighbour";
      return false;
    }
    head.inheritTr = CS::isDualITree( *cu.cs ) ? 1 : 0;
    m.numTimdMerge++;
  }
  c.timd = (int32_t) m.timdData.size();
  const int32_t* headWords = reinterpret_cast<const int32_t*>( &head );
  m.timdData.insert( m.timdData.end(), headWords, headWords + sizeof( head ) / 4 );
  m.timdData.insert( m.timdData.end(), unitBits.begin(), unitBits.end() );
  m.timdData.insert( m.timdData.end(), unitBitsSad.begin(), unitBitsSad.end() );
  // the CPU's fields (a TIMD merge CU's candidate) and transform types (not in ON mode: its modes are of a picture it
  // has not reconstructed)
  int32_t* cpuFields = &*m.cpuTimd.insert( m.cpuTimd.end(), TIMD_OUT, 0 );
  if( cu.timdMrg )
  {
    cpuTimdFields( cpuFields, cu.timdMrgList[0][0], cu.timdMrgList[0][1], cu.timdMrgList[0][2], cu.timdMrgIsBlended[0],
                   cu.timdMrgFusionWeight[0], cu.timdMrgLocDep[0], cu.timdMrgModeCheckWA[0][0], cu.timdMrgModeCheckWA[0][1] );
    cpuFields[TIMD_OUT_TR_HOR] = cu.timdmTrType[0][0];
    cpuFields[TIMD_OUT_TR_VER] = cu.timdmTrType[0][1];
    return true;
  }
  cpuTimdFields( cpuFields, cu.timdMode, cu.timdModeSecondary, cu.timdModeNonAng, cu.timdIsBlended, cu.timdFusionWeight,
                 cu.timdLocDep, cu.timdModeCheckWA, cu.timdModeSecondaryCheckWA );
  if( cu.timdSad )
  {
    cpuTimdFields( cpuFields + TIMD_OUT_SAD, cu.timdModeSad, cu.timdModeSecondarySad, cu.timdModeNonAngSad, cu.timdIsBlendedSad,
                   cu.timdFusionWeightSad, cu.timdLocDepSad, cu.timdModeCheckWASad, cu.timdModeSecondaryCheckWASad );
  }
  if( !m.on )
  {
    int trHor = DCT2, trVer = DCT2;
    if( !cu.lfnstIdx )
    {
      m.trQuant->getTrTypes( tu, COMPONENT_Y, trHor, trVer );
    }
    cpuFields[TIMD_OUT_TR_HOR] = trHor;
    cpuFields[TIMD_OUT_TR_VER] = trVer;
  }
  return true;
#else
  why = "TIMD";
  return false;
#endif
}

// the record of the CU of a block of the channel (made at the CU's first block): how its mode is found and the CUs that
// reads. With gpuDerivesModes() or DIMD the GPU derives the luma modes the MPM lists give (from the CUs at
// getSpatialIpm's positions, getPURestricted's, and with DIMD the CU's DIMD modes) and the chroma modes DM or the
// candidates' substitution give from those, in a node recorded before the CU's first block (intra_rec), which also
// derives a DIMD or an OBIC CU's modes; the host keeps the others, for luma the mode the neighbours' MPM lists take
// (PU::getIntraDirLuma: planar for MIP). -1 with why set where a neighbour cannot be read or a DIMD CU is of a kind
// the GPU lacks.
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
  c.kind     = CU_KNOWN;
  c.node     = -1;
  c.lumaCu   = -1;
  c.dimd     = -1;
  c.dimdNum  = 0;
  c.obic     = -1;
  c.merge    = -1;
  c.mtss     = -1;
  c.mtssNspt = 0;
  c.timd     = -1;
  c.tmrl     = -1;
  std::fill_n( c.nbr, MPM_NEIGHBOURS, -1 );
  c.modes[0] = c.modes[1] = -1;
  const CompArea& a = cu.blocks[chType == CHANNEL_TYPE_LUMA ? COMPONENT_Y : COMPONENT_Cb];
  c.x = a.x;
  c.y = a.y;
  c.w = a.width;
  c.h = a.height;
  int32_t mode = -1;
  std::pair<int32_t, int32_t> pending( -1, 0 );   // DIMD histograms kept aside (pendingDimd)
  if( isLuma( chType ) )
  {
    // DIMD (DecCu): the mode from the reconstruction around the CU; with OBIC the modes of the CUs around it (OBIC
    // without a mode cannot be decoded: the CPU would predict with mode -1)
    // with intra merge the parameters of a neighbour, ranked by template costs
    if( m.deriveModes && cu.dimd )
    {
      why = cu.bvgDimdFlag ? "BVG-DIMD" : cu.isBvDimd ? "DIMD with a block vector"
            : cu.intraMergeMode && cu.isBvDimdExtAvail ? "intra merge with a block vector"
            : cu.obicFlag && cu.obicMode[0] < 0 ? "OBIC without a mode" : nullptr;
      if( why || ( cu.obicFlag && !addObicNeighbours( m, cu, c, why ) )
          || ( cu.intraMergeMode && !addMergeRecord( m, cu, c, why ) ) )
      {
        return -1;
      }
      c.kind  = cu.intraMergeMode ? CU_MERGE : cu.obicFlag ? CU_OBIC : CU_DIMD;
      c.plIdx = cu.plIdx;
    }
    // (partial verification: TIMD's and TMRL's lists come from the MPM lists that MPM sorting, MDIP and PDP change,
    // which the GPU lacks: such a CU from the CPU)
    else if( m.partial && ( cu.timd || cu.tmrlFlag )
             && ( cu.cs->sps->getUseMpmSorting() || cu.cs->sps->getUseMdip() || cu.cs->sps->getUsePDP() ) )
    {
      why = "TIMD / TMRL with MPM sorting, MDIP or PDP";
      return -1;
    }
    // TIMD: from template costs of the modes of an MPM list, with its DIMD; TIMD merge: from its TIMD neighbours'
    // ranked by template costs, without DIMD and MPM list
    else if( m.deriveModes && cu.timd )
    {
      if( !addTimdRecord( m, cu, c, why ) || ( !cu.timdMrg && !addMpmNeighbours( m, cu, c, why ) ) )
      {
        return -1;
      }
      c.kind = CU_TIMD;
    }
    // TMRL: from template costs of a mode list's modes (getSpatialIpm's neighbours and its DIMD) on several reference
    // lines, the entry its index selects; the CPU's mode and line (from tmrlList), compared in verification
    else if( m.deriveModes && cu.tmrlFlag )
    {
      // its template costs are TIMD's SATD (xGetHADs_SIMD), which adds in 16-bit lanes, exactly only up to 10 bits
      if( cu.cs->sps->getBitDepth( CHANNEL_TYPE_LUMA ) > 10 )
      {
        why = "TMRL above 10 bits";
        return -1;
      }
      if( !addMpmNeighbours( m, cu, c, why ) )
      {
        return -1;
      }
      c.kind     = CU_TMRL;
      c.ipredIdx = cu.tmrlListIdx;
      c.tmrl     = (int32_t) ( m.cpuTmrl.size() / TMRL_OUT );
      m.cpuTmrl.push_back( pu.intraDir[CHANNEL_TYPE_LUMA] );
      m.cpuTmrl.push_back( pu.multiRefIdx );
    }
    // (partial verification: the CPU's mode where MPM sorting, MDIP or PDP change the MPM list, which the GPU lacks)
    else if( m.deriveModes && pu.parseLumaMode && !cu.mipFlag && !cu.bdpcmMode
             && !( m.partial && ( cu.cs->sps->getUseMpmSorting() || cu.cs->sps->getUseMdip() || cu.cs->sps->getUsePDP() ) ) )
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
      if( !addMpmNeighbours( m, cu, c, why ) )
      {
        return -1;
      }
    }
    else
    {
      mode = PU::getIntraDirLuma( pu );
#if JVET_AJ0249_NEURAL_NETWORK_BASED
      mode = mode == PNN_IDX ? PLANAR_IDX : mode;   // (an NN CU: the mode the MPM lists and chroma TMRL read)
#endif
    }
    // DIMD's histograms (deriveDimdMode): a DIMD CU's mode, the modes OBIC counts less, and the first entries of an
    // MPM list (also a TIMD CU's, not a TIMD merge CU's, and a TMRL CU's mode list); where the parsed syntax selects the
    // list's first entry, planar, only an intra merge CU reads the CU's DIMD (dimdBlending): kept aside until one does
    // (pendingDimd)
    if( cu.cs->sps->getUseDimd()
        && ( c.kind == CU_DIMD || c.kind == CU_OBIC || c.kind == CU_MPM || ( c.kind == CU_TIMD && !cu.timdMrg ) || c.kind == CU_TMRL ) )
    {
      addDimdCalls( m, cu, chType, c );
      if( c.kind == CU_MPM && c.modes[0] == PLANAR_IDX )
      {
        pending = { c.dimd, c.dimdNum };
        c.dimd    = -1;
        c.dimdNum = 0;
      }
    }
    // MTSS's mode list, in the CU's node: of a DIMD, OBIC, intra merge or TIMD CU, and of a MIP CU (CU_KNOWN), whose
    // node derives only that list (its block the rest, from its prediction)
    if( ( c.kind == CU_DIMD || c.kind == CU_OBIC || c.kind == CU_MERGE || c.kind == CU_TIMD || ( GPU_MIP_DIMD && cu.mipFlag ) )
        && usesMtss( *cu.firstTU ) )
    {
      addMtssCalls( m, cu, c );
    }
  }
  else
  {
    const auto luma = m.cuIndex[CHANNEL_TYPE_LUMA].find( PU::getCoLocatedLumaPU( pu ).cu );
    c.lumaCu = luma != m.cuIndex[CHANNEL_TYPE_LUMA].end() ? luma->second : -1;
    // (partial verification: the CPU's DM or candidate mode where chroma reordering reorders them, which the GPU lacks)
    const bool lumaDerived = c.lumaCu >= 0 && m.cus[c.lumaCu].kind != CU_KNOWN && !cu.bdpcmModeChroma
                             && !( m.partial && cu.cs->sps->getUseChromaReordering() );
    if( m.deriveModes && pu.chromaTmrlFlag )
    {
      // chroma TMRL (DecCu at the Cb block: deriveDimdChromaMode, then getChromaTmrlList): from its DIMD chroma, the
      // modes of the CUs getChromaTmrlSearchRange reads and costs of each mode on each chroma line, the entry its
      // index selects; the CPU's mode and line, compared in verification
      if( c.lumaCu < 0 || !addChromaTmrlNeighbours( m, pu, c, why ) )
      {
        why = why ? why : "chroma TMRL without its luma CU";
        return -1;
      }
      c.kind     = CU_CHROMA_TMRL;
      c.ipredIdx = pu.chromaTmrlIdx;
      c.tmrl     = (int32_t) ( m.cpuTmrl.size() / TMRL_OUT );
      m.cpuTmrl.push_back( pu.intraDir[CHANNEL_TYPE_CHROMA] );
      m.cpuTmrl.push_back( pu.chromaMrlIdx );
      addDimdCalls( m, cu, chType, c );
    }
    else if( m.deriveModes && ( pu.intraDir[CHANNEL_TYPE_CHROMA] == DIMD_CHROMA_IDX || ccpFusionDimd( pu ) )
             && !cu.bdpcmModeChroma )
    {
      // DIMD chroma (DecCu at the Cb block): from the reconstruction around the CU, then the co-located luma mode;
      // also a CCP merge CU's of fusion with DIMD chroma (DecCu derives it alike, its blocks predict by it)
      if( c.lumaCu < 0 )
      {
        why = "DIMD chroma without its luma CU";
        return -1;
      }
      c.kind = CU_CHROMA_DIMD;
      addDimdCalls( m, cu, chType, c );
    }
    else if( lumaDerived && pu.intraDir[CHANNEL_TYPE_CHROMA] == DM_CHROMA_IDX )
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
  // its node, which derives the mode before the CU's blocks (it reconstructs no samples), or for a MIP CU with MTSS
  // MTSS's mode list
  if( c.kind != CU_KNOWN || c.mtss >= 0 )
  {
    Tu node{};
    node.comp      = chType == CHANNEL_TYPE_LUMA ? COMPONENT_Y : COMPONENT_Cb;
    node.x         = c.x;
    node.y         = c.y;
    node.dbgOffset = (int32_t) m.numSamples;
    node.mipMatrix = -1;
    node.ispPrev = node.spCoded = node.spOffset = node.resiFrom = node.cflm = node.ccp = node.ccpMerge = -1;
    node.pdp = node.pnn = -1;
    node.cu        = index;
    // with what the template predictions of intra merge, TIMD and TMRL need (the bit depth, the SPS's PDPC switch; the
    // availability units of TMRL's reference lines, chroma's as xFillReferenceSamples has them for the CU; the SPS's
    // reference filter switch, which chroma TMRL's luma prediction takes)
    const bool noShift = cu.cs->pcv->noChroma2x2 && cu.blocks[node.comp].width == 4;
    node.derive    = DERIVE_NODE | ( cu.cs->sps->getDisablePdpc() ? DERIVE_NO_PDPC : 0 );
    node.bitDepth  = cu.cs->sps->getBitDepth( chType );
    node.unitW     = cu.cs->pcv->minCUWidth >> ( noShift ? 0 : getComponentScaleX( ComponentID( node.comp ), cu.cs->sps->getChromaFormatIdc() ) );
    node.unitH     = cu.cs->pcv->minCUHeight >> ( noShift ? 0 : getComponentScaleY( ComponentID( node.comp ), cu.cs->sps->getChromaFormatIdc() ) );
    node.disableRefFilter = cu.cs->sps->getDisableRefFilter();
    c.node         = (int32_t) m.rec.size();
    if( m.partial )
    {
      m.cuTus[&cu].push_back( c.node );
    }
    m.rec.push_back( node );
  }
  m.cus.push_back( c );
  m.cuMode.push_back( mode );
  m.pendingDimd.push_back( pending );
  m.cuMipDimd.push_back( 0 );
  m.cuIndex[chType][&cu] = index;
#if GPU_DIMD
  // the CPU's DIMD of the CU, which it derived before the CU's blocks (verification compares the node's with it); an
  // intra merge CU's merge parameters instead (in DimdOut's layout, which its node writes them in)
  int32_t* d = &*m.cpuDimd.insert( m.cpuDimd.end(), DIMD_OUT, 0 );
  if( c.kind == CU_CHROMA_DIMD || c.kind == CU_CHROMA_TMRL )
  {
    d[0] = cu.dimdChromaMode;
    d[1] = cu.dimdChromaModeSecond;
  }
  else if( c.kind == CU_MERGE )
  {
    d[0] = cu.dimdModeMerge;
    d[1] = 1;
    std::copy_n( cu.dimdBlendModeMerge, 6, d + 2 );
    d[8] = -1;
    std::copy_n( cu.dimdRelWeightMerge, 8, d + 9 );
    std::copy_n( cu.dimdLocDepMerge, 7, d + 17 );
  }
  else
  {
    d[0] = cu.dimdMode;
    d[1] = cu.dimdBlending ? 1 : 0;
    std::copy_n( cu.dimdBlendMode, 7, d + 2 );
    std::copy_n( cu.dimdRelWeight, 8, d + 9 );
    std::copy_n( cu.dimdLocDep, 7, d + 17 );
  }
#if GPU_DIMD_TRANSFORMS
  // DIMD's second strongest mode (a TIMD merge CU has no DIMD)
  if( c.kind == CU_DIMD || c.kind == CU_OBIC || c.kind == CU_MPM || ( c.kind == CU_TIMD && !cu.timdMrg ) || c.kind == CU_TMRL )
  {
    d[DIMD_OUT_TR_SECOND] = cu.candModeListForTransform[1];
  }
#if GPU_MIP_DIMD
  if( isLuma( chType ) && cu.mipFlag && !m.on )   // a MIP CU: deriveIpmForTransform's modes of its prediction instead
  {
    d[0]                  = cu.mipDimdMode;
    d[DIMD_OUT_TR_SECOND] = cu.dimdDerivedIntraDir2nd;
  }
#endif
#if JVET_AJ0249_NEURAL_NETWORK_BASED
  // an NN CU: those of its prediction (xPredIntraPnn's indicesRepresentationPnn)
  if( isLuma( chType ) && cu.firstPU->intraDir[CHANNEL_TYPE_LUMA] == PNN_IDX && !m.on )
  {
    d[0]                  = (int32_t) cu.indicesRepresentationPnn[COMPONENT_Y][0];
    d[DIMD_OUT_TR_SECOND] = (int32_t) cu.indicesRepresentationPnn[COMPONENT_Y][1];
  }
#endif
  // MTSS's mode list and the LFNST / NSPT mode and bucket it chose, for verification (in ON mode the CPU derives them
  // from samples it has not reconstructed)
  if( c.mtss >= 0 && !m.on )
  {
    CHECK( cu.candModeListForTransformMtss.size() != MTSS_LIST_SIZE, "MTSS's mode list" );
    std::copy_n( cu.candModeListForTransformMtss.begin(), MTSS_LIST_SIZE, d + DIMD_OUT_MTSS_LIST );
    bool secondBucket = false;
    d[DIMD_OUT_MTSS_MODE]   = (int32_t) PU::getFinalIntraModeForTransform( secondBucket, *cu.firstTU, COMPONENT_Y ).first;
    d[DIMD_OUT_MTSS_BUCKET] = secondBucket ? 1 : 0;
  }
#endif
  if( c.kind == CU_OBIC )   // and its OBIC
  {
    int32_t* o = &*m.cpuObic.insert( m.cpuObic.end(), OBIC_OUT, 0 );
    std::copy_n( cu.obicMode, OBIC_MODES, o );
    std::copy_n( cu.obicFusionWeight, OBIC_MODES, o + OBIC_MODES );
    std::copy_n( cu.obicLocDep, OBIC_MODES, o + 2 * OBIC_MODES );
  }
#endif
  return index;
}

// the CCP model record of PU pu (Tu::ccp; made at its first block, which each component's block writes its part of);
// isNew: made now
static int32_t ccpRecord( GpuIntraRec::Impl& m, const PredictionUnit& pu, bool* isNew = nullptr )
{
  const auto r = m.ccpIndex.emplace( &pu, m.numCcp );
  if( r.second )
  {
    m.numCcp++;
    m.ccpPu.push_back( &pu );
    m.ccpWriters.insert( m.ccpWriters.end(), CCP_WRITERS, -1 );
  }
  if( isNew )
  {
    *isNew = r.second;
  }
  return r.first->second;
}

#if JVET_AD0188_CCP_MERGE
// the entry of CCP merge PU pu in m.ccpMerge (made at its first block, before its own record goes into the history):
// the CTU row's history so far, idxNonLocalCCP - 1, its ordinal (its default records'), its adjustment (JVET-AL0126's
// ccpMergeAdjustFlag), then the spatial sources of PU::getCCPModelCandidateList - the PUs at its 5 adjacent positions
// and at the non-adjacent ones of its two rounds (7 distances each; their intervals and offsets by the PU's chType
// sizes: a single-tree CU's luma ones, the CPU's, added to the chroma position), each where getPUFromPos finds a PU (in
// the picture, decoded before: with one slice and tile, recorded before) with a CCP model (typed: a record), each
// record once (a later one is left out as equal to it)
static int32_t ccpMergeEntry( GpuIntraRec::Impl& m, const PredictionUnit& pu )
{
  std::vector<int32_t>& d = m.ccpMerge;
  const int32_t         e = (int32_t) d.size();
  d.push_back( (int32_t) m.ccpHistRowStart );
  d.push_back( (int32_t)( m.ccpHist.size() - m.ccpHistRowStart ) );
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION
  const bool ddFusion = pu.ddNonLocalCCPFusion > 0;   // (then idxNonLocalCCP is 1: the candidate is the pair's)
  d.push_back( ddFusion ? pu.ddNonLocalCCPFusion - 1 : pu.idxNonLocalCCP - 1 );
#else
  const bool ddFusion = false;
  d.push_back( pu.idxNonLocalCCP - 1 );
#endif
  d.push_back( m.numCcpMerge++ );
#if JVET_AL0126_CCP_MERGE_WITH_ADJUST
  const int adjust = pu.ccpMergeAdjustFlag;
#else
  const int adjust = 0;
#endif
  d.push_back( ddFusion ? CCP_KIND_DD_FUSION : ccpFusionBlend( pu ) ? CCP_KIND_FUSION_BLEND : adjust );
  d.push_back( 0 );
  const CodingStructure& cs  = *pu.cs;
  const CompArea&        cb  = pu.blocks[COMPONENT_Cb];
  const Position         pos = pu.chromaPos();
  const Size             chromaPic( cs.picture->chromaSize() );
  auto addAt = [&]( int dx, int dy )
  {
    const Position p = pos.offset( dx, dy );
    if( p.x < 0 || p.y < 0 || p.x >= (int) chromaPic.width || p.y >= (int) chromaPic.height )
    {
      return;
    }
    const PredictionUnit* q = cs.getPU( p, CHANNEL_TYPE_CHROMA );
    auto add = [&]( int32_t r )
    {
      if( std::find( d.begin() + e + CCP_MERGE_SPATIAL, d.end(), r ) == d.end() )
      {
        d.push_back( r );
      }
    };
    const auto r = q ? m.ccpIndex.find( q ) : m.ccpIndex.end();
    if( r != m.ccpIndex.end() )
    {
      add( r->second );
    }
#if JVET_AF0073_INTER_CCP_MERGE
    else if( q && !CU::isIntra( *q->cu ) && !m.ccpTuIndex.empty() )   // (partial verification: its TUs' models)
    {
      for( const TransformUnit& tu: CU::traverseTUs( *q->cu ) )
      {
        const auto t = m.ccpTuIndex.find( &tu );
        if( t != m.ccpTuIndex.end() )
        {
          add( t->second );
        }
      }
    }
#endif
  };
  const int iW = cb.width, iH = cb.height;
  addAt( -1, iH - 1 );
  addAt( iW - 1, -1 );
  addAt( -1, iH );
  addAt( iW, -1 );
  addAt( -1, -1 );
  const bool luma   = pu.chType == CHANNEL_TYPE_LUMA;
  const int  sizeW  = luma ? pu.Y().width : cb.width, sizeH = luma ? pu.Y().height : cb.height;
  const int  horInt = std::max( sizeW, 4 ), verInt = std::max( sizeH, 4 );   // horNAInterval, verNAInterval
  const int  offsetX2 = sizeW >> 1, offsetY2 = sizeH >> 1;
  for( int dist = 0; dist < 7; dist++ )   // round 1: 5 positions at the first distance, 9 at the others
  {
    const int dH = horInt * ( dist + 1 ), dV = verInt * ( dist + 1 );
    const int offsetX0 = -dH - 1, offsetY0 = verInt + dV - 1, offsetX1 = horInt + dH - 1, offsetY1 = -dV - 1;
    const int xs[9] = { offsetX0, offsetX1, offsetX2, offsetX0, offsetX0, -1, offsetX1, offsetX0 >> 1, offsetX1 };
    const int ys[9] = { offsetY0, offsetY1, offsetY1, offsetY2, offsetY1, offsetY0, -1, offsetY0, offsetY1 >> 1 };
    for( int k = 0; k < ( dist == 0 ? 5 : 9 ); k++ )
    {
      addAt( xs[k], ys[k] );
    }
  }
  for( int dist = 0; dist < 7; dist++ )   // round 2: 4 positions per distance
  {
    const int dH = horInt * ( dist + 1 ), dV = verInt * ( dist + 1 );
    const int offsetX0 = -dH - 1, offsetY0 = -dV - 1;
    addAt( offsetX0, offsetY2 + ( ( verInt + dV - 1 - offsetY2 ) >> 1 ) );
    addAt( offsetX2 + ( ( horInt + dH - 1 - offsetX2 ) >> 1 ), offsetY0 );
    addAt( offsetX0, offsetY0 + ( ( offsetY2 - offsetY0 ) >> 1 ) );
    addAt( offsetX0 + ( ( offsetX2 - offsetX0 ) >> 1 ), offsetY0 );
  }
  d[e + CCP_MERGE_NUM_SPATIAL] = (int32_t) d.size() - e - CCP_MERGE_SPATIAL;
  return e;
}

// the entry of a decoder-derived CCP fusion PU (intra_ccp.glsl derives it as a merge block of this kind): no history,
// no sources, its ordinal (its spare records hold its models)
static int32_t ccpDerivedEntry( GpuIntraRec::Impl& m )
{
  std::vector<int32_t>& d = m.ccpMerge;
  const int32_t         e = (int32_t) d.size();
  d.insert( d.end(), { 0, 0, 0, m.numCcpMerge++, CCP_KIND_DERIVED, 0 } );
  return e;
}
#endif

void GpuIntraRec::record( const TransformUnit& tu, const ComponentID compID, const IntraPrediction::IpaSnapshot& ipaIn,
                          const QpParam& qp, const CPelBuf* pred, const CPelBuf* resi )
{
  Impl& m = *m_impl;
  if( !m.unsupportedWhy.empty() )
  {
    return;
  }
  if( m.partial && m.cpuCus.count( tu.cu ) )   // (a block of a CU taken from the CPU)
  {
    cpuFillBlock( m, compID, tu.blocks[compID] );
    return;
  }
  const CodingUnit&      cu  = *tu.cu;
  const PredictionUnit&  pu  = *cu.firstPU;
  // a block the GPU path cannot reconstruct: with partial verification its CU from the CPU, else the whole picture
  auto reject = [&]( const char* reason )
  {
    if( m.partial )
    {
      cpuFillCu( m, cu, reason );
      cpuFillBlock( m, compID, tu.blocks[compID] );
    }
    else
    {
      unsupported( reason );
    }
  };
  // DecCu::xIntraRecBlk initializes no parameters for a CCCM block (no initIntraPatternChType): the snapshot is the
  // previous block's, a chroma TMRL CU's reference line too. An LM block reads none of them (t.lm below) but its line
  // is the first
  const IntraPrediction::IpaSnapshot ipa = !isLuma( compID ) && pu.cccmFlag ? IntraPrediction::IpaSnapshot{} : ipaIn;
  const CodingStructure& cs  = *tu.cs;
  const SPS&             sps = *cs.sps;
  const Slice&           slice = *cs.slice;
  const ChannelType      chType = toChannelType( compID );
  const CompArea&        area   = tu.blocks[compID];
  // MIP blocks (luma; IntraPrediction::predIntraMip) are recorded with the DC mode: they read the references DC reads
  // (a superset), and their bilateral filter, like DC's, has no angular texture gate; Tu::mipMatrix selects MIP
  const bool             mip     = isLuma( compID ) && cu.mipFlag;
  // NN intra prediction's blocks (luma; IntraPrediction::xPredIntraPnn, intra_nn.glsl) likewise: they read the
  // references DC reads (the upsampling's boundaries) and have the bilateral filter where DC has it
#if JVET_AJ0249_NEURAL_NETWORK_BASED
  const bool             pnn     = isLuma( compID ) && PU::getFinalIntraMode( pu, chType ) == PNN_IDX;
#else
  const bool             pnn     = false;
#endif
  int                    dirMode = mip || pnn ? DC_IDX : PU::getFinalIntraMode( pu, chType );
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION
  // a block of decoder-derived CCP fusion has the parsed LM_CHROMA_IDX (its CCLM model's), the GPU derives the rest; in
  // verification DecCu has left its choice (LM_CHROMA_IDX or MMLM_CHROMA_IDX) in the PU
  dirMode = !isLuma( compID ) && pu.decoderDerivedCcpMode ? LM_CHROMA_IDX : dirMode;
#endif

#if JVET_AH0209_PDP
  // PDP's size of the block (-1: no PDP), and a MIP block's filter set where PDP predicts it (predIntraMip, -1: none)
  const int pdp       = pdpSize( cu, pu, compID, ipa );
  int       pdpMipSet = -1;
#if JVET_AI0208_PDP_MIP
  if( pdp >= 0 && cu.mipFlag && !cu.dimd )
  {
    const int set = pu.intraDir[CHANNEL_TYPE_LUMA] + ( pu.mipTransposedFlag ? 16 : 0 );
    pdpMipSet     = g_pdpFiltersMip[set][pdp] ? set : -1;
  }
#endif
#endif

  // what the GPU path does not do yet
  const char* why = nullptr;
  if( !CU::isIntra( cu ) )                                          why = "not intra";
  else if( !isLuma( compID ) && PU::isMIP( pu, chType ) )           why = "chroma MIP (4:4:4)";
  else if( cu.ispMode && ipa.multiRefIndex )                        why = "ISP with MRL";
  // LFNST / NSPT of MIP blocks select their kernel by a DIMD mode the CPU derives from the MIP prediction
  // (deriveIpmForTransform), which intra_rec derives for a luma block that is the whole CU where the SPS has DIMD
  // (without, the CPU's is planar)
  else if( cu.mipFlag && cu.lfnstIdx
           && ( !GPU_MIP_DIMD || !isLuma( compID ) || !sps.getUseDimd() || area != cu.blocks[COMPONENT_Y] ) ) why = "LFNST of MIP";
  // transform skip with the regular residual coding is dequantized like a transform block (DepQuant::dequant)
  else if( ( tu.mtsIdx[compID] == MTS_SKIP || ( tu.jointCbCr && tu.mtsIdx[COMPONENT_Cr] == MTS_SKIP ) )
           && slice.getTSResidualCodingDisabledFlag() )             why = "transform skip with regular residual coding";
  else if( cu.tmpFlag )                                             why = "IntraTMP";
#if JVET_AC0071_DBV
  else if( !isLuma( compID ) && PU::isDbvMode( dirMode ) )           why = "DBV";
#endif
#if JVET_AH0209_PDP
  // PDP: a PDP size's CU is one block
  else if( pdpMipSet >= 0 && pdp >= PDP_SIZES )                     why = "PDP of MIP beyond 32x32";
  else if( pdp >= 0 && area != cu.blocks[COMPONENT_Y] )             why = "PDP of a CU with several blocks";
#endif
  // NN intra prediction: its CU is one block up to 32 x 32 (its transforms' mode, where they depend on one, a DIMD of
  // its prediction intra_rec derives: DERIVE_PNN_DIMD)
  else if( pnn && m.sh->pnnModel.empty() )                          why = "NN intra prediction without its models";
  else if( pnn && ( area != cu.blocks[COMPONENT_Y] || area.width > 32 || area.height > 32 ) )
    why = "NN intra prediction of a split CU or above 32 x 32";
  else if( pnn && !m.deriveModes && ( cu.lfnstIdx || tu.mtsIdx[COMPONENT_Y] > MTS_SKIP ) )
    why = "NN intra prediction with LFNST / NSPT or MTS without the GPU's mode derivation";
  // a DIMD CU (CU_DIMD, CU_OBIC, CU_MERGE) blends planar predicted from the CU's reference line at the block's size
  // (predIntraAng: initIntraPatternChType with the CU's area), the GPU from the block's: the same only where the CU is
  // one block (the intra configuration's CTU and transform sizes leave no other)
  else if( cu.dimd && isLuma( compID ) && area != cu.blocks[COMPONENT_Y] ) why = "DIMD CU with several transform blocks";
  else if( !GPU_DIMD_TRANSFORMS && cu.dimd && isLuma( compID ) && ( tu.mtsIdx[COMPONENT_Y] > MTS_SKIP || cu.lfnstIdx ) )
    why = "the transforms of a DIMD CU";
  // MTSS of other CUs than DIMD's, MIP's and TIMD's (IntraTMP, SGPM, EIP: rejected themselves)
  else if( isLuma( compID ) && usesMtss( tu ) && ( !( cu.dimd || cu.mipFlag || ( GPU_TIMD && cu.timd ) ) || !GPU_DIMD_TRANSFORMS ) ) why = "MTSS";
  // TIMD (luma; plain TIMD, TIMD-SAD and TIMD merge): its CU is one block, which predicts the CU's modes. TIMD with MRL
  // has no test stream (the stock encoder's TIMD-MRL candidates hardly ever survive its SATD checks)
  // TIMD-SAD's list takes the CU's DIMD modes, without DIMD in the SPS the CU's initial values (no test stream)
  else if( cu.timd && isLuma( compID ) && cu.timdSad && !cu.cs->sps->getUseDimd() ) why = "TIMD-SAD without DIMD";
  else if( cu.timd && isLuma( compID ) && ( cu.ispMode || ipa.multiRefIndex ) ) why = "TIMD with ISP / MRL";
  else if( cu.timd && isLuma( compID ) && area != cu.blocks[COMPONENT_Y] ) why = "TIMD CU with several transform blocks";
  else if( cu.sgpm )                                                why = "SGPM";
  // TMRL (luma): its CU is one block, which predicts the mode and line its node chooses
  else if( cu.tmrlFlag && isLuma( compID ) && area != cu.blocks[COMPONENT_Y] ) why = "TMRL CU with several transform blocks";
  // chroma TMRL (dual tree; the only chroma MRL): its CU is one block per component, which predicts the mode and line its
  // node chooses; the node's costs need its DIMD chroma, predict the co-located luma block with the chroma bit depth and
  // limit the luma availability by the SPS's picture size (the picture's in the kernels), and the SATD of the luma block
  // adds in 16-bit lanes (as TIMD's)
  else if( !isLuma( compID ) && pu.chromaTmrlFlag && !GPU_CHROMA_TMRL ) why = "chroma TMRL of another configuration";
  else if( !isLuma( compID ) && pu.chromaTmrlFlag && area != cu.blocks[compID] ) why = "chroma TMRL CU with several transform blocks";
  else if( !isLuma( compID ) && pu.chromaTmrlFlag && !sps.getUseDimd() ) why = "chroma TMRL without DIMD";
  else if( !isLuma( compID ) && pu.chromaTmrlFlag && ( sps.getBitDepth( CHANNEL_TYPE_LUMA ) > 10
                                                       || sps.getBitDepth( CHANNEL_TYPE_CHROMA ) != sps.getBitDepth( CHANNEL_TYPE_LUMA ) ) )
    why = "chroma TMRL above 10 bits or with unequal bit depths";
  else if( !isLuma( compID ) && pu.chromaTmrlFlag && ( (int) sps.getMaxPicWidthInLumaSamples() != cs.pps->getPicWidthInLumaSamples()
                                                       || (int) sps.getMaxPicHeightInLumaSamples() != cs.pps->getPicHeightInLumaSamples() ) )
    why = "chroma TMRL in a picture smaller than the SPS's";
  else if( cu.eipFlag )                                             why = "EIP";
  else if( !isLuma( compID ) && ipa.multiRefIndex && !pu.chromaTmrlFlag ) why = "chroma MRL";
  // CCP merge (JVET-AD0188: a model from a neighbour's, the history's or the defaults, the candidates ranked by
  // template costs; intra_ccp.glsl), its fusions (JVET-AG0059): with CCCM blended by JVET-AN0168's regression of the
  // template predictions where the SPS's ccpFusionMode asks for it (the encoder's outside all-intra), else averaged,
  // with DIMD chroma always averaged, its adjustment (JVET-AL0126: CCCM models only, on-the-fly ones added, the chosen
  // model's coefficients moved by template costs) and its decoder-derived fusion (JVET-AG0154: a pair of candidates
  // blended by that regression or averaged), also with LFNST / NSPT (JVET-AK0064: their mode a DIMD of the CU's final
  // predictions, intra_ccp.glsl's ccpMergeTransformMode); their SPS switches leave the other LM blocks as they are
  // (BVG-CCCM: otherCccm). Merge fusion exists in I slices with DIMD only (PU::hasCCPMergeFusionFlag): the GPU derives
  // the DIMD chroma mode there
#if JVET_AD0188_CCP_MERGE
  else if( !isLuma( compID ) && pu.idxNonLocalCCP && pu.ccpMergeFusionFlag && pu.ccpMergeFusionType != 0
           && !m.deriveModes )
    why = "CCP merge fusion with DIMD chroma without the GPU's mode derivation";
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION && !JVET_AN0168_REGRESSION_CCP_FUSION
  else if( !isLuma( compID ) && pu.idxNonLocalCCP && pu.ddNonLocalCCPFusion )
    why = "decoder-derived CCP merge fusion without the regression blend";
#endif
  else if( !isLuma( compID ) && pu.idxNonLocalCCP && !GPU_CCCM )    why = "CCP merge of another configuration";
#endif
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION
  // decoder-derived CCP fusion (JVET-AG0154: of a CCLM model and CCCM models fitted to the template the one of the
  // lowest probe-line cost, or two blended 3 : 1; intra_ccp.glsl), an LM mode (CABACReader parses it after the LM flag,
  // which needs LM chroma in the SPS), also with LFNST / NSPT (as a CCP merge block's)
  else if( !isLuma( compID ) && pu.decoderDerivedCcpMode && !GPU_CCCM )
    why = "decoder-derived CCP fusion of another configuration";
#endif
  // CCLM (chroma, the LM modes): the model from the template and the downsampled luma, its slope adjustment and LBCCP's
  // filter; with GLM (JVET-AA0126) the model of a gradient pattern's values instead of the downsampled luma; GLM with
  // luma (JVET-AB0092: glmIdc beyond the patterns) a CCCM model (intra_cccm.glsl), of the same kind in both components
  // (CABACReader::glmIdc gives Cr Cb's glmIdc; with LFNST / NSPT both predictions are the block's cccmPrediction's).
  // With LFNST / NSPT their mode is a DIMD of the CU's Cb and / or Cr predictions (deriveChromaIpmForTransform), which
  // without DIMD in the SPS leaves the mode the transforms read unset. Not above 10 bits (no test stream) nor with
  // unequal bit depths (the model takes the chroma bit depth, its division table m_auShiftLM the luma's)
  // CCCM (an LM mode with cccmFlag, JVET-AA0057): intra_cccm.glsl's plain models, GL-CCCM's (JVET-AC0054), those of
  // several downsampling filters (JVET-AD0202) and without subsampling (JVET-AC0147); with LFNST / NSPT their mode is
  // the DIMD of both components' CCCM predictions, which predIntraCCCM makes at the Cb block (each block on the GPU)
  else if( !isLuma( compID ) && pu.glmIdc.isActive() && !GPU_GLM ) why = "GLM of another configuration";
  else if( !isLuma( compID ) && pu.glmIdc.getIdc( compID, 0 ) > NUM_GLM_PATTERN && !GPU_CCCM ) why = "GLM with luma";
  else if( !isLuma( compID ) && cu.lfnstIdx
           && pu.glmIdc.getIdc( COMPONENT_Cb, 0 ) != pu.glmIdc.getIdc( COMPONENT_Cr, 0 )
           && std::max( pu.glmIdc.getIdc( COMPONENT_Cb, 0 ), pu.glmIdc.getIdc( COMPONENT_Cr, 0 ) ) > NUM_GLM_PATTERN )
    why = "GLM with luma with LFNST / NSPT, of another kind in the other component";
  else if( !isLuma( compID ) && pu.cccmFlag && otherCccm( pu ) )   why = otherCccm( pu );
  // LM chroma fusion (JVET-AC0119; geneChromaFusionPred with JVET-AL0191): its first type blends the block with plain
  // CCCM models of two (MMLM_CHROMA_IDX's), where PU::cccmMultiModeAvail allows them (else the CPU blends in a buffer
  // of a previous block); CFLM (types 2, 3) replaces it by a model of its own prediction and the luma
  // (xCflmCreateLumaRef downsamples the luma as xCccmGetLumaVal does: never without subsampling, which a cccmNoSubFlag
  // would select). The CPU predicts at the PU, the GPU at the block: one per component. Not above 10 bits nor with
  // unequal bit depths (the CCCM solvers' precision; CFLM's bias takes the luma bit depth)
  else if( !isLuma( compID ) && pu.isChromaFusion > 1 && !GPU_CFLM )         why = "CFLM";
  else if( !isLuma( compID ) && pu.isChromaFusion > 1 && pu.cccmNoSubFlag )  why = "CFLM without subsampling";
  else if( !isLuma( compID ) && pu.isChromaFusion && !GPU_CCCM )    why = "LM chroma fusion";
  else if( !isLuma( compID ) && pu.isChromaFusion == 1 && !PU::cccmMultiModeAvail( pu, MMLM_CHROMA_IDX, true ) )
    why = "LM chroma fusion without its CCCM models";
  else if( !isLuma( compID ) && pu.isChromaFusion && area != cu.blocks[compID] )
    why = "LM chroma fusion CU with several transform blocks";
  else if( !isLuma( compID ) && pu.isChromaFusion
           && ( sps.getBitDepth( CHANNEL_TYPE_LUMA ) > 10
                || sps.getBitDepth( CHANNEL_TYPE_CHROMA ) != sps.getBitDepth( CHANNEL_TYPE_LUMA ) ) )
    why = "LM chroma fusion above 10 bits or with unequal bit depths";
  else if( !isLuma( compID ) && PU::isLMCMode( dirMode ) && cu.lfnstIdx && !sps.getUseDimd() ) why = "CCLM with LFNST / NSPT without DIMD";
  // (DecCu::xIntraRecBlk with JVET-AK0064: the Cb block also predicts Cr, over the CU's area)
  else if( !isLuma( compID ) && PU::isLMCMode( dirMode ) && area != cu.blocks[compID] ) why = "CCLM CU with several transform blocks";
  else if( !isLuma( compID ) && PU::isLMCMode( dirMode ) && ( sps.getBitDepth( CHANNEL_TYPE_LUMA ) > 10
                                                              || sps.getBitDepth( CHANNEL_TYPE_CHROMA ) != sps.getBitDepth( CHANNEL_TYPE_LUMA ) ) )
    why = "CCLM above 10 bits or with unequal bit depths";
  else if( dirMode >= NUM_LUMA_MODE && !( ( cu.timd || cu.tmrlFlag ) && isLuma( compID ) ) && !( !isLuma( compID ) && PU::isLMCMode( dirMode ) ) )
    why = "intra mode beyond 66 (NN / ...)";
  else if( sps.getUseLmcs() && !m.reshape )                         why = "LMCS without the decoder's reshaper";
  else if( sps.getScalingListFlag() )                               why = "scaling lists";
  else if( sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag() ) why = "extended precision";
  // ISP splits luma into sub-partitions down to 1 x 4 / 4 x 1
  else if( ( ( area.width < 2 || area.height < 2 ) && !( cu.ispMode && isLuma( compID ) ) ) || area.width > 128 || area.height > 128 ) why = "block size";
  else if( ( area.width & ( area.width - 1 ) ) || ( area.height & ( area.height - 1 ) ) ) why = "block size not a power of 2";
  if( why )
  {
    reject( why );
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
      reject( why );
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
  // CCLM: the block's mark, the slope adjustments of the component's two models (cclmOffsets) and LBCCP's inside filter;
  // CCCM: the mark, LBCCP's inside filter and the template selection
  t.lm       = 0;
  t.cflm     = -1;
  t.ccp      = -1;
  t.ccpMerge = -1;
  // PDP (intra_pdp.glsl): a block it predicts where the mode has filters, with its references' padding and a MIP
  // block's filter set
#if JVET_AH0209_PDP
  t.pdp = pdp >= 0 && pdp < PDP_SIZES && ( !mip || pdpMipSet >= 0 )
            ? pdp | ipa.pdpPadW << 8 | ipa.pdpPadH << 16 | ( pdpMipSet + 1 ) << 24 : -1;
#else
  t.pdp = -1;
#endif
  // NN intra prediction (intra_nn.glsl): the block's record (IntraPredictionNN::gpuBlock's geometry), from the
  // unfiltered references without fusion (initPredIntraParams' PNN_IDX; fetchRef2nd)
  t.pnn = -1;
#if JVET_AJ0249_NEURAL_NETWORK_BASED
  if( pnn )
  {
    const IntraPredictionNN::GpuBlock b = IntraPredictionNN::gpuBlock( area, cu );
    if( b.model < 0 || b.model >= (int) m.sh->pnnModel.size() )
    {
      reject( "NN intra prediction of a shape without a model" );
      return;
    }
    const int32_t rec[PNN_REC] = { m.sh->pnnModel[b.model], b.factorVer, b.factorHor, b.transposed ? 1 : 0,
                                   b.maxNbLines[0], b.maxNbLines[1], b.maxCtxLength[0], b.maxCtxLength[1],
                                   b.ctxNbLines[0], b.ctxNbLines[1], b.ctxLength[0], b.ctxLength[1],
                                   b.availAbove, b.availLeft, b.nbLinesAbove, b.nbLinesLeft, b.heightTr, b.widthTr,
                                   b.leftBoundary ? 1 : 0, b.topBoundary ? 1 : 0 };
    static_assert( PNN_REC_TOP_BOUNDARY == PNN_REC - 1 && PNN_REC_AVAIL == 12 && PNN_REC_TARGET == 16, "PNN_REC" );
    t.pnn = (int32_t) m.timdData.size();
    m.timdData.insert( m.timdData.end(), rec, rec + PNN_REC );
    t.refFilter = t.fusion = 0;
  }
#endif
#if GPU_CCLM
  if( !isLuma( compID ) && PU::isLMCMode( dirMode ) )
  {
    const bool cb = compID == COMPONENT_Cb;
#if GPU_CCCM
    const bool glmLuma = pu.glmIdc.getIdc( compID, 0 ) > NUM_GLM_PATTERN;
#if JVET_AD0188_CCP_MERGE
    if( pu.idxNonLocalCCP )
    {
      // CCP merge (intra_ccp.glsl): the inherited model's kind, the GPU's (Tu::lm 0 until then, but merge fusion's);
      // the luma buffers of any kind in the workgroup's CCCM scratch (several filters' four, without subsampling the
      // luma itself) and the prediction, with merge fusion with CCCM also its CCCM prediction (at the scratch's end),
      // with decoder-derived merge fusion list[1]'s there, with fusion with DIMD chroma and LFNST / NSPT the DIMD
      // chroma mode's prediction of each component in turn (ccpMergeTransformMode); before it with either of them by a
      // blend model the candidates' (and the CCCM prediction's) template samples
      const bool fusionSlot = pu.ccpMergeFusionFlag && ( pu.ccpMergeFusionType == 0 || cu.lfnstIdx );
      const int  lumaWords  = std::max( 4 * cccmLumaArea( area, false ), cccmLumaArea( area, true ) );
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION
      const bool samples    = pu.ddNonLocalCCPFusion || ccpFusionBlend( pu );
#else
      const bool samples    = ccpFusionBlend( pu );
#endif
      const int  predWords  = samples ? ccpDerivedScratch( area ) : ( fusionSlot ? 2 : 1 ) * (int) area.area();
      m.cccmScratchWords  = std::max( m.cccmScratchWords, lumaWords + predWords );
    }
    else
#endif
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION
    if( pu.decoderDerivedCcpMode )
    {
      // decoder-derived CCP fusion (intra_ccp.glsl, through CCP merge's code): its CCLM models from the block's
      // template (an LM block's references: LM_ACTIVE, no slope adjustment), its CCCM models of one plain luma buffer
      // in the workgroup's CCCM scratch, after it the prediction, list[1]'s at the scratch's end and before that the
      // modes' probe samples
      t.lm = LM_ACTIVE | ( 8 << LM_DELTA0_SHIFT ) | ( 8 << LM_DELTA1_SHIFT );
      m.cccmScratchWords = std::max( m.cccmScratchWords, cccmLumaArea( area, false ) + ccpDerivedScratch( area ) );
    }
    else
#endif
    if( pu.cccmFlag || glmLuma )
    {
      // (GLM with luma: the template of its LM mode as a template selection, MDLM_L the left part, MDLM_T the one
      // above; its glmIdc in the GLM field)
      const int templateFlag = glmLuma ? ( dirMode == MDLM_L_IDX ? 2 : dirMode == MDLM_T_IDX ? 3 : 1 ) : pu.cccmFlag;
      t.lm = LM_ACTIVE | ( pu.ccInsideFilter ? LM_INSIDE : 0 ) | ( templateFlag << LM_CCCM_SHIFT )
             | ( pu.glCccmFlag ? LM_CCCM_GL : 0 ) | ( pu.cccmMultiFilterIdx << LM_CCCM_MDF_SHIFT )
             | ( pu.cccmNoSubFlag ? LM_CCCM_NOSUB : 0 )
             | ( glmLuma ? LM_CCCM_GLM_LUMA | ( pu.glmIdc.getIdc( compID, 0 ) << LM_GLM_SHIFT ) : 0 );
      // the luma buffers of several filters or of GLM with luma, a larger one or predictions beyond the shared ones
      // take the workgroup's CCCM scratch: room for the largest reference area's buffers and the predictions (with
      // LFNST / NSPT both components', which their mode's DIMD reads)
      const int numBuffers = glmLuma ? 2 : pu.cccmMultiFilterIdx == 0 ? 1 : pu.cccmMultiFilterIdx == 1 ? 4 : 2;
      const int lumaArea   = cccmLumaArea( area, pu.cccmNoSubFlag );
      const int predWords  = ( cu.lfnstIdx ? 2 : 1 ) * (int) area.area();
      if( numBuffers > 1 || lumaArea > CCCM_LUMA_MAX || predWords > CCCM_PRED_MAX )
      {
        m.cccmScratchWords = std::max( m.cccmScratchWords, numBuffers * lumaArea + predWords );
      }
    }
    else
#endif
    {
      t.lm = ( ( ( cb ? pu.cclmOffsets.cb0 : pu.cclmOffsets.cr0 ) + 8 ) << LM_DELTA0_SHIFT )
             | ( ( ( cb ? pu.cclmOffsets.cb1 : pu.cclmOffsets.cr1 ) + 8 ) << LM_DELTA1_SHIFT ) | ( pu.ccInsideFilter ? LM_INSIDE : 0 )
             | LM_ACTIVE | ( pu.glmIdc.getIdc( compID, 0 ) << LM_GLM_SHIFT );
    }
    // the predictions the transforms' DIMD reads by the TU's joint CbCr mode, the other's parameters (a CCCM block's -
    // also GLM with luma's - other component has the same kind of model, its kind's field is the other's)
    if( cu.lfnstIdx )
    {
      const ComponentID other = cb ? COMPONENT_Cr : COMPONENT_Cb;
      t.lm |= ( tu.jointCbCr != 2 ? LM_HIST_CB : 0 ) | ( tu.jointCbCr != 1 ? LM_HIST_CR : 0 );
      if( ( ( t.lm >> LM_CCCM_SHIFT ) & 3 ) == 0 )
      {
        t.lm |= ( ( ( ( cb ? pu.cclmOffsets.cr0 : pu.cclmOffsets.cb0 ) + 8 ) << LM_DELTA0_SHIFT )
                  | ( ( ( cb ? pu.cclmOffsets.cr1 : pu.cclmOffsets.cb1 ) + 8 ) << LM_DELTA1_SHIFT ) ) << LM_OTHER_SHIFT;
        t.lm |= pu.glmIdc.getIdc( other, 0 ) << LM_OTHER_GLM_SHIFT;
      }
    }
    // the LM prediction reads the unfiltered references (initPredIntraParams: no reference filter or fusion for a
    // chroma LM mode) and none of the angular and PDPC parameters, whose angles initPredIntraParams leaves as the
    // previous block's
    t.intraPredAngle = t.absInvAngle = t.angularScale = 0;
    t.refFilter = t.applyPDPC = t.gradPDPC = t.fusion = 0;
    bool newRecord = false;
    t.ccp = ccpRecord( m, pu, &newRecord );   // the CU's CCP model (CCP merge's curCand)
#if JVET_AD0188_CCP_MERGE
    if( newRecord )
    {
      // the CTU row's history (cs.ccpLut, reset at each CTU row's start: one slice, no tiles): a merge CU's entry
      // takes it as it is before the CU, then CU::saveModelsInHCCP pushes the CU's model (an LM mode's, but a single
      // tree's luma ISP CU's)
      const int row = ( pu.Cb().y << getComponentScaleY( COMPONENT_Cb, cu.chromaFormat ) ) / (int) sps.getMaxCUHeight();
      if( row != m.ccpHistRow )
      {
        m.ccpHistRow      = row;
        m.ccpHistRowStart = m.ccpHist.size();
      }
      if( pu.idxNonLocalCCP )
      {
        m.ccpMergeIndex[&pu] = ccpMergeEntry( m, pu );
      }
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION
      if( pu.decoderDerivedCcpMode )
      {
        m.ccpMergeIndex[&pu] = ccpDerivedEntry( m );
      }
#endif
      if( CS::isDualITree( *cu.cs ) || !cu.ispMode )
      {
        m.ccpHist.push_back( t.ccp );
      }
    }
    if( pu.idxNonLocalCCP )
    {
      // merge fusion with CCCM (predCCPCandidate's ccpMergeFusionType 0): LM chroma fusion's first type with the
      // weight 2, the plain CCCM models of MMLM_CHROMA_IDX averaged in, by a blend model the weight 0 that stands for
      // it (with DIMD chroma: DERIVE_CCP_DIMD, below)
      const int w0 = ccpFusionBlend( pu ) ? 0 : 2;
      t.lm       = pu.ccpMergeFusionFlag && pu.ccpMergeFusionType == 0
                   ? ( 1 << LM_CCCM_SHIFT ) | LM_FUSION | ( 1 << LM_FUSION_TYPE_SHIFT ) | ( w0 << LM_FUSION_W_SHIFT )
                   : 0;
      // with LFNST / NSPT the predictions their mode's DIMD reads, as an LM block's (intra_ccp.glsl's
      // ccpMergeTransformMode)
      t.lm      |= cu.lfnstIdx ? ( tu.jointCbCr != 2 ? LM_HIST_CB : 0 ) | ( tu.jointCbCr != 1 ? LM_HIST_CR : 0 ) : 0;
      t.ccpMerge = m.ccpMergeIndex.at( &pu );   // (relative until endPicture)
#if JVET_AL0126_CCP_MERGE_WITH_ADJUST
      m.numCcpAdjust += pu.ccpMergeAdjustFlag != 0;
#endif
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION
      m.numDdMergeFusion += pu.ddNonLocalCCPFusion > 0;
#endif
      m.numFusionBlend += ccpFusionBlend( pu );
    }
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION
    if( pu.decoderDerivedCcpMode )
    {
      t.ccpMerge = m.ccpMergeIndex.at( &pu );
      m.numDerivedCcp++;
    }
#endif
#endif
  }
#endif
#if GPU_CCCM
  // LM chroma fusion (geneChromaFusionPred, its first type): the plain CCCM models of MMLM_CHROMA_IDX blended in, the
  // weight of the block's own prediction by whether the CUs below-left and above-right of the CU are LM CUs (1 both,
  // 3 neither, else 2), its CCCM luma buffer and prediction in the scratch; the models are the CU's curCand (a CCP
  // merge candidate of its neighbours, not of the CTU row's history: CU::saveModelsInHCCP takes LM CUs only)
  if( !isLuma( compID ) && pu.isChromaFusion == 1 )
  {
    t.ccp = ccpRecord( m, pu );
    const Position        posBL = pu.Cb().bottomLeft(), posTR = pu.Cb().topRight();
    const PredictionUnit* left  = cu.cs->getPURestricted( posBL.offset( -1, 0 ), pu, CHANNEL_TYPE_CHROMA );
    const PredictionUnit* above = cu.cs->getPURestricted( posTR.offset( 0, -1 ), pu, CHANNEL_TYPE_CHROMA );
    const bool leftLm  = left && PU::isLMCMode( left->intraDir[1] );
    const bool aboveLm = above && PU::isLMCMode( above->intraDir[1] );
    const int  w0      = leftLm && aboveLm ? 1 : !leftLm && !aboveLm ? 3 : 2;
    t.lm = ( 1 << LM_CCCM_SHIFT ) | LM_FUSION | ( 1 << LM_FUSION_TYPE_SHIFT ) | ( w0 << LM_FUSION_W_SHIFT );
    m.cccmScratchWords = std::max( m.cccmScratchWords, cccmLumaArea( area, false ) + (int) area.area() );
  }
#endif
#if GPU_CFLM
  // CFLM (geneChromaFusionPred's types 2 and 3): one model or two of the block's own prediction and the luma, fitted
  // where the template (CFLM_TEMPLATE lines above / columns left of the block, those the picture has) has its
  // prediction by the block's mode from the template block's reference line (xCflmCreateChromaPred: the availability
  // of its units in the record at Tu::cflm); the area's luma buffer (padded by a sample all around) where it exceeds
  // shared memory and its chroma predictions (the template's and the block's) in the scratch
  if( !isLuma( compID ) && pu.isChromaFusion > 1 )
  {
    const int tw = std::min<int>( area.x, CFLM_TEMPLATE ), th = std::min<int>( area.y, CFLM_TEMPLATE );
    t.lm = LM_FUSION | ( pu.isChromaFusion << LM_FUSION_TYPE_SHIFT );
    if( tw > 0 || th > 0 )
    {
      std::vector<int32_t> bits;
      const TemplateUnits  u  = templateUnits( cu, area, tw, th, CFLM_REF_SHIFT, bits );
      const int32_t*       up = reinterpret_cast<const int32_t*>( &u );
      CHECK( u.totalUnits > TIMD_UNITS_MAX, "CFLM's reference units" );   // (a block of 64 x 64: 133)
      t.cflm = (int32_t) m.timdData.size();
      m.timdData.insert( m.timdData.end(), up, up + sizeof( u ) / 4 );
      m.timdData.insert( m.timdData.end(), bits.begin(), bits.end() );
    }
    const int areaW = area.width + tw, areaH = area.height + th;
    m.cccmScratchWords = std::max( m.cccmScratchWords, ( areaW + 2 ) * ( areaH + 2 ) + areaW * areaH );
  }
#endif
  // a chroma PU without a CCP record: verification expects its curCand untyped (no merge candidate of its neighbours)
  if( !isLuma( compID ) && t.ccp < 0 )
  {
    m.untypedCcpPu.push_back( &pu );
  }
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
    reject( "intra bilateral filter of a split CU" );
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
      reject( "MIP mode index out of range" );
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
      reject( "BDPCM with an intra mode other than its direction" );
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
    // a DIMD CU's fetchRef2nd without its DIMD term (the CPU's depends on the blending the GPU derives): DIMD has no
    // MRL, ISP, MIP or SGPM, so the size, the CTU's first row and the SPS's switch remain (intra_rec ends it with blending);
    // OBIC and intra merge CUs always blend (deriveObicMode adds planar), so have none; a TIMD CU (no MRL, ISP, MIP,
    // SGPM, directional planar either) the same as DIMD's, which counts where its modes do not blend (where they
    // blend, intra_rec has TIMD's rule, or TIMD merge's)
    // (ISP sub-partitions never fuse; their snapshot's flag can be the previous block's)
    const bool dimdCu      = m.cus[cuIdx].kind == CU_DIMD;
    const bool obicCu      = m.cus[cuIdx].kind == CU_OBIC;
    const bool mergeCu     = m.cus[cuIdx].kind == CU_MERGE;
    const bool timdCu      = isLuma( compID ) && m.cus[cuIdx].kind == CU_TIMD;
    const bool timdMergeCu = timdCu && cu.timdMrg;
    // (a TMRL CU's line is never 0, and is 12 only where y > 12: of initIntraPatternChType's line conditions none
    // applies, only the size and the SPS's switch count; a chroma TMRL CU's blocks have no fusion: fetchRef2nd is luma's)
    const bool tmrlCu      = isLuma( compID ) ? m.cus[cuIdx].kind == CU_TMRL : m.cus[cuIdx].kind == CU_CHROMA_TMRL;
    const bool fetch2nd = obicCu || mergeCu ? false
                          : dimdCu || timdCu ? area.area() >= SIZE_CONSTRAINT && cu.blocks[COMPONENT_Y].y % sps.getMaxCUWidth() != 0
                                                 && !cu.plIdx && !sps.getDisableIntraFusion()
                          : tmrlCu && isLuma( compID ) ? area.area() >= SIZE_CONSTRAINT && !sps.getDisableIntraFusion()
                                                       : ipa.fetchRef2nd && !( cu.ispMode && isLuma( compID ) );
    t.derive = DERIVE_MODE | ( fetch2nd ? DERIVE_FETCH2ND : 0 ) | ( bfBase ? DERIVE_BF : 0 )
               | ( sps.getDisablePdpc() ? DERIVE_NO_PDPC : 0 )
               | ( sps.getSpsRangeExtension().getIntraSmoothingDisabledFlag() ? DERIVE_NO_SMOOTHING : 0 )
               | ( sps.getDisableIntraFusion() ? DERIVE_NO_FUSION : 0 )
               | ( dimdCu ? DERIVE_DIMD : 0 ) | ( obicCu ? DERIVE_OBIC : 0 ) | ( mergeCu ? DERIVE_MERGE : 0 )
               | ( timdCu ? DERIVE_TIMD : 0 ) | ( timdMergeCu ? DERIVE_TIMD_MERGE : 0 ) | ( tmrlCu ? DERIVE_TMRL : 0 );
    // a CCP merge block of fusion with DIMD chroma (its CU's node a DIMD chroma one): intra_rec's blending of the merge
    // prediction and the mode's, each weighed 1, the average rounded
    const bool ccpDimd = ccpFusionDimd( pu );
    t.derive |= ccpDimd ? DERIVE_CCP_DIMD : 0;
    m.blendBlocks = m.blendBlocks || dimdCu || obicCu || mergeCu || timdCu || ccpDimd;
#if GPU_TIMD
    if( timdCu )
    {
      // the CPU's parameters of the main mode, for the comparison: where the CU blends, its snapshot after the
      // prediction holds the last blended mode's (predIntraAng's initPredIntraParams of each), and the parameters
      // are recomputed (not in ON mode: the CPU's TIMD is of a picture it has not reconstructed); the main mode's
      // fusion stays (those calls leave it) - a TIMD merge CU's is on (fetchRef2nd), unless the SPS disables it (its
      // secondary's then by its slope) - the bilateral filter filters only a blend. A TIMD-SAD CU predicts
      // deriveTimdModeSad's modes alike, a TIMD merge CU its candidate's (every wide angle getWideAngleExt's)
      const bool blended = timdMergeCu ? cu.timdMrgIsBlended[0] : cu.timdSad ? cu.timdIsBlendedSad : cu.timdIsBlended;
      if( blended && !m.on && timdMergeCu )
      {
        applyTimdModeParams( t, cu.timdMrgList[0][0], true );
        t.fusion = sps.getDisableIntraFusion() ? 0 : 1;
      }
      else if( blended && !m.on )
      {
        applyTimdModeParams( t, cu.timdSad ? cu.timdModeSad : cu.timdMode, cu.timdSad ? cu.timdModeCheckWASad : cu.timdModeCheckWA );
      }
      t.bf = blended && bfBase ? 1 : 0;
    }
#endif
    if( dimdCu || obicCu || mergeCu )
    {
      // the CPU's parameters of the main mode, for the comparison: its snapshot after the prediction holds the last
      // blended mode's (predIntraAng's initPredIntraParams of each); its fusion is the main mode's without blending,
      // else off (fetchRef2nd), and the bilateral filter filters the blend
      applyModeParams( t, dirMode, cu.plIdx );
      t.fusion = ipa.applyFusion;
      t.bf     = ( obicCu || mergeCu || cu.dimdBlending ) && bfBase ? 1 : 0;
    }
  }
  const PreCalcValues& pcv = *cs.pcv;
  const bool noShift = pcv.noChroma2x2 && area.width == 4;
  t.unitW    = pcv.minCUWidth >> ( noShift ? 0 : getComponentScaleX( compID, sps.getChromaFormatIdc() ) );
  t.unitH    = pcv.minCUHeight >> ( noShift ? 0 : getComponentScaleY( compID, sps.getChromaFormatIdc() ) );
  t.bitDepth = sps.getBitDepth( chType );
  if( t.mrl && t.unitW != t.unitH )   // xFillReferenceSamples counts the corner samples of both arms by the unit height
  {
    reject( "MRL with non-square availability units" );
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
      reject( "joint CbCr: a Cr block not right after its Cb block" );
      return;
    }
    t.jccr = JCCR_DERIVED;
    const int32_t cbIndex = (int32_t) m.rec.size() - 1;
    if( cb->spOffset >= 0 )   // sign prediction of the pair, decided by the coded component's block
    {
      if( std::abs( cb->jccr ) == 3 )   // Cr coded: this block decides and writes the Cb block's residual
      {
        t.spCoded               = cbIndex;
        m.rec[cbIndex].resiFrom = cbIndex + 1;
        // (with the Cb block's transforms; of a CCLM CU their mode is the DIMD this block then finds too)
        t.derive |= cb->derive & DERIVE_CCLM_DIMD;
      }
      else
      {
        t.resiFrom = cbIndex;
      }
    }
    else if( cb->derive & DERIVE_TRANSFORM )   // the Cb block computes both residuals in intra_rec
    {
      t.resiFrom = cbIndex;
    }
  }
  // sign prediction (TrQuant::predCoeffSigns): prepareSignPred took the predicted levels of the coded component before
  // the CPU decided their signs
  Impl::PendingSignPred& sign     = m.pendingSign;
  const bool             signPred = sign.num > 0 && t.jccr != JCCR_DERIVED;
  if( signPred && ( sign.comp != resiComp || sign.pos != tu.blocks[resiComp].pos() ) )
  {
    reject( "sign prediction taken from another block" );
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
      reject( "transform types without the decoder's TrQuant" );
      return;
    }
    Secondary sec;
    if( !invSecondary( tu, resiComp, sec ) )
    {
      reject( "LFNST without its extension (the VVC kernels)" );
      return;
    }
    int        trTypeHor, trTypeVer;
    const bool separable = m.trQuant->getInvTransformTypes( tu, resiComp, trTypeHor, trTypeVer );
    if( area.width > 1 && area.height > 1 && separable == ( sec.kind == SEC_NSPT ) )   // xIT: NSPT or the separable one
    {
      reject( "NSPT where xIT does not apply it" );
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
        reject( "an LFNST / NSPT matrix or scan the GPU lacks" );
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
          reject( "a transform type without a matrix of this size" );
          return;
        }
      }
      else
      {
        reject( "a transform the GPU lacks" );
        return;
      }
    }
    else
    {
      t.mtxV = m.sh->trOffset[trTypeVer][floorLog2( area.height ) - 1];
      t.mtxH = m.sh->trOffset[trTypeHor][floorLog2( area.width ) - 1];
      if( t.mtxV < 0 || t.mtxH < 0 )
      {
        reject( "a transform type without a matrix of this size" );
        return;
      }
    }
    // a derived mode's transforms (intra_rec's selectTransform): the kernel pair of ECM's explicit MTS
    // (TrQuant::getTrTypes with MTSExt) and the LFNST / NSPT matrix and transposition (getFinalIntraModeForTransform)
    // depend on the mode; the host gives what does not, and intra_rec computes the residual after the mode's node.
    // With MTSS the CU's node chooses the LFNST / NSPT mode and bucket (Cu::mtss). A MIP block's LFNST / NSPT mode is
    // a DIMD of its own prediction, which intra_rec derives after the prediction (MTS of MIP takes planar's kernels)
    const bool mtsByMode = isLuma( resiComp ) && sps.getUseMTS() && sps.getUseIntraMTS() && sps.getUseMTSExt()
                           && !cu.ispMode && tu.mtsIdx[resiComp] > MTS_SKIP;
    const bool mipDimd   = GPU_MIP_DIMD && mip && cu.lfnstIdx;
    // (a DIMD of the CU's chroma predictions: an LM block's, a CCP merge block's)
    const bool cclmDimd  = ( ( t.lm & LM_ACTIVE ) || t.ccpMerge >= 0 ) && cu.lfnstIdx;
    // an NN block's (MTS too): a DIMD of its prediction (indicesRepresentationPnn), LFNST / NSPT by the lfnstSecFlag
    // rule where it applies (getFinalIntraModeForTransform)
    const bool pnnDimd   = pnn;
#if JVET_AJ0249_NEURAL_NETWORK_BASED
    const bool pnnSec    = pnn && cu.lfnstSecFlag && isAllowedMultiple( area.width, area.height );
#else
    const bool pnnSec    = false;
#endif
    if( ( derived || mipDimd || cclmDimd || pnnDimd ) && ( mtsByMode || sec.kind != SEC_NONE ) )
    {
      CHECK( mipDimd && cuIdx < 0, "a MIP block's LFNST mode without its CU's record (DIMD has the GPU derive modes)" );
      CHECK( pnnDimd && cuIdx < 0, "an NN block's transforms' mode without its CU's record" );
      m.numCcpLfnst += cclmDimd && t.ccpMerge >= 0;
      t.derive   |= DERIVE_TRANSFORM | ( mipDimd ? DERIVE_MIP_DIMD : 0 ) | ( cclmDimd ? DERIVE_CCLM_DIMD : 0 )
                    | ( pnnDimd ? DERIVE_PNN_DIMD : 0 ) | ( pnnSec ? DERIVE_PNN_LFNST_SEC : 0 )
                    | ( isLuma( resiComp ) && usesMtss( tu ) ? DERIVE_MTSS : 0 );
      if( mipDimd || pnnDimd )
      {
        m.cuMipDimd[cuIdx] = pnnDimd ? 2 : 1;
      }
      t.mtsSet    = mtsByMode ? tu.mtsIdx[resiComp] - MTS_DST7_DST7 + 1 : 0;
      t.secIdx    = cu.lfnstIdx - 1;
      t.secBucket = sec.kind == SEC_NSPT ? PU::getNSPTBucket( tu, false, resiComp ) : 0;
    }
    if( ( t.derive & DERIVE_MTSS ) && m.cus[cuIdx].mtss < 0 )
    {
      reject( "MTSS without the CU's mode list" );
      return;
    }
    // a TIMD merge CU's luma block without LFNST in dual tree (TrQuant::xIT's TIMD merge branch; not with implicit
    // MTS, which the SPS has not): a side of 4 to 16 takes DST-VII and one above IntraMTSMaxSize DCT-II (the types
    // above), any other the type of the candidate its node chooses
    if( derived && isLuma( resiComp ) && cu.timdMrg && !cu.lfnstIdx && CS::isDualITree( cs ) && !sps.getUseImplicitMTS() )
    {
      auto inherits = [&]( int side ) { return ( side < 4 || side > 16 ) && side <= sps.getIntraMTSMaxSize(); };
      t.derive |= ( inherits( area.width ) ? DERIVE_INHERIT_HOR : 0 ) | ( inherits( area.height ) ? DERIVE_INHERIT_VER : 0 );
      if( t.derive & ( DERIVE_INHERIT_HOR | DERIVE_INHERIT_VER ) )
      {
        t.derive |= DERIVE_TRANSFORM;
      }
    }
  }
  if( signPred && ( !t.cbf || ts || area.width < 4 || area.height < 4 || area.width > 32 || area.height > 32 ) )
  {
    reject( "sign prediction of a block outside TU::getUseSignPred's conditions" );
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
    reject( "transform range above 16 bits" );
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
    // with a derived mode wherever the filter may apply (intra_rec decides it; the flag is read only where it does)
    if( ( t.bf || ( t.derive & DERIVE_BF ) ) && bfLmcs )
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
        reject( "LFNST with levels outside its region" );
        return;
      }
      t.nzRows = t.nzCols = t.secSize;
    }
    // intra_rec keeps the coefficients of a derived mode's LFNST / NSPT (at most 16 x 16) and of sign prediction's
    // blocks (at most 32 x 32) in 32 x 32 entries
    if( ( t.derive & DERIVE_TRANSFORM ) && ( t.secKind != SEC_NONE || signPred ) && t.nzRows * t.nzCols > 1024 )
    {
      reject( "a derived mode's transform with more than 32 x 32 coefficients" );
      return;
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
  if( t.ccp >= 0 )   // (a merge block's dependents wait for the blocks writing the records it reads)
  {
    m.ccpWriters[size_t( t.ccp ) * CCP_WRITERS + ( compID == COMPONENT_Cr ? 1 : 0 )] = (int32_t) m.rec.size();
  }
  if( m.partial )
  {
    m.cuTus[&cu].push_back( (int32_t) m.rec.size() );
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

  // CCP merge's data (Params::ccpData): the records' writers, the histories, the merge entries, with the positions
  // placed (an entry's history, its default records after the PUs' records; a merge block's entry)
  std::vector<int32_t> ccpData( m.ccpWriters );
  const int32_t        histBase = (int32_t) ccpData.size(), mergeBase = histBase + (int32_t) m.ccpHist.size();
  ccpData.insert( ccpData.end(), m.ccpHist.begin(), m.ccpHist.end() );
  // (each merge CU's default records: the Cb block's, the Cr block's; with LFNST / NSPT in the picture also those of
  // each block's prediction of the other component, intra_ccp.glsl's ccpMergeTransformMode)
  const int defaultSets = m.numCcpLfnst > 0 ? 4 : 2;
  for( const auto& pe: m.ccpMergeIndex )
  {
    int32_t* e = &m.ccpMerge[pe.second];
    e[CCP_MERGE_HIST] += histBase;
    e[CCP_MERGE_DEFAULTS] = m.numCcp + e[CCP_MERGE_DEFAULTS] * defaultSets * CCP_DEFAULTS;
  }
  ccpData.insert( ccpData.end(), m.ccpMerge.begin(), m.ccpMerge.end() );
  for( Tu& t: m.rec )
  {
    t.ccpMerge += t.ccpMerge >= 0 ? mergeBase : 0;
  }

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
    if( ( t.derive & DERIVE_NODE ) && !( t.derive & DERIVE_CPU_FILL ) )   // no samples
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
      // the owners of the samples of component comp in [x0, x1] x [y0, y1]
      auto addArea = [&]( int comp, int x0, int y0, int x1, int y1 )
      {
        const int ch = comp == COMPONENT_Y ? 0 : 1;
        for( int uy = y0 >> P.unitLog2Y[ch]; uy <= y1 >> P.unitLog2Y[ch]; uy++ )
        {
          for( int ux = x0 >> P.unitLog2X[ch]; ux <= x1 >> P.unitLog2X[ch]; ux++ )
          {
            if( ux >= 0 && uy >= 0 && ux < P.ownerW[ch] && uy < P.ownerH[ch] )
            {
              addBlock( ownerMap[comp][size_t( uy ) * P.ownerW[ch] + ux] );
            }
          }
        }
      };
      if( t.derive & DERIVE_NODE )   // a CU's mode: the derived modes it reads, from those CUs' nodes
      {
        const Cu& c = m.cus[t.cu];
        // the node that publishes CU n's mode: none for a mode the host knows (a MIP CU's node with MTSS derives only
        // MTSS's list)
        auto addModeNode = [&]( int n )
        {
          if( m.cus[n].kind != CU_KNOWN )
          {
            addBlock( m.cus[n].node );
          }
        };
        if( c.kind == CU_MPM || c.kind == CU_TIMD || c.kind == CU_TMRL )
        {
          for( int k = 0; k < MPM_NEIGHBOURS; k++ )
          {
            if( c.nbr[k] >= 0 )
            {
              addModeNode( c.nbr[k] );
            }
          }
        }
        else if( c.lumaCu >= 0 )
        {
          addModeNode( c.lumaCu );
        }
        for( int k = 0; c.obic >= 0 && k < OBIC_NBRS; k++ )   // OBIC: the modes of its neighbours
        {
          const ObicNbr& nb = m.obicNbrs[size_t( c.obic ) * OBIC_NBRS + k];
          if( nb.cu >= 0 )
          {
            addModeNode( nb.cu );
          }
        }
        // DIMD and MTSS's mode list: the samples their histograms read (dimdCallArea)
        auto addCallAreas = [&]( int first, int num )
        {
          for( int k = 0; k < num; k++ )
          {
            const DimdCall& call = m.dimdCalls[first + k];
            int x0, y0, x1, y1;
            if( dimdCallArea( call, x0, y0, x1, y1 ) )
            {
              addArea( call.comp, x0, y0, x1, y1 );
            }
          }
        };
        addCallAreas( c.dimd, c.dimdNum );
        addCallAreas( c.mtss, c.mtss >= 0 ? MTSS_CALLS : 0 );
        // a template: its samples and those of its references (templateReadAreas)
        auto addTemplateAreas = [&]( const TemplateUnits& tu )
        {
          if( tu.tw > 0 || tu.th > 0 )
          {
            int area[4][4];
            templateReadAreas( c, tu, area );
            for( int k = 0; k < 4; k++ )
            {
              addArea( COMPONENT_Y, area[k][0], area[k][1], area[k][2], area[k][3] );
            }
          }
        };
        // intra merge: the parameters of its neighbours, its template
        if( c.merge >= 0 )
        {
          const MergeHead& h = *reinterpret_cast<const MergeHead*>( &m.mergeData[c.merge] );
          for( int k = 0; k < h.nbrNum; k++ )
          {
            addBlock( m.cus[m.mergeData[c.merge + sizeof( MergeHead ) / 4 + 2 * k]].node );
          }
          addTemplateAreas( h.t );
        }
        if( c.timd >= 0 )   // TIMD: its template (TIMD-SAD: also deriveTimdModeSad's; TIMD merge: its candidates' TIMD)
        {
          const TimdHead& h = *reinterpret_cast<const TimdHead*>( &m.timdData[c.timd] );
          addTemplateAreas( h.t );
          if( h.sad )
          {
            addTemplateAreas( h.ts );
          }
          for( int k = 0; k < h.merge; k++ )
          {
            addModeNode( h.mergeNbr[k] );
          }
        }
        if( c.tmrl >= 0 && c.kind == CU_TMRL )   // TMRL: the reference lines it tries (whole arms), its one-sample template
        {
          for( int k = 0; k < tmrlNumLines( c.y ); k++ )
          {
            const int mrl = TMRL_LINES[k], x0 = c.x - 1 - mrl, y0 = c.y - 1 - mrl;
            addArea( COMPONENT_Y, x0, y0, c.x - 1 + ( c.w << 3 ), y0 );
            addArea( COMPONENT_Y, x0, y0, x0, c.y - 1 + ( c.h << 3 ) );
          }
          addArea( COMPONENT_Y, c.x, c.y - 1, c.x + c.w - 1, c.y - 1 );
          addArea( COMPONENT_Y, c.x - 1, c.y, c.x - 1, c.y + c.h - 1 );
        }
        // chroma TMRL (intra_deps' chromaTmrlDeps): the modes of the luma CUs and chroma neighbours it reads; of Cb and
        // Cr each line's whole arms (2 x (the side + 1)) and the template, of luma the co-located block and its lines
        // twice as far (arms of 2 x its sides) up to the CTU's right and bottom edges
        if( c.kind == CU_CHROMA_TMRL )
        {
          for( int k = CHROMA_TMRL_NBR_LUMA; k < CHROMA_TMRL_NBR_CHROMA + 5; k++ )
          {
            if( c.nbr[k] >= 0 )
            {
              addModeNode( c.nbr[k] );
            }
          }
          const int ctu = (int) cs.sps->getCTUSize(), lx = c.x << 1, ly = c.y << 1, lw = c.w << 1, lh = c.h << 1;
          const int ctuRight = ( lx / ctu + 1 ) * ctu - 1, ctuBottom = ( ly / ctu + 1 ) * ctu - 1;
          for( int k = 0; k < chromaTmrlNumLines( c.y ); k++ )
          {
            const int line = CHROMA_TMRL_LINES[k];
            for( int comp = COMPONENT_Cb; comp <= COMPONENT_Cr; comp++ )
            {
              const int x0 = c.x - 1 - line, y0 = c.y - 1 - line;
              addArea( comp, x0, y0, c.x - 1 + ( ( c.w + 1 ) << 1 ), y0 );
              addArea( comp, x0, y0, x0, c.y - 1 + ( ( c.h + 1 ) << 1 ) );
            }
            const int mrl = line << 1, x0 = lx - 1 - mrl, y0 = ly - 1 - mrl;
            addArea( COMPONENT_Y, x0, y0, std::min( lx - 1 + ( lw << 1 ), ctuRight ), y0 );
            addArea( COMPONENT_Y, x0, y0, x0, std::min( ly - 1 + ( lh << 1 ), ctuBottom ) );
          }
          for( int comp = COMPONENT_Cb; comp <= COMPONENT_Cr; comp++ )
          {
            addArea( comp, c.x, c.y - 1, c.x + c.w - 1, c.y - 1 );
            addArea( comp, c.x - 1, c.y, c.x - 1, c.y + c.h - 1 );
          }
          addArea( COMPONENT_Y, lx, ly, lx + lw - 1, ly + lh - 1 );
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
        else if( tm.lm & LM_ACTIVE )   // CCLM's template: entries 1 .. 2W / 2H (MDLM's longest)
        {
          topMax  = 2 * W;
          leftMax = 2 * H;
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
      // applies to one. A TIMD block's modes are of the 131-mode set, with wide angles beyond the regular ones, and
      // blend, fused from the next line also where the CU's fetchRef2nd is off: the whole arms of both lines, the
      // filter's neighbours where it may filter a blend. A TMRL block's line comes from its node: the whole arms of every
      // line the node may choose and, for luma, of the next one (intra fusion's), the filter's neighbours where it may
      // filter (DC); the block finds its line's availability itself (its avail and readMax stay 0)
      const bool derived      = ( t.derive & DERIVE_MODE ) != 0;
      const bool timdBlock    = ( t.derive & DERIVE_TIMD ) != 0;
      const bool tmrlBlock    = ( t.derive & DERIVE_TMRL ) != 0;
      const bool lumaTmrl     = t.comp == COMPONENT_Y;
      const int  numTmrlLines = !tmrlBlock ? 0 : lumaTmrl ? tmrlNumLines( t.y ) : chromaTmrlNumLines( t.y );
      const Cu*  cuRec        = derived ? &m.cus[t.cu] : nullptr;
      const int  numModes     = !derived ? 1 : timdBlock ? 0 : cuRec->modes[0] < 0 ? NUM_LUMA_MODE : cuRec->modes[1] < 0 ? 1 : 2;
      bool       bfNeeded     = ( !derived && t.bf != 0 ) || ( ( timdBlock || tmrlBlock ) && ( t.derive & DERIVE_BF ) );
      for( int k = 0; k < numTmrlLines; k++ )
      {
        const int line = lumaTmrl ? TMRL_LINES[k] : CHROMA_TMRL_LINES[k];
        for( int mrl = line; mrl <= line + ( lumaTmrl ? 1 : 0 ); mrl++ )
        {
          const int x0 = fx - 1 - mrl, y0 = fy - 1 - mrl;
          for( int x = x0; x <= fx - 1 + predSize; x++ )
          {
            addOwner( x, y0 );
          }
          for( int y = y0; y <= fy - 1 + predHSize; y++ )
          {
            addOwner( x0, y );
          }
        }
      }
      for( int ln = 0; ln < 2 && !tmrlBlock; ln++ )
      {
        const int mrl = t.mrl + ln;   // the reference line (multiRefIdx; the fusion line is the next one)
        int topMax = timdBlock ? topLen + mrl : -1, leftMax = timdBlock ? leftLen + mrl : -1;   // -1: the line is not read
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

        // the owners of the samples in range (the kernel's fill rule); a CCLM block with LFNST / NSPT that predicts the
        // other chroma component too (cclmTransformMode) reads that component's line at the same positions
        const bool other = ln == 0 && ( t.derive & DERIVE_CCLM_DIMD )
                           && ( t.lm & ( t.comp == COMPONENT_Cb ? LM_HIST_CR : LM_HIST_CB ) );
        for( int side = 0; side < 2; side++ )
        {
          for( int idx = 0; idx <= t.readMax[ln][side]; idx++ )
          {
            int sx, sy;
            if( refSource( t, ln, side == 0, idx, sx, sy ) )
            {
              addOwner( sx, sy );
              if( other )
              {
                addBlock( own( 3 - t.comp, sx, sy ) );
              }
            }
          }
        }
      }
      if( bfNeeded )   // the bilateral filter's neighbours (rows y-2, y-1; columns x-2, x-1)
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
      if( t.pdp >= 0 )   // PDP (intra_pdp.glsl): two lines above with the corner and left, arms of twice the sides
      {
        addArea( COMPONENT_Y, t.x - 2, t.y - 2, t.x + 2 * t.w - 1, t.y - 1 );
        addArea( COMPONENT_Y, t.x - 2, t.y, t.x - 1, t.y + 2 * t.h - 1 );
      }
      if( t.pnn >= 0 )   // NN intra prediction (intra_nn.glsl): its context, the lines above with the corner and left
      {
        const int32_t* a = &m.timdData[t.pnn + PNN_REC_AVAIL];   // the above arm, the left arm, the lines above / left
        addArea( COMPONENT_Y, t.x - a[3], t.y - a[2], t.x + a[0] - 1, t.y - 1 );
        addArea( COMPONENT_Y, t.x - a[3], t.y, t.x - 1, t.y + a[1] - 1 );
      }
      // CCLM: the luma its downsampling reads (xGetLumaRecPixels, 4:2:0): the co-located block, the rows above it (up
      // to 3; MDLM's template 2W wide) and the columns left of it (3; 2H high)
      if( t.lm & LM_ACTIVE )
      {
        const int lx = t.x << 1, ly = t.y << 1, lw = t.w << 1, lh = t.h << 1;
        addArea( COMPONENT_Y, lx, ly, lx + lw - 1, ly + lh - 1 );
        addArea( COMPONENT_Y, lx - 1, ly - 3, lx + 2 * lw, ly - 1 );
        addArea( COMPONENT_Y, lx - 3, ly - 1, lx - 1, ly + 2 * lh - 1 );
      }
      // CCCM (intra_cccm.glsl), for the largest reference area the block may have (6 lines above / left, arms of twice
      // the block): of both chroma components the template (with the offsets' samples, the costs' arms and the inside
      // filter's neighbours), of luma the area's downsampling with a sample of padding around (with the luma offset's
      // sample; the luma right of / below the block that a dual tree reads is its CTU's, decoded before). Also a CCP
      // merge block's (its candidates: CCCM's of any kind, CCLM's and GLM's luma), which waits for the blocks writing
      // the records it reads too: its sources' and its CTU row's history's (each block of the CU ranks the list itself:
      // the Cr block waiting for the Cb block's record would close a cycle where joint CbCr's sign prediction has the
      // Cb block wait for the Cr block, resiFrom)
      if( t.ccpMerge >= 0 )
      {
        auto addWriters = [&]( int r )
        {
          for( int k = 0; k < CCP_WRITERS; k++ )
          {
            addBlock( ccpData[size_t( r ) * CCP_WRITERS + k] );
          }
        };
        const int32_t* e = &ccpData[t.ccpMerge];
        for( int s = 0; s < e[CCP_MERGE_NUM_SPATIAL]; s++ )
        {
          addWriters( e[CCP_MERGE_SPATIAL + s] );
        }
        for( int h = 0; h < e[CCP_MERGE_HIST_NUM]; h++ )
        {
          addWriters( ccpData[e[CCP_MERGE_HIST] + h] );
        }
      }
      if( ( ( t.lm >> LM_CCCM_SHIFT ) & 3 ) || t.ccpMerge >= 0 )
      {
        const int sx = std::min( t.x, CCCM_WINDOW_SIZE ), sy = std::min( t.y, CCCM_WINDOW_SIZE );
        const int x1 = t.x + 2 * t.w - 1, y1 = t.y + 2 * t.h - 1;
        for( int comp = COMPONENT_Cb; comp <= COMPONENT_Cr; comp++ )
        {
          addArea( comp, t.x - sx, t.y - sy, x1, t.y - 1 );
          addArea( comp, t.x - sx, t.y, t.x - 1, y1 );
        }
        addArea( COMPONENT_Y, 2 * ( t.x - sx - 1 ) - 1, 2 * ( t.y - sy - 1 ) - 1, 2 * ( x1 + 1 ) + 1, 2 * ( y1 + 1 ) + 1 );
      }
      // CFLM (intra_cccm.glsl's cflmPrediction): of its component the template's block's reference line (arms of twice
      // its sides, the row's last unit up to sx samples beyond) and the template with the corner (the chroma offset's
      // sample), of luma the area's downsampling (with the luma offset's sample)
      if( lmFusionType( t ) > 1 )
      {
        const int sx = std::min( t.x, CFLM_TEMPLATE ), sy = std::min( t.y, CFLM_TEMPLATE );
        if( sx > 0 || sy > 0 )
        {
          const int x0 = t.x - sx - 1, y0 = t.y - sy - 1;
          addArea( t.comp, x0, y0, x0 + 2 * ( t.w + sx ) + sx, y0 );
          addArea( t.comp, x0, y0, x0, y0 + 2 * ( t.h + sy ) );
          addArea( t.comp, t.x - sx, t.y - sy, t.x + t.w - 1, t.y - 1 );
          addArea( t.comp, t.x - sx, t.y, t.x - 1, t.y + t.h - 1 );
        }
        addArea( COMPONENT_Y, 2 * ( t.x - sx ) - 1, 2 * ( t.y - sy ) - 1, 2 * ( t.x + t.w ) - 1,
                 2 * ( t.y + t.h ) - 1 );
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
      else if( t.derive & DERIVE_MTSS )   // a MIP block: MTSS's mode list, from the CU's node
      {
        addBlock( m.cus[t.cu].node );
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
  const int32_t* gDimdDl   = nullptr;
  const int32_t* gCcpDl    = nullptr;
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
  P.trTables    = m.sh->trTables;
  P.pdpTable    = m.sh->pdpTable;
  P.ctuSize     = (int32_t) cs.sps->getCTUSize();
  P.lumaUnitW   = (int32_t) cs.pcv->minCUWidth;
  P.lumaUnitH   = (int32_t) cs.pcv->minCUHeight;
  // CCLM's luma downsampling (xGetLumaRecPixels, JVET-AN0090): the mean of 2 x 2 samples without collocated chroma, the
  // 5-tap cross with it both ways, else 1 2 1 over two rows (intra_cclm.glsl's CCLM_FILTER_*)
  const bool cclmVer = cs.sps->getCclmVerCollocatedChromaFlag(), cclmHor = cs.sps->getCclmHorCollocatedChromaFlag();
  P.cclmFilter  = !cclmVer && !cclmHor ? 0 : cclmVer && cclmHor ? 1 : 2;
  // CCCM: in a dual tree its luma reaches right of / below the block (xCccmCreateLumaRef); LBCCP's second set of models
  // (predIntraCCCM: an intra slice with the TM non-inter tools; supported() takes intra slices only)
  P.dualTree  = CS::isDualITree( cs ) ? 1 : 0;
#if GPU_CCCM
  P.cccmLbccp = cs.sps->getTMnoninterToolsEnableFlag() ? 1 : 0;
#else
  P.cccmLbccp = 0;
#endif
#if JVET_AG0154_DECODER_DERIVED_CCP_FUSION && JVET_AN0168_REGRESSION_CCP_FUSION
  // decoder-derived CCP merge fusion's pairs of candidates (PU::getMaxNumCcpFusionModel: fewer without the SPS's
  // ccpFusionMode)
  P.ccpFusionPairs = cs.sps->getCcpFusionMode() ? MAX_CCP_FUSION_NUM : MAX_CCP_FUSION_NUM - 4;
#endif
  // the CU records, then their modes (the derived ones -1 until the CU's node writes them), their DIMD histograms, the
  // OBIC CUs' neighbours, the intra merge CUs' and the TIMD CUs' records (uploaded), then their DIMD (an intra merge
  // CU's merge parameters), the OBIC CUs' OBIC, the TIMD CUs' TIMD, the TMRL CUs' choice and the intra merge CUs'
  // template costs (written by their nodes); with verification the parameters intra_rec derives
  const size_t cusBytes  = m.cus.size() * sizeof( Cu ), modeBytes = m.cuMode.size() * 4;
  const size_t callBytes = m.dimdCalls.size() * sizeof( DimdCall ), nbrBytes = m.obicNbrs.size() * sizeof( ObicNbr );
  const size_t mergeBytes = m.mergeData.size() * 4, timdBytes = m.timdData.size() * 4;
  const size_t ccpDataBytes = ccpData.size() * 4;
  const size_t cuBytes   = cusBytes + modeBytes + callBytes + nbrBytes + mergeBytes + timdBytes + ccpDataBytes;
  const size_t dimdOutBytes   = m.cus.size() * DIMD_OUT * 4, obicOutBytes = m.cpuObic.size() * 4;
  const size_t timdOutBytes   = size_t( m.numTimd ) * TIMD_OUT * 4, tmrlOutBytes = m.cpuTmrl.size() * 4;
  const size_t nodeOutEnd     = cuBytes + dimdOutBytes + obicOutBytes + timdOutBytes + tmrlOutBytes;
  const size_t costOffset     = ( nodeOutEnd + 7 ) & ~size_t( 7 );   // 64-bit aligned
  const size_t costBytes      = size_t( m.numMerge ) * MERGE_COST * 8;
  // then the CUs' CCP models, written by their blocks (the PUs', compared in verification, then the CCP merge CUs'
  // default candidates)
  const size_t ccpOffset      = costOffset + costBytes, ccpBytes = size_t( m.numCcp ) * CCP_OUT * 4;
  const size_t ccpAllBytes    = ccpBytes + size_t( m.numCcpMerge ) * defaultSets * CCP_DEFAULTS * CCP_OUT * 4;
  const size_t dbgParamsBytes = m.on ? 4 : n * DBG_PARAMS * 4;
  m.cuBuf.ensure( ctx, std::max<size_t>( ccpOffset + ccpAllBytes, 4 ), gpu::Buffer::DEVICE );
  m.dbgParams.ensure( ctx, dbgParamsBytes, gpu::Buffer::DEVICE );
  P.cus       = m.cuBuf.addr;
  P.cuMode    = m.cuBuf.addr + cusBytes;
  P.dimdCalls = m.cuBuf.addr + cusBytes + modeBytes;
  P.obicNbrs  = m.cuBuf.addr + cusBytes + modeBytes + callBytes;
  P.mergeData = m.cuBuf.addr + cusBytes + modeBytes + callBytes + nbrBytes;
  P.timdData  = m.cuBuf.addr + cusBytes + modeBytes + callBytes + nbrBytes + mergeBytes;
  P.ccpData   = m.cuBuf.addr + cusBytes + modeBytes + callBytes + nbrBytes + mergeBytes + timdBytes;
  P.dimdOut   = m.cuBuf.addr + cuBytes;
  P.obicOut   = m.cuBuf.addr + cuBytes + dimdOutBytes;
  P.timdOut   = m.cuBuf.addr + cuBytes + dimdOutBytes + obicOutBytes;
  P.tmrlOut   = m.cuBuf.addr + cuBytes + dimdOutBytes + obicOutBytes + timdOutBytes;
  P.mergeCost = m.cuBuf.addr + costOffset;
  P.ccpOut    = m.cuBuf.addr + ccpOffset;
  P.dbgParams = m.dbgParams.addr;
  std::vector<int32_t> cuData( cuBytes / 4 );
  if( cuBytes )
  {
    int32_t* d = cuData.data();
    memcpy( d, m.cus.data(), cusBytes );
    memcpy( d + cusBytes / 4, m.cuMode.data(), modeBytes );
    memcpy( d + ( cusBytes + modeBytes ) / 4, m.dimdCalls.data(), callBytes );
    memcpy( d + ( cusBytes + modeBytes + callBytes ) / 4, m.obicNbrs.data(), nbrBytes );
    memcpy( d + ( cusBytes + modeBytes + callBytes + nbrBytes ) / 4, m.mergeData.data(), mergeBytes );
    memcpy( d + ( cusBytes + modeBytes + callBytes + nbrBytes + mergeBytes ) / 4, m.timdData.data(), timdBytes );
    memcpy( d + ( cusBytes + modeBytes + callBytes + nbrBytes + mergeBytes + timdBytes ) / 4, ccpData.data(),
            ccpDataBytes );
  }
  // the intra_rec pipeline with the features the picture uses (OBIC, intra merge, MTSS and the DIMD of MIP and CCLM
  // predictions are part of DIMD: their code, mipTransformMode's and cclmTransformMode's too, is in the pipelines with
  // WITH_DIMD only)
  bool dimd = false, derivedTransforms = false, cclm = false, cccm = false, cflm = false, ccpMerge = false;
  for( const Cu& c: m.cus )
  {
    dimd = dimd || c.dimd >= 0 || c.obic >= 0 || c.merge >= 0 || c.mtss >= 0;
  }
  for( const Tu& t: m.rec )
  {
    dimd              = dimd || ( t.derive & ( DERIVE_DIMD | DERIVE_OBIC | DERIVE_MERGE | DERIVE_MIP_DIMD | DERIVE_CCLM_DIMD ) );
    derivedTransforms = derivedTransforms || ( t.derive & DERIVE_TRANSFORM );
    cclm              = cclm || ( t.lm & LM_ACTIVE );
    cccm              = cccm || ( ( t.lm >> LM_CCCM_SHIFT ) & 3 );   // (LM chroma fusion's first type too)
    cflm              = cflm || lmFusionType( t ) > 1;
    ccpMerge          = ccpMerge || t.ccpMerge >= 0;
  }
  const IntraRecVariant variant = !m.cpuTmrl.empty() ? VARIANT_TMRL : m.numTimdMerge > 0 ? VARIANT_TIMD_MERGE
                                  : m.numTimdSad > 0 ? VARIANT_TIMD_SAD : m.numTimd > 0 ? VARIANT_TIMD
                                  : m.numMerge > 0 ? VARIANT_MERGE : dimd ? VARIANT_FULL
                                  : derivedTransforms ? VARIANT_DERIVED_TRANSFORMS : VARIANT_PLAIN;
  const LmLevel lmLevel = m.numCcpLfnst > 0 ? LEVEL_CCP_LFNST
                          : m.numDerivedCcp > 0 || m.numDdMergeFusion > 0 || m.numFusionBlend > 0 ? LEVEL_CCP_DERIVED
                          : ccpMerge ? LEVEL_CCP_MERGE : cflm ? LEVEL_CFLM : cccm ? LEVEL_CCCM : cclm ? LEVEL_CCLM
                          : LEVEL_NO_LM;
  gpu::Kernel& recKernel = intraRecKernel( *m.sh, ctx, variant, lmLevel );   // (before the stream begins)
  // the workgroups' scratch: per intra_rec workgroup the sums of DIMD / OBIC blending (three classes of a block of up
  // to 128 x 128), then per workgroup the largest CCCM block's luma buffers and prediction beyond shared memory
  const int blendWords = m.blendBlocks ? m.numGroups * BLEND_ACC : 0;
  const int cccmWords  = m.numGroups * m.cccmScratchWords;
  m.scratch.ensure( ctx, size_t( std::max( blendWords + cccmWords, 1 ) ) * 4, gpu::Buffer::DEVICE );
  P.scratch           = m.scratch.addr;
  P.cccmScratchOffset = blendWords;
  P.cccmScratchWords  = m.cccmScratchWords;
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
  // ECM_GPU_DUMP=<prefix>: the records each picture uploads, into <prefix>_<POC>_ver.bin / _on.bin. With the
  // reconstruction on, the CPU derives modes from samples it has not reconstructed: a field the host takes from them
  // for a mode the GPU derives is wrong there and right in verification, which therefore cannot see it;
  // perf/gpu/dumpdiff.py compares the two runs' records without the fields intra_rec derives
  if( getenv( "ECM_GPU_DUMP" ) )
  {
    const std::string name = std::string( getenv( "ECM_GPU_DUMP" ) ) + "_" + std::to_string( m.poc ) + ( m.on ? "_on" : "_ver" ) + ".bin";
    FILE* f = fopen( name.c_str(), "wb" );
    CHECK( !f, "ECM_GPU_DUMP: cannot write " << name );
    auto put = [&]( const void* p, size_t bytes ) { const uint64_t b = bytes; fwrite( &b, 8, 1, f ); fwrite( p, 1, bytes, f ); };
    put( m.rec.data(), tuBytes );
    put( m.coef.data(), coefBytes );
    put( m.signPred.data(), spBytes );
    put( cuData.data(), cuBytes );
    put( ccpData.data(), ccpDataBytes );   // (also in the CU data, at its end)
    fclose( f );
  }
  // ECM_GPU_DRY, in verification: the records built (and dumped) but not reconstructed on the GPU (the picture counts
  // as skipped; the CPU's reconstruction stays): the host's data can be examined where the GPU would fault
  if( getenv( "ECM_GPU_DRY" ) && !m.on )
  {
    m.numSkipped++;
    return;
  }

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

    // (partial verification: the planes, after the CU records, then the CCP records)
    size_t upPlane[3] = {}, planesUp = 0;
    for( int c = 0; c < numComp && m.partial; c++ )
    {
      upPlane[c] = sizeof( Params ) + tuBytes + coefBytes + lmcsBytes + spBytes + cuBytes + planesUp;
      planesUp  += ( planeBytes[c] + 3 ) & ~size_t( 3 );
    }
    const size_t upCcp   = sizeof( Params ) + tuBytes + coefBytes + lmcsBytes + spBytes + cuBytes + planesUp;
    const size_t ccpUp   = m.partial ? ccpBytes : 0;
    const size_t upBytes = upCcp + ccpUp;
    const size_t upSign  = sizeof( Params ) + tuBytes + coefBytes + lmcsBytes;   // where the signPred data goes
    const size_t upCu    = upSign + spBytes;                                      // the CU records and modes
    m.upload.ensure( ctx, upBytes, gpu::Buffer::UPLOAD );
    memcpy( m.upload.mapped, &P, sizeof( Params ) );
    memcpy( (char*) m.upload.mapped + sizeof( Params ), m.rec.data(), tuBytes );
    memcpy( (char*) m.upload.mapped + sizeof( Params ) + tuBytes, m.coef.data(), coefBytes );
    memcpy( (char*) m.upload.mapped + sizeof( Params ) + tuBytes + coefBytes, lmcsTable.data(), lmcsBytes );
    memcpy( (char*) m.upload.mapped + upSign, m.signPred.data(), spBytes );
    memcpy( (char*) m.upload.mapped + upCu, cuData.data(), cuBytes );
    // partial verification: the planes start as the CPU's reconstruction (the CUs taken from it, the inter, IBC and
    // palette CUs) but 0 in the GPU's blocks, so that a block the GPU leaves unwritten shows
    for( int c = 0; c < numComp && m.partial; c++ )
    {
      int16_t*      plane = (int16_t*) ( (char*) m.upload.mapped + upPlane[c] );
      const CPelBuf reco  = cs.picture->getRecoBuf( ComponentID( c ) );
      for( int y = 0; y < P.height[c]; y++ )
      {
        memcpy( plane + size_t( y ) * P.width[c], reco.bufAt( 0, y ), P.width[c] * sizeof( Pel ) );
      }
      for( const Tu& t: m.rec )
      {
        if( t.comp != c || ( t.derive & DERIVE_NODE ) )
        {
          continue;
        }
        for( int y = t.y; y < t.y + t.h; y++ )
        {
          std::fill_n( plane + size_t( y ) * P.width[c] + t.x, t.w, int16_t( 0 ) );
        }
      }
    }
    // and the CCP records of the CUs taken from the CPU, their curCand (the GPU's blocks write the others)
    if( m.partial )
    {
      int32_t* ccpWords = (int32_t*) ( (char*) m.upload.mapped + upCcp );
      std::fill_n( ccpWords, ccpBytes / 4, 0 );
#if JVET_AD0188_CCP_MERGE
      for( const auto& r: m.cpuCcp )
      {
        cpuCcpWords( *r.cand, r.merge, ccpWords + size_t( r.record ) * CCP_OUT );
      }
#endif
    }
    // downloaded: the dependency count (overflow check), with verification the CPU comparison data, the GPU's
    // scheduling data and the derived parameters, then the planes
    size_t downBytes = 8;
    const size_t dPred = downBytes, dResi = dPred + dbgBytes, dTus = dResi + dbgBytes, dOffset = dTus + tuBytes;
    const size_t dList = dOffset + ( n + 1 ) * 4, dParams = dList + cap * 4, dDimd = dParams + dbgParamsBytes;
    const size_t nodeOutBytes = costOffset + costBytes - cuBytes;   // DimdOut, ObicOut, the merge costs
    const size_t dCcp = dDimd + nodeOutBytes;                        // the CCP models
    if( !m.on )
    {
      downBytes = dCcp + ccpBytes;
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
    if( ccpUp )
    {
      s.copy( m.upload, upCcp, m.cuBuf, ccpOffset, ccpUp );
    }
    for( int c = 0; c < numComp; c++ )
    {
      s.fill( m.owner[c], 0, ownerBytes[c], 0xFFFFFFFFu );
      if( m.partial )
      {
        s.copy( m.upload, upPlane[c], m.planes[c], 0, ( planeBytes[c] + 3 ) & ~size_t( 3 ) );
      }
      else
      {
        s.fill( m.planes[c], 0, ( planeBytes[c] + 3 ) & ~size_t( 3 ), 0 );
      }
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
    s.dispatch( recKernel, &push, (uint32_t) std::min<size_t>( m.numGroups, n ) );
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
      if( dimdOutBytes )   // the CUs' DIMD, the OBIC CUs' OBIC, the intra merge CUs' template costs
      {
        s.copy( m.cuBuf, cuBytes, m.download, dDimd, nodeOutBytes );
      }
      if( ccpBytes )
      {
        s.copy( m.cuBuf, ccpOffset, m.download, dCcp, ccpBytes );
      }
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
    gDimdDl   = (const int32_t*) ( (const char*) m.download.mapped + dDimd );
    gCcpDl    = (const int32_t*) ( (const char*) m.download.mapped + dCcp );
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
  // the template costs of the intra merge CUs' nodes against the CPU's, where it ranked by them (before their choice)
  const char* gCostDl = (const char*) gDimdDl + ( costOffset - cuBytes );
  for( size_t u = 0; u < m.cus.size() && first.empty() && GPU_DIMD; u++ )
  {
    const Cu& c = m.cus[u];
    const int idx = c.merge >= 0 ? m.mergeData[c.merge] : -1;   // MergeHead::index
    if( idx < 0 || !m.cpuMergeCostValid[idx] )
    {
      continue;
    }
    for( int k = 0; k < MERGE_COST; k++ )
    {
      uint64_t g;
      memcpy( &g, gCostDl + ( size_t( idx ) * MERGE_COST + k ) * 8, 8 );
      const uint64_t h = m.cpuMergeCost[size_t( idx ) * MERGE_COST + k];
      if( g != h )
      {
        char buf[200];
        snprintf( buf, sizeof( buf ), "template cost of mode %d of CU %d (intra merge at %d,%d %dx%d): CPU %llu GPU %llu", k,
                  (int) u, c.x, c.y, c.w, c.h, (unsigned long long) h, (unsigned long long) g );
        first = buf;
        break;
      }
    }
  }
  // the DIMD the nodes derived against the CPU's: modes, blending, weights and location dependence (an intra merge
  // CU's merge parameters), the second strongest mode of a luma CU's own DIMD, with MTSS its mode list and choice
  for( size_t u = 0; u < m.cus.size() && first.empty() && GPU_DIMD; u++ )
  {
    const Cu&  c       = m.cus[u];
    const bool mipDimd = m.cuMipDimd[u] != 0;   // a MIP CU: its block's modes of its prediction (DERIVE_MIP_DIMD)
    const bool params  = c.dimd >= 0 || c.merge >= 0;   // DIMD_OUT_PARAMS (a TIMD merge CU has MTSS's fields only)
    const bool chroma  = c.kind == CU_CHROMA_DIMD || c.kind == CU_CHROMA_TMRL;   // its DIMD chroma's two modes
    if( !params && c.mtss < 0 && !mipDimd )
    {
      continue;
    }
    const int32_t* g = gDimdDl + u * DIMD_OUT;
    const int32_t* h = m.cpuDimd.data() + u * DIMD_OUT;
    for( int k = 0; k < DIMD_OUT; k++ )
    {
      const bool compared = mipDimd ? k == 0 || k == DIMD_OUT_TR_SECOND || ( k >= DIMD_OUT_MTSS_LIST && c.mtss >= 0 )
                            : k < DIMD_OUT_PARAMS ? params
                            : GPU_DIMD_TRANSFORMS && ( k == DIMD_OUT_TR_SECOND ? c.dimd >= 0 && !chroma : c.mtss >= 0 );
      if( compared && g[k] != h[k] )
      {
        const bool  pnnDimd = m.cuMipDimd[u] == 2;
        const char* field = pnnDimd && k == 0 ? "indicesRepresentationPnn" : pnnDimd ? "indicesRepresentationPnn[1]"
                            : mipDimd && k == 0 ? "mipDimdMode"
                            : mipDimd && k == DIMD_OUT_TR_SECOND ? "dimdDerivedIntraDir2nd"
                            : chroma ? ( k == 0 ? "dimdChromaMode" : "dimdChromaModeSecond" )
                            : k == 0 ? "dimdMode" : k == 1 ? "dimdBlending" : k < 9 ? "dimdBlendMode" : k < 17 ? "dimdRelWeight"
                            : k < DIMD_OUT_TR_SECOND ? "dimdLocDep" : k == DIMD_OUT_TR_SECOND ? "candModeListForTransform"
                            : k < DIMD_OUT_MTSS_MODE ? "candModeListForTransformMtss" : k == DIMD_OUT_MTSS_MODE ? "MTSS's LFNST / NSPT mode"
                            : "MTSS's secondBucket";
        const int   entry = k < 2 ? 0 : k < 9 ? k - 2 : k < 17 ? k - 9 : k < DIMD_OUT_TR_SECOND ? k - 17
                            : k == DIMD_OUT_TR_SECOND ? 1 : k < DIMD_OUT_MTSS_MODE ? k - DIMD_OUT_MTSS_LIST : 0;
        char buf[200];
        snprintf( buf, sizeof( buf ), "%s[%d] of CU %d (%s at %d,%d %dx%d): CPU %d GPU %d", field, entry, (int) u,
                  pnnDimd ? "NN" : mipDimd ? "MIP" : cuKindName( c.kind ), c.x, c.y, c.w, c.h, h[k], g[k] );
        first = buf;
        break;
      }
      if( chroma && k == 1 )   // dimdChromaMode and dimdChromaModeSecond
      {
        break;
      }
    }
  }
  // the OBIC the nodes derived against the CPU's: modes, weights and location dependence
  const int32_t* gObicDl = gDimdDl + m.cus.size() * DIMD_OUT;
  for( size_t u = 0; u < m.cus.size() && first.empty() && GPU_DIMD; u++ )
  {
    const Cu& c = m.cus[u];
    if( c.obic < 0 )
    {
      continue;
    }
    const int32_t* g = gObicDl + size_t( c.obic ) * OBIC_OUT;
    const int32_t* h = m.cpuObic.data() + size_t( c.obic ) * OBIC_OUT;
    for( int k = 0; k < OBIC_OUT; k++ )
    {
      if( g[k] != h[k] )
      {
        const char* field = k < OBIC_MODES ? "obicMode" : k < 2 * OBIC_MODES ? "obicFusionWeight" : "obicLocDep";
        char buf[200];
        snprintf( buf, sizeof( buf ), "%s[%d] of CU %d (OBIC at %d,%d %dx%d): CPU %d GPU %d", field, k % OBIC_MODES,
                  (int) u, c.x, c.y, c.w, c.h, h[k], g[k] );
        first = buf;
        break;
      }
    }
  }
  // the TIMD the nodes derived against the CPU's: every field deriveTimdMode sets (and those it leaves), of a TIMD-SAD
  // CU also deriveTimdModeSad's, of a TIMD merge CU its candidate's; with TIMD merge in the picture the transform types
  const int32_t* gTimdDl = gObicDl + m.cpuObic.size();
  for( size_t u = 0; u < m.cus.size() && first.empty() && GPU_TIMD; u++ )
  {
    const Cu& c = m.cus[u];
    if( c.timd < 0 )
    {
      continue;
    }
    const TimdHead& th  = *reinterpret_cast<const TimdHead*>( &m.timdData[c.timd] );
    const int32_t*  g   = gTimdDl + size_t( th.index ) * TIMD_OUT;
    const int32_t*  h   = m.cpuTimd.data() + size_t( th.index ) * TIMD_OUT;
    for( int k = 0; k < TIMD_OUT; k++ )
    {
      const bool compared = k < TIMD_OUT_SAD || ( k < TIMD_OUT_TR_HOR ? th.sad != 0 : m.numTimdMerge > 0 );
      if( compared && g[k] != h[k] )
      {
        static const char* const fields[TIMD_OUT_FIELDS] = { "timdMode",         "timdModeSecondary", "timdModeNonAng",
                                                          "timdIsBlended",    "timdFusionWeight[0]", "timdFusionWeight[1]",
                                                          "timdFusionWeight[2]", "timdLocDep[0]",  "timdLocDep[1]",
                                                          "timdLocDep[2]",    "timdModeCheckWA",   "timdModeSecondaryCheckWA" };
        const char* field = k == TIMD_OUT_TR_HOR ? "timdmTrType[0]" : k == TIMD_OUT_TR_VER ? "timdmTrType[1]"
                            : fields[k % TIMD_OUT_FIELDS];
        char buf[200];
        snprintf( buf, sizeof( buf ), "%s%s of CU %d (TIMD%s at %d,%d %dx%d): CPU %d GPU %d", field,
                  k >= TIMD_OUT_SAD && k < TIMD_OUT_TR_HOR ? " (Sad)" : "", (int) u, th.sad ? "-SAD" : th.merge ? " merge" : "",
                  c.x, c.y, c.w, c.h, h[k], g[k] );
        first = buf;
        break;
      }
    }
  }
  // the TMRL the nodes chose against the CPU's: the mode and the reference line (m_tmrlList[tmrlListIdx])
  const int32_t* gTmrlDl = gTimdDl + size_t( m.numTimd ) * TIMD_OUT;
  for( size_t u = 0; u < m.cus.size() && first.empty() && GPU_TMRL; u++ )
  {
    const Cu& c = m.cus[u];
    if( c.tmrl < 0 )
    {
      continue;
    }
    const int32_t* g = gTmrlDl + size_t( c.tmrl ) * TMRL_OUT;
    const int32_t* h = m.cpuTmrl.data() + size_t( c.tmrl ) * TMRL_OUT;
    for( int k = 0; k < TMRL_OUT; k++ )
    {
      if( g[k] != h[k] )
      {
        const bool chroma = c.kind == CU_CHROMA_TMRL;
        char buf[200];
        snprintf( buf, sizeof( buf ), "%s of CU %d (%s at %d,%d %dx%d, list index %d): CPU %d GPU %d",
                  k == TMRL_OUT_MODE ? ( chroma ? "intraDir[1]" : "intraDir" ) : chroma ? "chromaMrlIdx" : "multiRefIdx",
                  (int) u, chroma ? "chroma TMRL" : "TMRL", c.x, c.y, c.w, c.h, c.ipredIdx, h[k], g[k] );
        first = buf;
        break;
      }
    }
  }
#if JVET_AD0188_CCP_MERGE
  // the CCP models against the CPU's curCand after the picture (with CU::saveCcInsideFilterFlagInCCP's flag): the
  // fields of their type that CCPModelCandidate::operator== and predCCPCandidate read (CCLM's and GLM's patterns: per
  // component a, b and the shift, with two models also the second's and the threshold; GLM with luma's: its parameters
  // and the luma offset; CCCM's: the parameters, the template chroma's ranges, the luma offset, the corner offsets, the
  // filters, the parameters' number, with two models also the second's and the threshold) and the inside filter flag
  std::string firstCcp;   // (partial verification reports it beside any earlier check's)
  for( int r = 0; r < m.numCcp && ( first.empty() || m.partial ) && firstCcp.empty(); r++ )
  {
    if( !m.ccpPu[r] )   // (partial verification: a non-intra CU's TU's, from the CPU)
    {
      continue;
    }
    const int32_t*           g = gCcpDl + size_t( r ) * CCP_OUT;
    const CCPModelCandidate& h = m.ccpPu[r]->curCand;
    const char*              field = nullptr;
    long long                cpu = 0, gpu = 0;
    auto check = [&]( const char* name, long long a, long long b )
    {
      if( !field && a != b )
      {
        field = name;
        cpu   = a;
        gpu   = b;
      }
    };
    auto param = [&]( bool second, int c, int p )
    {
      const int o = ( second ? CCP_OUT_PARAMS2 : CCP_OUT_PARAMS ) + 2 * ( c * CCP_MAX_PARAMS + p );
      return (long long) ( uint64_t( uint32_t( g[o] ) ) | ( uint64_t( uint32_t( g[o + 1] ) ) << 32 ) );
    };
    const bool two = ( h.type & CCP_TYPE_MMLM ) != 0;
    check( "type", h.type, g[CCP_OUT_TYPE] );
    check( "ccInsideFilter", h.ccInsideFilter, g[CCP_OUT_INSIDE] );
    if( h.type & ( CCP_TYPE_CCLM | CCP_TYPE_GLM0123 ) )
    {
      check( "glmIdc", h.glmIdc, g[CCP_OUT_GLM_IDC] );
      for( int c = 0; c < 2; c++ )
      {
        check( "a", h.params[c][0], param( false, c, 0 ) );
        check( "b", h.params[c][1], param( false, c, 1 ) );
        check( "shift", h.shift[c], g[CCP_OUT_SHIFT + c] );
        if( two )
        {
          check( "a2", h.params2[c][0], param( true, c, 0 ) );
          check( "b2", h.params2[c][1], param( true, c, 1 ) );
          check( "shift2", h.shift2[c], g[CCP_OUT_SHIFT2 + c] );
        }
      }
    }
    else if( h.type & CCP_TYPE_GLM4567 )
    {
      check( "glmIdc", h.glmIdc, g[CCP_OUT_GLM_IDC] );
      check( "lumaOffset", h.lumaOffset, g[CCP_OUT_LUMA_OFFSET] );
      for( int c = 0; c < 2; c++ )
      {
        for( int p = 0; p < GLM_NUM_PARAMS; p++ )
        {
          check( "params", h.params[c][p], param( false, c, p ) );
        }
      }
    }
    else
    {
      const int ranges[2][2][3] = { { { h.rangeMinCb, h.rangeMaxCb, h.rangeChangeCb },
                                      { h.rangeMinCr, h.rangeMaxCr, h.rangeChangeCr } },
                                    { { h.rangeMinCb2, h.rangeMaxCb2, h.rangeChangeCb2 },
                                      { h.rangeMinCr2, h.rangeMaxCr2, h.rangeChangeCr2 } } };
      check( "lumaOffset", h.lumaOffset, g[CCP_OUT_LUMA_OFFSET] );
      check( "corOffX", h.corOffX, g[CCP_OUT_COR_OFF] );
      check( "corOffY", h.corOffY, g[CCP_OUT_COR_OFF + 1] );
      check( "cccmMultiFilterIdx", h.cccmMultiFilterIdx, g[CCP_OUT_MDF] );
      check( "numParams", h.numParams, g[CCP_OUT_NUM_PARAMS] );
      for( int k = 0; k < ( two ? 2 : 1 ); k++ )
      {
        for( int c = 0; c < 2; c++ )
        {
          for( int i = 0; i < 3; i++ )
          {
            check( "range", ranges[k][c][i], g[CCP_OUT_RANGE + 3 * ( 2 * k + c ) + i] );
          }
          for( int p = 0; p < std::min( h.numParams, CCP_MAX_PARAMS ); p++ )
          {
            check( k ? "params2" : "params", k ? h.params2[c][p] : h.params[c][p], param( k != 0, c, p ) );
          }
        }
      }
    }
    if( two )
    {
      check( "yThres", h.yThres, g[CCP_OUT_Y_THRES] );
    }
    if( field )
    {
      int b = 0;   // (the record's first block)
      while( b < numTus - 1 && m.rec[b].ccp != r )
      {
        b++;
      }
      char buf[200];
      snprintf( buf, sizeof( buf ),
                "CCP model's %s of record %d (block %d at %d,%d %dx%d, type %d): CPU %lld GPU %lld", field, r, b,
                m.rec[b].x, m.rec[b].y, m.rec[b].w, m.rec[b].h, h.type, cpu, gpu );
      firstCcp = buf;
    }
  }
  if( first.empty() )
  {
    first = firstCcp;
  }
  // the chroma PUs without a record have no CCP model either (else a merge CU next to them would inherit one the GPU
  // does not have)
  for( size_t k = 0; k < m.untypedCcpPu.size() && first.empty(); k++ )
  {
    const PredictionUnit& pu = *m.untypedCcpPu[k];
    if( pu.curCand.type != CCP_TYPE_NONE )
    {
      char buf[160];
      snprintf( buf, sizeof( buf ), "the chroma CU at %d,%d %dx%d has a CCP model (type %d) but no record",
                pu.Cb().x, pu.Cb().y, pu.Cb().width, pu.Cb().height, pu.curCand.type );
      first = buf;
    }
  }
#endif
  // the modes and parameters intra_rec derived against the CPU's: every field the kernels read for the mode (the
  // angular ones of angular modes, PDPC's scale for positive angles); of a MIP or CCLM block the transforms only
  for( int i = 0; i < numTus && first.empty(); i++ )
  {
    const Tu& h = m.rec[i];
    if( !( h.derive & ( DERIVE_MODE | DERIVE_MIP_DIMD | DERIVE_CCLM_DIMD ) ) )
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
    // (a CCP merge block of fusion with DIMD chroma: the CPU sets its mode's parameters only inside predCCPCandidate,
    // after the block's record, which holds the LM mode's; the node's DIMD chroma mode is compared with the CPU's)
    if( ( h.derive & DERIVE_MODE ) && !( h.derive & DERIVE_CCP_DIMD ) )
    {
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
    }
    if( h.derive & DERIVE_TRANSFORM )   // the transforms intra_rec chose for the mode
    {
      check( "mtxV", h.mtxV, g[12] );
      check( "mtxH", h.mtxH, g[13] );
      check( "secMatrix", h.secMatrix, g[14] );
      check( "secTranspose", h.secTranspose, g[15] );
      check( "secLayout", h.secLayout, g[16] );
    }
    if( field )
    {
      char buf[200];
      snprintf( buf, sizeof( buf ), "derived %s of block %d (comp %d at %d,%d %dx%d): CPU %d GPU %d", field, i, h.comp, h.x,
                h.y, h.w, h.h, cpu, gpu );
      first = buf;
    }
  }
  // (the first block whose prediction or residual differs; partial verification reports it beside any earlier check's)
  std::string firstBlock;
  for( size_t i = 0; i < m.rec.size() && ( first.empty() || m.partial ) && firstBlock.empty(); i++ )
  {
    const Tu& t = m.rec[i];
    for( int k = 0; k < t.w * t.h && !( t.derive & DERIVE_NODE ); k++ )
    {
      const int o = t.dbgOffset + k;
      const bool predBad = gPred[o] != m.cpuPred[o], resiBad = gResi[o] != m.cpuResi[o];
      if( predBad || resiBad )
      {
        char buf[320];
        snprintf( buf, sizeof( buf ),
                  "block %d (comp %d at %d,%d %dx%d, %s CU, mode %d, angle %d, ver %d, cubic %d, cbf %d, lm 0x%x, derive 0x%x, "
                  "ccp merge %d): %s at (%d,%d) CPU %d GPU %d",
                  (int) i, t.comp, t.x, t.y, t.w, t.h, t.cu >= 0 ? cuKindName( m.cus[t.cu].kind ) : "no", t.dirMode,
                  t.intraPredAngle, t.isModeVer, t.useCubic, t.cbf, t.lm, t.derive, t.ccpMerge,
                  predBad ? "prediction" : "residual", k % t.w, k / t.w, predBad ? m.cpuPred[o] : m.cpuResi[o],
                  predBad ? gPred[o] : gResi[o] );
        firstBlock = buf;
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
        firstBlock += above + left;
        break;
      }
    }
  }
  if( first.empty() )
  {
    first = firstBlock;
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
  int numFilter = 0, numPdpc = 0, numGrad = 0, numFusion = 0, numBf = 0, numNodes = 0, numDimd = 0, numDimdCus = 0, numObic = 0;
  int numDimdTransforms = 0, numMtss = 0, numMipDimd = 0, numInherited = 0;
  // (the LM modes' counts include CCCM blocks; CCCM's template selections: above alone, left alone; GLM's include GLM
  // with luma, CCCM's not)
  int numCclm = 0, numMmlm = 0, numMdlm = 0, numLbccp = 0, numCclmDimd = 0, numGlm = 0, numGlmLuma = 0, numLmFusion = 0;
  int numCflm = 0, numCcpMerge = 0, numCcpMergeFusion = 0, numCcpMergeDimd = 0, numCcpMergeLfnst = 0;
  int numCcpMergeDimdLfnst = 0;
  int numCccm = 0, numCccmAbove = 0, numCccmLeft = 0, numGlCccm = 0, numMdfCccm = 0, numNoSubCccm = 0;
  for( const Tu& t: m.rec )
  {
    if( t.lm & LM_ACTIVE )
    {
      const bool cccm = ( t.lm >> LM_CCCM_SHIFT ) != 0 && !( t.lm & LM_CCCM_GLM_LUMA );
      numCclm++;
      numMmlm += PU::isMultiModeLM( t.dirMode );
      numMdlm += t.dirMode != LM_CHROMA_IDX && t.dirMode != MMLM_CHROMA_IDX;
      numLbccp += ( t.lm & LM_INSIDE ) != 0;
      numCclmDimd += ( t.derive & DERIVE_CCLM_DIMD ) && ( t.derive & DERIVE_TRANSFORM );
      numGlm += ( ( t.lm >> LM_GLM_SHIFT ) & 15 ) != 0;
      numGlmLuma += ( t.lm & LM_CCCM_GLM_LUMA ) != 0;
      numCccm += cccm;
      numCccmAbove += cccm && ( t.lm >> LM_CCCM_SHIFT ) == 3;
      numCccmLeft += cccm && ( t.lm >> LM_CCCM_SHIFT ) == 2;
      numGlCccm += cccm && ( t.lm & LM_CCCM_GL ) != 0;
      numMdfCccm += cccm && ( ( t.lm >> LM_CCCM_MDF_SHIFT ) & 3 ) != 0;
      numNoSubCccm += cccm && ( t.lm & LM_CCCM_NOSUB ) != 0;
    }
    numLmFusion += lmFusionType( t ) != 0 && t.ccpMerge < 0;   // (a merge fusion block has LM chroma fusion's fields)
    numCflm += lmFusionType( t ) > 1;
    numCcpMerge += t.ccpMerge >= 0 && !( t.lm & LM_ACTIVE );   // (decoder-derived fusion's blocks are LM blocks)
    numCcpMergeFusion += t.ccpMerge >= 0 && lmFusionType( t ) != 0;   // (with CCCM)
    numCcpMergeDimd += ( t.derive & DERIVE_CCP_DIMD ) != 0;
    const bool mergeLfnst = t.ccpMerge >= 0 && !( t.lm & LM_ACTIVE ) && ( t.derive & DERIVE_CCLM_DIMD )
                            && ( t.derive & DERIVE_TRANSFORM );
    numCcpMergeLfnst += mergeLfnst;
    numCcpMergeDimdLfnst += mergeLfnst && ( t.derive & DERIVE_CCP_DIMD );
    numFilter += t.refFilter;
    numPdpc += t.applyPDPC;
    numGrad += t.applyPDPC && t.gradPDPC;
    numFusion += t.fusion;
    numBf += t.bf != 0;
    numNodes += ( t.derive & DERIVE_NODE ) != 0;
    numDimdTransforms += ( t.derive & DERIVE_TRANSFORM ) && ( t.derive & ( DERIVE_DIMD | DERIVE_OBIC | DERIVE_MERGE | DERIVE_TIMD | DERIVE_TMRL ) );
    numMtss += ( t.derive & DERIVE_MTSS ) != 0;
    numMipDimd += ( t.derive & DERIVE_MIP_DIMD ) != 0;
    numInherited += ( t.derive & ( DERIVE_INHERIT_HOR | DERIVE_INHERIT_VER ) ) != 0;
  }
  int numTmrl = 0, numChromaTmrl = 0;
  for( const Cu& c: m.cus )
  {
    numDimd += c.dimd >= 0;
    numDimdCus += c.kind == CU_DIMD || c.kind == CU_CHROMA_DIMD;
    numObic += c.kind == CU_OBIC;
    numTmrl += c.kind == CU_TMRL;
    numChromaTmrl += c.kind == CU_CHROMA_TMRL;
  }
  int numTimdBlended = 0, numTimdSad = 0;
  for( const Cu& c: m.cus )
  {
    if( c.timd >= 0 )   // the blend of the modes it predicts (timdIsBlended, a TIMD-SAD CU's timdIsBlendedSad)
    {
      const TimdHead& th = *reinterpret_cast<const TimdHead*>( &m.timdData[c.timd] );
      numTimdSad += th.sad;
      numTimdBlended += m.cpuTimd[size_t( th.index ) * TIMD_OUT + ( th.sad ? TIMD_OUT_SAD : 0 ) + TIMD_OUT_BLENDED] != 0;
    }
  }
  printf( "\n[gpu-intra] POC %d: %d blocks, chain %d (ref filter %d, PDPC %d, gradient PDPC %d, fusion %d, bilateral filter candidates %d, CU nodes %d, DIMD %d, DIMD CUs %d, OBIC CUs %d, intra merge CUs %d, TIMD CUs %d (blended %d, TIMD-SAD %d, TIMD merge %d, inheriting transforms %d), TMRL CUs %d, chroma TMRL CUs %d, their MTS / LFNST / NSPT blocks %d, MTSS %d, MIP LFNST / NSPT %d, LM blocks %d (MMLM %d, MDLM %d, LBCCP %d, LFNST / NSPT %d, GLM %d (with luma %d), CCCM %d: above %d, left %d, GL %d, several filters %d, without subsampling %d), LM chroma fusion %d (CFLM %d), CCP merge %d (fusion with CCCM %d (by a blend model %d), with DIMD chroma %d, adjusted %d, decoder-derived fusion %d, LFNST / NSPT %d (with DIMD chroma %d)), decoder-derived CCP fusion %d), GPU %.2f ms: %s%s%s", m.poc,
          (int) m.rec.size() - numNodes, maxDepth, numFilter, numPdpc, numGrad, numFusion, numBf, numNodes, numDimd, numDimdCus, numObic, m.numMerge, m.numTimd, numTimdBlended, numTimdSad, m.numTimdMerge, numInherited, numTmrl, numChromaTmrl, numDimdTransforms, numMtss, numMipDimd, numCclm, numMmlm, numMdlm, numLbccp, numCclmDimd, numGlm, numGlmLuma, numCccm, numCccmAbove, numCccmLeft, numGlCccm, numMdfCccm, numNoSubCccm, numLmFusion, numCflm, numCcpMerge, numCcpMergeFusion, m.numFusionBlend, numCcpMergeDimd, m.numCcpAdjust, m.numDdMergeFusion, numCcpMergeLfnst, numCcpMergeDimdLfnst, m.numDerivedCcp, ms, bad ? "MISMATCH, " : "identical",
          bad ? first.c_str() : "", numBad ? ( ", " + std::to_string( numBad ) + " picture samples differ" ).c_str() : "" );
  if( m.partial )
  {
    printf( "; %d CUs from the CPU%s%s%s%s", (int) m.cpuCus.size(),
            firstBlock.empty() || firstBlock == first ? "" : "; first block: ", firstBlock == first ? "" : firstBlock.c_str(),
            firstCcp.empty() || firstCcp == first ? "" : "; first CCP model: ", firstCcp == first ? "" : firstCcp.c_str() );
  }
}
