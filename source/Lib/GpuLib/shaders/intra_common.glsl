// layouts shared by the intra reconstruction kernels and GpuIntraRec.cpp
#ifndef INTRA_COMMON_GLSL
#define INTRA_COMMON_GLSL

// one transform block of one component (103 int32, layout shared with GpuIntraRec.cpp)
struct Tu
{
  int comp, x, y, w, h;
  int dirMode;                 // final intra mode (PLANAR_IDX 0, DC_IDX 1, angular 2..66)
  int isModeVer, intraPredAngle, absInvAngle, useCubic;
  int unitW, unitH;            // availability units (xFillReferenceSamples)
  int bitDepth;
  int cbf, coeffOffset;
  int dqScale, dqRightShift, dqInMin, dqInMax;   // Quant::dequant without scaling lists
  int trMin, trMax, shift1, shift2;              // TrQuant::xIT
  int recMin, recMax;                            // reconstruction clipping (adaptive for luma)
  int dbgOffset;                                 // sample offset of the block in the debug buffers
  int plIdx;                                     // directional planar (luma): 1 horizontal, 2 vertical
  int refFilter, applyPDPC, gradPDPC, angularScale, fusion, disableRefFilter;
  // per reference line (the block's multiRefIdx and the fusion line after it): the corner flag, then the available
  // prefix of the above / above-right / left / below-left units and of the MRL units left of and above the corner
  int avail[2][7];
  int readMax[2][2];   // per reference line: the highest top / left index the prediction reads (a superset)
  int nzRows, nzCols;   // the nonzero coefficient levels lie in rows < nzRows and columns < nzCols
  int bf;               // intra prediction bilateral filter possible: 1 planar / DC, 2 angular (also gated by the references)
  int qp;               // the CU's luma QP
  int lmcs;             // LMCS: bit 0 the bilateral filter works on inverse-mapped samples, bit 1 chroma residual scaling
  int crsX, crsY, crsW, crsH;   // chroma residual scaling: the CU's luma area (its left column and above row are averaged)
  int crsAvail;         // bit 0: there is a luma CU left of the area, bit 1: above it
  int quant;            // bit 0: dependent quantization (the coefficients are quantization indices), bit 1: QCS
  int jccr;             // joint CbCr: the Cb block's ICT mode (+-1..+-3; it computes both residuals), 4 for the Cr block
  int jccrOffset;       // the Cr block's dbgOffset (in the Cb block)
  int mtxV, mtxH;       // the inverse transform matrices (DCT-II or MTS) of the columns (size h) and the rows (size w)
  int transformSkip;    // the dequantized levels are the residual (with BDPCM the host accumulated the levels)
  int mipMatrix;        // MIP: the mode's matrix in the matrices buffer (-1: no MIP; dirMode is then DC)
  int mipTransposed, mipSizeId;
  // the inverse LFNST or NSPT (TrQuant::xInvLfnst, xInvNspt; SEC_*): its matrix (int8, at secMatrix in the kernels
  // buffer) takes secNumIn dequantized coefficients in the scan at secScan (in the matrices buffer, x | y << 16) and
  // gives secRowLength outputs per input; LFNST's go to the positions at secLayout in the top-left secSize x secSize
  // (the coefficients' region), whose dequantized coefficients the others keep, before the primary transform; NSPT's
  // are the residual, row after row (secTranspose: column after column)
  int secKind, secMatrix, secRowLength, secNumIn, secScan, secLayout, secSize, secTranspose;
  int mrl;             // multiRefIdx: the reference line (0, 1, 3, 5, 7, 12; luma)
  int bfAvail[3];       // the bilateral filter's padding: the corner flag, the available above / left units (line 0)
  int isp;              // ISP sub-partition (luma): 1 horizontal split, 2 vertical split (0: none)
  int cuX, cuY, cuW, cuH;   // ISP: the CU (its references, the bilateral filter's padding origin)
  int regX, regW;       // the predicted region: the block, or the 4-wide region of ISP sub-partitions 1 or 2 wide
  int ispSide;          // ISP: the side along the split is available (the CU's arm is shifted, not replaced)
  int ispPrev;          // ISP: the previous sub-partition of the CU (-1: none)
  // sign prediction (TrQuant::predCoeffSigns): the block whose prediction and neighbours decide the signs holds the
  // block with the coefficients (itself, or with joint CbCr whose Cr is coded the Cb block before it; -1: no sign
  // prediction); that block holds its data in the signPred buffer (spOffset, -1: none); with joint CbCr the other block
  // of the pair waits for the decision, which writes its residual (resiFrom, -1: intra_resi does)
  int spCoded;
  int spOffset;
  int resiFrom;
  // the block's CU record (Cu, -1: none) and DERIVE_*: with DERIVE_MODE intra_rec takes the mode its CU's node derived
  // and sets the fields above that depend on it (the host's are then the CPU's, for the comparison only)
  int cu;
  int derive;
  // DERIVE_TRANSFORM (intra_rec's selectTransform): what the transforms of the mode depend on besides it - explicit
  // MTS's kernel pair index + 1 (0: none), LFNST's / NSPT's index (lfnstIdx - 1) and NSPT's bucket
  int mtsSet;
  int secIdx;
  int secBucket;
  // a CCLM block's mark (dirMode LM_CHROMA_IDX ..; intra_cclm.glsl), slope adjustments and LBCCP filter (LM_* fields)
  int lm;
  // a CFLM block's template (intra_cccm.glsl's cflmPrediction): its record in Params::timdData, TemplateUnits followed
  // by its units' availability bits (-1: none)
  int cflm;
  // an LM or LM chroma fusion block's CU's CCP model, which the block writes its part of: its record in Params::ccpOut
  // (CCP_OUT_*; -1: none); a CCP merge block's the model its CU inherits, which its Cb block writes
  int ccp;
  // a CCP merge block (JVET-AD0188, intra_ccp.glsl): its CU's sources of candidates in Params::ccpData (CCP_MERGE_*;
  // -1: not a merge block)
  int ccpMerge;
};
layout( buffer_reference, std430, buffer_reference_align = 4 ) buffer BufTu { Tu v[]; };

// the chroma modes of the LM family (CommonDef.h with MMLM; a CCLM block's Tu::dirMode)
const int LM_CHROMA_IDX   = 67;   // NUM_LUMA_MODE
const int MMLM_CHROMA_IDX = LM_CHROMA_IDX + 1;
const int MDLM_L_IDX      = LM_CHROMA_IDX + 2;
const int MMLM_L_IDX      = LM_CHROMA_IDX + 3;
const int MDLM_T_IDX      = LM_CHROMA_IDX + 4;
const int MMLM_T_IDX      = LM_CHROMA_IDX + 5;

// Tu::lm: whether the block is a CCLM block (its mode is not enough: a luma TIMD / TMRL block derives modes of the
// 131-mode set, whose values from 67 on are angular), the slope adjustment (cclmOffsets, -4 .. 4) of its first and
// second model, each + 8 in 4 bits, and whether the CU's prediction takes LBCCP's inside filter (ccInsideFilter).
// With LFNST / NSPT (DERIVE_CCLM_DIMD) also the predictions whose DIMD gives their mode (deriveChromaIpmForTransform
// by the TU's joint CbCr mode: Cb's for 1, Cr's for 2, both for 0 and 3) and the other component's slope adjustments.
// With GLM (JVET-AA0126) the gradient pattern of the block (glmIdc 1 .. NUM_GLM_PATTERN, 0: none) and the other's.
// A CCCM block (JVET-AA0057, intra_cccm.glsl; no slope adjustments) has its template selection (cccmFlag: 1 above
// and left, 2 left, 3 above; 0: not CCCM) and its kind in the other component's field (with LFNST / NSPT the other
// component's kind is the same): GL-CCCM's gradients and locations (JVET-AC0054), several downsampling filters
// (JVET-AD0202's cccmMultiFilterIdx 1 .. 3), no subsampling (JVET-AC0147's cccmNoSubFlag), GLM with luma (JVET-AB0092:
// glmIdc 5 .. 8 in the GLM field, the template selection its LM mode's: 1 LM_CHROMA, 2 MDLM_L, 3 MDLM_T).
// A block of LM chroma fusion (JVET-AC0119: not an LM block) has LM_FUSION (a bit beyond every LM field: a CCLM block's
// other-component field reaches bit 19) and its type (isChromaFusion) in the second slope adjustment's field (it has no
// slope adjustments): type 1 the template selection 1 of the plain CCCM models it blends in and the weight of its own
// prediction (1 .. 3 of 4) in the first slope adjustment's field; CFLM's types 2 (one model) and 3 (two) Tu::cflm
const int LM_DELTA0_SHIFT = 0;
const int LM_DELTA1_SHIFT = 4;
const int LM_INSIDE       = 256;
const int LM_ACTIVE       = 512;
const int LM_HIST_CB      = 1024;
const int LM_HIST_CR      = 2048;
const int LM_OTHER_SHIFT  = 12;   // the other component's two adjustments, in LM_DELTA0 / 1's layout
const int LM_GLM_SHIFT    = 20;   // 4 bits
const int LM_OTHER_GLM_SHIFT = 24;
const int LM_CCCM_SHIFT   = 28;   // 2 bits
const int LM_CCCM_GL      = 4096;
const int LM_CCCM_MDF_SHIFT = 13;   // 2 bits
const int LM_CCCM_NOSUB   = 32768;
const int LM_CCCM_GLM_LUMA = 65536;
const int LM_FUSION       = 1 << 30;
const int LM_FUSION_W_SHIFT    = 0;   // 2 bits
const int LM_FUSION_TYPE_SHIFT = 4;   // 2 bits

// the type of a block of LM chroma fusion (PredictionUnit::isChromaFusion: 1 the blend with CCCM, 2 / 3 CFLM of one /
// two models), 0 for other blocks
int lmFusionType( Tu t )
{
  return ( t.lm & LM_FUSION ) != 0 ? ( t.lm >> LM_FUSION_TYPE_SHIFT ) & 3 : 0;
}

// CFLM's template: the lines above / columns left of the block (xCflmGetRefBuf; as many as the picture has)
const int CFLM_TEMPLATE = 2;

// the CCP model of an LM CU (JVET-AD0188's CCPModelCandidate: PredictionUnit::curCand, which predIntraChromaLM /
// predIntraCCCM set and CCP merge inherits; also an LM chroma fusion CU's, whose geneChromaFusionPred stores its CCCM
// models) as its blocks compute it (CCP_OUT int32 per record in Params::ccpOut at Tu::ccp; layout shared with
// GpuIntraRec.cpp): the type (CCP_TYPE_*), glmIdc, the luma threshold of two models (yThres), the luma offset, the
// reference area's columns / rows left of / above the block (corOffX, corOffY), the downsampling filters
// (cccmMultiFilterIdx), the parameters' number (numParams), per component the shifts of CCLM's two models, LBCCP's
// inside filter (ccInsideFilter) as the CU's curCand has it after the CU (CU::saveCcInsideFilterFlagInCCP: the CU's
// own flag) and as the CTU row's history holds it (CU::saveModelsInHCCP copies curCand before that: an LM CU's 0), per
// model and component the range of the template's chroma (rangeMin, rangeMax, rangeChange), then per component the
// first model's parameters (64-bit, two words each, low first) and the second's. A CCLM or GLM block writes its
// component's part, a CCCM block (the Cb block: predIntraCCCM predicts both components) both
const int CCP_OUT_TYPE = 0, CCP_OUT_GLM_IDC = 1, CCP_OUT_Y_THRES = 2, CCP_OUT_LUMA_OFFSET = 3, CCP_OUT_COR_OFF = 4;
const int CCP_OUT_MDF = 6, CCP_OUT_NUM_PARAMS = 7, CCP_OUT_SHIFT = 8, CCP_OUT_SHIFT2 = 10, CCP_OUT_INSIDE = 12;
const int CCP_OUT_INSIDE_HIST = 13, CCP_OUT_RANGE = 14;
const int CCP_MAX_PARAMS = 11;   // NUM_CCP_PARAMS
const int CCP_OUT_PARAMS = 26, CCP_OUT_PARAMS2 = CCP_OUT_PARAMS + 4 * CCP_MAX_PARAMS;
const int CCP_OUT = CCP_OUT_PARAMS2 + 4 * CCP_MAX_PARAMS;
// CCPType
const int CCP_TYPE_CCLM = 1, CCP_TYPE_MMLM = 2, CCP_TYPE_GLM0123 = 4, CCP_TYPE_GLM4567 = 8, CCP_TYPE_CCCM = 16;
const int CCP_TYPE_GLCCCM = 32, CCP_TYPE_NSCCCM = 64, CCP_TYPE_MDFCCCM = 128;
// Params::ccpData (int32; layout shared with GpuIntraRec.cpp): from 0 per CCP record its blocks (CCP_WRITERS each: the
// Cb block's, then the Cr block's index; -1: none), which intra_deps makes a merge block wait for; a CCP merge CU's
// entry at Tu::ccpMerge: where its CTU row's history begins in ccpData (the records CU::saveModelsInHCCP pushed into
// cs.ccpLut before the CU, in decoding order), their number, the candidate the bitstream selects (idxNonLocalCCP - 1),
// the first of the 24 records its blocks' default candidates take (after the PUs' records: the Cb block's 12, then
// the Cr block's), the number of spatial sources, then those records (PU::getCCPModelCandidateList's adjacent and
// non-adjacent positions' PUs, decoded before the CU, typed, each once)
const int CCP_WRITERS = 2;
const int CCP_MERGE_HIST = 0, CCP_MERGE_HIST_NUM = 1, CCP_MERGE_INDEX = 2, CCP_MERGE_DEFAULTS = 3;
const int CCP_MERGE_NUM_SPATIAL = 4, CCP_MERGE_SPATIAL = 5;

// Tu::derive
const int DERIVE_NODE     = 32;  // not a block: the derivation of the mode of CU Tu::cu, before the CU's blocks
const int DERIVE_MODE     = 1;   // the mode and its parameters come from the CU's mode (intra_modes.glsl)
const int DERIVE_FETCH2ND = 2;   // IntraPrediction's fetchRef2nd: the second reference line of intra fusion may be read
const int DERIVE_BF       = 4;   // the intra bilateral filter applies unless the mode excludes it
const int DERIVE_NO_PDPC  = 8;   // the SPS disables PDPC (DisablePdpc)
const int DERIVE_NO_SMOOTHING = 16;   // the SPS disables intra smoothing (no reference filter, no Gaussian interpolation)
// the transforms depend on the mode (explicit MTS of MTSExt, LFNST / NSPT): intra_rec chooses their matrices and
// computes the residual itself (intra_resi skips the block, with joint CbCr also the other block's)
const int DERIVE_TRANSFORM = 64;
// a block of a DIMD CU (CU_DIMD): where the CU's DIMD blends, the prediction blends its modes (the node's DimdOut)
const int DERIVE_DIMD = 128;
// a block of an OBIC CU (CU_OBIC): the prediction blends the CU's OBIC modes (the node's OBIC_OUT)
const int DERIVE_OBIC = 256;
// a block of an intra merge CU (CU_MERGE): the prediction blends the chosen candidate's modes (the node's DimdOut slot)
const int DERIVE_MERGE = 512;
// MTSS (JVET-AK0217, mdirIdx): the LFNST / NSPT mode and bucket are those chosen into the CU's DimdOut (DIMD_OUT_MTSS_*)
// by its node, or by a MIP block itself (mipTransformMode)
const int DERIVE_MTSS = 1024;
// a MIP block with LFNST / NSPT: intra_rec derives their mode from a DIMD of its prediction (mipDimdMode), then its
// residual
const int DERIVE_MIP_DIMD = 2048;
// a block of a TIMD CU (CU_TIMD): the prediction of the CU's TIMD modes (the node's TimdOut), blended where they blend
const int DERIVE_TIMD = 4096;
// the SPS disables intra fusion (DisableIntraFusion), also blended TIMD's fusion of close modes
const int DERIVE_NO_FUSION = 8192;
// a block of a TIMD merge CU (with DERIVE_TIMD): where its modes blend, the secondary mode is predicted from the next
// reference line wherever its slope is fractional (initPredIntraParams' TIMD merge branch), not by TIMD's rule
const int DERIVE_TIMD_MERGE = 16384;
// a block of a TIMD merge CU whose rows / columns take the transform type its chosen candidate gives (TrQuant::xIT's
// TIMD merge branch: a side of 4 to 16 takes DST-VII, one above IntraMTSMaxSize DCT-II, which the host sets): the
// type in the CU's TimdOut (TIMD_OUT_TR_HOR / TIMD_OUT_TR_VER), with DERIVE_TRANSFORM
const int DERIVE_INHERIT_HOR = 32768;
const int DERIVE_INHERIT_VER = 65536;
// a block of a TMRL CU (CU_TMRL): its mode (131-mode set, 1/64-sample angles) and reference line come from the CU's node
// (TmrlOut); the availability of that line, unknown to intra_deps, is found by the block itself (refLineAvailability)
const int DERIVE_TMRL = 131072;
// a CCLM block with LFNST / NSPT (JVET-AK0064): intra_rec derives their mode from a DIMD of the CU's chroma predictions
// (Tu::lm's LM_HIST_*; cclmTransformMode), then its residual
const int DERIVE_CCLM_DIMD = 262144;

// Params::dbgParams per block: the derived dirMode, isModeVer, intraPredAngle, absInvAngle, useCubic, refFilter,
// applyPDPC, gradPDPC, angularScale, fusion, bf, plIdx, then mtxV, mtxH, secMatrix, secTranspose, secLayout
const int DBG_PARAMS = 17;

// the positions of getSpatialIpm (the MPM list's neighbours): left, above, above-left, above-right, below-left, then
// the non-adjacent ones
const int MPM_NEIGHBOURS = 23;

// a coding unit of one channel type whose blocks were recorded (47 int32, layout shared with GpuIntraRec.cpp)
struct Cu
{
  int kind;              // CU_*: how its intra mode is found
  int node;              // the record (DERIVE_NODE) that derives the mode and publishes it (Params::cuMode); -1 CU_KNOWN,
                         // but a MIP CU with MTSS (CU_KNOWN), whose node derives only MTSS's mode list
  int x, y, w, h;        // the CU (luma samples for luma)
  int mpmFlag, secondMpmFlag, ipredIdx;   // luma: the parsed intra mode syntax (a TMRL CU's tmrlListIdx in ipredIdx)
  int nbFirst, nbLast;   // luma: the modes CU::isIdxModeValid removes at the start / the end of the angular range
  int plIdx;             // luma: the CU's directional planar index (Tu::plIdx of planar blocks)
  int candId;            // chroma: the parsed candidate index (CU_CHROMA_CAND)
  int lumaCu;            // chroma: the co-located luma CU (DM, and the candidates' substitution)
  int nbr[MPM_NEIGHBOURS];   // luma: the CU at each position of getSpatialIpm (-1: none, getPURestricted)
  // the modes the parsed syntax leaves (intra_deps reads the union of their reference ranges): modes[0] -1 any mode,
  // else modes[0] and, unless -1, modes[1]
  int modes[2];
  int dimd;              // the first of the CU's DIMD histograms in Params::dimdCalls (-1: no DIMD)
  int dimdNum;           // their number
  int obic;              // an OBIC CU's index among the OBIC CUs (Params::obicNbrs, obicOut; -1: not OBIC)
  int merge;             // an intra merge CU's record in Params::mergeData (MergeHead; -1: not intra merge)
  int mtss;              // with MTSS the first of the MTSS_CALLS histograms of its mode list in Params::dimdCalls (-1: none)
  int mtssNspt;          // with MTSS: its luma block may use NSPT (CU::isNSPTAllowed)
  int timd;              // a TIMD CU's record in Params::timdData (TimdHead; -1: not TIMD)
  int tmrl;              // a TMRL CU's index among the TMRL CUs (Params::tmrlOut; -1: not TMRL)
};
layout( buffer_reference, std430, buffer_reference_align = 4 ) buffer BufCu { Cu v[]; };

// Cu::kind
const int CU_KNOWN       = 0;   // the host knows the mode (Params::cuMode holds it)
const int CU_MPM         = 1;   // luma: from the MPM lists (PU::getIntraMPMs)
const int CU_CHROMA_DM   = 2;   // chroma: the co-located luma mode (PU::getCoLocatedIntraLumaMode)
const int CU_CHROMA_CAND = 3;   // chroma: candId of PU::getIntraChromaCandModes' first four
const int CU_DIMD        = 4;   // luma: a DIMD CU, its mode DIMD's (cu.dimd; intra_dimd.glsl)
const int CU_CHROMA_DIMD = 5;   // chroma: DIMD chroma (DIMD_CHROMA_IDX), compared with the co-located luma mode
const int CU_OBIC        = 6;   // luma: an OBIC CU, its mode OBIC's first (from its neighbours' modes; intra_obic.glsl)
const int CU_MERGE       = 7;   // luma: an intra merge CU, its mode its chosen candidate's (intra_merge.glsl)
const int CU_TIMD        = 8;   // luma: a TIMD CU, its modes from template costs (intra_timd.glsl), with its DIMD
const int CU_TMRL        = 9;   // luma: a TMRL CU, its mode and reference line from template costs (intra_tmrl.glsl), with
                                // its DIMD
const int CU_CHROMA_TMRL = 10;  // chroma: a chroma TMRL CU, its mode and reference line from costs of the co-located luma
                                // block's prediction and of its template (intra_tmrl.glsl), with its DIMD chroma (its
                                // Cu::nbr: CHROMA_TMRL_NBR_*)

// a TMRL CU's choice as its node derives it (TMRL_OUT int32 per TMRL CU in Params::tmrlOut): the mode (131-mode set)
// and the reference line (multiRefIdx) of m_tmrlList[tmrlListIdx], which its blocks predict with
const int TMRL_OUT      = 2;
const int TMRL_OUT_MODE = 0;
const int TMRL_OUT_MRL  = 1;

// the reference lines TMRL tries (EXT_REF_LINE_IDX), of a CU at picture row y those inside the picture, line L < y
// (getTmrlSearchRange: the lines before the first one at least y away)
const int TMRL_LINES[5] = int[5]( 1, 3, 5, 7, 12 );

int tmrlNumLines( int y )
{
  int n = 0;
  while( n < 5 && TMRL_LINES[n] < y )
  {
    n++;
  }
  return n;
}

// chroma TMRL (JVET-AJ0081): the chroma reference lines it tries (CHROMA_MULTI_REF_LINE_IDX), of a CU at chroma row y
// those inside the picture (getChromaTmrlSearchRange; PU::hasChromaTmrl allows it from row 4 on); the co-located luma
// block's line is twice as far. Its CU's Cu::nbr: the luma CUs at the five positions of getChromaTmrlSearchRange
// (CHROMA_TMRL_NBR_LUMA ..), then the chroma CUs at its five neighbour positions (CHROMA_TMRL_NBR_CHROMA ..; -1: none
// or not one it reads)
const int CHROMA_TMRL_LINES[3]   = int[3]( 1, 3, 5 );
const int CHROMA_TMRL_MODES      = 8;   // CHROMA_TMRL_MPM_SIZE
const int CHROMA_TMRL_NBR_LUMA   = 0;
const int CHROMA_TMRL_NBR_CHROMA = 5;

int chromaTmrlNumLines( int y )
{
  int n = 0;
  while( n < 3 && CHROMA_TMRL_LINES[n] < y )
  {
    n++;
  }
  return n;
}

// one gradient histogram of a CU's DIMD (16 int32, layout shared with GpuIntraRec.cpp; IntraPrediction's
// DimdHistogramCall, the arguments deriveDimdMode / deriveDimdModeAdaptive / deriveDimdChromaMode call buildHistogram
// / buildHistogramAdaptive with)
struct DimdCall
{
  int kind;            // 0: none, 1: buildHistogram, 2: buildHistogramAdaptive (luma)
  int histogram;       // 0: left, 1: above, 2: above-left (location dependence compares the first two); chroma 0
  int x, y;            // pReco in the plane of comp
  int height, width, direction, filterSizeIdx;
  int sizeExt[3];      // buildHistogramAdaptive: the length of each group of 4 template lines (uiSizeExt),
  int extra[3];        // whether the corner beyond each is available (isExtraAvailable),
  int maxAmp;          // the amplitude its sums stop at
  int comp;            // the component whose reconstruction it reads
};
layout( buffer_reference, std430, buffer_reference_align = 4 ) buffer BufDimdCall { DimdCall v[]; };

// a CU's DIMD as its node derives it (DIMD_OUT int32 per CU in Params::dimdOut): the CodingUnit fields deriveDimdMode
// sets (dimdMode, dimdBlending, dimdBlendMode, dimdRelWeight, dimdLocDep: DIMD_OUT_PARAMS, which an intra merge CU's
// node writes its choice in), the second strongest mode (candModeListForTransform[1]), with MTSS the mode list
// (candModeListForTransformMtss) and getFinalIntraModeForTransform's result: the LFNST / NSPT mode in NSPT's numbering
// and secondBucket; DIMD chroma's the final mode and the second one (dimdChromaMode, dimdChromaModeSecond). A MIP block
// with LFNST / NSPT writes its CU's: mipDimdMode as the mode, dimdDerivedIntraDir2nd as the second, with MTSS the result
// (its node the list)
const int DIMD_OUT             = 30;
const int DIMD_OUT_PARAMS      = 24;
const int DIMD_OUT_MODE        = 0;
const int DIMD_OUT_SECOND      = 1;
const int DIMD_OUT_BLENDING    = 1;
const int DIMD_OUT_BLEND_MODE  = 2;    // 7 entries, -1 where not set
const int DIMD_OUT_REL_WEIGHT  = 9;    // 8 entries, -1 where not set
const int DIMD_OUT_LOC_DEP     = 17;   // 7 entries
const int DIMD_OUT_TR_SECOND   = 24;
const int DIMD_OUT_MTSS_LIST   = 25;   // 3 entries
const int DIMD_OUT_MTSS_MODE   = 28;
const int DIMD_OUT_MTSS_BUCKET = 29;
const int MTSS_CALLS           = 3;    // the histograms of MTSS's mode list (deriveDimdModeList: left, above, above-left)

// a neighbour of an OBIC CU whose mode its histogram counts (layout shared with GpuIntraRec.cpp, OBIC_NBRS per OBIC CU
// in Params::obicNbrs; IntraPrediction::obicNeighbours): its record (-1: none) and the samples it counts
struct ObicNbr
{
  int cu;
  int numSamples;
};
layout( buffer_reference, std430, buffer_reference_align = 4 ) buffer BufObicNbr { ObicNbr v[]; };
const int OBIC_NBRS  = 20;

// an OBIC CU's OBIC as its node derives it (OBIC_OUT int32 per OBIC CU in Params::obicOut): the CodingUnit fields
// deriveObicMode sets, obicMode (-1 where not set), obicFusionWeight and obicLocDep
const int OBIC_MODES          = 6;
const int OBIC_OUT            = 3 * OBIC_MODES;
const int OBIC_OUT_MODE       = 0;
const int OBIC_OUT_WEIGHT     = OBIC_MODES;
const int OBIC_OUT_LOC_DEP    = 2 * OBIC_MODES;

// a template of TIMD's kind (10 int32, layout shared with GpuIntraRec.cpp): its arms, tw columns left of the CU and th
// rows above it (0: no such arm), and the units of the reference line of its block, the CU extended by the arms
// (IntraPrediction::TimdReferenceUnits), whose availability bits follow the record holding it (32 per word)
struct TemplateUnits
{
  int tw, th;
  int unitW, unitH;
  int leftTempUnitNum, aboveTempUnitNum, totalLeftUnits, totalAboveUnits, totalUnits, numIntraNeighbor;
};
const int TEMPLATE_UNITS = 10;

TemplateUnits templateUnitsAt( BufI32 b, int at )
{
  return TemplateUnits( b.v[at], b.v[at + 1], b.v[at + 2], b.v[at + 3], b.v[at + 4], b.v[at + 5], b.v[at + 6],
                        b.v[at + 7], b.v[at + 8], b.v[at + 9] );
}

// an intra merge CU's record in Params::mergeData (layout shared with GpuIntraRec.cpp): this header, then nbrNum pairs
// (neighbour record, MERGE_NBR_*) in deriveIntraMergeCand's order, then the availability bits of its template's
// reference units
struct MergeHead
{
  int           index;   // its index among the intra merge CUs (Params::mergeCost)
  int           nbrNum;
  TemplateUnits t;       // reorderIntraMergeCand's template (none: tw 0, the CU at the picture's left or top edge)
};
const int MERGE_HEAD_T    = 2;                         // the offset of MergeHead::t
const int MERGE_HEAD      = MERGE_HEAD_T + TEMPLATE_UNITS;

// a TIMD CU's record in Params::timdData (layout shared with GpuIntraRec.cpp): this header, then the availability
// bits of its template's reference units, then those of its TIMD-SAD template's
const int TIMD_MERGE_CANDS = 5;   // the neighbours searchTimdMrgCus keeps (its limitMaxNeigh)
struct TimdHead
{
  int           index;   // its index among the TIMD CUs (Params::timdOut)
  int           type;    // its template (CU::deriveTimdRefType): TIMD_NONE (the picture's top-left CU), TIMD_ABOVE,
                         // TIMD_LEFT, TIMD_BOTH
  int           waip;    // where deriveTimdMode's wide angles beyond the regular ones may reach: TIMD_WAIP_* bits
  TemplateUnits t;       // its template, an arm it lacks 0 (a TIMD merge CU's calcTimdMrgCandCosts', one sample thick)
  int           sad;     // a TIMD-SAD CU (CodingUnit::timdSad): its node also does deriveTimdModeSad
  TemplateUnits ts;      // deriveTimdModeSad's template (arms of up to 8; of the same type), where sad
  // the transform types of the CU's luma block as TrQuant::getTrTypes gives them (CodingUnit::timdmTrType, which a TIMD
  // merge CU may take): explicit MTS's pair index where the mode chooses its kernels (-1: none), else the types
  int           mtsTrIdx;
  int           trHor, trVer;
  // a TIMD merge CU (CodingUnit::timdMrg): its candidates' number (0: not TIMD merge) and records, searchTimdMrgCus'
  // neighbours in its order; inheritTr: the candidates give their transform types (genTimdMrgList, dual tree), else DCT-II
  int           merge;
  int           mergeNbr[TIMD_MERGE_CANDS];
  int           inheritTr;
};
const int TIMD_HEAD_T          = 3;                               // the offset of TimdHead::t
const int TIMD_HEAD_SAD        = TIMD_HEAD_T + TEMPLATE_UNITS;    // of TimdHead::sad
const int TIMD_HEAD_TS         = TIMD_HEAD_SAD + 1;               // of TimdHead::ts
const int TIMD_HEAD_MTS_TR_IDX = TIMD_HEAD_TS + TEMPLATE_UNITS;   // of TimdHead::mtsTrIdx
const int TIMD_HEAD_MERGE      = TIMD_HEAD_MTS_TR_IDX + 3;        // of TimdHead::merge
const int TIMD_HEAD            = TIMD_HEAD_MERGE + 2 + TIMD_MERGE_CANDS;
const int TIMD_NONE   = 0;
const int TIMD_ABOVE  = 1;
const int TIMD_LEFT   = 2;
const int TIMD_BOTH   = 3;
const int TIMD_WAIP_MAX_CLOSE = 1;   // increaseMaxClose: a PU above the CU's last column
const int TIMD_WAIP_MAX_FAR   = 2;   // increaseMaxFar: a PU above, a CU width further right
const int TIMD_WAIP_MIN_CLOSE = 4;   // decreaseMinClose: a PU left of the CU's last row
const int TIMD_WAIP_MIN_FAR   = 8;   // decreaseMinFar: a PU left, a CU height further down

// a TIMD CU's TIMD as its node derives it (TIMD_OUT int32 per TIMD CU in Params::timdOut): the CodingUnit fields
// deriveTimdMode sets, those it does not at their initial values (-1 modes and weights, 0 location dependence); from
// TIMD_OUT_SAD a TIMD-SAD CU's deriveTimdModeSad's alike (timdModeSad, ... timdModeSecondaryCheckWASad: the same
// fields at the same offsets), which its blocks predict; a TIMD merge CU's chosen candidate in the first fields (its
// timdMrgList[0], ... timdMrgModeCheckWA[0]). Then, in the pipelines with TIMD merge, the CU's transform types
// (timdmTrType: a TIMD merge CU's inherited ones, another's own)
const int TIMD_OUT_FIELDS          = 12;   // the fields of one derivation
const int TIMD_OUT_SAD             = TIMD_OUT_FIELDS;       // the offset of deriveTimdModeSad's
const int TIMD_OUT_TR_HOR          = 2 * TIMD_OUT_FIELDS;   // timdmTrType's horizontal type
const int TIMD_OUT_TR_VER          = TIMD_OUT_TR_HOR + 1;   // and vertical one
const int TIMD_OUT                 = TIMD_OUT_TR_VER + 1;
const int TIMD_OUT_MODE            = 0;    // timdMode (131-mode set; the regular mode of a wide angle beyond them)
const int TIMD_OUT_SECONDARY       = 1;    // timdModeSecondary
const int TIMD_OUT_NON_ANG         = 2;    // timdModeNonAng (planar / DC)
const int TIMD_OUT_BLENDED         = 3;    // timdIsBlended
const int TIMD_OUT_WEIGHT          = 4;    // timdFusionWeight, 3 entries: the mode's, the secondary's, the non-angular one's
const int TIMD_OUT_LOC_DEP         = 7;    // timdLocDep, 3 entries
const int TIMD_OUT_CHECK_WA        = 10;   // timdModeCheckWA: 0 for a wide angle beyond the regular ones
const int TIMD_OUT_SEC_CHECK_WA    = 11;   // timdModeSecondaryCheckWA
const int MERGE_NBR_MERGE = 1;   // an intra merge CU: always a candidate, its merge parameters (its DimdOut slot)
const int MERGE_NBR_OBIC  = 2;   // an OBIC CU: always, its OBIC converted
const int MERGE_NBR_DIMD  = 3;   // a DIMD CU or an MPM CU: where its DIMD blends, its DIMD
const int MERGE_NBR_TIMD  = 4;   // a TIMD CU: where its modes blend or its mode is angular, its TIMD converted (or its DIMD)
const int MERGE_NBR_TIMD_MERGE = 5;   // a TIMD merge CU: as a TIMD CU, but no candidate where its TIMD does not convert
layout( buffer_reference, std430, buffer_reference_align = 8 ) buffer BufU64 { uint64_t v[]; };

// the directory of the transform tables (in matrices at Params::trTables, GpuIntraRec.cpp's createShared): the
// offsets of the tables (in matrices), then values
const int TR_TAB_MATRIX       = 0;    // [type][log2 size - 1]: the inverse transform matrices' offsets (-1: none)
const int TR_TAB_IPM_TO_SET   = 1;    // g_aucIpmToTrSet[16][36]: explicit MTS's kernel set of size class and mode
const int TR_TAB_SET          = 2;    // g_aucTrSet[80][TR_TAB_SET_ROW]: the kernel pairs of a set
const int TR_TAB_IDX_TO_TR    = 3;    // g_aucTrIdxToTr[25][2]: a pair's vertical and horizontal type
const int TR_TAB_ID_LUT       = 4;    // g_aiIdLut[3][3]: the identity transform's mode distance per size
const int TR_TAB_LFNST_LUT    = 5;    // g_lfnstLut: the LFNST set of a mode
const int TR_TAB_NSPT_LUT     = 6;    // g_nsptLut: the NSPT set of a mode
const int TR_TAB_LFNST_MATRIX = 7;    // [3 region sizes][35 sets][3 indices]: the matrices' offsets in secKernels
const int TR_TAB_NSPT_MATRIX  = 8;    // [NSPT shape][35 sets][3 indices][3 buckets]: the matrices' offsets in secKernels
const int TR_TAB_LFNST_LAYOUT = 9;    // [3 region sizes][2 transposed]: the offsets of the LFNST output arrangements
const int TR_TAB_EXT_MODES    = 10;   // value: NUM_EXT_LUMA_MODE
const int TR_TAB_SET_ROW      = 11;   // value: the length of g_aucTrSet's rows
const int TR_TAB_IDTR         = 12;   // value: the identity transform's type (IDTR)
const int TR_TAB_NUM          = 13;
const int TR_DCT2             = 0;    // DCT-II's transform type (TransType)

// the NSPT shapes (w x h) of TR_TAB_NSPT_MATRIX
int nsptShape( int w, int h )
{
  return w == 4 && h == 4 ? 0 : w == 8 && h == 8 ? 1 : w == 4 && h == 8 ? 2 : w == 8 && h == 4 ? 3 : w == 4 && h == 16 ? 4
       : w == 16 && h == 4 ? 5 : w == 8 && h == 16 ? 6 : w == 16 && h == 8 ? 7 : w == 4 && h == 32 ? 8 : w == 32 && h == 4 ? 9
       : w == 8 && h == 32 ? 10 : 11;
}

// Tu::secKind
const int SEC_NONE  = 0;
const int SEC_LFNST = 1;
const int SEC_NSPT  = 2;

// the signPred data of a block (at Tu::spOffset): the number of predicted signs (1..8), the coded residual signs (bit
// i: predicted coefficient i is negative), the predicted coefficients in decision order (x | y << 16), then from
// SP_TEMPLATES on, per predicted coefficient, its template (intra_resi): its residual's left column bottom-up, then
// its top row (w + h samples)
const int SP_NUM       = 0;
const int SP_SIGNS     = 1;
const int SP_POS       = 2;
const int SP_TEMPLATES = 10;

// xFillReferenceSamples' rule (GpuIntraRec.cpp's fillSource): the reconstructed sample (sx, sy) whose value entry idx
// of the top or left arm of a reference line takes, for the area (fx, fy, fw, fh) filled with arms of predSize /
// predHSize samples from line mrl (multiRefIdx), given the availability av: the corner flag and the available prefix
// of the above, above-right, left, below-left, left MRL and above MRL units. False where no unit is available (the
// mid-level value). The units in the CPU's order: below-left and left (bottom-up), left MRL, the corner C, above MRL,
// above, above-right; a line beyond the first unit has lM MRL units left of and aM above the corner unit, which keeps
// entries 0 .. cLen of both arms (JVET-Y0116); an arm whose length is no multiple of the unit ends in a partial unit
// (chroma TMRL's arms of 2 x (the side + 1)). An unavailable unit takes the last sample of the unit before it in that
// order, units before the first available one take its first sample.
bool fillSource( int fx, int fy, int fw, int fh, int predSize, int predHSize, int mrl, int uW, int uH, int av[7],
                 bool top, int idx, out int sx, out int sy )
{
  const int tAU = ( predSize + uW - 1 ) / uW, tLU = ( predHSize + uH - 1 ) / uH;
  const int lM = mrl / uH, aM = mrl / uW, cLen = mrl - lM * uH, C = tLU + lM;
  const int totalUnits = tAU + tLU + 1 + lM + aM;
  const int numAbove = max( fw / uW, 1 ), numLeft = max( fh / uH, 1 );
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
  if( corner != 0 ) firstAvail = C;
  if( pLM > 0 ) firstAvail = lmS;
  if( pL > 0 ) firstAvail = lS;
  if( pBL > 0 ) firstAvail = blS;
  sx = 0;
  sy = 0;
  if( firstAvail >= totalUnits )
  {
    return false;
  }
  const int u = idx <= cLen ? C : top ? C + 1 + ( idx - 1 - cLen ) / uW : C - 1 - ( idx - 1 - cLen ) / uH;
  int pm = -1;   // the nearest available unit at or before u
  if( pBL > 0 && blS <= u ) pm = max( pm, min( u, blE ) );
  if( pL > 0 && lS <= u ) pm = max( pm, min( u, lE ) );
  if( pLM > 0 && lmS <= u ) pm = max( pm, min( u, lmE ) );
  if( corner != 0 && C <= u ) pm = max( pm, C );
  if( pAM > 0 && amS <= u ) pm = max( pm, min( u, amE ) );
  if( pA > 0 && aS <= u ) pm = max( pm, min( u, aE ) );
  if( pAR > 0 && arS <= u ) pm = max( pm, min( u, arE ) );
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

// the blocks' owner map of one component (Params::owner: per unit the block covering it) with its size and unit size
struct OwnerMap
{
  BufI32 owner;
  int    width, height, log2X, log2Y;
};

// sample (x, y) lies in a unit of a block decoded before block `before`: the kernels' reference availability
// ("decoded earlier"; supported() rejects what makes the CPU's getCURestricted differ)
bool decodedBefore( OwnerMap m, int x, int y, int before )
{
  if( x < 0 || y < 0 || ( x >> m.log2X ) >= m.width || ( y >> m.log2Y ) >= m.height )
  {
    return false;
  }
  const int o = m.owner.v[( y >> m.log2Y ) * m.width + ( x >> m.log2X )];
  return o >= 0 && o < before;
}

// the availability of reference line mrl of the area (fx, fy, fw, fh) whose arms have tAU above and tLU left units of
// uW x uH (xFillReferenceSamples' units), for block `before`, in Tu::avail's layout (fillSource): the corner flag, then
// the available prefixes of the above, above-right, left and below-left units, counted up to unitsTop / unitsLeft units
// of each arm (at least one unit each: intra_deps counts only the units a prediction reads), and of the MRL units left
// of the corner (downwards) and above it (rightwards)
void refLineAvailability( OwnerMap m, int before, int fx, int fy, int fw, int fh, int tAU, int tLU, int mrl, int uW, int uH,
                          int unitsTop, int unitsLeft, out int av[7] )
{
  const int x0 = fx - 1 - mrl, y0 = fy - 1 - mrl;
  const int lM = mrl / uH, aM = mrl / uW;
  const int numAbove = max( fw / uW, 1 ), numLeft = max( fh / uH, 1 );
  const int limA  = max( 1, min( numAbove, unitsTop ) );
  const int limAR = max( 1, min( tAU - numAbove, unitsTop - numAbove ) );
  const int limL  = max( 1, min( numLeft, unitsLeft ) );
  const int limBL = max( 1, min( tLU - numLeft, unitsLeft - numLeft ) );
  av[0] = decodedBefore( m, x0, y0, before ) ? 1 : 0;
  int k;
  for( k = 0; k < limA && decodedBefore( m, fx + k * uW, y0, before ); k++ ) {}
  av[1] = k;
  for( k = 0; k < limAR && decodedBefore( m, fx + fw - 1 + uW + k * uW, y0, before ); k++ ) {}
  av[2] = k;
  for( k = 0; k < limL && decodedBefore( m, x0, fy + k * uH, before ); k++ ) {}
  av[3] = k;
  for( k = 0; k < limBL && decodedBefore( m, x0, fy + fh - 1 + uH + k * uH, before ); k++ ) {}
  av[4] = k;
  for( k = 0; k < lM && decodedBefore( m, x0, fy - lM * uH + k * uH, before ); k++ ) {}
  av[5] = k;
  for( k = 0; k < aM && decodedBefore( m, fx - aM * uW + k * uW, y0, before ); k++ ) {}
  av[6] = k;
}

// the sample entry idx of the top or left arm of reference line ln of block t takes (false: the mid-level value;
// GpuIntraRec.cpp's refSource): xFillReferenceSamples at the block with arms of 8x its size, or for an ISP
// sub-partition initIntraPatternChTypeISP: the first one fills the CU's references with arms of twice the CU's size;
// a later one takes the CU's arm along the split shifted by its offset in the CU where that side of it is available
// (Tu::ispSide), else the first sample of the row / column it borders, and the other arm from the row above / column
// left of it (the previous sub-partition), its last sample repeated; the corner is the shifted arm's entry 0
bool refSource( Tu t, int ln, bool top, int idx, out int sx, out int sy )
{
  if( t.isp == 0 )
  {
    return fillSource( t.x, t.y, t.w, t.h, t.w << 3, t.h << 3, t.mrl + ln, t.unitW, t.unitH, t.avail[ln], top, idx, sx, sy );
  }
  const bool hor = t.isp == 1;
  const int  off = hor ? t.y - t.cuY : t.regX - t.cuX;
  const bool shiftedArm = hor ? !top : top;
  if( off == 0 || ( ( shiftedArm || idx == 0 ) && t.ispSide != 0 ) )   // from the CU's references
  {
    const bool cuTop = off == 0 ? top : !hor;
    const int  cuIdx = off == 0 ? idx : ( shiftedArm ? idx : 0 ) + off;
    return fillSource( t.cuX, t.cuY, t.cuW, t.cuH, 2 * t.cuW, 2 * t.cuH, 0, t.unitW, t.unitH, t.avail[0], cuTop, cuIdx, sx, sy );
  }
  if( shiftedArm || idx == 0 )
  {
    sx = hor ? t.regX : t.regX - 1;
    sy = hor ? t.y - 1 : t.y;
    return true;
  }
  const int j = min( idx, hor ? t.regW : t.h ) - 1;
  sx = hor ? t.regX + j : t.regX - 1;
  sy = hor ? t.y - 1 : t.y + j;
  return true;
}

struct Params
{
  uint64_t tus;
  uint64_t coeffs;      // int32 levels of each block's nonzero region (nzRows x nzCols, row-major) at coeffOffset
  uint64_t plane[3];    // int16 reconstruction (written), origin/stride below
  uint64_t owner[3];    // int32 per unit and component: the block covering it (-1: none); 4x4 luma, 2x2 chroma units
  uint64_t filters;     // int32: [taps 6,8,10,12,16][32][16] cubic tables, then the weak 4-tap table [32][4], the bilateral
                        // filter's tables, TIMD's 1/64 tables (intra_rec.comp's BF_LUT .. EXT_NTAP)
  uint64_t matrices;    // int32: the inverse transform matrices (each [k * N + i], at Tu::mtxV / mtxH), QCS weights at qcsOffset
  uint64_t dbgPred;     // int16 per block w*h at dbgOffset
  uint64_t dbgResi;     // int16 residual per block w*h at dbgOffset (written by intra_resi, added by intra_rec)
  uint64_t depCount;    // int32 per block: dependencies not reconstructed yet
  uint64_t depOffset;   // int32 per block + 1: its dependents in depList
  uint64_t depList;     // int32: the blocks reading samples of a block
  uint64_t queue;       // int32 per position: the ready blocks in the order they became ready (-1: not yet)
  uint64_t counter;     // int32 [4]: the next queue position to take, the next to fill, the reconstructed blocks
  uint64_t depStart;    // int32 per block + 1: its dependencies in depTmp (intra_deps)
  uint64_t depTmp;      // int32: the dependencies of each block (intra_deps)
  uint64_t numDep;      // int32 per block: its number of dependents, then the entries written (intra_deps)
  uint64_t lmcs;        // int32, lmcsLutSize entries each: the forward LUT, the inverse LUT, the chroma scale of each average luma
  uint64_t signPred;    // int32: the sign prediction data of the blocks with predicted signs (at Tu::spOffset, SP_*)
  uint64_t secKernels;  // int8: the LFNST and NSPT matrices (at Tu::secMatrix)
  uint64_t cus;         // the CU records (Tu::cu)
  uint64_t cuMode;      // int32 per CU: its mode (luma: PU::getIntraDirLuma, which the neighbours' MPM lists take; chroma:
                        // the final mode), the host's (CU_KNOWN) or written by the CU's node (-1 until then)
  uint64_t dbgParams;   // verification: per block the parameters intra_rec derived (DBG_PARAMS int32, DERIVE_MODE blocks)
  uint64_t dimdCalls;   // the DIMD histograms of the CUs (Cu::dimdNum DimdCall at Cu::dimd)
  uint64_t dimdOut;     // int32 per CU (DIMD_OUT): its DIMD, written by its node, a MIP CU's by its block (for its blocks,
                        // and the comparison)
  uint64_t obicNbrs;    // per OBIC CU (Cu::obic) its neighbours (OBIC_NBRS ObicNbr)
  uint64_t obicOut;     // int32 per OBIC CU (OBIC_OUT): its OBIC, written by its node (for its blocks, the comparison)
  uint64_t scratch;     // int32, intra_rec's: per workgroup the 3 x 128 x 128 sums of DIMD / OBIC / merge blending
                        // (BLEND_ACC), then from cccmScratchOffset per workgroup cccmScratchWords of CCCM data
  uint64_t mergeData;   // the intra merge CUs' records (Cu::merge, MergeHead)
  uint64_t mergeCost;   // uint64 per intra merge CU: the template cost of each mode (NUM_LUMA_MODE; ~0: none)
  uint64_t timdData;    // the TIMD CUs' records (Cu::timd, TimdHead)
  uint64_t timdOut;     // int32 per TIMD CU (TIMD_OUT): its TIMD, written by its node (for its blocks, the comparison)
  uint64_t tmrlOut;     // int32 per TMRL CU (TMRL_OUT): its mode and line, written by its node (for its blocks, the comparison)
  uint64_t ccpOut;      // int32 per LM / LM chroma fusion / CCP merge CU (CCP_OUT): its CCP model, which its blocks
                        // write (Tu::ccp)
  uint64_t ccpData;     // int32: per CCP record its writing blocks (CCP_WRITERS), the CTU rows' histories, the merge
                        // CUs' sources (Tu::ccpMerge; intra_ccp.glsl)
  int      origin[3];
  int      stride[3];
  int      width[3];
  int      height[3];
  int      ownerW[2];   // per channel type
  int      ownerH[2];
  int      unitLog2X[2];
  int      unitLog2Y[2];
  int      edgeCap;     // the capacity of depList / depTmp
  int      lmcsLutSize;              // entries of each LMCS table (1 << luma bit depth)
  int      lumaClipMin, lumaClipMax; // the slice's luma clipping range (the average luma of chroma residual scaling)
  int      qcsOffset;   // Quant's m_coeffShift64: the weight (of 1024) of the next larger index, per index magnitude < 64
  int      trTables;    // in matrices: the directory of the tables a derived mode's transforms are chosen with (TR_TAB_*)
  int      ctuSize;     // the SPS's CTU size (luma samples)
  int      lumaUnitW, lumaUnitH;   // the luma availability units (PreCalcValues::minCUWidth / minCUHeight)
  int      cclmFilter;  // CCLM's luma downsampling filter by the SPS's collocated chroma flags (CCLM_FILTER_*)
  int      dualTree;    // CS::isDualITree: luma and chroma are separate trees (CCCM reads luma right of / below a block)
  int      cccmLbccp;   // CCCM's two sets of models (LBCCP, JVET-AD0120: an intra slice with the TM non-inter tools)
  // scratch's CCCM part: a CCCM block's luma buffers and prediction beyond shared memory (several filters' buffers,
  // larger blocks), after blending's sums, which keep their constant stride (blending indexing by a stride read from
  // Params was 4 % slower; a buffer of its own, its address a further field, measured 0.7 % slower on CCCM streams)
  int      cccmScratchOffset;   // its start (int32)
  int      cccmScratchWords;    // its int32 per workgroup
};
layout( buffer_reference, std430, buffer_reference_align = 8 ) buffer BufParams { Params p; };

// Quant::dequant of a clipped nonzero level, in 32-bit arithmetic like the CPU's Intermediate_Int
int dequantScalar( Tu t, int qc, int add )
{
  return t.dqRightShift > 0 ? ( qc * t.dqScale + add ) >> t.dqRightShift : ( qc * t.dqScale ) << ( -t.dqRightShift );
}

// the dequantized coefficient of a nonzero level (Tu::quant bit 0: a quantization index of dependent quantization),
// clipped to the transform range; QCS (bit 1) weighs in the dequantization of the next larger index magnitude with
// the weight table at qcsOffset in the matrices buffer
int dequant( Tu t, int q, int add, BufI32 mt, int qcsOffset )
{
  const bool qcs = ( t.quant & 2 ) != 0;
  if( ( t.quant & 1 ) != 0 )   // DepQuant's dequantBlock, in 64-bit arithmetic; the indices are not clipped
  {
    int64_t c = ( int64_t( q ) * t.dqScale + add ) >> t.dqRightShift;
    if( qcs )
    {
      const int64_t c2 = ( int64_t( q + ( q > 0 ? 1 : -1 ) ) * t.dqScale + add ) >> t.dqRightShift;
      const int     w  = abs( q ) < 64 ? mt.v[qcsOffset + abs( q )] : 0;
      c = ( int64_t( 1024 - w ) * c + int64_t( w ) * c2 ) >> 10;
    }
    return int( min( max( c, int64_t( t.trMin ) ), int64_t( t.trMax ) ) );
  }
  const int qc = clip3( t.dqInMin, t.dqInMax, q );
  int       c  = dequantScalar( t, qc, add );
  if( qcs )
  {
    const int c2 = dequantScalar( t, qc + ( qc > 0 ? 1 : -1 ), add );
    const int w  = abs( qc ) < 64 ? mt.v[qcsOffset + abs( qc )] : 0;
    c = ( ( 1024 - w ) * c + w * c2 ) >> 10;
  }
  return clip3( t.trMin, t.trMax, c );
}

// the rounding offset of dequantScalar / dequant
int dequantAdd( Tu t )
{
  return t.dqRightShift > 0 ? 1 << ( t.dqRightShift - 1 ) : 0;
}

// TrQuant's invTransformCbCr: the residual of the other component of a joint CbCr pair from the coded one (ICT mode
// +-1, +-2: Cr from Cb, +-3: Cb from Cr)
int jccrDerived( int mode, int r )
{
  if( mode == 1 || mode == 3 )
  {
    return r >> 1;
  }
  if( mode == -1 || mode == -3 )
  {
    return ( -r ) >> 1;
  }
  if( mode == 2 )
  {
    return r;
  }
  return r == -32768 ? 32767 : -r;   // -2, with the CPU's clipping of -(-32768) to 16 bits
}

layout( push_constant ) uniform Push
{
  uint64_t params;
  int      numTus;
  int      verify;    // the prediction is copied for the comparison with the CPU
} pc;

#endif
