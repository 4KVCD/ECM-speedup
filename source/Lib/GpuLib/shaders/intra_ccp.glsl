// CCP merge on the GPU (JVET-AD0188's non-local cross-component prediction, with JVET-AE0097: no offset update): a
// merge CU inherits the CCP model (CCPModelCandidate, the records of Params::ccpOut: CCP_OUT_*) of a CU decoded
// before it, the one its bitstream index (idxNonLocalCCP) selects from a list that its template ranks. DecCu's
// xIntraRecBlk does it at the Cb block: PU::getCCPModelCandidateList - the models of the CUs at 5 adjacent and up to
// 87 non-adjacent positions, of the CTU row's history (cs.ccpLut) and up to 12 defaults, each unless equal to one
// before it (CCPModelCandidate::operator==), at most MAX_CCP_CAND_LIST_SIZE - then reorderCCPCandidates
// (xGetOneCCPCandCost: per candidate the SAD of both chroma components' predictions of the row above and the column
// left of the block, a stable sort by it), then predCCPCandidate with the chosen model, whose LBCCP inside filter flag
// may smooth the prediction. Here each block of the merge CU builds the list, ranks it and predicts its component with
// the chosen model, which the Cb block writes into the CU's record. The host gives the list's sources (Params::ccpData
// at Tu::ccpMerge: the records of the PUs at the positions, those of the CTU row's history, spare records for each
// block's defaults; its scheduling waits for the blocks writing them). Merge adjustment (JVET-AL0126)
// lists the CCCM models alone, adds models fitted on the fly (ccpFitModels) and moves the chosen model's coefficients
// by template costs (ccpMergeAdjust). Merge fusion with CCCM (predCCPCandidate's ccpMergeFusionType 0) is LM chroma
// fusion's first type with equal weights: the block computes the CCCM prediction before its own and storeRegion
// averages them (intra_cccm.glsl's cccmFusionBlend), or with the SPS's ccpFusionMode blends them by a model fitted to
// their template predictions (as decoder-derived merge fusion's pairs). Merge fusion with DIMD chroma
// (ccpMergeFusionType 1): the CU's node derives the DIMD chroma mode as a DIMD_CHROMA_IDX CU's, and intra_rec's
// blending averages the block's merge prediction and the mode's (two blended predictions of weight 1, the average
// rounded). A candidate predicts like the LM blocks of its type: a CCLM / GLM model as intra_cclm.glsl's (the slots in
// s_cclm), the others with intra_cccm.glsl's luma buffers and cccmPredict (the CPU's model applied to the CU's own
// luma, the candidate's luma offset taken in the buffer: xCccmGetLumaVal's luma minus m_cccmLumaOffset plus
// m_cccmLumaOffset minus the candidate's offset).
// Decoder-derived CCP fusion (JVET-AG0154's decoderDerivedCcpMode CUs; DecCu at the Cb block: decoderDerivedCcp,
// predDecoderDerivedIntraCCCMFusions) shares the merge code: its entry (kind CCP_KIND_DERIVED) has no sources and no
// history, its list is its modes - a CCLM model and CCCM models the block fits itself (ccpDerivedLinear,
// ccpFitModels) - whose costs are the SADs of their predictions of probe lines (the row above and the column left of
// the block; for the modes with LBCCP's inside filter that line filtered with the lines beside it), and the mode of
// the lowest cost alone or the pair of modes whose 3 : 1 blend of the probe lines costs less (ccpDerivedChoice)
// predicts: list[0] as a merge block's chosen candidate, list[1] before it into the fusion slot that storeRegion blends
// in (cccmFusionBlend with the weight 3). Each block derives the choice itself, from both components' costs. Its
// branches are in the pipelines with WITH_CCP_DERIVED alone (the merge pictures' pipeline without them is faster).
// With LFNST / NSPT (JVET-AK0064) their mode is a DIMD of the CU's final chroma predictions (ccpMergeTransformMode, in
// intra_rec's second module alone, CCP_LFNST_MODULE: the block predicts each component as its block does).
// glslang's inlining multiplies intra_rec.comp's SPIR-V (its ID bound limits the module): the large functions have
// few call sites here (the costs and the predictions one loop; the adjustment's costs and the fits one more each).
// Included by intra_rec.comp after intra_cccm.glsl
#ifndef INTRA_CCP_GLSL
#define INTRA_CCP_GLSL

const int CCP_MAX_CANDS     = 12;   // MAX_CCP_CAND_LIST_SIZE (also the defaults' number)
const int CCP_MAX_LIST      = 15;   // MAX_CCP_CAND_LIST_INIT_SIZE: the list with adjustment
const int CCP_FLY_MAX       = 6;    // MAX_NUM_DERIVED_CCP_ADJUST_CAND: adjustment's on-the-fly models
const int CCP_HIST_MAX      = 6;    // MAX_NUM_HCCP_CANDS: the history's models
const int CCP_DEFAULT_SHIFT = 3;    // the defaults' shift
// the defaults' slopes (getCCPModelCandidateList's defaultA): with a CCLM candidate before them its slope moved by
// these (towards 0 first), else these themselves
const int CCP_DEFAULT_A[CCP_MAX_CANDS] = int[]( 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6 );

// the candidate list in shared memory (s_dimdHist: a merge block derives no DIMD of its own; with LFNST / NSPT the
// histogram of its predictions after the list's words, CCP_DIMD_HIST), which invocation 0 builds: per
// candidate its record, its inside filter flag (the PU's curCand's for a spatial one, the history's copy's for a
// history one, 0 for a default or an on-the-fly model; a decoder-derived fusion mode's own) and its template cost;
// their number; the history's models (most recent first) and their number; the first CCLM candidate's slopes (Cb's,
// Cr's, 64-bit as two words) and shifts, which the defaults move, and whether there is one; which on-the-fly models are
// valid; decoder-derived fusion's costs of the pairs of modes and its choice (the list entries of list[0] and list[1],
// -1 without fusion; decoder-derived merge fusion's alike), the sorted candidates' places before the sort (its template
// samples' order), the chosen pair's model
#define s_ccp s_dimdHist
const int CCP_LIST = 0, CCP_LIST_INSIDE = 15, CCP_LIST_COST = 30, CCP_LIST_NUM = 45, CCP_LUT = 46, CCP_LUT_NUM = 52;
const int CCP_SCALE = 53, CCP_SHIFT = 57, CCP_FOUND = 59, CCP_FLY_VALID = 60, CCP_PAIR_COST = 66;
const int CCP_DERIVED_FIRST = 81, CCP_DERIVED_SECOND = 82, CCP_LIST_ORIG = 83, CCP_BLEND = 98;
const int CCP_DIMD_HIST = 2 * NUM_LUMA_MODE;   // (s_dimdHist's last NUM_LUMA_MODE words)

// decoder-derived CCP merge fusion (reorderCCPCandidates with ddNonLocalCCPFusion): pairs of the first 9 candidates,
// per pair in the scratch its words (CCP_PAIR_*: its cost, whether its model blends, the model's three parameters,
// shift and offset), the same words of the chosen pair at CCP_BLEND; a blend's weights of list[1] 1 .. 31 of 32
const int CCP_FUSION_CANDS = 9, CCP_PAIRS_MAX = 36, CCP_PAIR_WORDS = 7, CCP_FUSION_SLOTS_MAX = 12;
const int CCP_PAIR_COST_W = 0, CCP_PAIR_VALID = 1, CCP_PAIR_PARAMS = 2, CCP_PAIR_SHIFT = 5, CCP_PAIR_OFFSET = 6;

// decoder-derived CCP fusion's modes in the CPU's order (decoderDerivedCcp): the CCLM model (LM_CHROMA_IDX), plain
// CCCM of a single model without and with LBCCP's inside filter, of two models alike, GL-CCCM where the block has more
// than 32 samples (the GPU path's intra slices); each mode's record among the block's spare ones (CCLM's, the single
// model's, the two models', GL-CCCM's) and its inside filter flag
const int CCP_DERIVED_MODES = 6;
const int CCP_DERIVED_RECORD[CCP_DERIVED_MODES] = int[]( 0, 1, 1, 2, 2, 3 );
const int CCP_DERIVED_INSIDE[CCP_DERIVED_MODES] = int[]( 0, 0, 1, 0, 1, 0 );

int ccpWord( int r, int w )
{
  return CBufI32( P.ccpOut ).v[r * CCP_OUT + w];
}

int64_t ccpParam( int r, bool second, int c, int p )
{
  const int o = r * CCP_OUT + ( second ? CCP_OUT_PARAMS2 : CCP_OUT_PARAMS ) + 2 * ( c * CCP_MAX_PARAMS + p );
  return packInt2x32( ivec2( CBufI32( P.ccpOut ).v[o], CBufI32( P.ccpOut ).v[o + 1] ) );
}

// CCPModelCandidate::operator== of records x and y: the same type, then by it - CCLM's and GLM's patterns the slopes
// and shifts (not the offsets b), the others the luma offset and the parameters (CCCM's and GL-CCCM's 7, GLM with
// luma's 3, without subsampling x's number, several filters' 10 or 11 of the same filters), with two models also the
// second's and the threshold
bool ccpEqual( int x, int y )
{
  const int type = ccpWord( x, CCP_OUT_TYPE );
  if( type != ccpWord( y, CCP_OUT_TYPE ) )
  {
    return false;
  }
  const bool two = ( type & CCP_TYPE_MMLM ) != 0;
  if( two && ccpWord( x, CCP_OUT_Y_THRES ) != ccpWord( y, CCP_OUT_Y_THRES ) )
  {
    return false;
  }
  int n = 1;   // (the slopes: parameter 0)
  if( ( type & ( CCP_TYPE_CCLM | CCP_TYPE_GLM0123 ) ) != 0 )
  {
    for( int c = 0; c < 2; c++ )
    {
      if( ccpWord( x, CCP_OUT_SHIFT + c ) != ccpWord( y, CCP_OUT_SHIFT + c )
          || ( two && ccpWord( x, CCP_OUT_SHIFT2 + c ) != ccpWord( y, CCP_OUT_SHIFT2 + c ) ) )
      {
        return false;
      }
    }
  }
  else
  {
    const int mdf = ccpWord( x, CCP_OUT_MDF );
    n = ( type & ( CCP_TYPE_CCCM | CCP_TYPE_GLCCCM ) ) != 0 ? 7 : ( type & CCP_TYPE_GLM4567 ) != 0 ? GLM_PARAMS
      : ( type & CCP_TYPE_NSCCCM ) != 0 ? ccpWord( x, CCP_OUT_NUM_PARAMS ) : mdf == 1 ? 10 : 11;
    if( ccpWord( x, CCP_OUT_LUMA_OFFSET ) != ccpWord( y, CCP_OUT_LUMA_OFFSET )
        || ( ( type & CCP_TYPE_MDFCCCM ) != 0 && mdf != ccpWord( y, CCP_OUT_MDF ) ) )
    {
      return false;
    }
  }
  for( int i = 0; i < 2 * n; i++ )   // (component i / n, parameter i % n)
  {
    if( ccpParam( x, false, i / n, i % n ) != ccpParam( y, false, i / n, i % n )
        || ( two && ccpParam( x, true, i / n, i % n ) != ccpParam( y, true, i / n, i % n ) ) )
    {
      return false;
    }
  }
  return true;
}

// ----- the list (PU::getCCPModelCandidateList)

// the history of merge block t into s_ccp (invocation 0): cs.ccpLut, whose addCCPToLut drops a model equal to the new
// one (else the oldest when full) before appending it, so it holds the latest model of each of the last CCP_HIST_MAX
// distinct ones, read most recent first: the CTU row's models walked back from the latest, each unless equal to one
// taken. Then the first CCLM candidate's slopes for the defaults: of the first CCLM model among the spatial sources and
// the history (a CCLM candidate left out as equal to one before it has the same slopes and shifts)
void ccpMergeHistory( Tu t )
{
  BufI32    d = BufI32( P.ccpData );
  const int e = t.ccpMerge;
  int       lutNum = 0;
  for( int h = d.v[e + CCP_MERGE_HIST_NUM] - 1; h >= 0 && lutNum < CCP_HIST_MAX; h-- )
  {
    const int r = d.v[d.v[e + CCP_MERGE_HIST] + h];
    bool      known = false;
    for( int j = 0; j < lutNum && !known; j++ )
    {
      known = ccpEqual( s_ccp[CCP_LUT + j], r );
    }
    if( !known )
    {
      s_ccp[CCP_LUT + lutNum++] = r;
    }
  }
  s_ccp[CCP_LUT_NUM] = lutNum;
  const int numSpatial = d.v[e + CCP_MERGE_NUM_SPATIAL];
  s_ccp[CCP_FOUND] = 0;
  for( int s = 0; s < numSpatial + lutNum && s_ccp[CCP_FOUND] == 0; s++ )
  {
    const int r = s < numSpatial ? d.v[e + CCP_MERGE_SPATIAL + s] : s_ccp[CCP_LUT + s - numSpatial];
    if( ccpWord( r, CCP_OUT_TYPE ) == CCP_TYPE_CCLM )
    {
      for( int c = 0; c < 2; c++ )
      {
        const ivec2 a = unpackInt2x32( ccpParam( r, false, c, 0 ) );
        s_ccp[CCP_SCALE + 2 * c]     = a.x;
        s_ccp[CCP_SCALE + 2 * c + 1] = a.y;
        s_ccp[CCP_SHIFT + c]         = ccpWord( r, CCP_OUT_SHIFT + c );
      }
      s_ccp[CCP_FOUND] = 1;
    }
  }
}

// whether the block predicts the other chroma component (ccpMergeTransformMode): then it takes spare records of its own
// (the other block may be using that component's) and writes no CU record; never in intra_rec's first module
#ifdef CCP_LFNST_MODULE
bool g_ccpOther = false;
#else
const bool g_ccpOther = false;
#endif

// the first default record of merge block t: the Cb block's 12, then the Cr block's (each block ranks the list); with
// LFNST / NSPT 24 more, those of each block's prediction of the other component
int ccpDefaultRecords( Tu t )
{
  return BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_DEFAULTS]
         + ( t.comp - 1 + ( g_ccpOther ? 2 : 0 ) ) * CCP_MAX_CANDS;
}

// the default candidates' records of merge block t (all invocations, ending with a barrier): CCLM models of a slope k
// (CCP_DEFAULT_A[k], shift CCP_DEFAULT_SHIFT) or, with a CCLM candidate, its slope moved by k / 8 towards 0 first (the
// shift at least CCP_DEFAULT_SHIFT), the offset the mid level, 2 parameters, every other field 0
void ccpMergeDefaults( Tu t, int lid, int NT )
{
  const int first = ccpDefaultRecords( t );
  for( int i = lid; i < CCP_MAX_CANDS * CCP_OUT; i += NT )
  {
    const int k = i / CCP_OUT, w = i % CCP_OUT, p = ( w - CCP_OUT_PARAMS ) >> 1;   // (p: the word's parameter)
    int v = 0;
    if( w == CCP_OUT_TYPE )
    {
      v = CCP_TYPE_CCLM;
    }
    else if( w == CCP_OUT_NUM_PARAMS )
    {
      v = 2;
    }
    else if( w == CCP_OUT_SHIFT || w == CCP_OUT_SHIFT + 1 || ( w >= CCP_OUT_PARAMS && w < CCP_OUT_PARAMS2
                                                             && p % CCP_MAX_PARAMS < 2 ) )
    {
      const int c = w < CCP_OUT_PARAMS ? w - CCP_OUT_SHIFT : p / CCP_MAX_PARAMS, d0 = CCP_DEFAULT_A[k];
      int64_t   a     = int64_t( d0 );
      int       shift = CCP_DEFAULT_SHIFT;
      if( s_ccp[CCP_FOUND] != 0 && d0 != 0 )
      {
        a     = packInt2x32( ivec2( s_ccp[CCP_SCALE + 2 * c], s_ccp[CCP_SCALE + 2 * c + 1] ) );
        shift = s_ccp[CCP_SHIFT + c];
        int d = a > 0 ? -d0 : d0;
        if( shift < CCP_DEFAULT_SHIFT )
        {
          a   <<= CCP_DEFAULT_SHIFT - shift;
          shift = CCP_DEFAULT_SHIFT;
        }
        else
        {
          d <<= shift - CCP_DEFAULT_SHIFT;
        }
        a += int64_t( d );
      }
      const ivec2 pa = unpackInt2x32( p % CCP_MAX_PARAMS == 0 ? a : int64_t( 1 << ( t.bitDepth - 1 ) ) );
      v = w < CCP_OUT_PARAMS ? shift : ( ( w - CCP_OUT_PARAMS ) & 1 ) == 0 ? pa.x : pa.y;
    }
    CBufI32( P.ccpOut ).v[( first + k ) * CCP_OUT + w] = v;
  }
  cccmSync();
}

// the list of merge block t (invocation 0, after the history and the defaults): the spatial sources in order, the
// history, the defaults, each unless equal to one before it (the CPU compares a default with each the other way
// round, the same for a CCLM model), at most CCP_MAX_CANDS. With adjustment the CCCM models alone (of a type above
// CCP_TYPE_GLM4567's), at most CCP_MAX_LIST, no defaults (the on-the-fly models follow: ccpFitModels)
void ccpMergeList( Tu t )
{
  BufI32     d = BufI32( P.ccpData );
  const int  e = t.ccpMerge, numSpatial = d.v[e + CCP_MERGE_NUM_SPATIAL], numLut = s_ccp[CCP_LUT_NUM];
  const int  kind = d.v[e + CCP_MERGE_KIND];
  const bool adjust = kind != 0 && kind < CCP_KIND_DERIVED;
  const int  maxNum = adjust ? CCP_MAX_LIST : CCP_MAX_CANDS, numDefaults = adjust ? 0 : CCP_MAX_CANDS;
  int        num = 0;
  for( int s = 0; s < numSpatial + numLut + numDefaults && num < maxNum; s++ )
  {
    const int r = s < numSpatial ? d.v[e + CCP_MERGE_SPATIAL + s] : s < numSpatial + numLut
                ? s_ccp[CCP_LUT + s - numSpatial] : ccpDefaultRecords( t ) + s - numSpatial - numLut;
    if( adjust && ccpWord( r, CCP_OUT_TYPE ) <= CCP_TYPE_GLM4567 )
    {
      continue;
    }
    bool known = false;
    for( int j = 0; j < num && !known; j++ )
    {
      known = ccpEqual( r, s_ccp[CCP_LIST + j] );
    }
    if( !known )
    {
      s_ccp[CCP_LIST + num]        = r;
      s_ccp[CCP_LIST_INSIDE + num] = s < numSpatial ? ccpWord( r, CCP_OUT_INSIDE )
                                   : s < numSpatial + numLut ? ccpWord( r, CCP_OUT_INSIDE_HIST ) : 0;
      num++;
    }
  }
  s_ccp[CCP_LIST_NUM] = num;
}

// the list of decoder-derived CCP fusion block t (invocation 0): its modes
void ccpDerivedList( Tu t )
{
  const int num = t.w * t.h > 32 ? CCP_DERIVED_MODES : CCP_DERIVED_MODES - 1;
  for( int m = 0; m < num; m++ )
  {
    s_ccp[CCP_LIST + m]        = ccpDefaultRecords( t ) + CCP_DERIVED_RECORD[m];
    s_ccp[CCP_LIST_INSIDE + m] = CCP_DERIVED_INSIDE[m];
  }
  s_ccp[CCP_LIST_NUM] = num;
}

// reorderCCPCandidates' insertion sort by the costs (invocation 0): each candidate before the first one of a higher
// cost; in the pipelines with decoder-derived fusion each one's place before the sort too (CCP_LIST_ORIG)
void ccpMergeSort()
{
  for( int i = 0; i < s_ccp[CCP_LIST_NUM] && WITH_CCP_DERIVED; i++ )
  {
    s_ccp[CCP_LIST_ORIG + i] = i;
  }
  for( int i = 1; i < s_ccp[CCP_LIST_NUM]; i++ )
  {
    for( int j = 0; j < i; j++ )
    {
      if( s_ccp[CCP_LIST_COST + i] < s_ccp[CCP_LIST_COST + j] )
      {
        const int x = s_ccp[CCP_LIST + i], inside = s_ccp[CCP_LIST_INSIDE + i], cost = s_ccp[CCP_LIST_COST + i];
        const int orig = s_ccp[CCP_LIST_ORIG + i];
        for( int k = i; k > j; k-- )
        {
          s_ccp[CCP_LIST + k]        = s_ccp[CCP_LIST + k - 1];
          s_ccp[CCP_LIST_INSIDE + k] = s_ccp[CCP_LIST_INSIDE + k - 1];
          s_ccp[CCP_LIST_COST + k]   = s_ccp[CCP_LIST_COST + k - 1];
          if( WITH_CCP_DERIVED )
          {
            s_ccp[CCP_LIST_ORIG + k] = s_ccp[CCP_LIST_ORIG + k - 1];
          }
        }
        s_ccp[CCP_LIST + j]        = x;
        s_ccp[CCP_LIST_INSIDE + j] = inside;
        s_ccp[CCP_LIST_COST + j]   = cost;
        if( WITH_CCP_DERIVED )
        {
          s_ccp[CCP_LIST_ORIG + j] = orig;
        }
        break;
      }
    }
  }
}

// ----- a candidate's prediction

int g_ccpAngUsed = 0;   // the angular arrays' entries the CCCM luma buffers took (intra_rec clears them)

// whether candidate type `type` predicts as a CCLM block (CCLM, GLM's gradient patterns), else as a CCCM block
bool ccpLinear( int type ) { return ( type & ( CCP_TYPE_CCLM | CCP_TYPE_GLM0123 ) ) != 0; }

// CCCM's globals and luma buffers for record x (all invocations, ending with a barrier; the block's area in s_cclm):
// its kind, the luma buffers with its luma offset (GLM with luma's: glmIdc's pattern), its models in the parameter
// words (model 0 the first, 1 the second), its threshold, its clipping ranges (an inherited model's: CCCM_CLIP); the
// position features by its corner offsets where `ownLoc` is off (the costs; GL-CCCM's prediction), else by the
// block's (prediction with several filters: xCccmApplyModelOffset's m_cccmBlkArea - m_cccmRefArea)
void ccpCccmModel( Tu t, int x, bool ownLoc, int lid, int NT )
{
  const int type = ccpWord( x, CCP_OUT_TYPE ), mdf = ( type & CCP_TYPE_MDFCCCM ) != 0 ? ccpWord( x, CCP_OUT_MDF ) : 0;
  g_cccmGl         = ( type & CCP_TYPE_GLCCCM ) != 0;
  g_cccmMdf        = mdf;
  g_cccmNoSub      = ( type & CCP_TYPE_NSCCCM ) != 0;
  g_cccmGlmLuma    = ( type & CCP_TYPE_GLM4567 ) != 0;
  g_cccmGlmPattern = ccpWord( x, CCP_OUT_GLM_IDC ) - NUM_GLM_PATTERN;
  g_cccmFusion     = false;
  g_cccmCflm       = false;
  g_cccmBoth       = false;
  g_cccmSet        = 0;
  g_cccmShift      = g_cccmNoSub ? 1 : 0;
  g_cccmParams     = g_cccmGlmLuma ? GLM_PARAMS
                   : g_cccmNoSub ? ( P.cclmFilter == CCLM_FILTER_AVERAGE ? 9 : 11 )
                   : mdf == 0 ? 7 : mdf == 1 ? 10 : 11;
  cccmLoadArea();
  g_cccmLumaOffset = ccpWord( x, CCP_OUT_LUMA_OFFSET );
  const bool loc = ( g_cccmGl || mdf != 0 ) && !ownLoc;
  g_cccmLocDX      = loc ? ccpWord( x, CCP_OUT_COR_OFF ) - g_cccmSizeX : 0;
  g_cccmLocDY      = loc ? ccpWord( x, CCP_OUT_COR_OFF + 1 ) - g_cccmSizeY : 0;
  for( int b = 0; b < cccmNumBuffers(); b++ )
  {
    cccmBuildLuma( t, b, lid, NT );
  }
  if( g_cccmSharedLuma )
  {
    g_ccpAngUsed = max( g_ccpAngUsed, ( cccmBufferSize() + 3 ) >> 2 );
  }
  const bool two = ( type & CCP_TYPE_MMLM ) != 0;
  for( int i = lid; i < 4 * g_cccmParams; i += NT )   // (model, component, parameter)
  {
    const int m = i / ( 2 * g_cccmParams ), c = ( i / g_cccmParams ) & 1, p = i % g_cccmParams;
    cccmStore( cccmParamWord( m, c, p ), m == 0 || two ? ccpParam( x, m == 1, c, p ) : int64_t( 0 ) );
  }
  if( lid < 4 )   // (model, component)
  {
    const int   r    = CCP_OUT_RANGE + 3 * lid;
    const ivec2 clip = cccmClipRange( ccpWord( x, r ), ccpWord( x, r + 1 ), ccpWord( x, r + 2 ) != 0, true,
                                      t.w * t.h, t.bitDepth );
    s_cclm[CCCM_CLIP + lid * 2]     = clip.x;
    s_cclm[CCCM_CLIP + lid * 2 + 1] = clip.y;
  }
  if( lid == 0 )
  {
    s_cclm[CCCM_THRESHOLD] = two ? ccpWord( x, CCP_OUT_Y_THRES ) : 0;
  }
  cccmSync();
}

// record x's prediction of component c (0: Cb, 1: Cr) as a CCLM model at the template sample of the row above the
// block (top) or the column left of it, index q: the first model alone (xUpdateOffsetsAndGetCostCCLM's modelNum 1),
// neither clipped nor a Pel yet
int ccpLinearTemplate( Tu t, int x, int c, bool top, int q )
{
  const int lx = t.x << 1, ly = t.y << 1, glm = ccpWord( x, CCP_OUT_GLM_IDC );
  const int luma = top ? cclmLumaAbove( lx, ly, q, s_cclm[CCLM_LEFT] != 0, glm )
                       : cclmLumaLeft( lx, ly, q, s_cclm[CCLM_ABOVE] != 0, glm );
  const int a = int( ccpParam( x, false, c, 0 ) ), b = int( ccpParam( x, false, c, 1 ) );
  return ( ( a * luma ) >> ccpWord( x, CCP_OUT_SHIFT + c ) ) + b;
}

// record x's CCLM model of block t's component into CCLM's slot 0 (invocation 0): its first model's a, b and shift,
// its second's, the threshold between them
void ccpLoadLinear( Tu t, int x )
{
  const int m = CCLM_MODEL, c = t.comp - 1;
  s_cclm[m + CCLM_A]      = int( ccpParam( x, false, c, 0 ) );
  s_cclm[m + CCLM_B]      = int( ccpParam( x, false, c, 1 ) );
  s_cclm[m + CCLM_SHIFT]  = ccpWord( x, CCP_OUT_SHIFT + c );
  s_cclm[m + CCLM_A2]     = int( ccpParam( x, true, c, 0 ) );
  s_cclm[m + CCLM_B2]     = int( ccpParam( x, true, c, 1 ) );
  s_cclm[m + CCLM_SHIFT2] = ccpWord( x, CCP_OUT_SHIFT2 + c );
  s_cclm[m + CCLM_THRES]  = ccpWord( x, CCP_OUT_Y_THRES );
}

// ----- decoder-derived CCP fusion (JVET-AG0154): its CCLM models, the probe lines, the choice

// the CCLM models of decoder-derived CCP fusion block t (tmCostDecoderDerivedCcp's of LM_CHROMA_IDX: each component's
// xGetLMParametersLMS, no slope adjustment; invocation 0, after cclmAvailability and before cccmArea, whose words
// overlap CCLM's slots) into the first of the block's spare records: its own component's from its references (s_ref[0],
// slot 0), the other's from the other's (s_ref[1], slot 1); the type, 2 parameters, per component the shift, the slope
// and the offset (cclmModelToCcpParams), every other word 0
void ccpDerivedLinear( Tu t, int predStride )
{
  const int lx = t.x << 1, ly = t.y << 1, base = ccpDefaultRecords( t ) * CCP_OUT;
  for( int k = 0; k < 2; k++ )   // (cclmModel reads the references and the area of t alone, not its component)
  {
    cclmModel( t, lx, ly, predStride, k, k );
  }
  CBufI32 o = CBufI32( P.ccpOut );
  for( int w = 0; w < CCP_OUT; w++ )
  {
    o.v[base + w] = 0;
  }
  o.v[base + CCP_OUT_TYPE]       = CCP_TYPE_CCLM;
  o.v[base + CCP_OUT_NUM_PARAMS] = 2;
  for( int c = 0; c < 2; c++ )
  {
    const int m = CCLM_MODEL + ( c == t.comp - 1 ? 0 : CCLM_SLOT );
    o.v[base + CCP_OUT_SHIFT + c] = s_cclm[m + CCLM_SHIFT];
    ccpStoreParam( base, false, c, 0, int64_t( s_cclm[m + CCLM_A] ) );
    ccpStoreParam( base, false, c, 1, int64_t( s_cclm[m + CCLM_B] ) );
  }
}

// where decoder-derived fusion block t's probe samples (decoder-derived merge fusion's and blend-model merge fusion's
// template samples) are, the costs' step by step, Cb's then Cr's, then the CCP merge fusion's CCCM prediction's
// (intra_cccm.glsl's cccmPrediction: slot CCP_MAX_CANDS), then the pairs' words: in the workgroup's scratch before
// list[1]'s (or the CCCM) prediction at its end, which the block's own prediction may overlap later (GpuIntraRec.cpp's
// ccpDerivedScratch)
int ccpProbeBase( Tu t )
{
  return cccmFusionPredBase( t ) - ( CCP_MAX_CANDS + 1 ) * 2 * ( t.w + t.h ) - CCP_PAIRS_MAX * CCP_PAIR_WORDS;
}

int ccpPairBase( Tu t ) { return ccpProbeBase( t ) + ( CCP_MAX_CANDS + 1 ) * 2 * ( t.w + t.h ); }

int ccpFusionTemplateBase( Tu t ) { return ccpProbeBase( t ) + CCP_MAX_CANDS * 2 * ( t.w + t.h ); }

// the position of tap k (0 .. 8: 3 x 3, the line farthest from the block first) of decoder-derived fusion's probe
// sample q of the row above block t (top) or of the column left of it with LBCCP's inside filter
// (filterPredInsideProbeLine), tap 4 the sample itself: its position in the reference area to predict, or with `pad`
// the picture's position of a reconstructed sample. Its three lines: the one two away from the block where a CU is
// three away (else the next one again), the next one, the block's first; before a line's start the reconstruction
// beside it where the block has a CU there (else the line's first sample), after its end the line's last sample
ivec2 ccpProbeTap( Tu t, bool top, int q, int k, out bool pad )
{
  OwnerMap    om = OwnerMap( BufI32( P.owner[t.comp] ), P.ownerW[1], P.ownerH[1], P.unitLog2X[1], P.unitLog2Y[1] );
  const int   across = k / 3, along = q + k % 3 - 1;
  const ivec2 start = top ? ivec2( t.x - 1, t.y + across - 2 ) : ivec2( t.x + across - 2, t.y - 1 );
  pad = along < 0 && decodedBefore( om, start.x, start.y, g_ti );
  if( pad )
  {
    return start;
  }
  const ivec2 three = top ? ivec2( t.x, t.y - 3 ) : ivec2( t.x - 3, t.y );
  const bool  far   = across == 0 && decodedBefore( om, three.x, three.y, g_ti );
  const int   off   = across == 2 ? 0 : far ? -2 : -1, i = clamp( along, 0, ( top ? t.w : t.h ) - 1 );
  return top ? ivec2( g_cccmSizeX + i, g_cccmSizeY + off ) : ivec2( g_cccmSizeX + off, g_cccmSizeY + i );
}

// the modes of pair p of num modes, in the CPU's order: (0, 1), (0, 2) .. (0, num - 1), (1, 2) ..
ivec2 ccpDerivedPair( int p, int num )
{
  int a = 0;
  while( p >= num - 1 - a )
  {
    p -= num - 1 - a;
    a++;
  }
  return ivec2( a, a + 1 + p );
}

// decoder-derived CCP fusion's choice (decoderDerivedCcp; all invocations, ending with a barrier) from its modes'
// costs and probe samples: the first mode of the lowest cost alone where that cost is at most (W + H) / 2; else the
// pair of modes (in the CPU's order) whose blend of the probe samples costs least (ddccpFusionTemplateCost: Cb's and
// Cr's SAD of (3 P + P' + 2) >> 2, P the samples of the mode of the lower cost, of the first of equal ones), where it
// costs less than the mode alone: list[0] the pair's mode of the lower cost (the first of equal ones), list[1] the
// other
void ccpDerivedChoice( Tu t, int lid, int NT )
{
  const int  num = s_ccp[CCP_LIST_NUM], numPairs = num * ( num - 1 ) / 2, W = t.w, len = t.w + t.h;
  const bool above = s_cclm[CCLM_POS_ABOVE] != 0, left = s_cclm[CCLM_POS_LEFT] != 0;
  if( lid < numPairs )
  {
    s_ccp[CCP_PAIR_COST + lid] = 0;
  }
  cccmSync();   // (and the probe samples)
  for( int i = lid; i < numPairs * 2 * len; i += NT )
  {
    const int  p = i / ( 2 * len ), s = i % ( 2 * len ), c = s / len, pos = s % len, q = pos < W ? pos : pos - W;
    const bool top = pos < W;
    if( top ? above : left )
    {
      const ivec2 ab = ccpDerivedPair( p, num );
      const int   pa = CBufI32( P.scratch ).v[ccpProbeBase( t ) + ab.x * 2 * len + s];
      const int   pb = CBufI32( P.scratch ).v[ccpProbeBase( t ) + ab.y * 2 * len + s];
      const bool  bFirst = s_ccp[CCP_LIST_COST + ab.y] < s_ccp[CCP_LIST_COST + ab.x];
      const int   v = bFirst ? ( 3 * pb + pa + 2 ) >> 2 : ( 3 * pa + pb + 2 ) >> 2;
      const int   reco = top ? recoAt( c + 1, t.x + q, t.y - 1 ) : recoAt( c + 1, t.x - 1, t.y + q );
      atomicAdd( s_ccp[CCP_PAIR_COST + p], abs( reco - v ) );
    }
  }
  SYNC();
  if( lid == 0 )
  {
    int best = 0;
    for( int m = 1; m < num; m++ )
    {
      best = s_ccp[CCP_LIST_COST + m] < s_ccp[CCP_LIST_COST + best] ? m : best;
    }
    int first = best, second = -1;
    if( s_ccp[CCP_LIST_COST + best] > ( W + t.h ) >> 1 )
    {
      int bestPair = 0;
      for( int p = 1; p < numPairs; p++ )
      {
        bestPair = s_ccp[CCP_PAIR_COST + p] < s_ccp[CCP_PAIR_COST + bestPair] ? p : bestPair;
      }
      if( s_ccp[CCP_LIST_COST + best] > s_ccp[CCP_PAIR_COST + bestPair] )
      {
        const ivec2 ab = ccpDerivedPair( bestPair, num );
        first  = s_ccp[CCP_LIST_COST + ab.y] < s_ccp[CCP_LIST_COST + ab.x] ? ab.y : ab.x;
        second = ab.x + ab.y - first;
      }
    }
    s_ccp[CCP_DERIVED_FIRST]  = first;
    s_ccp[CCP_DERIVED_SECOND] = second;
  }
  SYNC();
}

// ----- decoder-derived CCP merge fusion (JVET-AG0154 with JVET-AN0168's regression blend)

// list[1]'s weight (1 .. 31 of 32) at (x, y) relative to the block by a blend model (AffineBlendingModel::compute: in
// 32 bits, as the CPU's ints)
int ccpBlendWeight( int p0, int p1, int p2, int shift, int offset, int x, int y )
{
  return clamp( ( p0 * x + p1 * y + p2 + offset ) >> shift, 1, 31 );
}

// the cost and blend model of the pair of candidates whose template samples are the a-th and b-th in the scratch (one
// invocation; reorderCCPCandidates with useBlendModelDdccp, always) into the pair's words at `dst`: the SAD of the
// average of their samples (xGetCostCCPFusion) or, where lower (or always without `compare`: merge fusion's
// ccpmComputeBlendFusionModelJoint), of their blend by xComputeBlendFusionModelJoint's
// model - list[1]'s weight linear in the position, fitted to both components' template samples by solve1: the
// features (P1 - P0) x where the block has a CU above, (P1 - P0) y where it has one left (x and y one less than the
// sample's own along its line: AML_MERGE_TEMPLATE_SIZE) and P1 - P0, the target the reconstruction minus P0, each a
// Pel; its parameters as ints (the CPU sizes them before it sets them, so it scales them by 1), its shift DECIM_BITS
// - 5; equal weights (16) where those at the block's corners differ by 4 at most
void ccpBlendModel( Tu t, int a, int b, int dst, bool compare )
{
  const bool above = s_cclm[CCLM_POS_ABOVE] != 0, left = s_cclm[CCLM_POS_LEFT] != 0;
  const int  W = t.w, H = t.h, len = W + H, bitDepth = t.bitDepth;
  const int  pa = ccpProbeBase( t ) + a * 2 * len, pb = ccpProbeBase( t ) + b * 2 * len;
  CBufI32    s = CBufI32( P.scratch );
  int64_t e[3][4];   // the normal equations' upper triangle and right-hand side, then the elimination's rows
  for( int i = 0; i < 12; i++ )
  {
    e[i / 4][i % 4] = 0;
  }
  int n = 0, costAverage = 0;
  for( int k = 0; k < 2 * len; k++ )   // (component k / len, template sample k % len: the row above, then the column)
  {
    const int  i = k % len, q = i < W ? i : i - W, c = k / len;
    const bool top = i < W;
    if( top ? above : left )
    {
      const int p0 = s.v[pa + k], p1 = s.v[pb + k], d = p1 - p0;
      const int reco = top ? recoAt( c + 1, t.x + q, t.y - 1 ) : recoAt( c + 1, t.x - 1, t.y + q );
      int f[4];
      f[0] = above ? int( int16_t( d * ( top ? q - 1 : -1 ) ) ) : 0;
      f[1] = left ? int( int16_t( d * ( top ? -1 : q - 1 ) ) ) : 0;
      f[2] = int( int16_t( d ) );
      f[3] = int( int16_t( reco - p0 ) );
      for( int r = 0; r < 3; r++ )
      {
        for( int col = r; col < 4; col++ )
        {
          e[r][col] += int64_t( f[r] * f[col] );
        }
      }
      n++;
      costAverage += abs( reco - ( ( p0 + p1 + 1 ) >> 1 ) );
    }
  }
  // solve1: scaled to 28 bits over the samples, regularized, eliminated with xDivide's scales, substituted back
  const int matrixShift = 28 - 2 * bitDepth - cccmCeilLog2( max( n, 1 ) );
  for( int i = 0; i < 12; i++ )   // (the upper triangle and the right-hand side scaled, then the lower triangle)
  {
    const int r = i / 4, col = i % 4;
    if( col >= r )
    {
      e[r][col] = matrixShift > 0 ? e[r][col] << matrixShift : e[r][col] >> -matrixShift;
    }
  }
  for( int r = 0; r < 3; r++ )
  {
    for( int col = 0; col < r; col++ )
    {
      e[r][col] = e[col][r];
    }
    e[r][r] += int64_t( 2 << ( bitDepth - 8 ) );
  }
  for( int piv = 0; piv < 3; piv++ )
  {
    int scale, round, shift;
    cccmDivScale( e[piv][piv] < int64_t( 1 ) ? int64_t( 1 ) : e[piv][piv], scale, round, shift );
    for( int j = piv + 1; j < 4; j++ )
    {
      e[piv][j] = ( e[piv][j] * int64_t( scale ) + int64_t( round ) ) >> shift;
    }
    for( int r = piv + 1; r < 3; r++ )
    {
      const int64_t f = e[r][piv];
      for( int j = piv + 1; j < 4; j++ )
      {
        e[r][j] -= ( f * e[piv][j] + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM;
      }
    }
  }
  int64_t x[3];
  for( int r = 2; r >= 0; r-- )
  {
    x[r] = e[r][3];
    for( int j = r + 1; j < 3; j++ )
    {
      x[r] -= ( e[r][j] * x[j] + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM;
    }
  }
  int p0 = int( x[0] ), p1 = int( x[1] ), p2 = int( x[2] ), shift = CCCM_DECIM - 5, offset = 1 << ( shift - 1 );
  int lo = 31, hi = 1;
  for( int corner = 0; corner < 4; corner++ )
  {
    const int wgt = ccpBlendWeight( p0, p1, p2, shift, offset, ( corner & 1 ) * ( W - 1 ), ( corner >> 1 ) * ( H - 1 ) );
    lo = min( lo, wgt );
    hi = max( hi, wgt );
  }
  if( hi - lo <= 4 )   // (AffineBlendingModel::average)
  {
    p0     = 0;
    p1     = 0;
    p2     = 16;
    shift  = 0;
    offset = 0;
  }
  int costModel = 0;
  for( int k = 0; k < 2 * len; k++ )
  {
    const int  i = k % len, q = i < W ? i : i - W, c = k / len;
    const bool top = i < W;
    if( top ? above : left )
    {
      const int p0s = s.v[pa + k], p1s = s.v[pb + k];
      const int reco = top ? recoAt( c + 1, t.x + q, t.y - 1 ) : recoAt( c + 1, t.x - 1, t.y + q );
      const int wgt = top ? ccpBlendWeight( p0, p1, p2, shift, offset, q, -1 )
                          : ccpBlendWeight( p0, p1, p2, shift, offset, -1, q );
      costModel += abs( reco - int( int16_t( ( ( 32 - wgt ) * p0s + wgt * p1s + 16 ) >> 5 ) ) );
    }
  }
  const bool blends = !compare || costModel < costAverage;
  s.v[dst + CCP_PAIR_COST_W]     = blends ? costModel : costAverage;
  s.v[dst + CCP_PAIR_VALID]      = blends ? 1 : 0;
  s.v[dst + CCP_PAIR_PARAMS]     = p0;
  s.v[dst + CCP_PAIR_PARAMS + 1] = p1;
  s.v[dst + CCP_PAIR_PARAMS + 2] = p2;
  s.v[dst + CCP_PAIR_SHIFT]      = shift;
  s.v[dst + CCP_PAIR_OFFSET]     = offset;
}

// decoder-derived CCP merge fusion's pair (reorderCCPCandidates' fusionList; all invocations, ending with a barrier),
// after the sort: per pair of the first 9 candidates (in the CPU's order) its cost and model (ccpBlendModel), then the
// pairs ranked by cost into P.ccpFusionPairs slots (each before the first slot of a higher cost), the entry's slot's
// pair as list[0] (its first candidate) and list[1] (its second), its model into CCP_BLEND. (A slot no pair reached,
// which the encoder never signals, takes the first candidate twice, averaged.)
void ccpDdFusionChoice( Tu t, int lid, int NT )
{
  const int num = min( s_ccp[CCP_LIST_NUM], CCP_FUSION_CANDS ), numPairs = num * ( num - 1 ) / 2;
  cccmSync();   // (the template samples)
  for( int p = lid; p < numPairs; p += NT )
  {
    const ivec2 ab = ccpDerivedPair( p, num );
    ccpBlendModel( t, s_ccp[CCP_LIST_ORIG + ab.x], s_ccp[CCP_LIST_ORIG + ab.y], ccpPairBase( t ) + p * CCP_PAIR_WORDS,
                   true );
  }
  cccmSync();
  if( lid == 0 )
  {
    CBufI32   s = CBufI32( P.scratch );
    const int slots = min( P.ccpFusionPairs, CCP_FUSION_SLOTS_MAX );
    int slotPair[CCP_FUSION_SLOTS_MAX], slotCost[CCP_FUSION_SLOTS_MAX];
    for( int m = 0; m < CCP_FUSION_SLOTS_MAX; m++ )
    {
      slotPair[m] = -1;
      slotCost[m] = 0x7fffffff;
    }
    for( int p = 0; p < numPairs; p++ )
    {
      const int cost = s.v[ccpPairBase( t ) + p * CCP_PAIR_WORDS + CCP_PAIR_COST_W];
      for( int m = 0; m < slots; m++ )
      {
        if( cost < slotCost[m] )
        {
          for( int k = slots - 1; k > m; k-- )
          {
            slotCost[k] = slotCost[k - 1];
            slotPair[k] = slotPair[k - 1];
          }
          slotCost[m] = cost;
          slotPair[m] = p;
          break;
        }
      }
    }
    const int   p  = slotPair[clamp( BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_INDEX], 0, slots - 1 )];
    const ivec2 ab = p >= 0 ? ccpDerivedPair( p, num ) : ivec2( 0 );
    s_ccp[CCP_DERIVED_FIRST]  = ab.x;
    s_ccp[CCP_DERIVED_SECOND] = ab.y;
    for( int w = 0; w < CCP_PAIR_WORDS; w++ )
    {
      s_ccp[CCP_BLEND + w] = p >= 0 ? s.v[ccpPairBase( t ) + p * CCP_PAIR_WORDS + w] : 0;
    }
  }
  SYNC();
}

// decoder-derived CCP merge fusion's prediction at (x, y) of block t (storeRegion's, cccmFusionBlend): list[0]'s p0
// blended with list[1]'s p1 by the chosen pair's model, ( ( 32 - w ) p0 + w p1 + 16 ) >> 5, else averaged
// (applyBlendFusion)
int ccpFusionBlend( int x, int y, int p0, int p1 )
{
  if( s_ccp[CCP_BLEND + CCP_PAIR_VALID] == 0 )
  {
    return ( p0 + p1 + 1 ) >> 1;
  }
  const int wgt = ccpBlendWeight( s_ccp[CCP_BLEND + CCP_PAIR_PARAMS], s_ccp[CCP_BLEND + CCP_PAIR_PARAMS + 1],
                                  s_ccp[CCP_BLEND + CCP_PAIR_PARAMS + 2], s_ccp[CCP_BLEND + CCP_PAIR_SHIFT],
                                  s_ccp[CCP_BLEND + CCP_PAIR_OFFSET], x, y );
  return int( int16_t( ( ( 32 - wgt ) * p0 + wgt * p1 + 16 ) >> 5 ) );
}

// ----- models the block fits itself; adjustment (JVET-AL0126): the chosen model's coefficients moved

// record words w of a model just fitted (two: MMLM; its parameters in the parameter words, its ranges in
// CCCM_MODEL_RANGE): cccmModelToCcpParams' fields - plain CCCM or GL-CCCM (g_cccmGl), the threshold of two models, the
// block's own luma offset and corner offsets, 7 parameters, the template chroma's ranges - every other word 0
int ccpFlyWord( int w, bool two )
{
  if( w == CCP_OUT_TYPE ) return ( g_cccmGl ? CCP_TYPE_GLCCCM : CCP_TYPE_CCCM ) | ( two ? CCP_TYPE_MMLM : 0 );
  if( w == CCP_OUT_Y_THRES ) return two ? s_cclm[CCCM_THRESHOLD] : 0;
  if( w == CCP_OUT_LUMA_OFFSET ) return g_cccmLumaOffset;
  if( w == CCP_OUT_COR_OFF ) return g_cccmSizeX;
  if( w == CCP_OUT_COR_OFF + 1 ) return g_cccmSizeY;
  if( w == CCP_OUT_NUM_PARAMS ) return 7;
  if( w >= CCP_OUT_RANGE && w < CCP_OUT_RANGE + ( two ? 12 : 6 ) )   // (per model and component: min, max, change)
  {
    const int i = w - CCP_OUT_RANGE, k = i / 3, r = CCCM_MODEL_RANGE + 2 * k;
    const int lo = s_cclm[r], hi = s_cclm[r + 1];
    return i % 3 == 0 ? lo : i % 3 == 1 ? hi : hi > lo ? 1 : 0;
  }
  const int pw = w - CCP_OUT_PARAMS, m = pw / ( 4 * CCP_MAX_PARAMS ), c = ( pw / ( 2 * CCP_MAX_PARAMS ) ) & 1;
  const int p = ( pw % ( 2 * CCP_MAX_PARAMS ) ) >> 1;
  if( pw >= 0 && pw < 8 * CCP_MAX_PARAMS && p < 7 && ( m == 0 || two ) )
  {
    const ivec2 v = unpackInt2x32( cccmLoad( cccmParamWord( m, c, p ) ) );
    return ( pw & 1 ) == 0 ? v.x : v.y;
  }
  return 0;
}

// the models merge block t fits itself (all invocations, ending with a barrier), each into a record among the block's
// spare ones, of the block's own luma buffer and luma offset, plain CCCM unless said otherwise:
// - with adjustment the on-the-fly models (deriveCcpMergeAdjustCands; the first CCP_FLY_MAX records), valid where one
//   of its coefficients but the bias is not 0 (CCP_FLY_VALID), over the whole template: two models split at
//   xCccmCalcRefAver's threshold (the template's mean luma), one model, one of the template's two lines next to the
//   block (trainingRange 2), one of the reference samples alone (useRefSampOnly), two of them, two of the two lines.
//   (The CPU selects the template by the block's position - the left part in the picture's top row, the part above in
//   its left column - which is the whole template there: a merge block's own template selection, 0, gives the same.)
// - decoder-derived CCP fusion's CCCM models (tmCostDecoderDerivedCcp; after its CCLM record): one model of the two
//   lines, two models over the whole template split at the block's mean luma (xCccmCalcRefAver's trainingRange 2, as
//   LBCCP's second set), where the block has more than 32 samples GL-CCCM's one over the whole template
void ccpFitModels( Tu t, bool derived, int lid, int NT )
{
  g_cccmGl         = false;
  g_cccmMdf        = 0;
  g_cccmNoSub      = false;
  g_cccmGlmLuma    = false;
  g_cccmFusion     = false;
  g_cccmCflm       = false;
  g_cccmBoth       = false;
  g_cccmSet        = 0;
  g_cccmShift      = 0;
  g_cccmParams     = 7;
  g_cccmLocDX      = 0;
  g_cccmLocDY      = 0;
  cccmLoadArea();
  cccmBuildLuma( t, 0, lid, NT );
  if( g_cccmSharedLuma )
  {
    g_ccpAngUsed = max( g_ccpAngUsed, ( cccmBufferSize() + 3 ) >> 2 );
  }
  if( lid < 2 )
  {
    s_red[2 + lid] = 0;
  }
  SYNC();
  // the threshold of two models: the template's mean luma, decoder-derived fusion's the block's
  const ivec4 r      = cccmTemplate( t, false );
  const ivec2 counts = cccmTemplateCounts( r );
  const int   num    = derived ? t.w * t.h : counts.x + counts.y;
  int sum = 0;
  for( int i = lid; i < num; i += NT )
  {
    const ivec2 p = derived ? ivec2( g_cccmSizeX + i % t.w, g_cccmSizeY + i / t.w )
                            : cccmTemplateSample( r, counts, i );
    sum += cccmLuma( p.x, p.y );
  }
  const int total = wgSum2( sum, 0, 2 ).x, shift = findMSB( t.w ) + findMSB( t.h );
  if( lid == 0 )
  {
    s_cclm[CCCM_THRESHOLD] = derived ? ( total + ( 1 << ( shift - 1 ) ) ) >> shift
                           : num == 0 ? 1 << ( t.bitDepth - 1 ) : cccmDivideLowPrec( total, num );
  }
  const int numFits = !derived ? CCP_FLY_MAX : t.w * t.h > 32 ? 3 : 2;
  for( int k = 0; k < numFits; k++ )
  {
    const bool two   = derived ? k == 1 : k == 0 || k >= 4;
    g_ccpFlyTwoLines = derived ? k == 0 : k == 2 || k == 5;
    g_ccpFlyRefOnly  = !derived && ( k == 3 || k == 4 );
    g_cccmGl         = derived && k == 2;
    for( int m = 0; m < ( two ? 2 : 1 ); m++ )
    {
      cccmAccumulate( t, m, two, lid, NT );
      cccmSolve( t, m, lid );
      if( lid < 2 )   // (the next model's cccmAccumulate resets CCCM_RANGE in the same invocations)
      {
        s_cclm[CCCM_MODEL_RANGE + 4 * m + 2 * lid]     = s_cclm[CCCM_RANGE + 2 * lid];
        s_cclm[CCCM_MODEL_RANGE + 4 * m + 2 * lid + 1] = s_cclm[CCCM_RANGE + 2 * lid + 1];
      }
    }
    SYNC();
    bool valid = false;
    for( int i = 0; i < 4 * 6 && !valid; i++ )   // (model, component, coefficient but the bias)
    {
      valid = ( i < 12 || two ) && cccmLoad( cccmParamWord( i / 12, ( i / 6 ) & 1, i % 6 ) ) != int64_t( 0 );
    }
    if( lid == 0 && !derived )
    {
      s_ccp[CCP_FLY_VALID + k] = valid ? 1 : 0;
    }
    const int base = ( ccpDefaultRecords( t ) + k + ( derived ? 1 : 0 ) ) * CCP_OUT;
    for( int w = lid; w < CCP_OUT; w += NT )
    {
      CBufI32( P.ccpOut ).v[base + w] = ccpFlyWord( w, two );
    }
    SYNC();   // (the next fit overwrites the parameter words)
  }
  g_ccpFlyTwoLines = false;
  g_ccpFlyRefOnly  = false;
  g_cccmGl         = false;
  cccmSync();
}

// adjustment's deltas in 1/32 of a coefficient's unit, tried in this order: of the co-located luma sample's
// coefficient (pass 1), of its neighbours' (pass 2)
const int CCP_ADJUST_DELTAS = 8;
const int CCP_ADJUST_DELTA[2 * CCP_ADJUST_DELTAS] = int[]( 1, -1, 3, -3, 5, -5, 7, -7, 1, -1, 2, -2, 3, -3, 4, -4 );

// coefficient v moved by delta away from 0 (towards it where positive), in 1/32 of the coefficient's unit
int64_t ccpAdjusted( int64_t v, int delta )
{
  return v + int64_t( ( v > int64_t( 0 ) ? -delta : delta ) << ( CCCM_DECIM - 5 ) );
}

// the coefficients pass `pass` of record x's models moves (adjustCCPCandidates' pMinIdx .. pMaxIdx - 1): the co-located
// sample's, else its neighbours' by the kind: plain CCCM's four, GL-CCCM's two gradients, without subsampling the
// spatial taps (4 of 9 parameters, of the cross filter 5, else 6), several filters' 3 of mode 1, else 2
ivec2 ccpAdjustCoefficients( int pass, int x )
{
  const int type = ccpWord( x, CCP_OUT_TYPE );
  const int end = pass == 1 ? 1
                : ( type & CCP_TYPE_CCCM ) != 0 ? 5 : ( type & CCP_TYPE_GLCCCM ) != 0 ? 3
                : ( type & CCP_TYPE_NSCCCM ) != 0
                  ? ( ccpWord( x, CCP_OUT_NUM_PARAMS ) == 9 ? 4 : P.cclmFilter == CCLM_FILTER_CROSS ? 5 : 6 )
                : ( type & CCP_TYPE_MDFCCCM ) != 0 ? ( ccpWord( x, CCP_OUT_MDF ) == 1 ? 4 : 3 ) : 2;
  return ivec2( pass == 1 ? 0 : 1, end );
}

// the adjustment of merge block t's chosen model x, flag `flag` (adjustCCPCandidates; its models in the parameter
// words, ccpCccmModel's with the candidate's position offsets; all invocations, ending with a barrier): of the block's
// component, per model, a delta for the coefficients ccpAdjustCoefficients gives - the one of the lowest template
// cost (xGetCostCCCM: the component's SAD on the template's row above the block and column left of it, each where the
// block has a CU there, over the model's samples alone with two models) among those tried in order: the first two
// decide between the positive and the negative ones (equal costs: the next two decide again), then those of the
// chosen sign only while not costlier than its first; ties take the later delta; a model of two without template
// samples takes the set's last delta. Flag 2 moves the co-located sample's coefficient first (a pass of flag 1), then
// its neighbours'
void ccpMergeAdjust( Tu t, int x, int flag, int lid, int NT )
{
  const int  c = t.comp - 1, W = t.w, len = t.w + t.h, bitDepth = t.bitDepth;
  const bool two = ( ccpWord( x, CCP_OUT_TYPE ) & CCP_TYPE_MMLM ) != 0;
  const bool above = s_cclm[CCLM_POS_ABOVE] != 0, left = s_cclm[CCLM_POS_LEFT] != 0;
  int64_t base[CCCM_MAX_PARAMS];   // (invocation 0's: the coefficients before the model's deltas)
  for( int pass = flag == 2 ? 1 : flag; pass <= flag; pass++ )
  {
    const ivec2 coef = ccpAdjustCoefficients( pass, x );
    const int   set  = ( pass - 1 ) * CCP_ADJUST_DELTAS;
    for( int m = 0; m < ( two ? 2 : 1 ); m++ )
    {
      for( int p = coef.x; p < coef.y && lid == 0; p++ )
      {
        base[p] = cccmLoad( cccmParamWord( m, c, p ) );
      }
      // (the deltas alternate in sign, positive first: the costs of the first positive and the first negative one
      // decide which sign continues, ties those of the next two)
      int  bestCost = 0x7fffffff, bestDelta = 0, posCost = 0x7fffffff, negCost = 0x7fffffff;
      bool onlyPos = false, onlyNeg = false;
      for( int i = 0; i < CCP_ADJUST_DELTAS; i++ )
      {
        if( ( onlyPos && ( i & 1 ) != 0 ) || ( onlyNeg && ( i & 1 ) == 0 ) )
        {
          continue;
        }
        const int delta = CCP_ADJUST_DELTA[set + i];
        for( int p = coef.x; p < coef.y && lid == 0; p++ )
        {
          cccmStore( cccmParamWord( m, c, p ), ccpAdjusted( base[p], delta ) );
        }
        if( lid < 2 )
        {
          s_red[2 + lid] = 0;
        }
        SYNC();
        int sad = 0, samples = 0;
        for( int s = lid; s < len; s += NT )
        {
          const bool top = s < W;
          const int  q = top ? s : s - W;
          const int  px = top ? g_cccmSizeX + q : g_cccmSizeX - 1, py = top ? g_cccmSizeY - 1 : g_cccmSizeY + q;
          const bool own = !two || ( cccmColocated( px, py ) > s_cclm[CCCM_THRESHOLD] ? 1 : 0 ) == m;
          if( ( top ? above : left ) && own )
          {
            const int reco = top ? recoAt( c + 1, t.x + q, t.y - 1 ) : recoAt( c + 1, t.x - 1, t.y + q );
            sad += abs( reco - cccmPredict( 0, c, px, py, two, bitDepth ) );
            samples++;
          }
        }
        const ivec2 total = wgSum2( sad, samples, 2 );
        if( two && total.y == 0 )
        {
          bestDelta = CCP_ADJUST_DELTA[set + CCP_ADJUST_DELTAS - 1];
          break;
        }
        const int cost = total.x;
        if( !onlyPos && !onlyNeg )
        {
          if( i == 0 || i == 2 )
          {
            posCost = cost;
          }
          else if( i == 1 || i == 3 )
          {
            negCost = cost;
            onlyPos = posCost < negCost;
            onlyNeg = negCost < posCost;
          }
        }
        else if( onlyPos ? cost > posCost : cost > negCost )
        {
          break;
        }
        if( cost <= bestCost )
        {
          bestCost  = cost;
          bestDelta = delta;
        }
      }
      for( int p = coef.x; p < coef.y && lid == 0; p++ )
      {
        cccmStore( cccmParamWord( m, c, p ), ccpAdjusted( base[p], bestDelta ) );
      }
      SYNC();
    }
  }
}

// merge block t's prediction (all invocations, ending with a barrier): the list, then per step first each candidate's
// template cost (xGetOneCCPCandCost: Cb's and Cr's SADs of the row above the block and the column left of it, each
// where the block has a CU there), then the choice (the sorted list's entry at idxNonLocalCCP - 1) and the prediction.
// Both blocks of the CU rank the list alike (the Cr block cannot wait for the Cb block: intra_deps' ccpMergeDeps); the
// Cb block writes the CU's record (the candidate's, with its inside filter flag both as the CU's and as the history's:
// the CPU copies the candidate into curCand, which the history takes; a merge CU has no LBCCP flag of its own). t
// becomes the LM block of the chosen model's type that cclmPrediction predicts: a CCLM model in s_cclm's slot 0 (its
// mode MMLM's with two models, its pattern), else its prediction in s_tmp or the scratch (a CCCM block's), with the
// inside filter flag.
// A block of decoder-derived CCP fusion ranks its modes by their probe samples' costs (decoderDerivedCclmCost: the
// CCLM model's clipped; decoderDerivedCccmCost), which the scratch keeps for the choice (ccpDerivedChoice), then
// predicts list[1] where it blends two (predDecoderDerivedIntraCCCMFusions: through its inside filter, into the fusion
// slot), then list[0]. The CU's record is list[0]'s model with the inside filter flags 0 (DecCu clears the PU's flag
// before CU::saveCcInsideFilterFlagInCCP would copy it).
// A merge block of decoder-derived merge fusion keeps its candidates' template samples alike, chooses its pair after
// the sort (ccpDdFusionChoice) and predicts like a decoder-derived fusion block (predDecoderDerivedCcpMergeFusion: the
// pair's second candidate, then its first, each through its inside filter), blended by the pair's model
// (ccpFusionBlend); the CU's record is the first candidate's, as a merge CU's.
// A merge block of fusion with CCCM by a blend model (the SPS's ccpFusionMode) keeps its candidates' template samples
// alike, and after the sort fits the blend model of the chosen candidate's and the CCCM prediction's (cccmPrediction
// keeps those: ccpmComputeTemplateFusionModel, ccpmComputeBlendFusionModelJoint) that storeRegion blends the two by
// (applyBlendFusion)
void ccpMergePrediction( inout Tu t, int predStride, int lid, int NT )
{
  const int  kind = BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_KIND];
  const bool derived = WITH_CCP_DERIVED && kind == CCP_KIND_DERIVED;
  const bool ddFusion = WITH_CCP_DERIVED && kind == CCP_KIND_DD_FUSION;
  const bool blendFusion = WITH_CCP_DERIVED && kind == CCP_KIND_FUSION_BLEND;
  const bool twoSteps = derived || ddFusion;   // (list[1]'s prediction, then list[0]'s)
  const int  adjust = twoSteps || blendFusion ? 0 : kind;
  g_ccpAngUsed = 0;
  if( twoSteps )   // (where a prediction goes, also where no CCCM model of the block set them: list[1] a CCLM model)
  {
    g_cccmFusion = false;
    g_cccmBoth   = false;
  }
  if( derived )   // (the other component's references, for its CCLM model)
  {
    Tu o = t;
    o.comp = 3 - t.comp;
    fillReferencePar( o, 0, 1, predStride, lid, NT );
    SYNC();
  }
  if( lid == 0 )
  {
    cclmAvailability( t );
    if( derived )
    {
      ccpDerivedLinear( t, predStride );
    }
    cccmArea( t );
    ccpMergeHistory( t );   // (a derived block's entry has neither sources nor a history)
  }
  SYNC();
  if( !derived && adjust == 0 )
  {
    ccpMergeDefaults( t, lid, NT );
  }
  if( lid == 0 )
  {
    if( derived )
    {
      ccpDerivedList( t );
    }
    else
    {
      ccpMergeList( t );
    }
  }
  SYNC();
  // the models the block fits itself: decoder-derived fusion's, adjustment's on-the-fly ones (appended while the list
  // has room)
  if( derived || ( adjust != 0 && s_ccp[CCP_LIST_NUM] < CCP_MAX_LIST ) )
  {
    ccpFitModels( t, derived, lid, NT );
    if( lid == 0 && !derived )
    {
      int num = s_ccp[CCP_LIST_NUM];
      for( int k = 0; k < CCP_FLY_MAX && num < CCP_MAX_LIST; k++ )
      {
        if( s_ccp[CCP_FLY_VALID + k] != 0 )
        {
          s_ccp[CCP_LIST + num]        = ccpDefaultRecords( t ) + k;
          s_ccp[CCP_LIST_INSIDE + num] = 0;
          num++;
        }
      }
      s_ccp[CCP_LIST_NUM] = num;
    }
    SYNC();
  }
  const int  W = t.w, H = t.h, len = W + H, numCost = s_ccp[CCP_LIST_NUM], maxPel = ( 1 << t.bitDepth ) - 1;
  const int  steps = numCost + ( twoSteps ? 2 : 1 );
  const bool above = s_cclm[CCLM_POS_ABOVE] != 0, left = s_cclm[CCLM_POS_LEFT] != 0;
  int x = 0, inside = 0;
  for( int step = 0; step < steps; step++ )
  {
    const bool cost = step < numCost, last = step == steps - 1, fusionStep = twoSteps && !cost && !last;
    if( cost )
    {
      x      = s_ccp[CCP_LIST + step];
      inside = derived ? s_ccp[CCP_LIST_INSIDE + step] : 0;   // (a derived mode's inside filter filters its probes)
    }
    else
    {
      if( step == numCost )   // the choice
      {
        if( derived )
        {
          ccpDerivedChoice( t, lid, NT );
        }
        else
        {
          if( lid == 0 )
          {
            ccpMergeSort();
          }
          SYNC();
          if( ddFusion )
          {
            ccpDdFusionChoice( t, lid, NT );
          }
          if( blendFusion )   // (the blend model of the chosen candidate's and the CCCM prediction's template samples)
          {
            cccmSync();
            if( lid == 0 )
            {
              const int idx = BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_INDEX];
              ccpBlendModel( t, s_ccp[CCP_LIST_ORIG + idx], CCP_MAX_CANDS, ccpPairBase( t ), false );
              for( int w = 0; w < CCP_PAIR_WORDS; w++ )
              {
                s_ccp[CCP_BLEND + w] = CBufI32( P.scratch ).v[ccpPairBase( t ) + w];
              }
            }
            SYNC();
          }
        }
      }
      const int idx = !twoSteps ? BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_INDEX]
                    : s_ccp[last ? CCP_DERIVED_FIRST : CCP_DERIVED_SECOND];
      if( derived && idx < 0 )   // (no list[1]: one mode alone)
      {
        continue;
      }
      x      = s_ccp[CCP_LIST + idx];
      inside = s_ccp[CCP_LIST_INSIDE + idx];
    }
    const int  type = ccpWord( x, CCP_OUT_TYPE );
    const bool linear = ccpLinear( type ), two = ( type & CCP_TYPE_MMLM ) != 0;
    const bool mdf = ( type & CCP_TYPE_MDFCCCM ) != 0;
    if( !linear )
    {
      // (the prediction places several filters' position features by the block's own area, the costs and the
      // adjustment by the candidate's)
      ccpCccmModel( t, x, !cost && mdf && adjust == 0, lid, NT );
    }
    if( last && adjust != 0 )   // (with adjustment the list holds CCCM models alone)
    {
      ccpMergeAdjust( t, x, adjust, lid, NT );
      g_cccmLocDX = mdf ? 0 : g_cccmLocDX;
      g_cccmLocDY = mdf ? 0 : g_cccmLocDY;
    }
    if( last && !g_ccpOther )
    {
      // the CU's record: the candidate's (with its inside filter flag both as the CU's and as the history's; a derived
      // block's 0), written by the Cb block; an adjusted model's coefficients of each component by its own block
      const int own = t.comp - 1, n = g_cccmParams;
      for( int w = lid; w < CCP_OUT; w += NT )
      {
        const int  pw = w - CCP_OUT_PARAMS, m = pw / ( 4 * CCP_MAX_PARAMS ), c = ( pw / ( 2 * CCP_MAX_PARAMS ) ) & 1;
        const int  p = ( pw % ( 2 * CCP_MAX_PARAMS ) ) >> 1;
        const bool coef = adjust != 0 && pw >= 0 && pw < 8 * CCP_MAX_PARAMS && p < n && ( m == 0 || two );
        if( coef ? c == own : own == 0 )
        {
          const ivec2 v     = coef ? unpackInt2x32( cccmLoad( cccmParamWord( m, c, p ) ) ) : ivec2( 0 );
          const bool  flags = w == CCP_OUT_INSIDE || w == CCP_OUT_INSIDE_HIST;
          CBufI32( P.ccpOut ).v[t.ccp * CCP_OUT + w] = coef ? ( ( pw & 1 ) == 0 ? v.x : v.y )
                                                     : flags ? ( derived ? 0 : inside ) : ccpWord( x, w );
        }
      }
    }
    if( lid < 2 )
    {
      s_red[2 + lid] = 0;
    }
    if( lid == 0 && fusionStep && linear )   // (list[1]'s CCLM model; list[0]'s follows after the steps)
    {
      ccpLoadLinear( t, x );
    }
    SYNC();
    // the template's samples of both components (costs) or the block's of its own (a CCCM prediction); decoder-derived
    // fusion's probe samples (those of its modes with the inside filter filtered: ccpProbeTap's taps, a CCLM model's
    // clipped) and its list[1]'s prediction (into the fusion slot, through the inside filter below where it has it)
    // apart, in the code of its pipelines alone
    const bool inTmp = cccmPredInTmp( t );
    int sad = 0;
    for( int s = lid; s < ( cost ? 2 * len : linear && last ? 0 : W * H ); s += NT )
    {
      const int  c = cost ? s / len : t.comp - 1, pos = s % len, q = pos < W ? pos : pos - W;
      const bool top = pos < W;
      if( !cost || ( top ? above : left ) )
      {
        int v = 0;
        if( derived && cost )   // (a probe sample)
        {
          const int taps = inside != 0 ? 9 : 1;
          int sum = 0;
          for( int k = 4 - ( taps >> 1 ); k <= 4 + ( taps >> 1 ); k++ )   // (tap 4 the sample itself)
          {
            bool pad = false;
            const ivec2 p = ccpProbeTap( t, top, q, k, pad );
            v = pad ? recoAt( c + 1, p.x, p.y ) : linear ? clamp( ccpLinearTemplate( t, x, c, top, q ), 0, maxPel )
              : cccmPredict( 0, c, p.x, p.y, two, t.bitDepth );
            sum += ( k != 4 ? 1 : taps == 1 ? 16 : 8 ) * v;
          }
          v = ( sum + 8 ) >> 4;
        }
        else if( fusionStep && linear )   // (list[1]'s CCLM / GLM prediction)
        {
          Tu tl = t;   // (a CCLM block of slot 0's model: its pattern, two models as MMLM's)
          tl.lm      = LM_ACTIVE | ( ccpWord( x, CCP_OUT_GLM_IDC ) << LM_GLM_SHIFT );
          tl.dirMode = two ? MMLM_CHROMA_IDX : LM_CHROMA_IDX;
          v = cclmPredSample( tl, t.x << 1, t.y << 1, s % W, s / W, maxPel, 0 );
        }
        else
        {
          const int px = !cost ? g_cccmSizeX + s % W : top ? g_cccmSizeX + q : g_cccmSizeX - 1;
          const int py = !cost ? g_cccmSizeY + s / W : top ? g_cccmSizeY - 1 : g_cccmSizeY + q;
          v = linear ? int( int16_t( ccpLinearTemplate( t, x, c, top, q ) ) )
                     : cccmPredict( 0, c, px, py, two, t.bitDepth );
        }
        if( cost )
        {
          sad += abs( ( top ? recoAt( c + 1, t.x + q, t.y - 1 ) : recoAt( c + 1, t.x - 1, t.y + q ) ) - v );
          if( twoSteps || blendFusion )
          {
            CBufI32( P.scratch ).v[ccpProbeBase( t ) + step * 2 * len + s] = v;
          }
        }
        else if( fusionStep && inside == 0 )
        {
          CBufI32( P.scratch ).v[cccmFusionPredBase( t ) + s] = v;
        }
        else if( inTmp )
        {
          s_tmp[s] = int16_t( v );
        }
        else
        {
          CBufI32( P.scratch ).v[cccmScratchPred() + s] = v;
        }
      }
    }
    if( fusionStep && inside != 0 )   // (list[1]'s prediction filtered into the fusion slot)
    {
      cccmSync();
      Tu tf = t;   // (a CCCM block: cclmPredSample reads the prediction it stored)
      tf.lm = 1 << LM_CCCM_SHIFT;
      for( int s = lid; s < W * H; s += NT )
      {
        CBufI32( P.scratch ).v[cccmFusionPredBase( t ) + s] = cclmInsideFiltered( tf, t.x << 1, t.y << 1, s % W, s / W,
                                                                                   maxPel, 0 );
      }
    }
    const int total = wgSum2( sad, 0, 2 ).x;
    if( cost && lid == 0 )
    {
      s_ccp[CCP_LIST_COST + step] = total;
    }
    SYNC();
  }
  // the chosen model as its LM block's; a merge fusion block's fields kept: storeRegion blends its CCCM prediction in,
  // decoder-derived fusion's list[1] with the weight 3 for list[0], decoder-derived merge fusion's by the pair's model
  // (the weight 0 that stands for it)
  const int type = ccpWord( x, CCP_OUT_TYPE );
  t.lm = ( t.lm & LM_FUSION_FIELDS ) | ( inside != 0 ? LM_INSIDE : 0 );
  if( ( derived && s_ccp[CCP_DERIVED_SECOND] >= 0 ) || ddFusion )
  {
    t.lm |= LM_FUSION | ( 1 << LM_FUSION_TYPE_SHIFT ) | ( ( derived ? 3 : 0 ) << LM_FUSION_W_SHIFT );
  }
  if( ccpLinear( type ) )
  {
    t.dirMode = ( type & CCP_TYPE_MMLM ) != 0 ? MMLM_CHROMA_IDX : LM_CHROMA_IDX;
    t.lm     |= LM_ACTIVE | ( ccpWord( x, CCP_OUT_GLM_IDC ) << LM_GLM_SHIFT );
    if( lid == 0 )
    {
      ccpLoadLinear( t, x );
    }
  }
  else
  {
    t.lm |= 1 << LM_CCCM_SHIFT;   // (cclmPredSample reads cccmBlockPrediction's)
  }
  cccmSync();
}

// ----- LFNST / NSPT of a CCP merge block or a decoder-derived CCP fusion block (JVET-AK0064; intra_rec's second
// module)

#ifdef CCP_LFNST_MODULE

// the CU's final prediction of component tc.comp at (x, y), after ccpMergePrediction( tc ): the chosen model's through
// its inside filter (cclmPrediction), blended with the fusion slot's prediction as storeRegion blends it
// (cccmFusionBlend: merge fusion with CCCM, decoder-derived fusion and its merge fusion), or as intra_rec's blending
// averages it with the prediction of the DIMD chroma mode there (merge fusion with DIMD chroma: two predictions of
// weight 1, the sum rounded)
int ccpMergeFinalSample( Tu tc, int x, int y, int maxPel )
{
  const int pr = cclmPrediction( tc, tc.x << 1, tc.y << 1, x, y, maxPel, 0 );
  if( ( tc.derive & DERIVE_CCP_DIMD ) != 0 )
  {
    return ( pr + CBufI32( P.scratch ).v[cccmFusionPredBase( tc ) + y * tc.w + x] + 1 ) >> 1;
  }
  return lmFusionType( tc ) == 1 ? cccmFusionBlend( tc, x, y, pr ) : pr;
}

// merge fusion with DIMD chroma's second prediction of component tc.comp (the CU's DIMD chroma mode, its node's) into
// the fusion slot, from the component's reference line in s_ref[0], as intra_rec's prediction section predicts a
// chroma block: planar / DC with their PDPC, angular from xPredIntraAng's arrays by two taps with PDPC, the angular
// arrays cleared again (all invocations, ending with a barrier)
void ccpDimdChromaPrediction( Tu tc, int predStride, int maxPel, int lid, int NT )
{
  Tu td = tc;
  applyModeParams( td, cuModeOf( tc.cu ), 0 );
  const int W = td.w, H = td.h, N = W * H, slot = cccmFusionPredBase( tc );
  const int topLen = W << 3, leftLen = H << 3;
  if( td.dirMode == PLANAR_IDX || td.dirMode == DC_IDX )
  {
    const int log2W = findMSB( W ), log2H = findMSB( H );
    if( td.dirMode == DC_IDX && lid == 0 )   // xGetPredValDc
    {
      const int denom = ( W == H ) ? ( W << 1 ) : max( W, H );
      int sum = 0;
      if( W >= H ) for( int k = 0; k < W; k++ ) sum += int( s_ref[0][1 + k] );
      if( W <= H ) for( int k = 0; k < H; k++ ) sum += int( s_ref[0][predStride + 1 + k] );
      s_dc = ( sum + ( denom >> 1 ) ) >> findMSB( denom );
    }
    SYNC();
    const int topRight = int( s_ref[0][1 + W] ), bottomLeft = int( s_ref[0][predStride + 1 + H] );
    const int offset = 1 << ( log2W + log2H ), finalShift = 1 + log2W + log2H;
    const int pdpcScale = ( log2W - 2 + log2H - 2 + 2 ) >> 2;
    for( int i = lid; i < N; i += NT )
    {
      const int x = i % W, y = i / W, T = int( s_ref[0][1 + x] ), L = int( s_ref[0][predStride + 1 + y] );
      int v;
      if( td.dirMode == DC_IDX )
      {
        v = s_dc;
      }
      else
      {
        const int horPred  = ( L << log2W ) + ( x + 1 ) * ( topRight - L );
        const int vertPred = ( T << log2H ) + ( y + 1 ) * ( bottomLeft - T );
        if( td.plIdx == 1 )      v = ( horPred + ( 1 << ( log2W - 1 ) ) ) >> log2W;
        else if( td.plIdx == 2 ) v = ( vertPred + ( 1 << ( log2H - 1 ) ) ) >> log2H;
        else                     v = ( ( horPred << log2H ) + ( vertPred << log2W ) + offset ) >> finalShift;
      }
      v = asPel( v );
      if( td.applyPDPC != 0 )   // xIntraPredPlanarDcPdpc
      {
        const int wT = 32 >> min( 31, ( y << 1 ) >> pdpcScale );
        const int wL = 32 >> min( 31, ( x << 1 ) >> pdpcScale );
        v = asPel( v + ( ( wL * ( L - v ) + wT * ( T - v ) + 32 ) >> 6 ) );
      }
      CBufI32( P.scratch ).v[slot + i] = v;
    }
  }
  else
  {
    const bool  isVer = td.isModeVer != 0;
    const int   angle = td.intraPredAngle, absInvAngle = td.absInvAngle;
    const ivec2 bases = angularArrays( isVer, W, H, angle, absInvAngle, topLen, leftLen, 0, predStride,
                                       td.disableRefFilter != 0, maxPel, lid, NT );
    SYNC();
    const int  mainBase = bases.x, sideBase = bases.y;
    const int  w = isVer ? W : H, h = isVer ? H : W;   // (rows along the main direction)
    const bool intSlope = ( abs( angle ) & 31 ) == 0;
    const int  pdpcScale = angle == 0 ? ( ( findMSB( w ) + findMSB( h ) - 2 ) >> 2 ) : td.angularScale;
    const int  pdpcLimit = td.applyPDPC != 0 ? min( 3 << pdpcScale, w ) : 0;
    for( int i = lid; i < N; i += NT )
    {
      const int x = i % w, y = i / w;   // position in the (possibly transposed) block
      int v;
      if( angle == 0 )
      {
        v = mainR( isVer, 0, mainBase + 1 + x );
        if( x < pdpcLimit )
        {
          const int scale = ( findMSB( w ) + findMSB( h ) - 2 ) >> 2;
          const int topLeft = mainR( isVer, 0, mainBase ), left = sideR( isVer, 0, sideBase + 1 + y );
          const int wL = 32 >> ( 2 * x >> scale );
          v = clip3i( 0, maxPel, v + ( ( wL * ( left - topLeft ) + 32 ) >> 6 ) );
        }
      }
      else
      {
        const int deltaPos = angle * ( y + 1 );
        const int deltaInt = deltaPos >> 5, deltaFract = deltaPos & 31;
        if( intSlope )
        {
          v = mainR( isVer, 0, mainBase + deltaInt + 1 + x );
        }
        else
        {
          const int p0 = mainR( isVer, 0, mainBase + deltaInt + x + 1 );
          const int p1 = mainR( isVer, 0, mainBase + deltaInt + x + 2 );
          v = asPel( p0 + ( ( deltaFract * ( p1 - p0 ) + 16 ) >> 5 ) );
        }
        v = asPel( v );
        if( x < pdpcLimit )
        {
          const int wL = 32 >> ( 2 * x >> td.angularScale );
          if( td.gradPDPC != 0 )
          {
            const int left = sideR( isVer, 0, sideBase + 1 + y );
            const int m0 = mainR( isVer, 0, mainBase + deltaInt ), m1 = mainR( isVer, 0, mainBase + deltaInt + 1 );
            const int topLeft = asPel( m0 + ( ( deltaFract * ( m1 - m0 ) + 16 ) >> 5 ) );
            v = clip3i( 0, maxPel, v + ( ( wL * ( left - topLeft ) + 32 ) >> 6 ) );
          }
          else
          {
            const int invAngleSum = 256 + ( x + 1 ) * absInvAngle;
            const int left = sideR( isVer, 0, sideBase + y + ( invAngleSum >> 9 ) + 1 );
            v = asPel( v + ( ( wL * ( left - v ) + 32 ) >> 6 ) );
          }
        }
      }
      CBufI32( P.scratch ).v[slot + ( isVer ? y * W + x : x * W + y )] = asPel( v );
    }
    // (zero again for the next prediction: IntraPrediction's angRefClearEnd)
    const int log2Ratio = findMSB( W ) - findMSB( H );
    const int angClear  = min( ANG, 2 * ( W + H ) + max( topLen, leftLen ) + 2 * PAD + 2 + ( 1 << abs( log2Ratio ) )
                                    + 16 );
    SYNC();
    for( int i = lid; i < angClear; i += NT )
    {
      s_angA[0][i] = int16_t( 0 );
      s_angL[0][i] = int16_t( 0 );
    }
  }
  SYNC();
}

// the LFNST / NSPT mode of CCP merge block t, also of a decoder-derived CCP fusion block (DERIVE_CCLM_DIMD): DecCu's
// deriveChromaIpmForTransform of the CU's final chroma predictions (Tu::lm's LM_HIST_*: Cb's, Cr's or both into one
// histogram) into s_predDimdMode. The block predicts each component the histogram reads as that component's block does,
// the other one first (g_ccpOther), then its own, which also leaves the references and merge fusion's CCCM prediction
// the block's again; the prediction section predicts the block again. A gradient's samples are computed again each
// time. All invocations, ending with a barrier
void ccpMergeTransformMode( Tu t, int predStride, int maxPel, int lid, int NT )
{
  const bool derived    = BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_KIND] == CCP_KIND_DERIVED;
  const bool fusionCccm = lmFusionType( t ) == 1;   // (merge fusion with CCCM: its CCCM prediction comes first)
  const bool fusionDimd = ( t.derive & DERIVE_CCP_DIMD ) != 0;   // (merge fusion with DIMD chroma: its prediction too)
  for( int i = lid; i < NUM_LUMA_MODE; i += NT )
  {
    s_dimdHist[CCP_DIMD_HIST + i] = 0;
  }
  const int iw = t.w - 2, inner = iw * ( t.h - 2 );
  for( int pass = 0; pass < 2; pass++ )
  {
    Tu tc = t;
    tc.comp = pass == 0 ? 3 - t.comp : t.comp;
    const bool hist = ( t.lm & ( tc.comp == 1 ? LM_HIST_CB : LM_HIST_CR ) ) != 0;
    if( pass == 0 && !hist )
    {
      continue;
    }
    g_ccpOther = pass == 0;
    // (the references: a derived block's CCLM model's, the merge code takes the other component's into s_ref[1]; those
    // of the DIMD chroma mode's prediction)
    if( derived || fusionDimd )
    {
      SYNC();
      fillReferencePar( tc, 0, 0, predStride, lid, NT );
      SYNC();
    }
    if( fusionCccm )
    {
      cccmPrediction( tc, false, lid, NT );
    }
    if( hist )
    {
      if( fusionDimd )
      {
        ccpDimdChromaPrediction( tc, predStride, maxPel, lid, NT );
      }
      ccpMergePrediction( tc, predStride, lid, NT );
      for( int k = lid; k < inner; k += NT )
      {
        const int   x = 1 + k % iw, y = 1 + k / iw;
        const ivec2 g = sobel3x3( ccpMergeFinalSample( tc, x - 1, y - 1, maxPel ),
                                  ccpMergeFinalSample( tc, x, y - 1, maxPel ),
                                  ccpMergeFinalSample( tc, x + 1, y - 1, maxPel ),
                                  ccpMergeFinalSample( tc, x - 1, y, maxPel ),
                                  ccpMergeFinalSample( tc, x + 1, y, maxPel ),
                                  ccpMergeFinalSample( tc, x - 1, y + 1, maxPel ),
                                  ccpMergeFinalSample( tc, x, y + 1, maxPel ),
                                  ccpMergeFinalSample( tc, x + 1, y + 1, maxPel ) );
        if( g.x != 0 || g.y != 0 )
        {
          atomicAdd( s_dimdHist[CCP_DIMD_HIST + dimdAngle( g )], abs( g.x ) + abs( g.y ) );
        }
      }
      SYNC();   // (the next pass's predictions overwrite these)
    }
  }
  g_ccpOther = false;
  if( lid == 0 )   // the first of equal amplitudes
  {
    int firstAmp = 0, first = 0;
    for( int i = 0; i < NUM_LUMA_MODE; i++ )
    {
      if( s_dimdHist[CCP_DIMD_HIST + i] > firstAmp )
      {
        firstAmp = s_dimdHist[CCP_DIMD_HIST + i];
        first    = i;
      }
    }
    s_predDimdMode = first;
  }
  SYNC();
}

#endif

#endif
