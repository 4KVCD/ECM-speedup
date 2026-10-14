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
// block's defaults; its scheduling waits for the blocks writing them) and rejects merge fusion with CCCM by a blend
// model (the SPS's ccpFusionMode 1: not all-intra) and decoder-derived CCP fusion. Merge adjustment (JVET-AL0126)
// lists the CCCM models alone, adds models fitted on the fly (ccpFlyModels) and moves the chosen model's coefficients
// by template costs (ccpMergeAdjust). Merge fusion with CCCM (predCCPCandidate's ccpMergeFusionType 0) is LM chroma
// fusion's first type with equal weights: the block computes the CCCM prediction before its own and storeRegion
// averages them (intra_cccm.glsl's cccmFusionBlend). Merge fusion with DIMD chroma (ccpMergeFusionType 1): the CU's
// node derives the DIMD chroma mode as a DIMD_CHROMA_IDX CU's, and intra_rec's blending averages the block's merge
// prediction and the mode's (two blended predictions of weight 1, the average rounded). A candidate predicts like the
// LM blocks of its type: a CCLM / GLM model as intra_cclm.glsl's (the slots in s_cclm), the others with
// intra_cccm.glsl's luma buffers and cccmPredict (the CPU's model applied to the CU's own luma, the candidate's luma
// offset taken in the buffer: xCccmGetLumaVal's luma minus m_cccmLumaOffset plus m_cccmLumaOffset minus the
// candidate's offset). glslang's inlining multiplies intra_rec.comp's SPIR-V (its ID bound limits the module): the
// large functions have few call sites here (the costs and the prediction one loop; the adjustment's costs and the
// on-the-fly fits one more each).
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

// the candidate list in shared memory (s_dimdHist: a merge block derives no DIMD), which invocation 0 builds: per
// candidate its record, its inside filter flag (the PU's curCand's for a spatial one, the history's copy's for a
// history one, 0 for a default or an on-the-fly model) and its template cost; their number; the history's models
// (most recent first) and their number; the first CCLM candidate's slopes (Cb's, Cr's, 64-bit as two words) and
// shifts, which the defaults move, and whether there is one; which on-the-fly models are valid
#define s_ccp s_dimdHist
const int CCP_LIST = 0, CCP_LIST_INSIDE = 15, CCP_LIST_COST = 30, CCP_LIST_NUM = 45, CCP_LUT = 46, CCP_LUT_NUM = 52;
const int CCP_SCALE = 53, CCP_SHIFT = 57, CCP_FOUND = 59, CCP_FLY_VALID = 60;

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

// the first default record of merge block t: the Cb block's 12, then the Cr block's (each block ranks the list)
int ccpDefaultRecords( Tu t )
{
  return BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_DEFAULTS] + ( t.comp - 1 ) * CCP_MAX_CANDS;
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
// CCP_TYPE_GLM4567's), at most CCP_MAX_LIST, no defaults (the on-the-fly models follow: ccpFlyModels)
void ccpMergeList( Tu t )
{
  BufI32     d = BufI32( P.ccpData );
  const int  e = t.ccpMerge, numSpatial = d.v[e + CCP_MERGE_NUM_SPATIAL], numLut = s_ccp[CCP_LUT_NUM];
  const bool adjust = d.v[e + CCP_MERGE_ADJUST] != 0;
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

// reorderCCPCandidates' insertion sort by the costs (invocation 0): each candidate before the first one of a higher
// cost
void ccpMergeSort()
{
  for( int i = 1; i < s_ccp[CCP_LIST_NUM]; i++ )
  {
    for( int j = 0; j < i; j++ )
    {
      if( s_ccp[CCP_LIST_COST + i] < s_ccp[CCP_LIST_COST + j] )
      {
        const int x = s_ccp[CCP_LIST + i], inside = s_ccp[CCP_LIST_INSIDE + i], cost = s_ccp[CCP_LIST_COST + i];
        for( int k = i; k > j; k-- )
        {
          s_ccp[CCP_LIST + k]        = s_ccp[CCP_LIST + k - 1];
          s_ccp[CCP_LIST_INSIDE + k] = s_ccp[CCP_LIST_INSIDE + k - 1];
          s_ccp[CCP_LIST_COST + k]   = s_ccp[CCP_LIST_COST + k - 1];
        }
        s_ccp[CCP_LIST + j]        = x;
        s_ccp[CCP_LIST_INSIDE + j] = inside;
        s_ccp[CCP_LIST_COST + j]   = cost;
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
// unclipped, as a Pel
int ccpLinearTemplate( Tu t, int x, int c, bool top, int q )
{
  const int lx = t.x << 1, ly = t.y << 1, glm = ccpWord( x, CCP_OUT_GLM_IDC );
  const int luma = top ? cclmLumaAbove( lx, ly, q, s_cclm[CCLM_LEFT] != 0, glm )
                       : cclmLumaLeft( lx, ly, q, s_cclm[CCLM_ABOVE] != 0, glm );
  const int a = int( ccpParam( x, false, c, 0 ) ), b = int( ccpParam( x, false, c, 1 ) );
  return int( int16_t( ( ( a * luma ) >> ccpWord( x, CCP_OUT_SHIFT + c ) ) + b ) );
}

// ----- adjustment (JVET-AL0126): on-the-fly models, then the chosen model's coefficients moved

// record words w of an on-the-fly model just fitted (two: MMLM; its parameters in the parameter words, its ranges in
// CCCM_MODEL_RANGE): cccmModelToCcpParams' fields - plain CCCM, the threshold of two models, the block's own luma
// offset and corner offsets, 7 parameters, the template chroma's ranges - every other word 0
int ccpFlyWord( int w, bool two )
{
  if( w == CCP_OUT_TYPE ) return CCP_TYPE_CCCM | ( two ? CCP_TYPE_MMLM : 0 );
  if( w == CCP_OUT_Y_THRES ) return two ? s_cclm[CCCM_THRESHOLD] : 0;
  if( w == CCP_OUT_LUMA_OFFSET ) return g_cccmLumaOffset;
  if( w == CCP_OUT_COR_OFF ) return g_cccmSizeX;
  if( w == CCP_OUT_COR_OFF + 1 ) return g_cccmSizeY;
  if( w == CCP_OUT_NUM_PARAMS ) return 7;
  if( w >= CCP_OUT_RANGE && w < CCP_OUT_RANGE + 12 )   // (per model and component: min, max, change)
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

// the on-the-fly models of merge block t with adjustment (deriveCcpMergeAdjustCands; all invocations, ending with a
// barrier), each a record among the block's spare ones (ccpDefaultRecords' first CCP_FLY_MAX), valid where one of its
// coefficients but the bias is not 0 (CCP_FLY_VALID): plain CCCM of the block's own luma buffer and luma offset over
// the whole template - two models split at xCccmCalcRefAver's threshold (the template's mean luma), one model, one of
// the template's two lines next to the block (trainingRange 2), one of the reference samples alone (useRefSampOnly),
// two of them, two of the two lines. (The CPU selects the template by the block's position - the left part in the
// picture's top row, the part above in its left column - which is the whole template there: a merge block's own
// template selection, 0, gives the same.)
void ccpFlyModels( Tu t, int lid, int NT )
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
  const ivec4 r      = cccmTemplate( t, false );
  const ivec2 counts = cccmTemplateCounts( r );
  int sum = 0;
  for( int i = lid; i < counts.x + counts.y; i += NT )
  {
    const ivec2 p = cccmTemplateSample( r, counts, i );
    sum += cccmLuma( p.x, p.y );
  }
  const int num = counts.x + counts.y, total = wgSum2( sum, 0, 2 ).x;
  if( lid == 0 )
  {
    s_cclm[CCCM_THRESHOLD] = num == 0 ? 1 << ( t.bitDepth - 1 ) : cccmDivideLowPrec( total, num );
  }
  for( int k = 0; k < CCP_FLY_MAX; k++ )
  {
    const bool two   = k == 0 || k >= 4;
    g_ccpFlyTwoLines = k == 2 || k == 5;
    g_ccpFlyRefOnly  = k == 3 || k == 4;
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
    if( lid == 0 )
    {
      s_ccp[CCP_FLY_VALID + k] = valid ? 1 : 0;
    }
    const int base = ( ccpDefaultRecords( t ) + k ) * CCP_OUT;
    for( int w = lid; w < CCP_OUT; w += NT )
    {
      CBufI32( P.ccpOut ).v[base + w] = ccpFlyWord( w, two );
    }
    SYNC();   // (the next fit overwrites the parameter words)
  }
  g_ccpFlyTwoLines = false;
  g_ccpFlyRefOnly  = false;
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
// inside filter flag
void ccpMergePrediction( inout Tu t, int lid, int NT )
{
  const int adjust = BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_ADJUST];
  g_ccpAngUsed = 0;
  if( lid == 0 )
  {
    cclmAvailability( t );
    cccmArea( t );
    ccpMergeHistory( t );
  }
  SYNC();
  if( adjust == 0 )
  {
    ccpMergeDefaults( t, lid, NT );
  }
  if( lid == 0 )
  {
    ccpMergeList( t );
  }
  SYNC();
  if( adjust != 0 && s_ccp[CCP_LIST_NUM] < CCP_MAX_LIST )   // (the on-the-fly models, appended while there is room)
  {
    ccpFlyModels( t, lid, NT );
    if( lid == 0 )
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
  const int  W = t.w, H = t.h, len = W + H, steps = s_ccp[CCP_LIST_NUM] + 1;
  const bool above = s_cclm[CCLM_POS_ABOVE] != 0, left = s_cclm[CCLM_POS_LEFT] != 0;
  int x = 0, inside = 0;
  for( int step = 0; step < steps; step++ )
  {
    const bool cost = step < steps - 1;
    if( cost )
    {
      x = s_ccp[CCP_LIST + step];
    }
    else
    {
      if( lid == 0 )
      {
        ccpMergeSort();
      }
      SYNC();
      const int idx = BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_INDEX];
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
    if( !cost && adjust != 0 )   // (with adjustment the list holds CCCM models alone)
    {
      ccpMergeAdjust( t, x, adjust, lid, NT );
      g_cccmLocDX = mdf ? 0 : g_cccmLocDX;
      g_cccmLocDY = mdf ? 0 : g_cccmLocDY;
    }
    if( !cost )
    {
      // the CU's record: the candidate's (with its inside filter flag both as the CU's and as the history's), written
      // by the Cb block; an adjusted model's coefficients of each component by its own block
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
          CBufI32( P.ccpOut ).v[t.ccp * CCP_OUT + w] = coef ? ( ( pw & 1 ) == 0 ? v.x : v.y ) : flags ? inside
                                                                                                    : ccpWord( x, w );
        }
      }
    }
    if( lid < 2 )
    {
      s_red[2 + lid] = 0;
    }
    SYNC();
    // the template's samples of both components (costs) or the block's of its own (a CCCM prediction)
    const bool inTmp = cccmPredInTmp( t );
    int sad = 0;
    for( int s = lid; s < ( cost ? 2 * len : linear ? 0 : W * H ); s += NT )
    {
      const int  c = cost ? s / len : t.comp - 1, pos = s % len, q = pos < W ? pos : pos - W;
      const bool top = pos < W;
      if( !cost || ( top ? above : left ) )
      {
        const int px = !cost ? g_cccmSizeX + s % W : top ? g_cccmSizeX + q : g_cccmSizeX - 1;
        const int py = !cost ? g_cccmSizeY + s / W : top ? g_cccmSizeY - 1 : g_cccmSizeY + q;
        const int v = linear ? ccpLinearTemplate( t, x, c, top, q ) : cccmPredict( 0, c, px, py, two, t.bitDepth );
        if( cost )
        {
          sad += abs( ( top ? recoAt( c + 1, t.x + q, t.y - 1 ) : recoAt( c + 1, t.x - 1, t.y + q ) ) - v );
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
    const int total = wgSum2( sad, 0, 2 ).x;
    if( cost && lid == 0 )
    {
      s_ccp[CCP_LIST_COST + step] = total;
    }
    SYNC();
  }
  // the chosen model as its LM block's (a merge fusion block's fields kept: storeRegion blends its CCCM prediction)
  const int type = ccpWord( x, CCP_OUT_TYPE );
  t.lm = ( t.lm & LM_FUSION_FIELDS ) | ( inside != 0 ? LM_INSIDE : 0 );
  if( ccpLinear( type ) )
  {
    t.dirMode = ( type & CCP_TYPE_MMLM ) != 0 ? MMLM_CHROMA_IDX : LM_CHROMA_IDX;
    t.lm     |= LM_ACTIVE | ( ccpWord( x, CCP_OUT_GLM_IDC ) << LM_GLM_SHIFT );
    if( lid == 0 )
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
  }
  else
  {
    t.lm |= 1 << LM_CCCM_SHIFT;   // (cclmPredSample reads cccmBlockPrediction's)
  }
  cccmSync();
}

#endif
