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
// model (the SPS's ccpFusionMode 1: not all-intra) and decoder-derived merge fusion. Merge adjustment (JVET-AL0126)
// lists the CCCM models alone, adds models fitted on the fly (ccpFitModels) and moves the chosen model's coefficients
// by template costs (ccpMergeAdjust). Merge fusion with CCCM (predCCPCandidate's ccpMergeFusionType 0) is LM chroma
// fusion's first type with equal weights: the block computes the CCCM prediction before its own and storeRegion
// averages them (intra_cccm.glsl's cccmFusionBlend). Merge fusion with DIMD chroma (ccpMergeFusionType 1): the CU's
// node derives the DIMD chroma mode as a DIMD_CHROMA_IDX CU's, and intra_rec's blending averages the block's merge
// prediction and the mode's (two blended predictions of weight 1, the average rounded). A candidate predicts like the
// LM blocks of its type: a CCLM / GLM model as intra_cclm.glsl's (the slots in s_cclm), the others with
// intra_cccm.glsl's luma buffers and cccmPredict (the CPU's model applied to the CU's own luma, the candidate's luma
// offset taken in the buffer: xCccmGetLumaVal's luma minus m_cccmLumaOffset plus m_cccmLumaOffset minus the
// candidate's offset).
// Decoder-derived CCP fusion (JVET-AG0154's decoderDerivedCcpMode CUs; DecCu at the Cb block: decoderDerivedCcp,
// predDecoderDerivedIntraCCCMFusions) shares the merge code: its entry (kind CCP_KIND_DERIVED) has no sources and no
// history, its list is its modes - a CCLM model and CCCM models the block fits itself (ccpDerivedLinear,
// ccpFitModels) - whose costs are the SADs of their predictions of probe lines (the row above and the column left of
// the block; for the modes with LBCCP's inside filter that line filtered with the lines beside it), and the mode of
// the lowest cost alone or the pair of modes whose 3 : 1 blend of the probe lines costs less (ccpDerivedChoice)
// predicts: list[0] as a merge block's chosen candidate, list[1] before it into the fusion slot that storeRegion blends
// in (cccmFusionBlend with the weight 3). Each block derives the choice itself, from both components' costs. Its
// branches are in the pipelines with WITH_CCP_DERIVED alone (the merge pictures' pipeline without them is faster).
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

// the candidate list in shared memory (s_dimdHist: a merge block derives no DIMD), which invocation 0 builds: per
// candidate its record, its inside filter flag (the PU's curCand's for a spatial one, the history's copy's for a
// history one, 0 for a default or an on-the-fly model; a decoder-derived fusion mode's own) and its template cost;
// their number; the history's models (most recent first) and their number; the first CCLM candidate's slopes (Cb's,
// Cr's, 64-bit as two words) and shifts, which the defaults move, and whether there is one; which on-the-fly models are
// valid; decoder-derived fusion's costs of the pairs of modes and its choice (the list entries of list[0] and list[1],
// -1 without fusion)
#define s_ccp s_dimdHist
const int CCP_LIST = 0, CCP_LIST_INSIDE = 15, CCP_LIST_COST = 30, CCP_LIST_NUM = 45, CCP_LUT = 46, CCP_LUT_NUM = 52;
const int CCP_SCALE = 53, CCP_SHIFT = 57, CCP_FOUND = 59, CCP_FLY_VALID = 60, CCP_PAIR_COST = 66;
const int CCP_DERIVED_FIRST = 81, CCP_DERIVED_SECOND = 82;

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
// CCP_TYPE_GLM4567's), at most CCP_MAX_LIST, no defaults (the on-the-fly models follow: ccpFitModels)
void ccpMergeList( Tu t )
{
  BufI32     d = BufI32( P.ccpData );
  const int  e = t.ccpMerge, numSpatial = d.v[e + CCP_MERGE_NUM_SPATIAL], numLut = s_ccp[CCP_LUT_NUM];
  const bool adjust = d.v[e + CCP_MERGE_KIND] != 0;
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

// where block t's probe samples are (the costs' step by step, Cb's then Cr's): the workgroup's scratch after the luma
// buffer (where it is there), where the predictions go later (the host allocates room for both)
int ccpProbeBase() { return cccmScratchPred(); }

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
      const int   pa = CBufI32( P.scratch ).v[ccpProbeBase() + ab.x * 2 * len + s];
      const int   pb = CBufI32( P.scratch ).v[ccpProbeBase() + ab.y * 2 * len + s];
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
// before CU::saveCcInsideFilterFlagInCCP would copy it)
void ccpMergePrediction( inout Tu t, int predStride, int lid, int NT )
{
  const int  kind = BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_KIND];
  const bool derived = WITH_CCP_DERIVED && kind == CCP_KIND_DERIVED;
  const int  adjust = derived ? 0 : kind;
  g_ccpAngUsed = 0;
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
  const int  steps = numCost + ( derived ? 2 : 1 );   // (decoder-derived fusion: list[1]'s prediction, then list[0]'s)
  const bool above = s_cclm[CCLM_POS_ABOVE] != 0, left = s_cclm[CCLM_POS_LEFT] != 0;
  int x = 0, inside = 0;
  for( int step = 0; step < steps; step++ )
  {
    const bool cost = step < numCost, last = step == steps - 1, fusionStep = derived && !cost && !last;
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
        }
      }
      const int idx = !derived ? BufI32( P.ccpData ).v[t.ccpMerge + CCP_MERGE_INDEX]
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
    if( last )
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
        else if( fusionStep && linear )   // (list[1]'s CCLM prediction)
        {
          Tu tl = t;   // (a CCLM block of slot 0's model)
          tl.lm      = LM_ACTIVE;
          tl.dirMode = LM_CHROMA_IDX;
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
          if( derived )
          {
            CBufI32( P.scratch ).v[ccpProbeBase() + step * 2 * len + s] = v;
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
  // decoder-derived fusion's list[1] with the weight 3 for list[0]
  const int type = ccpWord( x, CCP_OUT_TYPE );
  t.lm = ( t.lm & LM_FUSION_FIELDS ) | ( inside != 0 ? LM_INSIDE : 0 );
  if( derived && s_ccp[CCP_DERIVED_SECOND] >= 0 )
  {
    t.lm |= LM_FUSION | ( 1 << LM_FUSION_TYPE_SHIFT ) | ( 3 << LM_FUSION_W_SHIFT );
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

#endif
