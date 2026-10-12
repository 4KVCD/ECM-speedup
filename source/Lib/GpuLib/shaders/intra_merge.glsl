// Intra merge (JVET-AM0074) on the GPU, in an intra merge CU's node: deriveIntraMergeCand's candidates from the CU's
// neighbours (the host's list in the CPU's order, MergeHead: an intra merge or OBIC CU is always valid, a DIMD or MPM CU
// where its DIMD blends), each at a position not taken yet and of new parameters, at most five; reorderIntraMergeCand's
// ranking of two or more by the template costs of their modes (intra_timd.glsl); the choice into the CU's DimdOut slot
// (dimdModeMerge, dimdBlendModeMerge, dimdRelWeightMerge, dimdLocDepMerge in DimdOut's layout, which the CU's blocks
// blend like DIMD's), the CU's mode its first. Included by intra_rec.comp after intra_timd.glsl.
#ifndef INTRA_MERGE_GLSL
#define INTRA_MERGE_GLSL

// in s_spCost (free in a node): the candidates (IntraMergeCandidate), MERGE_CAND int32 each, their number, the modes
// to rank by (MC_MODES, their number at MC_NUM_MODES) and the template cost of each mode (MC_COST + mode)
const int MERGE_CANDS  = 5;    // MAX_NUM_INITIAL_INTRA_MERGE_CAND
const int MERGE_CAND   = 24;
const int MC_MODE      = 0;    // dimdModeMerge
const int MC_BLEND     = 1;    // dimdBlendModeMerge, 6
const int MC_WEIGHT    = 7;    // dimdRelWeightMerge, 8
const int MC_LOC_DEP   = 15;   // dimdLocDepMerge, 7
const int MC_X         = 22;   // the neighbour's position (isDifferentIntraBlk)
const int MC_Y         = 23;
const int MC_NUM       = MERGE_CANDS * MERGE_CAND;
const int MC_NUM_MODES = MC_NUM + 1;
const int MC_MODES     = MC_NUM + 2;
const int MC_COST      = MC_MODES + NUM_LUMA_MODE;   // ends at 256, s_spCost's size

int mcGet( int cand, int field )
{
  return int( s_spCost[cand * MERGE_CAND + field] );
}

void mcSet( int cand, int field, int v )
{
  s_spCost[cand * MERGE_CAND + field] = uint( v );
}

// in s_tmp (free before the template costs), while the candidates are found: per list entry its class where it is
// valid (0: not a candidate) and its position (MT_ENTRY int16 each, at most MT_ENTRIES entries), then the indices of the
// first MT_PREFETCH valid entries and their parameters' sources (their DimdOut / ObicOut, MT_WORDS each), fetched at once
const int MT_ENTRY    = 3;
const int MT_ENTRIES  = 192;   // deriveIntraMergeCand's positions of a CU up to 128 x 128 (the host's limit): at most
                               // 2 x 33 + 5 + 61 + 28
const int MT_PREFETCH = 16;
const int MT_WORDS    = DIMD_OUT_PARAMS;
const int MT_LIST     = MT_ENTRY * MT_ENTRIES;
const int MT_DATA     = MT_LIST + MT_PREFETCH + 1;

// word w of the parameters' source of neighbour n of class cls: its prefetched copy (slot pf), else its OBIC / DimdOut
int mergeSource( int n, int cls, int pf, int w )
{
  if( pf >= 0 )
  {
    return int( s_tmp[MT_DATA + pf * MT_WORDS + w] );
  }
  return cls == MERGE_NBR_OBIC ? CBufI32( P.obicOut ).v[BufCu( P.cus ).v[n].obic * OBIC_OUT + w]
                               : CBufI32( P.dimdOut ).v[n * DIMD_OUT + w];
}

// CU::setIntraMergeCandParam's parameters of neighbour n into candidate slot k: an OBIC CU's OBIC converted (its first
// four blend modes, their weights, planar's weight from the last weighted one), else the DIMD or merge parameters of its
// DimdOut slot (blend modes below 0 as planar)
void mergeCandidateParams( int k, int n, int cls, int pf )
{
  for( int f = 0; f < MERGE_CAND; f++ )
  {
    mcSet( k, f, 0 );
  }
  if( cls == MERGE_NBR_OBIC )
  {
    mcSet( k, MC_MODE, mergeSource( n, cls, pf, OBIC_OUT_MODE ) );
    mcSet( k, MC_WEIGHT, mergeSource( n, cls, pf, OBIC_OUT_WEIGHT ) );
    int last = 0;   // an index into the merge weights, read as one into OBIC's (as the CPU does)
    for( int i = 0; i < OBIC_MODES - 2; i++ )
    {
      const int m = mergeSource( n, cls, pf, OBIC_OUT_MODE + i + 1 ), w = mergeSource( n, cls, pf, OBIC_OUT_WEIGHT + i + 1 );
      mcSet( k, MC_BLEND + i, max( m, 0 ) );
      mcSet( k, MC_WEIGHT + i + 2, w );
      last = w > 0 ? i + 2 : last;
    }
    const int wLast = mergeSource( n, cls, pf, OBIC_OUT_WEIGHT + last );
    if( wLast > 0 )
    {
      mcSet( k, MC_WEIGHT + 1, wLast );
    }
    else
    {
      mcSet( k, MC_WEIGHT + last, 0 );
      mcSet( k, MC_WEIGHT + 1, mergeSource( n, cls, pf, OBIC_OUT_WEIGHT + last - 1 ) );
    }
  }
  else
  {
    mcSet( k, MC_MODE, mergeSource( n, cls, pf, DIMD_OUT_MODE ) );
    for( int i = 0; i < 6; i++ )
    {
      mcSet( k, MC_BLEND + i, max( mergeSource( n, cls, pf, DIMD_OUT_BLEND_MODE + i ), 0 ) );
    }
    for( int i = 0; i < 8; i++ )
    {
      mcSet( k, MC_WEIGHT + i, mergeSource( n, cls, pf, DIMD_OUT_REL_WEIGHT + i ) );
    }
    for( int i = 0; i < 7; i++ )
    {
      mcSet( k, MC_LOC_DEP + i, mergeSource( n, cls, pf, DIMD_OUT_LOC_DEP + i ) );
    }
  }
}

// CU::areIntraParamsSame: slot k has the parameters of an earlier candidate
bool mergeSameParams( int k )
{
  for( int i = 0; i < k; i++ )
  {
    bool same = true;
    for( int f = MC_MODE; f < MC_X && same; f++ )
    {
      same = mcGet( k, f ) == mcGet( i, f );
    }
    if( same )
    {
      return true;
    }
  }
  return false;
}

// deriveIntraMergeCand's walk over the list (one invocation; the entries' validity and positions and the first valid
// ones' parameters in s_tmp): a valid entry at a position no candidate has yet whose parameters no candidate has is
// the next candidate, up to five; their number at MC_NUM
void mergeWalk( int base, int nbrNum, int numFetch )
{
  BufI32 md = BufI32( P.mergeData );
  for( int f = 0; f < MC_NUM; f++ )
  {
    s_spCost[f] = 0u;
  }
  int cnt = 0, numValid = 0;
  for( int k = 0; k < nbrNum && cnt < MERGE_CANDS; k++ )
  {
    const int cls = int( s_tmp[MT_ENTRY * k] );
    if( cls == 0 )
    {
      continue;   // its DIMD does not blend: not a candidate
    }
    const int n  = md.v[base + MERGE_HEAD + 2 * k];
    const int pf = numValid < numFetch ? numValid : -1;
    numValid++;
    const int x = int( s_tmp[MT_ENTRY * k + 1] ), y = int( s_tmp[MT_ENTRY * k + 2] );
    bool taken = false;
    for( int i = 0; i < cnt; i++ )
    {
      taken = taken || ( mcGet( i, MC_X ) == x && mcGet( i, MC_Y ) == y );
    }
    if( taken )
    {
      continue;
    }
    mergeCandidateParams( cnt, n, cls, pf );
    mcSet( cnt, MC_X, x );
    mcSet( cnt, MC_Y, y );
    cnt += mergeSameParams( cnt ) ? 0 : 1;
  }
  s_spCost[MC_NUM] = uint( cnt );
}

// deriveIntraMergeCand over the CU's list (all invocations): the entries' validity (a DIMD or MPM neighbour where its
// DIMD blends) and positions, the first valid ones' parameters fetched at once, then the walk
void mergeCandidates( int base, int nbrNum, int lid, int NT )
{
  BufI32 md = BufI32( P.mergeData );
  for( int k = lid; k < nbrNum; k += NT )
  {
    const int n = md.v[base + MERGE_HEAD + 2 * k], cls = md.v[base + MERGE_HEAD + 2 * k + 1];
    const bool valid = cls != MERGE_NBR_DIMD || CBufI32( P.dimdOut ).v[n * DIMD_OUT + DIMD_OUT_BLENDING] != 0;
    s_tmp[MT_ENTRY * k]     = int16_t( valid ? cls : 0 );
    s_tmp[MT_ENTRY * k + 1] = int16_t( BufCu( P.cus ).v[n].x );
    s_tmp[MT_ENTRY * k + 2] = int16_t( BufCu( P.cus ).v[n].y );
  }
  SYNC();
  if( lid == 0 )   // the first valid entries, to fetch
  {
    int p = 0;
    for( int k = 0; k < nbrNum && p < MT_PREFETCH; k++ )
    {
      if( s_tmp[MT_ENTRY * k] != int16_t( 0 ) )
      {
        s_tmp[MT_LIST + p++] = int16_t( k );
      }
    }
    s_tmp[MT_LIST + MT_PREFETCH] = int16_t( p );
  }
  SYNC();
  const int numFetch = int( s_tmp[MT_LIST + MT_PREFETCH] );
  for( int i = lid; i < numFetch * MT_WORDS; i += NT )
  {
    const int pf = i / MT_WORDS, w = i % MT_WORDS, k = int( s_tmp[MT_LIST + pf] );
    const int n = md.v[base + MERGE_HEAD + 2 * k], cls = md.v[base + MERGE_HEAD + 2 * k + 1];
    if( cls != MERGE_NBR_OBIC || w < OBIC_OUT )
    {
      s_tmp[MT_DATA + i] = int16_t( mergeSource( n, cls, -1, w ) );
    }
  }
  SYNC();
  if( lid == 0 )
  {
    mergeWalk( base, nbrNum, numFetch );
  }
}

// reorderIntraMergeCand's included modes, as a set of bits: planar and every mode of the num candidates
uvec3 mergeIncluded( int num )
{
  uvec3 inc = uvec3( 1u << PLANAR_IDX, 0u, 0u );
  for( int i = 0; i < num; i++ )
  {
    for( int j = -1; j < 6; j++ )
    {
      const int m = mcGet( i, j < 0 ? MC_MODE : MC_BLEND + j );
      inc[m >> 5] |= 1u << ( m & 31 );
    }
  }
  return inc;
}

bool mergeHas( uvec3 inc, int m )
{
  return ( ( inc[m >> 5] >> ( m & 31 ) ) & 1u ) != 0u;
}

// the template costs of the modes listed at MC_MODES for CU c with template h (all invocations), into MC_COST + mode:
// as many modes at once as s_tmp holds their templates (each the above arm's rows of c.w, then the left arm's rows of
// tw), predicted in parallel, then the SATD of the reconstruction against each, block by block in parallel
void mergeTemplateCosts( Cu c, MergeHead h, int dc, int maxPel, bool noPdpc, int lid, int NT )
{
  const int   numModes = int( s_spCost[MC_NUM_MODES] );
  const int   nA = c.w * h.th, T = nA + h.tw * c.h;   // template samples per mode
  const int   perPass = max( 1, MAX_REF / T );
  const ivec2 ka = timdKernel( c.w, h.th ), kl = timdKernel( h.tw, c.h );
  const int   colsA = c.w / ka.x, colsL = h.tw / kl.x;
  const int   nbA = colsA * ( h.th / ka.y ), B = nbA + colsL * ( c.h / kl.y );   // SATD blocks per mode
  for( int first = 0; first < numModes; first += perPass )
  {
    const int n = min( perPass, numModes - first );
    for( int i = lid; i < n * T; i += NT )
    {
      const int        s    = i % T;
      const int        mode = timdMode131( int( s_spCost[MC_MODES + first + i / T] ) );
      const TimdParams p    = timdParams( mode, c.w, c.h, noPdpc );
      const int x = s < nA ? h.tw + s % c.w : ( s - nA ) % h.tw;
      const int y = s < nA ? s / c.w : h.th + ( s - nA ) / h.tw;
      s_tmp[i] = int16_t( timdPredict( mode, p, x, y, c.w, c.h, h.tw, h.th, dc, maxPel ) );
    }
    SYNC();
    for( int i = lid; i < n * B; i += NT )
    {
      const int m = i / B, b = i % B, k = b - nbA;
      const int satd = b < nbA ? timdHad( c.x, c.y - h.th, m * T, c.w, ( b % colsA ) * ka.x, ( b / colsA ) * ka.y, ka.x, ka.y )
                               : timdHad( c.x - h.tw, c.y, m * T + nA, h.tw, ( k % colsL ) * kl.x, ( k / colsL ) * kl.y, kl.x, kl.y );
      atomicAdd( s_spCost[MC_COST + s_spCost[MC_MODES + first + m]], uint( satd ) );
    }
    SYNC();
  }
}

// the node of intra merge CU c (record cuIdx), all invocations: the candidates, the template costs of their modes
// where they are ranked (with verification also into Params::mergeCost, ~0 where none), the choice into its DimdOut
// slot; returns its mode (invocation 0)
int mergeNode( Cu c, int cuIdx, int bitDepth, bool noPdpc, int lid, int NT )
{
  BufI32    md = BufI32( P.mergeData );
  MergeHead h  = MergeHead( md.v[c.merge], md.v[c.merge + 1], md.v[c.merge + 2], md.v[c.merge + 3], md.v[c.merge + 4],
                            md.v[c.merge + 5], md.v[c.merge + 6], md.v[c.merge + 7], md.v[c.merge + 8],
                            md.v[c.merge + 9], md.v[c.merge + 10], md.v[c.merge + 11] );
  mergeCandidates( c.merge, h.nbrNum, lid, NT );
  SYNC();
  const int  num  = int( s_spCost[MC_NUM] );
  const bool rank = num >= 2 && h.tw > 0;   // else candidate 0 (reorderIntraMergeCand without costs)
  if( rank )
  {
    if( lid == 0 )   // the modes to rank by, their costs zero
    {
      const uvec3 inc = mergeIncluded( num );
      int n = 0;
      for( int m = 0; m < NUM_LUMA_MODE; m++ )
      {
        s_spCost[MC_COST + m] = 0u;
        if( mergeHas( inc, m ) )
        {
          s_spCost[MC_MODES + n++] = uint( m );
        }
      }
      s_spCost[MC_NUM_MODES] = uint( n );
    }
    timdFillReference( h, c.merge + MERGE_HEAD + 2 * h.nbrNum, c.x, c.y, c.w, c.h, bitDepth, lid, NT );
    const int dc = timdDc( h, c.w, c.h, lid, NT );
    mergeTemplateCosts( c, h, dc, ( 1 << bitDepth ) - 1, noPdpc, lid, NT );
  }
  if( pc.verify != 0 )   // the costs, which verification compares (~0 for the modes not ranked by)
  {
    BufU64      costOut = BufU64( P.mergeCost );
    const uvec3 inc     = mergeIncluded( num );
    for( int m = lid; m < NUM_LUMA_MODE; m += NT )
    {
      costOut.v[h.index * NUM_LUMA_MODE + m] = rank && mergeHas( inc, m ) ? uint64_t( s_spCost[MC_COST + m] ) : ~0ul;
    }
  }
  if( lid != 0 )
  {
    return -1;
  }
  // the candidate of the least weighted cost (its modes' costs by their weights, blend modes that are planar cost
  // nothing; weights as the CPU's unsigned 64-bit products), the first of equal ones
  int best = 0;
  if( rank )
  {
    uint64_t bestCost = ~0ul;
    for( int i = 0; i < num; i++ )   // (all their modes are ranked by)
    {
      uint64_t s = uint64_t( int64_t( mcGet( i, MC_WEIGHT ) ) ) * uint64_t( s_spCost[MC_COST + mcGet( i, MC_MODE )] )
                 + uint64_t( int64_t( mcGet( i, MC_WEIGHT + 1 ) ) ) * uint64_t( s_spCost[MC_COST + PLANAR_IDX] );
      for( int j = 0; j < 6; j++ )
      {
        const int m = mcGet( i, MC_BLEND + j );
        s += m > 0 ? uint64_t( int64_t( mcGet( i, MC_WEIGHT + 2 + j ) ) ) * uint64_t( s_spCost[MC_COST + m] ) : 0ul;
      }
      s >>= 6;
      if( s < bestCost )
      {
        bestCost = s;
        best     = i;
      }
    }
  }
  CBufI32   d = CBufI32( P.dimdOut );
  const int o = cuIdx * DIMD_OUT;
  d.v[o + DIMD_OUT_MODE]     = mcGet( best, MC_MODE );
  d.v[o + DIMD_OUT_BLENDING] = 1;
  for( int i = 0; i < 6; i++ )
  {
    d.v[o + DIMD_OUT_BLEND_MODE + i] = mcGet( best, MC_BLEND + i );
  }
  d.v[o + DIMD_OUT_BLEND_MODE + 6] = -1;
  for( int i = 0; i < 8; i++ )
  {
    d.v[o + DIMD_OUT_REL_WEIGHT + i] = mcGet( best, MC_WEIGHT + i );
  }
  for( int i = 0; i < 7; i++ )
  {
    d.v[o + DIMD_OUT_LOC_DEP + i] = mcGet( best, MC_LOC_DEP + i );
  }
  return mcGet( best, MC_MODE );
}

#endif
