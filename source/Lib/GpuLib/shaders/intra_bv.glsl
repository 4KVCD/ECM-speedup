// Block vectors (JVET-AM0138 and the tools before it): the list of block vectors a CU takes from the CUs around it
// (PU::getItmpMergeCandidate, then getArBvMergeCandidate's cascade), from the records of those CUs' vectors
// (Params::bvOut, BV_OUT per CU by its index in the CodingStructure). A vector is in integer samples (but SGPM's own
// in a record, BV_OUT_SGPM: 1/16 samples). Included by intra_rec.comp.

// getArBvMergeCandidate's lengths (NUM_BV, NUM_TMP_ARBVP): a list grows to at least NUM_BV, else NUM_TMP_ARBVP beyond
// the vectors of the neighbours; BV_LIST_MAX the GPU's room (the host takes the lists that fit)
const int NUM_BV           = 50;
const int NUM_TMP_ARBVP    = 20;
const int BV_LIST_MAX      = 80;
const int TMP_TEMPLATE_SIZE = 4;

// the list in shared memory a node is free to take (sign prediction's coefficients): hor at 2 k, ver at 2 k + 1
#define s_bvList s_spCoef

int bvPack( ivec2 bv )
{
  return ( bv.x & 0xFFFF ) | ( bv.y << 16 );
}

ivec2 bvUnpack( int w )
{
  return ivec2( bitfieldExtract( w, 0, 16 ), w >> 16 );
}

ivec2 bvListAt( int k )
{
  return ivec2( int( s_bvList[2 * k] ), int( s_bvList[2 * k + 1] ) );
}

// PU::validItmpBv for the CU at (x, y) of w x h read by task `before`: the block at the vector and its template's
// top-left corner are decoded (CodingStructure::isDecomp: by blocks before it)
bool bvValid( int x, int y, int w, int h, ivec2 bv, int before )
{
  OwnerMap om = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  return decodedBefore( om, x + bv.x + w - 1, y + bv.y + h - 1, before )
         && decodedBefore( om, x + bv.x - TMP_TEMPLATE_SIZE, y + bv.y - TMP_TEMPLATE_SIZE, before );
}

// the index of the luma CU covering (x, y) (-1: outside the picture)
int bvCuAt( int x, int y )
{
  if( x < 0 || y < 0 || x >= P.width[0] || y >= P.height[0] )
  {
    return -1;
  }
  return BufI32( P.timdData ).v[P.bvMap + ( y >> 2 ) * P.bvMapW + ( x >> 2 )];
}

// PU::checkBvAvailable: the list's first n hold bv
bool bvListHas( int n, ivec2 bv )
{
  for( int k = 0; k < n; k++ )
  {
    if( bvListAt( k ) == bv )
    {
      return true;
    }
  }
  return false;
}

// bv appended to the list of n where it is valid for the CU and not in it; returns the new length
int bvListAdd( int n, ivec2 bv, Cu c, int before )
{
  if( n < BV_LIST_MAX && bvValid( c.x, c.y, c.w, c.h, bv, before ) && !bvListHas( n, bv ) )
  {
    s_bvList[2 * n]     = int16_t( bv.x );
    s_bvList[2 * n + 1] = int16_t( bv.y );
    n++;
  }
  return n;
}

// a CU's list record in Params::timdData (at Cu::bvList; layout shared with GpuIntraRec.cpp): its CodingStructure
// index, the length its list keeps (0: the cascade's), the indices of the BV_NBRS CUs at getItmpMergeCandidate's
// positions (-1 none)
const int BV_LIST_INDEX = 0;
const int BV_LIST_KEEP  = 1;
const int BV_LIST_NBRS  = 2;
const int BV_LIST       = BV_LIST_NBRS + BV_NBRS;

// The lists' candidates read ahead by the whole workgroup (bvValid's owner map loads, the records' loads) into the end
// of its scratch (BLEND_ACC's sums, free in a node), so that invocation 0 only compares and appends (bvListHas, in
// shared memory) instead of loading them one after the other: getItmpMergeCandidate's per neighbour and vector its
// vector and validity (2 words), a cascade round's per list vector and position a slot (the CU's number of vectors, -1
// for none, and its first cheapest one, BV_OUT_FUSION; then per vector moved and plain each its vector and validity)
const int BV_CAND_SLOT = 2 + 4 * BV_MAX;
const int BV_CAND_BASE = BLEND_ACC - BV_LIST_MAX * 5 * BV_CAND_SLOT;

shared int s_bvLen;   // the list's length (invocation 0's), for every invocation

int  bvcGet( int o )        { return WBufI32( P.scratch ).v[int( gl_WorkGroupID.x ) * BLEND_ACC + BV_CAND_BASE + o]; }
void bvcSet( int o, int v ) { WBufI32( P.scratch ).v[int( gl_WorkGroupID.x ) * BLEND_ACC + BV_CAND_BASE + o] = v; }

// a barrier after which the scratch's and shared memory's writes are visible to the workgroup
void bvSync()
{
  controlBarrier( gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsBuffer | gl_StorageSemanticsShared,
                  gl_SemanticsAcquireRelease );
}

// bv appended to the list of n where it is valid (read ahead) and not in it; returns the new length
int bvListAddKnown( int n, ivec2 bv, bool valid )
{
  if( n < BV_LIST_MAX && valid && !bvListHas( n, bv ) )
  {
    s_bvList[2 * n]     = int16_t( bv.x );
    s_bvList[2 * n + 1] = int16_t( bv.y );
    n++;
  }
  return n;
}

// getItmpMergeCandidate's list of CU c (all invocations, uniform control flow; the node of task `before`): the vectors
// of the CUs at its neighbour positions each in its order, each once and valid. Returns the length
int bvNeighbourList( Cu c, int before, int lid, int NT )
{
  BufI32  data = BufI32( P.timdData );
  CBufI32 recs = CBufI32( P.timdData );   // (records written by other workgroups' nodes)
  for( int i = lid; i < BV_NBRS * BV_MAX; i += NT )   // (neighbour i / BV_MAX, its vector i % BV_MAX)
  {
    const int   src = data.v[c.bvList + BV_LIST_NBRS + i / BV_MAX], at = P.bvOut + src * BV_OUT;
    const bool  has = src >= 0 && i % BV_MAX < recs.v[at];
    const ivec2 bv  = has ? bvUnpack( recs.v[at + 1 + i % BV_MAX] ) : ivec2( 0 );
    bvcSet( 2 * i, bvPack( bv ) );
    bvcSet( 2 * i + 1, has && bvValid( c.x, c.y, c.w, c.h, bv, before ) ? 1 : 0 );
  }
  bvSync();
  if( lid == 0 )
  {
    int n = 0;
    for( int i = 0; i < BV_NBRS * BV_MAX; i++ )
    {
      n = bvListAddKnown( n, bvUnpack( bvcGet( 2 * i ) ), bvcGet( 2 * i + 1 ) != 0 );
    }
    s_bvLen = n;
  }
  bvSync();
  const int n = s_bvLen;
  bvSync();   // (read before s_bvLen is written again)
  return n;
}

// the cascade's position p of CU c: its centre, then its corners
ivec2 bvCascadePos( Cu c, int p )
{
  return p == 0 ? ivec2( c.x + ( c.w >> 1 ), c.y + ( c.h >> 1 ) )
                : ivec2( ( p & 1 ) != 0 ? c.x : c.x + c.w - 1, p <= 2 ? c.y : c.y + c.h - 1 );
}

// one step of a cascade (invocation 0): the CUs at the five positions of CU c (its centre and corners) moved by cmv,
// decoded no later than it, each vector of those CUs (their getSparseArBvMergeCandidate set with sparse) added to the
// list of n moved by cmv, then as it is, until it has total. Returns the length
int bvCascadeStep( Cu c, int n, int total, ivec2 cmv, bool sparse, int before )
{
  BufI32      data    = BufI32( P.timdData );
  CBufI32     recs    = CBufI32( P.timdData );
  const int   cuIndex = data.v[c.bvList + BV_LIST_INDEX];
  for( int p = 0; p < 5 && n < total; p++ )
  {
    const ivec2 pos = bvCascadePos( c, p );
    const int   src = bvCuAt( pos.x + cmv.x, pos.y + cmv.y );
    if( src < 0 || src > cuIndex )   // (getPURestricted: decoded no later than the CU)
    {
      continue;
    }
    int       at     = P.bvOut + src * BV_OUT;
    int       num    = recs.v[at];
    const int fusion = sparse ? -1 : recs.v[at + BV_OUT_FUSION];
    if( sparse && recs.v[at + BV_OUT_SPARSE] >= 0 )   // (an IntraTMP fusion CU's own vector and its cheapest)
    {
      at  = at + BV_OUT_SPARSE;
      num = recs.v[at];
    }
    // each vector moved by cmv, then as it is, until the list has total; an IntraTMP fusion CU's cheapest vector after
    // its fused ones (from j = fusion) even so: the CPU's fusion loop leaves only itself at the list's length, so the
    // cheapest's moved, else plain, vector may still enter, and the step ends where one has entered at the list's
    // length (one call of bvListAdd: this is inlined at every list)
    for( int j = 0; j < num; j++ )
    {
      const bool tail = fusion >= 0 && j >= fusion;
      if( !tail && n >= total )
      {
        if( fusion < 0 )
        {
          break;
        }
        j = fusion - 1;   // (on to the cheapest)
        continue;
      }
      const ivec2 bv = bvUnpack( recs.v[at + 1 + j] );
      for( int k = 0; k < 2 && ( tail || n < total ); k++ )
      {
        const int m = bvListAdd( n, k == 0 ? cmv + bv : bv, c, before );
        if( tail && m > n && m >= total )
        {
          return m;
        }
        n = m;
      }
    }
  }
  return n;
}

// bvCascadeStep's of the cascade (not sparse) from the slots read ahead (invocation 0): the step of the round's list
// vector r
int bvCascadeStepKnown( int n, int total, int r )
{
  for( int p = 0; p < 5 && n < total; p++ )
  {
    const int o = ( r * 5 + p ) * BV_CAND_SLOT, num = bvcGet( o ), fusion = bvcGet( o + 1 );
    for( int j = 0; j < num; j++ )   // (no CU there: num -1)
    {
      const bool tail = fusion >= 0 && j >= fusion;
      if( !tail && n >= total )
      {
        if( fusion < 0 )
        {
          break;
        }
        j = fusion - 1;
        continue;
      }
      for( int k = 0; k < 2 && ( tail || n < total ); k++ )
      {
        const int w = o + 2 + 4 * j + 2 * k;
        const int m = bvListAddKnown( n, bvUnpack( bvcGet( w ) ), bvcGet( w + 1 ) != 0 );
        if( tail && m > n && m >= total )
        {
          return m;
        }
        n = m;
      }
    }
  }
  return n;
}

// getArBvMergeCandidate's cascade of the list of n (all invocations, uniform control flow): per vector of the list
// (those it adds too, in rounds) a cascade step, until the list has max(NUM_BV, n + NUM_TMP_ARBVP). Per round the
// workgroup reads the steps' CUs and vectors ahead (bvCascadeStep's, into the slots), then invocation 0 takes them in
// the steps' order. Returns the length
int bvCascade( Cu c, int n, int before, int lid, int NT )
{
  if( n == 0 )
  {
    return 0;
  }
  CBufI32   recs    = CBufI32( P.timdData );
  const int cuIndex = BufI32( P.timdData ).v[c.bvList + BV_LIST_INDEX];
  const int total   = max( NUM_BV, n + NUM_TMP_ARBVP );
  int       end     = 0;
  do
  {
    const int start = end;
    end = n;
    for( int i = lid; i < ( end - start ) * 5 * BV_MAX; i += NT )   // (slot i / BV_MAX, its vector i % BV_MAX)
    {
      const int   r = i / BV_MAX / 5, p = i / BV_MAX % 5, j = i % BV_MAX, o = ( r * 5 + p ) * BV_CAND_SLOT;
      const ivec2 cmv = bvListAt( start + r ), pos = bvCascadePos( c, p ) + cmv;
      const int   src = bvCuAt( pos.x, pos.y ), at = P.bvOut + src * BV_OUT;
      const bool  cu  = src >= 0 && src <= cuIndex;   // (getPURestricted: decoded no later than the CU)
      const int   num = cu ? recs.v[at] : -1;
      if( j == 0 )
      {
        bvcSet( o, num );
        bvcSet( o + 1, cu ? recs.v[at + BV_OUT_FUSION] : -1 );
      }
      if( j < num )
      {
        const ivec2 bv = bvUnpack( recs.v[at + 1 + j] );
        for( int k = 0; k < 2; k++ )
        {
          const ivec2 v = k == 0 ? cmv + bv : bv;
          bvcSet( o + 2 + 4 * j + 2 * k, bvPack( v ) );
          bvcSet( o + 3 + 4 * j + 2 * k, bvValid( c.x, c.y, c.w, c.h, v, before ) ? 1 : 0 );
        }
      }
    }
    bvSync();
    if( lid == 0 )
    {
      for( int m = start; m < end && n < total; m++ )
      {
        n = bvCascadeStepKnown( n, total, m - start );
      }
      s_bvLen = n;
    }
    bvSync();
    n = s_bvLen;
    bvSync();   // (read before invocation 0 writes the next round's)
  } while( n > end && n < total );
  return n;
}

// CU c's list of block vectors (all invocations, uniform control flow; the node of task `before`):
// getItmpMergeCandidate's, kept to its first BV_LIST_KEEP (DIMD's), else (TIMD's) cascaded. Returns the length
int bvMergeList( Cu c, int before, int lid, int NT )
{
  const int keep = BufI32( P.timdData ).v[c.bvList + BV_LIST_KEEP];
  const int n    = bvNeighbourList( c, before, lid, NT );
  return keep > 0 ? min( n, keep ) : bvCascade( c, n, before, lid, NT );
}

// CU c's record of its vectors (its node's choices: num of bv)
void bvStore( Cu c, int num, ivec2 bv[3] )
{
  CBufI32   data = CBufI32( P.timdData );
  const int at   = P.bvOut + BufI32( P.timdData ).v[c.bvList + BV_LIST_INDEX] * BV_OUT;
  data.v[at] = num;
  for( int j = 0; j < num; j++ )
  {
    data.v[at + 1 + j] = bvPack( bv[j] );
  }
  data.v[at + BV_OUT_SPARSE] = -1;   // (the same for getSparseArBvMergeCandidate)
  data.v[at + BV_OUT_FUSION] = -1;   // (an IntraTMP fusion CU's: tmpFusionSetup)
  data.v[at + BV_OUT_SGPM]   = -1;   // (SGPM's list takes them x 16)
}

// after bvStore: the vectors SGPM's list takes of CU c instead (num of mv, in 1/16 samples)
void bvStoreSgpm( Cu c, int num, ivec2 mv[2] )
{
  CBufI32   data = CBufI32( P.timdData );
  const int at   = P.bvOut + BufI32( P.timdData ).v[c.bvList + BV_LIST_INDEX] * BV_OUT + BV_OUT_SGPM;
  data.v[at] = num;
  for( int j = 0; j < 2; j++ )
  {
    data.v[at + 1 + 2 * j] = mv[j].x;
    data.v[at + 2 + 2 * j] = mv[j].y;
  }
}
