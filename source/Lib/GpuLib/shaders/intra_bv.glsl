// Block vectors (JVET-AM0138 and the tools before it): the list of block vectors a CU takes from the CUs around it
// (PU::getItmpMergeCandidate, then getArBvMergeCandidate's cascade), from the records of those CUs' vectors
// (Params::bvOut, BV_OUT per CU by its index in the CodingStructure). A vector is in integer samples. Included by
// intra_rec.comp.

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

// getItmpMergeCandidate's list of CU c (invocation 0; the node of task `before`): the vectors of the CUs at its
// neighbour positions each in its order, each once and valid. Returns the length
int bvNeighbourList( Cu c, int before )
{
  BufI32  data = BufI32( P.timdData );
  CBufI32 recs = CBufI32( P.timdData );   // (records written by other workgroups' nodes)
  int     n    = 0;
  for( int k = 0; k < BV_NBRS; k++ )
  {
    const int src = data.v[c.bvList + BV_LIST_NBRS + k];
    if( src < 0 )
    {
      continue;
    }
    const int at = P.bvOut + src * BV_OUT;
    for( int j = 0; j < recs.v[at]; j++ )
    {
      n = bvListAdd( n, bvUnpack( recs.v[at + 1 + j] ), c, before );
    }
  }
  return n;
}

// one step of a cascade (invocation 0): the CUs at the five positions of CU c (its centre and corners) moved by cmv,
// decoded no later than it, each vector of those CUs (their getSparseArBvMergeCandidate set with sparse) added to the
// list of n moved by cmv, then as it is, until it has total. Returns the length
int bvCascadeStep( Cu c, int n, int total, ivec2 cmv, bool sparse, int before )
{
  BufI32      data    = BufI32( P.timdData );
  CBufI32     recs    = CBufI32( P.timdData );
  const int   cuIndex = data.v[c.bvList + BV_LIST_INDEX];
  const ivec2 pos[5]  = ivec2[5]( ivec2( c.x + ( c.w >> 1 ), c.y + ( c.h >> 1 ) ), ivec2( c.x, c.y ),
                                  ivec2( c.x + c.w - 1, c.y ), ivec2( c.x, c.y + c.h - 1 ),
                                  ivec2( c.x + c.w - 1, c.y + c.h - 1 ) );
  for( int p = 0; p < 5 && n < total; p++ )
  {
    const int src = bvCuAt( pos[p].x + cmv.x, pos[p].y + cmv.y );
    if( src < 0 || src > cuIndex )   // (getPURestricted: decoded no later than the CU)
    {
      continue;
    }
    int at  = P.bvOut + src * BV_OUT;
    int num = recs.v[at];
    if( sparse && recs.v[at + BV_OUT_SPARSE] >= 0 )   // (an IntraTMP fusion CU's own vector and its cheapest)
    {
      at  = at + BV_OUT_SPARSE;
      num = recs.v[at];
    }
    for( int j = 0; j < num && n < total; j++ )
    {
      const ivec2 bv = bvUnpack( recs.v[at + 1 + j] );
      n = bvListAdd( n, cmv + bv, c, before );
      if( n < total )
      {
        n = bvListAdd( n, bv, c, before );
      }
    }
  }
  return n;
}

// getArBvMergeCandidate's cascade of the list of n (invocation 0): per vector of the list (those it adds too, in
// rounds) a cascade step, until the list has max(NUM_BV, n + NUM_TMP_ARBVP). Returns the length
int bvCascade( Cu c, int n, int before )
{
  if( n == 0 )
  {
    return 0;
  }
  const int total = max( NUM_BV, n + NUM_TMP_ARBVP );
  int       end   = 0;
  do
  {
    const int start = end;
    end = n;
    for( int m = start; m < end && n < total; m++ )
    {
      n = bvCascadeStep( c, n, total, bvListAt( m ), false, before );
    }
  } while( n > end && n < total );
  return n;
}

// CU c's list of block vectors (invocation 0; the node of task `before`): getItmpMergeCandidate's, kept to its first
// BV_LIST_KEEP (DIMD's), else (TIMD's) cascaded. Returns the length
int bvMergeList( Cu c, int before )
{
  const int keep = BufI32( P.timdData ).v[c.bvList + BV_LIST_KEEP];
  const int n    = bvNeighbourList( c, before );
  return keep > 0 ? min( n, keep ) : bvCascade( c, n, before );
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
}
