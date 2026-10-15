// IntraTMP's search (IntraPrediction::searchCandidateFromOnePicIntra, JVET-V0130 and the proposals after it): the
// node of an IntraTMP CU finds the block vectors of its list (MTMP_NUM) as the decoder does - a sparse search of the
// regions around the CU, the vectors of its neighbours (getItmpMergeCandidate, sortInOut, getArBvMergeCandidate) and
// those cascaded from the sparse search's best (getSparseArBvMergeCandidate) with windows clustered into the sparse
// lists, a refinement of each candidate's window, the lists of the top and the left template merged into the
// L-shaped one's - into its DimdOut (DIMD_OUT_TMP_*), whose entry tmpIdx its block copies. The template costs
// (calcTemplateDiff) are computed in parallel, in chunks of positions in the CPU's order, and the CPU's lists replayed
// from them by invocation 0: every test of a cost is "less than a threshold", which the CPU's early ends cannot
// change. No LIC, fusion, FLM or sub-sample refinement yet (the host keeps such CUs on the CPU). An EIP CU with a
// vector runs the sparse search alone, for its best (TMP_HEAD_ONLY_BEST). Included by intra_rec.comp after
// intra_bv.glsl.

const int TMP_COST_SHIFT    = 3;    // TMP_TEMPLATE_COST_SHIFT: the template row and column next to the block weigh 8
const int MTMP_NUM          = 19;
const int MTMP_NUM_SPARSE   = 30;
const int TL_NUM            = 3;
const int TL_NUM_SPARSE     = 6;
const int TMP_MERGE_RANGE   = 5;
const int TMP_ARBVP_RANGE   = 2;
const int TMP_MRG_REG_ID    = 6;    // a neighbour's vector's region (7: one cascaded, 8: one from the sparse search)
const int TMP_LOG2MINCOST   = 5;
const int NUM_TMP_ARBVP_S   = 5;
const int MAX_TO_KEEP_MERGE = 36;
const int INIT_TL_POS       = MTMP_NUM - TL_NUM_SPARSE;
const int TMP_L_SHAPE = 1, TMP_LEFT = 2, TMP_ABOVE = 3;   // RefTemplateType

// an IntraTMP CU's record in Params::timdData (at Cu::tmp; layout shared with GpuIntraRec.cpp)
const int TMP_HEAD_TEMP_TYPE  = 0;   // CodingUnit::tempType
const int TMP_HEAD_IDX        = 1;   // tmpIdx
const int TMP_HEAD_SAMPLING   = 2;   // TMP_SAMPLING (the SPS's ItmpLicMode: 3, else 4)
const int TMP_HEAD_LIC_MODE   = 3;   // the SPS's ItmpLicMode
const int TMP_HEAD_NEED_TL    = 4;   // needTopLeft: tmpIdx reaches the list's entries of the top / left template
const int TMP_HEAD_NUM_REFINE = 5;   // mtmpNumRefine[0] (tmpIdx + 1 without needTopLeft)
const int TMP_HEAD_BIG_CTU    = 6;   // the CTU is larger than 128 (the clustering's windows)
const int TMP_HEAD_ONLY_BEST  = 7;   // an EIP CU with a vector: the sparse search's best alone (mtmpNumSparse 1)
const int TMP_HEAD_REGIONS    = 8;   // per regionId mvXMin, mvXMax, mvYMin, mvYMax (clipMvIntraConstraint), 6 x 4
const int TMP_HEAD            = TMP_HEAD_REGIONS + 24;

// the node's work area in the scratch (the workgroup's blending sums, free in a node): a chunk of positions and their
// costs, the lists (a length, then entries of TE ints), the vector list's costs, the clustering's result
const int TMP_CHUNK = 256;
const int TE        = 8;   // an entry: x, y, its window's top-left x / y and bottom-right x / y, region, cost
const int TE_X = 0, TE_Y = 1, TE_TLX = 2, TE_TLY = 3, TE_BRX = 4, TE_BRY = 5, TE_REGION = 6, TE_COST = 7;
const int TS_POS    = 0;                              // per position: x | y << 16, its tag (region / list entry)
const int TS_COST   = TS_POS + 2 * TMP_CHUNK;         // per position its 3 costs
const int TS_LIST   = 1 + ( MTMP_NUM_SPARSE + 1 ) * TE;
const int TS_SPARSE = TS_COST + 3 * TMP_CHUNK;        // 3 lists: the L-shape's (or the only one), the top's, the left's
const int TS_REFINE = TS_SPARSE + 3 * TS_LIST;        // 3 lists alike
const int TS_MCOST  = TS_REFINE + 3 * TS_LIST;        // per vector of the list its 3 costs
const int TS_WIN    = TS_MCOST + 3 * BV_LIST_MAX;     // which lists took vectors (bits), the refinement's 3 thresholds

shared int s_tmpCount;   // the chunk's positions
shared int s_tmpDone;    // the last chunk

int  tsBase()              { return int( gl_WorkGroupID.x ) * BLEND_ACC; }
int  tsGet( int o )        { return CBufI32( P.scratch ).v[tsBase() + o]; }
void tsSet( int o, int v ) { CBufI32( P.scratch ).v[tsBase() + o] = v; }

void tsSync()
{
  controlBarrier( gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsBuffer | gl_StorageSemanticsShared,
                  gl_SemanticsAcquireRelease );
}

// ----- the template costs

// calcTemplateDiff's three costs of vector (bx, by) for CU c with template type tempType (exact; the CPU's early ends
// only stop where a cost is no longer below its threshold): x the weighted one (the row and the column next to the
// block weigh 2^TMP_COST_SHIFT; the L-shape's with its corner, else the above or left template's), y the L-shape's
// top part, z its left part
ivec3 tmpCosts( Cu c, int tempType, int bx, int by )
{
  int sum = 0, top = 0, left = 0;
  if( tempType != TMP_LEFT )   // the rows above (with the corner of the L-shape)
  {
    const int x0 = tempType == TMP_L_SHAPE ? -TMP_TEMPLATE_SIZE : 0;
    for( int y = -TMP_TEMPLATE_SIZE; y < 0; y++ )
    {
      int corner = 0, row = 0;
      for( int x = x0; x < c.w; x++ )
      {
        const int d = abs( recoAt( 0, c.x + bx + x, c.y + by + y ) - recoAt( 0, c.x + x, c.y + y ) );
        corner += x < 0 ? d : 0;
        row    += x < 0 ? 0 : d;
      }
      top += row;
      sum += corner + ( y == -1 ? row << TMP_COST_SHIFT : row );
    }
  }
  if( tempType != TMP_ABOVE )   // the columns left
  {
    for( int y = 0; y < c.h; y++ )
    {
      int row = 0, next = 0;
      for( int x = -TMP_TEMPLATE_SIZE; x < 0; x++ )
      {
        next = abs( recoAt( 0, c.x + bx + x, c.y + by + y ) - recoAt( 0, c.x + x, c.y + y ) );   // (x -1 last)
        row += next;
      }
      left += row;
      sum += row + next * ( ( 1 << TMP_COST_SHIFT ) - 1 );
    }
  }
  return ivec3( sum, top, left );
}

// the costs of the chunk's s_tmpCount positions (all invocations, after a tsSync; positions at TS_POS), into TS_COST
void tmpChunkCosts( Cu c, int tempType, int lid, int NT )
{
  for( int k = lid; k < s_tmpCount; k += NT )
  {
    const int   p  = tsGet( TS_POS + 2 * k );
    const ivec3 cs = tmpCosts( c, tempType, bitfieldExtract( p, 0, 16 ), p >> 16 );
    tsSet( TS_COST + 3 * k, cs.x );
    tsSet( TS_COST + 3 * k + 1, cs.y );
    tsSet( TS_COST + 3 * k + 2, cs.z );
  }
  tsSync();
}

void tmpEmit( inout int k, int x, int y, int tag )
{
  tsSet( TS_POS + 2 * k, ( x & 0xFFFF ) | ( y << 16 ) );
  tsSet( TS_POS + 2 * k + 1, tag );
  k++;
}

// ----- the lists (invocation 0)

int  tlSize( int list )                       { return tsGet( list ); }
int  tlAt( int list, int i, int f )           { return tsGet( list + 1 + i * TE + f ); }
void tlSetAt( int list, int i, int f, int v ) { tsSet( list + 1 + i * TE + f, v ); }

// updateCandList (UnitTools.h, uint64 costs): entry e into the list of capacity num after the entries of equal cost;
// where the list is full, the last drops out, or e where it costs no less than the last
void tlUpdate( int list, int e[TE], int num )
{
  const int size = min( tlSize( list ), num );
  int shift = 0;
  while( shift < size && e[TE_COST] < tlAt( list, size - 1 - shift, TE_COST ) )
  {
    shift++;
  }
  if( size >= num && shift == 0 )
  {
    return;
  }
  const int last = size >= num ? size - 1 : size;   // (the new last entry)
  for( int i = last; i > size - shift; i-- )
  {
    for( int f = 0; f < TE; f++ )
    {
      tlSetAt( list, i, f, tlAt( list, i - 1, f ) );
    }
  }
  for( int f = 0; f < TE; f++ )
  {
    tlSetAt( list, size - shift, f, e[f] );
  }
  tsSet( list, last + 1 );
}

void tlErase( int list, int i )
{
  const int size = tlSize( list );
  for( int j = i; j < size - 1; j++ )
  {
    for( int f = 0; f < TE; f++ )
    {
      tlSetAt( list, j, f, tlAt( list, j + 1, f ) );
    }
  }
  tsSet( list, size - 1 );
}

bool tlHas( int list, int x, int y )
{
  for( int i = 0; i < tlSize( list ); i++ )
  {
    if( tlAt( list, i, TE_X ) == x && tlAt( list, i, TE_Y ) == y )
    {
      return true;
    }
  }
  return false;
}

// a list entry's values
int[TE] tlEntry( int x, int y, ivec2 tl, ivec2 br, int region, int cost )
{
  return int[TE]( x, y, tl.x, tl.y, br.x, br.y, region, cost );
}

// region r of the record at head: mvXMin, mvXMax, mvYMin, mvYMax
ivec4 tmpRegion( int head, int r )
{
  BufI32    data = BufI32( P.timdData );
  const int at   = head + TMP_HEAD_REGIONS + 4 * r;
  return ivec4( data.v[at], data.v[at + 1], data.v[at + 2], data.v[at + 3] );
}

// PU::sortInOut (more than MAX_TO_KEEP_MERGE vectors): those outside every region first, then those inside, 24 and 12
// of them (more of one where the other has fewer); returns the length (itmpSize)
int tmpSortInOut( int head, int n )
{
  if( n <= MAX_TO_KEEP_MERGE )
  {
    return n;
  }
  bool outside[BV_LIST_MAX];
  int  numOut = 0;
  for( int i = 0; i < n; i++ )
  {
    const ivec2 bv     = bvListAt( i );
    bool        inside = false;
    for( int r = 0; r < 6; r++ )
    {
      const ivec4 g = tmpRegion( head, r );
      inside = inside || ( g.y >= g.x && g.w >= g.z && bv.x >= g.x && bv.x <= g.y && bv.y >= g.z && bv.y <= g.w );
    }
    outside[i] = !inside;
    numOut += inside ? 0 : 1;
  }
  const int numIn = n - numOut, keepOut = MAX_TO_KEEP_MERGE - MAX_TO_KEEP_MERGE / 3, keepIn = MAX_TO_KEEP_MERGE / 3;
  int takeOut = numOut, takeIn = numIn;
  if( numOut > keepOut && numIn > keepIn )
  {
    takeOut = keepOut;
    takeIn  = keepIn;
  }
  else if( numOut > keepOut )
  {
    takeOut = min( numOut, MAX_TO_KEEP_MERGE - numIn );
  }
  else if( numIn > keepIn )
  {
    takeIn = min( numIn, MAX_TO_KEEP_MERGE - numOut );
  }
  int sorted[BV_LIST_MAX];   // (packed)
  int m = 0;
  for( int pass = 0; pass < 2; pass++ )   // outside first
  {
    int taken = 0;
    for( int i = 0; i < n && taken < ( pass == 0 ? takeOut : takeIn ); i++ )
    {
      if( outside[i] == ( pass == 0 ) )
      {
        sorted[m++] = bvPack( bvListAt( i ) );
        taken++;
      }
    }
  }
  for( int i = 0; i < m; i++ )
  {
    const ivec2 bv = bvUnpack( sorted[i] );
    s_bvList[2 * i]     = int16_t( bv.x );
    s_bvList[2 * i + 1] = int16_t( bv.y );
  }
  return m;
}

// clipMvIntraConstraintRefine for a candidate at (px, py) of region r (bounds g) not transferred: its window around it
// of S / 2 (one less to the right / below for even S), within the region but where the exception allows (region 3 /
// 5 left, 4 up past its edge where the picture lets the template fit; 3 right, 1 down)
ivec4 tmpRefineWindow( ivec4 g, int px, int py, int r, int S, bool extLeft, bool extTop, bool exception )
{
  const int rr = S >> 1, odd = S % 2 != 0 ? 0 : 1;
  const int xMin = exception && ( r == 3 || r == 5 ) && extLeft ? px - rr : max( g.x, px - rr );
  const int xMax = exception && r == 3 ? px + rr - odd : min( g.y, px + rr - odd );
  const int yMin = exception && r == 4 && extTop ? py - rr : max( g.z, py - rr );
  const int yMax = exception && r == 1 ? py + rr - odd : min( g.w, py + rr - odd );
  return ivec4( xMin, xMax, yMin, yMax );
}

// ----- the node

// the node of IntraTMP CU c (record cuIndex; all invocations, uniform control flow): its list of vectors into its
// DimdOut, the vector its block copies (and the list's first) into its record of vectors
void tmpNode( Cu c, int cuIndex, int bitDepth, int lid, int NT )
{
  BufI32     data     = BufI32( P.timdData );
  const int  head     = c.tmp;
  const int  tempType = data.v[head + TMP_HEAD_TEMP_TYPE];
  const int  S        = data.v[head + TMP_HEAD_SAMPLING];
  const bool licMode  = data.v[head + TMP_HEAD_LIC_MODE] != 0;
  const bool needTL   = data.v[head + TMP_HEAD_NEED_TL] != 0;
  const bool bigCtu   = data.v[head + TMP_HEAD_BIG_CTU] != 0;
  const int  numTypes = needTL && tempType == TMP_L_SHAPE ? 3 : 1;   // (checkAllTmpTypes)
  const int  W = c.w, H = c.h;
  OwnerMap   om = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  const bool onlyBest = data.v[head + TMP_HEAD_ONLY_BEST] != 0;   // (an EIP CU with a vector)
  const int  numSparse[3] = int[3]( onlyBest ? 1 : MTMP_NUM_SPARSE, TL_NUM_SPARSE, TL_NUM_SPARSE );
  const int  numRefine[3] = int[3]( data.v[head + TMP_HEAD_NUM_REFINE], TL_NUM, TL_NUM );
  const int  numTL = ( W + TMP_TEMPLATE_SIZE ) * TMP_TEMPLATE_SIZE + TMP_TEMPLATE_SIZE * H;   // numPixTopLeft
  const int  numTop = W * TMP_TEMPLATE_SIZE, numLeft = TMP_TEMPLATE_SIZE * H;

  // invocation 0's state: the vector list's parts (itmpSize, itmpSize1), the thresholds, the enumeration
  int itmpSize = 0, itmpSize1 = 0, numList = 0;
  int pDiff[3] = int[3]( numTL << bitDepth, numTop << bitDepth, numLeft << bitDepth );
  if( lid == 0 )
  {
    itmpSize  = tmpSortInOut( head, bvNeighbourList( c, g_ti ) );
    itmpSize1 = bvCascade( c, itmpSize, g_ti );
    for( int t = 0; t < 3; t++ )
    {
      tsSet( TS_SPARSE + t * TS_LIST, 0 );
      tsSet( TS_REFINE + t * TS_LIST, 0 );
    }
  }

  // the sparse search: the regions 3, 4, 5, 0, 1, 2 (but for a small block a region inside a neighbour vector's
  // window), rows from the bottom up by S, row r's first position S - 1 - (r - 1) % S left of the right edge, then
  // to the left by S; in regions 4 and 5 (beyond the CU in its CTU) the positions whose block is decoded, until a
  // row's first such. Each chunk costed in parallel, then put into the sparse lists by invocation 0
  int   checkIdx = 0, region = 0, y = 0, x = 0, shiftX = 0;
  bool  inRegion = false, newRow = false, rowFound = false, checkR4R5 = true;
  ivec4 g = ivec4( 0 );
  for( ;; )
  {
    if( lid == 0 )
    {
      int k = 0;
      while( k < TMP_CHUNK && checkIdx < 6 )
      {
        if( !inRegion )
        {
          region = ( checkIdx + 3 ) % 6;
          g      = tmpRegion( head, region );
          bool skip = g.y < g.x || g.w < g.z;
          checkR4R5 = true;
          if( !skip && ( region == 4 || region == 5 ) )
          {
            skip      = !decodedBefore( om, c.x + g.x + W - 1, c.y + g.z + H - 1, g_ti );
            checkR4R5 = !decodedBefore( om, c.x + g.y + W - 1, c.y + g.w + H - 1, g_ti );
          }
          for( int m = 0; !skip && W <= 8 && H <= 8 && m < itmpSize1; m++ )
          {
            const ivec2 bv = bvListAt( m );
            const int   r  = m < itmpSize ? TMP_MERGE_RANGE : TMP_ARBVP_RANGE;
            skip = g.x >= bv.x - r && g.z >= bv.y - r && g.y <= bv.x + r && g.w <= bv.y + r;
          }
          if( skip )
          {
            checkIdx++;
            continue;
          }
          inRegion = true;
          newRow   = true;
          shiftX   = 0;
          y        = g.w;
        }
        if( newRow )
        {
          if( y < g.z )
          {
            inRegion = false;
            checkIdx++;
            continue;
          }
          shiftX   = g.y != g.x ? shiftX % S : 0;
          x        = g.y - shiftX;
          shiftX++;
          rowFound = false;
          newRow   = false;
        }
        if( x < g.x )
        {
          y -= S;
          newRow = true;
          continue;
        }
        const int px = x;
        x -= S;
        if( ( region == 4 || region == 5 ) && !rowFound && checkR4R5 )
        {
          if( !decodedBefore( om, c.x + px + W - 1, c.y + y + H - 1, g_ti ) )
          {
            continue;
          }
          rowFound = true;
        }
        tmpEmit( k, px, y, region );
      }
      s_tmpCount = k;
      s_tmpDone  = checkIdx >= 6 ? 1 : 0;
    }
    tsSync();
    tmpChunkCosts( c, tempType, lid, NT );
    const bool last = s_tmpDone != 0;   // (read before invocation 0 may write the next chunk's)
    if( lid == 0 )
    {
      const int rr = S >> 1;
      for( int k = 0; k < s_tmpCount; k++ )
      {
        const int p = tsGet( TS_POS + 2 * k ), px = bitfieldExtract( p, 0, 16 ), py = p >> 16;
        for( int t = 0; t < numTypes; t++ )
        {
          const int cost = tsGet( TS_COST + 3 * k + t ), list = TS_SPARSE + t * TS_LIST;
          if( cost < pDiff[t] )
          {
            const int region = tsGet( TS_POS + 2 * k + 1 );
            tlUpdate( list, tlEntry( px, py, ivec2( px - rr, py - rr ), ivec2( px + rr, py + rr ), region, cost ),
                      numSparse[t] );
            if( tlSize( list ) == numSparse[t] )
            {
              pDiff[t] = min( tlAt( list, numSparse[t] - 1, TE_COST ), pDiff[t] );
            }
          }
        }
      }
    }
    tsSync();
    if( last )
    {
      break;
    }
  }

  // an EIP CU with a vector: the sparse search's best (m_tmpSparseSearchXdisp; the number found, 0 or 1) into its
  // DimdOut and its record of vectors (pu.bv), which is all it searches for
  if( onlyBest )
  {
    if( lid == 0 )
    {
      const int   n  = tlSize( TS_SPARSE ), at = cuIndex * DIMD_OUT;
      const ivec2 bv = n > 0 ? ivec2( tlAt( TS_SPARSE, 0, TE_X ), tlAt( TS_SPARSE, 0, TE_Y ) ) : ivec2( 0 );
      CBufI32( P.dimdOut ).v[at + DIMD_OUT_TMP_NUM] = n;
      CBufI32( P.dimdOut ).v[at + DIMD_OUT_BV]      = bvPack( bv );
      bvStore( c, n, ivec2[3]( bv, bv, bv ) );
    }
    tsSync();
    return;
  }

  // the vectors cascaded from the L-shape's sparse candidates (getSparseArBvMergeCandidate: up to NUM_TMP_ARBVP_S
  // more), then the costs of every vector of the list
  if( lid == 0 )
  {
    numList = itmpSize1;
    const int total = numList + NUM_TMP_ARBVP_S;
    for( int i = 0; i < tlSize( TS_SPARSE ) && numList < total; i++ )
    {
      numList = bvCascadeStep( c, numList, total, ivec2( tlAt( TS_SPARSE, i, TE_X ), tlAt( TS_SPARSE, i, TE_Y ) ), true,
                               g_ti );
    }
    s_tmpCount = numList;
    int k = 0;   // (tmpEmit counts it on)
    for( int i = 0; i < numList; i++ )
    {
      tmpEmit( k, bvListAt( i ).x, bvListAt( i ).y, i );
    }
  }
  tsSync();
  tmpChunkCosts( c, tempType, lid, NT );   // (at most BV_LIST_MAX)

  // the windows of the neighbours' vectors (adaptive: by the cost against its threshold), the cascaded ones' fixed;
  // then the clustering (invocation 0)
  if( lid == 0 )
  {
    for( int i = 0; i < numList; i++ )
    {
      tsSet( TS_MCOST + 3 * i, tsGet( TS_COST + 3 * i ) );
      tsSet( TS_MCOST + 3 * i + 1, tsGet( TS_COST + 3 * i + 1 ) );
      tsSet( TS_MCOST + 3 * i + 2, tsGet( TS_COST + 3 * i + 2 ) );
    }
    const int numTemp   = tempType == TMP_L_SHAPE ? numTL : tempType == TMP_LEFT ? numLeft : numTop;
    const int threshold = numTemp << ( ( licMode ? TMP_LOG2MINCOST : TMP_LOG2MINCOST + 1 ) + bitDepth - 10 );
    const int arbvpS    = licMode ? 2 : 1;   // ARBVPS_RANGE (- 1 without the LIC mode)
    bool clustered[3] = bool[3]( false, false, false );
    for( int m = 0; m < numList; m++ )
    {
      const ivec2 cand = bvListAt( m );
      ivec2       winTL, winBR;
      int         regionId;
      if( m < itmpSize )
      {
        const int cost  = tsGet( TS_MCOST + 3 * m );
        const int slope = cost < threshold ? TMP_MERGE_RANGE * cost / threshold : TMP_MERGE_RANGE;
        winTL    = cand - slope;
        winBR    = cand + slope;
        regionId = TMP_MRG_REG_ID;
      }
      else
      {
        const int r = m < itmpSize1 ? TMP_ARBVP_RANGE : arbvpS;
        winTL    = cand - r;
        winBR    = cand + r;
        regionId = m < itmpSize1 ? TMP_MRG_REG_ID + 1 : TMP_MRG_REG_ID + 2;
      }
      for( int t = 0; t < numTypes; t++ )
      {
        const int list = TS_SPARSE + t * TS_LIST, diff = tsGet( TS_MCOST + 3 * m + t );
        bool overlap = false;
        for( int i = 0; i < tlSize( list ); i++ )
        {
          if( tlAt( list, i, TE_X ) == cand.x && tlAt( list, i, TE_Y ) == cand.y )
          {
            overlap = true;
            break;
          }
          ivec2 sTL = ivec2( tlAt( list, i, TE_TLX ), tlAt( list, i, TE_TLY ) );
          ivec2 sBR = ivec2( tlAt( list, i, TE_BRX ), tlAt( list, i, TE_BRY ) );
          if( !( winBR.x < sTL.x || winTL.x > sBR.x ) && !( winBR.y < sTL.y || winTL.y > sBR.y ) )
          {
            if( diff < tlAt( list, i, TE_COST ) )   // the vector replaces the sparse one, their windows merged
            {
              sTL = bigCtu ? min( sTL, winTL ) : min( sTL + 1, winTL );
              sBR = bigCtu ? max( sBR, winBR ) : max( sBR - 1, winBR );
              tlErase( list, i );
              tlUpdate( list, tlEntry( cand.x, cand.y, sTL, sBR, regionId, diff ), numSparse[t] );
              if( tlSize( list ) == numSparse[t] )
              {
                pDiff[t] = min( tlAt( list, numSparse[t] - 1, TE_COST ), pDiff[t] );
              }
            }
            else   // the sparse one's window takes the vector in (also the window the next types test with)
            {
              if( !bigCtu )
              {
                const int r = regionId == TMP_MRG_REG_ID ? 2 : 1;
                winTL = cand - r;
                winBR = cand + r;
              }
              const ivec2 aTL = bigCtu ? cand : winTL, aBR = bigCtu ? cand : winBR;
              tlSetAt( list, i, TE_TLX, min( sTL.x, aTL.x ) );
              tlSetAt( list, i, TE_TLY, min( sTL.y, aTL.y ) );
              tlSetAt( list, i, TE_BRX, max( sBR.x, aBR.x ) );
              tlSetAt( list, i, TE_BRY, max( sBR.y, aBR.y ) );
              tlSetAt( list, i, TE_REGION, regionId );
            }
            clustered[t] = true;
            overlap      = true;
            break;
          }
        }
        if( !overlap && diff < pDiff[t] )
        {
          tlUpdate( list, tlEntry( cand.x, cand.y, winTL, winBR, regionId, diff ), numSparse[t] );
          if( tlSize( list ) == numSparse[t] )
          {
            pDiff[t] = min( tlAt( list, numSparse[t] - 1, TE_COST ), pDiff[t] );
          }
          clustered[t] = true;
        }
      }
    }
    tsSet( TS_WIN, ( clustered[0] ? 1 : 0 ) | ( clustered[1] ? 2 : 0 ) | ( clustered[2] ? 4 : 0 ) );
    for( int t = 0; t < 3; t++ )   // the refinement's thresholds
    {
      tsSet( TS_WIN + 1 + t, pDiff[t] + ( tlSize( TS_SPARSE + t * TS_LIST ) < numSparse[t] ? 0 : 1 ) );
    }
  }
  tsSync();

  // the refinement: per list type its candidates' windows (a sparse one's clipped to its region but where the
  // exception allows, a vector's whole, its positions checked where the window may leave the decoded area), each
  // position once per list, costed in chunks; the lists of the top and the left template keep TL_NUM
  int  type = 0, cand = 0, wy = 0, wx = 0, curType = -1;
  bool open = false, found = false, validWindow = true;
  ivec4 win = ivec4( 0 );
  int   candRegion = 0;
  int   rDiff[3] = int[3]( 0, 0, 0 );
  const int clusteredBits = tsGet( TS_WIN );
  for( ;; )
  {
    if( lid == 0 )
    {
      int k = 0;
      while( k < TMP_CHUNK && type < numTypes )
      {
        const int list = TS_SPARSE + type * TS_LIST;
        if( !open )
        {
          if( cand >= tlSize( list ) )
          {
            type++;
            cand = 0;
            continue;
          }
          candRegion = tlAt( list, cand, TE_REGION );
          const int cx = tlAt( list, cand, TE_X ), cy = tlAt( list, cand, TE_Y );
          if( candRegion > 5 )
          {
            win = ivec4( tlAt( list, cand, TE_TLX ), tlAt( list, cand, TE_BRX ), tlAt( list, cand, TE_TLY ),
                         tlAt( list, cand, TE_BRY ) );
            validWindow = bvValid( c.x, c.y, W, H, win.xz, g_ti ) && bvValid( c.x, c.y, W, H, win.yw, g_ti );
          }
          else
          {
            const ivec4 rg = tmpRegion( head, candRegion );
            win = tmpRefineWindow( rg, cx, cy, candRegion, S, c.x + rg.x >= TMP_TEMPLATE_SIZE + ( S >> 1 ),
                                   c.y + rg.z >= TMP_TEMPLATE_SIZE + ( S >> 1 ), !licMode );
          }
          open  = true;
          wy    = win.w;
          wx    = win.y;
          found = false;
        }
        if( wy < win.z )
        {
          open = false;
          cand++;
          continue;
        }
        if( wx < win.x )
        {
          wy--;
          wx    = win.y;
          found = false;
          continue;
        }
        const int px = wx, py = wy;
        wx--;
        const bool checked = candRegion == 4 || candRegion == 5
                             || ( !licMode && ( ( candRegion == 3 && px > 0 ) || ( candRegion == 1 && py > 0 ) ) );
        if( checked && !found )
        {
          if( !decodedBefore( om, c.x + px + W - 1, c.y + py + H - 1, g_ti ) )
          {
            continue;
          }
          found = true;
        }
        else if( candRegion > 5 && !validWindow && !bvValid( c.x, c.y, W, H, ivec2( px, py ), g_ti ) )
        {
          continue;
        }
        tmpEmit( k, px, py, type | cand << 2 );
      }
      s_tmpCount = k;
      s_tmpDone  = type >= numTypes ? 1 : 0;
    }
    tsSync();
    tmpChunkCosts( c, tempType, lid, NT );
    const bool last = s_tmpDone != 0;
    if( lid == 0 )
    {
      for( int k = 0; k < s_tmpCount; k++ )
      {
        const int p = tsGet( TS_POS + 2 * k ), px = bitfieldExtract( p, 0, 16 ), py = p >> 16;
        const int tag = tsGet( TS_POS + 2 * k + 1 ), t = tag & 3, ci = tag >> 2;
        const int sparse = TS_SPARSE + t * TS_LIST, refine = TS_REFINE + t * TS_LIST;
        if( t != curType )   // a type's thresholds: its own, the others' 0
        {
          curType = t;
          for( int i = 0; i < 3; i++ )
          {
            rDiff[i] = i == t ? tsGet( TS_WIN + 1 + i ) : 0;
          }
        }
        if( ( ( clusteredBits >> t ) & 1 ) != 0 && tlHas( refine, px, py ) )
        {
          continue;
        }
        const bool atCand = px == tlAt( sparse, ci, TE_X ) && py == tlAt( sparse, ci, TE_Y );
        const int  cost   = atCand ? tlAt( sparse, ci, TE_COST ) : tsGet( TS_COST + 3 * k + t );
        if( cost < rDiff[t] )
        {
          const ivec2 tl = ivec2( tlAt( sparse, ci, TE_TLX ), tlAt( sparse, ci, TE_TLY ) );
          const ivec2 br = ivec2( tlAt( sparse, ci, TE_BRX ), tlAt( sparse, ci, TE_BRY ) );
          tlUpdate( refine, tlEntry( px, py, tl, br, tlAt( sparse, ci, TE_REGION ), cost ), numRefine[t] );
          if( tlSize( refine ) == numRefine[t] )
          {
            rDiff[t] = min( tlAt( refine, numRefine[t] - 1, TE_COST ), rDiff[t] );
          }
        }
      }
    }
    tsSync();
    if( last )
    {
      break;
    }
  }

  // the list: the L-shape's (or the only one's), with the top and the left template's merged in from INIT_TL_POS on
  // (each vector once; where the list is shorter, appended); its vectors cyclically repeated to MTMP_NUM
  if( lid == 0 )
  {
    const int main = TS_REFINE;
    if( numTypes == 3 && tlSize( main ) >= INIT_TL_POS )
    {
      const int top = TS_REFINE + TS_LIST, left = TS_REFINE + 2 * TS_LIST;
      int cnt = 0;
      for( int t = 2; t > 0; t-- )
      {
        const int list = TS_REFINE + t * TS_LIST;
        for( int i = 0; i < tlSize( list ); i++ )
        {
          const int cx = tlAt( list, i, TE_X ), cy = tlAt( list, i, TE_Y );
          if( tlHas( main, cx, cy ) )
          {
            continue;
          }
          const int pos = MTMP_NUM - tlSize( top ) - tlSize( left ) + cnt++, size = tlSize( main );
          if( pos < size )
          {
            for( int j = size - 1; j > pos; j-- )
            {
              for( int f = 0; f < TE; f++ )
              {
                tlSetAt( main, j, f, tlAt( main, j - 1, f ) );
              }
            }
          }
          else
          {
            tsSet( main, size + 1 );
          }
          const int at = pos < size ? pos : size;
          for( int f = 0; f < TE; f++ )
          {
            tlSetAt( main, at, f, tlAt( list, i, f ) );
          }
        }
      }
    }
    // (with none found the CPU leaves tmpXYdisp as it was: no decodable stream has such a CU)
    CBufI32   o  = CBufI32( P.dimdOut );
    const int n  = tlSize( main ), at = cuIndex * DIMD_OUT;
    o.v[at + DIMD_OUT_TMP_NUM] = n;
    for( int i = 0; i < MTMP_NUM && n > 0; i++ )
    {
      o.v[at + DIMD_OUT_TMP_BV + i] = bvPack( ivec2( tlAt( main, i % n, TE_X ), tlAt( main, i % n, TE_Y ) ) );
    }
    // its record of vectors (getNeighBv): the one its block copies (pu.bv), then the list's first where that is
    // another entry
    const int tmpIdx = data.v[head + TMP_HEAD_IDX];
    if( n > 0 )
    {
      const ivec2 bv[3] = ivec2[3]( bvUnpack( o.v[at + DIMD_OUT_TMP_BV + tmpIdx] ),
                                    bvUnpack( o.v[at + DIMD_OUT_TMP_BV] ), ivec2( 0 ) );
      bvStore( c, tmpIdx > 0 ? 2 : 1, bv );
      o.v[at + DIMD_OUT_BV] = bvPack( bv[0] );   // (the one its block copies)
    }
  }
}

// ----- the block

// IntraTMP block t's prediction into s_spPred (all invocations): the reconstruction at the vector of its CU's list that
// tmpIdx selects (generateTMPrediction), which its CU's node has put in its DimdOut's DIMD_OUT_BV
void tmpPredict( Tu t, int lid, int NT )
{
  const ivec2 bv = bvUnpack( CBufI32( P.dimdOut ).v[t.cu * DIMD_OUT + DIMD_OUT_BV] );
  for( int i = lid; i < t.w * t.h; i += NT )
  {
    s_spPred[i] = int16_t( recoAt( 0, t.x + bv.x + i % t.w, t.y + bv.y + i / t.w ) );
  }
  SYNC();
}
