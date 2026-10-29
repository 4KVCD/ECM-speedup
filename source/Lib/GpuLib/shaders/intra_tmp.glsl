// IntraTMP's search (IntraPrediction::searchCandidateFromOnePicIntra, JVET-V0130 and the proposals after it): the node
// of an IntraTMP CU finds the block vectors of its list (MTMP_NUM) as the decoder does - a sparse search of the regions
// around the CU, the vectors of its neighbours (getItmpMergeCandidate, sortInOut, getArBvMergeCandidate) and those
// cascaded from the sparse search's best (getSparseArBvMergeCandidate) with windows clustered into the sparse lists, a
// refinement of each candidate's window, the lists of the top and the left template merged into the L-shaped one's -
// into its DimdOut (DIMD_OUT_TMP_*), whose entry tmpIdx its block copies. The template costs (calcTemplateDiff) are
// computed in parallel, in chunks of positions in the CPU's order, and the CPU's lists replayed from them by invocation
// 0: every test of a cost is "less than a threshold", which the CPU's early ends cannot change; for tmpFracIdx 1 the
// sub-sample search (tmpFracSearch). With LIC (tmpLicFlag, JVET-AE0078) its costs are mean-removed SADs
// (each template part's mean difference taken out) and the sparse search's positions move half a step towards its
// region's middle (iXOffsetMetric: "transferred" back at the region's edge), whose flags its refinement window
// keeps; its block's LIC model then comes from the one-sample templates at its vector (tmpLicModel). A fusion CU's
// block blends the vectors of its index's group (tmpFusionSetup, tmpFusionSample); an FLM CU's filters the area at its
// vector with a model of 5 taps fitted on the template (tmpFlmSetup). An EIP CU
// with a vector runs the sparse search alone, for its best (TMP_SEARCH_BEST); a BVG-DIMD CU the sparse search for its
// best, its neighbours' vectors against it and the refinement of the best one's window (TMP_SEARCH_BVG; intra_bvg.glsl
// takes the refined list on). Included by intra_rec.comp after intra_bv.glsl.

const int TMP_COST_SHIFT    = 3;    // TMP_TEMPLATE_COST_SHIFT: the template row and column next to the block weigh 8
const int MTMP_NUM          = 19;
const int MTMP_NUM_SPARSE   = 30;
// the sparse lists with LIC without the SPS's LIC mode (TL_NUM_SPARSE_FOR_LIC: TL_NUM)
const int MTMP_NUM_SPARSE_FOR_LIC = 15;
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
const int TMP_HEAD_SEARCH     = 7;   // what the search is for (TMP_SEARCH_*)
const int TMP_HEAD_REGIONS    = 8;   // per regionId mvXMin, mvXMax, mvYMin, mvYMax (clipMvIntraConstraint), 6 x 4
const int TMP_HEAD_FRAC       = TMP_HEAD_REGIONS + 24;   // tmpFracIdx: its block's vector is the sub-sample search's
const int TMP_HEAD_OBMC       = TMP_HEAD_FRAC + 1;       // its OBMC record (intra_obmc.glsl's OBMC_*), else -1
const int TMP_HEAD_LIC        = TMP_HEAD_OBMC + 1;       // with LIC (tmpLicFlag) its ibcLicIdx + 1, else 0
const int TMP_HEAD_LIC_EXT    = TMP_HEAD_LIC + 1;        // the SPS's ItmpLicExtension (LIC on the L-shape always)
const int TMP_HEAD_LIC_SLOT   = TMP_HEAD_LIC_EXT + 1;    // where SGPM's lists read its LIC model (SGPM_LIC), else -1
const int TMP_HEAD_FUSION     = TMP_HEAD_LIC_SLOT + 1;   // tmpFusionFlag
const int TMP_HEAD_FLM        = TMP_HEAD_FUSION + 1;     // tmpFlmFlag
const int TMP_HEAD            = TMP_HEAD_FLM + 1;
const int TMP_SEARCH_FULL = 0;   // an IntraTMP CU's list
const int TMP_SEARCH_BEST = 1;   // an EIP CU with a vector: the sparse search's best alone (mtmpNumSparse 1)
const int TMP_SEARCH_BVG  = 2;   // a BVG-DIMD CU's (JVET-AL0108): MAX_BVG_ROUGH_NUM 1, its own refinement

// BVG-DIMD's search: the window of a neighbour's vector (5 around it), the refined list's length and its cut-off at
// 1.2 times its cheapest (BVG_DIMD_BV_LIST_THRD; the CPU's double products are exact as 6 / 5 in integers)
const int BVG_REFINE_RANGE = 5;
const int BVG_NUM_REFINE   = 5;   // BVG_DIMD_BV_LIST_SIZE
int bvgThreshold( int cost ) { return int( int64_t( cost ) * 6 / 5 ) + 1; }   // (int)( cost * 1.2 ) + 1

// the node's work area in the scratch (the workgroup's blending sums, free in a node): a chunk of positions and their
// costs, the lists (a length, then entries of TE ints), the vector list's costs, the clustering's result
const int TMP_CHUNK = 256;
const int TE        = 8;   // an entry: x, y, its window's top-left x / y and bottom-right x / y, region, cost
const int TE_X = 0, TE_Y = 1, TE_TLX = 2, TE_TLY = 3, TE_BRX = 4, TE_BRY = 5, TE_REGION = 6, TE_COST = 7;
const int TS_POS    = 0;                              // per position: x | y << 16, its tag (region / list entry)
const int TS_COST   = TS_POS + 2 * TMP_CHUNK;         // per position its 3 costs
const int TL_CAP    = MTMP_NUM_SPARSE + 1;            // a list's room (the longest list, one more)
const int TS_LIST   = 1 + TL_CAP * ( 1 + TE );        // a list: its size, its order, its entries (tlAt)
const int TS_SPARSE = TS_COST + 3 * TMP_CHUNK;        // 3 lists: the L-shape's (or the only one), the top's, the left's
const int TS_REFINE = TS_SPARSE + 3 * TS_LIST;        // 3 lists alike
const int TS_MCOST  = TS_REFINE + 3 * TS_LIST;        // per vector of the list its 3 costs
const int TS_WIN    = TS_MCOST + 3 * BV_LIST_MAX;     // which lists took vectors (bits), the refinement's 3 thresholds

shared int s_tmpCount;    // the chunk's positions
shared int s_tmpDone;     // the last chunk
shared int s_tmpThr[3];   // per list type its threshold at the chunk's start (the lists' costs only fall)

int  tsBase()              { return int( gl_WorkGroupID.x ) * BLEND_ACC; }
int  tsGet( int o )        { return WBufI32( P.scratch ).v[tsBase() + o]; }
void tsSet( int o, int v ) { WBufI32( P.scratch ).v[tsBase() + o] = v; }

void tsSync()
{
  controlBarrier( gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsBuffer | gl_StorageSemanticsShared,
                  gl_SemanticsAcquireRelease );
}

// ----- the lists (invocation 0 updates them): per template type the sparse search's and the refinement's, in the
// scratch at their handles (TS_SPARSE + t * TS_LIST, TS_REFINE + t * TS_LIST): the size; the order, TL_CAP slots
// numbers, the entries' in the list's order, then the free ones; per slot an entry of TE words. An entry stays in its
// slot while the list changes, so an insertion or an erasure moves one word per entry behind it, not TE

int  tlSize( int list )                       { return tsGet( list ); }
int  tlSlot( int list, int i )                { return tsGet( list + 1 + i ); }
int  tlEntryAt( int list, int slot )          { return list + 1 + TL_CAP + slot * TE; }
int  tlAt( int list, int i, int f )           { return tsGet( tlEntryAt( list, tlSlot( list, i ) ) + f ); }
void tlSetAt( int list, int i, int f, int v ) { tsSet( tlEntryAt( list, tlSlot( list, i ) ) + f, v ); }

// an empty list, every slot free
void tlClear( int list )
{
  tsSet( list, 0 );
  for( int k = 0; k < TL_CAP; k++ )
  {
    tsSet( list + 1 + k, k );
  }
}

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
  const int slot = tlSlot( list, last );             // (a free slot, or with the list full the dropped entry's)
  for( int i = last; i > size - shift; i-- )
  {
    tsSet( list + 1 + i, tlSlot( list, i - 1 ) );
  }
  tsSet( list + 1 + size - shift, slot );
  for( int f = 0; f < TE; f++ )
  {
    tsSet( tlEntryAt( list, slot ) + f, e[f] );
  }
  tsSet( list, last + 1 );
}

void tlErase( int list, int i )
{
  const int size = tlSize( list ), slot = tlSlot( list, i );
  for( int j = i; j < size - 1; j++ )
  {
    tsSet( list + 1 + j, tlSlot( list, j + 1 ) );
  }
  tsSet( list + size, slot );   // (free again: the order's entry size - 1)
  tsSet( list, size - 1 );
}

// the slot for an entry at position pos (the main list's merge): before the entry at pos, the entries from there on
// one further back and the list's last dropped; past the end appended
int tlPlace( int list, int pos )
{
  const int size = tlSize( list );
  if( pos >= size )
  {
    tsSet( list, size + 1 );
    return tlSlot( list, size );
  }
  const int slot = tlSlot( list, size - 1 );
  for( int j = size - 1; j > pos; j-- )
  {
    tsSet( list + 1 + j, tlSlot( list, j - 1 ) );
  }
  tsSet( list + 1 + pos, slot );
  return slot;
}

bool tlHas( int list, int x, int y )
{
  for( int i = 0; i < tlSize( list ); i++ )
  {
    const int e = tlEntryAt( list, tlSlot( list, i ) );
    if( tsGet( e + TE_X ) == x && tsGet( e + TE_Y ) == y )
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

// ----- the template costs

// the CU's own template (its rows above, with the L-shape's corner, then its columns left), which every cost reads, in
// shared memory during the node: the first stage of sign prediction's transforms (s_spStage1), free in a node - not
// s_spCoef, which holds the node's vector list (intra_bv.glsl's s_bvList) - (at most 4 x 132 + 4 x 128 samples: 1024)
int tmpCurIndex( Cu c, int x, int y )
{
  return y < 0 ? ( y + TMP_TEMPLATE_SIZE ) * ( c.w + TMP_TEMPLATE_SIZE ) + x + TMP_TEMPLATE_SIZE
               : TMP_TEMPLATE_SIZE * ( c.w + TMP_TEMPLATE_SIZE ) + y * TMP_TEMPLATE_SIZE + x + TMP_TEMPLATE_SIZE;
}

int tmpCur( Cu c, int x, int y )
{
  return int( s_spStage1[tmpCurIndex( c, x, y )] );
}

void tmpCurLoad( Cu c, int lid, int NT )
{
  const int numTop = TMP_TEMPLATE_SIZE * ( c.w + TMP_TEMPLATE_SIZE ), numLeft = TMP_TEMPLATE_SIZE * c.h;
  for( int i = lid; i < numTop + numLeft; i += NT )
  {
    const int k = i - numTop, rowW = c.w + TMP_TEMPLATE_SIZE;
    const int x = ( i < numTop ? i % rowW : k % TMP_TEMPLATE_SIZE ) - TMP_TEMPLATE_SIZE;
    const int y = i < numTop ? i / rowW - TMP_TEMPLATE_SIZE : k / TMP_TEMPLATE_SIZE;
    // (a sample outside the picture is never compared: its template type has no such part)
    s_spStage1[i] = int16_t( c.x + x >= 0 && c.y + y >= 0 ? recoAt( 0, c.x + x, c.y + y ) : 0 );
  }
  tsSync();
}

// the search's mean-removed costs (useMR: an IntraTMP CU with LIC) and its template's means (calcTargetMean): the
// top part's (the rows above the block, without the L-shape's corner) and the left part's (the columns left of it),
// each sum floored by its size's log2 (both parts are powers of two)
bool  g_tmpMr   = false;
ivec2 g_tmpMean = ivec2( 0 );

// The picture's samples come from L2 (the plane is coherent), so these loops read a template in steps of 4 x 4 samples
// whose loads are independent: one thread per position has 16 loads in flight instead of waiting for each (a CU's
// sides are multiples of 4)
ivec2 tmpMeans( Cu c, int tempType, int bx, int by )
{
  int top = 0, left = 0;
  for( int x = 0; x < c.w && tempType != TMP_LEFT; x += 4 )
  {
    [[unroll]] for( int j = 0; j < 4; j++ )
    {
      [[unroll]] for( int y = -TMP_TEMPLATE_SIZE; y < 0; y++ )
      {
        top += recoAt( 0, c.x + bx + x + j, c.y + by + y );
      }
    }
  }
  for( int y = 0; y < c.h && tempType != TMP_ABOVE; y += 4 )
  {
    [[unroll]] for( int i = 0; i < 4; i++ )
    {
      [[unroll]] for( int x = -TMP_TEMPLATE_SIZE; x < 0; x++ )
      {
        left += recoAt( 0, c.x + bx + x, c.y + by + y + i );
      }
    }
  }
  return ivec2( top >> findMSB( TMP_TEMPLATE_SIZE * c.w ), left >> findMSB( TMP_TEMPLATE_SIZE * c.h ) );
}

// calcTemplateDiff's three costs of vector (bx, by) for CU c with template type tempType (exact; the CPU's early ends
// only stop where a cost is no longer below its threshold): x the weighted one (the row and the column next to the
// block weigh 2^TMP_COST_SHIFT; the L-shape's with its corner, else the above or left template's), y the L-shape's
// top part, z its left part. With g_tmpMr each part's samples less the difference of its means (the top part's for the
// corner too)
ivec3 tmpCosts( Cu c, int tempType, int bx, int by )
{
  const ivec2 dm = g_tmpMr ? tmpMeans( c, tempType, bx, by ) - g_tmpMean : ivec2( 0 );
  int sum = 0, top = 0, left = 0;
  if( tempType != TMP_LEFT )   // the rows above (with the corner of the L-shape: the first 4 columns, x < 0)
  {
    int corner = 0, upper = 0, next = 0;   // (next: the row next to the block)
    for( int x = tempType == TMP_L_SHAPE ? -TMP_TEMPLATE_SIZE : 0; x < c.w; x += 4 )
    {
      int part = 0, nextPart = 0;
      [[unroll]] for( int j = 0; j < 4; j++ )
      {
        [[unroll]] for( int y = -TMP_TEMPLATE_SIZE; y < 0; y++ )
        {
          const int d = abs( recoAt( 0, c.x + bx + x + j, c.y + by + y ) - tmpCur( c, x + j, y ) - dm.x );
          part     += y < -1 ? d : 0;
          nextPart += y < -1 ? 0 : d;
        }
      }
      corner += x < 0 ? part + nextPart : 0;
      upper  += x < 0 ? 0 : part;
      next   += x < 0 ? 0 : nextPart;
    }
    top = upper + next;
    sum = corner + upper + ( next << TMP_COST_SHIFT );
  }
  if( tempType != TMP_ABOVE )   // the columns left (the one next to the block, x -1, weighs 2^TMP_COST_SHIFT)
  {
    int part = 0, next = 0;
    for( int y = 0; y < c.h; y += 4 )
    {
      [[unroll]] for( int i = 0; i < 4; i++ )
      {
        [[unroll]] for( int x = -TMP_TEMPLATE_SIZE; x < 0; x++ )
        {
          const int d = abs( recoAt( 0, c.x + bx + x, c.y + by + y + i ) - tmpCur( c, x, y + i ) - dm.y );
          part += x < -1 ? d : 0;
          next += x < -1 ? 0 : d;
        }
      }
    }
    left = part + next;
    sum += part + ( next << TMP_COST_SHIFT );
  }
  return ivec3( sum, top, left );
}

// The refinement's walk (invocation 0) takes a position of type t into the type's list where its cost is below the
// list's threshold, which only falls as the list fills (not with BVG-DIMD's threshold by the list's cheapest: caps 0,
// which leaves this out). A position below the threshold at the chunk's start then ends in the list only where fewer
// than the list's cap (caps[t]) rank before it: the list's entries that cost no more, which stay while it does, and
// the chunk's positions of its type below the threshold that cost less or as much and come earlier, each of which
// either enters before the position would leave or shows that the threshold fell to the position's cost. Of a
// clustered type, whose list takes each position once, a position counts only where none equal to it is in the list or
// earlier in the chunk (else the walk may skip it). The others are left out of the walk (may: whether the position's
// threshold test passed; all invocations, after the chunk's costs, ending with a barrier; one position per invocation,
// its cost as its list takes it, its type; s_spCost per position its cost, below 2^24, and its type, or ~0 for one that
// does not count)
void tmpRefineRanks( int cost, int type, int px, int py, ivec3 caps, inout bool may, int lid )
{
  const int  n = s_tmpCount, k = lid, list = TS_REFINE + type * TS_LIST;
  const bool dedupe = ( ( tsGet( TS_WIN ) >> type ) & 1 ) != 0;
  bool counts = k < n && may;
  if( counts && dedupe )
  {
    const int p = ( px & 0xFFFF ) | ( py << 16 );
    counts = !tlHas( list, px, py );
    for( int i = 0; i < k && counts; i++ )
    {
      counts = tsGet( TS_POS + 2 * i ) != p || ( tsGet( TS_POS + 2 * i + 1 ) & 3 ) != type;
    }
  }
  if( k < n )
  {
    s_spCost[k] = counts ? uint( cost ) | uint( type ) << 28 : ~0u;
  }
  tsSync();
  if( k < n && may )
  {
    int lo = 0, hi = tlSize( list );   // (the list's entries costing no more: it rises by cost)
    while( lo < hi )
    {
      const int mid = ( lo + hi ) >> 1;
      lo = tlAt( list, mid, TE_COST ) <= cost ? mid + 1 : lo;
      hi = tlAt( list, mid, TE_COST ) <= cost ? hi : mid;
    }
    int before = lo;
    for( int j = 0; j < n && before < caps[type]; j++ )
    {
      const uint v = s_spCost[j];
      const int  cj = int( v & 0x0fffffffu );
      before += j != k && v != ~0u && int( v >> 28 ) == type && ( cj < cost || ( cj == cost && j < k ) ) ? 1 : 0;
    }
    may = before < caps[type];
  }
  tsSync();   // (s_spCost read before the chunk's flags replace it)
}

// the costs of the chunk's s_tmpCount positions (all invocations, after a tsSync; positions at TS_POS), into TS_COST,
// and per position whether it may change a list (s_spCost, sign prediction's costs, free in a node), which invocation
// 0 then walks alone. The refinement's (refine): a cost of its own type (tag & 3; at its candidate's own position that
// candidate's cost) below the type's threshold at the chunk's start (s_tmpThr), as a list's threshold only falls, and
// with caps (the lists' lengths) few enough ranking before it (tmpRefineRanks). The
// sparse search's (the first numTypes lists, of caps entries): a position that is in a list after the chunk. Such a
// list keeps the first positions by cost, then by order (updateCandList inserts after equal costs; none enters past
// the threshold, which is the last entry's once the list is full), so a position below the threshold ends in it
// exactly where fewer than the list's cap rank before it: the list's entries and the chunk's earlier positions that
// cost no more, its later ones that cost less. The others leave it again within the chunk if they enter at all. (One
// position per invocation: TMP_CHUNK is intra_rec's workgroup size)
void tmpChunkCosts( Cu c, int tempType, int numTypes, ivec3 caps, bool refine, int lid, int NT )
{
  bool may = false;
  int  cost = 0, type = 0, px = 0, py = 0;   // (the refinement's: the position's cost as its list takes it, its type)
  for( int k = lid; k < s_tmpCount; k += NT )
  {
    const int p = tsGet( TS_POS + 2 * k );
    px = bitfieldExtract( p, 0, 16 );
    py = p >> 16;
    const ivec3 cs = tmpCosts( c, tempType, px, py );
    tsSet( TS_COST + 3 * k, cs.x );
    tsSet( TS_COST + 3 * k + 1, cs.y );
    tsSet( TS_COST + 3 * k + 2, cs.z );
    if( refine )
    {
      const int tag = tsGet( TS_POS + 2 * k + 1 ), t = tag & 3, ci = tag >> 2, sparse = TS_SPARSE + t * TS_LIST;
      const bool atCand = px == tlAt( sparse, ci, TE_X ) && py == tlAt( sparse, ci, TE_Y );
      type = t;
      cost = atCand ? tlAt( sparse, ci, TE_COST ) : cs[t];
      may  = cost < s_tmpThr[t];
    }
  }
  tsSync();
  if( refine && caps.x > 0 )
  {
    tmpRefineRanks( cost, type, px, py, caps, may, lid );
  }
  for( int t = 0; t < numTypes && !refine; t++ )   // (the sparse search's: the chunk's costs of type t in s_spCost)
  {
    const int k = lid, cost = k < s_tmpCount ? tsGet( TS_COST + 3 * k + t ) : 0;
    if( k < s_tmpCount )
    {
      s_spCost[k] = uint( cost );
    }
    tsSync();
    if( k < s_tmpCount && cost < s_tmpThr[t] )
    {
      const int list = TS_SPARSE + t * TS_LIST;
      int       lo = 0, hi = tlSize( list );   // (the list's entries costing no more: it rises by cost)
      while( lo < hi )
      {
        const int mid = ( lo + hi ) >> 1;
        lo = tlAt( list, mid, TE_COST ) <= cost ? mid + 1 : lo;
        hi = tlAt( list, mid, TE_COST ) <= cost ? hi : mid;
      }
      int before = lo;
      for( int j = 0; j < s_tmpCount && before < caps[t]; j++ )
      {
        const int cj = int( s_spCost[j] );
        before += cj < cost || ( cj == cost && j < k ) ? 1 : 0;
      }
      may = may || before < caps[t];
    }
    tsSync();
  }
  if( lid < s_tmpCount )
  {
    s_spCost[lid] = may ? 1u : 0u;
  }
  tsSync();
}

void tmpEmit( inout int k, int x, int y, int tag )
{
  tsSet( TS_POS + 2 * k, ( x & 0xFFFF ) | ( y << 16 ) );
  tsSet( TS_POS + 2 * k + 1, tag );
  k++;
}

// region r of the record at head: mvXMin, mvXMax, mvYMin, mvYMax
ivec4 tmpRegion( int head, int r )
{
  BufI32    data = BufI32( P.timdData );
  const int at   = head + TMP_HEAD_REGIONS + 4 * r;
  return ivec4( data.v[at], data.v[at + 1], data.v[at + 2], data.v[at + 3] );
}

// ----- LIC (JVET-AE0078: InterPrediction::LicItmp -> xLocalIlluComp with ibcLicFlag)

const int IBC_LIC_IDX_T = 1, IBC_LIC_IDX_L = 2, IBC_LIC_IDX_M = 3;   // ibcLicIdx (0: both templates, one model)
// a LIC model as SGPM's lists keep one (intra_sgpm.glsl's SGPM_LIC_*: a BvInfo's): whether it has two, the first's
// scale, shift and offset, the second's, the mean between them (the first up to it)
const int TMP_LIC = 8;

// getMSB (CommonDef.h, INTER_LIC): floorLog2 + 1, 1 for 0
int licMsb( uint v )
{
  return v == 0u ? 1 : findMSB( v ) + 1;
}

// sample idx of the reference template above (top) or left of chroma block c (in its samples) of component comp at
// vector mv, whose chroma part has a fraction (xGetIbcLicPredBlkTpl of a chroma component): with the SPS's fractional
// IBC predicted at the vector moved by one luma sample - half a chroma sample - with the bilinear filter, else copied
// at its integer part's row above / column left
int tmpLicChromaRef( Cu c, int comp, ivec2 mv, bool top, int idx, int bitDepth )
{
  if( P.ibcFrac != 0 )
  {
    const ivec2 moved = mv + ( top ? ivec2( 0, -16 ) : ivec2( -16, 0 ) );
    return ibcChromaPredict( comp, c.x, c.y, top ? c.w : 1, top ? 1 : c.h, moved, top ? idx : 0, top ? 0 : idx, g_ti,
                             bitDepth, true, false );
  }
  const ivec2 b = mv >> 5;
  return top ? recoAt( comp, c.x + b.x + idx, c.y + b.y - 1 ) : recoAt( comp, c.x + b.x - 1, c.y + b.y + idx );
}

// the LIC model of IntraTMP CU c at vector mv (1/16 samples; invocation 0; the scratch from TS_POS its work area):
// xGetSublkTemplate's one-sample templates - the row above the CU (but for IBC_LIC_IDX_L or required template 2) and
// the column left of it (but for IBC_LIC_IDX_T or required template 1) where the CU has a neighbour there and the row /
// column at the vector is valid (checkIsIBCCandidateValid of the vector moved by a sample, its integer part), at a
// sub-sample vector predicted with the bilinear filter (xGetIbcLicPredBlkTpl -> getPredIBCBlk; delta: the padding's
// region offset of the CU's vector, as tmpPredict's) -, sampled as
// xGetLICParamGeneral steps (numSteps per side, by the smaller side but for one side's own index), fitted as inter
// LIC's one model (m_licShift 5, the sums cropped, m_licMultApprox's division) or with IBC_LIC_IDX_M as MMLM's two
// split at the reference samples' mean (xLMSampleClassifiedTraining over 2^cntShift samples, the means without
// division; cclmFit). comp: the component; a chroma one (4:2:0; c in its samples) is that of a single tree's DBV copy
// at a vector with a chroma fraction (predIntraDbv's getPredIBCBlk, whose xPredInterBlk applies xLocalIlluComp of the
// component): the luma check of the vector moved by a chroma sample, the templates tmpLicChromaRef's
int[TMP_LIC] tmpLicModel( Cu c, ivec2 mv, ivec2 delta, int licIdx, int req, int bitDepth, int comp )
{
  const int   W = c.w, H = c.h, precShift = max( 0, bitDepth - 12 ), sc = comp != 0 ? 1 : 0;
  const ivec2 mvTop = mv + ivec2( 0, -16 << sc ), mvLeft = mv + ivec2( -16 << sc, 0 );
  const bool  above = c.y > 0 && licIdx != IBC_LIC_IDX_L && req != 2
                      && ibcAreaValid( c.x << sc, c.y << sc, W << sc, 1, mvTop >> 4, 0, 0, g_ti );
  const bool  left  = c.x > 0 && licIdx != IBC_LIC_IDX_T && req != 1
                      && ibcAreaValid( c.x << sc, c.y << sc, 1, H << sc, mvLeft >> 4, 0, 0, g_ti );
  const int  minDimBit = licIdx == IBC_LIC_IDX_T ? findMSB( W ) : licIdx == IBC_LIC_IDX_L ? findMSB( H )
                                                                                          : findMSB( min( W, H ) );
  const int  maxNumMinus1 = 30 - 2 * min( bitDepth, 12 ) - 1;
  int        minStepBit   = ( 1 << minDimBit ) > 8 ? 1 : 0;
  while( minDimBit > minStepBit + maxNumMinus1 )
  {
    minStepBit++;
  }
  const int numSteps = ( 1 << minDimBit ) >> minStepBit, dimShift = minDimBit - minStepBit;
  const int REF = TS_POS, REC = TS_POS + 128;   // the samples (refSamples, recSamples)
  int x = 0, y = 0, xx = 0, xy = 0, cntShift = 0, n = 0;
  tsSet( REF, 0 );   // (none: the mean of 2^0 zero samples)
  tsSet( REC, 0 );
  for( int side = 0; side < 2; side++ )
  {
    if( side == 0 ? !above : !left )
    {
      continue;
    }
    for( int k = 0; k < numSteps; k++ )
    {
      const int idx = ( k * ( side == 0 ? W : H ) ) >> dimShift;
      const int rf  = ( comp != 0 ? tmpLicChromaRef( c, comp, mv, side == 0, idx, bitDepth )
                        : ibcPredict( c.x, c.y, side == 0 ? W : 1, side == 0 ? 1 : H, side == 0 ? mvTop : mvLeft, true,
                                      side == 0 ? idx : 0, side == 0 ? 0 : idx, g_ti, bitDepth, delta, false ) )
                      >> precShift;
      const int rc  = ( side == 0 ? recoAt( comp, c.x + idx, c.y - 1 ) : recoAt( comp, c.x - 1, c.y + idx ) )
                      >> precShift;
      tsSet( REF + n, rf );
      tsSet( REC + n, rc );
      n++;
      x  += rf;
      y  += rc;
      xx += rf * rf;
      xy += rf * rc;
    }
    cntShift += side == 0 ? dimShift : cntShift != 0 ? 1 : dimShift;
  }
  int r[TMP_LIC] = int[TMP_LIC]( 0, 32, 5, 0, 0, 0, 0, 0 );
  if( licIdx == IBC_LIC_IDX_M )
  {
    const int count = 1 << cntShift;   // (avgCnt)
    int       mean = 0, meanC = 0;
    for( int i = 0; i < count; i++ )
    {
      mean  += tsGet( REF + i );
      meanC += tsGet( REC + i );
    }
    const int divSigTable[16] = int[16]( 0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0 );
    int       lx = findMSB( count );
    const int normDiff = ( count << 4 >> lx ) & 15, v = divSigTable[normDiff] | 8;
    lx += normDiff != 0 ? 1 : 0;
    mean  = ( mean * v ) >> ( lx + 3 );
    meanC = ( meanC * v ) >> ( lx + 3 );
    ivec3 m[2] = ivec3[2]( ivec3( 0, 1 << ( bitDepth - 1 ), 0 ), ivec3( 0, 1 << ( bitDepth - 1 ), 0 ) );
    if( count >= 4 )   // xLMSampleClassifiedTraining: the groups from TS_COST (reference, then current), 128 each
    {
      const int split = max( 1, mean );
      int       groupCount[2] = int[2]( 0, 0 );
      for( int i = 0; i < count; i++ )
      {
        const int g = tsGet( REF + i ) <= split ? 0 : 1;
        tsSet( TS_COST + g * 128 + groupCount[g], tsGet( REF + i ) );
        tsSet( TS_COST + 256 + g * 128 + groupCount[g], tsGet( REC + i ) );
        groupCount[g]++;
      }
      for( int g = 0; g < 2; g++ )
      {
        const int exist = groupCount[g];
        if( exist < 2 )
        {
          m[g] = ivec3( 0, meanC - mean, 0 );
          continue;
        }
        const int upper = 1 << ( findMSB( exist - 1 ) + 1 ), lower = 1 << findMSB( exist );
        if( upper != lower )   // padded to a power of two
        {
          const int numPadded = min( exist, upper - exist ), step = exist / numPadded;
          for( int i = 0; i < numPadded; i++ )
          {
            tsSet( TS_COST + g * 128 + exist + i, tsGet( TS_COST + g * 128 + i * step ) );
            tsSet( TS_COST + 256 + g * 128 + exist + i, tsGet( TS_COST + 256 + g * 128 + i * step ) );
          }
          groupCount[g] = upper;
        }
        int64_t sx = 0, sy = 0, sxx = 0, sxy = 0;
        for( int i = 0; i < groupCount[g]; i++ )
        {
          const int64_t l = tsGet( TS_COST + g * 128 + i ), cc = tsGet( TS_COST + 256 + g * 128 + i );
          sx  += l;
          sy  += cc;
          sxx += l * l;
          sxy += l * cc;
        }
        m[g] = cclmFit( sx, sy, sxx, sxy, findMSB( groupCount[g] ), bitDepth );
      }
    }
    r = int[TMP_LIC]( 1, m[0].x, m[0].z, m[0].y, m[1].x, m[1].z, m[1].y, mean );
  }
  else if( cntShift != 0 )
  {
    const int shift = 5, licRegShift = 7, licShiftDiff = 12;   // (m_licShift, m_licRegShift, m_licShiftDiff)
    const int cropShift = max( 0, bitDepth - precShift + cntShift - 15 );
    const int xzOffset  = xx >> licRegShift;
    const int sumX = x << precShift, sumY = y << precShift;
    const int sumXX = ( ( xx + xzOffset ) >> ( cropShift << 1 ) ) << cntShift;
    const int sumXY = ( ( xy + xzOffset ) >> ( cropShift << 1 ) ) << cntShift;
    const int sumXsumX = ( x >> cropShift ) * ( x >> cropShift ), sumXsumY = ( x >> cropShift ) * ( y >> cropShift );
    int       a1 = sumXY - sumXsumY, a2 = sumXX - sumXsumX;
    int       scaleShiftA2 = licMsb( uint( abs( a2 ) ) ) - 6, scaleShiftA1 = scaleShiftA2 - licShiftDiff;
    scaleShiftA2 = max( 0, scaleShiftA2 );
    scaleShiftA1 = max( 0, scaleShiftA1 );
    const int scaleShiftA = scaleShiftA2 + 15 - shift - scaleShiftA1;
    a1 >>= scaleShiftA1;
    a2 = clamp( a2 >> scaleShiftA2, 0, 63 );
    const int approx = a2 == 0 ? 0 : ( ( 1 << 15 ) + ( a2 >> 1 ) ) / a2;   // (m_licMultApprox)
    const int scale  = clamp( int( ( int64_t( a1 ) * int64_t( approx ) ) >> scaleShiftA ), 0, 1 << ( shift + 2 ) );
    const int maxOffset = ( 1 << ( bitDepth - 1 ) ) - 1;
    const int offset = ( sumY - ( ( scale * sumX ) >> shift ) + ( ( 1 << cntShift ) >> 1 ) ) >> cntShift;
    r = int[TMP_LIC]( 0, scale, shift, clamp( offset, -1 - maxOffset, maxOffset ), 0, 0, 0, 0 );
  }
  return r;
}

// sample v through LIC model m (linearTransform; with two models linearTransforms, the second above the mean),
// clipped
int tmpLicSample( int v, int m[TMP_LIC], int maxPel )
{
  const int k = m[0] != 0 && v > m[7] ? 4 : 1, scaled = m[k] * v, shift = m[k + 1];
  return clamp( ( shift >= 0 ? scaled >> shift : scaled << -shift ) + m[k + 2], 0, maxPel );
}

// IntraTMP block t's LIC (all invocations; after the DIMD of its prediction - deferredPred's -, as DecCu applies it
// after generateTMPrediction): its CU's model from its DimdOut through every sample
void tmpLicApply( Tu t, int lid, int NT )
{
  CBufI32   o  = CBufI32( P.dimdOut );
  const int at = t.cu * DIMD_OUT + DIMD_OUT_TMP_LIC, maxPel = ( 1 << t.bitDepth ) - 1;
  int       m[TMP_LIC];
  for( int k = 0; k < TMP_LIC; k++ )
  {
    m[k] = o.v[at + k];
  }
  for( int i = lid; i < t.w * t.h; i += NT )
  {
    const int x = i % t.w, y = i / t.w;
    deferredSet( t.w, x, y, tmpLicSample( deferredPred( t.w, t.h, x, y ), m, maxPel ) );
  }
}

// ----- the sub-sample search (JVET-AM0229's searchFracCandidate, for tmpFracIdx 1)

const int TMP_PAD        = 2;                     // TMP_SUBPEL_PAD_NUM
const int TS_FRAC_REQ    = TS_WIN + 4;            // the block vector's required template (m_mtmpRequiredTemplate)
const int TS_FBUF        = 4096;                  // the padded template region at that vector (row-major)
const int TS_FCOST       = TS_FBUF + 72 * 72;     // per candidate its cost, -1 where the search skips it
const int TMP_FRAC_CANDS = 4 * 4 * 8;             // fracPrecX 1..4, fracPrecY 1..4, the 8 directions (TmpSubPelDirType)

// defineFilterTmpSubpel0 / 1's 4-tap cubic filters (getExtIntraCubicFilter) per fracPrec 1..4; the distances in 1/16
// samples generateTMPrediction moves the vector by per fracPrec
const ivec4 TMP_SUBPEL0[4] = ivec4[4]( ivec4( -16, 144, 144, -16 ), ivec4( -10, 70, 210, -14 ),
                                       ivec4( -5, 34, 236, -9 ), ivec4( -14, 107, 179, -16 ) );
const ivec4 TMP_SUBPEL1[4] = ivec4[4]( ivec4( -16, 144, 144, -16 ), ivec4( -14, 210, 70, -10 ),
                                       ivec4( -9, 236, 34, -5 ), ivec4( -16, 179, 107, -14 ) );
const int   TMP_FRAC_DIST[4] = int[4]( 8, 4, 2, 6 );

// the padded region's sample (bx, by) (relative to the template's top-left at the vector; rows of `stride`), in shared
// memory: sign prediction's low and high halves and templates (s_spLow, s_spHigh, s_spTempl; 2560 samples, the region
// of a 128 x 8 CU 2176), free in a node; every candidate's cost reads it 4 to 16 times per sample. A larger region (a
// CU of 64 x 32 or more: ( W + 8 ) x ( H + 8 )) is in the node's scratch from TS_FBUF (g_tfScratch)
bool g_tfScratch = false;

int tfGet( int stride, int bx, int by )
{
  const int i = ( by + TMP_PAD ) * stride + bx + TMP_PAD;
  if( g_tfScratch )
  {
    return tsGet( TS_FBUF + i );
  }
  return int( i < 1024 ? s_spLow[i] : i < 2048 ? s_spHigh[i - 1024] : s_spTempl[i - 2048] );
}

void tfSet( int stride, int bx, int by, int v )
{
  const int i = ( by + TMP_PAD ) * stride + bx + TMP_PAD;
  if( g_tfScratch )
  {
    tsSet( TS_FBUF + i, v );
  }
  else if( i < 1024 )
  {
    s_spLow[i] = int16_t( v );
  }
  else if( i < 2048 )
  {
    s_spHigh[i - 1024] = int16_t( v );
  }
  else
  {
    s_spTempl[i - 2048] = int16_t( v );
  }
}

// whether searchFracCandidate's loop takes (fracPrecX fx, fracPrecY fy, direction dir): 3 / 4 in both only the
// diagonal directions; 1 / 2 in both the straight ones only where equal, without some combinations of 1; others none
bool tmpFracAllowed( int fx, int fy, int dir )
{
  if( fx >= 3 && fy >= 3 )
  {
    return dir >= 4;
  }
  if( fx > 2 || fy > 2 )
  {
    return false;
  }
  const bool skip = ( fx == 1 && fy == 1 && ( dir == 1 || dir == 3 ) ) || ( fy == 1 && dir == 6 )
                    || ( ( fx == 1 || fy == 1 ) && dir == 7 ) || ( fx == 1 && dir == 5 );
  return !skip && ( fx == fy || dir >= 4 );
}

// sample (bx, by) of the template predicted at candidate (fx, fy, dir) (xTmpFracSearchIF): the 4-tap filter across
// (left / right: the taps from 2 / 1 samples before), down (above / below), or across then down for a diagonal
// direction, each pass rounded to the sample and clipped (m_filterHor / m_filterVer with isFirst and isLast)
int tmpFracSample( int stride, int fx, int fy, int dir, int bx, int by, int maxPel )
{
  if( dir < 4 )   // (left, right, above, below: fracPrecX's filter)
  {
    const bool  first = dir == 0 || dir == 2, hor = dir < 2;
    const ivec4 p     = first ? TMP_SUBPEL0[fx - 1] : TMP_SUBPEL1[fx - 1];
    int         sum   = 0;
    for( int k = 0; k < 4; k++ )
    {
      const int o = ( first ? -2 : -1 ) + k;
      sum += p[k] * tfGet( stride, hor ? bx + o : bx, hor ? by : by + o );
    }
    return clamp( ( sum + 128 ) >> 8, 0, maxPel );
  }
  const bool  left = dir == 4 || dir == 6, above = dir == 4 || dir == 5;
  const ivec4 ph   = left ? TMP_SUBPEL0[fx - 1] : TMP_SUBPEL1[fx - 1];
  const ivec4 pv   = above ? TMP_SUBPEL0[fy - 1] : TMP_SUBPEL1[fy - 1];
  int         sum  = 0;
  for( int k = 0; k < 4; k++ )
  {
    int row = 0;
    for( int m = 0; m < 4; m++ )
    {
      row += ph[m] * tfGet( stride, bx + ( left ? -2 : -1 ) + m, by + ( above ? -2 : -1 ) + k );
    }
    sum += pv[k] * clamp( ( row + 128 ) >> 8, 0, maxPel );
  }
  return clamp( ( sum + 128 ) >> 8, 0, maxPel );
}

// calcTemplateDiff's cost of candidate n that required template req takes (tmpCosts' weighted one, the L-shape's top
// part or its left part) of the template predicted there against CU c's
int tmpFracCost( Cu c, int tempType, int req, int stride, int n, int maxPel )
{
  const int fx = 1 + n / 32, fy = 1 + ( n / 8 ) % 4, dir = n % 8;
  const int extW = tempType != TMP_ABOVE ? TMP_TEMPLATE_SIZE : 0, extH = tempType != TMP_LEFT ? TMP_TEMPLATE_SIZE : 0;
  int       sum = 0, top = 0, left = 0;
  if( tempType != TMP_LEFT && req != 2 )   // the rows above (with the corner for the L-shape's whole)
  {
    const int x0 = tempType == TMP_L_SHAPE && req == 0 ? -TMP_TEMPLATE_SIZE : 0;
    for( int y = -TMP_TEMPLATE_SIZE; y < 0; y++ )
    {
      int corner = 0, row = 0;
      for( int x = x0; x < c.w; x++ )
      {
        const int p = tmpFracSample( stride, fx, fy, dir, x + extW, y + extH, maxPel );
        const int d = abs( p - tmpCur( c, x, y ) );
        corner += x < 0 ? d : 0;
        row    += x < 0 ? 0 : d;
      }
      top += row;
      sum += corner + ( y == -1 ? row << TMP_COST_SHIFT : row );
    }
  }
  if( tempType != TMP_ABOVE && req != 1 )   // the columns left
  {
    for( int y = 0; y < c.h; y++ )
    {
      int row = 0, next = 0;
      for( int x = -TMP_TEMPLATE_SIZE; x < 0; x++ )
      {
        const int p = tmpFracSample( stride, fx, fy, dir, x + extW, y + extH, maxPel );
        next = abs( p - tmpCur( c, x, y ) );   // (x -1 last)
        row += next;
      }
      left += row;
      sum += row + next * ( ( 1 << TMP_COST_SHIFT ) - 1 );
    }
  }
  return req == 1 ? top : req == 2 ? left : sum;
}

// the sub-sample search of IntraTMP CU c (all invocations; after its list, with its block's vector in its DimdOut and
// its required template at TS_FRAC_REQ): xPadForFracSearchInterpolation's region around the template at the vector
// (the copy, then its padding where units are not decoded, step by step as the CPU pads it), each candidate's cost (one
// invocation each), the first cheapest (updateCandList keeps the first of equal costs; tmpFracIdx 1 takes the list's
// first) as the block's vector moved by its distances (DIMD_OUT_TMP_MV), which SGPM's lists take too
void tmpFracSearch( Cu c, int cuIndex, int tempType, int bitDepth, int lid, int NT )
{
  OwnerMap    om   = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  CBufI32     o    = CBufI32( P.dimdOut );
  const int   at   = cuIndex * DIMD_OUT, T = TMP_TEMPLATE_SIZE, D = TMP_PAD;
  const int   extW = tempType != TMP_ABOVE ? T : 0, extH = tempType != TMP_LEFT ? T : 0;
  const int   eW = c.w + extW, eH = c.h + extH, stride = eW + 2 * D, maxPel = ( 1 << bitDepth ) - 1;
  g_tfScratch = stride * ( eH + 2 * D ) > 2560;
  const ivec2 bv   = bvUnpack( o.v[at + DIMD_OUT_BV] );
  const ivec2 pos  = ivec2( c.x + bv.x - extW, c.y + bv.y - extH );
  const int   req  = tsGet( TS_FRAC_REQ );
  // the copy: the above template's rows and the left one's columns, D more on each side (clamped into the picture,
  // which the GPU's planes do not pad: the padding below replaces every sample outside it, as none is decoded)
  for( int i = lid; i < stride * ( eH + 2 * D ); i += NT )
  {
    const int bx = i % stride - D, by = i / stride - D;
    if( ( tempType != TMP_LEFT && by < T + D ) || ( tempType != TMP_ABOVE && bx < T + D ) )
    {
      const int x = clamp( pos.x + bx, 0, P.width[0] - 1 ), y = clamp( pos.y + by, 0, P.height[0] - 1 );
      tfSet( stride, bx, by, recoAt( 0, x, y ) );
    }
  }
  tsSync();
  if( lid == 0 )
  {
    // the rows above where their first column is not decoded: the row below
    const int padW = tempType != TMP_LEFT ? eW : T + D;
    for( int j = -1; j >= -D; j-- )
    {
      for( int i = 0; i < padW && !decodedBefore( om, pos.x, pos.y + j, g_ti ); i++ )
      {
        tfSet( stride, i, j, tfGet( stride, i, j + 1 ) );
      }
    }
    // the rows below the left template from their first column not decoded: the row above
    for( int j = eH; j < eH + D && tempType != TMP_ABOVE; j++ )
    {
      int i = 0;
      for( ; i < T + D && decodedBefore( om, pos.x + i, pos.y + j, g_ti ); i++ ) {}
      for( int k = i; k < T + D; k++ )
      {
        tfSet( stride, k, j, tfGet( stride, k, j - 1 ) );
      }
    }
    // the columns left where row 0's is not decoded: from there on the column right, in the template's rows
    const int padH = tempType == TMP_ABOVE ? T + D : eH;
    int       i0   = -1;
    for( ; i0 >= -D && decodedBefore( om, pos.x + i0, pos.y, g_ti ); i0-- ) {}
    for( int j = 0; j < padH && i0 >= -D; j++ )
    {
      for( int k = i0; k >= -D; k-- )
      {
        tfSet( stride, k, j, tfGet( stride, k + 1, j ) );
      }
    }
    // the rows above and below the template left of it, from their first column not decoded: the column right
    for( int j = -D; j < ( tempType == TMP_ABOVE ? 0 : eH + D ); j++ )
    {
      if( j >= 0 && j < eH )
      {
        continue;
      }
      int i = -1;
      for( ; i >= -D && decodedBefore( om, pos.x + i, pos.y + j, g_ti ); i-- ) {}
      for( int k = i; k >= -D; k-- )
      {
        tfSet( stride, k, j, tfGet( stride, k + 1, j ) );
      }
    }
    // the columns right of the above template from the first not decoded: the column left
    for( int j = -D; j < T + D && tempType != TMP_LEFT; j++ )
    {
      int i = eW;
      for( ; i < eW + D && decodedBefore( om, pos.x + i, pos.y + j, g_ti ); i++ ) {}
      for( int k = i; k < eW + D; k++ )
      {
        tfSet( stride, k, j, tfGet( stride, k - 1, j ) );
      }
    }
  }
  tsSync();
  // with LIC its model of the vector (the node's, in its DimdOut) through the padded region's parts the search reads:
  // an L-shape's rows above (from the left template's columns on for the top template alone) and below them its left
  // columns (the left template alone: all from the rows above's last D), else all
  const int licIdc = BufI32( P.timdData ).v[c.tmp + TMP_HEAD_LIC];
  if( licIdc != 0 )
  {
    int m[TMP_LIC];
    for( int k = 0; k < TMP_LIC; k++ )
    {
      m[k] = o.v[at + DIMD_OUT_TMP_LIC + k];
    }
    for( int i = lid; i < stride * ( eH + 2 * D ); i += NT )
    {
      const int bx = i % stride - D, by = i / stride - D;
      bool      part = true;
      if( tempType == TMP_L_SHAPE )
      {
        part = req == 0 ? by < T + D || bx < T + D
           : req == 1 ? by < T + D && bx >= T - D
                      : by >= T - D && bx < T + D;
      }
      if( part && ( ( tempType != TMP_LEFT && by < T + D ) || ( tempType != TMP_ABOVE && bx < T + D ) ) )
      {
        tfSet( stride, bx, by, tmpLicSample( tfGet( stride, bx, by ), m, maxPel ) );
      }
    }
    tsSync();
  }
  for( int n = lid; n < TMP_FRAC_CANDS; n += NT )
  {
    const bool allowed = tmpFracAllowed( 1 + n / 32, 1 + ( n / 8 ) % 4, n % 8 );
    tsSet( TS_FCOST + n, allowed ? tmpFracCost( c, tempType, req, stride, n, maxPel ) : -1 );
  }
  tsSync();
  if( lid == 0 )
  {
    int best = -1;
    for( int n = 0; n < TMP_FRAC_CANDS; n++ )
    {
      const int cost = tsGet( TS_FCOST + n );
      best = cost >= 0 && ( best < 0 || cost < tsGet( TS_FCOST + best ) ) ? n : best;
    }
    const int   fx = 1 + best / 32, fy = 1 + ( best / 8 ) % 4, dir = best % 8;
    const int   dx = TMP_FRAC_DIST[fx - 1], dy = TMP_FRAC_DIST[fy - 1];
    const bool  l  = dir == 0 || dir == 4 || dir == 6, r = dir == 1 || dir == 5 || dir == 7;
    const bool  a  = dir == 2 || dir == 4 || dir == 5, b = dir == 3 || dir == 6 || dir == 7;
    const ivec2 mv = ( bv << 4 ) + ivec2( l ? -dx : r ? dx : 0, a ? -dy : b ? dy : 0 );
    o.v[at + DIMD_OUT_TMP_MV]     = mv.x;
    o.v[at + DIMD_OUT_TMP_MV + 1] = mv.y;
    // (SGPM's lists take mv[0], then the list's first x 16 where the CU has an index: getNeighBv)
    const int tmpIdx = BufI32( P.timdData ).v[c.tmp + TMP_HEAD_IDX];
    bvStoreSgpm( c, tmpIdx > 0 ? 2 : 1, ivec2[2]( mv, bvUnpack( o.v[at + DIMD_OUT_TMP_BV] ) << 4 ) );
    // with LIC the model again, at this vector (DecCu's LicItmp after generateTMPrediction with tmpIsSubPel)
    if( licIdc != 0 )
    {
      BufI32      data   = BufI32( P.timdData );
      const ivec2 delta  = ivec2( ( mv.x >> 4 ) < bv.x ? 1 : 0, ( mv.y >> 4 ) < bv.y ? 1 : 0 );
      const int   reqLic = data.v[c.tmp + TMP_HEAD_LIC_EXT] != 0 ? 0 : req;
      const int   model[TMP_LIC] = tmpLicModel( c, mv, delta, licIdc - 1, reqLic, bitDepth, 0 );
      const int   slot = data.v[c.tmp + TMP_HEAD_LIC_SLOT];
      for( int k = 0; k < TMP_LIC; k++ )
      {
        o.v[at + DIMD_OUT_TMP_LIC + k] = model[k];
        if( slot >= 0 )
        {
          data.v[slot + k] = model[k];
        }
      }
    }
  }
}

// PU::sortInOut (more than MAX_TO_KEEP_MERGE vectors): those outside every region first, then those inside, 24 and 12
// of them (more of one where the other has fewer); returns the length (itmpSize)
int tmpSortInOut( int head, int n )
{
  if( n <= MAX_TO_KEEP_MERGE )
  {
    return n;
  }
  bool  outside[BV_LIST_MAX];
  int   numOut = 0;
  ivec4 g[6];   // (the regions, loaded once)
  for( int r = 0; r < 6; r++ )
  {
    g[r] = tmpRegion( head, r );
  }
  for( int i = 0; i < n; i++ )
  {
    const ivec2 bv     = bvListAt( i );
    bool        inside = false;
    for( int r = 0; r < 6; r++ )
    {
      inside = inside || ( g[r].y >= g[r].x && g[r].w >= g[r].z && bv.x >= g[r].x && bv.x <= g[r].y && bv.y >= g[r].z
                           && bv.y <= g[r].w );
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

// clipMvIntraConstraintRefine for a candidate at (px, py) of region r (bounds g): its window around it of S / 2 (one
// less to the right / below for even S) - where transferred (tl, tt: LIC's sparse search) from it to S - 1 beyond -,
// within the region but where the exception allows (region 3 / 5 left, 4 up past its edge where the picture lets the
// template fit; 3 right, 1 down)
ivec4 tmpRefineWindow( ivec4 g, int px, int py, int r, int S, bool extLeft, bool extTop, bool exception, bool tl,
                       bool tt )
{
  const int rr = S >> 1, odd = S % 2 != 0 ? 0 : 1;
  const int x0 = tl ? px : px - rr, x1 = tl ? px + S - 1 : px + rr - odd;
  const int y0 = tt ? py : py - rr, y1 = tt ? py + S - 1 : py + rr - odd;
  const int xMin = exception && ( r == 3 || r == 5 ) && extLeft ? x0 : max( g.x, x0 );
  const int xMax = exception && r == 3 ? x1 : min( g.y, x1 );
  const int yMin = exception && r == 4 && extTop ? y0 : max( g.z, y0 );
  const int yMax = exception && r == 1 ? y1 : min( g.w, y1 );
  return ivec4( xMin, xMax, yMin, yMax );
}

// ----- fusion (xTMPBuildFusionCandidate, generateTMPrediction's fusion, xTMPFusionCalcModels / xTMPFusionApplyModel)

const int TMP_FUSION_NUM = 5, TMP_GROUP_IDX = 4, TMP_BV_FUS_CAND_MRG = 5, MAX_NON_LOC_DEP_NUM = 3;

// a fusion index's group of the list: its first entry and its number (getTmpFusionIdx, getTmpFusionNumber)
ivec2 tmpFusionGroup( int tmpIdx )
{
  const int first = ( tmpIdx % TMP_GROUP_IDX ) * TMP_FUSION_NUM;
  return ivec2( first, min( TMP_FUSION_NUM, MTMP_NUM - first ) );
}

// fusion candidate j's sample at (x, y) relative to the CU c of DimdOut at (its group from `first`): the sample at the
// candidate's vector, with LIC through the candidate's model (getMemLicParams( 0, j ): the fusion CU's index 0, one
// model)
int tmpFusionRef( Cu c, int at, int first, int j, int x, int y, bool lic, int maxPel )
{
  CBufI32     o  = CBufI32( P.dimdOut );
  const ivec2 bv = bvUnpack( o.v[at + DIMD_OUT_TMP_BV + first + j] );
  const int   v  = recoAt( 0, c.x + bv.x + x, c.y + bv.y + y );
  if( !lic )
  {
    return v;
  }
  const int k = at + DIMD_OUT_TMP_FUS_LIC + 3 * j, scaled = o.v[k] * v, shift = o.v[k + 1];
  return clamp( ( shift >= 0 ? scaled >> shift : scaled << -shift ) + o.v[k + 2], 0, maxPel );
}

// the filter model's reference point relative to the CU (refPosX / refPosY of the template: its corner sample next to
// the block; the above template's column 0, the left one's row 0)
ivec2 tmpFusionRefPoint( int tempType )
{
  return ivec2( tempType == TMP_ABOVE ? 0 : -1, tempType == TMP_LEFT ? 0 : -1 );
}

// fusion candidate j's input to the filter model at (x, y): its sample less its sample at the reference point (with
// LIC both through its model: m_memOffsetsFusionUseMR)
int tmpFusionInput( Cu c, int at, int first, int j, int x, int y, int tempType, bool lic, int maxPel )
{
  const ivec2 rp = tmpFusionRefPoint( tempType );
  return tmpFusionRef( c, at, first, j, x, y, lic, maxPel ) - tmpFusionRef( c, at, first, j, rp.x, rp.y, lic, maxPel );
}

// convertDiff2Weight: n template costs into weights summing to 64, each against the others' total (a zero cost as 1),
// normalized by the divSig table, then rounded to the sum from the last (down) or the first (up)
void tmpFusionWeights( int cost[TMP_FUSION_NUM], int n, out int w[TMP_FUSION_NUM] )
{
  for( int i = 0; i < TMP_FUSION_NUM; i++ )
  {
    w[i] = 0;
  }
  if( n <= 1 )
  {
    w[0] = 64;
    return;
  }
  int sumW = 0;
  for( int i = 0; i < n; i++ )
  {
    cost[i] = cost[i] == 0 ? 1 : cost[i];
    sumW += cost[i];
  }
  for( int i = 0; i < n; i++ )
  {
    cost[i] = sumW - cost[i];
  }
  const int divSigTable[16] = int[16]( 0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0 );
  const int sumDiff = ( n - 1 ) * sumW;
  int       x = findMSB( sumDiff );
  const int norm = ( sumDiff << 4 >> x ) & 15, v = divSigTable[norm] | 8;
  x += norm != 0 ? 1 : 0;
  const int shift = x + 3, add = 1 << ( shift - 1 );
  sumW = 0;
  for( int j = 0; j < n; j++ )
  {
    w[j] = ( cost[j] * v * 64 + add ) >> shift;
    sumW += w[j];
  }
  while( sumW > 64 )
  {
    for( int j = n - 1; j >= 0 && sumW > 64; j-- )
    {
      if( w[j] != 0 )
      {
        w[j]--;
        sumW--;
      }
    }
  }
  while( sumW < 64 )
  {
    for( int j = 0; j < n && sumW < 64; j++ )
    {
      w[j]++;
      sumW++;
    }
  }
}

// CccmCovariance::solve1 of M (<= 6) parameters from the products' sums over `samples` samples (ata's upper triangle,
// atb): the offset taken out of the cross-correlations, both scaled to 28 bits less the samples' and the bit depth's
// (cccmMatrixBits: 32 above 10 bits), gaussElimination with the regularization 2 << (bitDepth - 8) and xDivide's
// division, the back substitution, the offset added to the bias; without samples clearModel's (a bias of 1)
int64_t[6] tmpSolve1( int64_t ata[6][6], int64_t atb[6], int M, int samples, int offset, int bitDepth )
{
  int64_t p[6] = int64_t[6]( 0, 0, 0, 0, 0, 0 );
  if( samples == 0 )   // (clearModel)
  {
    for( int i = 0; i < 6; i++ )
    {
      p[i] = i == M - 1 ? int64_t( 1 ) << cccmDecim() : int64_t( 0 );
    }
  }
  else
  {
    for( int i = 0; i < M; i++ )
    {
      atb[i] -= ( ata[i][M - 1] * int64_t( offset ) ) >> ( bitDepth - 1 );
    }
    const int shift = cccmMatrixBits( GLM_MATRIX ) - 2 * bitDepth - cccmCeilLog2( samples );
    int64_t   cm[6][7];
    for( int i = 0; i < M; i++ )
    {
      for( int j = 0; j < M; j++ )
      {
        const int64_t v = j >= i ? ata[i][j] : ata[j][i];
        cm[i][j] = shift > 0 ? v << shift : v >> -shift;
      }
      cm[i][M] = shift > 0 ? atb[i] << shift : atb[i] >> -shift;
      cm[i][i] += int64_t( 2 << ( bitDepth - 8 ) );
    }
    for( int i = 0; i < M; i++ )   // gaussElimination
    {
      int sc, rd, sh;
      cccmDivScale( cm[i][i] < int64_t( 1 ) ? int64_t( 1 ) : cm[i][i], sc, rd, sh );
      for( int j = i + 1; j <= M; j++ )
      {
        cm[i][j] = ( cm[i][j] * int64_t( sc ) + int64_t( rd ) ) >> sh;
      }
      for( int r = i + 1; r < M; r++ )
      {
        const int64_t f = cm[r][i];
        for( int k = i + 1; k <= M; k++ )
        {
          cm[r][k] -= ( f * cm[i][k] + int64_t( 1 << ( cccmDecim() - 1 ) ) ) >> cccmDecim();
        }
      }
    }
    for( int i = M - 1; i >= 0; i-- )   // gaussBacksubstitution
    {
      int64_t v = cm[i][M];
      for( int j = i + 1; j < M; j++ )
      {
        v -= ( cm[i][j] * p[j] + int64_t( 1 << ( cccmDecim() - 1 ) ) ) >> cccmDecim();
      }
      p[i] = v;
    }
    p[M - 1] += int64_t( offset ) << ( cccmDecim() - ( bitDepth - 1 ) );
  }
  return p;
}

// the node's part of fusion CU c (invocation 0, with its list of n entries in TS_REFINE, `main`): its group's vectors
// as later CUs' lists take them (cpuBlockVectors' fusion branch: the group's TMP_BV_FUS_CAND_MRG, then the list's first
// where the group is not the first; getSparseArBvMergeCandidate's: the group's first and the list's first where its
// index is not 0), with LIC each candidate's model (LicItmp at its vector with its required template), then by its
// index a filter model (from TMP_GROUP_IDX on: xTMPFusionCalcParams' samples of the template - each candidate's less
// its reference point's, a bias -, solve1) or the blend's (a candidate of the top or the left template alone: the
// first of each and up to 3 others for the location-dependent blend; else weights of the costs)
void tmpFusionSetup( Cu c, int cuIndex, int head, int tempType, int n, int bitDepth )
{
  BufI32    data = BufI32( P.timdData );
  CBufI32   o    = CBufI32( P.dimdOut );
  const int at = cuIndex * DIMD_OUT, tmpIdx = data.v[head + TMP_HEAD_IDX], maxPel = ( 1 << bitDepth ) - 1;
  const ivec2 grp = tmpFusionGroup( tmpIdx );
  const int first = grp.x, num = grp.y;
  const bool lic = data.v[head + TMP_HEAD_LIC] != 0, ext = data.v[head + TMP_HEAD_LIC_EXT] != 0;
  // its vectors for later CUs, its own (pu.bv, mv: the group's first)
  const int bvAt = P.bvOut + data.v[c.bvList + BV_LIST_INDEX] * BV_OUT;
  const int numBv = min( TMP_BV_FUS_CAND_MRG, MTMP_NUM - first );
  data.v[bvAt] = numBv + ( first != 0 ? 1 : 0 );
  for( int j = 0; j < numBv; j++ )
  {
    data.v[bvAt + 1 + j] = o.v[at + DIMD_OUT_TMP_BV + first + j];
  }
  if( first != 0 )
  {
    data.v[bvAt + 1 + numBv] = o.v[at + DIMD_OUT_TMP_BV];
  }
  data.v[bvAt + BV_OUT_SPARSE]     = tmpIdx > 0 ? 2 : 1;
  data.v[bvAt + BV_OUT_SPARSE + 1] = o.v[at + DIMD_OUT_TMP_BV + first];
  data.v[bvAt + BV_OUT_SPARSE + 2] = o.v[at + DIMD_OUT_TMP_BV];
  data.v[bvAt + BV_OUT_FUSION]     = first != 0 ? numBv : -1;
  data.v[bvAt + BV_OUT_SGPM]       = -1;
  const ivec2 own = bvUnpack( o.v[at + DIMD_OUT_TMP_BV + first] );
  o.v[at + DIMD_OUT_BV]         = o.v[at + DIMD_OUT_TMP_BV + first];
  o.v[at + DIMD_OUT_TMP_MV]     = own.x << 4;
  o.v[at + DIMD_OUT_TMP_MV + 1] = own.y << 4;
  // the candidates' required templates (cyclic as the list) and LIC models
  int req[TMP_FUSION_NUM];
  for( int j = 0; j < num; j++ )
  {
    req[j] = tlAt( TS_REFINE, ( first + j ) % n, TE_REGION ) >> 8;
    if( lic )
    {
      const ivec2 bv = bvUnpack( o.v[at + DIMD_OUT_TMP_BV + first + j] );
      const int   m[TMP_LIC] = tmpLicModel( c, bv << 4, ivec2( 0 ), 0, ext ? 0 : req[j], bitDepth, 0 );
      o.v[at + DIMD_OUT_TMP_FUS_LIC + 3 * j]     = m[1];
      o.v[at + DIMD_OUT_TMP_FUS_LIC + 3 * j + 1] = m[2];
      o.v[at + DIMD_OUT_TMP_FUS_LIC + 3 * j + 2] = m[3];
    }
  }
  if( tmpIdx < TMP_GROUP_IDX )   // the blend's
  {
    int  topFirst = -1, leftFirst = -1, nonLoc[3] = int[3]( -1, -1, -1 ), nb = 0;
    bool conversion = true;
    for( int j = first; j < first + num; j++ )
    {
      const int r = req[j - first];
      if( r == 1 || r == 2 )
      {
        topFirst  = r == 1 && topFirst < 0 ? j : topFirst;
        leftFirst = r == 2 && leftFirst < 0 ? j : leftFirst;
        conversion = false;
      }
      else if( nb < MAX_NON_LOC_DEP_NUM )
      {
        nonLoc[nb++] = j;
      }
    }
    if( nb == 0 )   // (the group's last 3 but the top and left ones)
    {
      for( int k = MAX_NON_LOC_DEP_NUM; k > 0; k-- )
      {
        const int idx = first + num - k;
        if( idx != topFirst && idx != leftFirst )
        {
          nonLoc[nb++] = idx;
        }
      }
    }
    int cost[TMP_FUSION_NUM], w[TMP_FUSION_NUM];
    for( int j = 0; j < TMP_FUSION_NUM; j++ )
    {
      cost[j] = j < num ? tlAt( TS_REFINE, ( first + j ) % n, TE_COST ) : 0;
    }
    tmpFusionWeights( cost, conversion ? num : 1, w );
    o.v[at + DIMD_OUT_TMP_FUS]     = topFirst;
    o.v[at + DIMD_OUT_TMP_FUS + 1] = leftFirst;
    for( int k = 0; k < 3; k++ )
    {
      o.v[at + DIMD_OUT_TMP_FUS + 2 + k] = nonLoc[k];
    }
    for( int j = 0; j < TMP_FUSION_NUM; j++ )
    {
      o.v[at + DIMD_OUT_TMP_FUS + 5 + j] = w[j];
    }
    return;
  }
  // the filter model: num + 1 parameters over the template's samples (the rows above the block, then the columns left
  // of it; the L-shape without its corner), solve1 with the current template's reference point as the offset
  const int   M = num + 1, T = TMP_TEMPLATE_SIZE;
  const ivec2 rp = tmpFusionRefPoint( tempType );
  const int   offset = recoAt( 0, c.x + rp.x, c.y + rp.y ), bias = 1 << ( bitDepth - 1 );
  int64_t ata[6][6], atb[6];
  for( int i = 0; i < 6; i++ )
  {
    atb[i] = 0;
    for( int j = 0; j < 6; j++ )
    {
      ata[i][j] = 0;
    }
  }
  const int numTop = tempType != TMP_LEFT ? T * c.w : 0, numLeft = tempType != TMP_ABOVE ? T * c.h : 0;
  const int samples = numTop + numLeft;
  for( int s = 0; s < samples; s++ )
  {
    const int x = s < numTop ? s % c.w : s % T - T, y = s < numTop ? s / c.w - T : ( s - numTop ) / T;
    int a[6];
    for( int j = 0; j < 6; j++ )
    {
      a[j] = j < num ? tmpFusionInput( c, at, first, j, x, y, tempType, lic, maxPel ) : j == num ? bias : 0;
    }
    const int cb = recoAt( 0, c.x + x, c.y + y );
    for( int i = 0; i < M; i++ )
    {
      for( int j = i; j < M; j++ )
      {
        ata[i][j] += int64_t( a[i] * a[j] );
      }
      atb[i] += int64_t( a[i] * cb );
    }
  }
  const int64_t p[6] = tmpSolve1( ata, atb, M, samples, offset, bitDepth );
  for( int i = 0; i < 6; i++ )
  {
    o.v[at + DIMD_OUT_TMP_FUS_PAR + 2 * i]     = int( uint64_t( p[i] ) );
    o.v[at + DIMD_OUT_TMP_FUS_PAR + 2 * i + 1] = int( uint64_t( p[i] ) >> 32 );
  }
}

// ----- FLM (xGetTmpFlmRefBuf / xCalcTmpFlmRefArea, xCalTmpFlmParam, xGenerateTmpFlmPred)

// the FLM area's sample (ax, ay) - relative to its top-left, the template's at the CU's vector bv: the block with the
// template's rows above / columns left -, less the area's offset, padded where pads says (bits 1 left, 2 right, 4 top,
// 8 bottom: the area's edge sample for the one beyond it)
int tmpFlmRef( Cu c, ivec2 bv, int tempType, int pads, int ax, int ay, int offset )
{
  const int rx = tempType != TMP_ABOVE ? TMP_TEMPLATE_SIZE : 0, ry = tempType != TMP_LEFT ? TMP_TEMPLATE_SIZE : 0;
  const int aw = c.w + rx, ah = c.h + ry;
  ax = ( pads & 1 ) != 0 && ax < 0 ? 0 : ( pads & 2 ) != 0 && ax >= aw ? aw - 1 : ax;
  ay = ( pads & 4 ) != 0 && ay < 0 ? 0 : ( pads & 8 ) != 0 && ay >= ah ? ah - 1 : ay;
  const int x = clamp( c.x + bv.x - rx + ax, 0, P.width[0] - 1 ), y = clamp( c.y + bv.y - ry + ay, 0, P.height[0] - 1 );
  return recoAt( 0, x, y ) - offset;
}

// FLM's feature j at area sample (ax, ay): the sample, the ones above, below, left and right of it, the bias
int tmpFlmFeature( Cu c, ivec2 bv, int tempType, int pads, int offset, int j, int ax, int ay, int bitDepth )
{
  const ivec2 d = j == 1 ? ivec2( 0, -1 ) : j == 2 ? ivec2( 0, 1 ) : j == 3 ? ivec2( -1, 0 ) : j == 4 ? ivec2( 1, 0 )
                                                                                                     : ivec2( 0 );
  return j == 5 ? 1 << ( bitDepth - 1 ) : tmpFlmRef( c, bv, tempType, pads, ax + d.x, ay + d.y, offset );
}

// the node's part of FLM CU c (invocation 0, with its list of n entries in TS_REFINE): its area's padding (an entry of
// a neighbour's vector or the cascade all sides; else the sides where the vector is at its region's bound, and for the
// regions 1, 3, 4 and 5 right / below where the samples there are not decoded), its offset (the template's corner
// sample at the vector), the model's 6 parameters (solve1 over the template parts its required template takes: the
// rows above the block, else the columns left; with the whole L-shape both; the current template's corner as the
// offset) into its DimdOut
void tmpFlmSetup( Cu c, int cuIndex, int head, int tempType, int n, int bitDepth )
{
  BufI32      data = BufI32( P.timdData );
  CBufI32     o    = CBufI32( P.dimdOut );
  OwnerMap    om   = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  const int   at = cuIndex * DIMD_OUT, tmpIdx = data.v[head + TMP_HEAD_IDX], T = TMP_TEMPLATE_SIZE;
  const ivec2 bv = bvUnpack( o.v[at + DIMD_OUT_TMP_BV + tmpIdx] );
  const int   tag = tlAt( TS_REFINE, tmpIdx % n, TE_REGION ), region = tag & 15, req = tag >> 8;
  int pads = 15;
  if( region <= 5 )
  {
    const ivec4 g = tmpRegion( head, region );
    pads = ( bv.x > g.x ? 0 : 1 ) | ( bv.x < g.y ? 0 : 2 ) | ( bv.y > g.z ? 0 : 4 ) | ( bv.y < g.w ? 0 : 8 );
    if( region != 0 && region != 2 )
    {
      pads |= decodedBefore( om, c.x + bv.x + c.w - 1, c.y + bv.y + c.h, g_ti ) ? 0 : 8;
      pads |= decodedBefore( om, c.x + bv.x + c.w, c.y + bv.y + c.h - 1, g_ti ) ? 0 : 2;
    }
  }
  const int   rx = tempType != TMP_ABOVE ? T : 0, ry = tempType != TMP_LEFT ? T : 0;
  const ivec2 rp = tmpFusionRefPoint( tempType );
  const int   offset = tmpFlmRef( c, bv, tempType, 0, rx + rp.x, ry + rp.y, 0 );   // (the area's corner)
  const int   curOffset = recoAt( 0, c.x + rp.x, c.y + rp.y );
  const bool  aboveDone = ( tempType == TMP_L_SHAPE && req != 2 ) || tempType == TMP_ABOVE;
  const int   numFirst = aboveDone ? T * c.w : T * c.h;
  const int   samples = numFirst + ( tempType == TMP_L_SHAPE && req == 0 ? T * c.h : 0 );
  int64_t ata[6][6], atb[6];
  for( int i = 0; i < 6; i++ )
  {
    atb[i] = 0;
    for( int j = 0; j < 6; j++ )
    {
      ata[i][j] = 0;
    }
  }
  for( int s = 0; s < samples; s++ )
  {
    // (x, y) relative to the CU: the rows above it, else / then the columns left of it
    const bool top = aboveDone && s < numFirst;
    const int  k = s < numFirst ? s : s - numFirst;
    const int  x = top ? k % c.w : k % T - T, y = top ? k / c.w - T : k / T;
    int a[6];
    for( int j = 0; j < 6; j++ )
    {
      a[j] = tmpFlmFeature( c, bv, tempType, pads, offset, j, x + rx, y + ry, bitDepth );
    }
    const int cb = recoAt( 0, c.x + x, c.y + y );
    for( int i = 0; i < 6; i++ )
    {
      for( int j = i; j < 6; j++ )
      {
        ata[i][j] += int64_t( a[i] * a[j] );
      }
      atb[i] += int64_t( a[i] * cb );
    }
  }
  const int64_t p[6] = tmpSolve1( ata, atb, 6, samples, curOffset, bitDepth );
  for( int i = 0; i < 6; i++ )
  {
    o.v[at + DIMD_OUT_TMP_FUS_PAR + 2 * i]     = int( uint64_t( p[i] ) );
    o.v[at + DIMD_OUT_TMP_FUS_PAR + 2 * i + 1] = int( uint64_t( p[i] ) >> 32 );
  }
  o.v[at + DIMD_OUT_TMP_FUS]     = pads;
  o.v[at + DIMD_OUT_TMP_FUS + 1] = offset;
}

// the fusion CU c's prediction at (x, y) relative to it (its node's DimdOut at `at`): the filter model's convolution
// (as a Pel, clipped), the location-dependent blend (xLocationdepBlending with range 16 of the top / left candidates
// over the others' blend: one alone, two 32 : 32, three 22 : 21 : 21), else the weighted blend (tmpFusionBlending)
int tmpFusionSample( Cu c, int at, int x, int y, int bitDepth )
{
  CBufI32     o      = CBufI32( P.dimdOut );
  BufI32      data   = BufI32( P.timdData );
  const int   tmpIdx = data.v[c.tmp + TMP_HEAD_IDX], tempType = data.v[c.tmp + TMP_HEAD_TEMP_TYPE];
  const bool  lic    = data.v[c.tmp + TMP_HEAD_LIC] != 0;
  const ivec2 grp    = tmpFusionGroup( tmpIdx );
  const int   first = grp.x, num = grp.y, maxPel = ( 1 << bitDepth ) - 1;
  const bool  flm   = data.v[c.tmp + TMP_HEAD_FLM] != 0;
  if( flm || tmpIdx >= TMP_GROUP_IDX )   // (a model's convolution: FLM's or fusion's filter)
  {
    int64_t sum = 0;
    for( int j = 0; j <= ( flm ? 5 : num ); j++ )
    {
      const int64_t pj = int64_t( uint( o.v[at + DIMD_OUT_TMP_FUS_PAR + 2 * j] ) )
                         | ( int64_t( o.v[at + DIMD_OUT_TMP_FUS_PAR + 2 * j + 1] ) << 32 );
      if( flm )
      {
        const int rx = tempType != TMP_ABOVE ? TMP_TEMPLATE_SIZE : 0, ry = tempType != TMP_LEFT ? TMP_TEMPLATE_SIZE : 0;
        sum += pj * int64_t( tmpFlmFeature( c, bvUnpack( o.v[at + DIMD_OUT_BV] ), tempType, o.v[at + DIMD_OUT_TMP_FUS],
                                            o.v[at + DIMD_OUT_TMP_FUS + 1], j, x + rx, y + ry, bitDepth ) );
        continue;
      }
      sum += pj * int64_t( j < num ? tmpFusionInput( c, at, first, j, x, y, tempType, lic, maxPel )
                                   : 1 << ( bitDepth - 1 ) );
    }
    return clamp( int( int16_t( ( sum + int64_t( 1 << ( cccmDecim() - 1 ) ) ) >> cccmDecim() ) ), 0, maxPel );
  }
  const int topFirst = o.v[at + DIMD_OUT_TMP_FUS], leftFirst = o.v[at + DIMD_OUT_TMP_FUS + 1];
  if( topFirst < 0 && leftFirst < 0 )
  {
    int blend = 32;
    for( int j = 0; j < num; j++ )
    {
      blend += o.v[at + DIMD_OUT_TMP_FUS + 5 + j] * tmpFusionRef( c, at, first, j, x, y, lic, maxPel );
    }
    return int( int16_t( blend >> 6 ) );
  }
  int nb = 0, non = 32;
  const int wn[3] = int[3]( 22, 21, 21 );
  for( int k = 0; k < 3; k++ )
  {
    nb += o.v[at + DIMD_OUT_TMP_FUS + 2 + k] >= 0 ? 1 : 0;
  }
  for( int k = 0; k < nb; k++ )
  {
    const int v = tmpFusionRef( c, at, first, o.v[at + DIMD_OUT_TMP_FUS + 2 + k] - first, x, y, lic, maxPel );
    non += ( nb == 1 ? 64 : nb == 2 ? 32 : wn[k] ) * v;
  }
  const int pNon = int( int16_t( non >> 6 ) );
  const int mode = topFirst >= 0 ? ( leftFirst >= 0 ? 0 : 1 ) : 2;
  const int wDir = mode == 0 ? ( nb == 1 ? 16 : 8 ) : 16, wNon = 64 - ( mode == 0 ? 2 * wDir : wDir );
  const int range = 16, W1 = c.w - 1, H1 = c.h - 1;
  const int pVer = topFirst >= 0 ? tmpFusionRef( c, at, first, topFirst - first, x, y, lic, maxPel ) : 0;
  const int pHor = leftFirst >= 0 ? tmpFusionRef( c, at, first, leftFirst - first, x, y, lic, maxPel ) : 0;
  const int rangeVer = c.h > 64 ? 2 * range : range, rangeHor = c.w > 64 ? 2 * range : range;
  int wv = 64, wh = 64;
  if( mode == 0 )   // (the clip ranges: each direction's weight plus its half of the rest, exact here)
  {
    const int clipVer = wDir + dimdScaledWeight( wDir, 2 * wDir, wNon ), clipHor = clipVer;
    const bool needVer = wDir + rangeVer > clipVer || wDir - rangeVer < 0;
    const bool needHor = wDir + rangeHor > clipHor || wDir - rangeHor < 0;
    wv = H1 != 0 ? divTrunc( ( wDir + rangeVer ) * H1 - ( ( rangeVer * y ) << 1 ), H1 ) : 64;
    wv = H1 != 0 && needVer ? clamp( wv, 0, clipVer ) : wv;
    wh = W1 != 0 ? divTrunc( ( wDir + rangeHor ) * W1 - ( ( rangeHor * x ) << 1 ), W1 ) : 64 - wv;
    wh = W1 != 0 && needHor ? clamp( wh, 0, clipHor ) : wh;
    return int( int16_t( ( pVer * wv + pHor * wh + pNon * ( 64 - wv - wh ) + 32 ) >> 6 ) );
  }
  if( mode == 1 )
  {
    const bool need = wDir + rangeVer > 64 || wDir - rangeVer < 0;
    wv = H1 != 0 ? divTrunc( ( wDir + rangeVer ) * H1 - ( ( rangeVer * y ) << 1 ), H1 ) : 64;
    wv = H1 != 0 && need ? clamp( wv, 0, 64 ) : wv;
    return int( int16_t( ( pVer * wv + pNon * ( 64 - wv ) + 32 ) >> 6 ) );
  }
  const bool need = wDir + rangeHor > 64 || wDir - rangeHor < 0;
  wh = W1 != 0 ? divTrunc( ( wDir + rangeHor ) * W1 - ( ( rangeHor * x ) << 1 ), W1 ) : 64;
  wh = W1 != 0 && need ? clamp( wh, 0, 64 ) : wh;
  return int( int16_t( ( pHor * wh + pNon * ( 64 - wh ) + 32 ) >> 6 ) );
}

// ----- the node

// the rows of a chunk of the sparse search or the refinement (cut by invocation 0 in tmpNode): per row TS_RROW_WORDS
// words - the refinement's: its y, its right and its left end, its positions' tag, its candidate's region with whether
// its window is valid (16), whether the row's walk has found a decoded block yet (32) and whether the row goes on in
// the next chunk (64), its slot in TS_RBUF; the sparse search's: its y, its first x, its number of positions, its
// region, whether its region checks decoded blocks (1), whether the row has found one yet (2) and whether it goes on
// (4), its slot, its region's right and bottom bound -, its positions' number (TS_RCNT), where they go in the chunk
// (TS_RDST); per position in TS_RBUF two words as at TS_POS; the state a row going on leaves for the next chunk
// (TS_RFOUND). After the sub-sample search's area
const int TS_RROW_WORDS = 8;
const int TS_RROW       = TS_FCOST + 4 * TMP_FRAC_CANDS;
const int TS_RCNT       = TS_RROW + TS_RROW_WORDS * TMP_CHUNK;
const int TS_RDST       = TS_RCNT + TMP_CHUNK;
const int TS_RBUF       = TS_RDST + TMP_CHUNK;
const int TS_RFOUND     = TS_RBUF + 2 * TMP_CHUNK;

// row r of the refinement's chunk (one invocation): its positions as the walk of the windows takes them, from the row's
// right end leftwards - a checked one (a region's position where the window may leave the decoded area) only from
// the first whose block is decoded on, a vector window's off the decoded area (not validWindow) only where its block
// is valid (bvValid; BVG-DIMD's: from the first whose block is decoded, where the template's corner is) - into
// TS_RBUF from its slot on, their number into TS_RCNT, the walk's state where the row goes on into TS_RFOUND
void tmpRefineRow( Cu c, int r, bool licMode, bool bvg, OwnerMap om )
{
  const int  rb = TS_RROW + TS_RROW_WORDS * r, py = tsGet( rb ), xR = tsGet( rb + 1 ), xL = tsGet( rb + 2 );
  const int  tag = tsGet( rb + 3 ), mode = tsGet( rb + 4 ), slot = tsGet( rb + 5 );
  const int  candRegion = mode & 15, W = c.w, H = c.h;
  const bool validWindow = ( mode & 16 ) != 0;
  bool       found = ( mode & 32 ) != 0;
  int        n = 0;
  for( int px = xR; px >= xL; px-- )
  {
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
    else if( candRegion > 5 && !validWindow && !bvg && !bvValid( c.x, c.y, W, H, ivec2( px, py ), g_ti ) )
    {
      continue;
    }
    else if( candRegion > 5 && !validWindow && bvg )   // (validItmpBv with isAvailablePairFound)
    {
      if( ( !found && !decodedBefore( om, c.x + px + W - 1, c.y + py + H - 1, g_ti ) )
          || !decodedBefore( om, c.x + px - TMP_TEMPLATE_SIZE, c.y + py - TMP_TEMPLATE_SIZE, g_ti ) )
      {
        continue;
      }
      found = true;
    }
    tsSet( TS_RBUF + 2 * ( slot + n ), ( px & 0xFFFF ) | ( py << 16 ) );
    tsSet( TS_RBUF + 2 * ( slot + n ) + 1, tag );
    n++;
  }
  tsSet( TS_RCNT + r, n );
  if( ( mode & 64 ) != 0 )
  {
    tsSet( TS_RFOUND, found ? 1 : 0 );
  }
}

// row r of the sparse search's chunk (one invocation): its positions from its first x leftwards by S - in regions 4
// and 5 where the region checks decoded blocks only from the first whose block is decoded on (rowFound); with LIC (mr)
// a position's cost and entry those S / 2 right and below it, but where that leaves the region (transferred, bits 4
// and 5 of its tag) or, before rowFound, its block is not decoded (both transferred) - into TS_RBUF from its slot on,
// their number into TS_RCNT, rowFound where the row goes on into TS_RFOUND
void tmpSparseRow( Cu c, int r, bool mr, int S, OwnerMap om )
{
  const int  rb = TS_RROW + TS_RROW_WORDS * r, y = tsGet( rb ), x0 = tsGet( rb + 1 ), take = tsGet( rb + 2 );
  const int  region = tsGet( rb + 3 ), mode = tsGet( rb + 4 ), slot = tsGet( rb + 5 );
  const int  gRight = tsGet( rb + 6 ), gBottom = tsGet( rb + 7 ), W = c.w, H = c.h;
  const bool checkR4R5 = ( mode & 1 ) != 0;
  bool       rowFound = ( mode & 2 ) != 0;
  int        n = 0;
  for( int k = 0; k < take; k++ )
  {
    const int px = x0 - k * S;
    int       mx = px + ( S >> 1 ), my = y + ( S >> 1 ), flags = 0;
    if( mx > gRight )
    {
      mx = px;
      flags |= 16;
    }
    if( my > gBottom )
    {
      my = y;
      flags |= 32;
    }
    if( ( region == 4 || region == 5 ) && !rowFound && checkR4R5 )
    {
      if( !decodedBefore( om, c.x + px + W - 1, c.y + y + H - 1, g_ti ) )
      {
        continue;
      }
      if( mr && !decodedBefore( om, c.x + mx + W - 1, c.y + my + H - 1, g_ti ) )
      {
        mx    = px;
        my    = y;
        flags = 48;
      }
      else
      {
        rowFound = true;
      }
    }
    const ivec2 p = mr ? ivec2( mx, my ) : ivec2( px, y );
    tsSet( TS_RBUF + 2 * ( slot + n ), ( p.x & 0xFFFF ) | ( p.y << 16 ) );
    tsSet( TS_RBUF + 2 * ( slot + n ) + 1, mr ? region | flags : region );
    n++;
  }
  tsSet( TS_RCNT + r, n );
  if( ( mode & 4 ) != 0 )
  {
    tsSet( TS_RFOUND, rowFound ? 1 : 0 );
  }
}

// the chunk's rows' positions after each other in the rows' order into TS_POS, their number into s_tmpCount, which
// held the rows' (all invocations, after tmpSparseRow / tmpRefineRow and a tsSync; ending with a barrier)
void tmpChunkPack( int lid, int NT )
{
  const int numRows = s_tmpCount;
  int       total   = 0;
  if( lid == 0 )
  {
    for( int r = 0; r < numRows; r++ )
    {
      tsSet( TS_RDST + r, total );
      total += tsGet( TS_RCNT + r );
    }
  }
  tsSync();
  for( int r = lid; r < numRows; r += NT )
  {
    const int slot = tsGet( TS_RROW + TS_RROW_WORDS * r + 5 ), n = tsGet( TS_RCNT + r ), dst = tsGet( TS_RDST + r );
    for( int j = 0; j < n; j++ )
    {
      tsSet( TS_POS + 2 * ( dst + j ), tsGet( TS_RBUF + 2 * ( slot + j ) ) );
      tsSet( TS_POS + 2 * ( dst + j ) + 1, tsGet( TS_RBUF + 2 * ( slot + j ) + 1 ) );
    }
  }
  tsSync();
  if( lid == 0 )
  {
    s_tmpCount = total;
  }
  tsSync();
}

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
  const int  search   = data.v[head + TMP_HEAD_SEARCH];
  const bool onlyBest = search == TMP_SEARCH_BEST, bvg = search == TMP_SEARCH_BVG;
  const bool mr       = data.v[head + TMP_HEAD_LIC] != 0;   // (useMR: LIC's mean-removed costs)
  const int  tlSparse = mr && !licMode ? TL_NUM : TL_NUM_SPARSE;
  const int  numSparse[3] = int[3]( search != TMP_SEARCH_FULL ? 1
                                    : mr && !licMode       ? MTMP_NUM_SPARSE_FOR_LIC
                                                           : MTMP_NUM_SPARSE,
                                    tlSparse, tlSparse );
  const int  numRefine[3] = int[3]( data.v[head + TMP_HEAD_NUM_REFINE], TL_NUM, TL_NUM );
  const int  numTL = ( W + TMP_TEMPLATE_SIZE ) * TMP_TEMPLATE_SIZE + TMP_TEMPLATE_SIZE * H;   // numPixTopLeft
  const int  numTop = W * TMP_TEMPLATE_SIZE, numLeft = TMP_TEMPLATE_SIZE * H;

  g_tmpMr   = mr;
  g_tmpMean = mr ? tmpMeans( c, tempType, 0, 0 ) : ivec2( 0 );
  tmpCurLoad( c, lid, NT );

  // invocation 0's state: the vector list's parts (itmpSize, itmpSize1), the thresholds, the enumeration
  int itmpSize = 0, itmpSize1 = 0, numList = 0;
  int pDiff[3] = int[3]( numTL << bitDepth, numTop << bitDepth, numLeft << bitDepth );
  const int numNeighbours = bvNeighbourList( c, g_ti, lid, NT );
  if( lid == 0 )
  {
    s_bvLen = tmpSortInOut( head, numNeighbours );
  }
  tsSync();
  itmpSize  = s_bvLen;
  tsSync();   // (read before bvCascade writes it)
  itmpSize1 = bvCascade( c, itmpSize, g_ti, lid, NT );
  if( lid == 0 )
  {
    for( int t = 0; t < 3; t++ )
    {
      tlClear( TS_SPARSE + t * TS_LIST );
      tlClear( TS_REFINE + t * TS_LIST );
    }
  }
  clockMark( 0 );   // (BLOCK_CLOCK: the setup)

  // the sparse search: the regions 3, 4, 5, 0, 1, 2 (but for a small block a region inside a neighbour vector's
  // window), rows from the bottom up by S, row r's first position S - 1 - (r - 1) % S left of the right edge, then
  // to the left by S; in regions 4 and 5 (beyond the CU in its CTU) the positions whose block is decoded, until a
  // row's first such. With LIC a position's cost and entry are those S / 2 right and below it (iXOffsetMetric), but
  // where that leaves the region (transferred: the window then extends from it by S - 1) or, in regions 4 and 5 before
  // the row's first such, its block is not decoded (both transferred). Each chunk costed in parallel, then put into the
  // sparse lists by invocation 0 (a region's tag with the transferred flags as bits 4 and 5)
  // (a chunk's positions: the regions' next rows, invocation 0 cutting them as the refinement's below, each row's
  // positions taken by an invocation of its own, tmpSparseRow)
  int   checkIdx = 0, region = 0, y = 0, x = 0, shiftX = 0;
  bool  inRegion = false, newRow = false, rowFound = false, checkR4R5 = true, goesOn = false;
  ivec4 g = ivec4( 0 );
  for( ;; )
  {
    if( lid == 0 )
    {
      if( goesOn )   // (a row the last chunk began goes on with the state its invocation left)
      {
        rowFound = tsGet( TS_RFOUND ) != 0;
        goesOn   = false;
      }
      int numRows = 0, width = 0;
      while( width < TMP_CHUNK && checkIdx < 6 )
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
        // the row's positions from x leftwards by S, as many as the chunk has room for
        const int count = ( x - g.x ) / S + 1, take = min( count, TMP_CHUNK - width );
        const int rb    = TS_RROW + TS_RROW_WORDS * numRows;
        goesOn          = take < count;
        tsSet( rb, y );
        tsSet( rb + 1, x );
        tsSet( rb + 2, take );
        tsSet( rb + 3, region );
        tsSet( rb + 4, ( checkR4R5 ? 1 : 0 ) | ( rowFound ? 2 : 0 ) | ( goesOn ? 4 : 0 ) );
        tsSet( rb + 5, width );
        tsSet( rb + 6, g.y );
        tsSet( rb + 7, g.w );
        numRows++;
        width += take;
        x     -= take * S;
        if( goesOn )
        {
          break;
        }
      }
      s_tmpCount = numRows;   // (the rows; the positions below)
      s_tmpDone  = checkIdx >= 6 ? 1 : 0;
      for( int t = 0; t < numTypes; t++ )
      {
        s_tmpThr[t] = pDiff[t];
      }
    }
    tsSync();
    for( int r = lid; r < s_tmpCount; r += NT )
    {
      tmpSparseRow( c, r, mr, S, om );
    }
    tsSync();
    tmpChunkPack( lid, NT );
    tmpChunkCosts( c, tempType, numTypes, ivec3( numSparse[0], numSparse[1], numSparse[2] ), false, lid, NT );
    const bool last = s_tmpDone != 0;   // (read before invocation 0 may write the next chunk's)
    if( lid == 0 )
    {
      const int rr = S >> 1;
      for( int k = 0; k < s_tmpCount; k++ )
      {
        if( s_spCost[k] == 0u )
        {
          continue;
        }
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

  clockMark( 1 );   // (BLOCK_CLOCK: the sparse search)

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
  // more; BVG-DIMD's search takes none), then the costs of every vector of the list
  if( lid == 0 )
  {
    numList = itmpSize1;
    const int total = numList + NUM_TMP_ARBVP_S;
    for( int i = 0; i < tlSize( TS_SPARSE ) && numList < total && !bvg; i++ )
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
  tmpChunkCosts( c, tempType, 0, ivec3( 0 ), false, lid, NT );   // (at most BV_LIST_MAX)
  clockMark( 2 );   // (BLOCK_CLOCK: the vector list's cascade and costs)

  // BVG-DIMD (invocation 0): of the sparse search's best and the neighbours' vectors that cost less than it, in turn,
  // the cheapest alone (bvgCandList of MAX_BVG_ROUGH_NUM 1), a vector's window BVG_REFINE_RANGE around it; the
  // refinement's threshold 1.2 times its cost, plus 1, without clustering (no position is tried twice)
  if( lid == 0 && bvg )
  {
    int maxCost = pDiff[0];   // (the sparse best's cost, else the initial threshold: maxSparseCost)
    for( int m = 0; m < numList; m++ )
    {
      const int diff = tsGet( TS_COST + 3 * m );
      if( diff < maxCost )
      {
        const ivec2 cand = bvListAt( m );
        tlUpdate( TS_SPARSE, tlEntry( cand.x, cand.y, cand - BVG_REFINE_RANGE, cand + BVG_REFINE_RANGE, TMP_MRG_REG_ID,
                                      diff ), 1 );
        maxCost = diff;
      }
    }
    tsSet( TS_WIN, 0 );
    tsSet( TS_WIN + 1, bvgThreshold( maxCost ) );
  }
  // the windows of the neighbours' vectors (adaptive: by the cost against its threshold), the cascaded ones' fixed;
  // then the clustering (invocation 0)
  else if( lid == 0 )
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

  clockMark( 3 );   // (BLOCK_CLOCK: the clustering)

  // the refinement: per list type its candidates' windows (a sparse one's clipped to its region but where the
  // exception allows, a vector's whole, its positions checked where the window may leave the decoded area), each
  // position once per list, costed in chunks; the lists of the top and the left template keep TL_NUM. A chunk's
  // positions are the windows' next rows (invocation 0 cuts them, a row too long for the chunk in pieces), each row's
  // taken by an invocation of its own (tmpRefineRow), then put after each other in the rows' order (tmpChunkPack)
  int  type = 0, cand = 0, wy = 0, wx = 0, curType = -1, numRows = 0;
  bool open = false, found = false, validWindow = true, rowGoesOn = false;
  ivec4 win = ivec4( 0 );
  int   candRegion = 0;
  int   rDiff[3] = int[3]( 0, 0, 0 );
  const int clusteredBits = tsGet( TS_WIN );
  for( ;; )
  {
    if( lid == 0 )
    {
      if( rowGoesOn )   // (a row the last chunk began goes on with the state its invocation left)
      {
        found     = tsGet( TS_RFOUND ) != 0;
        rowGoesOn = false;
      }
      numRows = 0;
      int width = 0;
      while( width < TMP_CHUNK && type < numTypes )
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
          const int tag = tlAt( list, cand, TE_REGION );   // (a sparse entry's transferred flags: bits 4, 5)
          candRegion = tag & 15;
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
                                   c.y + rg.z >= TMP_TEMPLATE_SIZE + ( S >> 1 ), !licMode, ( tag & 16 ) != 0,
                                   ( tag & 32 ) != 0 );
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
        // the row's positions from wx leftwards, as many as the chunk has room for
        const int take = min( wx - win.x + 1, TMP_CHUNK - width ), rb = TS_RROW + TS_RROW_WORDS * numRows;
        rowGoesOn      = take < wx - win.x + 1;
        tsSet( rb, wy );
        tsSet( rb + 1, wx );
        tsSet( rb + 2, wx - take + 1 );
        tsSet( rb + 3, type | cand << 2 );
        tsSet( rb + 4, candRegion | ( validWindow ? 16 : 0 ) | ( found ? 32 : 0 ) | ( rowGoesOn ? 64 : 0 ) );
        tsSet( rb + 5, width );
        numRows++;
        width += take;
        wx    -= take;
        if( rowGoesOn )
        {
          break;
        }
      }
      s_tmpCount = numRows;   // (the rows; the positions below)
      s_tmpDone  = type >= numTypes ? 1 : 0;
      for( int t = 0; t < numTypes; t++ )   // (a type not begun yet starts from its own threshold)
      {
        s_tmpThr[t] = t == curType ? rDiff[t] : tsGet( TS_WIN + 1 + t );
      }
    }
    tsSync();
    for( int r = lid; r < s_tmpCount; r += NT )
    {
      tmpRefineRow( c, r, licMode, bvg, om );
    }
    tsSync();
    tmpChunkPack( lid, NT );
    tmpChunkCosts( c, tempType, numTypes, bvg ? ivec3( 0 ) : ivec3( numRefine[0], numRefine[1], numRefine[2] ), true,
                   lid, NT );
    const bool last = s_tmpDone != 0;
    if( lid == 0 )
    {
      for( int k = 0; k < s_tmpCount; k++ )
      {
        if( s_spCost[k] == 0u )   // (a type's first positions skipped: its thresholds are set at its first one kept)
        {
          continue;
        }
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
          if( tlSize( refine ) == numRefine[t] )   // (BVG-DIMD's: also no more than 1.2 times the cheapest, plus 1)
          {
            const int lastCost = tlAt( refine, numRefine[t] - 1, TE_COST );
            rDiff[t] = min( bvg ? min( bvgThreshold( tlAt( refine, 0, TE_COST ) ), lastCost ) : lastCost, rDiff[t] );
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
  if( bvg )   // (its refined list in TS_REFINE: intra_bvg.glsl's bvgNode goes on)
  {
    return;
  }

  clockMark( 4 );   // (BLOCK_CLOCK: the refinement)

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
          const int pos = MTMP_NUM - tlSize( top ) - tlSize( left ) + cnt++;
          const int e   = tlEntryAt( main, tlPlace( main, pos ) );
          for( int f = 0; f < TE; f++ )
          {
            tsSet( e + f, tlAt( list, i, f ) );
          }
          // (m_mtmpRequiredTemplate: t)
          tsSet( e + TE_REGION, ( tlAt( list, i, TE_REGION ) & 255 ) | t << 8 );
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
      o.v[at + DIMD_OUT_BV]         = bvPack( bv[0] );   // (the one its block copies)
      o.v[at + DIMD_OUT_TMP_MV]     = bv[0].x << 4;      // (pu.mv[0], but a sub-sample search's below)
      o.v[at + DIMD_OUT_TMP_MV + 1] = bv[0].y << 4;
      tsSet( TS_FRAC_REQ, tlAt( main, tmpIdx % n, TE_REGION ) >> 8 );
      // with LIC its model (LicItmp at the vector with getRequiredTemplate: with the SPS's ItmpLicExtension the
      // L-shape's) into its DimdOut, and where SGPM's lists may take it, into its slot there (the host's BV_OUT_LIC)
      const int licIdc = data.v[head + TMP_HEAD_LIC];
      if( data.v[head + TMP_HEAD_FUSION] != 0 )
      {
        tmpFusionSetup( c, cuIndex, head, tempType, n, bitDepth );
      }
      else if( data.v[head + TMP_HEAD_FLM] != 0 )
      {
        tmpFlmSetup( c, cuIndex, head, tempType, n, bitDepth );
      }
      else if( licIdc != 0 )
      {
        const int req = data.v[head + TMP_HEAD_LIC_EXT] != 0 ? 0 : tsGet( TS_FRAC_REQ );
        const int model[TMP_LIC] = tmpLicModel( c, bv[0] << 4, ivec2( 0 ), licIdc - 1, req, bitDepth, 0 );
        const int slot = data.v[head + TMP_HEAD_LIC_SLOT];
        for( int k = 0; k < TMP_LIC; k++ )
        {
          o.v[at + DIMD_OUT_TMP_LIC + k] = model[k];
          if( slot >= 0 )
          {
            data.v[slot + k] = model[k];
          }
        }
      }
    }
  }
  if( data.v[head + TMP_HEAD_FRAC] != 0 )
  {
    tsSync();
    tmpFracSearch( c, cuIndex, tempType, bitDepth, lid, NT );
  }
}

// ----- the block

// (IntraTMP block t's prediction, tmpPredict: intra_obmc.glsl, which predicts its OBMC's motion runs with it)
