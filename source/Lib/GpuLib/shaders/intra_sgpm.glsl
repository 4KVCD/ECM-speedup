// SGPM (spatial geometric partitioning mode, JVET-AB0155, with JVET-AJ0107's partitions) on the GPU. An SGPM CU's node
// (DecCu's SGPM branch) has its DIMD (intra_dimd.glsl) and TIMD's best horizontal and vertical modes (intra_timd.glsl's
// timdNode), then derives IntraPrediction::deriveSgpmModeOrdered without regression SGPM (the host leaves those CUs to
// the CPU): its list of block vectors (getItmpMergeCandidate's for SGPM, the vectors of the CUs around it valid for its
// one-sample template, in 1/16 samples) ordered by their template costs, the cheapest of them modes from SGPM_BV_START
// on; per partition of g_sgpmSplitDir the lists of its two parts, three modes (PU::getSgpmIntraMPMs) and those vectors;
// the template costs of every mode they name - its prediction of the template (predTimdIntraAng with SGPM's wide
// angles, a vector's copy of the reconstruction: getPredIBCBlk's bilinear interpolation at a fraction, intra_ibc.glsl),
// its SAD against the reconstruction (sadTM) and per partition that SAD weighted by the partition's template weights
// (sgpmSadTM) - and of each partition's mode pairs (the first mode's weighted SAD plus the second's SAD less its
// weighted one), then the entry of the list those costs rank (updateCandList: ascending, the first of equal costs
// first) that the CU's index selects. Regression SGPM (JVET-AJ0112) instead pairs the entries of a list of the MPM
// list's first six modes and those vectors, fits each pair's blend to the template by least squares
// (deriveSgpmBlending: a weight linear in the position) and ranks the pairs by the SAD of that blend. Its block
// predicts the two modes (a vector's: its copy, at a fraction the luma filter's interpolation) and blends them by the
// partition's weights (xWeightedSgpm) or the regression's (weightAffineBlk, weightedBlendBlk), in intra_rec.comp's
// blending, and takes its transforms by a DIMD of that blend. Included by intra_rec.comp after intra_tmp.glsl.
#ifndef INTRA_SGPM_GLSL
#define INTRA_SGPM_GLSL

// SGPM's tables (in matrices at trTab( TR_TAB_SGPM ); GpuIntraRec.cpp's sgpmTables): per partition the mode of its
// angle (g_geoAngle2IntraAng), then which arms its parts' templates have (g_geoTmShape: bit 0 the first part's left
// arm, bit 1 its above arm, bits 2 and 3 the second part's)
const int SGPM_PARTITIONS = 26;   // SGPM_TOTAL_NUM_PARTITIONS
const int SGPM_TAB_ANGLE  = 0;
const int SGPM_TAB_ARMS   = SGPM_PARTITIONS;
const int SGPM_LIST       = 3;                           // SGPM_NUM_MPM: a part's modes
const int SGPM_NUM_BVS    = 6;                           // the list's cheapest vectors that become modes
const int SGPM_BV_START   = NUM_LUMA_MODE;               // SGPM_BV_START_IDX: mode SGPM_BV_START + j their j-th
const int SGPM_LIST_MAX   = SGPM_LIST + SGPM_NUM_BVS;    // a part's entries with them
const int SGPM_VIPM_SIZE  = 256;   // MAX_SGPM_VIPM_SIZE: the CUs below it whose transforms check the DIMD's modes
const int RSGPM_MPMS      = 6;     // NUM_PRIMARY_MOST_PROBABLE_MODES: regression SGPM's modes before its vectors

// the node's shared memory: in s_spCoef (free in a node: TIMD's pass of an SGPM CU takes no vectors) the list of
// vectors (s_bvList, intra_bv.glsl) and from SG_LIST_AT the parts' entries as int16 (partition, part, entry); in
// s_dimdHist per mode of SGPM_BV_START + SGPM_NUM_BVS its slot among the modes the lists name (-1: none), per slot its
// mode, their number, the CU's DIMD mode, TIMD's two modes, the chosen pair, the number of vectors that are modes; in
// s_spCost the list's vectors' template costs, then the order of the cheapest, the vectors' LIC models, their
// fractions (x & 15 | ( y & 15 ) << 4: s_bvList holds their integer parts); in s_tmp each slot's template
// differences. In the workgroup's scratch (its blending sums, free in a node): per list entry its weighted SAD, per
// slot its SAD, per mode pair its cost, the list's vectors' template copies; with regression SGPM (its list's entries
// from SG_LIST_AT) per pair its cost and model, per slot its template prediction
const int SG_LIST_AT   = 2 * BV_LIST_MAX;
const int SG_SLOTS     = SGPM_BV_START + SGPM_NUM_BVS;
const int SG_MODES     = SG_SLOTS;
const int SG_NUM       = SG_MODES + SG_SLOTS;
const int SG_DIMD      = SG_NUM + 1;
const int SG_TIMD      = SG_DIMD + 1;
const int SG_CHOICE    = SG_TIMD + 2;
const int SG_NUM_BVS   = SG_CHOICE + 1;
const int SG_BV_ORDER  = BV_LIST_MAX;                    // (in s_spCost)
const int SG_LIC       = SG_BV_ORDER + SGPM_NUM_BVS;     // (in s_spCost: the list's LIC models' offsets, -1 none)
const int SG_FRAC      = SG_LIC + BV_LIST_MAX;           // (in s_spCost: the list's vectors' fractions)
const int SC_UNIT      = 0;
const int SC_SAD       = SC_UNIT + SGPM_PARTITIONS * 2 * SGPM_LIST_MAX;
const int SC_PAIR      = SC_SAD + SG_SLOTS;
// (per vector its template's W + H)
const int SC_COPY      = SC_PAIR + SGPM_PARTITIONS * SGPM_LIST_MAX * SGPM_LIST_MAX;
const int SC_RMODEL    = SC_PAIR + 66;                     // (regression: per pair of 12 entries 4 words; in SC_PAIR's)
const int SC_RPRED     = SC_COPY + BV_LIST_MAX * 128;      // (regression: per slot its template's W + H)

int  scGet( int o )        { return CBufI32( P.scratch ).v[int( gl_WorkGroupID.x ) * BLEND_ACC + o]; }
void scSet( int o, int v ) { CBufI32( P.scratch ).v[int( gl_WorkGroupID.x ) * BLEND_ACC + o] = v; }

int sgpmTable( int i )
{
  return BufI32( P.matrices ).v[trTab( TR_TAB_SGPM ) + i];
}

// getSgpmIntraMPMs' order of the neighbours (Cu::sgpm's records: 0 left, 1 above, 2 above-left, 3 above-right, 4
// below-left): the intra ones left and above, the inter ones there, the intra ones below-left, above-right and
// above-left, the inter ones there
const int SGPM_NBR_ORDER[10] = int[10]( 0, 1, 0, 1, 4, 3, 2, 4, 3, 2 );
const int SGPM_NBR_KIND[10]  = int[10]( 1, 1, 2, 2, 1, 1, 1, 2, 2, 2 );   // 1 intra, 2 inter or IBC

// the modes of part `part` of partition s of the SGPM CU of record rec (getSgpmIntraMPMs; 67-mode set): TIMD's
// horizontal and vertical modes (131-mode set; where angular), the partition's angle, the CU's DIMD mode, the
// neighbours' modes (those left and below-left where the part's template has a left arm, those above and above-right
// where it has an above arm), the first mode's opposite (32 away); each mode once, three at most, the last entry
// planar where fewer (even a repeated planar)
void sgpmList( int rec, int s, int part, int timdHor, int timdVer, int dimdMode, out int list[SGPM_LIST] )
{
  BufI32     td    = BufI32( P.timdData );
  const int  arms  = sgpmTable( SGPM_TAB_ARMS + s ) >> ( 2 * part );
  const bool left  = ( arms & 1 ) != 0, above = ( arms & 2 ) != 0;
  const int  kinds = td.v[rec + SGPM_HEAD_NBR_KIND];
  uint included[3] = uint[3]( 0u, 0u, 0u );
  int  n = 0;
  list = int[SGPM_LIST]( PLANAR_IDX, PLANAR_IDX, PLANAR_IDX );
  for( int step = 0; step < 15 && n < SGPM_LIST; step++ )
  {
    int m = -1;
    if( step < 2 )
    {
      const int timd = step == 0 ? timdHor : timdVer;
      m = timd > DC_IDX ? timdMode67( timd ) : -1;
    }
    else if( step == 2 )
    {
      m = sgpmTable( SGPM_TAB_ANGLE + s );
    }
    else if( step == 3 )
    {
      m = dimdMode;
    }
    else if( step < 14 )
    {
      const int  k    = SGPM_NBR_ORDER[step - 4];
      const bool side = k == 2 || ( k == 0 || k == 4 ? left : above );
      const bool kind = ( ( kinds >> ( 2 * k ) ) & 3 ) == SGPM_NBR_KIND[step - 4];
      m = side && kind ? cuModeOf( td.v[rec + SGPM_HEAD_NBR + k] ) : -1;
    }
    else
    {
      m = list[0] > DIA_IDX ? list[0] - 32 : list[0] + 32;
    }
    if( m >= 0 && !modeIn( included, m ) )
    {
      modeAdd( included, m );
      list[n++] = m;
    }
  }
}

// a LIC model of SGPM's list in Params::timdData (GpuIntraRec.cpp's BV_OUT_LIC records; a BvInfo's licScale /
// licShift / licOffset / mean): whether it has two, the first's scale, shift and offset, the second's, the mean between
const int SGPM_LIC_TWO = 0, SGPM_LIC_FIRST = 1, SGPM_LIC_SECOND = 4, SGPM_LIC_MEAN = 7, SGPM_LIC = 8;

// sample v through LIC model `at` (-1: none): linearTransform, with two models linearTransforms (the second above the
// mean), clipped
int sgpmLic( int v, int at, int maxPel )
{
  if( at < 0 )
  {
    return v;
  }
  BufI32    d = BufI32( P.timdData );
  const int m = at + ( d.v[at + SGPM_LIC_TWO] != 0 && v > d.v[at + SGPM_LIC_MEAN] ? SGPM_LIC_SECOND : SGPM_LIC_FIRST );
  const int scaled = d.v[m] * v, shift = d.v[m + 1];
  return clamp( ( shift >= 0 ? scaled >> shift : scaled << -shift ) + d.v[m + 2], 0, maxPel );
}

// whether LIC models at a and b are alike (BvInfo's operator==: none both, or every field equal)
bool sgpmLicSame( int a, int b )
{
  bool same = ( a < 0 ) == ( b < 0 );
  for( int k = 0; a >= 0 && b >= 0 && k < SGPM_LIC; k++ )
  {
    same = same && BufI32( P.timdData ).v[a + k] == BufI32( P.timdData ).v[b + k];
  }
  return same;
}

// the list's vector k in 1/16 samples
ivec2 sgpmMvAt( int k )
{
  const int f = int( s_spCost[SG_FRAC + k] );
  return ( bvListAt( k ) << 4 ) + ivec2( f & 15, f >> 4 );
}

// getItmpMergeCandidate's list for SGPM of CU c (invocation 0; the node of task `before`; c.bvList's record): the
// vectors of the CUs at its neighbour positions each in its order (getNeighBv's for SGPM, in 1/16 samples: their
// records' x 16, else those the records give SGPM's list; a CU's first with its LIC model), each once (with its model)
// where the block at it and its template's corner are decoded (validIBCItmpMv with SGPM's one-sample template, at the
// vector's integer part), the first BV_LIST_KEEP of them (DecCu's maxToKeep): the vectors' integer parts into
// s_bvList, their fractions and their models' offsets into s_spCost. Returns the length
int sgpmVectorList( Cu c, int before )
{
  BufI32    data = BufI32( P.timdData );
  CBufI32   recs = CBufI32( P.timdData );   // (records written by other workgroups' nodes)
  OwnerMap  om   = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  const int keep = data.v[c.bvList + BV_LIST_KEEP];
  int       n    = 0;
  for( int k = 0; k < BV_NBRS && n < keep; k++ )
  {
    const int src = data.v[c.bvList + BV_LIST_NBRS + k];
    const int at  = P.bvOut + src * BV_OUT;
    const int own = src >= 0 ? recs.v[at + BV_OUT_SGPM] : -1;   // (SGPM's own vectors, else -1)
    for( int j = 0; src >= 0 && j < ( own >= 0 ? own : recs.v[at] ) && n < keep; j++ )
    {
      const ivec2 mv  = own >= 0 ? ivec2( recs.v[at + BV_OUT_SGPM + 1 + 2 * j], recs.v[at + BV_OUT_SGPM + 2 + 2 * j] )
                                 : bvUnpack( recs.v[at + 1 + j] ) << 4;
      const ivec2 bv  = mv >> 4;
      const int   lic = j == 0 ? data.v[at + BV_OUT_LIC] : -1;
      bool        listed = false;
      for( int e = 0; e < n && !listed; e++ )
      {
        listed = sgpmMvAt( e ) == mv && sgpmLicSame( int( s_spCost[SG_LIC + e] ), lic );
      }
      if( !listed && decodedBefore( om, c.x + bv.x + c.w - 1, c.y + bv.y + c.h - 1, before )
          && decodedBefore( om, c.x + bv.x - 1, c.y + bv.y - 1, before ) )
      {
        s_bvList[2 * n]       = int16_t( bv.x );
        s_bvList[2 * n + 1]   = int16_t( bv.y );
        s_spCost[SG_FRAC + n] = uint( ( mv.x & 15 ) | ( ( mv.y & 15 ) << 4 ) );
        s_spCost[SG_LIC + n]  = uint( lic );
        n++;
      }
    }
  }
  return n;
}

// the template sample s (the above row's W, then the left column's H) of CU c predicted from the block at the list's
// vector k (getPredIBCBlk with the bilinear filter at the vector moved by the template's offset, the template's
// region at the CU's position), through its LIC model; for the node of block `before`
int sgpmCopyAt( Cu c, int k, int s, int bitDepth, int before )
{
  const bool top = s < c.w;
  const int  v   = ibcPredict( c.x, c.y, top ? c.w : 1, top ? 1 : c.h,
                               sgpmMvAt( k ) + ( top ? ivec2( 0, -16 ) : ivec2( -16, 0 ) ), true, top ? s : 0,
                               top ? 0 : s - c.w, before, bitDepth, ivec2( 0 ) );
  return sgpmLic( v, int( s_spCost[SG_LIC + k] ), ( 1 << bitDepth ) - 1 );
}

// regression SGPM's blend weight (of 32) of its second prediction at (x, y) relative to the CU by a model (p0, p1, p2,
// shift: AffineBlendingModel::compute, clipped to 1 .. 31, in 32 bits as the CPU's ints; its offset follows from shift)
int sgpmModelWeight( ivec4 model, int x, int y )
{
  const int offset = model.w != 0 ? 1 << ( model.w - 1 ) : 0;
  return clamp( ( model.x * x + model.y * y + model.z + offset ) >> model.w, 1, 31 );
}

// deriveSgpmBlending of regression SGPM CU c's template predictions in the scratch from pa (the pair's first mode, P0)
// and pb (its second, P1), one invocation: the second's weight linear in the position, fitted to the template (the row
// above at y = -1, the column left at x = -1) by solve1 with a CccmModel of 3 parameters - the features (P1 - P0) x,
// (P1 - P0) y and P1 - P0, the target the reconstruction minus P0, each a Pel; the dot products as cccmDotProduct sums
// them (in each full group of 8 samples a pair's two products added in 32 bits: madd), the matrix scaled to 28 bits
// (32 above 10 bits), regularized, eliminated with xGetDivScaleRoundShift's scales and the decimation bits (16, 22
// above 10 bits), substituted back -, its parameters as ints (the CPU's shiftA is 0: it sizes the model it was given,
// the initial one), its shift the decimation bits less 5; equal weights (16) where those at the CU's corners differ by
// 4 at most. Returns the SAD of the blend ( ( 32 - w ) P0 + w P1 + 16 ) >> 5 against the template's reconstruction
int sgpmRegression( Cu c, int pa, int pb, int bitDepth, out ivec4 model )
{
  const int W = c.w, T = c.w + c.h, full = T & ~7, decim = bitDepth > 10 ? 22 : 16;
  int64_t e[3][4];   // the normal equations' upper triangle and right-hand side, then the elimination's rows
  for( int i = 0; i < 12; i++ )
  {
    e[i / 4][i % 4] = 0;
  }
  int f[4], g[4] = int[4]( 0, 0, 0, 0 );   // (a sample's features and target, the previous sample's)
  for( int i = 0; i < T; i++ )
  {
    const bool top  = i < W;
    const int  p0   = scGet( SC_RPRED + pa + i ), d = scGet( SC_RPRED + pb + i ) - p0;
    const int  reco = top ? recoAt( 0, c.x + i, c.y - 1 ) : recoAt( 0, c.x - 1, c.y + i - W );
    f[0] = int( int16_t( d * ( top ? i : -1 ) ) );
    f[1] = int( int16_t( d * ( top ? -1 : i - W ) ) );
    f[2] = int( int16_t( d ) );
    f[3] = int( int16_t( reco - p0 ) );
    for( int r = 0; r < 3; r++ )
    {
      for( int col = r; col < 4; col++ )
      {
        if( i >= full || ( i & 1 ) != 0 )
        {
          e[r][col] += int64_t( ( i < full ? g[r] * g[col] : 0 ) + f[r] * f[col] );
        }
      }
    }
    g = f;
  }
  const int matrixShift = ( bitDepth > 10 ? 32 : 28 ) - 2 * bitDepth - cccmCeilLog2( T );
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
  const int64_t rnd = int64_t( 1 << ( decim - 1 ) );
  for( int piv = 0; piv < 3; piv++ )
  {
    int scale, round, shift;
    cccmDivScale( e[piv][piv] < int64_t( 1 ) ? int64_t( 1 ) : e[piv][piv], scale, round, shift );
    scale <<= decim - CCCM_DECIM;
    for( int j = piv + 1; j < 4; j++ )
    {
      e[piv][j] = ( e[piv][j] * int64_t( scale ) + int64_t( round ) ) >> shift;
    }
    for( int r = piv + 1; r < 3; r++ )
    {
      const int64_t q = e[r][piv];
      for( int j = piv + 1; j < 4; j++ )
      {
        e[r][j] -= ( q * e[piv][j] + rnd ) >> decim;
      }
    }
  }
  int64_t x[3];
  for( int r = 2; r >= 0; r-- )
  {
    x[r] = e[r][3];
    for( int j = r + 1; j < 3; j++ )
    {
      x[r] -= ( e[r][j] * x[j] + rnd ) >> decim;
    }
  }
  model = ivec4( int( x[0] ), int( x[1] ), int( x[2] ), decim - 5 );
  int lo = 31, hi = 1;
  for( int corner = 0; corner < 4; corner++ )
  {
    const int w = sgpmModelWeight( model, ( corner & 1 ) * ( W - 1 ), ( corner >> 1 ) * ( c.h - 1 ) );
    lo = min( lo, w );
    hi = max( hi, w );
  }
  model = hi - lo <= 4 ? ivec4( 0, 0, 16, 0 ) : model;
  int sad = 0;
  for( int i = 0; i < T; i++ )
  {
    const bool top  = i < W;
    const int  w    = sgpmModelWeight( model, top ? i : -1, top ? -1 : i - W );
    const int  reco = top ? recoAt( 0, c.x + i, c.y - 1 ) : recoAt( 0, c.x - 1, c.y + i - W );
    const int  pr   = ( ( 32 - w ) * scGet( SC_RPRED + pa + i ) + w * scGet( SC_RPRED + pb + i ) + 16 ) >> 5;
    sad += abs( int( int16_t( pr ) ) - reco );
  }
  return sad;
}

// regression SGPM's pair p (the CPU's order: the first entry ascending, then the second after it) of a list of L
// entries
ivec2 sgpmRegressionPair( int p, int L )
{
  int a = 0;
  while( p >= L - 1 - a )
  {
    p -= L - 1 - a;
    a++;
  }
  return ivec2( a, a + 1 + p );
}

// the node of SGPM CU c (all invocations, uniform control flow; after timdNode's pass, which wrote TIMD's two modes
// into its DimdOut, where DecCu derives them - else they are -1, initData's): its list's entry into its DimdOut
// (sgpmSplitDir, sgpmMode0, sgpmMode1, the vectors of the modes from vectors, regression SGPM's blend model), its
// partition's angle (regression SGPM: sgpmDimdMode, which DecCu leaves at initData's planar) into the record the MPM
// lists read of it, its vectors into its record for the lists of the CUs after it; returns (invocation 0) its first
// mode as the CU's (planar for a vector's: chroma DM and the lists read that). cuIndex: the CU's record; nbrMode,
// dimdModes: its MPM list's neighbours' modes and its DIMD's modes (deriveLumaMode's, invocation 0's)
int sgpmNode( Cu c, int cuIndex, int nbrMode[MPM_NEIGHBOURS], int dimdModes[3], int bitDepth, bool noPdpc, int lid,
              int NT )
{
  BufI32     td    = BufI32( P.timdData );
  CBufI32    o     = CBufI32( P.dimdOut );
  const int  rec   = c.sgpm, base = cuIndex * DIMD_OUT;
  const int  W     = c.w, H = c.h, T = W + H, maxPel = ( 1 << bitDepth ) - 1;
  const int  flags = td.v[rec + SGPM_HEAD_FLAGS];
  const bool regression = ( flags & SGPM_REGRESSION ) != 0;
  // its list of vectors (none without a list record: no CU around it has one)
  if( lid == 0 )
  {
    if( ( flags & SGPM_NO_TIMD ) != 0 )
    {
      o.v[base + DIMD_OUT_SGPM_TIMD]     = -1;
      o.v[base + DIMD_OUT_SGPM_TIMD + 1] = -1;
    }
    s_dimdHist[SG_DIMD]     = dimdModes[0];
    s_dimdHist[SG_TIMD]     = o.v[base + DIMD_OUT_SGPM_TIMD];
    s_dimdHist[SG_TIMD + 1] = o.v[base + DIMD_OUT_SGPM_TIMD + 1];
    s_dimdHist[SG_CHOICE]   = 0;
    s_dimdHist[SG_NUM_BVS]  = c.bvList >= 0 ? sgpmVectorList( c, g_ti ) : 0;
  }
  for( int m = lid; m < SG_SLOTS; m += NT )
  {
    s_dimdHist[m] = 0;
  }
  for( int k = lid; k < BV_LIST_MAX; k += NT )
  {
    s_spCost[k] = 0u;
  }
  SYNC();
  // the vectors' template costs (sadTM of their copies, which the scratch keeps for the modes' costs below), then the
  // cheapest SGPM_NUM_BVS in order (stableSortSmall), which become the modes from SGPM_BV_START on
  const int numVectors = s_dimdHist[SG_NUM_BVS];
  for( int i = lid; i < numVectors * T; i += NT )
  {
    const int s    = i % T;
    const int reco = s < W ? recoAt( 0, c.x + s, c.y - 1 ) : recoAt( 0, c.x - 1, c.y + s - W );
    const int copy = sgpmCopyAt( c, i / T, s, bitDepth, g_ti );
    scSet( SC_COPY + i, copy );
    atomicAdd( s_spCost[i / T], uint( abs( reco - copy ) ) );
  }
  tsSync();
  if( lid == 0 )
  {
    uint used = 0u;
    const int numBvs = min( numVectors, SGPM_NUM_BVS );
    for( int j = 0; j < numBvs; j++ )
    {
      int best = -1;
      for( int k = 0; k < numVectors; k++ )
      {
        best = ( used & ( 1u << k ) ) == 0u && ( best < 0 || s_spCost[k] < s_spCost[best] ) ? k : best;
      }
      used |= 1u << best;
      s_spCost[SG_BV_ORDER + j] = uint( best );
    }
    s_dimdHist[SG_NUM_BVS] = numBvs;
  }
  SYNC();
  // the parts' lists, one invocation each (three modes, then the vectors' modes), the modes they name marked; then
  // each named mode's slot. Regression SGPM's list instead (invocation 0): deriveSgpmModeOrdered's mpmList - planar,
  // getSpatialIpm (forced valid, without MDIP's exclusions: isModeExcluded false; the CU's DIMD modes, the
  // neighbours'), fillMPMList, RSGPM_MPMS entries - then the vectors' modes
  const int numBvs = s_dimdHist[SG_NUM_BVS], L = ( regression ? RSGPM_MPMS : SGPM_LIST ) + numBvs;
  if( regression && lid == 0 )
  {
    int  mpm[NUM_MPM];
    uint included[3] = uint[3]( 0u, 0u, 0u );
    mpmFill( min( mpmSpatial( c, nbrMode, dimdModes, true, mpm, included ), RSGPM_MPMS ), RSGPM_MPMS, mpm, included );
    for( int k = 0; k < L; k++ )
    {
      const int m = k < RSGPM_MPMS ? mpm[k] : SGPM_BV_START + k - RSGPM_MPMS;
      s_spCoef[SG_LIST_AT + k] = int16_t( m );
      s_dimdHist[m]            = 1;
    }
  }
  for( int u = lid; !regression && u < 2 * SGPM_PARTITIONS; u += NT )
  {
    int list[SGPM_LIST];
    sgpmList( rec, u >> 1, u & 1, s_dimdHist[SG_TIMD], s_dimdHist[SG_TIMD + 1], s_dimdHist[SG_DIMD], list );
    for( int k = 0; k < L; k++ )
    {
      const int m = k < SGPM_LIST ? list[k] : SGPM_BV_START + k - SGPM_LIST;
      s_spCoef[SG_LIST_AT + u * L + k] = int16_t( m );
      s_dimdHist[m]                    = 1;
    }
  }
  SYNC();
  if( lid == 0 )
  {
    int n = 0;
    for( int m = 0; m < SG_SLOTS; m++ )
    {
      if( s_dimdHist[m] != 0 )
      {
        s_dimdHist[SG_MODES + n] = m;
        s_dimdHist[m]            = n++;
      }
      else
      {
        s_dimdHist[m] = -1;
      }
    }
    s_dimdHist[SG_NUM] = n;
  }
  // the template's reference line (its barriers order the slots), DC's value
  const TemplateUnits t = templateUnitsAt( td, rec + SGPM_HEAD_T );
  timdFillReference( t, td, rec + SGPM_HEAD_T + TEMPLATE_UNITS, 0, c.x, c.y, W, H, TIMD_REF_SHIFT, bitDepth, lid, NT );
  const int dc = timdDc( t, W, H, lid, NT );

  // the named modes' template differences, as many modes at once as s_tmp holds (the above row's W samples, then the
  // left column's H; a vector's mode its copy), then each mode's SAD and each list entry's SAD weighted by its
  // partition (one invocation each), into the scratch; with regression SGPM the predictions themselves
  const int numModes = s_dimdHist[SG_NUM], perPass = MAX_REF / T, tplWeights = td.v[rec + SGPM_HEAD_WEIGHTS];
  const int numUnits = 2 * SGPM_PARTITIONS * L;
  for( int first = 0; first < numModes; first += perPass )
  {
    const int n = min( perPass, numModes - first );
    for( int i = lid; i < n * T; i += NT )
    {
      const int s = i % T, mode = s_dimdHist[SG_MODES + first + i / T];
      const int reco = s < W ? recoAt( 0, c.x + s, c.y - 1 ) : recoAt( 0, c.x - 1, c.y + s - W );
      int       pred;
      if( mode >= SGPM_BV_START )
      {
        pred = scGet( SC_COPY + int( s_spCost[SG_BV_ORDER + mode - SGPM_BV_START] ) * T + s );
      }
      else
      {
        const int        m131 = timdMode131( mode );
        const TimdParams p    = timdParamsOf( m131, sgpmWideAngle( W, H, m131 ), W, H, noPdpc );
        pred = timdPredict( m131, p, s < W ? 1 + s : 0, s < W ? 0 : 1 + s - W, W, H, 1, 1, dc, maxPel );
      }
      if( regression )
      {
        scSet( SC_RPRED + first * T + i, pred );
      }
      else
      {
        s_tmp[i] = int16_t( abs( reco - pred ) );
      }
    }
    SYNC();
    for( int k = lid; !regression && k < n + numUnits; k += NT )
    {
      if( k < n )
      {
        int sum = 0;
        for( int i = 0; i < T; i++ )
        {
          sum += int( s_tmp[k * T + i] );
        }
        scSet( SC_SAD + first + k, sum );
      }
      else
      {
        const int u = k - n, slot = s_dimdHist[int( s_spCoef[SG_LIST_AT + u] )];
        if( slot >= first && slot < first + n )
        {
          const int w0  = tplWeights + ( u / ( 2 * L ) ) * T;
          int       sum = 0;
          for( int i = 0; i < T; i++ )
          {
            sum += int( s_tmp[( slot - first ) * T + i] ) * BufI32( P.matrices ).v[w0 + i];
          }
          scSet( SC_UNIT + u, sum );
        }
      }
    }
    tsSync();
  }

  // a pair's cost: its first mode's weighted SAD plus its second's SAD less its weighted one (a second part's entry
  // turned into the latter); a pair of one mode twice takes none (the largest cost). Regression SGPM: per pair of its
  // list's entries the SAD of their regression's blend and the model (sgpmRegression, one invocation each)
  for( int p = lid; regression && p < L * ( L - 1 ) / 2; p += NT )
  {
    const ivec2 ab = sgpmRegressionPair( p, L );
    ivec4       model;
    const int   cost = sgpmRegression( c, s_dimdHist[int( s_spCoef[SG_LIST_AT + ab.x] )] * T,
                                       s_dimdHist[int( s_spCoef[SG_LIST_AT + ab.y] )] * T, bitDepth, model );
    scSet( SC_PAIR + p, cost );
    for( int k = 0; k < 4; k++ )
    {
      scSet( SC_RMODEL + 4 * p + k, model[k] );
    }
  }
  for( int u = lid; !regression && u < numUnits; u += NT )
  {
    if( ( ( u / L ) & 1 ) != 0 )
    {
      scSet( SC_UNIT + u, scGet( SC_SAD + s_dimdHist[int( s_spCoef[SG_LIST_AT + u] )] ) - scGet( SC_UNIT + u ) );
    }
  }
  tsSync();
  const int numPairs = regression ? L * ( L - 1 ) / 2 : SGPM_PARTITIONS * L * L;
  for( int i = lid; !regression && i < numPairs; i += NT )
  {
    const int s  = i / ( L * L );
    const int u0 = s * 2 * L + ( i / L ) % L;   // (the first part's entry)
    const int u1 = s * 2 * L + L + i % L;       // (the second part's)
    const bool distinct = s_spCoef[SG_LIST_AT + u0] != s_spCoef[SG_LIST_AT + u1];
    scSet( SC_PAIR + i, distinct ? scGet( SC_UNIT + u0 ) + scGet( SC_UNIT + u1 ) : 0x7fffffff );
  }
  tsSync();
  // the list's entry: the pair of that rank (by cost, then by the CPU's order of the pairs); a pair counts the cheaper
  // ones until it knows it ranks after the entry
  const int idx = td.v[rec + SGPM_HEAD_IDX];
  for( int i = lid; i < numPairs; i += NT )
  {
    const int cost = scGet( SC_PAIR + i );
    int       rank = 0;
    for( int j = 0; j < numPairs && cost != 0x7fffffff && rank <= idx; j++ )
    {
      const int other = scGet( SC_PAIR + j );
      rank += other < cost || ( other == cost && j < i ) ? 1 : 0;
    }
    if( cost != 0x7fffffff && rank == idx )
    {
      s_dimdHist[SG_CHOICE] = i;
    }
  }
  SYNC();
  if( lid != 0 )
  {
    return -1;
  }
  const int   i  = s_dimdHist[SG_CHOICE], s = regression ? 0 : i / ( L * L );
  const ivec2 ab = regression ? sgpmRegressionPair( i, L ) : ivec2( s * 2 * L + ( i / L ) % L, s * 2 * L + L + i % L );
  const int   mode0 = int( s_spCoef[SG_LIST_AT + ab.x] ), mode1 = int( s_spCoef[SG_LIST_AT + ab.y] );
  o.v[base + DIMD_OUT_SGPM_SPLIT]    = s;
  o.v[base + DIMD_OUT_SGPM_MODE]     = mode0;
  o.v[base + DIMD_OUT_SGPM_MODE + 1] = mode1;
  for( int k = 0; k < 4; k++ )
  {
    o.v[base + DIMD_OUT_SGPM_MODEL + k] = regression ? scGet( SC_RMODEL + 4 * i + k ) : 0;
  }
  CBufI32( P.cuMode ).v[td.v[rec + SGPM_HEAD_VIEW]] = regression ? PLANAR_IDX : sgpmTable( SGPM_TAB_ANGLE + s );
  // the vectors of its modes from vectors (1/16 samples) with their LIC models for its block, the vectors in its record
  // for the lists after it (getNeighBv's SGPM branch: the first mode's, then the second's; without their models;
  // rounded to integers, changePrecision's rounding, and as they are for SGPM's lists)
  ivec2 bvs[3] = ivec2[3]( ivec2( 0 ), ivec2( 0 ), ivec2( 0 ) ), mvs[2] = ivec2[2]( ivec2( 0 ), ivec2( 0 ) );
  int   num = 0;
  for( int k = 0; k < 2; k++ )
  {
    const int   m     = k == 0 ? mode0 : mode1;
    const int   entry = m >= SGPM_BV_START ? int( s_spCost[SG_BV_ORDER + m - SGPM_BV_START] ) : -1;
    const ivec2 mv    = entry >= 0 ? sgpmMvAt( entry ) : ivec2( 0 );
    o.v[base + DIMD_OUT_SGPM_BV + 2 * k]     = mv.x;
    o.v[base + DIMD_OUT_SGPM_BV + 2 * k + 1] = mv.y;
    o.v[base + DIMD_OUT_SGPM_LIC + k]        = entry >= 0 ? int( s_spCost[SG_LIC + entry] ) : -1;
    if( entry >= 0 )
    {
      mvs[num] = mv;
      bvs[num] = ( mv + mix( ivec2( 8 ), ivec2( 7 ), greaterThanEqual( mv, ivec2( 0 ) ) ) ) >> 4;
      num++;
    }
  }
  if( c.bvList >= 0 )
  {
    bvStore( c, num, bvs );
    bvStoreSgpm( c, num, mvs );
  }
  return mode0 >= SGPM_BV_START ? PLANAR_IDX : mode0;
}

// ----- an SGPM CU's luma block

// the first mode's weight (of 32) per sample of SGPM block t (all invocations, uniform control flow, ending with a
// barrier), row-major into the workgroup's scratch from SGPM_BLOCK_W: its node's partition's (xWeightedSgpm's walk of
// g_geoWeights with the size's blending width, without blending its first: in matrices), regression SGPM's 32 less the
// second's by the model its node chose (weightAffineBlk)
void sgpmBlockWeights( Tu t, int lid, int NT )
{
  CBufI32     o       = CBufI32( P.dimdOut );
  const int   base    = t.cu * DIMD_OUT, rec = BufCu( P.cus ).v[t.cu].sgpm;
  const bool  rsgpm   = ( BufI32( P.timdData ).v[rec + SGPM_HEAD_FLAGS] & SGPM_REGRESSION ) != 0;
  const int   weights = BufI32( P.timdData ).v[rec + SGPM_HEAD_WEIGHTS + 1]
                        + o.v[base + DIMD_OUT_SGPM_SPLIT] * t.w * t.h;
  const ivec4 model   = ivec4( o.v[base + DIMD_OUT_SGPM_MODEL], o.v[base + DIMD_OUT_SGPM_MODEL + 1],
                               o.v[base + DIMD_OUT_SGPM_MODEL + 2], o.v[base + DIMD_OUT_SGPM_MODEL + 3] );
  for( int i = lid; i < t.w * t.h; i += NT )
  {
    scSet( SGPM_BLOCK_W + i,
           rsgpm ? 32 - sgpmModelWeight( model, i % t.w, i / t.w ) : BufI32( P.matrices ).v[weights + i] );
  }
  tsSync();
}

// whether SGPM block t's SPS has no DIMD (its block's deriveIpmForTransform then derives nothing)
bool sgpmNoDimd( Tu t )
{
  return ( BufI32( P.timdData ).v[BufCu( P.cus ).v[t.cu].sgpm + SGPM_HEAD_FLAGS] & SGPM_NO_DIMD ) != 0;
}

// the modes of SGPM block t's transforms (after mipTransformMode's barrier, which has the DIMD of its prediction in its
// CU's DimdOut): x the partition's angle, which explicit MTS takes (TrQuant::getTrTypes); y the LFNST / NSPT mode
// (getFinalIntraModeForTransform's SGPM branch, JVET-AK0222): the DIMD's main mode (dimdDerivedIntraDir), in a CU below
// SGPM_VIPM_SIZE samples where it is within 2 of either of the CU's two modes, else its second mode where that is,
// else the angle. Regression SGPM: sgpmDimdMode (initData's planar: DecCu never sets it) for MTS, the DIMD's main mode
// for the angle
ivec2 sgpmTransformModes( Tu t )
{
  CBufI32    o      = CBufI32( P.dimdOut );
  const int  base   = t.cu * DIMD_OUT;
  const bool rsgpm  = ( BufI32( P.timdData ).v[BufCu( P.cus ).v[t.cu].sgpm + SGPM_HEAD_FLAGS] & SGPM_REGRESSION ) != 0;
  const int  first  = o.v[base + DIMD_OUT_SGPM_PRED], second = o.v[base + DIMD_OUT_SGPM_PRED + 1];
  const int  angle  = rsgpm ? first : sgpmTable( SGPM_TAB_ANGLE + o.v[base + DIMD_OUT_SGPM_SPLIT] );
  const int  mts    = rsgpm ? PLANAR_IDX : angle;
  const int  mode0  = o.v[base + DIMD_OUT_SGPM_MODE], mode1 = o.v[base + DIMD_OUT_SGPM_MODE + 1];
  if( t.w * t.h >= SGPM_VIPM_SIZE )
  {
    return ivec2( mts, first );
  }
  const bool firstFar  = abs( first - mode0 ) > 2 && abs( first - mode1 ) > 2;     // SGPM_VIPM_TH
  const bool secondFar = abs( second - mode0 ) > 2 && abs( second - mode1 ) > 2;
  return ivec2( mts, firstFar && secondFar ? angle : firstFar ? second : first );
}

#endif
