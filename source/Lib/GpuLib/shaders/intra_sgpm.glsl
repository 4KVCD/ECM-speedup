// SGPM (spatial geometric partitioning mode, JVET-AB0155, with JVET-AJ0107's partitions) on the GPU. An SGPM CU's node
// (DecCu's SGPM branch) has its DIMD (intra_dimd.glsl) and TIMD's best horizontal and vertical modes (intra_timd.glsl's
// timdNode), then derives IntraPrediction::deriveSgpmModeOrdered without regression SGPM (the host leaves those CUs to
// the CPU): its list of block vectors (getItmpMergeCandidate's for SGPM, the vectors of the CUs around it valid for its
// one-sample template; integer ones: the host takes no CU whose list may hold another) ordered by their template costs,
// the cheapest of them modes from SGPM_BV_START on; per partition of g_sgpmSplitDir the lists of its two parts, three
// modes (PU::getSgpmIntraMPMs) and those vectors; the template costs of every mode they name - its prediction of the
// template (predTimdIntraAng with SGPM's wide angles, a vector's copy of the reconstruction), its SAD against the
// reconstruction (sadTM) and per partition that SAD weighted by the partition's template weights (sgpmSadTM) - and of
// each partition's mode pairs (the first mode's weighted SAD plus the second's SAD less its weighted one), then the
// entry of the list those costs rank (updateCandList: ascending, the first of equal costs first) that the CU's index
// selects. Its block predicts the two modes (a vector's: its copy) and blends them by the partition's weights
// (xWeightedSgpm, in intra_rec.comp's blending), and takes its transforms by a DIMD of that blend. Included by
// intra_rec.comp after intra_tmp.glsl.
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

// the node's shared memory: in s_spCoef (free in a node: TIMD's pass of an SGPM CU takes no vectors) the list of
// vectors (s_bvList, intra_bv.glsl) and from SG_LIST_AT the parts' entries as int16 (partition, part, entry); in
// s_dimdHist per mode of SGPM_BV_START + SGPM_NUM_BVS its slot among the modes the lists name (-1: none), per slot its
// mode, their number, the CU's DIMD mode, TIMD's two modes, the chosen pair, the number of vectors that are modes; in
// s_spCost the list's vectors' template costs, then the order of the cheapest; in s_tmp each slot's template
// differences. In the workgroup's scratch (its blending sums, free in a node): per list entry its weighted SAD, per
// slot its SAD, per mode pair its cost
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
const int SC_UNIT      = 0;
const int SC_SAD       = SC_UNIT + SGPM_PARTITIONS * 2 * SGPM_LIST_MAX;
const int SC_PAIR      = SC_SAD + SG_SLOTS;

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

// getItmpMergeCandidate's list for SGPM of CU c (invocation 0; the node of task `before`; c.bvList's record): the
// vectors of the CUs at its neighbour positions each in its order (their records', which are SGPM's in integer
// samples; a CU's first with its LIC model), each once (with its model) where the block at it and its template's
// corner are decoded (validIBCItmpMv with SGPM's one-sample template), the first BV_LIST_KEEP of them (DecCu's
// maxToKeep): the vectors into s_bvList, their models' offsets into s_spCost. Returns the length
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
    for( int j = 0; src >= 0 && j < recs.v[at] && n < keep; j++ )
    {
      const ivec2 bv  = bvUnpack( recs.v[at + 1 + j] );
      const int   lic = j == 0 ? data.v[at + BV_OUT_LIC] : -1;
      bool        listed = false;
      for( int e = 0; e < n && !listed; e++ )
      {
        listed = bvListAt( e ) == bv && sgpmLicSame( int( s_spCost[SG_LIC + e] ), lic );
      }
      if( !listed && decodedBefore( om, c.x + bv.x + c.w - 1, c.y + bv.y + c.h - 1, before )
          && decodedBefore( om, c.x + bv.x - 1, c.y + bv.y - 1, before ) )
      {
        s_bvList[2 * n]     = int16_t( bv.x );
        s_bvList[2 * n + 1] = int16_t( bv.y );
        s_spCost[SG_LIC + n] = uint( lic );
        n++;
      }
    }
  }
  return n;
}

// the template sample s (the above row's W, then the left column's H) of CU c copied from the block at the list's
// vector k (getPredIBCBlk at the vector moved by the template's offset), through its LIC model
int sgpmCopyAt( Cu c, int k, int s, int maxPel )
{
  const ivec2 bv = bvListAt( k );
  const int   v  = s < c.w ? recoAt( 0, c.x + s + bv.x, c.y - 1 + bv.y )
                           : recoAt( 0, c.x - 1 + bv.x, c.y + s - c.w + bv.y );
  return sgpmLic( v, int( s_spCost[SG_LIC + k] ), maxPel );
}

// the node of SGPM CU c (all invocations, uniform control flow; after timdNode's pass, which wrote TIMD's two modes
// into its DimdOut): its list's entry into its DimdOut (sgpmSplitDir, sgpmMode0, sgpmMode1, the vectors of the modes
// from vectors), its partition's angle into the record the MPM lists read of it, its vectors into its record for the
// lists of the CUs after it; returns (invocation 0) its first mode as the CU's (planar for a vector's: chroma DM and
// the lists read that). cuIndex: the CU's record; dimdMode: its DIMD's mode (invocation 0's)
int sgpmNode( Cu c, int cuIndex, int dimdMode, int bitDepth, bool noPdpc, int lid, int NT )
{
  BufI32    td   = BufI32( P.timdData );
  CBufI32   o    = CBufI32( P.dimdOut );
  const int rec  = c.sgpm, base = cuIndex * DIMD_OUT;
  const int W    = c.w, H = c.h, T = W + H, maxPel = ( 1 << bitDepth ) - 1;
  // its list of vectors (none without a list record: no CU around it has one)
  if( lid == 0 )
  {
    s_dimdHist[SG_DIMD]     = dimdMode;
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
  // the vectors' template costs (sadTM of their copies), then the cheapest SGPM_NUM_BVS in order (stableSortSmall),
  // which become the modes from SGPM_BV_START on
  const int numVectors = s_dimdHist[SG_NUM_BVS];
  for( int i = lid; i < numVectors * T; i += NT )
  {
    const int s = i % T;
    const int reco = s < W ? recoAt( 0, c.x + s, c.y - 1 ) : recoAt( 0, c.x - 1, c.y + s - W );
    atomicAdd( s_spCost[i / T], uint( abs( reco - sgpmCopyAt( c, i / T, s, maxPel ) ) ) );
  }
  SYNC();
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
  // each named mode's slot
  const int numBvs = s_dimdHist[SG_NUM_BVS], L = SGPM_LIST + numBvs;
  for( int u = lid; u < 2 * SGPM_PARTITIONS; u += NT )
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
  // partition (one invocation each), into the scratch
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
        pred = sgpmCopyAt( c, int( s_spCost[SG_BV_ORDER + mode - SGPM_BV_START] ), s, maxPel );
      }
      else
      {
        const int        m131 = timdMode131( mode );
        const TimdParams p    = timdParamsOf( m131, sgpmWideAngle( W, H, m131 ), W, H, noPdpc );
        pred = timdPredict( m131, p, s < W ? 1 + s : 0, s < W ? 0 : 1 + s - W, W, H, 1, 1, dc, maxPel );
      }
      s_tmp[i] = int16_t( abs( reco - pred ) );
    }
    SYNC();
    for( int k = lid; k < n + numUnits; k += NT )
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
  // turned into the latter); a pair of one mode twice takes none (the largest cost)
  for( int u = lid; u < numUnits; u += NT )
  {
    if( ( ( u / L ) & 1 ) != 0 )
    {
      scSet( SC_UNIT + u, scGet( SC_SAD + s_dimdHist[int( s_spCoef[SG_LIST_AT + u] )] ) - scGet( SC_UNIT + u ) );
    }
  }
  tsSync();
  const int numPairs = SGPM_PARTITIONS * L * L;
  for( int i = lid; i < numPairs; i += NT )
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
  const int i = s_dimdHist[SG_CHOICE], s = i / ( L * L );
  const int mode0 = int( s_spCoef[SG_LIST_AT + s * 2 * L + ( i / L ) % L] );
  const int mode1 = int( s_spCoef[SG_LIST_AT + s * 2 * L + L + i % L] );
  o.v[base + DIMD_OUT_SGPM_SPLIT]    = s;
  o.v[base + DIMD_OUT_SGPM_MODE]     = mode0;
  o.v[base + DIMD_OUT_SGPM_MODE + 1] = mode1;
  CBufI32( P.cuMode ).v[td.v[rec + SGPM_HEAD_VIEW]] = sgpmTable( SGPM_TAB_ANGLE + s );
  // the vectors of its modes from vectors with their LIC models for its block, the vectors in its record for the
  // lists after it (getNeighBv's SGPM branch: the first mode's, then the second's; without their models)
  ivec2 bvs[3];
  int   num = 0;
  for( int k = 0; k < 2; k++ )
  {
    const int m     = k == 0 ? mode0 : mode1;
    const int entry = m >= SGPM_BV_START ? int( s_spCost[SG_BV_ORDER + m - SGPM_BV_START] ) : -1;
    o.v[base + DIMD_OUT_SGPM_BV + k]  = entry >= 0 ? bvPack( bvListAt( entry ) ) : 0;
    o.v[base + DIMD_OUT_SGPM_LIC + k] = entry >= 0 ? int( s_spCost[SG_LIC + entry] ) : -1;
    if( entry >= 0 )
    {
      bvs[num++] = bvListAt( entry );
    }
  }
  if( c.bvList >= 0 )
  {
    bvStore( c, num, bvs );
  }
  return mode0 >= SGPM_BV_START ? PLANAR_IDX : mode0;
}

// ----- an SGPM CU's luma block

// the offset in matrices of the weights of the partition SGPM block t's node chose: the first mode's weight (of 32) per
// sample, row-major (xWeightedSgpm's walk of g_geoWeights with the size's blending width)
int sgpmBlockWeights( Tu t )
{
  const int split = CBufI32( P.dimdOut ).v[t.cu * DIMD_OUT + DIMD_OUT_SGPM_SPLIT];
  return BufI32( P.timdData ).v[BufCu( P.cus ).v[t.cu].sgpm + SGPM_HEAD_WEIGHTS + 1] + split * t.w * t.h;
}

// the modes of SGPM block t's transforms (after mipTransformMode's barrier, which has the DIMD of its prediction in its
// CU's DimdOut): x the partition's angle, which explicit MTS takes (TrQuant::getTrTypes); y the LFNST / NSPT mode
// (getFinalIntraModeForTransform's SGPM branch, JVET-AK0222): the DIMD's main mode (dimdDerivedIntraDir), in a CU below
// SGPM_VIPM_SIZE samples where it is within 2 of either of the CU's two modes, else its second mode where that is,
// else the angle
ivec2 sgpmTransformModes( Tu t )
{
  CBufI32   o      = CBufI32( P.dimdOut );
  const int base   = t.cu * DIMD_OUT;
  const int angle  = sgpmTable( SGPM_TAB_ANGLE + o.v[base + DIMD_OUT_SGPM_SPLIT] );
  const int first  = o.v[base + DIMD_OUT_SGPM_PRED], second = o.v[base + DIMD_OUT_SGPM_PRED + 1];
  const int mode0  = o.v[base + DIMD_OUT_SGPM_MODE], mode1 = o.v[base + DIMD_OUT_SGPM_MODE + 1];
  if( t.w * t.h >= SGPM_VIPM_SIZE )
  {
    return ivec2( angle, first );
  }
  const bool firstFar  = abs( first - mode0 ) > 2 && abs( first - mode1 ) > 2;     // SGPM_VIPM_TH
  const bool secondFar = abs( second - mode0 ) > 2 && abs( second - mode1 ) > 2;
  return ivec2( angle, firstFar && secondFar ? angle : firstFar ? second : first );
}

#endif
