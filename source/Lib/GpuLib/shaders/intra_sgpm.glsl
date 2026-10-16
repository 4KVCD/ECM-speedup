// SGPM (spatial geometric partitioning mode, JVET-AB0155, with JVET-AJ0107's partitions) on the GPU. An SGPM CU's node
// (DecCu's SGPM branch) has its DIMD (intra_dimd.glsl) and TIMD's best horizontal and vertical modes (intra_timd.glsl's
// timdNode), then derives IntraPrediction::deriveSgpmModeOrdered without block vectors or regression SGPM (the host
// leaves those CUs to the CPU): per partition of g_sgpmSplitDir the lists of three modes of its two parts
// (PU::getSgpmIntraMPMs), the template costs of every mode they name - its prediction of the one-sample template
// (predTimdIntraAng with SGPM's wide angles), its SAD against the reconstruction (sadTM) and per partition that SAD
// weighted by the partition's template weights (sgpmSadTM) - and of each partition's mode pairs (the first mode's
// weighted SAD plus the second's SAD less its weighted one), then the entry of the list those costs rank
// (updateCandList: ascending, the first of equal costs first) that the CU's index selects. Its block predicts the two
// modes and blends them by the partition's weights (xWeightedSgpm, in intra_rec.comp's blending), and takes its
// transforms by a DIMD of that blend. Included by intra_rec.comp after intra_timd.glsl.
#ifndef INTRA_SGPM_GLSL
#define INTRA_SGPM_GLSL

// SGPM's tables (in matrices at trTab( TR_TAB_SGPM ); GpuIntraRec.cpp's sgpmTables): per partition the mode of its
// angle (g_geoAngle2IntraAng), then which arms its parts' templates have (g_geoTmShape: bit 0 the first part's left
// arm, bit 1 its above arm, bits 2 and 3 the second part's)
const int SGPM_PARTITIONS = 26;   // SGPM_TOTAL_NUM_PARTITIONS
const int SGPM_TAB_ANGLE  = 0;
const int SGPM_TAB_ARMS   = SGPM_PARTITIONS;
const int SGPM_LIST       = 3;                                          // SGPM_NUM_MPM: a part's modes
const int SGPM_UNITS      = SGPM_PARTITIONS * 2 * SGPM_LIST;            // the lists' entries (partition, part, entry)
const int SGPM_CANDS      = SGPM_PARTITIONS * SGPM_LIST * SGPM_LIST;    // the mode pairs (partition, first, second)
const int SGPM_VIPM_SIZE  = 256;   // MAX_SGPM_VIPM_SIZE: the CUs below it whose transforms check the DIMD's modes

// the node's shared memory: the lists' entries as int16 in s_spCoef (free in a node without block vectors); in
// s_dimdHist per mode of the 67 its slot among the modes the lists name (-1: none), per slot its mode, their number,
// the CU's DIMD mode, TIMD's two modes and the chosen pair; in s_spCost per list entry its cost, then per slot its SAD;
// in s_tmp each slot's template differences, then the pairs' costs (two int16 each)
#define s_sgpmList s_spCoef
const int SG_MODES  = NUM_LUMA_MODE;
const int SG_NUM    = SG_MODES + NUM_LUMA_MODE;
const int SG_DIMD   = SG_NUM + 1;
const int SG_TIMD   = SG_DIMD + 1;
const int SG_CHOICE = SG_TIMD + 2;
const int SG_SAD    = SGPM_UNITS;

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

// a pair's cost in s_tmp (two int16)
void sgpmStoreCost( int i, int cost )
{
  s_tmp[2 * i]     = int16_t( cost & 0xffff );
  s_tmp[2 * i + 1] = int16_t( cost >> 16 );
}

int sgpmCost( int i )
{
  return ( int( s_tmp[2 * i + 1] ) << 16 ) | ( int( s_tmp[2 * i] ) & 0xffff );
}

// the node of SGPM CU c (all invocations, uniform control flow; after timdNode's pass, which wrote TIMD's two modes
// into its DimdOut): its list's entry into its DimdOut (sgpmSplitDir, sgpmMode0, sgpmMode1) and its partition's angle
// into the record the MPM lists read of it; returns (invocation 0) its first mode, which chroma DM reads. cuIndex: the
// CU's record; dimdMode: its DIMD's mode (invocation 0's)
int sgpmNode( Cu c, int cuIndex, int dimdMode, int bitDepth, bool noPdpc, int lid, int NT )
{
  BufI32    td   = BufI32( P.timdData );
  CBufI32   o    = CBufI32( P.dimdOut );
  const int rec  = c.sgpm, base = cuIndex * DIMD_OUT;
  const int W    = c.w, H = c.h, T = W + H, maxPel = ( 1 << bitDepth ) - 1;
  if( lid == 0 )
  {
    s_dimdHist[SG_DIMD]     = dimdMode;
    s_dimdHist[SG_TIMD]     = o.v[base + DIMD_OUT_SGPM_TIMD];
    s_dimdHist[SG_TIMD + 1] = o.v[base + DIMD_OUT_SGPM_TIMD + 1];
    s_dimdHist[SG_CHOICE]   = 0;
  }
  for( int m = lid; m < NUM_LUMA_MODE; m += NT )
  {
    s_dimdHist[m] = 0;
  }
  SYNC();
  // the parts' lists, one invocation each, the modes they name marked; then each named mode's slot
  for( int u = lid; u < 2 * SGPM_PARTITIONS; u += NT )
  {
    int list[SGPM_LIST];
    sgpmList( rec, u >> 1, u & 1, s_dimdHist[SG_TIMD], s_dimdHist[SG_TIMD + 1], s_dimdHist[SG_DIMD], list );
    for( int k = 0; k < SGPM_LIST; k++ )
    {
      s_sgpmList[u * SGPM_LIST + k] = int16_t( list[k] );
      s_dimdHist[list[k]]           = 1;
    }
  }
  SYNC();
  if( lid == 0 )
  {
    int n = 0;
    for( int m = 0; m < NUM_LUMA_MODE; m++ )
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
  // left column's H), then each mode's SAD and each list entry's SAD weighted by its partition (one invocation each)
  const int numModes = s_dimdHist[SG_NUM], perPass = MAX_REF / T, tplWeights = td.v[rec + SGPM_HEAD_WEIGHTS];
  for( int first = 0; first < numModes; first += perPass )
  {
    const int n = min( perPass, numModes - first );
    for( int i = lid; i < n * T; i += NT )
    {
      const int        s    = i % T, mode = timdMode131( s_dimdHist[SG_MODES + first + i / T] );
      const TimdParams p    = timdParamsOf( mode, sgpmWideAngle( W, H, mode ), W, H, noPdpc );
      const bool       top  = s < W;
      const int        pred = timdPredict( mode, p, top ? 1 + s : 0, top ? 0 : 1 + s - W, W, H, 1, 1, dc, maxPel );
      const int        reco = top ? recoAt( 0, c.x + s, c.y - 1 ) : recoAt( 0, c.x - 1, c.y + s - W );
      s_tmp[i] = int16_t( abs( reco - pred ) );
    }
    SYNC();
    for( int k = lid; k < n + SGPM_UNITS; k += NT )
    {
      if( k < n )
      {
        int sum = 0;
        for( int i = 0; i < T; i++ )
        {
          sum += int( s_tmp[k * T + i] );
        }
        s_spCost[SG_SAD + first + k] = uint( sum );
      }
      else
      {
        const int u = k - n, slot = s_dimdHist[int( s_sgpmList[u] )];
        if( slot >= first && slot < first + n )
        {
          const int w0  = tplWeights + ( u / ( 2 * SGPM_LIST ) ) * T;
          int       sum = 0;
          for( int i = 0; i < T; i++ )
          {
            sum += int( s_tmp[( slot - first ) * T + i] ) * BufI32( P.matrices ).v[w0 + i];
          }
          s_spCost[u] = uint( sum );
        }
      }
    }
    SYNC();
  }

  // a pair's cost: its first mode's weighted SAD plus its second's SAD less its weighted one (a second part's entry
  // turned into the latter); pairs of one mode twice take none (the largest cost)
  for( int u = lid; u < SGPM_UNITS; u += NT )
  {
    if( ( ( u / SGPM_LIST ) & 1 ) != 0 )
    {
      s_spCost[u] = uint( int( s_spCost[SG_SAD + s_dimdHist[int( s_sgpmList[u] )]] ) - int( s_spCost[u] ) );
    }
  }
  SYNC();
  for( int i = lid; i < SGPM_CANDS; i += NT )
  {
    const int s  = i / ( SGPM_LIST * SGPM_LIST );
    const int u0 = s * 2 * SGPM_LIST + ( i / SGPM_LIST ) % SGPM_LIST;   // (the first part's entry)
    const int u1 = s * 2 * SGPM_LIST + SGPM_LIST + i % SGPM_LIST;       // (the second part's)
    sgpmStoreCost( i, s_sgpmList[u0] != s_sgpmList[u1] ? int( s_spCost[u0] ) + int( s_spCost[u1] ) : 0x7fffffff );
  }
  SYNC();
  // the list's entry: the pair of that rank (by cost, then by the CPU's order of the pairs)
  const int idx = td.v[rec + SGPM_HEAD_IDX];
  for( int i = lid; i < SGPM_CANDS; i += NT )
  {
    const int cost = sgpmCost( i );
    int       rank = 0;
    for( int j = 0; j < SGPM_CANDS && cost != 0x7fffffff; j++ )
    {
      const int other = sgpmCost( j );
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
  const int i = s_dimdHist[SG_CHOICE], s = i / ( SGPM_LIST * SGPM_LIST );
  const int mode0 = int( s_sgpmList[s * 2 * SGPM_LIST + ( i / SGPM_LIST ) % SGPM_LIST] );
  const int mode1 = int( s_sgpmList[s * 2 * SGPM_LIST + SGPM_LIST + i % SGPM_LIST] );
  o.v[base + DIMD_OUT_SGPM_SPLIT]    = s;
  o.v[base + DIMD_OUT_SGPM_MODE]     = mode0;
  o.v[base + DIMD_OUT_SGPM_MODE + 1] = mode1;
  CBufI32( P.cuMode ).v[td.v[rec + SGPM_HEAD_VIEW]] = sgpmTable( SGPM_TAB_ANGLE + s );
  return mode0;
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
