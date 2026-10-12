// OBIC (occurrence-based intra coding, JVET-AH0076, with JVET-AK0056's weights) on the GPU, in an OBIC CU's node: the
// histogram of occurrence of the modes of the CU's neighbours (IntraPrediction::obicNeighbours chose them), each
// counted with the samples of its CU, per location dependence class (histoLocDep); then deriveObicMode's end (the
// strongest modes but planar, their classes and weights, planar added) into Params::obicOut. Included by intra_rec.comp
// after intra_dimd.glsl: the histogram takes s_dimdHist once the CU's DIMD is selected.
#ifndef INTRA_OBIC_GLSL
#define INTRA_OBIC_GLSL

// the counts deriveObicMode adds to its histogram directly, without a location dependence class (an intra merge CU's
// modes): they add to the totals, not to the classes
shared int s_obicPlain[NUM_LUMA_MODE];

// histoLocDep[mode][cls] += amount (in s_dimdHist: class-major, as DIMD's three histograms). An OBIC neighbour's mode
// can have class -1 (obicSelect): the CPU indexes histoLocDep[mode][-1] then, which in its row-major [mode][3] array is
// histoLocDep[mode - 1][2] (such a mode is above DC)
void obicCount( int mode, int cls, int amount )
{
  if( amount != 0 )
  {
    atomicAdd( s_dimdHist[cls >= 0 ? cls * NUM_LUMA_MODE + mode : 2 * NUM_LUMA_MODE + mode - 1], amount );
  }
}

// deriveObicMode's step 4 over the neighbours of OBIC CU c, one invocation each, all invocations (ends with a barrier).
// A DIMD CU counts its main mode, planar and its blend modes, those but the main one by their weights relative to the
// main one's; an OBIC CU its modes, each with all the samples; an intra merge CU its modes, each with all the samples,
// without a class (s_obicPlain); another intra CU its mode (MIP CUs, which count nothing in intra slices, are not in
// the list)
void obicHistogram( Cu c, int lid, int NT )
{
  for( int i = lid; i < 3 * NUM_LUMA_MODE; i += NT )
  {
    s_dimdHist[i] = 0;
  }
  for( int i = lid; WITH_MERGE && i < NUM_LUMA_MODE; i += NT )
  {
    s_obicPlain[i] = 0;
  }
  SYNC();
  const ObicNbr nb = lid < OBIC_NBRS ? BufObicNbr( P.obicNbrs ).v[c.obic * OBIC_NBRS + lid] : ObicNbr( -1, 0 );
  if( nb.cu >= 0 )
  {
    const int kind = BufCu( P.cus ).v[nb.cu].kind;
    const int ns   = nb.numSamples;
    if( kind == CU_DIMD )   // its node's DimdOut
    {
      CBufI32    d        = CBufI32( P.dimdOut );
      const int  o        = nb.cu * DIMD_OUT;
      const int  mode     = d.v[o + DIMD_OUT_MODE];
      const int  w0       = d.v[o + DIMD_OUT_REL_WEIGHT];
      const int  w1       = d.v[o + DIMD_OUT_REL_WEIGHT + 1];
      const bool blending = d.v[o + DIMD_OUT_BLENDING] != 0;
      if( mode >= 0 && mode < NUM_LUMA_MODE )
      {
        obicCount( mode, d.v[o + DIMD_OUT_LOC_DEP], ns );
      }
      if( w1 > 0 )
      {
        obicCount( PLANAR_IDX, 0, ns * w1 / w0 );
      }
      for( int k = 0; k < DIMD_FUSION_NUM - 2 && blending; k++ )
      {
        const int m = d.v[o + DIMD_OUT_BLEND_MODE + k], w = d.v[o + DIMD_OUT_REL_WEIGHT + k + 2];
        if( m >= 0 && m < NUM_LUMA_MODE && w > 0 )
        {
          obicCount( m, d.v[o + DIMD_OUT_LOC_DEP + k + 1], ns * w / w0 );
        }
      }
    }
    else if( WITH_MERGE && kind == CU_MERGE )   // its merge parameters (its DimdOut slot), all samples to each mode, no class
    {
      CBufI32   d        = CBufI32( P.dimdOut );
      const int o        = nb.cu * DIMD_OUT;
      const int mainMode = d.v[o + DIMD_OUT_MODE];
      if( mainMode >= 0 && mainMode < NUM_LUMA_MODE )
      {
        atomicAdd( s_obicPlain[mainMode], ns );
      }
      for( int k = 0; k < DIMD_FUSION_NUM - 2; k++ )
      {
        const int m = d.v[o + DIMD_OUT_BLEND_MODE + k];
        if( m >= 0 && m < NUM_LUMA_MODE && d.v[o + DIMD_OUT_REL_WEIGHT + k + 2] > 0 )
        {
          atomicAdd( s_obicPlain[m], ns );
        }
      }
    }
    else if( kind == CU_OBIC )   // its node's OBIC (always blended)
    {
      CBufI32   b = CBufI32( P.obicOut );
      const int o = BufCu( P.cus ).v[nb.cu].obic * OBIC_OUT;
      obicCount( b.v[o + OBIC_OUT_MODE], b.v[o + OBIC_OUT_LOC_DEP], ns );
      for( int k = 1; k < OBIC_MODES; k++ )
      {
        const int m = b.v[o + OBIC_OUT_MODE + k];
        if( m >= 0 && b.v[o + OBIC_OUT_WEIGHT + k] > 0 )
        {
          obicCount( m, b.v[o + OBIC_OUT_LOC_DEP + k], ns );
        }
      }
    }
    else   // its intra mode (firstPU->intraDir[0])
    {
      obicCount( cuModeOf( nb.cu ), 0, ns );
    }
  }
  SYNC();
}

// deriveObicMode's steps 5 and 6 for OBIC CU c, one invocation, from its histogram (obicHistogram) and its DIMD (d,
// whose main mode and, with planar blended, first blend mode count half, keeping OBIC apart from DIMD): the six
// strongest modes but planar (the first of equal counts), each in its dominant class; their weights out of 64 less
// planar's in proportion to the counts (DIMD's division by table), the excess taken off the last modes or the
// shortfall put on the first one that takes it, then planar after the modes with a weight. Writes the CU's OBIC and
// returns its first mode. (A stream with OBIC where no neighbour gives a mode cannot be decoded, the CPU's mode being
// -1: planar keeps the GPU's in range.)
int obicSelect( Cu c, DimdOut d )
{
  int amp[OBIC_MODES], best[OBIC_MODES];
  for( int i = 0; i < OBIC_MODES; i++ )
  {
    amp[i]  = 0;
    best[i] = -1;
  }
  for( int m = PLANAR_IDX + 1; m < NUM_LUMA_MODE; m++ )
  {
    int h = s_dimdHist[m] + s_dimdHist[NUM_LUMA_MODE + m] + s_dimdHist[2 * NUM_LUMA_MODE + m] + ( WITH_MERGE ? s_obicPlain[m] : 0 );
    h >>= m == d.mode ? 1 : 0;
    h >>= d.blending != 0 && m == d.blendMode[0] && d.relWeight[1] > 0 ? 1 : 0;
    for( int j = 0; j < OBIC_MODES; j++ )
    {
      if( h > amp[j] )
      {
        for( int k = OBIC_MODES - 1; k > j; k-- )
        {
          amp[k]  = amp[k - 1];
          best[k] = best[k - 1];
        }
        amp[j]  = h;
        best[j] = m;
        break;
      }
    }
  }
  int mode[OBIC_MODES], weight[OBIC_MODES], locDep[OBIC_MODES];
  int count = 0;
  for( int i = 0; i < OBIC_MODES; i++ )
  {
    mode[i]   = best[i];
    weight[i] = 0;
    locDep[i] = 0;
    count += best[i] >= 0 ? 1 : 0;
    if( best[i] > DC_IDX )   // the class with the most of the mode's count (the first of equal ones; -1 where the
    {                        // mode has only counts without a class, s_obicPlain)
      int top = 0, cls = -1;
      for( int j = 0; j < 3; j++ )
      {
        const int v = s_dimdHist[j * NUM_LUMA_MODE + best[i]];
        cls = v > top ? j : cls;
        top = max( top, v );
      }
      locDep[i] = cls;
    }
  }
  const int planarWeight = count == 1 ? 21 : 64 / 4;
  if( best[0] < 0 )
  {
    mode[0]   = PLANAR_IDX;
    weight[0] = 64;
  }
  else if( best[1] < 0 )   // one mode: blended with planar
  {
    weight[0] = 64 - planarWeight;
    mode[1]   = PLANAR_IDX;
    weight[1] = planarWeight;
    locDep[1] = 0;
  }
  else
  {
    const int sumWeight = 64 - planarWeight;
    if( count == OBIC_MODES )   // planar takes the last place
    {
      amp[OBIC_MODES - 1] = 0;
    }
    int s1 = 0;
    for( int i = 0; i < OBIC_MODES; i++ )
    {
      amp[i] *= 10;
      s1 += amp[i];
    }
    int x = findMSB( s1 );
    const int normS1 = ( s1 << 4 >> x ) & 15, v = GRAD_DIV_TABLE[normS1] | 8;
    x += normS1 != 0 ? 1 : 0;
    const int shift = x + 3;
    const uint64_t add = uint64_t( 1 ) << ( shift - 1 );
    int sum = 0;
    for( int i = 0; i < OBIC_MODES; i++ )
    {
      weight[i] = amp[i] == 0 ? 0 : min( int( ( uint64_t( amp[i] ) * uint64_t( v ) * uint64_t( sumWeight ) + add ) >> shift ), sumWeight );
      sum += weight[i];
    }
    int diff = sum - sumWeight;
    for( int i = OBIC_MODES - 1; i >= 0 && diff > 0; i-- )   // the excess off the last modes, each down to 0
    {
      const int take = min( weight[i], diff );
      weight[i] -= take;
      diff -= take;
    }
    for( int i = 0; i < OBIC_MODES && diff < 0; i++ )   // the shortfall onto the first weighted mode it keeps below the sum
    {
      if( weight[i] > 0 && weight[i] + diff < sumWeight )
      {
        weight[i] -= diff;
        diff = 0;
      }
    }
    count = 0;
    for( int i = 0; i < OBIC_MODES; i++ )
    {
      count += weight[i] > 0 ? 1 : 0;
    }
    const int p = min( count, OBIC_MODES - 1 );
    mode[p]   = PLANAR_IDX;
    weight[p] = planarWeight;
    locDep[p] = 0;
  }
  CBufI32   o    = CBufI32( P.obicOut );
  const int base = c.obic * OBIC_OUT;
  for( int i = 0; i < OBIC_MODES; i++ )
  {
    o.v[base + OBIC_OUT_MODE + i]    = mode[i];
    o.v[base + OBIC_OUT_WEIGHT + i]  = weight[i];
    o.v[base + OBIC_OUT_LOC_DEP + i] = locDep[i];
  }
  return mode[0];
}

#endif
