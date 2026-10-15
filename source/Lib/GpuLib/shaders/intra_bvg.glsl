// BVG-DIMD (block-vector-guided DIMD, JVET-AL0108) on the GPU, in a BVG-DIMD CU's node after IntraTMP's search for it
// (intra_tmp.glsl's TMP_SEARCH_BVG, whose refined list it takes on): the vectors that cost no more than 1.2 times the
// cheapest (m_bvListSorted) and their fusion weights (convertDiff2Weight), then deriveBvgDimdMode - the DIMD histogram
// of the reconstruction at the vectors (the blocks at overlapping vectors merged into one area while it stays small
// and its last sample decoded), its two strongest modes and their weights, planar added - into the CU's DimdOut in
// DIMD's layout, so that its blocks blend as a DIMD block does (DERIVE_DIMD with DERIVE_BVG; planar's prediction the
// vectors' fusion, bvgFusionSample). Included by intra_rec.comp after intra_dimd.glsl and intra_tmp.glsl.
#ifndef INTRA_BVG_GLSL
#define INTRA_BVG_GLSL

shared ivec4 s_bvgArea[BVG_NUM_REFINE];   // the histogram's areas: x, y of the first sample, width, height
shared int   s_bvgAreas;

// convertDiff2Weight: weights out of 64 for num costs (cost[i] 0 taken as 1), each by how much less than the others'
// sum it costs (DIMD's division by table; the excess off the last weights, the shortfall onto the first); one alone 64.
// The products wrap in 32 bits as the CPU's int ones
void bvgWeights( int cost[BVG_NUM_REFINE], int num, out int w[BVG_NUM_REFINE] )
{
  for( int i = 0; i < BVG_NUM_REFINE; i++ )
  {
    w[i] = 0;
  }
  if( num <= 1 )
  {
    w[0] = 64;
    return;
  }
  int d[BVG_NUM_REFINE];
  int sumW = 0;
  for( int i = 0; i < num; i++ )
  {
    d[i] = max( cost[i], 1 );
    sumW += d[i];
  }
  for( int i = 0; i < num; i++ )
  {
    d[i] = sumW - d[i];
  }
  const int sumDiff = ( num - 1 ) * sumW;
  int       x       = findMSB( sumDiff );
  const int norm    = ( sumDiff << 4 >> x ) & 15, v = GRAD_DIV_TABLE[norm] | 8;
  x += norm != 0 ? 1 : 0;
  const int shift = x + 3, add = 1 << ( shift - 1 );
  int sum = 0;
  for( int j = 0; j < num; j++ )
  {
    w[j] = ( d[j] * v * 64 + add ) >> shift;
    sum += w[j];
  }
  while( sum > 64 )
  {
    for( int j = num - 1; j >= 0 && sum > 64; j-- )
    {
      if( w[j] != 0 )
      {
        w[j]--;
        sum--;
      }
    }
  }
  while( sum < 64 )
  {
    for( int j = 0; j < num && sum < 64; j++ )
    {
      w[j]++;
      sum++;
    }
  }
}

// the node of BVG-DIMD CU c (record cuIndex; all invocations, uniform control flow) once tmpNode has refined its
// vectors: its vectors, their weights and its blend into its DimdOut (DIMD_OUT_BVG_*, DIMD's fields), its first vector
// into its record of vectors (pu.bv); returns (invocation 0) its first mode (the CU's intraDir), in second its second
// entry (bvgDimdMode[1], MTSS's first try: planar where one mode blends with planar). A CU whose histogram is empty
// (the CPU's modes -1, which no decodable stream has) predicts planar
int bvgNode( Cu c, int cuIndex, out int second, int lid, int NT )
{
  CBufI32    o  = CBufI32( P.dimdOut );
  const int  at = cuIndex * DIMD_OUT;
  OwnerMap   om = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  const int  W = c.w, H = c.h;
  // the list, its weights, the areas of the histogram (invocation 0)
  if( lid == 0 )
  {
    ivec2 bv[BVG_NUM_REFINE];
    int   cost[BVG_NUM_REFINE];
    int   num = 0;
    for( int i = 0; i < tlSize( TS_REFINE ) && i < BVG_NUM_REFINE; i++ )
    {
      const int ci = tlAt( TS_REFINE, i, TE_COST );
      if( int64_t( ci ) * 5 > int64_t( tlAt( TS_REFINE, 0, TE_COST ) ) * 6 )   // (above 1.2 times the cheapest)
      {
        break;
      }
      bv[num]   = ivec2( tlAt( TS_REFINE, i, TE_X ), tlAt( TS_REFINE, i, TE_Y ) );
      cost[num] = ci;
      num++;
    }
    int w[BVG_NUM_REFINE];
    bvgWeights( cost, num, w );
    o.v[at + DIMD_OUT_BVG_NUM] = num;
    for( int i = 0; i < BVG_NUM_REFINE; i++ )
    {
      o.v[at + DIMD_OUT_BVG_BV + i]     = i < num ? bvPack( bv[i] ) : 0;
      o.v[at + DIMD_OUT_BVG_WEIGHT + i] = i < num ? w[i] : 0;
    }
    bvStore( c, 1, ivec2[3]( num > 0 ? bv[0] : ivec2( 0 ), ivec2( 0 ), ivec2( 0 ) ) );
    // each vector's block (but where an earlier vector's area took it in): with those of later vectors that overlap
    // its area while their bounding box stays within 2 blocks' samples plus one per vector taken and its last sample
    // is decoded; the histogram reads its inside (buildHistogram's 3x3 gradients)
    bool taken[BVG_NUM_REFINE] = bool[BVG_NUM_REFINE]( false, false, false, false, false );
    int  areas = 0;
    for( int a = 0; a < num; a++ )
    {
      if( taken[a] )
      {
        continue;
      }
      ivec2 tl = bv[a], br = bv[a] + ivec2( W - 1, H - 1 );
      int   maxSize = 2 * W * H;
      for( int b = a + 1; b < num; b++ )
      {
        const ivec2 ntl = bv[b], nbr = bv[b] + ivec2( W - 1, H - 1 );
        if( taken[b] || br.x < ntl.x || tl.x > nbr.x || br.y < ntl.y || tl.y > nbr.y )
        {
          continue;
        }
        const ivec2 utl = min( tl, ntl ), ubr = max( br, nbr );
        if( ( ubr.x - utl.x + 1 ) * ( ubr.y - utl.y + 1 ) <= maxSize
            && decodedBefore( om, c.x + ubr.x, c.y + ubr.y, g_ti ) )
        {
          taken[b] = true;
          maxSize += W * H;
          tl = utl;
          br = ubr;
        }
      }
      s_bvgArea[areas++] = ivec4( c.x + tl.x + 1, c.y + tl.y + 1, br.x - tl.x - 1, br.y - tl.y - 1 );
    }
    s_bvgAreas = areas;
  }
  for( int i = lid; i < NUM_LUMA_MODE; i += NT )
  {
    s_dimdHist[i] = 0;
  }
  SYNC();
  // the histogram (buildHistogram with MTSS's subsampling: every second sample both ways from MIN_MTSS_SIZE on)
  for( int a = 0; a < s_bvgAreas; a++ )
  {
    const ivec4 area = s_bvgArea[a];
    const int   step = area.z * area.w >= MIN_MTSS_SIZE ? 2 : 1;
    const int   nx = ( area.z + step - 1 ) / step, ny = ( area.w + step - 1 ) / step;
    for( int i = lid; i < nx * ny; i += NT )
    {
      const ivec2 g = dimdGradient( 0, area.x + ( i % nx ) * step, area.y + ( i / nx ) * step, 0 );
      if( g.x != 0 || g.y != 0 )
      {
        atomicAdd( s_dimdHist[dimdAngle( g )], abs( g.x ) + abs( g.y ) );
      }
    }
  }
  SYNC();
  second = -1;
  if( lid != 0 )
  {
    return -1;
  }
  // the two strongest modes (the first of equal amplitudes), their weights: one mode 43 and planar 21, two by their
  // amplitudes out of 32 (DIMD's division by table; the excess off the second, the shortfall onto the first that
  // takes it) and planar 32
  int amp[2] = int[2]( 0, 0 ), best[2] = int[2]( 0, 0 );
  for( int m = 0; m < NUM_LUMA_MODE; m++ )
  {
    const int h = s_dimdHist[m];
    if( h > amp[0] )
    {
      amp[1]  = amp[0];
      best[1] = best[0];
      amp[0]  = h;
      best[0] = m;
    }
    else if( h > amp[1] )
    {
      amp[1]  = h;
      best[1] = m;
    }
  }
  int mode[3] = int[3]( PLANAR_IDX, -1, -1 ), weight[3] = int[3]( 64, 0, 0 );
  if( best[0] > 0 && best[1] <= 0 )
  {
    mode   = int[3]( best[0], PLANAR_IDX, -1 );
    weight = int[3]( 64 - 21, 21, 0 );
  }
  else if( best[0] > 0 )
  {
    const int sumWeight = 32, s1 = amp[0] + amp[1];
    int       x         = findMSB( s1 );
    const int norm      = ( s1 << 4 >> x ) & 15, v = GRAD_DIV_TABLE[norm] | 8;
    x += norm != 0 ? 1 : 0;
    const int shift = x + 3, add = 1 << ( shift - 1 );
    int w[2];
    for( int i = 0; i < 2; i++ )
    {
      w[i] = int( ( uint64_t( amp[i] ) * uint64_t( v * sumWeight ) + uint64_t( add ) ) >> shift );
      w[i] = amp[i] == 0 ? 0 : min( w[i], sumWeight );
    }
    int d = w[0] + w[1] - sumWeight;
    for( int i = 1; d > 0 && i >= 0; i-- )   // (each takes off up to the excess it saw, while it has weight)
    {
      const int take = min( d, w[i] );
      w[i] -= take;
      d    -= take;
    }
    for( int i = 0; d < 0 && i < 2; i++ )   // (the first with weight whose sum with the shortfall stays below 32)
    {
      if( w[i] > 0 && w[i] + d < sumWeight )
      {
        w[i] -= d;
        d     = 0;
      }
    }
    mode   = int[3]( best[0], best[1], PLANAR_IDX );
    weight = int[3]( w[0], w[1], 32 );
  }
  // DIMD's layout: the main mode, planar's weight, the second mode as the first blend mode (others planar, weight 0),
  // no location dependence; the transforms' second mode bvgDimdMode[1]
  o.v[at + DIMD_OUT_MODE]       = mode[0];
  o.v[at + DIMD_OUT_BLENDING]   = 1;
  o.v[at + DIMD_OUT_REL_WEIGHT] = weight[0];
  o.v[at + DIMD_OUT_REL_WEIGHT + 1] = mode[1] == PLANAR_IDX ? weight[1] : weight[2];
  for( int k = 0; k < 7; k++ )
  {
    const bool blended = k == 0 && mode[1] > PLANAR_IDX;
    o.v[at + DIMD_OUT_BLEND_MODE + k]    = blended ? mode[1] : PLANAR_IDX;
    o.v[at + DIMD_OUT_REL_WEIGHT + k + 2] = blended ? weight[1] : 0;
    o.v[at + DIMD_OUT_LOC_DEP + k]       = 0;
  }
  o.v[at + DIMD_OUT_TR_SECOND] = mode[1];
  o.v[at + DIMD_OUT_BV_FLAG]   = 0;
  o.v[at + DIMD_OUT_BV]        = 0;
  second = mode[1];
  return mode[0];
}

// planar's prediction of a BVG-DIMD CU's block at sample (x, y) of the picture: its CU's vectors' fusion
// (tmpFusionBlending: the reconstruction at each by its weight out of 64, rounded)
int bvgFusionSample( int cu, int x, int y )
{
  CBufI32   d   = CBufI32( P.dimdOut );
  const int o   = cu * DIMD_OUT;
  int       sum = 32;
  for( int i = 0; i < d.v[o + DIMD_OUT_BVG_NUM]; i++ )
  {
    const ivec2 bv = bvUnpack( d.v[o + DIMD_OUT_BVG_BV + i] );
    sum += d.v[o + DIMD_OUT_BVG_WEIGHT + i] * recoAt( 0, x + bv.x, y + bv.y );
  }
  return sum >> 6;
}

#endif
