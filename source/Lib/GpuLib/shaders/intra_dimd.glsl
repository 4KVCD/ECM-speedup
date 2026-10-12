// DIMD (decoder-side intra mode derivation) on the GPU, in a CU's node: the histograms of the gradient angles of the
// reconstruction around the CU (buildHistogram / buildHistogramAdaptive with the arguments the host gives,
// IntraPrediction::dimdHistogramCalls / dimdChromaHistogramCalls), then a luma CU's modes, blending and weights as
// deriveDimdMode / deriveDimdModeAdaptive choose them (dimdSelect) or DIMD chroma's mode (dimdChromaSelect); and the
// integer arithmetic of DIMD's blending. Included by intra_rec.comp after its shared memory, Params P, recoAt and
// intra_modes.glsl.
#ifndef INTRA_DIMD_GLSL
#define INTRA_DIMD_GLSL

const int DIMD_FUSION_NUM       = 8;
const int DIMD_FUSION_NUM_SMALL = 6;
const int GRAD_DIV_TABLE[16]  = int[16]( 0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0 );   // g_gradDivTable
const int GRAD_ANG_TABLE[17]  = int[17]( 0, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864,
                                         40960, 47104, 53248, 59392, 65536 );
const int DIMD_LUT[64]        = int[64]( 16, 15, 14, 14, 13, 12, 11, 11, 10, 9, 9, 8, 8, 7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4,
                                         4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
const int DIMD_WEIGHT_TABLE[12] = int[12]( 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1 );   // buildHistogramAdaptive's weightTable
const int DIMD_MAX_OUTER        = 12;   // the template lines buildHistogramAdaptive scans (MAX_DIMD_TEMPLATE_SIZE)

shared int s_dimdHist[3 * NUM_LUMA_MODE];   // DimdCall::histogram's sums (left, above, above-left)
shared int s_dimdLo[DIMD_MAX_OUTER];        // buildHistogramAdaptive's sample order (dimdOuterRanges)
shared int s_dimdHi[DIMD_MAX_OUTER];
shared int s_dimdStart[DIMD_MAX_OUTER + 1];
shared int s_dimdSg[33];                    // per subgroup (at least 8 invocations each): a batch's prefix sums

// DIMD's 3x3 Sobel gradient (iDx, iDy) from the eight samples around a position: the row above (tl, t, tr), left and
// right (l, r), the row below (bl, b, br)
ivec2 sobel3x3( int tl, int t, int tr, int l, int r, int bl, int b, int br )
{
  return ivec2( bl + 2 * b + br - tl - 2 * t - tr, tl + 2 * l + bl - tr - 2 * r - br );
}

// the gradient (iDx, iDy) of component comp's reconstruction at (x, y): DIMD's 3x3 Sobel operator, or its 2x2 one
// (filterSizeIdx 1, JVET-AJ0203)
ivec2 dimdGradient( int comp, int x, int y, int filterSizeIdx )
{
  if( filterSizeIdx == 1 )
  {
    const int a = recoAt( comp, x - 1, y - 1 ), b = recoAt( comp, x, y - 1 ), c = recoAt( comp, x - 1, y ), d = recoAt( comp, x, y );
    return ivec2( c + d - a - b, a + c - b - d );
  }
  return sobel3x3( recoAt( comp, x - 1, y - 1 ), recoAt( comp, x, y - 1 ), recoAt( comp, x + 1, y - 1 ),
                   recoAt( comp, x - 1, y ), recoAt( comp, x + 1, y ),
                   recoAt( comp, x - 1, y + 1 ), recoAt( comp, x, y + 1 ), recoAt( comp, x + 1, y + 1 ) );
}

// the region of a gradient with both components nonzero (mapXgrY1 / mapXgrY0)
int dimdRegion( ivec2 g )
{
  const int signx = g.x < 0 ? 1 : 0, signy = g.y < 0 ? 1 : 0;
  const bool gtY  = abs( g.x ) > abs( g.y );
  return gtY ? ( signy == signx ? 1 : 0 ) : ( signy == signx ? 2 : 3 );
}

// buildHistogram's angle (the LUT ratio of JVET-X0149 and gradAngleIdx)
int dimdAngle( ivec2 g )
{
  if( g.x == 0 || g.y == 0 )
  {
    return g.x == 0 ? VER_IDX : HOR_IDX;
  }
  const int  region = dimdRegion( g );
  const int  absx = abs( g.x ), absy = abs( g.y );
  const bool gtY  = absx > absy;
  const int  s0 = gtY ? absy : absx, s1 = gtY ? absx : absy;
  int x = findMSB( s1 );
  const int norm = ( s1 << 4 >> x ) & 15, v = GRAD_DIV_TABLE[norm] | 8;
  x += norm != 0 ? 1 : 0;
  const int shift = 13 - x;
  const int ratio = shift < 0 ? ( s0 * v + ( 1 << ( -shift - 1 ) ) ) >> -shift : ( s0 * v ) << shift;
  int below = 0;   // gradAngleIdx: the nearest entry of the table, the upper one on ties, 16 above it
  for( int i = 1; i < 17; i++ )
  {
    below += ratio > GRAD_ANG_TABLE[i] ? 1 : 0;
  }
  int idx = 16;
  if( below < 16 )
  {
    const int i = below + 1;
    idx = ratio - GRAD_ANG_TABLE[i - 1] < GRAD_ANG_TABLE[i] - ratio ? i - 1 : i;
  }
  return ( region < 2 ? HOR_IDX : VER_IDX ) + ( ( region & 1 ) != 0 ? idx : -idx );
}

// buildHistogramAdaptive's angle (the logarithmic ratio and dimdLut)
int dimdAngleAdaptive( ivec2 g )
{
  if( g.x == 0 || g.y == 0 )
  {
    return g.x == 0 ? VER_IDX : HOR_IDX;
  }
  const int  region = dimdRegion( g );
  const int  absx = abs( g.x ), absy = abs( g.y );
  const bool gtY  = absx > absy;
  const int  s0 = gtY ? absy : absx, s1 = gtY ? absx : absy;
  const int  bs1 = findMSB( s1 ), bs0 = findMSB( s0 );
  const int  offset1 = bs1 > 0 ? 1 << ( bs1 - 1 ) : 0, offset0 = bs0 > 0 ? 1 << ( bs0 - 1 ) : 0;
  const int  ls1 = ( bs1 << 6 ) + ( ( ( s1 << 6 ) + offset1 ) >> bs1 );   // bitShift 3 + 3
  const int  ls0 = ( bs0 << 6 ) + ( ( ( s0 << 6 ) + offset0 ) >> bs0 );
  const int  idx1 = min( ( ls1 - ls0 + 4 ) >> 3, 63 );
  return ( region < 2 ? HOR_IDX : VER_IDX ) + ( ( region & 1 ) != 0 ? DIMD_LUT[idx1] : -DIMD_LUT[idx1] );
}

// buildHistogram over the samples of call k, all invocations: integer sums, in any order
void dimdHistogram( DimdCall k, int lid, int NT )
{
  for( int i = lid; i < k.height * k.width; i += NT )
  {
    const int y = i / k.width, x = i % k.width;
    if( k.direction == 3 && x == k.width - 1 && y == k.height - 1 )
    {
      continue;
    }
    const ivec2 g = dimdGradient( k.comp, k.x + x, k.y + y, k.filterSizeIdx );
    if( g.x != 0 || g.y != 0 )
    {
      atomicAdd( s_dimdHist[k.histogram * NUM_LUMA_MODE + dimdAngle( g )], abs( g.x ) + abs( g.y ) );
    }
  }
}

// buildHistogramAdaptive's sample order: the outer loop over the template's columns (left, direction 1) / rows (above,
// direction 2) from the CU outwards, the inner one along each over [s_dimdLo[o], s_dimdHi[o]) (the groups' sizes and
// the corner's availability around each group boundary); s_dimdStart[o]: the samples before outer index o
int dimdOuterCount( DimdCall k, int cuW, int cuH )
{
  const int sides = k.filterSizeIdx == 0 ? 2 : 1;
  const int maxTemplate = ( k.direction == 1 ? cuW : cuH ) <= 8 ? 8 - sides : DIMD_MAX_OUTER - sides;
  return max( 0, min( ( k.direction == 1 ? k.width : k.height ) - sides, maxTemplate ) );
}

void dimdOuterRanges( DimdCall k, int numOuter )
{
  const int so = ( k.filterSizeIdx == 0 ? 2 : 1 ) >> 1;
  int start = 0;
  for( int o = 0; o < numOuter; o++ )
  {
    const int i = o >> 2;
    int hi = k.sizeExt[i] - so, lo = k.extra[i] == 0 ? 1 : 0;
    if( o > 0 && ( o % 4 ) == 0 )
    {
      hi = max( k.sizeExt[i - 1] - so, hi );
      lo = lo != 0 || k.extra[i - 1] == 0 ? 1 : 0;
    }
    else if( ( o % 4 ) == 3 )
    {
      hi = max( k.sizeExt[i + 1] - so, hi );
      lo = lo != 0 || k.extra[i + 1] == 0 ? 1 : 0;
    }
    s_dimdLo[o]    = lo;
    s_dimdHi[o]    = hi;
    s_dimdStart[o] = start;
    start += max( 0, hi - lo );
  }
  s_dimdStart[numOuter] = start;
}

// the position of the sample at outer index o, inner position j (direction 1: currX = width - 1 - o - sideOffset,
// currY = j), relative to pReco
ivec2 dimdAdaptivePos( DimdCall k, int o, int j )
{
  const int so = ( k.filterSizeIdx == 0 ? 2 : 1 ) >> 1;
  return k.direction == 1 ? ivec2( k.width - 1 - o - so, j ) : ivec2( j, k.height - 1 - o - so );
}

// the weighted amplitude of a sample with gradient g and angle ang at outer index o, position p: 0 for the angles
// between horizontal and vertical beyond the CU's side
int dimdAdaptiveAmp( DimdCall k, ivec2 g, int ang, int o, ivec2 p, int cuW, int cuH )
{
  const bool beyond = k.direction == 1 ? p.y > cuH : p.x > cuW;
  return beyond && ang > HOR_IDX && ang < VER_IDX ? 0 : ( ( abs( g.x ) + abs( g.y ) ) * DIMD_WEIGHT_TABLE[o] + 2 ) >> 2;
}

// buildHistogramAdaptive over call k, all invocations. A CU 64 or more high (left) / wide (above) compares each
// sample of the first column / row with the gradient one further out, whose angle counts if it lies in the region of
// the last angle computed (an earlier sample's, carried along): the gradients of that column / row first, in parallel
// (into s_spCoef, free in a node: the amplitude, angle, regions and the outer gradient's angle at 0, 256, 512, 768),
// then one invocation carries the region along. The sums stop at the first sample that brings the total to maxAmp:
// per batch of samples in the CPU's order a prefix sum of the amplitudes decides which count
void dimdHistogramAdaptive( DimdCall k, int cuW, int cuH, int lid, int NT )
{
  const int numOuter = dimdOuterCount( k, cuW, cuH );
  if( lid == 0 )
  {
    dimdOuterRanges( k, numOuter );
  }
  SYNC();
  const bool firstInOrder = numOuter > 0 && ( k.direction == 1 ? cuH : cuW ) >= 64;
  if( firstInOrder )
  {
    const int lo = s_dimdLo[0], n = s_dimdHi[0] - lo;
    for( int i = lid; i < n; i += NT )
    {
      const ivec2 p = dimdAdaptivePos( k, 0, lo + i );
      const ivec2 g = dimdGradient( 0, k.x + p.x, k.y + p.y, k.filterSizeIdx );
      // the outer gradient: one column left (direction 1) / row up
      const ivec2 g2 = dimdGradient( 0, k.x + p.x - ( k.direction == 1 ? 1 : 0 ), k.y + p.y - ( k.direction == 1 ? 0 : 1 ), k.filterSizeIdx );
      const bool both = g.x != 0 && g.y != 0, both2 = g2.x != 0 && g2.y != 0;
      const int  ang  = g.x != 0 || g.y != 0 ? dimdAngleAdaptive( g ) : -1;
      s_spCoef[i]       = int16_t( ang >= 0 ? dimdAdaptiveAmp( k, g, ang, 0, p, cuW, cuH ) : 0 );
      s_spCoef[256 + i] = int16_t( ang );
      s_spCoef[512 + i] = int16_t( ( both ? dimdRegion( g ) : 15 ) | ( both2 ? dimdRegion( g2 ) : 15 ) << 4 );
      s_spCoef[768 + i] = int16_t( both2 ? dimdAngleAdaptive( g2 ) : g2.x == 0 ? VER_IDX : HOR_IDX );
    }
    SYNC();
    if( lid == 0 )
    {
      int region = 0;   // the CPU's, from sample to sample
      for( int i = 0; i < n; i++ )
      {
        const int ang = int( s_spCoef[256 + i] ), regions = int( s_spCoef[512 + i] );
        if( ang < 0 )   // no gradient: skipped
        {
          continue;
        }
        if( ( regions & 15 ) != 15 )
        {
          region = regions & 15;
        }
        int amp = int( s_spCoef[i] );
        if( amp != 0 )
        {
          const int otherRegion = region;
          int angWeight = int( s_spCoef[768 + i] );
          if( ( regions >> 4 ) != 15 )
          {
            region    = regions >> 4;
            angWeight = region == otherRegion ? angWeight : -1;
          }
          if( abs( ang - angWeight ) <= 1 )
          {
            amp += amp >> 3;
          }
        }
        s_spCoef[i] = int16_t( amp );
      }
    }
    SYNC();
  }
  const int total = s_dimdStart[numOuter];
  int       running = 0;   // the amplitudes summed before the batch (the CPU's totAmp)
  for( int base = 0; base < total && running < k.maxAmp; base += NT )
  {
    // the batch's samples in subgroup order, so that the inclusive sums run in the CPU's order
    const int s = base + int( gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID );
    int amp = 0, ang = 0;
    if( s < total )
    {
      int o = 0;
      while( s >= s_dimdStart[o + 1] )
      {
        o++;
      }
      const int j = s_dimdLo[o] + s - s_dimdStart[o];
      if( o == 0 && firstInOrder )
      {
        amp = int( s_spCoef[s] );
        ang = int( s_spCoef[256 + s] );
      }
      else
      {
        const ivec2 p = dimdAdaptivePos( k, o, j );
        const ivec2 g = dimdGradient( 0, k.x + p.x, k.y + p.y, k.filterSizeIdx );
        if( g.x != 0 || g.y != 0 )
        {
          ang = dimdAngleAdaptive( g );
          amp = dimdAdaptiveAmp( k, g, ang, o, p, cuW, cuH );
        }
      }
    }
    const int incl = subgroupInclusiveAdd( amp );
    if( gl_SubgroupInvocationID == gl_SubgroupSize - 1 )
    {
      s_dimdSg[gl_SubgroupID] = incl;
    }
    SYNC();
    if( lid == 0 )   // the subgroups' totals into exclusive sums, the batch's total after them
    {
      int sum = 0;
      for( int g = 0; g < int( gl_NumSubgroups ); g++ )
      {
        const int v = s_dimdSg[g];
        s_dimdSg[g] = sum;
        sum += v;
      }
      s_dimdSg[gl_NumSubgroups] = sum;
    }
    SYNC();
    if( amp != 0 && running + s_dimdSg[gl_SubgroupID] + incl - amp < k.maxAmp )
    {
      atomicAdd( s_dimdHist[k.histogram * NUM_LUMA_MODE + ang], amp );
    }
    running += s_dimdSg[gl_NumSubgroups];
    SYNC();
  }
}

// the histograms of calls first .. first + num - 1 of a CU of w x h into s_dimdHist (zeroed first), all invocations,
// ending with a barrier
void dimdHistogramsOf( int first, int num, int w, int h, int lid, int NT )
{
  for( int i = lid; i < 3 * NUM_LUMA_MODE; i += NT )
  {
    s_dimdHist[i] = 0;
  }
  SYNC();
  for( int n = 0; n < num; n++ )
  {
    const DimdCall k = BufDimdCall( P.dimdCalls ).v[first + n];
    if( k.kind == 1 )
    {
      dimdHistogram( k, lid, NT );
    }
    else if( k.kind == 2 )
    {
      dimdHistogramAdaptive( k, w, h, lid, NT );
    }
  }
  SYNC();
}

// the histograms of CU c's DIMD (its dimdNum calls), all invocations, ending with a barrier
void dimdHistograms( Cu c, int lid, int NT )
{
  dimdHistogramsOf( c.dimd, c.dimdNum, c.w, c.h, lid, NT );
}

// DIMD chroma's mode (deriveDimdChromaMode with JVET-AC0094, then DecCu at the Cb block): the strongest mode of its one
// histogram, or where that is the co-located luma mode the second strongest (DC where they are the same); one
// invocation
int dimdChromaSelect( int lumaMode, out int second )
{
  int firstAmp = 0, secondAmp = 0, first = 0;
  second = 0;
  for( int i = 0; i < NUM_LUMA_MODE; i++ )
  {
    const int curAmp = s_dimdHist[i];
    if( curAmp > firstAmp )
    {
      secondAmp = firstAmp;
      second    = first;
      firstAmp  = curAmp;
      first     = i;
    }
    else if( curAmp > secondAmp )
    {
      secondAmp = curAmp;
      second    = i;
    }
  }
  return lumaMode != first ? first : first == second ? DC_IDX : second;
}

// a CU's DIMD (CodingUnit's fields)
struct DimdOut
{
  int mode;
  int blending;
  int blendMode[7];
  int relWeight[8];
  int locDep[7];
  int trSecond;   // the second strongest mode (candModeListForTransform[1]; 0 where fewer modes have an amplitude)
};

// the DIMD modes, blending and weights of a luma CU of w x h from its histograms (deriveDimdMode's and
// deriveDimdModeAdaptive's common end, without MDIP's excluded modes); one invocation
DimdOut dimdSelect( int w, int h )
{
  DimdOut d;
  int amp[DIMD_FUSION_NUM - 1], mode[DIMD_FUSION_NUM - 1];
  for( int i = 0; i < DIMD_FUSION_NUM - 1; i++ )
  {
    amp[i] = 0;
    mode[i] = 0;
    d.blendMode[i] = -1;
    d.locDep[i] = 0;
  }
  for( int i = 0; i < DIMD_FUSION_NUM; i++ )
  {
    d.relWeight[i] = -1;
  }
  // the strongest modes, in order (the first of equal amplitudes)
  const int numPred = w * h >= 128 ? DIMD_FUSION_NUM : DIMD_FUSION_NUM_SMALL;
  for( int i = 0; i < NUM_LUMA_MODE; i++ )
  {
    const int curAmp = s_dimdHist[i] + s_dimdHist[NUM_LUMA_MODE + i] + s_dimdHist[2 * NUM_LUMA_MODE + i];
    for( int j = 0; j < numPred - 1; j++ )
    {
      if( curAmp > amp[j] )
      {
        for( int k = numPred - 2; k > j; k-- )
        {
          amp[k]  = amp[k - 1];
          mode[k] = mode[k - 1];
        }
        amp[j]  = curAmp;
        mode[j] = i;
        break;
      }
    }
  }
  // location dependence (JVET-AC0098): little of a mode's amplitude from the left (1) / from above (2)
  for( int i = 0; i < DIMD_FUSION_NUM - 1; i++ )
  {
    const int m = mode[i];
    if( m > DC_IDX )
    {
      const int ampAll = s_dimdHist[m] + s_dimdHist[NUM_LUMA_MODE + m] + s_dimdHist[2 * NUM_LUMA_MODE + m];
      d.locDep[i] = s_dimdHist[m] < ampAll / 3 ? 1 : s_dimdHist[NUM_LUMA_MODE + m] < ampAll / 3 ? 2 : 0;
    }
  }
  d.mode     = mode[0];
  d.trSecond = mode[1];
  bool blending = amp[1] > 0 && mode[1] > DC_IDX && mode[0] > DC_IDX;
  if( d.locDep[0] != 0 && amp[1] == 0 )   // one location-dependent mode: blended with planar
  {
    blending = mode[0] > DC_IDX;
    mode[1]  = 0;
  }
  int countBlendMode = 2;
  if( blending )
  {
    for( int i = 0; i < DIMD_FUSION_NUM - 2; i++ )
    {
      d.blendMode[i] = mode[i + 1];
      countBlendMode += d.blendMode[i] != PLANAR_IDX ? 1 : 0;
    }
  }
  const int dimdPlanarWeight = 64 / 4;
  int sumWeight = 64;
  if( blending && mode[1] == 0 )
  {
    d.relWeight[0] = sumWeight - 21;
    d.relWeight[1] = 21;
    for( int i = 2; i < 6; i++ )
    {
      d.relWeight[i] = 0;
    }
  }
  else if( blending )   // the weights by amplitude (JVET-X0149's division by table), planar's fixed
  {
    sumWeight -= dimdPlanarWeight;
    int s1 = 0;
    for( int i = 0; i < DIMD_FUSION_NUM - 1; i++ )
    {
      s1 += amp[i];
    }
    int x = findMSB( s1 );
    const int normS1 = ( s1 << 4 >> x ) & 15, v = GRAD_DIV_TABLE[normS1] | 8;
    x += normS1 != 0 ? 1 : 0;
    const int shift = x + 3, add = 1 << ( shift - 1 );
    int iRatio[DIMD_FUSION_NUM - 1];
    for( int i = 0; i < DIMD_FUSION_NUM - 1; i++ )
    {
      iRatio[i] = amp[i] == 0 ? 0 : min( ( amp[i] * v * sumWeight + add ) >> shift, sumWeight );
    }
    d.relWeight[0] = iRatio[0];
    int sumWeightReal = iRatio[0], countBlendModeNew = countBlendMode, lastFilled = 0;
    for( int i = 1; i < countBlendMode - 2; i++ )
    {
      d.relWeight[i + 1] = iRatio[i];
      sumWeightReal += iRatio[i];
      if( sumWeightReal > sumWeight )
      {
        lastFilled = i;
        break;
      }
    }
    if( sumWeightReal > sumWeight )   // the modes after the one that exceeds the sum are dropped
    {
      for( int i = lastFilled + 1; i <= countBlendMode - 2; i++ )
      {
        iRatio[i] = 0;
        d.relWeight[i + 1] = 0;
        d.blendMode[i - 1] = 0;
        countBlendModeNew--;
      }
    }
    countBlendMode = countBlendModeNew;
    if( sumWeightReal > sumWeight )   // the excess off the last modes, one each in turn
    {
      int diff = sumWeightReal - sumWeight;
      for( int j = 0; j < sumWeight && diff != 0; j++ )
      {
        for( int i = lastFilled; i >= 1; i-- )
        {
          iRatio[i] -= 1;
          d.relWeight[i + 1] = iRatio[i];
          if( d.relWeight[i + 1] == 0 )
          {
            d.blendMode[i - 1] = 0;
            countBlendMode--;
          }
          diff--;
          if( diff == 0 )
          {
            break;
          }
        }
      }
    }
    d.relWeight[countBlendMode - 1] = sumWeight;   // the last mode takes the rest
    for( int i = 0; i < countBlendMode - 2; i++ )
    {
      d.relWeight[countBlendMode - 1] -= iRatio[i];
      if( d.relWeight[countBlendMode - 1] == 0 )
      {
        d.blendMode[countBlendMode - 3] = 0;
      }
    }
    d.relWeight[1] = dimdPlanarWeight;
  }
  else
  {
    d.relWeight[0] = sumWeight;
    for( int i = 1; i < DIMD_FUSION_NUM; i++ )
    {
      d.relWeight[i] = 0;
    }
  }
  d.blending = blending ? 1 : 0;
  return d;
}

// a CU's DIMD into Params::dimdOut (device-coherent: its blocks read it)
void dimdStore( int cu, DimdOut d )
{
  CBufI32   o    = CBufI32( P.dimdOut );
  const int base = cu * DIMD_OUT;
  o.v[base + DIMD_OUT_MODE]      = d.mode;
  o.v[base + DIMD_OUT_BLENDING]  = d.blending;
  o.v[base + DIMD_OUT_TR_SECOND] = d.trSecond;
  for( int i = 0; i < 7; i++ )
  {
    o.v[base + DIMD_OUT_BLEND_MODE + i] = d.blendMode[i];
    o.v[base + DIMD_OUT_LOC_DEP + i]    = d.locDep[i];
  }
  for( int i = 0; i < 8; i++ )
  {
    o.v[base + DIMD_OUT_REL_WEIGHT + i] = d.relWeight[i];
  }
}

// prediction sub of the blending of DIMD CU cu (predIntraAng; its node's DimdOut): 0 the main mode, 1 planar, 2 .. 7
// the blend modes (those that are not planar); its mode, weight (dimdRelWeight, 0: not predicted) and location
// dependence class (dimdLocDep; planar's 0)
void dimdPrediction( int cu, int sub, out int mode, out int w, out int cls )
{
  CBufI32   d = CBufI32( P.dimdOut );
  const int o = cu * DIMD_OUT;
  if( sub == 0 )
  {
    mode = d.v[o + DIMD_OUT_MODE];
    w    = d.v[o + DIMD_OUT_REL_WEIGHT];
    cls  = d.v[o + DIMD_OUT_LOC_DEP];
  }
  else if( sub == 1 )
  {
    mode = PLANAR_IDX;
    w    = d.v[o + DIMD_OUT_REL_WEIGHT + 1];
    cls  = 0;
  }
  else
  {
    mode = d.v[o + DIMD_OUT_BLEND_MODE + sub - 2];
    w    = mode != PLANAR_IDX ? d.v[o + DIMD_OUT_REL_WEIGHT + sub] : 0;
    cls  = d.v[o + DIMD_OUT_LOC_DEP + sub - 1];
  }
}

// C++'s integer division (truncated toward zero, also for negative operands)
int divTrunc( int a, int b )
{
  const int q = abs( a ) / abs( b );
  return ( a < 0 ) != ( b < 0 ) ? -q : q;
}

// (int)( ( (double) a / (double) t ) * (double) c ) of xLocationdepBlending's clip ranges, in integers (a, c >= 0,
// t >= 1, all below 2^12): floor(a c / t), but one less where a c / t is an integer k the two roundings of the double
// expression fall short of. That needs a / t inexact (t over gcd(a, t) not a power of two) and rounded down: its
// mantissa M = floor(a 2^S / t), S = 52 - floor(log2(a / t)), rounded down where the remainder R = a 2^S mod t is
// below t / 2 (never a tie); the product then lies c R / t 2^(e - 52) below k, which rounds to k unless that exceeds
// half the spacing of the doubles below k (ties go to k, whose mantissa is even)
int dimdScaledWeight( int a, int t, int c )
{
  if( a == 0 || c == 0 )
  {
    return 0;
  }
  const int k = ( a * c ) / t;
  if( ( a * c ) % t != 0 )
  {
    return k;
  }
  int g = a, h = t;   // gcd
  while( h != 0 )
  {
    const int r = g % h;
    g = h;
    h = r;
  }
  const int tp = t / g;
  if( ( tp & ( tp - 1 ) ) == 0 )   // a / t exact
  {
    return k;
  }
  const int ea = findMSB( a ), et = findMSB( t );
  const int e  = ( a << et ) >= ( t << ea ) ? ea - et : ea - et - 1;   // floor(log2(a / t))
  int R = a % t;
  for( int i = 0; i < 52 - e; i++ )
  {
    R = ( 2 * R ) % t;
  }
  if( 2 * R > t )   // rounded up: the product is at least k
  {
    return k;
  }
  const int ek = findMSB( k );
  const int Ep = ( k & ( k - 1 ) ) == 0 ? ek - 1 : ek;   // the exponent of the spacing below k (2^(Ep - 52))
  const int p  = e - Ep + 1;
  // c R 2^p <= t: within half the spacing
  const bool rounds = p >= 0 ? ( c * R ) << p <= t : c * R <= t << -p;
  return rounds ? k : k - 1;
}

#endif
