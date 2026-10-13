// CCLM on the GPU (cross-component linear models of a chroma block: LM_CHROMA, MDLM_L / MDLM_T and their MMLM
// variants of two models): IntraPrediction::xGetLumaRecPixels downsamples the co-located luma block and the luma
// beside its templates (4:2:0), xGetLMParametersLMS fits a model (ECM's least squares) to the template's luma and
// chroma samples, JVET-Z0050 adjusts its slope, predIntraChromaLM applies it, and LBCCP (JVET-AD0120) can smooth the
// prediction with the reconstruction around it. Included by intra_rec.comp; the block's references (its chroma line)
// are in s_ref[0]
#ifndef INTRA_CCLM_GLSL
#define INTRA_CCLM_GLSL

// Params::cclmFilter: xGetLumaRecPixels' downsampling filter by the SPS's collocated chroma flags (JVET-AN0090)
const int CCLM_FILTER_AVERAGE = 0;   // neither flag: the mean of 2 x 2 samples
const int CCLM_FILTER_CROSS   = 1;   // both: the 5-tap cross
const int CCLM_FILTER_6TAP    = 2;   // one of them: 1 2 1 over two rows

// the model in shared memory (invocation 0 fits it, all apply it), in MIP's arrays, which an LM block does not use:
// the first model's a, b, shift, the second's, and the luma threshold between them
#define s_cclm s_mipBdry
const int CCLM_A = 0, CCLM_B = 1, CCLM_SHIFT = 2, CCLM_A2 = 3, CCLM_B2 = 4, CCLM_SHIFT2 = 5, CCLM_THRES = 6;
// (the template's: whether all units above / left are available, the available units above-right / below-left;
// LBCCP's: whether the positions above, left, above-left, above-right and below-left of the block are)
#define s_cclmAvail s_mipRed
const int CCLM_ABOVE = 0, CCLM_LEFT = 1, CCLM_AR_UNITS = 2, CCLM_BL_UNITS = 3;
const int CCLM_POS_ABOVE = 4, CCLM_POS_LEFT = 5, CCLM_POS_LEFT_ABOVE = 6, CCLM_POS_ABOVE_RIGHT = 7, CCLM_POS_LEFT_BELOW = 8;

// the availability of block t's template units (xGetLMParametersLMS: chroma units of 2) and of LBCCP's positions
// (isPosAvailable) into s_cclmAvail (invocation 0)
void cclmAvailability( Tu t )
{
  const int ch = 1;
  OwnerMap  om = OwnerMap( BufI32( P.owner[t.comp] ), P.ownerW[ch], P.ownerH[ch], P.unitLog2X[ch], P.unitLog2Y[ch] );
  const int unit = 2, tAU = t.w, tLU = t.h;   // the template's arms of 2W / 2H in units of 2
  int av[7];
  refLineAvailability( om, g_ti, t.x, t.y, t.w, t.h, tAU, tLU, 0, unit, unit, tAU, tLU, av );
  const bool above = av[1] == t.w / unit, left = av[3] == t.h / unit;
  s_cclmAvail[CCLM_ABOVE]    = above ? 1 : 0;
  s_cclmAvail[CCLM_LEFT]     = left ? 1 : 0;
  s_cclmAvail[CCLM_AR_UNITS] = above ? av[2] : 0;   // (none where the units above are not all available)
  s_cclmAvail[CCLM_BL_UNITS] = left ? av[4] : 0;
  s_cclmAvail[CCLM_POS_ABOVE]       = decodedBefore( om, t.x, t.y - 1, g_ti ) ? 1 : 0;
  s_cclmAvail[CCLM_POS_LEFT]        = decodedBefore( om, t.x - 1, t.y, g_ti ) ? 1 : 0;
  s_cclmAvail[CCLM_POS_LEFT_ABOVE]  = decodedBefore( om, t.x - 1, t.y - 1, g_ti ) ? 1 : 0;
  s_cclmAvail[CCLM_POS_ABOVE_RIGHT] = decodedBefore( om, t.x + t.w, t.y - 1, g_ti ) ? 1 : 0;
  s_cclmAvail[CCLM_POS_LEFT_BELOW]  = decodedBefore( om, t.x - 1, t.y + t.h, g_ti ) ? 1 : 0;
}

bool cclmMultiModel( int mode ) { return mode == MMLM_CHROMA_IDX || mode == MMLM_L_IDX || mode == MMLM_T_IDX; }
bool cclmLeftOnly( int mode ) { return mode == MDLM_L_IDX || mode == MMLM_L_IDX; }
bool cclmTopOnly( int mode ) { return mode == MDLM_T_IDX || mode == MMLM_T_IDX; }

// ----- the downsampled luma (xGetLumaRecPixels; 4:2:0)

// the co-located luma block's downsampled sample (i, j) (luma origin (lx, ly)); the side not available pads the filter
int cclmLumaInner( int lx, int ly, int i, int j, bool leftAvail, bool aboveAvail )
{
  const int x = lx + 2 * i, y = ly + 2 * j, l = i == 0 && !leftAvail ? 0 : 1;
  if( P.cclmFilter == CCLM_FILTER_AVERAGE )
  {
    return ( recoAt( 0, x, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x, y + 1 ) + recoAt( 0, x + 1, y + 1 ) + 2 ) >> 2;
  }
  if( P.cclmFilter == CCLM_FILTER_CROSS )
  {
    const int t = j == 0 && !aboveAvail ? 0 : 1;
    return ( recoAt( 0, x, y - t ) + 4 * recoAt( 0, x, y ) + recoAt( 0, x - l, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x, y + 1 ) + 4 ) >> 3;
  }
  return ( 2 * recoAt( 0, x, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x - l, y ) + 2 * recoAt( 0, x, y + 1 ) + recoAt( 0, x + 1, y + 1 )
           + recoAt( 0, x - l, y + 1 ) + 4 ) >> 3;
}

// the downsampled sample i of the row above the co-located block: in a CTU's first row from its one luma row above
// (the line buffer), else from the two rows above it
int cclmLumaAbove( int lx, int ly, int i, bool leftAvail )
{
  const int x = lx + 2 * i, l = i == 0 && !leftAvail ? 0 : 1;
  if( ( ly & ( P.ctuSize - 1 ) ) == 0 )
  {
    return P.cclmFilter == CCLM_FILTER_AVERAGE ? ( recoAt( 0, x, ly - 1 ) + recoAt( 0, x + 1, ly - 1 ) + 1 ) >> 1
                                              : ( 2 * recoAt( 0, x, ly - 1 ) + recoAt( 0, x - l, ly - 1 ) + recoAt( 0, x + 1, ly - 1 ) + 2 ) >> 2;
  }
  const int y = ly - 2;
  if( P.cclmFilter == CCLM_FILTER_AVERAGE )
  {
    return ( recoAt( 0, x, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x, y + 1 ) + recoAt( 0, x + 1, y + 1 ) + 2 ) >> 2;
  }
  if( P.cclmFilter == CCLM_FILTER_CROSS )
  {
    return ( recoAt( 0, x, y - 1 ) + 4 * recoAt( 0, x, y ) + recoAt( 0, x - l, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x, y + 1 ) + 4 ) >> 3;
  }
  return ( 2 * recoAt( 0, x, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x - l, y ) + 2 * recoAt( 0, x, y + 1 ) + recoAt( 0, x + 1, y + 1 )
           + recoAt( 0, x - l, y + 1 ) + 4 ) >> 3;
}

// the downsampled sample j of the column left of the co-located block (from the luma columns lx - 3 .. lx - 1)
int cclmLumaLeft( int lx, int ly, int j, bool aboveAvail )
{
  const int x = lx - 2, y = ly + 2 * j;
  if( P.cclmFilter == CCLM_FILTER_AVERAGE )
  {
    return ( recoAt( 0, x, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x, y + 1 ) + recoAt( 0, x + 1, y + 1 ) + 2 ) >> 2;
  }
  if( P.cclmFilter == CCLM_FILTER_CROSS )
  {
    const int t = j == 0 && !aboveAvail ? 0 : 1;
    return ( recoAt( 0, x, y - t ) + 4 * recoAt( 0, x, y ) + recoAt( 0, x - 1, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x, y + 1 ) + 4 ) >> 3;
  }
  return ( 2 * recoAt( 0, x, y ) + recoAt( 0, x + 1, y ) + recoAt( 0, x - 1, y ) + 2 * recoAt( 0, x, y + 1 ) + recoAt( 0, x + 1, y + 1 )
           + recoAt( 0, x - 1, y + 1 ) + 4 ) >> 3;
}

// ----- the model (xGetLMParametersLMS, xLMSampleClassifiedTraining, xCalcLMParametersGeneralized; one invocation)

// floorLog2 of a positive 64-bit value (floorLog2Uint64)
int cclmLog2( uint64_t v )
{
  const uint hi = uint( v >> 32 );
  return hi != 0u ? 32 + findMSB( hi ) : findMSB( uint( v ) );
}

// xCalcLMParametersGeneralized from the sums over 1 << countShift samples (64-bit sums: JVET-AJ0237): a, b and shift of
// chroma = ( a * luma >> shift ) + b; the division by the luma variance through m_auShiftLM (computed here: its table
// for the luma bit depth, the chroma's: the host rejects unequal ones)
ivec3 cclmFit( int64_t x, int64_t y, int64_t xx, int64_t xy, int countShift, int bitDepth )
{
  const int tempShift = bitDepth + countShift - ( bitDepth > 10 ? 31 : 15 );
  if( tempShift > 0 )
  {
    const int64_t rounding = int64_t( 1 ) << ( tempShift - 1 );
    x  = ( x + rounding ) >> tempShift;
    y  = ( y + rounding ) >> tempShift;
    xx = ( xx + rounding ) >> tempShift;
    xy = ( xy + rounding ) >> tempShift;
    countShift -= tempShift;
  }
  const int64_t avgX = x >> countShift, avgY = y >> countShift;
  const int64_t errX = x & ( ( int64_t( 1 ) << countShift ) - 1 ), errY = y & ( ( int64_t( 1 ) << countShift ) - 1 );
  const int iB = 7;
  int shift = 13 - iB;
  if( countShift == 0 )
  {
    return ivec3( 0, 1 << ( bitDepth - 1 ), 0 );
  }
  const int64_t a1 = xy - ( avgX * avgY << countShift ) - avgX * errY - avgY * errX;
  const int64_t a2 = xx - ( avgX * avgX << countShift ) - 2 * avgX * errX;
  const int shiftA1 = bitDepth - 2, shiftA2 = 5, accuracyShift = bitDepth + 4;
  const int scaleShiftA1 = a1 == 0 ? 0 : max( 0, cclmLog2( uint64_t( abs( a1 ) ) ) - shiftA1 );
  const int scaleShiftA2 = a2 == 0 ? 0 : max( 0, cclmLog2( uint64_t( abs( a2 ) ) ) - shiftA2 );
  const int scaleShiftA  = scaleShiftA2 + accuracyShift - shift - scaleShiftA1;
  const int64_t a2s = a2 >> scaleShiftA2, a1s = a1 >> scaleShiftA1;
  int a = 0;
  if( a2s >= 32 )
  {
    const int     i   = int( a2s );   // 32 .. 63 (a2s keeps 6 significant bits)
    const int64_t a2t = ( ( int64_t( 1 ) << ( bitDepth + 4 ) ) + i / 2 ) / i;
    a = int( a1s * a2t );
  }
  a = scaleShiftA < 0 ? a << -scaleShiftA : a >> scaleShiftA;
  a = clamp( a, -( 1 << ( 15 - iB ) ), ( 1 << ( 15 - iB ) ) - 1 ) << iB;
  const int n = a != 0 ? findMSB( abs( a ) + ( ( a < 0 ? -1 : 1 ) - 1 ) / 2 ) - 5 : 0;
  shift = shift + iB - n;
  a >>= n;
  return ivec3( a, int( avgY - ( ( a * avgX ) >> shift ) ), shift );
}

// PU::getMeanValue: sum / div without division (JVET-AD0184)
int cclmMean( int sum, int div )
{
  const int sign = sum < 0 ? -1 : 1;
  sum = abs( sum );
  const int divTable[16] = int[16]( 0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0 );
  int x = findMSB( div );
  const int normNum1 = ( div << 4 >> x ) & 15, v = divTable[normNum1] | 8;
  x += normNum1 != 0 ? 1 : 0;
  int shift = 13 - x, retVal;
  if( shift < 0 )
  {
    shift  = -shift;
    retVal = ( sum * v + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  else
  {
    retVal = ( sum * v ) << shift;
  }
  return sign * ( retVal >> 16 );
}

// xUpdateCclmModel (JVET-Z0050): the slope adjusted by delta / 8 around the template's middle luma value
void cclmAdjust( inout int a, inout int b, inout int shift, int midLuma, int delta )
{
  if( delta == 0 )
  {
    return;
  }
  const int dShift = 3;
  delta = a > 0 ? -delta : delta;
  if( shift < dShift )
  {
    a <<= dShift - shift;
    shift = dShift;
  }
  else if( shift > dShift )
  {
    delta <<= shift - dShift;
  }
  a += delta;
  b -= ( delta * midLuma ) >> shift;
}

// the model of CCLM block t (invocation 0) into s_cclm: the template's luma (downsampled, into s_tmp from 0) and chroma
// samples (from its references in s_ref[0], into s_tmp from 256) as xGetLMParametersLMS picks them, fitted as one model
// (LM_CHROMA, MDLM: the MDLM template padded to a power of two) or as two split at the luma mean (MMLM:
// xLMSampleClassifiedTraining, each group padded to a power of two), then the slope adjustments of Tu::lm
void cclmModel( Tu t, int lx, int ly, int predStride )
{
  const int  W = t.w, H = t.h, mode = t.dirMode, bitDepth = t.bitDepth;
  const bool leftOnly = cclmLeftOnly( mode ), topOnly = cclmTopOnly( mode );
  bool aboveAvail = s_cclmAvail[CCLM_ABOVE] != 0, leftAvail = s_cclmAvail[CCLM_LEFT] != 0;
  const bool aboveAvailFill = aboveAvail, leftAvailFill = leftAvail;   // (the downsampling's padding)
  const int  unit = 2;   // baseUnitSize >> 1: the chroma units of the template's availability
  int actualTop = 0, actualLeft = 0, minDim = 1;
  if( topOnly )
  {
    leftAvail = false;
    actualTop = aboveAvail ? unit * ( W / unit + s_cclmAvail[CCLM_AR_UNITS] ) : 0;
    minDim    = actualTop;
  }
  else if( leftOnly )
  {
    aboveAvail = false;
    actualLeft = leftAvail ? unit * ( H / unit + s_cclmAvail[CCLM_BL_UNITS] ) : 0;
    minDim     = actualLeft;
  }
  else
  {
    actualTop  = W;
    actualLeft = H;
    minDim     = leftAvail && aboveAvail ? 1 << findMSB( min( actualLeft, actualTop ) ) : 1 << findMSB( leftAvail ? actualLeft : actualTop );
  }
  const int numSteps = minDim;
  int cntT = 0, cntL = 0, sumLuma = 0;
  const int numPels = aboveAvail && leftAvail ? 2 * numSteps : aboveAvail || leftAvail ? numSteps : 0;
  if( aboveAvail )
  {
    cntT = numSteps;
    for( int j = 0; j < numSteps; j++ )
    {
      const int idx = ( j * actualTop ) / minDim;
      s_tmp[j]       = int16_t( cclmLumaAbove( lx, ly, idx, leftAvailFill ) );
      s_tmp[256 + j] = s_ref[0][1 + idx];
      sumLuma       += int( s_tmp[j] );
    }
  }
  if( leftAvail )
  {
    cntL = numSteps;
    for( int i = 0; i < numSteps; i++ )
    {
      const int idx = ( i * actualLeft ) / minDim;
      s_tmp[cntT + i]       = int16_t( cclmLumaLeft( lx, ly, idx, aboveAvailFill ) );
      s_tmp[256 + cntT + i] = s_ref[0][predStride + 1 + idx];
      sumLuma              += int( s_tmp[cntT + i] );
    }
  }
  int midLuma = numPels != 0 ? cclmMean( sumLuma + ( numPels >> 1 ), numPels ) : 1 << ( bitDepth - 1 );
  int midLuma2 = midLuma;
  ivec3 m1 = ivec3( 0, 1 << ( bitDepth - 1 ), 0 ), m2 = m1;
  int thres = 0;
  if( cclmMultiModel( mode ) )
  {
    const int count = cntT + cntL;
    int mean = 0, meanC = 0;
    for( int i = 0; i < count; i++ )
    {
      mean  += int( s_tmp[i] );
      meanC += int( s_tmp[256 + i] );
    }
    if( count != 0 )   // the means without division (divSigTable)
    {
      const int divSigTable[16] = int[16]( 0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0 );
      int x = findMSB( count );
      const int normDiff = ( count << 4 >> x ) & 15, v = divSigTable[normDiff] | 8;
      x += normDiff != 0 ? 1 : 0;
      mean  = ( mean * v ) >> ( x + 3 );
      meanC = ( meanC * v ) >> ( x + 3 );
    }
    thres = mean;
    if( count >= 4 )   // xLMSampleClassifiedTraining: the groups into s_tmp from 512 (luma) and 768 (chroma), 128 each
    {
      const int meanDiff = meanC - mean, split = max( 1, mean );
      int groupCount[2] = int[2]( 0, 0 );
      for( int i = 0; i < count; i++ )
      {
        const int g = int( s_tmp[i] ) <= split ? 0 : 1;
        s_tmp[512 + g * 128 + groupCount[g]] = s_tmp[i];
        s_tmp[768 + g * 128 + groupCount[g]] = s_tmp[256 + i];
        groupCount[g]++;
      }
      for( int g = 0; g < 2; g++ )
      {
        const int exist = groupCount[g];
        int64_t x = 0, y = 0, xx = 0, xy = 0;
        if( exist >= 2 )
        {
          const int upper = 1 << ( findMSB( exist - 1 ) + 1 ), lower = 1 << findMSB( exist );
          if( upper != lower )   // padded to a power of two
          {
            const int numPadded = min( exist, upper - exist ), step = exist / numPadded;
            for( int i = 0; i < numPadded; i++ )
            {
              s_tmp[512 + g * 128 + exist + i] = s_tmp[512 + g * 128 + i * step];
              s_tmp[768 + g * 128 + exist + i] = s_tmp[768 + g * 128 + i * step];
            }
            groupCount[g] = upper;
          }
          for( int i = 0; i < groupCount[g]; i++ )
          {
            const int64_t l = s_tmp[512 + g * 128 + i], c = s_tmp[768 + g * 128 + i];
            x += l;  y += c;  xx += l * l;  xy += l * c;
          }
          const ivec3 m = cclmFit( x, y, xx, xy, findMSB( groupCount[g] ), bitDepth );
          if( g == 0 ) m1 = m; else m2 = m;
        }
        else
        {
          const ivec3 m = ivec3( 0, meanDiff, 0 );
          if( g == 0 ) m1 = m; else m2 = m;
        }
      }
    }
    // the middle luma of each model's samples (unpadded, split at the unclamped mean)
    int sum0 = 0, sum1 = 0, num0 = 0, num1 = 0;
    for( int i = 0; i < count; i++ )
    {
      if( int( s_tmp[i] ) <= mean ) { sum0 += int( s_tmp[i] ); num0++; }
      else                          { sum1 += int( s_tmp[i] ); num1++; }
    }
    midLuma  = num0 != 0 ? cclmMean( sum0 + ( num0 >> 1 ), num0 ) : mean;
    midLuma2 = num1 != 0 ? cclmMean( sum1 + ( num1 >> 1 ), num1 ) : mean;
  }
  else
  {
    int64_t x = 0, y = 0, xx = 0, xy = 0;
    int countShift = 0;
    bool fit = true;
    int count = cntT + cntL;
    if( leftOnly || topOnly )   // MDLM: its template padded to a power of two (xPadMdlmTemplateSample)
    {
      const int orgNum = topOnly ? ( aboveAvail ? W : 0 ) : ( leftAvail ? H : 0 ), exist = topOnly ? actualTop : actualLeft;
      fit = orgNum != 0 && exist != 0;
      if( fit )
      {
        const int target = 1 << ( findMSB( exist - 1 ) + 1 );
        if( target != exist )
        {
          const int step = exist / ( target - exist );
          for( int i = 0; i < target - exist; i++ )
          {
            s_tmp[exist + i]       = s_tmp[i * step];
            s_tmp[256 + exist + i] = s_tmp[256 + i * step];
          }
        }
        count      = target;
        countShift = findMSB( target );
      }
    }
    else
    {
      fit = aboveAvail || leftAvail;
      countShift = ( aboveAvail ? findMSB( minDim ) : 0 ) + ( leftAvail ? ( aboveAvail ? 1 : findMSB( minDim ) ) : 0 );
    }
    if( fit )
    {
      for( int i = 0; i < count; i++ )
      {
        const int64_t l = s_tmp[i], c = s_tmp[256 + i];
        x += l;  y += c;  xx += l * l;  xy += l * c;
      }
      m1 = cclmFit( x, y, xx, xy, countShift, bitDepth );
    }
  }
  // the slope adjustments (cclmOffsets of the component), then the model into shared memory
  cclmAdjust( m1.x, m1.y, m1.z, midLuma, ( ( t.lm >> LM_DELTA0_SHIFT ) & 15 ) - 8 );
  cclmAdjust( m2.x, m2.y, m2.z, midLuma2, ( ( t.lm >> LM_DELTA1_SHIFT ) & 15 ) - 8 );
  s_cclm[CCLM_A]  = m1.x;  s_cclm[CCLM_B]  = m1.y;  s_cclm[CCLM_SHIFT]  = m1.z;
  s_cclm[CCLM_A2] = m2.x;  s_cclm[CCLM_B2] = m2.y;  s_cclm[CCLM_SHIFT2] = m2.z;
  s_cclm[CCLM_THRES] = thres;
}

// ----- the prediction (predIntraChromaLM, LBCCP's filterPredInside)

// the model applied to the co-located block's downsampled sample (i, j) (two models: split at the threshold)
int cclmPredSample( Tu t, int lx, int ly, int i, int j, int maxPel )
{
  const int luma = cclmLumaInner( lx, ly, i, j, s_cclmAvail[CCLM_LEFT] != 0, s_cclmAvail[CCLM_ABOVE] != 0 );
  const bool second = cclmMultiModel( t.dirMode ) && luma > s_cclm[CCLM_THRES];
  const int  v = second ? ( ( s_cclm[CCLM_A2] * luma ) >> s_cclm[CCLM_SHIFT2] ) + s_cclm[CCLM_B2]
                        : ( ( s_cclm[CCLM_A] * luma ) >> s_cclm[CCLM_SHIFT] ) + s_cclm[CCLM_B];
  return clamp( v, 0, maxPel );
}

// filterPredInside's padding above (x = 0 .. W - 1) and left (y = 0 .. H - 1) of the prediction: the reconstruction
// where that side is available, else the prediction's first row / column
int cclmInsideAbove( Tu t, int lx, int ly, int x, int maxPel )
{
  return s_cclmAvail[CCLM_POS_ABOVE] != 0 ? recoAt( t.comp, t.x + x, t.y - 1 ) : cclmPredSample( t, lx, ly, x, 0, maxPel );
}

int cclmInsideLeft( Tu t, int lx, int ly, int y, int maxPel )
{
  return s_cclmAvail[CCLM_POS_LEFT] != 0 ? recoAt( t.comp, t.x - 1, t.y + y ) : cclmPredSample( t, lx, ly, 0, y, maxPel );
}

// filterPredInside's padded prediction at (x, y), -1 <= x <= W, -1 <= y <= H: the prediction inside and, right of and
// below it, its last column / row; above and left cclmInsideAbove / Left; the three corners the reconstruction where
// available, else the top-left one the mean of its two neighbours, the others the padding next to them
int cclmInsideSample( Tu t, int lx, int ly, int x, int y, int maxPel )
{
  const int W = t.w, H = t.h;
  if( y < 0 && x < 0 )
  {
    return s_cclmAvail[CCLM_POS_LEFT_ABOVE] != 0 ? recoAt( t.comp, t.x - 1, t.y - 1 )
         : ( cclmInsideAbove( t, lx, ly, 0, maxPel ) + cclmInsideLeft( t, lx, ly, 0, maxPel ) + 1 ) >> 1;
  }
  if( y < 0 )
  {
    return x < W ? cclmInsideAbove( t, lx, ly, x, maxPel )
         : s_cclmAvail[CCLM_POS_ABOVE_RIGHT] != 0 ? recoAt( t.comp, t.x + W, t.y - 1 ) : cclmInsideAbove( t, lx, ly, W - 1, maxPel );
  }
  if( x < 0 )
  {
    return y < H ? cclmInsideLeft( t, lx, ly, y, maxPel )
         : s_cclmAvail[CCLM_POS_LEFT_BELOW] != 0 ? recoAt( t.comp, t.x - 1, t.y + H ) : cclmInsideLeft( t, lx, ly, H - 1, maxPel );
  }
  return cclmPredSample( t, lx, ly, min( x, W - 1 ), min( y, H - 1 ), maxPel );
}

// LBCCP's filtered sample (x, y): the padded prediction around it weighed 1, itself 8, the sum over 16
int cclmInsideFiltered( Tu t, int lx, int ly, int x, int y, int maxPel )
{
  int sum = 0;
  for( int dy = -1; dy <= 1; dy++ )
  {
    for( int dx = -1; dx <= 1; dx++ )
    {
      sum += ( dx == 0 && dy == 0 ? 8 : 1 ) * cclmInsideSample( t, lx, ly, x + dx, y + dy, maxPel );
    }
  }
  return ( sum + 8 ) >> 4;
}

#endif
