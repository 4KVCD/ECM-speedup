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

// the availability and the models in shared memory (invocation 0 finds them, all apply them), in MIP's reduced
// prediction, which an LM block does not use. The availability: the template's (whether all units above / left are
// available, the available units above-right / below-left), LBCCP's (whether the positions above, left, above-left,
// above-right and below-left of the block are). The models: per slot (0: the block's, 1: with LFNST / NSPT the other
// component's, cclmTransformMode) at CCLM_MODEL + slot * CCLM_SLOT the first model's a, b, shift, the second's, and the
// luma threshold between them
#define s_cclm s_mipRed
const int CCLM_ABOVE = 0, CCLM_LEFT = 1, CCLM_AR_UNITS = 2, CCLM_BL_UNITS = 3;
const int CCLM_POS_ABOVE = 4, CCLM_POS_LEFT = 5, CCLM_POS_LEFT_ABOVE = 6, CCLM_POS_ABOVE_RIGHT = 7, CCLM_POS_LEFT_BELOW = 8;
const int CCLM_MODEL = 16, CCLM_SLOT = 8;
const int CCLM_A = 0, CCLM_B = 1, CCLM_SHIFT = 2, CCLM_A2 = 3, CCLM_B2 = 4, CCLM_SHIFT2 = 5, CCLM_THRES = 6;

// the availability of block t's template units (xGetLMParametersLMS: chroma units of 2) and of LBCCP's positions
// (isPosAvailable) into s_cclm (invocation 0)
void cclmAvailability( Tu t )
{
  const int ch = 1;
  OwnerMap  om = OwnerMap( BufI32( P.owner[t.comp] ), P.ownerW[ch], P.ownerH[ch], P.unitLog2X[ch], P.unitLog2Y[ch] );
  const int unit = 2, tAU = t.w, tLU = t.h;   // the template's arms of 2W / 2H in units of 2
  int av[7];
  refLineAvailability( om, g_ti, t.x, t.y, t.w, t.h, tAU, tLU, 0, unit, unit, tAU, tLU, av );
  const bool above = av[1] == t.w / unit, left = av[3] == t.h / unit;
  s_cclm[CCLM_ABOVE]    = above ? 1 : 0;
  s_cclm[CCLM_LEFT]     = left ? 1 : 0;
  s_cclm[CCLM_AR_UNITS] = above ? av[2] : 0;   // (none where the units above are not all available)
  s_cclm[CCLM_BL_UNITS] = left ? av[4] : 0;
  s_cclm[CCLM_POS_ABOVE]       = decodedBefore( om, t.x, t.y - 1, g_ti ) ? 1 : 0;
  s_cclm[CCLM_POS_LEFT]        = decodedBefore( om, t.x - 1, t.y, g_ti ) ? 1 : 0;
  s_cclm[CCLM_POS_LEFT_ABOVE]  = decodedBefore( om, t.x - 1, t.y - 1, g_ti ) ? 1 : 0;
  s_cclm[CCLM_POS_ABOVE_RIGHT] = decodedBefore( om, t.x + t.w, t.y - 1, g_ti ) ? 1 : 0;
  s_cclm[CCLM_POS_LEFT_BELOW]  = decodedBefore( om, t.x - 1, t.y + t.h, g_ti ) ? 1 : 0;
}

bool cclmMultiModel( int mode ) { return mode == MMLM_CHROMA_IDX || mode == MMLM_L_IDX || mode == MMLM_T_IDX; }
bool cclmLeftOnly( int mode ) { return mode == MDLM_L_IDX || mode == MMLM_L_IDX; }
bool cclmTopOnly( int mode ) { return mode == MDLM_T_IDX || mode == MMLM_T_IDX; }

// ----- the downsampled luma (xGetLumaRecPixels; 4:2:0), or with GLM its gradient

// GLM's gradient patterns (Rom.cpp's g_glmPattern with JVET-AN0090 and JVET-AP0105, NUM_GLM_PATTERN 4 with CCCM) per
// chroma location type (xGetLumaRecPixels' chromaloctype: 0 one collocated flag, 1 neither, 2 both), 9 taps each: the
// samples left of, at and right of the luma sample in its row, the row below, the row above (type 2 only)
const int NUM_GLM_PATTERN = 4;
const int GLM_PATTERN[3 * NUM_GLM_PATTERN * 9] = int[](
   1,  0, -1,  1,  0, -1,  0,  0,  0,     1,  2,  1, -1, -2, -1,  0,  0,  0,
   2,  1, -1,  1, -1, -2,  0,  0,  0,    -1,  1,  2, -2, -1,  1,  0,  0,  0,
   0,  1, -1,  0,  1, -1,  0,  0,  0,     0,  1,  1,  0, -1, -1,  0,  0,  0,
   0,  1,  0,  0,  0, -1,  0,  0,  0,     0,  0,  1,  0, -1,  0,  0,  0,  0,
   2,  0, -2,  1,  0, -1,  1,  0, -1,     0,  0,  0, -1, -2, -1,  1,  2,  1,
   1,  0, -1,  0, -1, -2,  2,  1,  0,    -1,  0,  1, -2, -1,  0,  0,  1,  2 );

// xGlmGetLumaVal (JVET-AA0126; with JVET-AB0092 the gradient alone) of pattern glm (1 .. NUM_GLM_PATTERN) at the luma
// sample (x, y) the downsampling centres on: the samples x - l .. x + 1 of rows y, y + 1 and y - u (l, u: 0 where that
// side pads, as the downsampling's)
int glmGradient( int glm, int x, int y, int l, int u )
{
  const int type = P.cclmFilter == CCLM_FILTER_6TAP ? 0 : P.cclmFilter == CCLM_FILTER_AVERAGE ? 1 : 2;
  const int c    = ( type * NUM_GLM_PATTERN + glm - 1 ) * 9;
  int g = GLM_PATTERN[c] * recoAt( 0, x - l, y ) + GLM_PATTERN[c + 1] * recoAt( 0, x, y ) + GLM_PATTERN[c + 2] * recoAt( 0, x + 1, y )
          + GLM_PATTERN[c + 3] * recoAt( 0, x - l, y + 1 ) + GLM_PATTERN[c + 4] * recoAt( 0, x, y + 1 )
          + GLM_PATTERN[c + 5] * recoAt( 0, x + 1, y + 1 );
  if( type == 2 )
  {
    g += GLM_PATTERN[c + 6] * recoAt( 0, x - l, y - u ) + GLM_PATTERN[c + 7] * recoAt( 0, x, y - u )
         + GLM_PATTERN[c + 8] * recoAt( 0, x + 1, y - u );
  }
  return int( int16_t( g ) );   // (a Pel)
}

// the co-located luma block's downsampled sample (i, j) (luma origin (lx, ly)), with GLM pattern glm (0: none) its
// gradient; the side not available pads the filter
int cclmLumaInner( int lx, int ly, int i, int j, bool leftAvail, bool aboveAvail, int glm )
{
  const int x = lx + 2 * i, y = ly + 2 * j, l = i == 0 && !leftAvail ? 0 : 1;
  if( glm != 0 )
  {
    return glmGradient( glm, x, y, l, j == 0 && !aboveAvail ? 0 : 1 );
  }
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
// (the line buffer), else from the two rows above it; with GLM the gradient around row ly - 2 (also in a CTU's first
// row)
int cclmLumaAbove( int lx, int ly, int i, bool leftAvail, int glm )
{
  const int x = lx + 2 * i, l = i == 0 && !leftAvail ? 0 : 1;
  if( glm != 0 )
  {
    return glmGradient( glm, x, ly - 2, l, 1 );
  }
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

// the downsampled sample j of the column left of the co-located block (from the luma columns lx - 3 .. lx - 1), with
// GLM the gradient
int cclmLumaLeft( int lx, int ly, int j, bool aboveAvail, int glm )
{
  const int x = lx - 2, y = ly + 2 * j;
  if( glm != 0 )
  {
    return glmGradient( glm, x, y, 1, j == 0 && !aboveAvail ? 0 : 1 );
  }
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

// the model of CCLM block t (invocation 0) into slot `slot` of s_cclm: the template's luma (downsampled, into s_tmp from
// 0) and chroma samples (from its references in s_ref[ref], into s_tmp from 256) as xGetLMParametersLMS picks them,
// fitted as one model (LM_CHROMA, MDLM: the MDLM template padded to a power of two) or as two split at the luma mean
// (MMLM: xLMSampleClassifiedTraining, each group padded to a power of two), then the slope adjustments of Tu::lm
void cclmModel( Tu t, int lx, int ly, int predStride, int ref, int slot )
{
  const int  W = t.w, H = t.h, mode = t.dirMode, bitDepth = t.bitDepth;
  const int  glm = ( t.lm >> LM_GLM_SHIFT ) & 15;   // GLM's gradient pattern (0: none)
  const bool leftOnly = cclmLeftOnly( mode ), topOnly = cclmTopOnly( mode );
  bool aboveAvail = s_cclm[CCLM_ABOVE] != 0, leftAvail = s_cclm[CCLM_LEFT] != 0;
  const bool aboveAvailFill = aboveAvail, leftAvailFill = leftAvail;   // (the downsampling's padding)
  const int  unit = 2;   // baseUnitSize >> 1: the chroma units of the template's availability
  int actualTop = 0, actualLeft = 0, minDim = 1;
  if( topOnly )
  {
    leftAvail = false;
    actualTop = aboveAvail ? unit * ( W / unit + s_cclm[CCLM_AR_UNITS] ) : 0;
    minDim    = actualTop;
  }
  else if( leftOnly )
  {
    aboveAvail = false;
    actualLeft = leftAvail ? unit * ( H / unit + s_cclm[CCLM_BL_UNITS] ) : 0;
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
      s_tmp[j]       = int16_t( cclmLumaAbove( lx, ly, idx, leftAvailFill, glm ) );
      s_tmp[256 + j] = s_ref[ref][1 + idx];
      sumLuma       += int( s_tmp[j] );
    }
  }
  if( leftAvail )
  {
    cntL = numSteps;
    for( int i = 0; i < numSteps; i++ )
    {
      const int idx = ( i * actualLeft ) / minDim;
      s_tmp[cntT + i]       = int16_t( cclmLumaLeft( lx, ly, idx, aboveAvailFill, glm ) );
      s_tmp[256 + cntT + i] = s_ref[ref][predStride + 1 + idx];
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
  const int m = CCLM_MODEL + slot * CCLM_SLOT;
  s_cclm[m + CCLM_A]  = m1.x;  s_cclm[m + CCLM_B]  = m1.y;  s_cclm[m + CCLM_SHIFT]  = m1.z;
  s_cclm[m + CCLM_A2] = m2.x;  s_cclm[m + CCLM_B2] = m2.y;  s_cclm[m + CCLM_SHIFT2] = m2.z;
  s_cclm[m + CCLM_THRES] = thres;
}

// ----- the prediction (predIntraChromaLM, LBCCP's filterPredInside); `slot`: the model's in s_cclm

int cccmBlockPrediction( Tu t, int x, int y );   // (intra_cccm.glsl)

// the model applied to the co-located block's downsampled sample (i, j) (two models: split at the threshold); a CCCM
// block's prediction
int cclmPredSample( Tu t, int lx, int ly, int i, int j, int maxPel, int slot )
{
  if( WITH_CCCM && ( ( t.lm >> LM_CCCM_SHIFT ) & 3 ) != 0 )
  {
    return cccmBlockPrediction( t, i, j );
  }
  const int  m      = CCLM_MODEL + slot * CCLM_SLOT;
  const int  luma   = cclmLumaInner( lx, ly, i, j, s_cclm[CCLM_LEFT] != 0, s_cclm[CCLM_ABOVE] != 0, ( t.lm >> LM_GLM_SHIFT ) & 15 );
  const bool second = cclmMultiModel( t.dirMode ) && luma > s_cclm[m + CCLM_THRES];
  const int  v = second ? ( ( s_cclm[m + CCLM_A2] * luma ) >> s_cclm[m + CCLM_SHIFT2] ) + s_cclm[m + CCLM_B2]
                        : ( ( s_cclm[m + CCLM_A] * luma ) >> s_cclm[m + CCLM_SHIFT] ) + s_cclm[m + CCLM_B];
  return clamp( v, 0, maxPel );
}

// filterPredInside's padding above (x = 0 .. W - 1) and left (y = 0 .. H - 1) of the prediction: the reconstruction
// where that side is available, else the prediction's first row / column
int cclmInsideAbove( Tu t, int lx, int ly, int x, int maxPel, int slot )
{
  return s_cclm[CCLM_POS_ABOVE] != 0 ? recoAt( t.comp, t.x + x, t.y - 1 ) : cclmPredSample( t, lx, ly, x, 0, maxPel, slot );
}

int cclmInsideLeft( Tu t, int lx, int ly, int y, int maxPel, int slot )
{
  return s_cclm[CCLM_POS_LEFT] != 0 ? recoAt( t.comp, t.x - 1, t.y + y ) : cclmPredSample( t, lx, ly, 0, y, maxPel, slot );
}

// filterPredInside's padded prediction at (x, y), -1 <= x <= W, -1 <= y <= H: the prediction inside and, right of and
// below it, its last column / row; above and left cclmInsideAbove / Left; the three corners the reconstruction where
// available, else the top-left one the mean of its two neighbours, the others the padding next to them
int cclmInsideSample( Tu t, int lx, int ly, int x, int y, int maxPel, int slot )
{
  const int W = t.w, H = t.h;
  if( y < 0 && x < 0 )
  {
    return s_cclm[CCLM_POS_LEFT_ABOVE] != 0 ? recoAt( t.comp, t.x - 1, t.y - 1 )
         : ( cclmInsideAbove( t, lx, ly, 0, maxPel, slot ) + cclmInsideLeft( t, lx, ly, 0, maxPel, slot ) + 1 ) >> 1;
  }
  if( y < 0 )
  {
    return x < W ? cclmInsideAbove( t, lx, ly, x, maxPel, slot )
         : s_cclm[CCLM_POS_ABOVE_RIGHT] != 0 ? recoAt( t.comp, t.x + W, t.y - 1 ) : cclmInsideAbove( t, lx, ly, W - 1, maxPel, slot );
  }
  if( x < 0 )
  {
    return y < H ? cclmInsideLeft( t, lx, ly, y, maxPel, slot )
         : s_cclm[CCLM_POS_LEFT_BELOW] != 0 ? recoAt( t.comp, t.x - 1, t.y + H ) : cclmInsideLeft( t, lx, ly, H - 1, maxPel, slot );
  }
  return cclmPredSample( t, lx, ly, min( x, W - 1 ), min( y, H - 1 ), maxPel, slot );
}

// LBCCP's filtered sample (x, y): the padded prediction around it weighed 1, itself 8, the sum over 16
int cclmInsideFiltered( Tu t, int lx, int ly, int x, int y, int maxPel, int slot )
{
  int sum = 0;
  for( int dy = -1; dy <= 1; dy++ )
  {
    for( int dx = -1; dx <= 1; dx++ )
    {
      sum += ( dx == 0 && dy == 0 ? 8 : 1 ) * cclmInsideSample( t, lx, ly, x + dx, y + dy, maxPel, slot );
    }
  }
  return ( sum + 8 ) >> 4;
}

// the predicted sample (x, y) of CCLM block t: with LBCCP's filter where the CU takes it (Tu::lm's LM_INSIDE)
int cclmPrediction( Tu t, int lx, int ly, int x, int y, int maxPel, int slot )
{
  return ( t.lm & LM_INSIDE ) != 0 ? cclmInsideFiltered( t, lx, ly, x, y, maxPel, slot ) : cclmPredSample( t, lx, ly, x, y, maxPel, slot );
}

// ----- LFNST / NSPT of a CCLM block (JVET-AK0064)

// the CU's prediction of chroma component comp (1: Cb, 2: Cr) at (x, y) for cclmTransformMode's histogram: from s_tmp
// where it holds them (`stored`), else computed again (block t's own component from slot 0, o's from slot 1)
int cclmHistSample( Tu t, Tu o, bool stored, int comp, int x, int y, int maxPel )
{
  if( stored )
  {
    return int( s_tmp[( comp == 1 ? 0 : 1024 ) + y * t.w + x] );
  }
  const int lx = t.x << 1, ly = t.y << 1;
  return comp == t.comp ? cclmPrediction( t, lx, ly, x, y, maxPel, 0 ) : cclmPrediction( o, lx, ly, x, y, maxPel, 1 );
}

// the LFNST / NSPT mode of CCLM block t (DERIVE_CCLM_DIMD): deriveChromaIpmForTransform, which DecCu calls with the Cb
// block - the strongest mode of buildHistogram's 3x3 gradients (no subsampling) at the samples inside the CU's Cb
// prediction, its Cr prediction or both (Tu::lm's LM_HIST_*) - into s_predDimdMode. The block of the other component
// has the same area, its template's and LBCCP's availability is the block's (decodedBefore on both components' owner
// maps; neither reads the CU), its references go into s_ref[1], its model into slot 1 (its slope adjustments from
// Tu::lm); the predictions into s_tmp, Cb's from 0, Cr's from 1024, where they fit (up to 32 x 32; larger blocks, rare,
// compute each sample of the gradients again). All invocations, ending with a barrier; the prediction section predicts
// the block again
void cclmTransformMode( Tu t, int predStride, int maxPel, int lid, int NT )
{
  const int  lx = t.x << 1, ly = t.y << 1, W = t.w, H = t.h;
  const bool histCb = ( t.lm & LM_HIST_CB ) != 0, histCr = ( t.lm & LM_HIST_CR ) != 0;
  const bool own    = t.comp == 1 ? histCb : histCr;
  const bool other  = t.comp == 1 ? histCr : histCb;
  const bool stored = W * H <= 1024;
  Tu o = t;
  o.comp = 3 - t.comp;
  o.lm   = ( t.lm & ~( 0xff | ( 15 << LM_GLM_SHIFT ) ) ) | ( ( t.lm >> LM_OTHER_SHIFT ) & 0xff )
           | ( ( ( t.lm >> LM_OTHER_GLM_SHIFT ) & 15 ) << LM_GLM_SHIFT );
  if( other )
  {
    fillReferencePar( o, 0, 1, predStride, lid, NT );
  }
  if( lid == 0 )
  {
    cclmAvailability( t );
  }
  SYNC();
  if( lid == 0 )   // (cclmModel's samples share s_tmp: one after the other)
  {
    if( own )
    {
      cclmModel( t, lx, ly, predStride, 0, 0 );
    }
    if( other )
    {
      cclmModel( o, lx, ly, predStride, 1, 1 );
    }
  }
  SYNC();
  for( int i = lid; i < W * H && stored; i += NT )
  {
    const int x = i % W, y = i / W;
    if( own )
    {
      s_tmp[( t.comp == 1 ? 0 : 1024 ) + i] = int16_t( cclmPrediction( t, lx, ly, x, y, maxPel, 0 ) );
    }
    if( other )
    {
      s_tmp[( o.comp == 1 ? 0 : 1024 ) + i] = int16_t( cclmPrediction( o, lx, ly, x, y, maxPel, 1 ) );
    }
  }
  for( int i = lid; i < NUM_LUMA_MODE; i += NT )
  {
    s_dimdHist[i] = 0;
  }
  SYNC();
  const int iw = W - 2, ih = H - 2, inner = iw * ih;
  for( int k = lid; k < 2 * inner; k += NT )
  {
    const int comp = k < inner ? 1 : 2, i = k % inner, x = 1 + i % iw, y = 1 + i / iw;
    if( comp == 1 ? histCb : histCr )
    {
      const ivec2 g = sobel3x3( cclmHistSample( t, o, stored, comp, x - 1, y - 1, maxPel ),
                                cclmHistSample( t, o, stored, comp, x, y - 1, maxPel ),
                                cclmHistSample( t, o, stored, comp, x + 1, y - 1, maxPel ),
                                cclmHistSample( t, o, stored, comp, x - 1, y, maxPel ),
                                cclmHistSample( t, o, stored, comp, x + 1, y, maxPel ),
                                cclmHistSample( t, o, stored, comp, x - 1, y + 1, maxPel ),
                                cclmHistSample( t, o, stored, comp, x, y + 1, maxPel ),
                                cclmHistSample( t, o, stored, comp, x + 1, y + 1, maxPel ) );
      if( g.x != 0 || g.y != 0 )
      {
        atomicAdd( s_dimdHist[dimdAngle( g )], abs( g.x ) + abs( g.y ) );
      }
    }
  }
  SYNC();
  if( lid == 0 )   // the first of equal amplitudes
  {
    int firstAmp = 0, first = 0;
    for( int i = 0; i < NUM_LUMA_MODE; i++ )
    {
      if( s_dimdHist[i] > firstAmp )
      {
        firstAmp = s_dimdHist[i];
        first    = i;
      }
    }
    s_predDimdMode = first;
  }
  SYNC();
}

#endif
