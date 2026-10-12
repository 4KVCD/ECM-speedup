// Template predictions of TIMD's kind (IntraPrediction::initTimdIntraPatternLuma, initPredTimdIntraParams,
// predTimdIntraAng) for a luma template with both arms, and their SATD costs (RdCost::xGetHADs_SIMD), which intra merge
// ranks its candidates with. The template block is the CU extended by tw columns to the left and th rows above
// (W + tw by H + th); the template is its top th rows right of the corner and its left tw columns below the corner. Its
// reference line, one sample further out, is filled into s_ref[0] (the row: (W + tw) * 8 + 1 samples from the corner)
// and s_ref[1] (the column: (H + th) * 8 + 1). Modes are of the 131-mode set (MAP67TO131). Included by intra_rec.comp.
#ifndef INTRA_TIMD_GLSL
#define INTRA_TIMD_GLSL

// initPredTimdIntraParams' angle tables (1/64 sample per row; the inverse: 512 * 64 / angle)
const int EXT_ANG_TABLE[64]     = int[64]( 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
                                           36, 38, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 74, 78, 84, 90, 96, 102,
                                           108, 114, 121, 128, 137, 146, 159, 172, 188, 204, 230, 256, 299, 342, 427, 512,
                                           597, 682, 853, 1024, 1536, 2048, 3072 );
const int EXT_INV_ANG_TABLE[64] = int[64]( 0, 32768, 16384, 10923, 8192, 6554, 5461, 4681, 4096, 3277, 2731, 2341, 2048,
                                           1820, 1638, 1489, 1365, 1260, 1170, 1092, 1024, 964, 910, 862, 819, 762, 712,
                                           669, 630, 596, 565, 537, 512, 489, 468, 443, 420, 390, 364, 341, 321, 303, 287,
                                           271, 256, 239, 224, 206, 191, 174, 161, 142, 128, 110, 96, 77, 64, 55, 48, 38,
                                           32, 21, 16, 11 );
const int EXT_HOR_IDX  = 34;
const int EXT_DIA_IDX  = 66;
const int EXT_VER_IDX  = 98;
const int EXT_VDIA_IDX = 130;

// MAP67TO131
int timdMode131( int mode )
{
  return mode < 2 ? mode : ( mode << 1 ) - 2;
}

// ----- the reference line

// the availability bits of the reference's units (timdFillReference copies them from the merge record): at most 2 x
// (8 x 132 / 4) + 3 units of 4 (the host takes intra merge CUs up to 128 x 128 only)
shared uint s_timdUnits[17];

bool timdUnitAvailable( int unit )
{
  return ( ( s_timdUnits[unit >> 5] >> ( unit & 31 ) ) & 1u ) != 0u;
}

// the unit of sample j of the row / i of the column (xFillTimdReferenceSamples' order: the left units from the bottom up
// to the corner unit C, which holds the corner and the samples beside the template's arms, then the above units)
int timdTopUnit( MergeHead h, int j )
{
  const int c = h.totalLeftUnits + h.leftTempUnitNum, corner = h.tw - h.aboveTempUnitNum * h.unitW;
  return j <= corner ? c : c + 1 + ( j - corner - 1 ) / h.unitW;
}

int timdLeftUnit( MergeHead h, int i )
{
  const int c = h.totalLeftUnits + h.leftTempUnitNum, corner = h.th - h.leftTempUnitNum * h.unitH;
  return i <= corner ? c : c - 1 - ( i - corner - 1 ) / h.unitH;
}

// xFillTimdReferenceSamples' step 2 for the template of CU (cx, cy, W, H), all invocations (ends with a barrier): the
// samples of the available units from the reconstruction (in parallel), then the CPU's padding of the others in its
// order (one invocation: each unit takes the last sample before it); without any unit the mid-level value
void timdFillReference( MergeHead h, int flagsBase, int cx, int cy, int W, int H, int bitDepth, int lid, int NT )
{
  const int predSize = ( W + h.tw ) << 3, predHSize = ( H + h.th ) << 3;
  const int x0 = cx - h.tw - 1, y0 = cy - h.th - 1;
  if( lid < ( h.totalUnits + 31 ) >> 5 )
  {
    s_timdUnits[lid] = uint( BufI32( P.mergeData ).v[flagsBase + lid] );
  }
  SYNC();
  for( int j = lid; j <= predSize; j += NT )
  {
    s_ref[0][j] = int16_t( h.numIntraNeighbor == 0 ? 1 << ( bitDepth - 1 )
                           : timdUnitAvailable( timdTopUnit( h, j ) ) ? recoAt( 0, x0 + j, y0 ) : 0 );
  }
  for( int i = lid; i <= predHSize; i += NT )
  {
    s_ref[1][i] = int16_t( h.numIntraNeighbor == 0 ? 1 << ( bitDepth - 1 )
                           : timdUnitAvailable( timdLeftUnit( h, i ) ) ? recoAt( 0, x0, y0 + i ) : 0 );
  }
  SYNC();
  if( lid == 0 && h.numIntraNeighbor != 0 && h.numIntraNeighbor != h.totalUnits )
  {
    const int c = h.totalLeftUnits + h.leftTempUnitNum, uw = h.unitW, uh = h.unitH;
    const int cornerCol = h.tw - h.aboveTempUnitNum * uw, cornerRow = h.th - h.leftTempUnitNum * uh;
    int lastAvailUnit = 0;
    if( !timdUnitAvailable( 0 ) )   // from the first available sample down to the last below-left one
    {
      int firstAvailUnit = 1;
      while( firstAvailUnit < h.totalUnits && !timdUnitAvailable( firstAvailUnit ) )
      {
        firstAvailUnit++;
      }
      int firstAvailRow = -1, firstAvailCol = 0;
      if( firstAvailUnit < c )
      {
        firstAvailRow = ( c - firstAvailUnit ) * uh + cornerRow;
      }
      else if( firstAvailUnit == c )
      {
        firstAvailRow = cornerRow;
      }
      else
      {
        firstAvailCol = ( firstAvailUnit - c - 1 ) * uw + 1 + cornerCol;
      }
      const int16_t v = firstAvailRow < 0 ? s_ref[0][firstAvailCol] : s_ref[1][firstAvailRow];
      for( int i = predHSize; i > firstAvailRow; i-- )
      {
        s_ref[1][i] = v;
      }
      for( int j = 0; j < firstAvailCol; j++ )
      {
        s_ref[0][j] = v;
      }
      lastAvailUnit = firstAvailUnit;
    }
    for( int unit = lastAvailUnit + 1; unit < h.totalUnits; unit++ )   // each unit after it from the one before
    {
      if( !timdUnitAvailable( unit ) )
      {
        int lastAvailRow = -1, lastAvailCol = 0;
        if( lastAvailUnit < c )
        {
          lastAvailRow = ( c - lastAvailUnit - 1 ) * uh + cornerRow + 1;
        }
        else
        {
          lastAvailCol = ( lastAvailUnit - c ) * uw + cornerCol;
        }
        const int16_t v = lastAvailRow < 0 ? s_ref[0][lastAvailCol] : s_ref[1][lastAvailRow];
        if( unit < c )
        {
          for( int i = lastAvailRow - 1; i >= lastAvailRow - uh; i-- )
          {
            s_ref[1][i] = v;
          }
        }
        else if( unit == c )
        {
          for( int i = 0; i <= cornerRow; i++ )
          {
            s_ref[1][i] = v;
          }
          for( int j = 0; j <= cornerCol; j++ )
          {
            s_ref[0][j] = v;
          }
        }
        else
        {
          const int rest = ( predSize - h.tw ) % uw, num = unit == h.totalUnits - 1 && rest != 0 ? rest : uw;
          for( int j = lastAvailCol + 1; j <= lastAvailCol + num; j++ )
          {
            s_ref[0][j] = v;
          }
        }
      }
      lastAvailUnit = unit;
    }
  }
  SYNC();
}

// ----- the prediction

// initPredTimdIntraParams for a CU of W x H (the angular ones of an angular mode)
struct TimdParams
{
  bool isModeVer;
  int  angle;          // intraPredAngle, 1/64 sample per row
  int  invAngle;       // absInvAngle
  bool pdpc;           // applyPDPC
  int  angularScale;
  bool gradPdpc;       // useGradPDPC (positive angles)
};

// IntraPrediction::getWideAngleExt (without SGPM)
int timdWideAngle( int W, int H, int mode )
{
  if( mode > 1 && mode <= EXT_VDIA_IDX )
  {
    const int modeShift[6] = int[6]( 0, 11, 19, 23, 27, 29 );
    const int deltaSize    = abs( findMSB( W ) - findMSB( H ) );
    if( W > H && mode < 2 + modeShift[deltaSize] )
    {
      mode += EXT_VDIA_IDX - 1;
    }
    else if( H > W && mode > EXT_VDIA_IDX - modeShift[deltaSize] )
    {
      mode -= EXT_VDIA_IDX - 1;
    }
  }
  return mode;
}

TimdParams timdParams( int mode, int W, int H, bool noPdpc )
{
  TimdParams p;
  const int predMode  = timdWideAngle( W, H, mode );
  p.isModeVer         = predMode >= EXT_DIA_IDX;
  const int angleMode = p.isModeVer ? predMode - EXT_VER_IDX : -( predMode - EXT_HOR_IDX );
  const int absMode   = abs( angleMode );
  p.angle             = angleMode < 0 ? -EXT_ANG_TABLE[absMode] : EXT_ANG_TABLE[absMode];
  p.invAngle          = EXT_INV_ANG_TABLE[absMode];
  p.pdpc              = W >= 4 && H >= 4;
  p.angularScale      = 0;
  p.gradPdpc          = false;
  if( mode > 1 && angleMode < 0 )
  {
    p.pdpc = false;
  }
  else if( mode > 1 && angleMode > 0 )
  {
    p.angularScale = min( 2, findMSB( p.isModeVer ? H : W ) - ( findMSB( 3 * p.invAngle - 2 ) - 8 ) );
    if( p.angularScale < 0 )
    {
      p.angularScale = ( findMSB( W ) + findMSB( H ) - 2 ) >> 2;
      p.gradPdpc     = true;
    }
  }
  p.pdpc = p.pdpc && !noPdpc;
  return p;
}

// the reference row (top) or column at index k
int timdTop( int k )
{
  return int( s_ref[0][k] );
}

int timdLeft( int k )
{
  return int( s_ref[1][k] );
}

// xPredTimdIntraAng's refSide and refMain at index k: negative angles extend refMain below 0 by projecting refSide
// (the block's sides wT x hT, the template block's), positive ones repeat its last sample beyond the reference length
int timdSide( TimdParams p, int k )
{
  return p.isModeVer ? timdLeft( k ) : timdTop( k );
}

int timdMain( TimdParams p, int k, int wT, int hT )
{
  if( p.angle < 0 && k < 0 )
  {
    const int sizeSide = p.isModeVer ? hT : wT;
    return timdSide( p, min( ( -k * p.invAngle + 256 ) >> 9, sizeSide ) );
  }
  const int refLength = ( p.isModeVer ? wT : hT ) << 3;
  const int idx = p.angle >= 0 ? min( k, refLength ) : k;
  return p.isModeVer ? timdTop( idx ) : timdLeft( idx );
}

// a workgroup sum of a, all invocations (uniform control flow), through s_red's first two slots: zeroed before, free
// again after
int timdWgSum( int a, int lid )
{
  if( lid == 0 )
  {
    s_red[0] = 0;
    s_red[1] = 0;
  }
  SYNC();
  const int s = wgSum2( a, 0, 0 ).x;
  SYNC();
  return s;
}

// xGetPredTimdValDc for a template with both arms: the mean of the reference row above the CU and / or the column left
// of it, all invocations
int timdDc( MergeHead h, int W, int H, int lid, int NT )
{
  int sum = 0;
  for( int k = lid; W >= H && k < W; k += NT ) sum += timdTop( h.tw + 1 + k );
  for( int k = lid; W <= H && k < H; k += NT ) sum += timdLeft( h.th + 1 + k );
  const int denom = W == H ? W << 1 : max( W, H );
  return ( timdWgSum( sum, lid ) + ( denom >> 1 ) ) >> findMSB( denom );
}

// sample (x, y) of the template (the template block's coordinates) predicted with mode (131) of parameters p
// (predTimdIntraAng: planar / DC with their PDPC, angular with the 1/64 cubic filter and its PDPC kinds, clipped as the CPU
// clips); dc: timdDc's value
int timdPredict( int mode, TimdParams p, int x, int y, int W, int H, int tw, int th, int dc, int maxPel )
{
  const int wT = W + tw, hT = H + th;
  int v;
  if( mode <= 1 )
  {
    if( mode == 0 )   // xPredTimdIntraPlanar: the template above and the template left as two planar blocks
    {
      const bool above = y < th;
      const int  bw = above ? W : tw, bh = above ? th : H;
      const int  bx = above ? x - tw : x, by = above ? y : y - th;
      const int  log2W = findMSB( bw ), log2H = findMSB( bh );
      const int  left0 = above ? 1 : th + 1, top0 = above ? tw + 1 : 1;   // the block's left column / top row start
      const int  leftCol = timdLeft( left0 + by ), topRow = timdTop( top0 + bx );
      const int  topRight = timdTop( top0 + bw ), bottomLeft = timdLeft( left0 + bh );
      const int  hor = ( leftCol << log2W ) + ( bx + 1 ) * ( topRight - leftCol );
      const int  ver = ( topRow << log2H ) + ( by + 1 ) * ( bottomLeft - topRow );
      v = ( ( hor << log2H ) + ( ver << log2W ) + ( 1 << ( log2W + log2H ) ) ) >> ( 1 + log2W + log2H );
    }
    else
    {
      v = dc;
    }
    if( p.pdpc )   // xIntraPredTimdPlanarDcPdpc
    {
      const int scale = ( findMSB( wT ) - 2 + findMSB( hT ) - 2 + 2 ) >> 2;
      const int wTop  = 32 >> min( 31, ( y << 1 ) >> scale ), wL = 32 >> min( 31, ( x << 1 ) >> scale );
      v = v + ( ( wL * ( timdLeft( y + 1 ) - v ) + wTop * ( timdTop( x + 1 ) - v ) + 32 ) >> 6 );
    }
    return v;
  }
  // angular: in the transposed block for horizontal modes (rows along the main reference)
  const int xt = p.isModeVer ? x : y, yt = p.isModeVer ? y : x;
  const int wt = p.isModeVer ? wT : hT, ht = p.isModeVer ? hT : wT;
  const int twt = p.isModeVer ? tw : th, tht = p.isModeVer ? th : tw;
  const bool upper = yt < tht;   // the region of the transposed template: rows of the arm along the main reference
  if( p.angle == 0 )
  {
    v = timdMain( p, xt + 1, wT, hT );
    const int scale = ( findMSB( wt ) + findMSB( ht ) - 2 ) >> 2;
    if( p.pdpc && xt < ( 3 << scale ) )   // xIntraPredTimdHorVerPdpc
    {
      const int wL = 32 >> ( ( 2 * xt ) >> scale );
      v = clamp( v + ( ( wL * ( timdSide( p, 1 + yt ) - timdMain( p, 0, wT, hT ) ) + 32 ) >> 6 ), 0, maxPel );
    }
    return v;
  }
  const int deltaPos = p.angle * ( yt + 1 ), deltaInt = deltaPos >> 6, frac = deltaPos & 63;
  if( ( abs( p.angle ) & 63 ) != 0 )   // xIntraPredTimdAngLuma
  {
    const int idx = deltaInt + 1 + xt;
    BufI32    f   = BufI32( P.filters );
    int sum = 128;
    for( int k = 0; k < 4; k++ )
    {
      sum += f.v[EXT_CUBIC + frac * 4 + k] * timdMain( p, idx - 1 + k, wT, hT );
    }
    v = clamp( sum >> 8, 0, maxPel );
  }
  else
  {
    v = timdMain( p, xt + deltaInt + 1, wT, hT );
  }
  if( !p.pdpc )
  {
    return v;
  }
  const int xOffset = upper ? twt : 0, regionW = upper ? wt : twt, scale = p.angularScale;
  if( p.gradPdpc )   // xIntraPredTimdAngGradPdpc (the weight from the region's start)
  {
    if( xt < min( 3 << scale, regionW ) )
    {
      const int wL      = 32 >> ( ( 2 * ( xt - xOffset ) ) >> scale );
      const int m0      = timdMain( p, deltaInt, wT, hT ), m1 = timdMain( p, deltaInt + 1, wT, hT );
      const int topLeft = m0 + ( ( frac * ( m1 - m0 ) + 32 ) >> 6 );
      v = clamp( v + ( ( wL * ( timdSide( p, 1 + yt ) - topLeft ) + 32 ) >> 6 ), 0, maxPel );
    }
  }
  else if( xt < ( regionW < 4 ? 2 : min( 3 << scale, regionW ) ) )   // xIntraPredTimdAngPdpc (no clip)
  {
    const int invAngleSum = 256 + ( xt - xOffset + 1 ) * p.invAngle;
    const int wL          = 32 >> ( ( 2 * xt ) >> scale );
    const int left        = timdSide( p, yt + ( invAngleSum >> 9 ) + 1 );
    v = v + ( ( wL * ( left - v ) + 32 ) >> 6 );
  }
  return v;
}

// ----- the SATD (RdCost::xGetHADs_SIMD's kernels for the template's arms: 2x2, 4x4, 8x4, 4x8)

// the difference of the original (the reconstruction of the template) and the prediction at (x, y) of an arm
// (s_tmp from predOffset, rows of w samples; the original from (ox, oy))
int timdDiff( int ox, int oy, int predOffset, int w, int x, int y )
{
  return recoAt( 0, ox + x, oy + y ) - int( s_tmp[predOffset + y * w + x] );
}

// (int)( sad / sqrt( 4.0 * 8 ) * 2 ) of the 8x4 / 4x8 kernels: floor( sad / sqrt( 8 ) ) = the largest q with 8 q^2 <=
// sad^2 (sad below 2^24: the double expression has no rounding that crosses an integer, checked exhaustively)
int timdScaleSqrt8( int sad )
{
  const int64_t s2 = int64_t( sad ) * int64_t( sad );
  int q = int( float( sad ) * 0.35355339f );
  while( 8 * int64_t( q + 1 ) * int64_t( q + 1 ) <= s2 ) q++;
  while( q > 0 && 8 * int64_t( q ) * int64_t( q ) > s2 ) q--;
  return q;
}

// the SATD of block (bx, by) of size kw x kh (2x2, 4x4, 8x4 or 4x8) of an arm
int timdHad( int ox, int oy, int predOffset, int w, int bx, int by, int kw, int kh )
{
  int d[32];
  for( int j = 0; j < kh; j++ )
  {
    for( int i = 0; i < kw; i++ )
    {
      d[j * kw + i] = timdDiff( ox, oy, predOffset, w, bx + i, by + j );
    }
  }
  if( kw == 2 )   // xCalcHADs2x2 (the DC term a quarter: JVET-R0164)
  {
    const int m0 = d[0] + d[2], m1 = d[1] + d[3], m2 = d[0] - d[2], m3 = d[1] - d[3];
    return ( abs( m0 + m1 ) >> 2 ) + abs( m0 - m1 ) + abs( m2 + m3 ) + abs( m2 - m3 );
  }
  // the Hadamard transform of rows of kw, then of columns of kh, in place (the order of the butterflies does not
  // change the absolute values: each output is a signed sum of the inputs)
  int t[32];
  for( int j = 0; j < kh; j++ )
  {
    for( int i = 0; i < kw; i++ )
    {
      int s = 0;
      for( int k = 0; k < kw; k++ ) s += ( bitCount( uint( i & k ) ) & 1 ) != 0 ? -d[j * kw + k] : d[j * kw + k];
      t[j * kw + i] = s;
    }
  }
  int sad = 0, dc = 0;
  for( int i = 0; i < kw; i++ )
  {
    for( int j = 0; j < kh; j++ )
    {
      int s = 0;
      for( int k = 0; k < kh; k++ ) s += ( bitCount( uint( j & k ) ) & 1 ) != 0 ? -t[k * kw + i] : t[k * kw + i];
      sad += abs( s );
      dc = i == 0 && j == 0 ? abs( s ) : dc;
    }
  }
  sad = sad - dc + ( dc >> 2 );
  return kw == kh ? ( sad + 1 ) >> 1 : timdScaleSqrt8( sad );
}

// the kernel xGetHADs_SIMD takes for an arm of w x h (the template's arms are 2 or 4 thin)
ivec2 timdKernel( int w, int h )
{
  return w > h && ( w & 7 ) == 0 && ( h & 3 ) == 0 ? ivec2( 8, 4 )
       : w < h && ( h & 7 ) == 0 && ( w & 3 ) == 0 ? ivec2( 4, 8 )
       : ( h & 3 ) == 0 && ( w & 3 ) == 0         ? ivec2( 4, 4 ) : ivec2( 2, 2 );
}

#endif
