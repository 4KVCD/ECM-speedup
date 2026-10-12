// Template predictions of TIMD's kind (IntraPrediction::initTimdIntraPatternLuma, initPredTimdIntraParams,
// predTimdIntraAng) for a luma template, and their SATD costs (RdCost::xGetHADs_SIMD), which intra merge ranks its
// candidates with. The template block is the CU extended by tw columns to the left and th rows above (W + tw by H +
// th; an arm the template lacks is 0, CU::deriveTimdRefType's ABOVE_NEIGHBOR / LEFT_NEIGHBOR at the picture's left /
// top edge); the template is its top th rows right of the corner and its left tw columns below the corner. Its
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

// MAP67TO131, MAP131TO67
int timdMode131( int mode )
{
  return mode < 2 ? mode : ( mode << 1 ) - 2;
}

int timdMode67( int mode )
{
  return mode < 2 ? mode : ( mode >> 1 ) + 1;
}

// ----- the reference line

// the availability bits of the reference's units (timdFillReference copies them from the CU's record): at most 2 x
// (8 x 132 / 4) + 3 units of 4 (the host takes CUs up to 128 x 128 only)
shared uint s_timdUnits[17];

bool timdUnitAvailable( int unit )
{
  return ( ( s_timdUnits[unit >> 5] >> ( unit & 31 ) ) & 1u ) != 0u;
}

// the unit of sample j of the row / i of the column (xFillTimdReferenceSamples' order: the left units from the bottom up
// to the corner unit C, which holds the corner and the samples beside the template's arms, then the above units)
int timdTopUnit( TemplateUnits t, int j )
{
  const int c = t.totalLeftUnits + t.leftTempUnitNum, corner = t.tw - t.aboveTempUnitNum * t.unitW;
  return j <= corner ? c : c + 1 + ( j - corner - 1 ) / t.unitW;
}

int timdLeftUnit( TemplateUnits t, int i )
{
  const int c = t.totalLeftUnits + t.leftTempUnitNum, corner = t.th - t.leftTempUnitNum * t.unitH;
  return i <= corner ? c : c - 1 - ( i - corner - 1 ) / t.unitH;
}

// xFillTimdReferenceSamples' step 2 for template t of CU (cx, cy, W, H), its units' availability bits at flags.v[flagsBase],
// all invocations (ends with a barrier): the samples of the available units from the reconstruction (in parallel),
// then the CPU's padding of the others in its order (one invocation: each unit takes the last sample before it);
// without any unit the mid-level value
void timdFillReference( TemplateUnits t, BufI32 flags, int flagsBase, int cx, int cy, int W, int H, int bitDepth, int lid, int NT )
{
  const int predSize = ( W + t.tw ) << 3, predHSize = ( H + t.th ) << 3;
  const int x0 = cx - t.tw - 1, y0 = cy - t.th - 1;
  if( lid < ( t.totalUnits + 31 ) >> 5 )
  {
    s_timdUnits[lid] = uint( flags.v[flagsBase + lid] );
  }
  SYNC();
  for( int j = lid; j <= predSize; j += NT )
  {
    s_ref[0][j] = int16_t( t.numIntraNeighbor == 0 ? 1 << ( bitDepth - 1 )
                           : timdUnitAvailable( timdTopUnit( t, j ) ) ? recoAt( 0, x0 + j, y0 ) : 0 );
  }
  for( int i = lid; i <= predHSize; i += NT )
  {
    s_ref[1][i] = int16_t( t.numIntraNeighbor == 0 ? 1 << ( bitDepth - 1 )
                           : timdUnitAvailable( timdLeftUnit( t, i ) ) ? recoAt( 0, x0, y0 + i ) : 0 );
  }
  SYNC();
  if( lid == 0 && t.numIntraNeighbor != 0 && t.numIntraNeighbor != t.totalUnits )
  {
    const int c = t.totalLeftUnits + t.leftTempUnitNum, uw = t.unitW, uh = t.unitH;
    const int cornerCol = t.tw - t.aboveTempUnitNum * uw, cornerRow = t.th - t.leftTempUnitNum * uh;
    int lastAvailUnit = 0;
    if( !timdUnitAvailable( 0 ) )   // from the first available sample down to the last below-left one
    {
      int firstAvailUnit = 1;
      while( firstAvailUnit < t.totalUnits && !timdUnitAvailable( firstAvailUnit ) )
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
    for( int unit = lastAvailUnit + 1; unit < t.totalUnits; unit++ )   // each unit after it from the one before
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
          const int rest = ( predSize - t.tw ) % uw, num = unit == t.totalUnits - 1 && rest != 0 ? rest : uw;
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

// initPredTimdIntraParams for a CU of W x H (the angular ones of an angular mode, unused for planar / DC)
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

// TIMD's wide angles beyond the regular ones (JVET-AC0094: deriveTimdMode tries them where the reference reaches, and a
// TIMD CU may take one): getTimdWideAngleExt, the angle of a mode of the 131-mode set, and getTimdRegularAngleExt, the
// mode of an angle (its inverse but at the ends of the range)
const int TIMD_MODE_SHIFT[6] = int[6]( 0, 11, 19, 23, 27, 29 );

int timdWaipAngle( int W, int H, int mode )
{
  const int maxIndex = TIMD_MODE_SHIFT[min( abs( findMSB( W ) - findMSB( H ) ), 5 )];
  if( W > H && mode < 2 + maxIndex )   // a regular mode wide angles replace: the one before it
  {
    return mode == EXT_VDIA_IDX ? 2 : mode + 1;
  }
  if( H > W && mode > EXT_VDIA_IDX - maxIndex )
  {
    return mode == 2 ? EXT_VDIA_IDX : mode - 1;
  }
  // a mode the block's wide angles do not take: the angle beyond the other end of the range
  return mode > EXT_DIA_IDX ? mode - ( EXT_VDIA_IDX - 2 ) : mode < EXT_DIA_IDX ? mode + ( EXT_VDIA_IDX - 2 ) : mode;
}

int timdRegularAngle( int W, int H, int angle )
{
  const int maxIndex = TIMD_MODE_SHIFT[min( abs( findMSB( W ) - findMSB( H ) ) + 2, 5 )];
  if( angle == EXT_VDIA_IDX || angle == 2 )
  {
    return angle == EXT_VDIA_IDX ? 2 : EXT_VDIA_IDX;
  }
  if( W > H && angle <= 2 + maxIndex && angle > 1 )
  {
    return angle - 1;
  }
  if( H > W && angle >= EXT_VDIA_IDX - maxIndex && angle <= EXT_VDIA_IDX )
  {
    return angle + 1;
  }
  return angle > EXT_VER_IDX ? angle - ( EXT_VDIA_IDX - 2 ) : angle < EXT_HOR_IDX ? angle + ( EXT_VDIA_IDX - 2 ) : angle;
}

// the parameters of mode dirMode, its wide angle getWideAngleExt's (checkWA, every mode but TIMD's beyond the regular
// angles) or getTimdWideAngleExt's
TimdParams timdParams( int dirMode, int W, int H, bool noPdpc, bool checkWA )
{
  TimdParams p;
  const int predMode  = checkWA ? timdWideAngle( W, H, dirMode ) : timdWaipAngle( W, H, dirMode );
  p.isModeVer         = predMode >= EXT_DIA_IDX;
  const int angleMode = p.isModeVer ? predMode - EXT_VER_IDX : -( predMode - EXT_HOR_IDX );
  const int absMode   = abs( angleMode );
  p.angle             = angleMode < 0 ? -EXT_ANG_TABLE[absMode] : EXT_ANG_TABLE[absMode];
  p.invAngle          = EXT_INV_ANG_TABLE[absMode];
  p.pdpc              = W >= 4 && H >= 4;
  p.angularScale      = 0;
  p.gradPdpc          = false;
  if( dirMode > 1 && angleMode < 0 )
  {
    p.pdpc = false;
  }
  else if( dirMode > 1 && angleMode > 0 )
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

// xGetPredTimdValDc for template t of a CU of W x H, all invocations: with one arm the mean of the reference beside it
// (the row above an above-only template, the column left of a left-only one), with both the mean of the row above the
// CU and / or the column left of it, as for a block
int timdDc( TemplateUnits t, int W, int H, int lid, int NT )
{
  const bool top = t.tw == 0 || ( t.th > 0 && W >= H ), left = t.th == 0 || ( t.tw > 0 && W <= H );
  int sum = 0;
  for( int k = lid; top && k < W; k += NT ) sum += timdTop( t.tw + 1 + k );
  for( int k = lid; left && k < H; k += NT ) sum += timdLeft( t.th + 1 + k );
  const int denom = t.tw == 0 ? W : t.th == 0 ? H : W == H ? W << 1 : max( W, H );
  return ( timdWgSum( sum, lid ) + ( denom >> 1 ) ) >> findMSB( denom );
}

// sample (x, y) of the template (the template block's coordinates) predicted with mode (131) of parameters p
// (predTimdIntraAng: planar / DC with their PDPC, angular - any other mode, also WAIP's wide angles below 0 - with the
// 1/64 cubic filter and its PDPC kinds, clipped as the CPU clips); dc: timdDc's value
int timdPredict( int mode, TimdParams p, int x, int y, int W, int H, int tw, int th, int dc, int maxPel )
{
  const int wT = W + tw, hT = H + th;
  int v;
  if( mode == 0 || mode == 1 )
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

// ----- the costs of a list of candidates

// a candidate (a word of s_spCost): the mode predicted, the mode of its parameters and whether they take
// getWideAngleExt's wide angles (timdParams), and the slot its costs are added to
uint timdCand( int mode, int dirMode, bool checkWA, int slot )
{
  return uint( mode + 256 ) | uint( dirMode ) << 10 | ( checkWA ? 1u : 0u ) << 18 | uint( slot ) << 19;
}

int  timdCandMode( uint w ) { return int( w & 1023u ) - 256; }
int  timdCandDir( uint w )  { return int( ( w >> 10 ) & 255u ); }
bool timdCandWA( uint w )   { return ( w & ( 1u << 18 ) ) != 0u; }
int  timdCandSlot( uint w ) { return int( w >> 19 ); }

// the template costs of the numCands candidates at s_spCost[candBase] for CU c with template t (all invocations; the
// references filled, dc timdDc's): each arm's SATD added to its slot (the above arm's at costA, the left arm's at costL;
// zeroed by the caller), as many candidates at once as s_tmp holds their templates (the above arm's rows of c.w, then the
// left arm's rows of tw), predicted in parallel, then the SATD of the reconstruction against each, block by block in
// parallel
void timdTemplateCosts( Cu c, TemplateUnits t, int candBase, int numCands, int costA, int costL, int dc, int maxPel, bool noPdpc,
                        int lid, int NT )
{
  const int   nA = c.w * t.th, T = nA + t.tw * c.h;   // template samples per candidate
  const int   perPass = max( 1, MAX_REF / T );
  const ivec2 ka = timdKernel( c.w, t.th ), kl = timdKernel( t.tw, c.h );
  const int   colsA = c.w / ka.x, colsL = t.tw / kl.x;
  const int   nbA = colsA * ( t.th / ka.y ), B = nbA + colsL * ( c.h / kl.y );   // SATD blocks per candidate
  for( int first = 0; first < numCands; first += perPass )
  {
    const int n = min( perPass, numCands - first );
    for( int i = lid; i < n * T; i += NT )
    {
      const int        s    = i % T;
      const uint       cand = s_spCost[candBase + first + i / T];
      const TimdParams p    = timdParams( timdCandDir( cand ), c.w, c.h, noPdpc, timdCandWA( cand ) );
      const int x = s < nA ? t.tw + s % c.w : ( s - nA ) % t.tw;
      const int y = s < nA ? s / c.w : t.th + ( s - nA ) / t.tw;
      s_tmp[i] = int16_t( timdPredict( timdCandMode( cand ), p, x, y, c.w, c.h, t.tw, t.th, dc, maxPel ) );
    }
    SYNC();
    for( int i = lid; i < n * B; i += NT )
    {
      const int m = i / B, b = i % B, k = b - nbA;
      const int satd = b < nbA ? timdHad( c.x, c.y - t.th, m * T, c.w, ( b % colsA ) * ka.x, ( b / colsA ) * ka.y, ka.x, ka.y )
                               : timdHad( c.x - t.tw, c.y, m * T + nA, t.tw, ( k % colsL ) * kl.x, ( k / colsL ) * kl.y, kl.x, kl.y );
      atomicAdd( s_spCost[( b < nbA ? costA : costL ) + timdCandSlot( s_spCost[candBase + first + m] )], uint( satd ) );
    }
    SYNC();
  }
}

// ----- a TIMD CU's node (IntraPrediction::deriveTimdMode without TIMD-SAD, TIMD merge and block vectors): the template
// costs of its candidates in two batches - planar, DC, the angular modes of its MPM list and the wide angles beyond the
// regular ones, then the neighbours of the best and the secondary mode - computed in parallel, the CPU's sequential
// choice among them replayed by invocation 0 (with its early ends: computing more costs than the CPU does changes no
// choice), then the fusion weights

// in s_spCost (free in a node): the candidates (timdCand words; a first-batch candidate's cost slot is its index, a
// second-batch one's numCands + its neighbour index, so that an invalid neighbour leaves its slot unused) and their
// arms' costs
const int TIMD_CANDS = 64;                        // at most 2 + 21 + 2 + the wide angles beyond (below 12), + 4
const int TC_CAND    = 0;
const int TC_COST_A  = TC_CAND + TIMD_CANDS;      // the above arm's (tmpCost0)
const int TC_COST_L  = TC_COST_A + TIMD_CANDS;    // the left arm's (tmpCost1)
const int TC_NUM     = TC_COST_L + TIMD_CANDS;    // the first batch's candidates
const int TC_NUM_B   = TC_NUM + 1;                // the second batch's candidates
const int TC_END_MPM = TC_NUM + 2;                // the end of the MPM list's modes in the first batch

// deriveTimdMode's running choice (invocation 0): the best and the secondary mode with their costs and whether their
// parameters take getWideAngleExt's wide angles (else getTimdWideAngleExt's: the mode is a wide angle beyond the regular
// ones), the best of planar and DC, the location dependences (timdLocDep)
struct TimdScan
{
  uint64_t best, secondary, nonAng;   // costs (~0: none yet)
  int      bestMode, secMode, nonAngMode;
  bool     bestWA, secWA;
  int      locDep[3];
};

// the location dependence of a mode of arm costs c0 (above) and c1 (left): the arm a template lacks the other's, with
// both the arm whose cost per sample is less than half the other's (1 above, 2 left), else none
int timdLocDep( int type, uint c0, uint c1, int log2A, int log2L )
{
  if( type != TIMD_BOTH )
  {
    return type == TIMD_LEFT ? 2 : 1;
  }
  return ( c0 >> log2A ) < ( c1 >> ( log2L + 1 ) ) ? 1 : ( c1 >> log2L ) < ( c0 >> ( log2A + 1 ) ) ? 2 : 0;
}

// a candidate's cost: its arms' sum
uint64_t timdCost( int slot )
{
  return uint64_t( s_spCost[TC_COST_A + slot] ) + uint64_t( s_spCost[TC_COST_L + slot] );
}

// X0149's weight of the best mode of cost best beside the secondary of cost secondary (best + secondary > 0): the
// secondary cost's share of their sum in 1/64, the division by a table of the reciprocal of the sum's leading bits
int timdLutWeight( uint64_t best, uint64_t secondary )
{
  const int      divTable[16] = int[16]( 0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0 );
  const uint64_t s1           = best + secondary;
  int            x            = s1 >> 32 != 0ul ? 32 + findMSB( uint( s1 >> 32 ) ) : findMSB( uint( s1 ) );
  const int      normS1       = int( ( s1 << 4 ) >> x ) & 15;
  const int      v            = divTable[normS1] | 8;
  x += normS1 != 0 ? 1 : 0;
  const int shift = x + 3;
  return min( 64, int( ( secondary * uint64_t( v ) * 64ul + ( 1ul << ( shift - 1 ) ) ) >> shift ) );
}

// the node of TIMD CU c (all invocations, uniform control flow): its TIMD into its TimdOut; returns (invocation 0) its
// mode as the neighbours read it (MAP131TO67) and in mtssSecond the second mode MTSS tries first (the secondary where
// the CU blends, else -1: none). nbrMode, dimdModes: deriveLumaMode's (invocation 0's)
int timdNode( Cu c, int nbrMode[MPM_NEIGHBOURS], int dimdModes[3], int bitDepth, bool noPdpc, out int mtssSecond, int lid, int NT )
{
  BufI32         td = BufI32( P.timdData );
  const TimdHead h  = TimdHead( td.v[c.timd], td.v[c.timd + 1], td.v[c.timd + 2], templateUnitsAt( td, c.timd + TIMD_HEAD_T ) );
  const int      W = c.w, H = c.h;
  CBufI32        o    = CBufI32( P.timdOut );
  const int      base = h.index * TIMD_OUT;
  mtssSecond = -1;
  if( h.type == TIMD_NONE )   // the picture's top-left CU: planar
  {
    if( lid == 0 )
    {
      for( int k = 0; k < TIMD_OUT; k++ )
      {
        o.v[base + k] = k >= TIMD_OUT_LOC_DEP && k < TIMD_OUT_LOC_DEP + 3 ? 0 : k >= TIMD_OUT_CHECK_WA ? 1 : -1;
      }
      o.v[base + TIMD_OUT_MODE]    = PLANAR_IDX;
      o.v[base + TIMD_OUT_BLENDED] = 0;
    }
    return PLANAR_IDX;
  }
  // the regular angles of the CU's shape (the wide angles of WAIP included), and those beyond its reference reaches,
  // further where PUs lie further out above-right / below-left (deriveTimdMode's increase / decrease Close / Far)
  const int  deltaSize = abs( findMSB( W ) - findMSB( H ) );
  const int  maxOrig   = W >= H ? EXT_VDIA_IDX + TIMD_MODE_SHIFT[deltaSize] : EXT_VDIA_IDX - TIMD_MODE_SHIFT[deltaSize];
  const int  minOrig   = W >= H ? 2 + TIMD_MODE_SHIFT[deltaSize] : 2 - TIMD_MODE_SHIFT[deltaSize];
  const bool maxClose = ( h.waip & TIMD_WAIP_MAX_CLOSE ) != 0, maxFar = ( h.waip & TIMD_WAIP_MAX_FAR ) != 0;
  const bool minClose = ( h.waip & TIMD_WAIP_MIN_CLOSE ) != 0, minFar = ( h.waip & TIMD_WAIP_MIN_FAR ) != 0;
  int newMin = minOrig, newMax = maxOrig;
  if( W >= H && ( maxFar || maxClose ) )
  {
    newMax += TIMD_MODE_SHIFT[min( deltaSize + ( maxFar ? 2 : 1 ), 5 )] - TIMD_MODE_SHIFT[deltaSize];
  }
  if( W <= H && ( minFar || minClose ) )
  {
    newMin -= TIMD_MODE_SHIFT[min( deltaSize + ( minFar ? 2 : 1 ), 5 )] - TIMD_MODE_SHIFT[deltaSize];
  }
  const int extRef = minFar || maxFar ? 2 : minClose || maxClose ? 1 : 0;
  for( int d = 1, previous = deltaSize; d <= extRef; d++ )
  {
    const int step = abs( TIMD_MODE_SHIFT[abs( previous )] - TIMD_MODE_SHIFT[min( abs( deltaSize - d ), 5 )] );
    newMin -= W > H ? step : 0;
    newMax += W < H ? step : 0;
    previous = deltaSize - d;
  }

  // the first batch (invocation 0): planar, DC, the angular modes of the MPM list with isForcedValid (all 22 entries)
  // and HOR, VER where it lacks them (the CPU also appends DC, then skips it: DC is the second candidate already), then
  // the wide angles beyond the regular ones from newMin + 1 to newMax in steps of 5, skipping the regular range (the
  // CPU's loop jumps to maxOrig - 1: the next is maxOrig + 4)
  SYNC();   // (dimdSelect has read its histograms)
  if( lid == 0 )
  {
    int n = 0;
    s_spCost[TC_CAND + n] = timdCand( PLANAR_IDX, PLANAR_IDX, true, n );
    n++;
    s_spCost[TC_CAND + n] = timdCand( DC_IDX, DC_IDX, true, n );
    n++;
    int  mpm[NUM_MPM];
    uint included[3] = uint[3]( 0u, 0u, 0u );
    const int numSpatial = mpmSpatial( c, nbrMode, dimdModes, true, mpm, included );
    mpmFill( numSpatial, mpm, included );
    for( int i = 0; i < NUM_MPM; i++ )
    {
      if( mpm[i] > DC_IDX )
      {
        s_spCost[TC_CAND + n] = timdCand( timdMode131( mpm[i] ), timdMode131( mpm[i] ), true, n );
        n++;
      }
    }
    for( int k = 0; k < 2; k++ )
    {
      const int mode = k == 0 ? HOR_IDX : VER_IDX;
      if( !modeIn( included, mode ) )
      {
        s_spCost[TC_CAND + n] = timdCand( timdMode131( mode ), timdMode131( mode ), true, n );
        n++;
      }
    }
    s_spCost[TC_END_MPM] = uint( n );
    for( int i = newMin + 1; i < newMax; i += 5 )
    {
      if( i >= minOrig && i <= maxOrig )
      {
        i = maxOrig - 1;
        continue;
      }
      const int regular = timdRegularAngle( W, H, i );
      s_spCost[TC_CAND + n] = timdCand( timdWaipAngle( W, H, regular ), regular, false, n );
      n++;
    }
    s_spCost[TC_NUM] = uint( n );
    for( int i = 0; i < TIMD_CANDS; i++ )
    {
      s_spCost[TC_COST_A + i] = 0u;
      s_spCost[TC_COST_L + i] = 0u;
    }
  }
  // the template's references, DC's value, the first batch's costs
  timdFillReference( h.t, td, c.timd + TIMD_HEAD, c.x, c.y, W, H, bitDepth, lid, NT );
  const int dc       = timdDc( h.t, W, H, lid, NT );
  const int maxPel   = ( 1 << bitDepth ) - 1;
  const int numCands = int( s_spCost[TC_NUM] );
  timdTemplateCosts( c, h.t, TC_CAND, numCands, TC_COST_A, TC_COST_L, dc, maxPel, noPdpc, lid, NT );

  // the CPU's scan of the first batch (invocation 0): the best and the secondary cost, strictly less replacing; planar
  // and DC set the non-angular mode; the MPM list's modes, then the wide angles (where the scan gets that far) end once
  // the secondary cost is at most maxCost, the wide angles after one at least. Then the second batch: the neighbours
  // of the best and of the secondary mode (of a wide angle the neighbouring wide angles outside the regular range)
  const int tw = W <= 8 ? 2 : 4, th = H <= 8 ? 2 : 4;   // (the costs' scale: of both arms, also where one is 0)
  const uint64_t maxCost = uint64_t( tw * H + th * W );
  const int      log2A = findMSB( th ) + findMSB( W ), log2L = findMSB( tw ) + findMSB( H );
  TimdScan s;
  int      modeB[4];    // the second batch's modes, around the best mode (0, 1) and the secondary (2, 3),
  bool     validB[4];   // where the CPU tries them
  if( lid == 0 )
  {
    s.best = s.secondary = s.nonAng = ~0ul;
    s.bestMode = s.secMode = s.nonAngMode = PLANAR_IDX;
    s.bestWA = s.secWA = true;
    s.locDep = int[3]( 0, 0, 0 );
    const int endMpm = int( s_spCost[TC_END_MPM] );
    for( int k = 0; k < numCands; k++ )
    {
      const uint64_t cost = timdCost( k );
      const int      mode = timdCandMode( s_spCost[TC_CAND + k] );
      const bool     wa   = timdCandWA( s_spCost[TC_CAND + k] );
      const int      ld   = timdLocDep( h.type, s_spCost[TC_COST_A + k], s_spCost[TC_COST_L + k], log2A, log2L );
      if( cost < s.best )
      {
        s.secondary = s.best;
        s.secMode   = s.bestMode;
        s.secWA     = s.bestWA;
        s.best      = cost;
        s.bestMode  = mode;
        s.bestWA    = wa;
        s.locDep[1] = s.locDep[0];
        s.locDep[0] = ld;
        if( k < 2 )   // planar, DC
        {
          s.nonAng     = cost;
          s.nonAngMode = mode;
          s.locDep[2]  = ld;
        }
      }
      else if( cost < s.secondary )
      {
        s.secondary = cost;
        s.secMode   = mode;
        s.secWA     = wa;
        s.locDep[1] = ld;
      }
      if( k >= 2 && s.secondary <= maxCost )
      {
        if( k >= endMpm )
        {
          break;          // the wide angles' scan ends
        }
        k = endMpm - 1;   // the MPM list's ends, the wide angles' starts
      }
    }
    // the second batch: of a mode's neighbours those the CPU tries (angular ones; of a wide angle beyond the regular
    // ones those outside the regular range, as such)
    int nB = 0;
    for( int j = 0; j < 4; j++ )
    {
      const int  mid = j < 2 ? s.bestMode : s.secMode;
      const bool wa  = j < 2 ? s.bestWA : s.secWA;
      const int  m   = mid + ( ( j & 1 ) == 0 ? -1 : 1 );
      validB[j] = wa ? m > DC_IDX && m <= EXT_VDIA_IDX : m < minOrig || m > maxOrig;
      if( validB[j] )
      {
        const int regular = wa ? m : timdRegularAngle( W, H, m );
        modeB[j] = wa ? m : timdWaipAngle( W, H, regular );
        s_spCost[TC_CAND + numCands + nB] = timdCand( modeB[j], regular, wa, numCands + j );
        nB++;
      }
      s_spCost[TC_COST_A + numCands + j] = 0u;
      s_spCost[TC_COST_L + numCands + j] = 0u;
    }
    s_spCost[TC_NUM_B] = uint( nB );
  }
  SYNC();
  timdTemplateCosts( c, h.t, TC_CAND + numCands, int( s_spCost[TC_NUM_B] ), TC_COST_A, TC_COST_L, dc, maxPel, noPdpc, lid, NT );
  if( lid != 0 )
  {
    return -1;
  }

  // the CPU's refinement, where the best cost is above maxCost: of the best mode (an angular one, or a wide angle beyond
  // the regular ones) until that cost is at most maxCost, a cheaper neighbour replacing it (the best does not become the
  // secondary); then of the secondary mode alike, until the secondary cost is (on the condition of the best cost, as
  // the CPU has it)
  for( int pass = 0; pass < 2; pass++ )
  {
    const bool ofBest = pass == 0;
    if( !( s.best > maxCost && ( ( ofBest ? s.bestMode : s.secMode ) > DC_IDX || !( ofBest ? s.bestWA : s.secWA ) ) ) )
    {
      continue;
    }
    for( int j = 2 * pass; j < 2 * pass + 2; j++ )
    {
      if( !validB[j] )
      {
        continue;
      }
      const uint64_t cost = timdCost( numCands + j );
      const int      ld   = timdLocDep( h.type, s_spCost[TC_COST_A + numCands + j], s_spCost[TC_COST_L + numCands + j], log2A, log2L );
      if( ofBest && cost < s.best )
      {
        s.best      = cost;
        s.bestMode  = modeB[j];
        s.locDep[0] = ld;
      }
      else if( !ofBest && cost < s.secondary )
      {
        s.secondary = cost;
        s.secMode   = modeB[j];
        s.locDep[1] = ld;
      }
      if( ( ofBest ? s.best : s.secondary ) <= maxCost )
      {
        break;
      }
    }
  }

  // the final choice: the regular modes of the wide angles beyond, the cheaper as the mode; a secondary costing twice
  // the best or more replaced by an angular best (which then blends with itself, JVET-AG0092); blending where the
  // secondary costs less than twice the best, with the non-angular mode where it costs less than 1.5 times the best and
  // is neither of them (weights from the three costs), else by X0149's weights of the two
  if( !s.bestWA )
  {
    s.bestMode = timdRegularAngle( W, H, s.bestMode );
  }
  if( !s.secWA )
  {
    s.secMode = timdRegularAngle( W, H, s.secMode );
  }
  if( s.secondary < s.best )
  {
    const uint64_t cost = s.secondary;
    s.secondary = s.best;
    s.best      = cost;
    const int mode = s.secMode;
    s.secMode  = s.bestMode;
    s.bestMode = mode;
    const bool wa = s.secWA;
    s.secWA  = s.bestWA;
    s.bestWA = wa;
    const int ld = s.locDep[1];
    s.locDep[1] = s.locDep[0];
    s.locDep[0] = ld;
  }
  if( !( s.secondary - s.best < s.best ) && s.bestMode > DC_IDX )
  {
    s.secondary = s.best;
    s.secMode   = s.bestMode;
    s.secWA     = s.bestWA;
  }
  const bool blended = s.secondary - s.best < s.best;
  int r[TIMD_OUT] = int[TIMD_OUT]( s.bestMode, -1, -1, blended ? 1 : 0, -1, -1, -1, s.locDep[0], s.locDep[1], s.locDep[2],
                                   s.bestWA ? 1 : 0, 1 );
  if( blended )
  {
    r[TIMD_OUT_SECONDARY]    = s.secMode;
    r[TIMD_OUT_SEC_CHECK_WA] = s.secWA ? 1 : 0;
    const bool nonAngFusion = ( s.nonAng < s.best || s.nonAng - s.best < ( s.best >> 1 ) ) && s.bestMode != s.nonAngMode
                              && s.secMode != s.nonAngMode;
    if( nonAngFusion )   // each weight (sum - cost) / (2 sum) in 1/64 rounded, as the CPU's doubles give it
    {
      const uint64_t sum = s.best + s.secondary + s.nonAng;
      r[TIMD_OUT_NON_ANG]    = s.nonAngMode;
      r[TIMD_OUT_WEIGHT]     = int( ( 64ul * ( sum - s.best ) + sum ) / ( 2ul * sum ) );
      r[TIMD_OUT_WEIGHT + 1] = int( ( 64ul * ( sum - s.secondary ) + sum ) / ( 2ul * sum ) );
      r[TIMD_OUT_WEIGHT + 2] = 64 - r[TIMD_OUT_WEIGHT] - r[TIMD_OUT_WEIGHT + 1];
    }
    else
    {
      r[TIMD_OUT_WEIGHT]     = timdLutWeight( s.best, s.secondary );
      r[TIMD_OUT_WEIGHT + 1] = 64 - r[TIMD_OUT_WEIGHT];
      r[TIMD_OUT_WEIGHT + 2] = 0;
    }
    mtssSecond = timdMode67( s.secMode );
  }
  for( int k = 0; k < TIMD_OUT; k++ )
  {
    o.v[base + k] = r[k];
  }
  return timdMode67( s.bestMode );
}

#endif
