// Template predictions of TIMD's kind (IntraPrediction::initTimdIntraPatternLuma, initPredTimdIntraParams,
// predTimdIntraAng) for a luma template, and their SATD costs (RdCost::xGetHADs_SIMD), which intra merge ranks its
// candidates with, or TIMD-SAD's mean-removed SAD; the nodes of TIMD CUs and of TIMD merge CUs. The template block is
// the CU extended by tw columns to the left and th rows above (W + tw by H + th; an arm the template lacks is 0,
// CU::deriveTimdRefType's ABOVE_NEIGHBOR / LEFT_NEIGHBOR at the picture's left / top edge); the template is its top th
// rows right of the corner and its left tw columns below the corner. Its reference line, one sample further out, is
// filled into s_ref[0] (the row: (W + tw) * 8 + 1 samples from the corner) and s_ref[1] (the column: (H + th) * 8 + 1).
// Modes are of the 131-mode set (MAP67TO131). Included by intra_rec.comp.
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

// ----- the reference line

// the availability bits of the reference's units (timdFillReference copies them from the record): at most 2 x
// (8 x 132 / 4) + 3 units of 4 for arms of 4 (the host takes CUs up to 128 x 128 only), fewer for TIMD-SAD's arms of 8
// (intra slices keep its CUs at 1024 samples) and CFLM's (133 at most); the host rejects a template of more
// (TIMD_UNITS_MAX)
shared uint s_timdUnits[17];

// the arms of the template block's reference line: 2^TIMD_REF_SHIFT x its sides (initTimdIntraPatternLuma), CFLM's
// chroma 2^CFLM_REF_SHIFT x (xCflmCreateChromaPred)
const int TIMD_REF_SHIFT = 3;
const int CFLM_REF_SHIFT = 1;

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

// xFillTimdReferenceSamples' step 2 for template t of block (cx, cy, W, H) of component comp (a CU's luma, CFLM's
// chroma block), its units' availability bits at flags.v[flagsBase], the arms 2^refShift x the template block's sides
// (TIMD's 8x, CFLM's 2x), all invocations (ends with a barrier): the samples of the available units from the
// reconstruction (in parallel), then the CPU's padding of the others in its order (one invocation: each unit takes the
// last sample before it); without any unit the mid-level value.
// With some units available the CPU fills the row's units right of the corner unit, predSize / unitW of them rounded up
// (beside the template's arm too): with an arm as thick as a unit (CFLM's chroma template of 2) the last one reaches
// past the row (to topEnd), and its samples there land in the column's first entries (the CPU's buffer holds the
// column right after the row, at predStride = predSize + 1): an available unit's in the fill, which the padding may
// overwrite, an unavailable one's in its padding, the last write (TIMD's luma units of 4 end with the row)
void timdFillReference( TemplateUnits t, BufI32 flags, int flagsBase, int comp, int cx, int cy, int W, int H,
                        int refShift, int bitDepth, int lid, int NT )
{
  const int  predSize = ( W + t.tw ) << refShift, predHSize = ( H + t.th ) << refShift;
  const int  x0 = cx - t.tw - 1, y0 = cy - t.th - 1;
  const bool partial = t.numIntraNeighbor != 0 && t.numIntraNeighbor != t.totalUnits;
  const int  rest = ( predSize - t.tw ) % t.unitW;   // the last unit's samples (lastSample): rest, or a whole unit
  // the last entry of the row the units write: from 1 + tw - aboveTempUnitNum x unitW on (luma's end with the row)
  const int  topEnd = !WITH_CFLM ? predSize
                      : t.tw + ( t.totalAboveUnits - 1 ) * t.unitW + ( rest != 0 ? rest : t.unitW );
  if( lid < ( t.totalUnits + 31 ) >> 5 )
  {
    s_timdUnits[lid] = uint( flags.v[flagsBase + lid] );
  }
  SYNC();
  for( int j = lid; j <= ( partial ? max( predSize, topEnd ) : predSize ); j += NT )
  {
    s_ref[0][j] = int16_t( t.numIntraNeighbor == 0 ? 1 << ( bitDepth - 1 )
                           : timdUnitAvailable( timdTopUnit( t, j ) ) ? recoAt( comp, x0 + j, y0 ) : 0 );
  }
  for( int i = lid; i <= predHSize; i += NT )
  {
    s_ref[1][i] = int16_t( t.numIntraNeighbor == 0 ? 1 << ( bitDepth - 1 )
                           : timdUnitAvailable( timdLeftUnit( t, i ) ) ? recoAt( comp, x0, y0 + i ) : 0 );
  }
  SYNC();
  if( lid == 0 && partial )
  {
    const int c = t.totalLeftUnits + t.leftTempUnitNum, uw = t.unitW, uh = t.unitH;
    const int cornerCol = t.tw - t.aboveTempUnitNum * uw, cornerRow = t.th - t.leftTempUnitNum * uh;
    if( WITH_CFLM && timdUnitAvailable( t.totalUnits - 1 ) )   // (the fill's samples past the row)
    {
      for( int j = predSize + 1; j <= topEnd; j++ )
      {
        s_ref[1][j - predSize - 1] = s_ref[0][j];
      }
    }
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
          const int num = unit == t.totalUnits - 1 && rest != 0 ? rest : uw;
          for( int j = lastAvailCol + 1; j <= lastAvailCol + num; j++ )
          {
            s_ref[0][j] = v;
            if( WITH_CFLM && j > predSize )   // (past the row: the column's first entries)
            {
              s_ref[1][j - predSize - 1] = v;
            }
          }
        }
      }
      lastAvailUnit = unit;
    }
  }
  SYNC();
}

// ----- the prediction

// initPredTimdIntraParams for a CU of W x H (the angular ones of an angular mode, unused for planar / DC); for CFLM's
// chroma template (chroma: xPredTimdIntraAng with a chroma channel type) initPredIntraParams' of the chroma block, its
// regular 1/32-sample angles interpolated by two taps (xIntraPredTimdAngChroma) from the arms of its template block's
// reference line, twice the sides long
struct TimdParams
{
  bool isModeVer;
  int  angle;          // intraPredAngle, 1/64 sample per row (chroma: 1/32)
  int  invAngle;       // absInvAngle
  bool pdpc;           // applyPDPC
  int  angularScale;
  bool gradPdpc;       // useGradPDPC (positive angles)
  bool chroma;
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
  p.pdpc   = p.pdpc && !noPdpc;
  p.chroma = false;
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
// (m_topRefLength / m_leftRefLength: 8 x the template block's side, CFLM's chroma 2 x)
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
  const int refLength = ( p.isModeVer ? wT : hT ) << ( WITH_CFLM && p.chroma ? 1 : 3 );
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
// 1/64 cubic filter and its PDPC kinds, clipped as the CPU clips; CFLM's chroma template by p's regular angle, the
// 1/32 two-tap interpolation unclipped); dc: timdDc's value
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
  const int fracBits = WITH_CFLM && p.chroma ? 5 : 6, fracMask = ( 1 << fracBits ) - 1;
  const int deltaPos = p.angle * ( yt + 1 ), deltaInt = deltaPos >> fracBits, frac = deltaPos & fracMask;
  const int idx      = deltaInt + 1 + xt;
  if( ( abs( p.angle ) & fracMask ) == 0 )
  {
    v = timdMain( p, idx, wT, hT );
  }
  else if( WITH_CFLM && p.chroma )   // xIntraPredTimdAngChroma
  {
    const int m0 = timdMain( p, idx, wT, hT );
    v = m0 + ( ( frac * ( timdMain( p, idx + 1, wT, hT ) - m0 ) + 16 ) >> 5 );
  }
  else   // xIntraPredTimdAngLuma
  {
    BufI32 f   = BufI32( P.filters );
    int    sum = 128;
    for( int k = 0; k < 4; k++ )
    {
      sum += f.v[EXT_CUBIC + frac * 4 + k] * timdMain( p, idx - 1 + k, wT, hT );
    }
    v = clamp( sum >> 8, 0, maxPel );
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
      const int topLeft = m0 + ( ( frac * ( m1 - m0 ) + ( 1 << ( fracBits - 1 ) ) ) >> fracBits );
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

// (int)( sad / sqrt( 8.0 ) * 2 ) of the 1x8 kernel: floor( sad / sqrt( 2 ) ) = the largest q with 2 q^2 <= sad^2 (sad
// below 2^24: the double expression has no rounding that crosses an integer, checked exhaustively)
int timdScaleSqrt2( int sad )
{
  const int64_t s2 = int64_t( sad ) * int64_t( sad );
  int q = int( float( sad ) * 0.70710678f );
  while( 2 * int64_t( q + 1 ) * int64_t( q + 1 ) <= s2 ) q++;
  while( q > 0 && 2 * int64_t( q ) * int64_t( q ) > s2 ) q--;
  return q;
}

// the SATD of the n differences d along an arm one sample thick (JVET-AJ0096's kernels of xGetHADs_SIMD, the DC term a
// quarter: JVET-R0164): 16 by xCalcHADs1x16_SSE - the butterflies of strides 8, 4 and 2 only (its last stage left
// out), the DC term that of the even samples, the sum halved; 8 by xCalcHADs1x8_SSE - the full transform, scaled by
// 2 / sqrt( 8 ); 4 by xCalcHADs1xN - the full transform, scaled by 2 / sqrt( 4 ) = 1
int timdHadLine( int d[32], int n )
{
  if( n == 16 )
  {
    int m1[4][4];   // per 4 lanes k: the stride-8 sum / difference, each with the stride-4 sum and difference
    for( int k = 0; k < 4; k++ )
    {
      const int s0 = d[k] + d[k + 8], s1 = d[k + 4] + d[k + 12], t0 = d[k] - d[k + 8], t1 = d[k + 4] - d[k + 12];
      m1[0][k] = s0 + s1;
      m1[1][k] = s0 - s1;
      m1[2][k] = t0 + t1;
      m1[3][k] = t0 - t1;
    }
    int sad = 0;
    for( int v = 0; v < 4; v++ )
    {
      sad += abs( m1[v][0] + m1[v][2] ) + abs( m1[v][1] + m1[v][3] ) + abs( m1[v][0] - m1[v][2] ) + abs( m1[v][1] - m1[v][3] );
    }
    const int dc = abs( m1[0][0] + m1[0][2] );
    return ( sad - dc + ( dc >> 2 ) ) >> 1;
  }
  int sad = 0, dc = 0;
  for( int i = 0; i < n; i++ )
  {
    int s = 0;
    for( int k = 0; k < n; k++ ) s += ( bitCount( uint( i & k ) ) & 1 ) != 0 ? -d[k] : d[k];
    sad += abs( s );
    dc = i == 0 ? abs( s ) : dc;
  }
  sad = sad - dc + ( dc >> 2 );
  return n == 8 ? timdScaleSqrt2( sad ) : sad;
}

// the SATD of block (bx, by) of size kw x kh (2x2, 4x4, 8x4 or 4x8; along an arm one sample thick 16, 8 or 4 samples)
// of an arm
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
  if( WITH_THIN_TEMPLATES && ( kw == 1 || kh == 1 ) )   // (TIMD merge's and TMRL's templates only)
  {
    return timdHadLine( d, kw * kh );
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

// the kernel xGetHADs_SIMD takes for an arm of w x h: of the arms 2 or 4 thick of TIMD's templates 8x4, 4x8, 4x4 or 2x2;
// along an arm one sample thick (TIMD merge's and TMRL's templates) 16 or 8 samples where they divide it, else the
// whole arm
ivec2 timdKernel( int w, int h )
{
  if( WITH_THIN_TEMPLATES && ( w == 1 || h == 1 ) )   // (TIMD merge's and TMRL's templates only)
  {
    const int len = max( w, h ), n = len % 16 == 0 ? 16 : len % 8 == 0 ? 8 : len;
    return h == 1 ? ivec2( n, 1 ) : ivec2( 1, n );
  }
  return w > h && ( w & 7 ) == 0 && ( h & 3 ) == 0 ? ivec2( 8, 4 )
       : w < h && ( h & 7 ) == 0 && ( w & 3 ) == 0 ? ivec2( 4, 8 )
       : ( h & 3 ) == 0 && ( w & 3 ) == 0         ? ivec2( 4, 4 ) : ivec2( 2, 2 );
}

// ----- the costs of a list of candidates

// a candidate (a word of s_spCost): the mode predicted, the mode of its parameters and whether they take
// getWideAngleExt's wide angles (timdParams), and the slot its costs are added to. A mode from a block vector (from
// TIMD_BV_IDX) takes the field's top quarter
uint timdCand( int mode, int dirMode, bool checkWA, int slot )
{
  const int field = mode >= TIMD_BV_IDX ? mode - TIMD_BV_IDX + 768 : mode + 256;
  return uint( field ) | uint( dirMode ) << 10 | ( checkWA ? 1u : 0u ) << 18 | uint( slot ) << 19;
}

int  timdCandMode( uint w ) { const int f = int( w & 1023u ); return f >= 768 ? f - 768 + TIMD_BV_IDX : f - 256; }
int  timdCandDir( uint w )  { return int( ( w >> 10 ) & 255u ); }
bool timdCandWA( uint w )   { return ( w & ( 1u << 18 ) ) != 0u; }
int  timdCandSlot( uint w ) { return int( w >> 19 ); }

// the difference of the original (the reconstruction) and the prediction at sample s of a template (timdTemplateCosts'
// order: the above arm's rows of c.w, then the left arm's rows of t.tw) predicted into s_tmp from predOffset
int timdTemplateDiff( Cu c, TemplateUnits t, int predOffset, int s )
{
  const int nA = c.w * t.th;
  return s < nA ? timdDiff( c.x, c.y - t.th, predOffset, c.w, s % c.w, s / c.w )
                : timdDiff( c.x - t.tw, c.y, predOffset + nA, t.tw, ( s - nA ) % t.tw, ( s - nA ) / t.tw );
}

// the mean-removed SAD's sums of the differences per candidate of a pass and arm (above, left), 2 x TIMD_MRSAD_CANDS
// ints. They take s_dimdHist (3 x 67 ints), which a TIMD node no longer reads once dimdSelect has (and which MTSS clears
// before its histograms): shared memory of their own slowed every pipeline by ~3 %
const int TIMD_MRSAD_CANDS = 64;
#define s_timdArmDiff s_dimdHist

// the template costs of the numCands candidates at s_spCost[candBase] for CU c with template t (all invocations; the
// references filled, dc timdDc's): each arm's cost added to its slot (the above arm's at costA, the left arm's at costL;
// zeroed by the caller), as many candidates at once as s_tmp holds their templates (the above arm's rows of c.w, then the
// left arm's rows of tw), predicted in parallel, then the cost of the reconstruction against each in parallel: the SATD
// block by block, or with mrsad deriveTimdModeSad's mean-removed SAD (RdCost::xGetMRSAD per arm: the sum of the
// differences' distances from their mean, the mean truncated toward 0; its SIMD forms, whose 16-bit lanes saturate,
// agree up to 11 bits, the GPU's TIMD up to 10)
void timdTemplateCosts( Cu c, TemplateUnits t, int candBase, int numCands, int costA, int costL, int dc, int maxPel, bool noPdpc,
                        bool mrsad, int lid, int NT )
{
  const int   nA = c.w * t.th, T = nA + t.tw * c.h;   // template samples per candidate
  const int   perPass = mrsad ? min( max( 1, MAX_REF / T ), TIMD_MRSAD_CANDS ) : max( 1, MAX_REF / T );
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
      const int mode = timdCandMode( cand );
      if( mode >= TIMD_BV_IDX )   // a block vector's: the reconstruction there (predTimdIbcItmp; clamped to the picture
      {                           // for a vector that TIMD-SAD's thicker template passes over)
        const ivec2 bv = bvListAt( mode - TIMD_BV_IDX );
        s_tmp[i] = int16_t( recoAt( 0, clamp( c.x - t.tw + x + bv.x, 0, P.width[0] - 1 ),
                                    clamp( c.y - t.th + y + bv.y, 0, P.height[0] - 1 ) ) );
      }
      else
      {
        s_tmp[i] = int16_t( timdPredict( mode, p, x, y, c.w, c.h, t.tw, t.th, dc, maxPel ) );
      }
    }
    for( int i = lid; mrsad && i < 2 * n; i += NT )
    {
      s_timdArmDiff[i] = 0;
    }
    SYNC();
    if( mrsad )
    {
      for( int i = lid; i < n * T; i += NT )
      {
        atomicAdd( s_timdArmDiff[2 * ( i / T ) + ( i % T < nA ? 0 : 1 )], timdTemplateDiff( c, t, ( i / T ) * T, i % T ) );
      }
      SYNC();
      for( int i = lid; i < n * T; i += NT )
      {
        const int m = i / T, s = i % T, arm = s < nA ? 0 : 1;
        const int sum = s_timdArmDiff[2 * m + arm], count = arm == 0 ? nA : T - nA;
        const int mean = sum < 0 ? -( -sum / count ) : sum / count;
        atomicAdd( s_spCost[( arm == 0 ? costA : costL ) + timdCandSlot( s_spCost[candBase + first + m] )],
                   uint( abs( timdTemplateDiff( c, t, m * T, s ) - mean ) ) );
      }
    }
    else
    {
      for( int i = lid; i < n * B; i += NT )
      {
        const int m = i / B, b = i % B, k = b - nbA;
        const int satd = b < nbA ? timdHad( c.x, c.y - t.th, m * T, c.w, ( b % colsA ) * ka.x, ( b / colsA ) * ka.y, ka.x, ka.y )
                                 : timdHad( c.x - t.tw, c.y, m * T + nA, t.tw, ( k % colsL ) * kl.x, ( k / colsL ) * kl.y, kl.x, kl.y );
        atomicAdd( s_spCost[( b < nbA ? costA : costL ) + timdCandSlot( s_spCost[candBase + first + m] )], uint( satd ) );
      }
    }
    SYNC();
  }
}

// ----- a TIMD CU's node (IntraPrediction::deriveTimdMode, a TIMD-SAD CU's also deriveTimdModeSad; without block
// vectors; TIMD merge CUs have a node of their own, below): the template costs of its candidates in two batches -
// planar, DC, the angular modes of a list and the wide angles beyond the regular ones, then the neighbours of the best
// and the secondary mode - computed in parallel, the CPU's sequential choice among them replayed by invocation 0 (with
// its early ends: computing more costs than the CPU does changes no choice), then the fusion weights. deriveTimdMode's
// list is the MPM list and its costs the SATD; deriveTimdModeSad's list is made of the modes deriveTimdMode costed
// first, the CU's DIMD modes and the modes around those found, and its costs the mean-removed SAD of a template up to
// twice as thick

// in s_spCost (free in a node): the candidates (timdCand words; a first-batch candidate's cost slot is its index, a
// second-batch one's numCands + its neighbour index, so that an invalid neighbour leaves its slot unused) and their
// arms' costs
const int TIMD_CANDS  = 64;                        // at most 2 + 25 (deriveTimdModeSad's list) + the wide angles beyond
                                                   // (below 12), + 4
const int TC_CAND     = 0;
const int TC_COST_A   = TC_CAND + TIMD_CANDS;      // the above arm's (tmpCost0)
const int TC_COST_L   = TC_COST_A + TIMD_CANDS;    // the left arm's (tmpCost1)
const int TC_NUM      = TC_COST_L + TIMD_CANDS;    // the first batch's candidates
const int TC_NUM_B    = TC_NUM + 1;                // the second batch's candidates
const int TC_END_LIST = TC_NUM + 2;                // the end of the list's modes in the first batch
const int TC_NUM_BV   = TC_NUM + 3;                // the length of the CU's list of block vectors

const int TIMD_COST_LIST     = NUM_MPM + 2;   // m_timdModeCostList: the MPM list's angular modes, HOR and VER
const int TIMD_SAD_LIST      = 25;            // deriveTimdModeSad's list (TIMD_NUM_MODES_SORTED + 10)
const int TIMD_SAD_SORTED    = 15;            // of which the first modes of m_timdModeCostList (TIMD_NUM_MODES_SORTED)
const int TIMD_SAD_REFINED   = 12;            // the modes around the modes found it tries (deriveTimdModeSad's numRefine)
const int TIMD_WIDE_STEP     = 5;             // the step between the wide angles beyond the regular ones it tries
const int TIMD_SECOND_BATCH  = 4;             // the neighbours of the best mode (0, 1) and of the secondary (2, 3)

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

// the modes of the 131-mode set deriveTimdMode tries for a CU's shape: the regular ones (minOrig to maxOrig, the wide
// angles of WAIP included) and the wide angles beyond them that its reference reaches (newMin to newMax)
struct TimdAngleRange
{
  int minOrig, maxOrig;
  int newMin, newMax;
};

// a TIMD CU's record (TimdHead) at timdData[at]
TimdHead timdHeadAt( int at )
{
  BufI32    timdData = BufI32( P.timdData );
  const int tr = at + TIMD_HEAD_MTS_TR_IDX, merge = at + TIMD_HEAD_MERGE;
  int mergeNbr[TIMD_MERGE_CANDS];
  for( int k = 0; k < TIMD_MERGE_CANDS; k++ )
  {
    mergeNbr[k] = timdData.v[merge + 1 + k];
  }
  return TimdHead( timdData.v[at], timdData.v[at + 1], timdData.v[at + 2], templateUnitsAt( timdData, at + TIMD_HEAD_T ),
                   timdData.v[at + TIMD_HEAD_SAD], templateUnitsAt( timdData, at + TIMD_HEAD_TS ), timdData.v[tr],
                   timdData.v[tr + 1], timdData.v[tr + 2], timdData.v[merge], mergeNbr,
                   timdData.v[merge + 1 + TIMD_MERGE_CANDS] );
}

// deriveTimdMode's range of modes for a CU of W x H whose PUs above-right / below-left reach further (waip:
// TIMD_WAIP_* bits; its increase / decrease Close / Far)
TimdAngleRange timdAngleRange( int waip, int W, int H )
{
  const int  deltaSize = abs( findMSB( W ) - findMSB( H ) );
  const bool maxClose = ( waip & TIMD_WAIP_MAX_CLOSE ) != 0, maxFar = ( waip & TIMD_WAIP_MAX_FAR ) != 0;
  const bool minClose = ( waip & TIMD_WAIP_MIN_CLOSE ) != 0, minFar = ( waip & TIMD_WAIP_MIN_FAR ) != 0;
  TimdAngleRange range;
  range.maxOrig = W >= H ? EXT_VDIA_IDX + TIMD_MODE_SHIFT[deltaSize] : EXT_VDIA_IDX - TIMD_MODE_SHIFT[deltaSize];
  range.minOrig = W >= H ? 2 + TIMD_MODE_SHIFT[deltaSize] : 2 - TIMD_MODE_SHIFT[deltaSize];
  range.newMin  = range.minOrig;
  range.newMax  = range.maxOrig;
  if( W >= H && ( maxFar || maxClose ) )
  {
    range.newMax += TIMD_MODE_SHIFT[min( deltaSize + ( maxFar ? 2 : 1 ), 5 )] - TIMD_MODE_SHIFT[deltaSize];
  }
  if( W <= H && ( minFar || minClose ) )
  {
    range.newMin -= TIMD_MODE_SHIFT[min( deltaSize + ( minFar ? 2 : 1 ), 5 )] - TIMD_MODE_SHIFT[deltaSize];
  }
  const int extRef = minFar || maxFar ? 2 : minClose || maxClose ? 1 : 0;
  for( int e = 1, previous = deltaSize; e <= extRef; e++ )
  {
    const int step = abs( TIMD_MODE_SHIFT[abs( previous )] - TIMD_MODE_SHIFT[min( abs( deltaSize - e ), 5 )] );
    range.newMin -= W > H ? step : 0;
    range.newMax += W < H ? step : 0;
    previous = deltaSize - e;
  }
  return range;
}

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

// ----- the first batches (invocation 0)

// the wide angles beyond the regular ones as candidates from slot n, newMin + 1 to newMax in steps of 5, skipping the
// regular range (the CPU's loop jumps to maxOrig - 1: the next is maxOrig + 4); returns the next slot
int timdWideAnglesBeyond( int n, int W, int H, TimdAngleRange range )
{
  for( int i = range.newMin + 1; i < range.newMax; i += TIMD_WIDE_STEP )
  {
    if( i >= range.minOrig && i <= range.maxOrig )
    {
      i = range.maxOrig - 1;
      continue;
    }
    const int regular = timdRegularAngle( W, H, i );
    s_spCost[TC_CAND + n] = timdCand( timdWaipAngle( W, H, regular ), regular, false, n );
    n++;
  }
  return n;
}

// a first batch's end: its list's modes ending at endList, the wide angles after them, the numbers, the costs zeroed
void timdFirstBatchEnd( int endList, int W, int H, TimdAngleRange range )
{
  s_spCost[TC_END_LIST] = uint( endList );
  s_spCost[TC_NUM]      = uint( timdWideAnglesBeyond( endList, W, H, range ) );
  for( int i = 0; i < TIMD_CANDS; i++ )
  {
    s_spCost[TC_COST_A + i] = 0u;
    s_spCost[TC_COST_L + i] = 0u;
  }
}

// a list mode m (67-mode set) as the candidate of slot n; returns the next slot
int timdListCand( int n, int m )
{
  s_spCost[TC_CAND + n] = timdCand( timdMode131( m ), timdMode131( m ), true, n );
  return n + 1;
}

// deriveTimdMode's first batch: planar, DC, the angular modes of the MPM list with isForcedValid (all 22 entries) and
// HOR, VER where it lacks them (the CPU also appends DC, then skips it: DC is the second candidate already), then the
// wide angles beyond the regular ones. nbrMode, dimdModes: deriveLumaMode's
void timdMpmCandidates( Cu c, int nbrMode[MPM_NEIGHBOURS], int dimdModes[3], TimdAngleRange range )
{
  int n = 0;
  s_spCost[TC_CAND + n] = timdCand( PLANAR_IDX, PLANAR_IDX, true, n );
  n++;
  s_spCost[TC_CAND + n] = timdCand( DC_IDX, DC_IDX, true, n );
  n++;
  int  mpm[NUM_MPM];
  uint included[3] = uint[3]( 0u, 0u, 0u );
  const int numSpatial = mpmSpatial( c, nbrMode, dimdModes, true, mpm, included );
  mpmFill( numSpatial, NUM_MPM, mpm, included );
  for( int i = 0; i < NUM_MPM; i++ )
  {
    if( mpm[i] > DC_IDX )
    {
      n = timdListCand( n, mpm[i] );
    }
  }
  if( !modeIn( included, HOR_IDX ) )
  {
    n = timdListCand( n, HOR_IDX );
  }
  if( !modeIn( included, VER_IDX ) )
  {
    n = timdListCand( n, VER_IDX );
  }
  timdFirstBatchEnd( n, c.w, c.h, range );
}

// the i-th of the 12 modes around those found that deriveTimdModeSad's list tries (its refinedList): around TIMD's mode
// and secondary (67-mode set), DIMD's main mode, DIMD's first blend mode where DIMD blends (else further around its
// main mode), then again around TIMD's two
int timdSadRefinedMode( int i, int timd67, int second67, int dimdMode, int dimdBlend, bool dimdBlending )
{
  switch( i )
  {
    case 0:  return timd67 + 1;
    case 1:  return timd67 - 1;
    case 2:  return second67 + 1;
    case 3:  return second67 - 1;
    case 4:  return dimdMode + 1;
    case 5:  return dimdMode - 1;
    case 6:  return dimdBlending ? dimdBlend + 1 : dimdMode + 2;
    case 7:  return dimdBlending ? dimdBlend - 1 : dimdMode - 2;
    case 8:  return timd67 + 2;
    case 9:  return timd67 - 2;
    case 10: return second67 + 2;
    default: return second67 - 2;
  }
}

// one step of deriveTimdModeSad's list after its sorted costs' modes: mode m, where the list (n of TIMD_SAD_LIST) is
// not full and m is angular, taken as used where TIMD found it (its secondary only where TIMD blends), else added where
// unused
void timdSadListAdd( int m, int timdMode, int timdSecondary, bool blended, inout uint used[3], inout int list[TIMD_SAD_LIST],
                     inout int n )
{
  if( n == TIMD_SAD_LIST || m <= DC_IDX || m >= NUM_LUMA_MODE )
  {
    return;
  }
  if( timdMode131( m ) == timdMode || ( blended && timdMode131( m ) == timdSecondary ) )
  {
    used[m >> 5] |= 1u << ( m & 31 );
  }
  if( !modeIn( used, m ) )
  {
    list[n++] = m;
    used[m >> 5] |= 1u << ( m & 31 );
  }
}

// deriveTimdModeSad's first batch: planar, DC, then a list of up to 25 modes - the first 15 angular modes of
// m_timdModeCostList (those deriveTimdMode costed in its MPM list's scan to its early end, the first batch's candidates
// 2 to lastList, by cost, then mode: std::stable_sort of the pairs), the CU's DIMD modes (its DimdOut slot cuIndex),
// HOR, VER and the modes around those deriveTimdMode found (timdFields) and around DIMD's (timdSadRefinedMode); each
// mode once, none of the modes TIMD found (its secondary only where they blend) - then the wide angles beyond the
// regular ones as deriveTimdMode's (TIMD-SAD's code runs in a pipeline of its own: VARIANT_TIMD_SAD). The list into
// list (n modes) first, the candidates from it by timdSadSlots (with block vectors their batches in between)
void timdSadList( int timdFields[TIMD_OUT_FIELDS], int cuIndex, int lastList, out int list[TIMD_SAD_LIST], out int n )
{
  uint64_t costs[TIMD_COST_LIST];
  int      modes[TIMD_COST_LIST];
  int      numCosts = 0;
  for( int k = 2; k <= lastList; k++ )   // (insertion sort)
  {
    const uint64_t cost = timdCost( k );
    const int      mode = timdMode67( timdCandMode( s_spCost[TC_CAND + k] ) );
    int i = numCosts++;
    for( ; i > 0 && ( costs[i - 1] > cost || ( costs[i - 1] == cost && modes[i - 1] > mode ) ); i-- )
    {
      costs[i] = costs[i - 1];
      modes[i] = modes[i - 1];
    }
    costs[i] = cost;
    modes[i] = mode;
  }
  const int  timdMode = timdFields[TIMD_OUT_MODE], timdSecondary = timdFields[TIMD_OUT_SECONDARY];
  const bool blended  = timdFields[TIMD_OUT_BLENDED] != 0;
  uint used[3] = uint[3]( 0u, 0u, 0u );
  n = 0;
  // the sorted modes until min(15, the list's size) of them are taken (those skipped not counted)
  for( int i = 0; i < numCosts && n < min( numCosts, TIMD_SAD_SORTED ); i++ )
  {
    const int m = modes[i];
    if( timdMode131( m ) != timdMode && !( blended && timdMode131( m ) == timdSecondary ) && m > DC_IDX )
    {
      list[n++] = m;
      used[m >> 5] |= 1u << ( m & 31 );
    }
  }
  CBufI32   dimdOut  = CBufI32( P.dimdOut );
  const int dimdBase = cuIndex * DIMD_OUT;
  const int dimdMode = dimdOut.v[dimdBase + DIMD_OUT_MODE];
  for( int i = 0; i < DIMD_FUSION_NUM; i++ )   // DIMD's main mode and blend modes
  {
    const int m = i == 0 ? dimdMode : dimdOut.v[dimdBase + DIMD_OUT_BLEND_MODE + i - 1];
    timdSadListAdd( m, timdMode, timdSecondary, blended, used, list, n );
  }
  timdSadListAdd( HOR_IDX, timdMode, timdSecondary, blended, used, list, n );
  timdSadListAdd( VER_IDX, timdMode, timdSecondary, blended, used, list, n );
  const int  timd67 = timdMode67( timdMode ), second67 = timdMode67( timdSecondary );
  const int  dimdBlend    = dimdOut.v[dimdBase + DIMD_OUT_BLEND_MODE];
  const bool dimdBlending = dimdOut.v[dimdBase + DIMD_OUT_BLENDING] != 0;
  for( int i = 0; i < TIMD_SAD_REFINED; i++ )
  {
    timdSadListAdd( timdSadRefinedMode( i, timd67, second67, dimdMode, dimdBlend, dimdBlending ), timdMode, timdSecondary,
                    blended, used, list, n );
  }
}

// deriveTimdModeSad's first batch from its list (n modes, timdSadList): planar, DC, the list, the wide angles
void timdSadSlots( int list[TIMD_SAD_LIST], int n, int W, int H, TimdAngleRange range )
{
  int slot = 0;
  s_spCost[TC_CAND + slot] = timdCand( PLANAR_IDX, PLANAR_IDX, true, slot );
  slot++;
  s_spCost[TC_CAND + slot] = timdCand( DC_IDX, DC_IDX, true, slot );
  slot++;
  for( int i = 0; i < n; i++ )
  {
    slot = timdListCand( slot, list[i] );
  }
  timdFirstBatchEnd( slot, W, H, range );
}

// ----- the CPU's choice (invocation 0)

// the CPU's scan of the first batch of numCands candidates: the best and the secondary cost, strictly less replacing;
// planar and DC set the non-angular mode; the list's modes (to endList), then the wide angles (where the scan gets that
// far) end once the secondary cost is at most maxCost, the wide angles after one at least. With skipAbove
// (deriveTimdModeSad with both arms, JVET-AN0093) the scan passes over a list mode whose above arm alone costs the
// secondary cost or more, before that test. Returns the slot of the last list mode the scan took (its early end).
// continued: planar, DC and the block vectors were scanned before (timdScanVectors), the scan starts with the list
int timdScanFirst( inout TimdScan s, int numCands, int endList, int type, int log2A, int log2L, uint64_t maxCost,
                   bool skipAbove, bool continued )
{
  if( !continued )
  {
    s.best = s.secondary = s.nonAng = ~0ul;
    s.bestMode = s.secMode = s.nonAngMode = PLANAR_IDX;
    s.bestWA = s.secWA = true;
    s.locDep = int[3]( 0, 0, 0 );
  }
  int lastList = endList - 1;
  for( int k = continued ? 2 : 0; k < numCands; k++ )
  {
    if( skipAbove && k >= 2 && k < endList && uint64_t( s_spCost[TC_COST_A + k] ) >= s.secondary )
    {
      continue;
    }
    const uint64_t cost = timdCost( k );
    const int      mode = timdCandMode( s_spCost[TC_CAND + k] );
    const bool     wa   = timdCandWA( s_spCost[TC_CAND + k] );
    const int      ld   = timdLocDep( type, s_spCost[TC_COST_A + k], s_spCost[TC_COST_L + k], log2A, log2L );
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
      if( k >= endList )
      {
        break;          // the wide angles' scan ends
      }
      lastList = k;
      k = endList - 1;   // the list's scan ends, the wide angles' starts
    }
  }
  return lastList;
}

// the CPU's first loop with block vectors (deriveTimdMode's bFull loop, deriveTimdModeSad's alike): planar, DC, then
// the list's vectors (modes from TIMD_BV_IDX), num of them at slots 0 .. num - 1 (first: the loop's start, planar at
// slot 0), strictly less replacing, each one becoming the best also the non-angular mode. A vector's mode counts as
// angular for the location dependences, which it zeroes (deriveTimdModeSad's stay 0). deriveTimdModeSad's loop (sad)
// passes over a vector whose template block would start left of or above the picture (template t of CU c)
void timdScanVectors( inout TimdScan s, int num, bool first, int type, int log2A, int log2L, bool sad, Cu c,
                      TemplateUnits t )
{
  if( first )
  {
    s.best = s.secondary = s.nonAng = ~0ul;
    s.bestMode = s.secMode = s.nonAngMode = PLANAR_IDX;
    s.bestWA = s.secWA = true;
    s.locDep = int[3]( 0, 0, 0 );
  }
  for( int k = 0; k < num; k++ )
  {
    const uint64_t cost = timdCost( k );
    const int      mode = timdCandMode( s_spCost[TC_CAND + k] );
    if( sad && mode >= TIMD_BV_IDX )
    {
      const ivec2 bv = bvListAt( mode - TIMD_BV_IDX );
      if( c.x - t.tw + bv.x < 0 || c.y - t.th + bv.y < 0 )
      {
        continue;
      }
    }
    const int      ld   = timdLocDep( type, s_spCost[TC_COST_A + k], s_spCost[TC_COST_L + k], log2A, log2L );
    if( cost < s.best )
    {
      s.secondary  = s.best;
      s.secMode    = s.bestMode;
      s.best       = cost;
      s.bestMode   = mode;
      s.nonAng     = cost;
      s.nonAngMode = mode;
      s.locDep     = mode > DC_IDX ? int[3]( 0, 0, 0 ) : int[3]( ld, s.locDep[0], ld );
    }
    else if( cost < s.secondary )
    {
      s.secondary = cost;
      s.secMode   = mode;
      s.locDep[1] = mode > DC_IDX ? 0 : ld;
    }
  }
}

// the CPU's first loop with numBv block vectors of the CU's list (all invocations, uniform control flow): planar, DC
// and the vectors as candidates in batches of TIMD_CANDS, their costs (mrsad: deriveTimdModeSad's), each batch scanned
// by invocation 0 (timdScanVectors)
void timdVectorBatches( Cu c, TemplateUnits t, int numBv, int dc, int maxPel, bool noPdpc, bool mrsad, inout TimdScan s,
                        int type, int log2A, int log2L, int lid, int NT )
{
  for( int base = 0; base < 2 + numBv; base += TIMD_CANDS )
  {
    const int n = min( TIMD_CANDS, 2 + numBv - base );
    for( int k = lid; k < n; k += NT )
    {
      const int e = base + k;   // planar, DC, then the vectors
      s_spCost[TC_CAND + k]   = timdCand( e < 2 ? e : TIMD_BV_IDX + e - 2, e < 2 ? e : PLANAR_IDX, true, k );
      s_spCost[TC_COST_A + k] = 0u;
      s_spCost[TC_COST_L + k] = 0u;
    }
    SYNC();
    timdTemplateCosts( c, t, TC_CAND, n, TC_COST_A, TC_COST_L, dc, maxPel, noPdpc, mrsad, lid, NT );
    if( lid == 0 )
    {
      timdScanVectors( s, n, base == 0, type, log2A, log2L, mrsad, c, t );   // (mrsad: deriveTimdModeSad's loop)
    }
    SYNC();
  }
}

// the second batch: of the best and the secondary mode's neighbours those the CPU tries (angular ones; of a wide angle
// beyond the regular ones those outside the regular range, as such), as candidates from slot numCands; their modes in
// neighbourMode (TIMD_SECOND_BATCH order) where neighbourValid
void timdSecondBatch( TimdScan s, int numCands, int W, int H, TimdAngleRange range, out int neighbourMode[TIMD_SECOND_BATCH],
                      out bool neighbourValid[TIMD_SECOND_BATCH] )
{
  int numSecond = 0;
  for( int j = 0; j < TIMD_SECOND_BATCH; j++ )
  {
    const int  mid = j < 2 ? s.bestMode : s.secMode;
    const bool wa  = j < 2 ? s.bestWA : s.secWA;
    const int  m   = mid + ( ( j & 1 ) == 0 ? -1 : 1 );
    neighbourValid[j] = wa ? m > DC_IDX && m <= EXT_VDIA_IDX : m < range.minOrig || m > range.maxOrig;
    neighbourMode[j]  = -1;
    if( neighbourValid[j] )
    {
      const int regular = wa ? m : timdRegularAngle( W, H, m );
      neighbourMode[j] = wa ? m : timdWaipAngle( W, H, regular );
      s_spCost[TC_CAND + numCands + numSecond] = timdCand( neighbourMode[j], regular, wa, numCands + j );
      numSecond++;
    }
    s_spCost[TC_COST_A + numCands + j] = 0u;
    s_spCost[TC_COST_L + numCands + j] = 0u;
  }
  s_spCost[TC_NUM_B] = uint( numSecond );
}

// the CPU's refinement, where the best cost is above maxCost: of the best mode (an angular one, or a wide angle beyond
// the regular ones) until that cost is at most maxCost, a cheaper neighbour replacing it (the best does not become the
// secondary); then of the secondary mode alike, until the secondary cost is (on the condition of the best cost, as the
// CPU has it)
void timdRefine( inout TimdScan s, int numCands, int neighbourMode[TIMD_SECOND_BATCH], bool neighbourValid[TIMD_SECOND_BATCH],
                 int type, int log2A, int log2L, uint64_t maxCost )
{
  for( int pass = 0; pass < 2; pass++ )
  {
    const bool ofBest = pass == 0;
    if( !( s.best > maxCost && ( ( ofBest ? s.bestMode : s.secMode ) > DC_IDX || !( ofBest ? s.bestWA : s.secWA ) ) ) )
    {
      continue;
    }
    for( int j = 2 * pass; j < 2 * pass + 2; j++ )
    {
      if( !neighbourValid[j] )
      {
        continue;
      }
      const uint64_t cost = timdCost( numCands + j );
      const int      ld   = timdLocDep( type, s_spCost[TC_COST_A + numCands + j], s_spCost[TC_COST_L + numCands + j], log2A, log2L );
      if( ofBest && cost < s.best )
      {
        s.best      = cost;
        s.bestMode  = neighbourMode[j];
        s.locDep[0] = ld;
      }
      else if( !ofBest && cost < s.secondary )
      {
        s.secondary = cost;
        s.secMode   = neighbourMode[j];
        s.locDep[1] = ld;
      }
      if( ( ofBest ? s.best : s.secondary ) <= maxCost )
      {
        break;
      }
    }
  }
}

// the final choice: the regular modes of the wide angles beyond, the cheaper as the mode; a secondary costing twice the
// best or more replaced by an angular best (which then blends with itself, JVET-AG0092); blending where the secondary
// costs less than twice the best, with the non-angular mode where it costs less than 1.5 times the best and is neither
// of them (weights from the three costs), else by X0149's weights of the two. Into fields the fields deriveTimdMode
// sets (TIMD_OUT_*), with sad deriveTimdModeSad's, whose location dependences stay 0
void timdFinal( TimdScan s, int W, int H, bool sad, out int fields[TIMD_OUT_FIELDS] )
{
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
  if( sad )
  {
    s.locDep = int[3]( 0, 0, 0 );
  }
  // a mode from a block vector is planar in the fields, its vector beside them (isBvTimd, timdBv)
  const bool bvBest = s.bestMode >= TIMD_BV_IDX, bvSec = blended && s.secMode >= TIMD_BV_IDX;
  fields = int[TIMD_OUT_FIELDS]( bvBest ? PLANAR_IDX : s.bestMode, -1, -1, blended ? 1 : 0, -1, -1, -1, s.locDep[0],
                                 s.locDep[1], s.locDep[2], s.bestWA ? 1 : 0, 1, 0, 0, 0, 0 );
  if( bvBest )
  {
    fields[TIMD_OUT_BV_FLAGS] = 1;
    fields[TIMD_OUT_BV]       = bvPack( bvListAt( s.bestMode - TIMD_BV_IDX ) );
  }
  if( blended )
  {
    fields[TIMD_OUT_SECONDARY]    = bvSec ? PLANAR_IDX : s.secMode;
    fields[TIMD_OUT_SEC_CHECK_WA] = s.secWA ? 1 : 0;
    if( bvSec )
    {
      fields[TIMD_OUT_BV_FLAGS] |= 2;
      fields[TIMD_OUT_BV + 1]    = bvPack( bvListAt( s.secMode - TIMD_BV_IDX ) );
    }
    const bool nonAngFusion = ( s.nonAng < s.best || s.nonAng - s.best < ( s.best >> 1 ) ) && s.bestMode != s.nonAngMode
                              && s.secMode != s.nonAngMode && ( s.nonAngMode < TIMD_BV_IDX || ( !bvBest && !bvSec ) );
    if( nonAngFusion )   // each weight (sum - cost) / (2 sum) in 1/64 rounded, as the CPU's doubles give it
    {
      const uint64_t sum = s.best + s.secondary + s.nonAng;
      fields[TIMD_OUT_NON_ANG]    = s.nonAngMode >= TIMD_BV_IDX ? PLANAR_IDX : s.nonAngMode;
      if( s.nonAngMode >= TIMD_BV_IDX )
      {
        fields[TIMD_OUT_BV_FLAGS] |= 4;
        fields[TIMD_OUT_BV + 2]    = bvPack( bvListAt( s.nonAngMode - TIMD_BV_IDX ) );
      }
      fields[TIMD_OUT_WEIGHT]     = int( ( 64ul * ( sum - s.best ) + sum ) / ( 2ul * sum ) );
      fields[TIMD_OUT_WEIGHT + 1] = int( ( 64ul * ( sum - s.secondary ) + sum ) / ( 2ul * sum ) );
      fields[TIMD_OUT_WEIGHT + 2] = 64 - fields[TIMD_OUT_WEIGHT] - fields[TIMD_OUT_WEIGHT + 1];
    }
    else
    {
      fields[TIMD_OUT_WEIGHT]     = timdLutWeight( s.best, s.secondary );
      fields[TIMD_OUT_WEIGHT + 1] = 64 - fields[TIMD_OUT_WEIGHT];
      fields[TIMD_OUT_WEIGHT + 2] = 0;
    }
  }
}

// ----- the node

// fields into TimdOut from outBase
void timdStore( int outBase, int fields[TIMD_OUT_FIELDS] )
{
  CBufI32 timdOut = CBufI32( P.timdOut );
  for( int k = 0; k < TIMD_OUT_FIELDS; k++ )
  {
    timdOut.v[outBase + k] = fields[k];
  }
}

// in the pipelines with TIMD merge, which reads them: the transform types of TIMD CU c's luma block (the CU's
// timdmTrType as DecCu sets it after the reconstruction, TrQuant::getTrTypes), explicit MTS's by its mode (67-mode set)
// where that chooses them, else the host's, into its TimdOut
void timdStoreTransforms( TimdHead head, Cu c, int mode )
{
  if( WITH_TIMD_MERGE )
  {
    const ivec2 types   = head.mtsTrIdx >= 0 ? explicitMtsTypes( c.w, c.h, mode, 0, head.mtsTrIdx ) : ivec2( head.trHor, head.trVer );
    CBufI32     timdOut = CBufI32( P.timdOut );
    const int   outBase = head.index * TIMD_OUT;
    timdOut.v[outBase + TIMD_OUT_TR_HOR] = types.x;
    timdOut.v[outBase + TIMD_OUT_TR_VER] = types.y;
  }
}

// the node of TIMD CU c (all invocations, uniform control flow): its TIMD (a TIMD-SAD CU's also its TIMD-SAD) into its
// TimdOut; returns (invocation 0) the mode its blocks predict as the neighbours read it (MAP131TO67) and in mtssSecond
// the second mode MTSS tries first (the secondary where those modes blend, else -1: none). nbrMode, dimdModes:
// deriveLumaMode's (invocation 0's); cuIndex: the CU's record, whose DimdOut slot holds its DIMD.
// Each derivation is written out at its own call sites, with its kind (SATD / mean-removed SAD) a constant there: one
// function for both, its kind a parameter, made the TIMD-SAD pictures ~8 % slower
int timdNode( Cu c, int cuIndex, int nbrMode[MPM_NEIGHBOURS], int dimdModes[3], int bitDepth, bool noPdpc,
              out int mtssSecond, int lid, int NT )
{
  const TimdHead head    = timdHeadAt( c.timd );
  const int      outBase = head.index * TIMD_OUT;
  const bool     sad     = WITH_TIMD_SAD && head.sad != 0;   // (TIMD-SAD in its own pipeline only)
  const int      maxPel  = ( 1 << bitDepth ) - 1;
  int            fields[TIMD_OUT_FIELDS];
  mtssSecond = -1;
  if( head.type == TIMD_NONE )   // the picture's top-left CU: planar, the other fields at their initial values
  {
    fields = int[TIMD_OUT_FIELDS]( PLANAR_IDX, -1, -1, 0, -1, -1, -1, 0, 0, 0, 1, 1, 0, 0, 0, 0 );
    if( lid == 0 )
    {
      timdStore( outBase, fields );
      if( sad )
      {
        timdStore( outBase + TIMD_OUT_SAD, fields );
      }
      timdStoreTransforms( head, c, PLANAR_IDX );
    }
    return PLANAR_IDX;
  }
  const TimdAngleRange range = timdAngleRange( head.waip, c.w, c.h );
  // the location dependences' scale, and deriveTimdMode's costs': TIMD's arms of both sides (also where the template
  // lacks one)
  const int      tw = c.w <= 8 ? 2 : 4, th = c.h <= 8 ? 2 : 4;
  const int      log2A = findMSB( th ) + findMSB( c.w ), log2L = findMSB( tw ) + findMSB( c.h );
  TimdScan       scan;
  int            neighbourMode[TIMD_SECOND_BATCH];
  bool           neighbourValid[TIMD_SECOND_BATCH];

  // deriveTimdMode: with block vectors their list, then planar, DC and the vectors in batches, scanned in turn; its
  // first batch, the template's references, DC's value and costs, the CPU's scan, the second batch's costs, the
  // refinement and the final choice
  SYNC();   // (dimdSelect has read its histograms)
  if( c.bvList >= 0 && lid == 0 )
  {
    s_spCost[TC_NUM_BV] = uint( bvMergeList( c, g_ti ) );
  }
  timdFillReference( head.t, BufI32( P.timdData ), c.timd + TIMD_HEAD, 0, c.x, c.y, c.w, c.h, TIMD_REF_SHIFT,
                     bitDepth, lid, NT );
  const int dc    = timdDc( head.t, c.w, c.h, lid, NT );   // (a barrier: the list's length visible)
  const int numBv = c.bvList >= 0 ? int( s_spCost[TC_NUM_BV] ) : 0;
  if( numBv > 0 )
  {
    timdVectorBatches( c, head.t, numBv, dc, maxPel, noPdpc, false, scan, head.type, log2A, log2L, lid, NT );
  }
  if( lid == 0 )
  {
    timdMpmCandidates( c, nbrMode, dimdModes, range );
  }
  SYNC();
  const int numCands = int( s_spCost[TC_NUM] );
  timdTemplateCosts( c, head.t, TC_CAND, numCands, TC_COST_A, TC_COST_L, dc, maxPel, noPdpc, false, lid, NT );
  const uint64_t maxCost  = uint64_t( tw * c.h + th * c.w );
  int            lastList = 0;
  if( lid == 0 )
  {
    lastList = timdScanFirst( scan, numCands, int( s_spCost[TC_END_LIST] ), head.type, log2A, log2L, maxCost, false,
                              numBv > 0 );
    timdSecondBatch( scan, numCands, c.w, c.h, range, neighbourMode, neighbourValid );
  }
  SYNC();
  timdTemplateCosts( c, head.t, TC_CAND + numCands, int( s_spCost[TC_NUM_B] ), TC_COST_A, TC_COST_L, dc, maxPel, noPdpc,
                     false, lid, NT );
  if( lid == 0 )
  {
    timdRefine( scan, numCands, neighbourMode, neighbourValid, head.type, log2A, log2L, maxCost );
    timdFinal( scan, c.w, c.h, false, fields );
    timdStore( outBase, fields );
  }

  // deriveTimdModeSad (a TIMD-SAD CU) alike: its first batch from deriveTimdMode's (its costs, the modes it found), its
  // own template (its reference units after TIMD's) whose arms' sizes scale its costs (2 for an arm it lacks, as the
  // CPU clips it), the mean-removed SAD, JVET-AN0093's pass with both arms, no location dependence
  if( sad )
  {
    int sadList[TIMD_SAD_LIST];   // (invocation 0's)
    int sadNum = 0;
    if( lid == 0 )
    {
      timdSadList( fields, cuIndex, lastList, sadList, sadNum );
    }
    SYNC();
    const int unitsAt = c.timd + TIMD_HEAD + ( ( head.t.totalUnits + 31 ) >> 5 );
    timdFillReference( head.ts, BufI32( P.timdData ), unitsAt, 0, c.x, c.y, c.w, c.h, TIMD_REF_SHIFT, bitDepth, lid,
                       NT );
    const int dcSad = timdDc( head.ts, c.w, c.h, lid, NT );
    if( numBv > 0 )
    {
      timdVectorBatches( c, head.ts, numBv, dcSad, maxPel, noPdpc, true, scan, head.type, log2A, log2L, lid, NT );
    }
    if( lid == 0 )
    {
      timdSadSlots( sadList, sadNum, c.w, c.h, range );
    }
    SYNC();
    const int numSad = int( s_spCost[TC_NUM] );
    timdTemplateCosts( c, head.ts, TC_CAND, numSad, TC_COST_A, TC_COST_L, dcSad, maxPel, noPdpc, true, lid, NT );
    const uint64_t maxCostSad = uint64_t( ( head.type == TIMD_ABOVE ? 2 : head.ts.tw ) * c.h
                                          + ( head.type == TIMD_LEFT ? 2 : head.ts.th ) * c.w );
    if( lid == 0 )
    {
      timdScanFirst( scan, numSad, int( s_spCost[TC_END_LIST] ), head.type, log2A, log2L, maxCostSad,
                     head.type == TIMD_BOTH, numBv > 0 );
      timdSecondBatch( scan, numSad, c.w, c.h, range, neighbourMode, neighbourValid );
    }
    SYNC();
    timdTemplateCosts( c, head.ts, TC_CAND + numSad, int( s_spCost[TC_NUM_B] ), TC_COST_A, TC_COST_L, dcSad, maxPel, noPdpc,
                       true, lid, NT );
    if( lid == 0 )
    {
      timdRefine( scan, numSad, neighbourMode, neighbourValid, head.type, log2A, log2L, maxCostSad );
      timdFinal( scan, c.w, c.h, true, fields );
      timdStore( outBase + TIMD_OUT_SAD, fields );
    }
  }
  if( lid != 0 )
  {
    return -1;
  }
  // the modes the blocks predict: a TIMD-SAD CU's deriveTimdModeSad's; its vectors for the lists of the CUs after it
  // (getNeighBv takes a TIMD-SAD CU's of deriveTimdModeSad)
  if( fields[TIMD_OUT_BLENDED] != 0 )
  {
    mtssSecond = timdMode67( fields[TIMD_OUT_SECONDARY] );
  }
  if( c.bvList >= 0 )
  {
    ivec2 bv[3];
    int   num = 0;
    for( int k = 0; k < 3; k++ )
    {
      if( ( ( fields[TIMD_OUT_BV_FLAGS] >> k ) & 1 ) != 0 )
      {
        bv[num++] = bvUnpack( fields[TIMD_OUT_BV + k] );
      }
    }
    bvStore( c, num, bv );
  }
  const int mode = timdMode67( fields[TIMD_OUT_MODE] );
  timdStoreTransforms( head, c, mode );
  return mode;
}

// ----- a DIMD, OBIC or intra merge CU's choice of planar or a block vector (IntraPrediction::getBestNonAnglularMode,
// JVET-AG0146): with both arms of TIMD's template (its record after the list's, BV_LIST), the SATD of planar's template
// prediction and of the reconstruction at each vector of its list (getItmpMergeCandidate's, kept to its first 30 / 15)
// against the template, the first of the least cost; a vector chosen replaces planar's predictions of the CU
// (isBvDimd, bvDimd: into its DimdOut and its record of vectors). All invocations, uniform control flow
void dimdVectorChoice( Cu c, int cuIndex, int bitDepth, bool noPdpc, int lid, int NT )
{
  BufI32              data = BufI32( P.timdData );
  const TemplateUnits t    = templateUnitsAt( data, c.bvList + BV_LIST );
  SYNC();   // (the node's shared arrays are free)
  if( lid == 0 )
  {
    s_spCost[TC_NUM_BV] = uint( bvMergeList( c, g_ti ) );
  }
  timdFillReference( t, data, c.bvList + BV_LIST + TEMPLATE_UNITS, 0, c.x, c.y, c.w, c.h, TIMD_REF_SHIFT, bitDepth, lid,
                     NT );
  SYNC();
  const int n = 1 + int( s_spCost[TC_NUM_BV] );   // planar, then the vectors (at most 31 candidates)
  for( int k = lid; k < n; k += NT )
  {
    s_spCost[TC_CAND + k]   = timdCand( k == 0 ? PLANAR_IDX : TIMD_BV_IDX + k - 1, PLANAR_IDX, true, k );
    s_spCost[TC_COST_A + k] = 0u;
    s_spCost[TC_COST_L + k] = 0u;
  }
  SYNC();
  timdTemplateCosts( c, t, TC_CAND, n, TC_COST_A, TC_COST_L, 0, ( 1 << bitDepth ) - 1, noPdpc, false, lid, NT );
  if( lid == 0 )
  {
    int best = 0;
    for( int k = 1; k < n; k++ )
    {
      best = timdCost( k ) < timdCost( best ) ? k : best;
    }
    CBufI32     o  = CBufI32( P.dimdOut );
    const ivec2 bv = best > 0 ? bvListAt( best - 1 ) : ivec2( 0 );
    o.v[cuIndex * DIMD_OUT + DIMD_OUT_BV_FLAG] = best > 0 ? 1 : 0;
    o.v[cuIndex * DIMD_OUT + DIMD_OUT_BV]      = best > 0 ? bvPack( bv ) : 0;
    ivec2 bvs[3] = ivec2[3]( bv, bv, bv );
    bvStore( c, best > 0 ? 1 : 0, bvs );
  }
}

// ----- a TIMD merge CU's node (IntraPrediction::deriveTimdMergeModes, JVET-AJ0061; NUM_TIMD_MERGE_MODES 1): its
// candidates from the TIMD of its neighbours (genTimdMrgList over searchTimdMrgCus' neighbours, which the host found),
// with two or more ranked by the template cost of their modes (calcTimdMrgCandCosts: the SATD of each candidate's mode
// on the CU's template of arms one sample thick, TimdHead::t), the choice into its TimdOut

// calcTimdMrgCandCosts' swap: whether a candidate of cost second ranks before the cheapest one, of cost best, as the
// CPU compares second < 1.1 * best in doubles. 1.1 is a double 0.4 * 2^-52 above 1.1, so only where 1.1 * best is an
// integer n can the rounding decide: second = n is below where the product rounds up from n, i.e. where best * 0.4 *
// 2^-52 exceeds half n's unit in the last place, 2^(e - 53) for 2^e <= n < 2^(e + 1) (a tie rounds to n, which is
// even there); checked against the doubles for every best below 2^20 and millions of others up to 2^40
bool timdMergeSwaps( uint64_t second, uint64_t best )
{
  if( best % 10ul != 0ul )
  {
    return 10ul * second < 11ul * best;
  }
  const uint64_t n = best / 10ul * 11ul;
  if( second != n || best == 0ul )
  {
    return second < n;
  }
  const int e = n >> 32 != 0ul ? 32 + findMSB( uint( n >> 32 ) ) : findMSB( uint( n ) );
  return 4ul * best > 5ul << e;
}

// the node of TIMD merge CU c (all invocations, uniform control flow): its candidate's fields into its TimdOut, its
// transform types those the candidate gives (in dual tree, else DCT-II); returns (invocation 0) its mode as the
// neighbours read it (MAP131TO67) and in mtssSecond the second mode MTSS tries first (the secondary where the
// candidate's modes blend, else -1)
int timdMergeNode( Cu c, int bitDepth, bool noPdpc, out int mtssSecond, int lid, int NT )
{
  const TimdHead head    = timdHeadAt( c.timd );
  const int      outBase = head.index * TIMD_OUT;
  CBufI32        timdOut = CBufI32( P.timdOut );
  mtssSecond = -1;
  // genTimdMrgList (invocation 0): per neighbour the fields its blocks predict (a TIMD merge neighbour's candidate, a
  // TIMD-SAD one's TIMD-SAD), unless a candidate has its mode and secondary mode already or its mode is not set (the
  // CPU checks a TIMD merge or TIMD-SAD neighbour's only: a TIMD CU's is always set), as the candidates of the costs
  int source[TIMD_MERGE_CANDS];   // the neighbour each candidate comes from (invocation 0's)
  int num = 0;
  if( lid == 0 )
  {
    int modes[TIMD_MERGE_CANDS], secondaries[TIMD_MERGE_CANDS];
    for( int k = 0; k < head.merge; k++ )
    {
      const int predBase  = timdPredBase( head.mergeNbr[k] );
      const int mode      = timdOut.v[predBase + TIMD_OUT_MODE];
      const int secondary = timdOut.v[predBase + TIMD_OUT_SECONDARY];
      bool known = false;
      for( int i = 0; i < num; i++ )
      {
        known = known || ( modes[i] == mode && secondaries[i] == secondary );
      }
      if( !known && mode != -1 )
      {
        source[num]      = head.mergeNbr[k];
        modes[num]       = mode;
        secondaries[num] = secondary;
        s_spCost[TC_CAND + num]   = timdCand( mode, mode, true, num );
        s_spCost[TC_COST_A + num] = 0u;
        s_spCost[TC_COST_L + num] = 0u;
        num++;
      }
    }
    s_spCost[TC_NUM] = uint( num );
  }
  SYNC();
  // calcTimdMrgCandCosts where there are two or more: the SATD of each candidate's mode on the template
  const int numCands = int( s_spCost[TC_NUM] );
  if( numCands >= 2 )
  {
    timdFillReference( head.t, BufI32( P.timdData ), c.timd + TIMD_HEAD, 0, c.x, c.y, c.w, c.h, TIMD_REF_SHIFT,
                       bitDepth, lid, NT );
    const int dc = timdDc( head.t, c.w, c.h, lid, NT );
    timdTemplateCosts( c, head.t, TC_CAND, numCands, TC_COST_A, TC_COST_L, dc, ( 1 << bitDepth ) - 1, noPdpc, false, lid, NT );
  }
  if( lid != 0 )
  {
    return -1;
  }
  // the two cheapest (updateCandList: the earlier of equal costs first), the second where it costs less than 1.1 times
  // the cheapest
  int chosen = 0;
  if( num >= 2 )
  {
    int best = 0, second = -1;
    for( int i = 1; i < num; i++ )
    {
      if( timdCost( i ) < timdCost( best ) )
      {
        second = best;
        best   = i;
      }
      else if( second < 0 || timdCost( i ) < timdCost( second ) )
      {
        second = i;
      }
    }
    chosen = timdMergeSwaps( timdCost( second ), timdCost( best ) ) ? second : best;
  }
  // its fields (every mode's wide angle getWideAngleExt's: timdMrgModeCheckWA), its transform types; without a
  // candidate (the CPU cannot decode such a CU) planar
  int fields[TIMD_OUT_FIELDS] = int[TIMD_OUT_FIELDS]( PLANAR_IDX, -1, -1, 0, -1, -1, -1, 0, 0, 0, 1, 1, 0, 0, 0, 0 );
  int trHor = TR_DCT2, trVer = TR_DCT2;
  if( num > 0 )
  {
    const int predBase = timdPredBase( source[chosen] ), sourceBase = timdOutBase( source[chosen] );
    for( int f = 0; f < TIMD_OUT_CHECK_WA; f++ )
    {
      fields[f] = timdOut.v[predBase + f];
    }
    if( head.inheritTr != 0 )
    {
      trHor = timdOut.v[sourceBase + TIMD_OUT_TR_HOR];
      trVer = timdOut.v[sourceBase + TIMD_OUT_TR_VER];
    }
  }
  timdStore( outBase, fields );
  timdOut.v[outBase + TIMD_OUT_TR_HOR] = trHor;
  timdOut.v[outBase + TIMD_OUT_TR_VER] = trVer;
  if( fields[TIMD_OUT_BLENDED] != 0 )
  {
    mtssSecond = timdMode67( fields[TIMD_OUT_SECONDARY] );
  }
  return timdMode67( fields[TIMD_OUT_MODE] );
}

#endif
