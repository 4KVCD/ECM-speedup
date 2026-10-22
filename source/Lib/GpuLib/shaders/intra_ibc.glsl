// IBC's interpolation (InterPrediction::getPredIBCBlk -> xPredInterBlk with isIBC: luma, rndRes, without LIC): the
// prediction of a region of w x h at (px, py) - a CU's position, which checkValidBv takes, even for a template - from
// the reconstruction at a vector in 1/16 samples. At an integer vector (or without the SPS's fractional IBC, which
// takes the vector's integer part) a copy; else InterpolationFilter's luma filter of 8 taps (filterIdx -1:
// m_lumaFilter, IF_12TAP's x 4 table) or the bilinear one (filterIdx 1: SGPM's templates), in one pass rounded to the
// sample or in two through the 14-bit intermediate. Where the filter's reach is not all decoded (checkValidBv:
// IBC_INT_BV_VALID), xPredIBCBlkPadding's padding. Included by intra_rec.comp after intra_bv.glsl.
#ifndef INTRA_IBC_GLSL
#define INTRA_IBC_GLSL

// m_lumaFilter (8 taps, x 4: IF_FILTER_PREC 8) per fraction of 1/16
const int IBC_LUMA_FILTER[16 * 8] = int[16 * 8](
    0,   0,   0, 256,   0,   0,   0,  0,     0,   4, -12, 252,  16,  -8,   4,  0,
   -4,   8, -20, 248,  32, -12,   4,  0,    -4,  12, -32, 240,  52, -16,   4,  0,
   -4,  16, -40, 232,  68, -20,   4,  0,    -4,  16, -44, 208, 104, -32,  12, -4,
   -4,  12, -36, 188, 124, -40,  16, -4,    -4,  16, -44, 180, 136, -40,  16, -4,
   -4,  16, -44, 160, 160, -44,  16, -4,    -4,  16, -40, 136, 180, -44,  16, -4,
   -4,  16, -40, 124, 188, -36,  12, -4,    -4,  12, -32, 104, 208, -44,  16, -4,
    0,   4, -20,  68, 232, -40,  16, -4,     0,   4, -16,  52, 240, -32,  12, -4,
    0,   4, -12,  32, 248, -20,   8, -4,     0,   4,  -8,  16, 252, -12,   4,  0 );

// tap k of fraction f: the luma filter's, or the bilinear one's (m_bilinearFilter, x 4)
int ibcTap( bool bilinear, int f, int k )
{
  return bilinear ? ( k == 0 ? 256 - 16 * f : 16 * f ) : IBC_LUMA_FILTER[f * 8 + k];
}

// PU::searchBv with checkValidBv's filter reach (xTap / yTap taps where the vector has a fraction; JVET-AJ0172's
// reference area): the region of w x h at (px, py) moved by integer offset b reads only decoded samples for block
// `before` - inside the picture, not right of and below the region's position at once, not below its CTU row nor
// right of its CTU in that row, its corners decoded
bool ibcAreaValid( int px, int py, int w, int h, ivec2 b, int xTap, int yTap, int before )
{
  OwnerMap  om = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  const int s  = findLSB( P.ctuSize );
  int       l  = px + b.x, r = px + b.x + w - 1, t = py + b.y, d = py + b.y + h - 1;
  if( xTap > 0 )
  {
    l -= ( xTap >> 1 ) - 1;
    r += xTap >> 1;
  }
  if( yTap > 0 )
  {
    t -= ( yTap >> 1 ) - 1;
    d += yTap >> 1;
  }
  return l >= 0 && r < P.width[0] && t >= 0 && d < P.height[0] && !( r >= px && d >= py )
         && ( d >> s ) <= ( py >> s ) && !( ( t >> s ) == ( py >> s ) && ( r >> s ) > ( px >> s ) )
         && decodedBefore( om, r, d, before ) && decodedBefore( om, l, t, before );
}

// the reference samples of a region's interpolation: their top-left (the region at the vector's integer part), the
// region's size, the filter's taps across and down (0: that fraction is 0), whether they are padded, and how far the
// padding's region lies right of / below that top-left (xPredIBCBlkPadding's deltaX / deltaY: 1 for an IntraTMP CU's
// sub-sample vector whose integer part is left of / above its integer vector; the filter reads from the top-left
// still, but the padding takes the region there)
struct IbcRef
{
  int  ox, oy, w, h, xTap, yTap;
  bool pad;
  int  before, dx, dy;
  int  comp;   // (the component; DBV's chroma: ibcChromaPredict)
};

// reference sample (x, y) of g (relative to its top-left), padded as xPredIBCBlkPadding pads: first per row of the
// region the samples left / right of it from the first unit that is not decoded on take the sample next to that unit,
// then per column the rows above / below it take the one nearer the region where their unit is not decoded
int ibcRef( IbcRef g, int x, int y )
{
  const int ch = g.comp == 0 ? 0 : 1;
  if( g.pad )
  {
    OwnerMap  om = OwnerMap( BufI32( P.owner[g.comp] ), P.ownerW[ch], P.ownerH[ch], P.unitLog2X[ch], P.unitLog2Y[ch] );
    const int L  = om.log2X, U = 1 << L;
    const int rx = g.ox + g.dx, ry = g.oy + g.dy;   // (the padding's region; x, y relative to it from here)
    x -= g.dx;
    y -= g.dy;
    // (rows outside the region exist only with yTap: the nearest decoded row towards it, else its edge row)
    const int dir = y < 0 ? 1 : -1;
    for( ; ( y < 0 || y >= g.h ) && !decodedBefore( om, rx + x, ry + y, g.before ); y += dir ) {}
    if( y >= 0 && y < g.h )
    {
      // (columns outside it exist only with xTap)
      if( x < 0 )
      {
        for( int u = ( rx >> L ) - 1; ( u << L ) + U - 1 - rx >= 1 - g.dx - ( g.xTap >> 1 ); u-- )
        {
          if( !decodedBefore( om, u << L, ry + y, g.before ) )
          {
            const int c = ( u << L ) + U - 1 - rx;   // (the unit's last column)
            x = x <= c ? c + 1 : x;
            break;
          }
        }
      }
      else if( x >= g.w )
      {
        for( int u = ( ( rx + g.w - 1 ) >> L ) + 1; ( u << L ) - ( rx + g.w - 1 ) <= ( g.xTap >> 1 ) - g.dx; u++ )
        {
          if( !decodedBefore( om, u << L, ry + y, g.before ) )
          {
            const int c = ( u << L ) - rx;   // (the unit's first column)
            x = x >= c ? c - 1 : x;
            break;
          }
        }
      }
    }
    return recoAt( g.comp, rx + x, ry + y );
  }
  return recoAt( g.comp, g.ox + x, g.oy + y );
}

// sample (i, j) of the prediction of the region of w x h at (px, py) at vector mv (1/16 samples) with the luma filter
// or the bilinear one, for block `before`; delta: the padding's deltaX / deltaY (IbcRef). (Few call sites: ibcRef
// inlines.)
int ibcPredict( int px, int py, int w, int h, ivec2 mv, bool bilinear, int i, int j, int before, int bitDepth,
                ivec2 delta )
{
  IbcRef    g;
  const int fx = P.ibcFrac != 0 ? mv.x & 15 : 0, fy = P.ibcFrac != 0 ? mv.y & 15 : 0;
  const int taps = bilinear ? 2 : 8, lead = ( taps >> 1 ) - 1;   // (taps before the sample)
  g.ox     = px + ( mv.x >> 4 );
  g.oy     = py + ( mv.y >> 4 );
  g.w      = w;
  g.h      = h;
  g.xTap   = fx != 0 ? taps : 0;
  g.yTap   = fy != 0 ? taps : 0;
  g.before = before;
  g.dx     = delta.x;
  g.dy     = delta.y;
  g.comp   = 0;
  g.pad    = ( fx != 0 || fy != 0 )
           && !ibcAreaValid( px, py, w, h, ivec2( g.ox - px, g.oy - py ), g.xTap, g.yTap, before );
  // per row of the filter's reach down (one without yTap) the horizontal pass (the sample without xTap), then the
  // vertical one: rounded to the sample where only one pass filters, else the first pass to the intermediate
  // (IF_INTERNAL_PREC 14, less its offset) and the second from it
  const bool both     = fx != 0 && fy != 0;
  const int  headRoom = max( 2, 14 - bitDepth ), sh1 = 8 - headRoom, sh2 = 8 + headRoom;
  const int  nx = fx != 0 ? taps : 1, ny = fy != 0 ? taps : 1;
  int        sum = 0, row = 0;
  for( int n = 0; n < nx * ny; n++ )
  {
    const int k = n / nx, m = n % nx;
    const int v = ibcRef( g, fx != 0 ? i - lead + m : i, fy != 0 ? j - lead + k : j )
                  * ( fx != 0 ? ibcTap( bilinear, fx, m ) : 1 );
    if( !both )
    {
      sum += v * ( fy != 0 ? ibcTap( bilinear, fy, k ) : 1 );
      continue;
    }
    // (the two-pass form: a row's horizontal sum, then its intermediate into the vertical one)
    row = m == 0 ? v : row + v;
    if( m == nx - 1 )
    {
      sum += ibcTap( bilinear, fy, k ) * ( ( row - ( 8192 << sh1 ) ) >> sh1 );
    }
  }
  const int maxPel = ( 1 << bitDepth ) - 1;
  if( fx == 0 && fy == 0 )
  {
    return sum;
  }
  return clamp( both ? ( sum + ( 1 << ( sh2 - 1 ) ) + ( 8192 << 8 ) ) >> sh2 : ( sum + 128 ) >> 8, 0, maxPel );
}

// ----- DBV's chroma (IntraPrediction::predIntraDbv of a single tree: getPredIBCBlk of a chroma component, 4:2:0)

// m_chromaFilter (6 taps, x 4: IF_FILTER_PREC 8) per fraction of 1/32
const int IBC_CHROMA_FILTER[32 * 6] = int[32 * 6](
    0,   0, 256,   0,   0,   0,      1,  -6, 256,   7,  -2,   0,
    2, -11, 253,  15,  -4,   1,      3, -16, 251,  23,  -6,   1,
    4, -21, 248,  33, -10,   2,      5, -25, 244,  42, -12,   2,
    7, -30, 239,  53, -17,   4,      7, -32, 234,  62, -19,   4,
    8, -35, 227,  73, -22,   5,      9, -38, 220,  84, -26,   7,
   10, -40, 213,  95, -29,   7,     10, -41, 204, 106, -31,   8,
   10, -42, 196, 117, -34,   9,     10, -41, 187, 127, -35,   8,
   11, -42, 177, 138, -38,  10,     10, -41, 168, 148, -39,  10,
   10, -40, 158, 158, -40,  10,     10, -39, 148, 168, -41,  10,
   10, -38, 138, 177, -42,  11,      8, -35, 127, 187, -41,  10,
    9, -34, 117, 196, -42,  10,      8, -31, 106, 204, -41,  10,
    7, -29,  95, 213, -40,  10,      7, -26,  84, 220, -38,   9,
    5, -22,  73, 227, -35,   8,      4, -19,  62, 234, -32,   7,
    4, -17,  53, 239, -30,   7,      2, -12,  42, 244, -25,   5,
    2, -10,  33, 248, -21,   4,      1,  -6,  23, 251, -16,   3,
    1,  -4,  15, 253, -11,   2,      0,  -2,   7, 256,  -6,   1 );

// PU::searchBv of chroma component comp (checkValidBv's form, in luma samples: the region of w x h at chroma (px, py)
// and the integer part of vector mv (1/16 luma samples, 1/32 chroma) x 2, its last column / row one luma sample
// short; with the SPS's fractional IBC xTap / yTap luma taps of reach): inside the picture, not right of and below the
// region's position at once, not below its CTU row nor right of its CTU in that row, its corners (in chroma samples)
// decoded for block `before`
bool ibcChromaValid( int comp, int px, int py, int w, int h, ivec2 mv, int xTap, int yTap, int before )
{
  OwnerMap  om = OwnerMap( BufI32( P.owner[comp] ), P.ownerW[1], P.ownerH[1], P.unitLog2X[1], P.unitLog2Y[1] );
  const int s  = findLSB( P.ctuSize );
  const int x0 = px << 1, y0 = py << 1;
  int       l  = x0 + ( ( mv.x >> 5 ) << 1 ), t = y0 + ( ( mv.y >> 5 ) << 1 );
  int       r  = l + ( w << 1 ) - 2, d = t + ( h << 1 ) - 2;
  if( P.ibcFrac != 0 && xTap > 0 )
  {
    l -= ( xTap >> 1 ) - 2;
    r += xTap >> 1;
  }
  if( P.ibcFrac != 0 && yTap > 0 )
  {
    t -= ( yTap >> 1 ) - 2;
    d += yTap >> 1;
  }
  return l >= 0 && r < P.width[0] && t >= 0 && d < P.height[0] && !( r >= x0 && d >= y0 )
         && ( d >> s ) <= ( y0 >> s ) && !( ( t >> s ) == ( y0 >> s ) && ( r >> s ) > ( x0 >> s ) )
         && decodedBefore( om, r >> 1, d >> 1, before ) && decodedBefore( om, l >> 1, t >> 1, before );
}

// sample (i, j) of the chroma prediction of the region of w x h at (px, py) of component comp at vector mv (1/16 luma
// samples) for block `before` (xPredInterBlk with isIBC: without the SPS's fractional IBC, or at an integer chroma
// vector, the copy at the integer part; else m_chromaFilter in one pass rounded to the sample or in two through the
// 14-bit intermediate, padded as xPredIBCBlkPadding pads where checkValidBv's reach - 6 chroma taps - is not all
// decoded: IBC_INT_BV_VALID)
int ibcChromaPredict( int comp, int px, int py, int w, int h, ivec2 mv, int i, int j, int before, int bitDepth )
{
  IbcRef    g;
  const int fx = P.ibcFrac != 0 ? mv.x & 31 : 0, fy = P.ibcFrac != 0 ? mv.y & 31 : 0;
  const int taps = 6, lead = 2;
  g.ox     = px + ( mv.x >> 5 );
  g.oy     = py + ( mv.y >> 5 );
  g.w      = w;
  g.h      = h;
  g.xTap   = fx != 0 ? taps : 0;
  g.yTap   = fy != 0 ? taps : 0;
  g.before = before;
  g.dx     = 0;
  g.dy     = 0;
  g.comp   = comp;
  g.pad    = ( fx != 0 || fy != 0 ) && !ibcChromaValid( comp, px, py, w, h, mv, fx != 0 ? 2 * taps : 0,
                                                       fy != 0 ? 2 * taps : 0, before );
  const bool both     = fx != 0 && fy != 0;
  const int  headRoom = max( 2, 14 - bitDepth ), sh1 = 8 - headRoom, sh2 = 8 + headRoom;
  const int  nx = fx != 0 ? taps : 1, ny = fy != 0 ? taps : 1;
  int        sum = 0, row = 0;
  for( int n = 0; n < nx * ny; n++ )
  {
    const int k = n / nx, m = n % nx;
    const int v = ibcRef( g, fx != 0 ? i - lead + m : i, fy != 0 ? j - lead + k : j )
                  * ( fx != 0 ? IBC_CHROMA_FILTER[fx * 6 + m] : 1 );
    if( !both )
    {
      sum += v * ( fy != 0 ? IBC_CHROMA_FILTER[fy * 6 + k] : 1 );
      continue;
    }
    row = m == 0 ? v : row + v;
    if( m == nx - 1 )
    {
      sum += IBC_CHROMA_FILTER[fy * 6 + k] * ( ( row - ( 8192 << sh1 ) ) >> sh1 );
    }
  }
  if( fx == 0 && fy == 0 )
  {
    return sum;
  }
  return clamp( both ? ( sum + ( 1 << ( sh2 - 1 ) ) + ( 8192 << 8 ) ) >> sh2 : ( sum + 128 ) >> 8, 0,
                ( 1 << bitDepth ) - 1 );
}

#endif
