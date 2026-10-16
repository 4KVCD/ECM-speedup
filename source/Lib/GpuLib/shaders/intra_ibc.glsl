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
// region's size, the filter's taps across and down (0: that fraction is 0), whether they are padded
struct IbcRef
{
  int  ox, oy, w, h, xTap, yTap;
  bool pad;
  int  before;
};

// reference sample (x, y) of g (relative to its top-left), padded as xPredIBCBlkPadding pads: first per row of the
// region the samples left / right of it from the first unit that is not decoded on take the sample next to that unit,
// then per column the rows above / below it take the one nearer the region where their unit is not decoded
int ibcRef( IbcRef g, int x, int y )
{
  if( g.pad )
  {
    OwnerMap  om = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
    const int L  = om.log2X, U = 1 << L;
    // (rows outside the region exist only with yTap: the nearest decoded row towards it, else its edge row)
    const int dir = y < 0 ? 1 : -1;
    for( ; ( y < 0 || y >= g.h ) && !decodedBefore( om, g.ox + x, g.oy + y, g.before ); y += dir ) {}
    if( y >= 0 && y < g.h )
    {
      // (columns outside it exist only with xTap)
      if( x < 0 )
      {
        for( int u = ( g.ox >> L ) - 1; ( u << L ) + U - 1 - g.ox >= 1 - ( g.xTap >> 1 ); u-- )
        {
          if( !decodedBefore( om, u << L, g.oy + y, g.before ) )
          {
            const int c = ( u << L ) + U - 1 - g.ox;   // (the unit's last column)
            x = x <= c ? c + 1 : x;
            break;
          }
        }
      }
      else if( x >= g.w )
      {
        for( int u = ( ( g.ox + g.w - 1 ) >> L ) + 1; ( u << L ) - ( g.ox + g.w - 1 ) <= ( g.xTap >> 1 ); u++ )
        {
          if( !decodedBefore( om, u << L, g.oy + y, g.before ) )
          {
            const int c = ( u << L ) - g.ox;   // (the unit's first column)
            x = x >= c ? c - 1 : x;
            break;
          }
        }
      }
    }
  }
  return recoAt( 0, g.ox + x, g.oy + y );
}

// sample (i, j) of the prediction of the region of w x h at (px, py) at vector mv (1/16 samples) with the luma filter
// or the bilinear one, for block `before` (one call site per module: ibcRef inlines)
int ibcPredict( int px, int py, int w, int h, ivec2 mv, bool bilinear, int i, int j, int before, int bitDepth )
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

#endif
