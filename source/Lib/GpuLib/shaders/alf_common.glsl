// helpers of the ALF kernels (AdaptiveLoopFilter.cpp); all arithmetic is the integer arithmetic of the C++ code

int clipAlf2( int clip, int ref, int v0, int v1 )
{
  return clamp( v0 - ref, -clip, clip ) + clamp( v1 - ref, -clip, clip );
}

int clipAlf1( int clip, int ref, int v )
{
  return clamp( v - ref, -clip, clip );
}

// sample of a picture-size plane whose padding is the mirror image of the picture (edge sample repeated)
int mirror( int v, int size )
{
  return v < 0 ? -1 - v : ( v >= size ? 2 * size - 1 - v : v );
}

int ldMirror( Plane p, int x, int y, int w, int h )
{
  return ld( p, mirror( x, w ), mirror( y, h ) );
}

// per-sample gradients of AdaptiveLoopFilter::deriveClassificationLaplacian: vertical, horizontal, two diagonals
uvec4 gradients( Plane p, int x, int y )
{
  const int c2 = ld( p, x, y ) << 1;
  const int u  = ld( p, x, y - 1 ), d = ld( p, x, y + 1 );
  const int l  = ld( p, x - 1, y ), r = ld( p, x + 1, y );
  const int ul = ld( p, x - 1, y - 1 ), dr = ld( p, x + 1, y + 1 );
  const int ur = ld( p, x + 1, y - 1 ), dl = ld( p, x - 1, y + 1 );
  return uvec4( abs( c2 - u - d ), abs( c2 - l - r ), abs( c2 - ul - dr ), abs( c2 - ur - dl ) );
}

const int ALF_TH193[193] = int[193](
  0,  1,  2,  3,  4,  4,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
  11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
  12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
  13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
  14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
  14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15 );
const int ALF_TH16[16]      = int[16]( 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4 );
const int ALF_DIVSHIFT2[16] = int[16]( 2, 2, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10, 10 );
const int ALF_SQRTSUM[50]   = int[50]( 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                       5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7 );
const int ALF_TRANSPOSE[8]  = int[8]( 0, 1, 0, 2, 2, 3, 1, 3 );
const int ALF_MULT[6]       = int[6]( 5628, 1407, 624, 351, 225, 156 );

// AdaptiveLoopFilter::m_mappingDir: index of the lower triangle, 0 above it
int mappingDir( int i, int j )
{
  return j <= i ? ( i * ( i + 1 ) ) / 2 + j : 0;
}

// class of a 2x2 block from the window sums of the gradients (v, h, d0, d1): AdaptiveLoopFilter::calcClass0Var
// (fixed filters: classDir 1, 7 directions, 16 activities, variance) and calcClass (classDir 0, 5 / 5)
int edgeStrength( uint a1, uint a0 )
{
  return a1 > 8u * a0 ? 6 : a1 * 2u > 9u * a0 ? 5 : a1 > 3u * a0 ? 4 : a1 > 2u * a0 ? 3 : a1 * 2u > 3u * a0 ? 2
         : a1 * 4u > 5u * a0 ? 1 : 0;
}

int classFixed( uvec4 g, int mult, int bitDepth, uint variance )
{
  const int  shift = 9 + bitDepth;
  const int  sumV = int( g.x ), sumH = int( g.y ), sumD0 = int( g.z ), sumD1 = int( g.w );
  const int  sum  = sumV + sumH;
  int        hv1, hv0, d1, d0, dirTempHV, dirTempD;
  if( sumV > sumH ) { hv1 = sumV; hv0 = sumH; dirTempHV = 1; } else { hv1 = sumH; hv0 = sumV; dirTempHV = 3; }
  if( sumD0 > sumD1 ) { d1 = sumD0; d0 = sumD1; dirTempD = 0; } else { d1 = sumD1; d0 = sumD0; dirTempD = 2; }
  int mainDirection, secondaryDirection;
  if( uint64_t( d1 ) * uint64_t( hv0 ) > uint64_t( d0 ) * uint64_t( hv1 ) )
  {
    mainDirection = dirTempD; secondaryDirection = dirTempHV;
  }
  else
  {
    mainDirection = dirTempHV; secondaryDirection = dirTempD;
  }
  const int activity = ALF_TH193[min( ( mult * sum ) >> ( shift - 4 ), 192 )];
  const int esHV = edgeStrength( uint( hv1 ), uint( hv0 ) );
  const int esD  = edgeStrength( uint( d1 ), uint( d0 ) );
  const int direction = uint64_t( hv1 ) * uint64_t( d0 ) > uint64_t( hv0 ) * uint64_t( d1 ) ? mappingDir( esHV, esD )
                                                                                         : 28 + mappingDir( esD, esHV );
  int        actDirInd = 7 * 8 * activity + direction;
  const uint sum2      = min( variance >> uint( ALF_DIVSHIFT2[activity] ), 49u );
  actDirInd            = ALF_SQRTSUM[sum2] * 7 * 8 * 16 + actDirInd;
  return ( actDirInd << 2 ) + ALF_TRANSPOSE[mainDirection * 2 + ( secondaryDirection >> 1 )];
}

int classBand( uvec4 g, int mult, int bitDepth )   // calcClass with classDir 0, 5 directions, 5 activities
{
  const int shift = 9 + bitDepth;
  const int sumV = int( g.x ), sumH = int( g.y ), sumD0 = int( g.z ), sumD1 = int( g.w );
  const int sum  = sumV + sumH;
  int       hv1, hv0, d1, d0, dirTempHV, dirTempD;
  if( sumV > sumH ) { hv1 = sumV; hv0 = sumH; dirTempHV = 1; } else { hv1 = sumH; hv0 = sumV; dirTempHV = 3; }
  if( sumD0 > sumD1 ) { d1 = sumD0; d0 = sumD1; dirTempD = 0; } else { d1 = sumD1; d0 = sumD0; dirTempD = 2; }
  int mainDirection, secondaryDirection, hvd1, hvd0;
  if( uint64_t( d1 ) * uint64_t( hv0 ) > uint64_t( d0 ) * uint64_t( hv1 ) )
  {
    mainDirection = dirTempD; secondaryDirection = dirTempHV; hvd1 = d1; hvd0 = d0;
  }
  else
  {
    mainDirection = dirTempHV; secondaryDirection = dirTempD; hvd1 = hv1; hvd0 = hv0;
  }
  const int activity          = ALF_TH16[min( ( mult * 3 * sum ) >> shift, 15 )];
  const int directionStrength = hvd1 * 2 > 9 * hvd0 ? 2 : ( hvd1 > 2 * hvd0 ? 1 : 0 );
  const int direction         = directionStrength != 0 ? ( ( mainDirection & 1 ) << 1 ) + directionStrength : 0;
  const int actDirInd         = 5 * activity + direction;
  return ( actDirInd << 2 ) + ALF_TRANSPOSE[mainDirection * 2 + ( secondaryDirection >> 1 )];
}

// AdaptiveLoopFilter::fixedFilterBlk / fixedFilteringResi: stronger correction next to block boundaries / residuals
int codingInfoAdjust( int sum, int ciClass, bool useBound, bool useResi, bool intra, int bsFactor, int resiFactor,
                      int offsetClip )
{
  if( useBound )
  {
    sum            = clamp( sum, -offsetClip, offsetClip );
    const int sign = sum < 0 ? -1 : 1;
    sum            = ( ciClass >> 1 ) != 0 ? sign * ( ( abs( sum ) * ( 16 + bsFactor ) + 8 ) >> 4 ) : sum;
  }
  if( useResi )
  {
    sum            = clamp( sum, -offsetClip, offsetClip );
    const int sign = sum < 0 ? -1 : 1;
    sum            = ( ciClass & 1 ) != 0 ? sign * ( ( abs( sum ) * ( 16 + resiFactor ) + 8 ) >> 4 ) : sum;
  }
  return sum;
}
