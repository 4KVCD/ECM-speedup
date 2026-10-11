// IEEE 754 binary64 arithmetic in integer operations, for GPUs without FP64. Values are the raw bits in a uint64_t.
// Every operation rounds to nearest even and keeps subnormals, which is what SSE2 does with the default MXCSR, so
// the results equal the CPU decoder's double results bit for bit. NaN results follow x86: an invalid operation
// gives the default NaN 0xFFF8000000000000, and a NaN operand is returned quieted (the first one if both are NaN).
// The rounding core follows the structure of Berkeley SoftFloat 3: a significand normalized to bit 62 with the
// value sig * 2^(exp - 1084).

const uint64_t F64_SIGN      = 0x8000000000000000UL;
const uint64_t F64_ABS       = 0x7FFFFFFFFFFFFFFFUL;
const uint64_t F64_FRAC      = 0x000FFFFFFFFFFFFFUL;
const uint64_t F64_HIDDEN    = 0x0010000000000000UL;
const uint64_t F64_QUIET     = 0x0008000000000000UL;
const uint64_t F64_DEFAULT_NAN = 0xFFF8000000000000UL;
const uint64_t F64_ONE       = 0x3FF0000000000000UL;

int      f64Exp( uint64_t a )  { return int( ( a >> 52 ) & 0x7FFUL ); }
uint64_t f64Frac( uint64_t a ) { return a & F64_FRAC; }
bool     f64Sign( uint64_t a ) { return ( a >> 63 ) != 0UL; }
bool     f64IsNaN( uint64_t a ) { return ( a & F64_ABS ) > 0x7FF0000000000000UL; }
uint64_t f64Pack( bool sign, int exp, uint64_t sig ) { return ( sign ? F64_SIGN : 0UL ) + ( uint64_t( exp ) << 52 ) + sig; }

int clz64( uint64_t a )
{
  const uint hi = uint( a >> 32 );
  return hi != 0u ? 31 - findMSB( hi ) : 63 - findMSB( uint( a ) );   // findMSB(0) is -1, so clz64(0) is 64
}

uint64_t shiftRightJam64( uint64_t a, int dist )
{
  if( dist <= 0 )
  {
    return a;
  }
  return dist < 63 ? ( a >> dist ) | ( ( a << ( 64 - dist ) ) != 0UL ? 1UL : 0UL ) : ( a != 0UL ? 1UL : 0UL );
}

// Propagates a NaN operand the way SSE2 does.
uint64_t f64PropagateNaN( uint64_t a, uint64_t b )
{
  return f64IsNaN( a ) ? ( a | F64_QUIET ) : ( b | F64_QUIET );
}

uint64_t f64RoundPack( bool sign, int exp, uint64_t sig )
{
  uint roundBits = uint( sig & 0x3FFUL );
  if( exp < 0 || exp >= 0x7FD )
  {
    if( exp < 0 )
    {
      sig       = shiftRightJam64( sig, -exp );
      exp       = 0;
      roundBits = uint( sig & 0x3FFUL );
    }
    else if( exp > 0x7FD || sig + 0x200UL >= 0x8000000000000000UL )
    {
      return f64Pack( sign, 0x7FF, 0UL );
    }
  }
  sig = ( sig + 0x200UL ) >> 10;
  if( roundBits == 0x200u )
  {
    sig &= ~1UL;   // a tie rounds to even
  }
  if( sig == 0UL )
  {
    exp = 0;
  }
  return f64Pack( sign, exp, sig );
}

uint64_t f64NormRoundPack( bool sign, int exp, uint64_t sig )
{
  const int shiftDist = clz64( sig ) - 1;
  exp -= shiftDist;
  if( shiftDist >= 10 && exp >= 0 && exp < 0x7FD )
  {
    return f64Pack( sign, sig != 0UL ? exp : 0, sig << ( shiftDist - 10 ) );
  }
  return f64RoundPack( sign, exp, sig << shiftDist );
}

uint64_t f64AddMags( uint64_t a, uint64_t b, bool signZ )
{
  int      expA = f64Exp( a ), expB = f64Exp( b );
  uint64_t sigA = f64Frac( a ), sigB = f64Frac( b );
  const int expDiff = expA - expB;
  int      expZ;
  uint64_t sigZ;
  if( expDiff == 0 )
  {
    if( expA == 0 )
    {
      return a + sigB;
    }
    if( expA == 0x7FF )
    {
      return ( sigA | sigB ) != 0UL ? f64PropagateNaN( a, b ) : a;
    }
    expZ = expA;
    sigZ = ( 0x0020000000000000UL + sigA + sigB ) << 9;
    return f64RoundPack( signZ, expZ, sigZ );
  }
  sigA <<= 9;
  sigB <<= 9;
  if( expDiff < 0 )
  {
    if( expB == 0x7FF )
    {
      return sigB != 0UL ? f64PropagateNaN( a, b ) : f64Pack( signZ, 0x7FF, 0UL );
    }
    expZ = expB;
    sigA = expA != 0 ? sigA + 0x2000000000000000UL : sigA << 1;
    sigA = shiftRightJam64( sigA, -expDiff );
  }
  else
  {
    if( expA == 0x7FF )
    {
      return sigA != 0UL ? f64PropagateNaN( a, b ) : a;
    }
    expZ = expA;
    sigB = expB != 0 ? sigB + 0x2000000000000000UL : sigB << 1;
    sigB = shiftRightJam64( sigB, expDiff );
  }
  sigZ = 0x2000000000000000UL + sigA + sigB;
  if( sigZ < 0x4000000000000000UL )
  {
    expZ--;
    sigZ <<= 1;
  }
  return f64RoundPack( signZ, expZ, sigZ );
}

uint64_t f64SubMags( uint64_t a, uint64_t b, bool signZ )
{
  int      expA = f64Exp( a ), expB = f64Exp( b );
  uint64_t sigA = f64Frac( a ), sigB = f64Frac( b );
  const int expDiff = expA - expB;
  if( expDiff == 0 )
  {
    if( expA == 0x7FF )
    {
      return ( sigA | sigB ) != 0UL ? f64PropagateNaN( a, b ) : F64_DEFAULT_NAN;
    }
    int64_t sigDiff = int64_t( sigA ) - int64_t( sigB );
    if( sigDiff == 0L )
    {
      return 0UL;
    }
    if( expA != 0 )
    {
      expA--;
    }
    if( sigDiff < 0L )
    {
      signZ   = !signZ;
      sigDiff = -sigDiff;
    }
    int shiftDist = clz64( uint64_t( sigDiff ) ) - 11;
    int expZ      = expA - shiftDist;
    if( expZ < 0 )
    {
      shiftDist = expA;
      expZ      = 0;
    }
    return f64Pack( signZ, expZ, uint64_t( sigDiff ) << shiftDist );
  }
  sigA <<= 10;
  sigB <<= 10;
  int      expZ;
  uint64_t sigZ;
  if( expDiff < 0 )
  {
    signZ = !signZ;
    if( expB == 0x7FF )
    {
      return sigB != 0UL ? f64PropagateNaN( a, b ) : f64Pack( signZ, 0x7FF, 0UL );
    }
    sigA += expA != 0 ? 0x4000000000000000UL : sigA;
    sigA = shiftRightJam64( sigA, -expDiff );
    sigB |= 0x4000000000000000UL;
    expZ = expB;
    sigZ = sigB - sigA;
  }
  else
  {
    if( expA == 0x7FF )
    {
      return sigA != 0UL ? f64PropagateNaN( a, b ) : a;
    }
    sigB += expB != 0 ? 0x4000000000000000UL : sigB;
    sigB = shiftRightJam64( sigB, expDiff );
    sigA |= 0x4000000000000000UL;
    expZ = expA;
    sigZ = sigA - sigB;
  }
  return f64NormRoundPack( signZ, expZ - 1, sigZ );
}

uint64_t f64Add( uint64_t a, uint64_t b )
{
  const bool signA = f64Sign( a );
  return signA == f64Sign( b ) ? f64AddMags( a, b, signA ) : f64SubMags( a, b, signA );
}

uint64_t f64Sub( uint64_t a, uint64_t b )
{
  const bool signA = f64Sign( a );
  return signA != f64Sign( b ) ? f64AddMags( a, b, signA ) : f64SubMags( a, b, signA );
}

// Normalizes a subnormal significand: returns the shifted significand and sets exp.
uint64_t f64NormSubnormal( uint64_t sig, out int exp )
{
  const int shiftDist = clz64( sig ) - 11;
  exp = 1 - shiftDist;
  return sig << shiftDist;
}

// 64 x 64 -> 128-bit product as (high, low).
u64vec2 mul64To128( uint64_t a, uint64_t b )
{
  const uint64_t a0 = a & 0xFFFFFFFFUL, a1 = a >> 32, b0 = b & 0xFFFFFFFFUL, b1 = b >> 32;
  const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const uint64_t mid = ( p00 >> 32 ) + ( p01 & 0xFFFFFFFFUL ) + ( p10 & 0xFFFFFFFFUL );
  const uint64_t lo  = ( mid << 32 ) | ( p00 & 0xFFFFFFFFUL );
  const uint64_t hi  = p11 + ( p01 >> 32 ) + ( p10 >> 32 ) + ( mid >> 32 );
  return u64vec2( hi, lo );
}

uint64_t f64Mul( uint64_t a, uint64_t b )
{
  const bool signZ = f64Sign( a ) != f64Sign( b );
  int      expA = f64Exp( a ), expB = f64Exp( b );
  uint64_t sigA = f64Frac( a ), sigB = f64Frac( b );
  if( expA == 0x7FF || expB == 0x7FF )
  {
    if( f64IsNaN( a ) || f64IsNaN( b ) )
    {
      return f64PropagateNaN( a, b );
    }
    const bool otherZero = expA == 0x7FF ? ( expB == 0 && sigB == 0UL ) : ( expA == 0 && sigA == 0UL );
    return otherZero ? F64_DEFAULT_NAN : f64Pack( signZ, 0x7FF, 0UL );
  }
  if( expA == 0 )
  {
    if( sigA == 0UL )
    {
      return f64Pack( signZ, 0, 0UL );
    }
    sigA = f64NormSubnormal( sigA, expA );
  }
  if( expB == 0 )
  {
    if( sigB == 0UL )
    {
      return f64Pack( signZ, 0, 0UL );
    }
    sigB = f64NormSubnormal( sigB, expB );
  }
  int expZ = expA + expB - 0x3FF;
  sigA     = ( sigA | F64_HIDDEN ) << 10;
  sigB     = ( sigB | F64_HIDDEN ) << 11;
  const u64vec2 p = mul64To128( sigA, sigB );
  uint64_t sigZ = p.x | ( p.y != 0UL ? 1UL : 0UL );
  if( sigZ < 0x4000000000000000UL )
  {
    expZ--;
    sigZ <<= 1;
  }
  return f64RoundPack( signZ, expZ, sigZ );
}

uint64_t f64Div( uint64_t a, uint64_t b )
{
  const bool signZ = f64Sign( a ) != f64Sign( b );
  int      expA = f64Exp( a ), expB = f64Exp( b );
  uint64_t sigA = f64Frac( a ), sigB = f64Frac( b );
  if( f64IsNaN( a ) || f64IsNaN( b ) )
  {
    return f64PropagateNaN( a, b );
  }
  if( expA == 0x7FF )
  {
    return expB == 0x7FF ? F64_DEFAULT_NAN : f64Pack( signZ, 0x7FF, 0UL );
  }
  if( expB == 0x7FF )
  {
    return f64Pack( signZ, 0, 0UL );
  }
  if( expB == 0 )
  {
    if( sigB == 0UL )
    {
      return ( expA == 0 && sigA == 0UL ) ? F64_DEFAULT_NAN : f64Pack( signZ, 0x7FF, 0UL );
    }
    sigB = f64NormSubnormal( sigB, expB );
  }
  if( expA == 0 )
  {
    if( sigA == 0UL )
    {
      return f64Pack( signZ, 0, 0UL );
    }
    sigA = f64NormSubnormal( sigA, expA );
  }
  int expZ = expA - expB + 0x3FE;
  sigA |= F64_HIDDEN;
  sigB |= F64_HIDDEN;
  if( sigA < sigB )
  {
    expZ--;
    sigA <<= 11;
  }
  else
  {
    sigA <<= 10;
  }
  sigB <<= 11;
  // Restoring division: q = floor(sigA * 2^63 / sigB) has 63 bits; the remainder only sets the sticky bit.
  uint64_t r = sigA, q = 0UL;
  for( int i = 0; i < 63; i++ )
  {
    const bool carry = ( r >> 63 ) != 0UL;
    r <<= 1;
    q <<= 1;
    if( carry || r >= sigB )
    {
      r -= sigB;
      q |= 1UL;
    }
  }
  return f64RoundPack( signZ, expZ, q | ( r != 0UL ? 1UL : 0UL ) );
}

uint64_t f64Sqrt( uint64_t a )
{
  int      expA = f64Exp( a );
  uint64_t sigA = f64Frac( a );
  if( f64IsNaN( a ) )
  {
    return a | F64_QUIET;
  }
  if( f64Sign( a ) )
  {
    return ( a & F64_ABS ) == 0UL ? a : F64_DEFAULT_NAN;
  }
  if( expA == 0x7FF )
  {
    return a;
  }
  if( expA == 0 )
  {
    if( sigA == 0UL )
    {
      return a;
    }
    sigA = f64NormSubnormal( sigA, expA );
  }
  const int expZ = ( ( expA - 0x3FF ) >> 1 ) + 0x3FE;
  sigA |= F64_HIDDEN;
  // The radicand is sigA * 2^72 for an even unbiased exponent and sigA * 2^73 for an odd one, so its root lies in
  // [2^62, 2^63). As a 128-bit value its low word is zero.
  const bool oddExp = ( ( expA - 0x3FF ) & 1 ) != 0;
  uint64_t remHi = sigA << ( oddExp ? 9 : 8 ), remLo = 0UL;
  uint64_t root  = 0UL;
  for( int bit = 62; bit >= 0; bit-- )
  {
    // trial = (root << (bit + 1)) + (1 << (2 * bit)), as a 128-bit value
    uint64_t tHi = bit + 1 >= 64 ? root << ( bit + 1 - 64 ) : ( bit + 1 == 0 ? 0UL : root >> ( 63 - bit ) );
    uint64_t tLo = bit + 1 >= 64 ? 0UL : root << ( bit + 1 );
    const int sq = 2 * bit;
    if( sq >= 64 )
    {
      tHi += 1UL << ( sq - 64 );
    }
    else
    {
      const uint64_t add = 1UL << sq;
      tLo += add;
      if( tLo < add )
      {
        tHi++;
      }
    }
    if( remHi > tHi || ( remHi == tHi && remLo >= tLo ) )
    {
      const uint64_t borrow = remLo < tLo ? 1UL : 0UL;
      remLo -= tLo;
      remHi -= tHi + borrow;
      root |= 1UL << bit;
    }
  }
  return f64RoundPack( false, expZ, root | ( ( remHi | remLo ) != 0UL ? 1UL : 0UL ) );
}

uint64_t f64FromI64( int64_t a )
{
  const bool     sign = a < 0L;
  const uint64_t absA = sign ? uint64_t( -a ) : uint64_t( a );   // -INT64_MIN wraps to 2^63, which is right
  if( absA == 0UL )
  {
    return 0UL;
  }
  if( ( absA >> 63 ) != 0UL )
  {
    return 0xC3E0000000000000UL;   // INT64_MIN is -2^63 exactly
  }
  return f64NormRoundPack( sign, 0x43C, absA );
}

uint64_t f64FromI32( int a ) { return f64FromI64( int64_t( a ) ); }

uint64_t f64FromU64( uint64_t a )
{
  if( a == 0UL )
  {
    return 0UL;
  }
  if( ( a >> 63 ) != 0UL )
  {
    return f64RoundPack( false, 0x43D, shiftRightJam64( a, 1 ) );
  }
  return f64NormRoundPack( false, 0x43C, a );
}

// Conversion with truncation toward zero, as a C cast. Out-of-range values and NaN give INT64_MIN like cvttsd2si.
int64_t f64ToI64( uint64_t a )
{
  const int exp = f64Exp( a );
  if( exp < 0x3FF )
  {
    return 0L;
  }
  if( exp >= 0x3FF + 63 )
  {
    return int64_t( 0x8000000000000000UL );
  }
  const uint64_t sig   = f64Frac( a ) | F64_HIDDEN;
  const int      shift = exp - 1075;
  const uint64_t mag   = shift >= 0 ? sig << shift : sig >> ( -shift );
  return f64Sign( a ) ? -int64_t( mag ) : int64_t( mag );
}

// Out-of-range values and NaN give INT32_MIN like cvttsd2si.
int f64ToI32( uint64_t a )
{
  const int exp = f64Exp( a );
  if( exp < 0x3FF )
  {
    return 0;
  }
  if( exp >= 0x3FF + 31 )
  {
    // Only -2^31 itself is in range among these values.
    return int( 0x80000000u );
  }
  const uint64_t sig = f64Frac( a ) | F64_HIDDEN;
  const int      mag = int( sig >> ( 1075 - exp ) );
  return f64Sign( a ) ? -mag : mag;
}

bool f64Eq( uint64_t a, uint64_t b )
{
  if( f64IsNaN( a ) || f64IsNaN( b ) )
  {
    return false;
  }
  return a == b || ( ( a | b ) & F64_ABS ) == 0UL;
}

bool f64Lt( uint64_t a, uint64_t b )
{
  if( f64IsNaN( a ) || f64IsNaN( b ) )
  {
    return false;
  }
  const bool signA = f64Sign( a ), signB = f64Sign( b );
  if( signA != signB )
  {
    return signA && ( ( a | b ) & F64_ABS ) != 0UL;
  }
  return a != b && ( signA != ( a < b ) );
}

bool f64Le( uint64_t a, uint64_t b )
{
  if( f64IsNaN( a ) || f64IsNaN( b ) )
  {
    return false;
  }
  const bool signA = f64Sign( a ), signB = f64Sign( b );
  if( signA != signB )
  {
    return signA || ( ( a | b ) & F64_ABS ) == 0UL;
  }
  return a == b || ( signA != ( a < b ) );
}

uint64_t f64Neg( uint64_t a ) { return a ^ F64_SIGN; }
uint64_t f64Abs( uint64_t a ) { return a & F64_ABS; }
