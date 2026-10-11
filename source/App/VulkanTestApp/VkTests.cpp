/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     VkTests.cpp
    \brief    GPU kernel unit tests
*/

#include "VkTests.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace vkt
{

static std::string testSelf( vkd::Core& core )
{
  const int   n   = 100000;
  vkd::Buffer src = core.createBuffer( n * 4, vkd::Mem::HostWrite );
  vkd::Buffer dst = core.createBuffer( n * 4, vkd::Mem::HostCached );
  for( int i = 0; i < n; i++ )
  {
    src.ptr<int>()[i] = i * 7 - 50000;
  }
  struct
  {
    uint64_t src, dst;
    int      count;
  } push = { src.address, dst.address, n };
  core.begin();
  core.dispatch( "selftest", ( n + 63 ) / 64, 1, 1, &push, sizeof( push ) );
  core.barrier();
  core.submit();
  std::string err;
  for( int i = 0; i < n && err.empty(); i++ )
  {
    if( dst.ptr<int>()[i] != 3 * src.ptr<int>()[i] + 1 )
    {
      err = "element " + std::to_string( i ) + " differs";
    }
  }
  core.destroyBuffer( src );
  core.destroyBuffer( dst );
  return err;
}

// ---- software double arithmetic (sf64.glsl) ----------------------------------------------------------------------

static uint64_t bitsOf( double d )
{
  uint64_t u;
  memcpy( &u, &d, 8 );
  return u;
}

static double dbl( uint64_t u )
{
  double d;
  memcpy( &d, &u, 8 );
  return d;
}

// The CPU result of the operation codes of test_sf64.comp.
static uint64_t sf64Reference( int op, uint64_t a, uint64_t b )
{
  switch( op )
  {
  case 0: return bitsOf( dbl( a ) + dbl( b ) );
  case 1: return bitsOf( dbl( a ) - dbl( b ) );
  case 2: return bitsOf( dbl( a ) * dbl( b ) );
  case 3: return bitsOf( dbl( a ) / dbl( b ) );
  case 4: return bitsOf( sqrt( dbl( a ) ) );
  case 5: return bitsOf( (double) (int64_t) a );
  case 6: return (uint64_t) (int64_t) dbl( a );
  case 7: return (uint64_t) (int64_t) (int) dbl( a );
  case 8: return dbl( a ) < dbl( b );
  case 9: return dbl( a ) <= dbl( b );
  case 10: return dbl( a ) == dbl( b );
  case 11: return bitsOf( (double) a );
  case 12: return bitsOf( (double) (int) a );
  }
  return 0;
}

static std::string testSf64( vkd::Core& core )
{
  std::mt19937_64 rng( 20261010 );
  std::vector<int>      ops;
  std::vector<uint64_t> args;
  auto add = [&]( int op, uint64_t a, uint64_t b ) { ops.push_back( op ); args.push_back( a ); args.push_back( b ); };
  // A double of random sign, significand and exponent in [e0, e1] (biased).
  auto randDouble = [&]( int e0, int e1 ) {
    const uint64_t e = e0 + rng() % ( e1 - e0 + 1 );
    return ( rng() & 0x8000000000000000ull ) | ( e << 52 ) | ( rng() & 0x000FFFFFFFFFFFFFull );
  };
  const uint64_t specials[] = { 0, 0x8000000000000000ull, 0x3FF0000000000000ull, 0xBFF0000000000000ull,
                                0x7FF0000000000000ull, 0xFFF0000000000000ull, 0x7FF8000000000000ull, 0x7FF0000000000001ull,
                                0x0000000000000001ull, 0x000FFFFFFFFFFFFFull, 0x0010000000000000ull, 0x7FEFFFFFFFFFFFFFull,
                                0x43E0000000000000ull, 0xC3E0000000000000ull, 0x41E0000000000000ull, 0xC1E0000000000000ull,
                                0x3FE0000000000000ull, 0x4330000000000000ull, 0x4340000000000000ull };
  for( int op = 0; op <= 12; op++ )
  {
    for( uint64_t a: specials )
    {
      for( uint64_t b: specials )
      {
        add( op, a, b );
      }
    }
  }
  for( int i = 0; i < 300000; i++ )
  {
    for( int op = 0; op <= 3; op++ )
    {
      // Full-range values, values of nearby magnitude (cancellation, ties) and subnormal neighbourhoods.
      add( op, rng(), rng() );
      const uint64_t a = randDouble( 900, 1150 );
      add( op, a, randDouble( 900, 1150 ) );
      const int e = 1 + rng() % 2000;
      add( op, randDouble( e, e + 1 ), randDouble( e > 60 ? e - 60 : 1, e + 1 ) );
      add( op, randDouble( 0, 60 ), randDouble( 0, 60 ) );
      add( op, randDouble( 1000, 1100 ), rng() % 2 ? ( a ^ ( rng() & 0x3FF ) ) : ( a ^ 0x8000000000000000ull ^ ( rng() & 0xF ) ) );
    }
    // Integers as the decoder uses them: sums of products of samples, ratios of costs.
    const int64_t x = (int64_t) ( rng() >> ( rng() % 64 ) ) * ( rng() % 2 ? 1 : -1 );
    const int64_t y = (int64_t) ( rng() >> ( 1 + rng() % 63 ) ) + 1;
    add( 0, bitsOf( (double) x ), bitsOf( (double) y ) );
    add( 2, bitsOf( (double) ( x >> 20 ) ), bitsOf( (double) ( y >> 30 ) ) );
    add( 3, bitsOf( (double) x ), bitsOf( (double) y ) );
    add( 4, rng() & 0x7FFFFFFFFFFFFFFFull, 0 );
    add( 4, bitsOf( (double) ( y >> 10 ) * (double) ( y >> 10 ) ), 0 );
    add( 4, randDouble( 0, 60 ) & 0x7FFFFFFFFFFFFFFFull, 0 );
    add( 5, (uint64_t) x, 0 );
    add( 11, rng() >> ( rng() % 64 ), 0 );
    add( 12, (uint64_t) (int64_t) (int) rng(), 0 );
    add( 6, randDouble( 1000, 1085 ), 0 );
    add( 6, rng(), 0 );
    add( 7, randDouble( 1000, 1054 ), 0 );
    add( 7, randDouble( 1020, 1060 ), 0 );
    for( int op = 8; op <= 10; op++ )
    {
      const uint64_t a = randDouble( 1000, 1050 );
      add( op, a, rng() % 4 ? randDouble( 1000, 1050 ) : ( rng() % 2 ? a : a ^ 0x8000000000000000ull ) );
      add( op, rng(), rng() );
    }
  }
  const int   n    = (int) ops.size();
  vkd::Buffer bOps = core.createBuffer( n * 4, vkd::Mem::HostWrite );
  vkd::Buffer bArg = core.createBuffer( n * 16, vkd::Mem::HostWrite );
  vkd::Buffer bRes = core.createBuffer( n * 8, vkd::Mem::HostCached );
  memcpy( bOps.mapped, ops.data(), n * 4 );
  memcpy( bArg.mapped, args.data(), n * 16 );
  struct
  {
    uint64_t ops, args, res;
    int      count;
  } push = { bOps.address, bArg.address, bRes.address, n };
  core.begin();
  core.dispatch( "test_sf64", ( n + 63 ) / 64, 1, 1, &push, sizeof( push ) );
  core.barrier();
  core.submit();
  std::string err;
  int         numBad = 0;
  for( int i = 0; i < n; i++ )
  {
    const uint64_t a = args[2 * i], b = args[2 * i + 1];
    const uint64_t want = sf64Reference( ops[i], a, b );
    const uint64_t got  = bRes.ptr<uint64_t>()[i];
    if( want != got )
    {
      if( numBad++ < 8 )
      {
        char buf[200];
        snprintf( buf, sizeof( buf ), "%sop %d a %016llx b %016llx: cpu %016llx gpu %016llx", err.empty() ? "" : "; ", ops[i],
                  (unsigned long long) a, (unsigned long long) b, (unsigned long long) want, (unsigned long long) got );
        err += buf;
      }
    }
  }
  if( numBad )
  {
    err = std::to_string( numBad ) + " of " + std::to_string( n ) + " differ: " + err;
  }
  core.destroyBuffer( bOps );
  core.destroyBuffer( bArg );
  core.destroyBuffer( bRes );
  return err;
}

static const Test s_tests[] = {
  { "selftest", testSelf },
  { "sf64", testSf64 },
  { nullptr, nullptr },
};

const Test* tests()
{
  return s_tests;
}

}   // namespace vkt
