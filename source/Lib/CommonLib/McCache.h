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

/** \file     McCache.h
    \brief    Cache of motion compensation interpolation results (encoder speed-up, bit-exact)
*/

#ifndef __MCCACHE__
#define __MCCACHE__

#include "CommonDef.h"

#include <cstring>
#include <vector>

//! \ingroup CommonLib
//! \{

// The RD search interpolates the same reference block with the same sub-sample phase and filter many times
// (merge candidates in the SATD and RD passes, OBMC and template tools, the same CU reached through different
// partitionings); about half of the calls in random access repeat an earlier one within the picture. The result of
// the interpolation stage depends only on the reference samples, the block size, the phases and the filter
// configuration, and the reference pictures do not change while a picture is coded. Entries are therefore keyed on
// exactly these values (the reference position by address) and invalidated whenever the coded picture changes.
struct McCacheKey
{
  const Pel* src;
  int        srcStride;
  int16_t    width;
  int16_t    height;
  int16_t    xFrac;
  int16_t    yFrac;
  int8_t     compID;
  int8_t     filterIdx;
  int8_t     chFmt;
  uint8_t    flags;   // bilinear, alternative half-pel filter, bi filter, 4-tap TM filter, rounded result
  ClpRng     clpRng;

  bool operator==( const McCacheKey& o ) const
  {
    return src == o.src && srcStride == o.srcStride && width == o.width && height == o.height && xFrac == o.xFrac
           && yFrac == o.yFrac && compID == o.compID && filterIdx == o.filterIdx && chFmt == o.chFmt && flags == o.flags
           && clpRng.min == o.clpRng.min && clpRng.max == o.clpRng.max && clpRng.bd == o.clpRng.bd && clpRng.n == o.clpRng.n;
  }

  uint64_t hash() const
  {
    uint64_t h = (uint64_t) (uintptr_t) src;
    auto     mix = [&]( uint64_t v ) { h = ( h ^ v ) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; };
    mix( (uint64_t) (uint32_t) srcStride | ( (uint64_t) (uint16_t) width << 32 ) | ( (uint64_t) (uint16_t) height << 48 ) );
    mix( (uint64_t) (uint16_t) xFrac | ( (uint64_t) (uint16_t) yFrac << 16 ) | ( (uint64_t) (uint8_t) compID << 32 )
         | ( (uint64_t) (uint8_t) filterIdx << 40 ) | ( (uint64_t) (uint8_t) chFmt << 48 ) | ( (uint64_t) flags << 56 ) );
    mix( (uint64_t) (uint32_t) clpRng.max | ( (uint64_t) (uint32_t) clpRng.bd << 32 ) );
    return h;
  }
};

class McBlockCache
{
public:
  // reference pictures are fixed while one picture is coded; any change of the coded picture invalidates all entries
  void setPicture( const void* pic, int poc )
  {
    if( pic != m_pic || poc != m_poc )
    {
      m_pic = pic;
      m_poc = poc;
      m_epoch++;
    }
  }

  // on a hit, writes the cached block to dst and returns true
  bool lookup( const McCacheKey& key, Pel* dst, int dstStride )
  {
    if( m_slots.empty() )
    {
      m_slots.resize( size_t( 1 ) << LOG2_SLOTS );
      m_arena.resize( ARENA_SIZE );
    }
    const Slot& s = m_slots[key.hash() >> ( 64 - LOG2_SLOTS )];
    if( s.epoch != m_epoch || !( s.key == key ) || m_writePos > s.pos + ARENA_SIZE )
    {
      return false;
    }
    const Pel* src = &m_arena[s.pos % ARENA_SIZE];
    for( int y = 0; y < key.height; y++, src += key.width, dst += dstStride )
    {
      memcpy( dst, src, key.width * sizeof( Pel ) );
    }
    return true;
  }

  void store( const McCacheKey& key, const Pel* src, int srcStride )
  {
    const uint64_t size = uint64_t( key.width ) * key.height;
    if( size > ARENA_SIZE / 16 )
    {
      return;
    }
    if( m_writePos % ARENA_SIZE + size > ARENA_SIZE )
    {
      m_writePos += ARENA_SIZE - m_writePos % ARENA_SIZE;   // keep every block contiguous
    }
    Slot& s  = m_slots[key.hash() >> ( 64 - LOG2_SLOTS )];
    s.key    = key;
    s.epoch  = m_epoch;
    s.pos    = m_writePos;
    Pel* dst = &m_arena[m_writePos % ARENA_SIZE];
    for( int y = 0; y < key.height; y++, src += srcStride, dst += key.width )
    {
      memcpy( dst, src, key.width * sizeof( Pel ) );
    }
    m_writePos += size;
  }

  // only worth it when recomputing is clearly more expensive than a lookup: 2-D interpolation of larger blocks
  static bool worthCaching( int width, int height, int xFrac, int yFrac )
  {
    return xFrac != 0 && yFrac != 0 && width * height >= 64;
  }

private:
  // small enough to stay cache resident: a lookup that misses the CPU caches costs as much as a small interpolation
  static constexpr int      LOG2_SLOTS = 13;
  static constexpr uint64_t ARENA_SIZE = uint64_t( 1 ) << 21;   // samples (4 MB)

  struct Slot
  {
    McCacheKey key   = {};
    uint64_t   epoch = 0;
    uint64_t   pos   = 0;
  };

  std::vector<Slot> m_slots;
  std::vector<Pel>  m_arena;
  uint64_t          m_writePos = 0;
  uint64_t          m_epoch    = 1;
  const void*       m_pic      = nullptr;
  int               m_poc      = 0;
};

//! \}

#endif // __MCCACHE__
