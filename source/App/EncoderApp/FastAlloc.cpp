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

/** \file     FastAlloc.cpp
    \brief    perf: small-object allocator behind the global operator new / delete of the encoder

    The RD search allocates and frees many small, short-lived std::vector buffers; the Windows heap spends ~1.3 %
    of the encoding time on them. Blocks of up to 1 KB come from per-size-class free lists (memory is kept for reuse,
    never returned), larger ones from malloc. Every block starts with a 16-byte header holding its size class, so
    blocks stay 16-byte aligned and delete knows where a block came from. A spinlock makes it thread-safe (the
    encoder itself is single-threaded, so the lock is uncontended). Over-aligned new/delete keep their defaults.
*/

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <new>

namespace
{

constexpr size_t HEADER        = 16;
constexpr int    NUM_CLASSES   = 16;
constexpr size_t CLASS_SIZE[NUM_CLASSES] = { 16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024 };
constexpr size_t MAX_SMALL     = 1024;
constexpr size_t CHUNK_BYTES   = 1 << 20;   // carved into blocks of one class at a time
constexpr uint32_t LARGE_CLASS = 0xFFFFFFFFu;

struct FreeBlock
{
  FreeBlock* next;
};

struct Allocator
{
  std::atomic_flag lock      = ATOMIC_FLAG_INIT;
  FreeBlock*       freeList[NUM_CLASSES] = {};
  char*            chunkPos  = nullptr;
  size_t           chunkLeft = 0;
  uint8_t          classOf[MAX_SMALL / 16 + 1];   // (size + 15) / 16 -> class

  Allocator()
  {
    int c = 0;
    for( size_t i = 0; i <= MAX_SMALL / 16; i++ )
    {
      while( CLASS_SIZE[c] < i * 16 )
      {
        c++;
      }
      classOf[i] = (uint8_t) c;
    }
  }
};

Allocator& allocator()
{
  static Allocator a;   // constructed on first use (also during static initialization of other units)
  return a;
}

struct LockGuard
{
  std::atomic_flag& f;
  explicit LockGuard( std::atomic_flag& flag ) : f( flag )
  {
    while( f.test_and_set( std::memory_order_acquire ) )
    {
    }
  }
  ~LockGuard() { f.clear( std::memory_order_release ); }
};

void* allocate( size_t size )
{
  if( size == 0 )
  {
    size = 1;
  }
  if( size > MAX_SMALL )
  {
    char* p = (char*) std::malloc( size + HEADER );
    if( !p )
    {
      return nullptr;
    }
    *(uint32_t*) p = LARGE_CLASS;
    return p + HEADER;
  }
  Allocator&  a   = allocator();
  const int   c   = a.classOf[( size + 15 ) >> 4];
  LockGuard   guard( a.lock );
  FreeBlock*  blk = a.freeList[c];
  if( blk )
  {
    a.freeList[c] = blk->next;
    return blk;
  }
  const size_t need = CLASS_SIZE[c] + HEADER;
  if( a.chunkLeft < need )
  {
    char* chunk = (char*) std::malloc( CHUNK_BYTES );
    if( !chunk )
    {
      return nullptr;
    }
    a.chunkPos  = chunk;
    a.chunkLeft = CHUNK_BYTES;
  }
  char* p = a.chunkPos;
  a.chunkPos += need;
  a.chunkLeft -= need;
  *(uint32_t*) p = (uint32_t) c;
  return p + HEADER;
}

void release( void* ptr )
{
  if( !ptr )
  {
    return;
  }
  char*          p = (char*) ptr - HEADER;
  const uint32_t c = *(const uint32_t*) p;
  if( c == LARGE_CLASS )
  {
    std::free( p );
    return;
  }
  Allocator& a = allocator();
  LockGuard  guard( a.lock );
  FreeBlock* blk = (FreeBlock*) ptr;
  blk->next      = a.freeList[c];
  a.freeList[c]  = blk;
}

void* allocateOrThrow( size_t size )
{
  void* p = allocate( size );
  if( !p )
  {
    throw std::bad_alloc();
  }
  return p;
}

}   // namespace

void* operator new( size_t size ) { return allocateOrThrow( size ); }
void* operator new[]( size_t size ) { return allocateOrThrow( size ); }
void* operator new( size_t size, const std::nothrow_t& ) noexcept { return allocate( size ); }
void* operator new[]( size_t size, const std::nothrow_t& ) noexcept { return allocate( size ); }
void  operator delete( void* ptr ) noexcept { release( ptr ); }
void  operator delete[]( void* ptr ) noexcept { release( ptr ); }
void  operator delete( void* ptr, size_t ) noexcept { release( ptr ); }
void  operator delete[]( void* ptr, size_t ) noexcept { release( ptr ); }
void  operator delete( void* ptr, const std::nothrow_t& ) noexcept { release( ptr ); }
void  operator delete[]( void* ptr, const std::nothrow_t& ) noexcept { release( ptr ); }
