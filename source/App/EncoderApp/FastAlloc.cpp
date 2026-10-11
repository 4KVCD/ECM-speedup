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
    blocks stay 16-byte aligned and delete knows where a block came from. A spinlock makes it thread-safe (with
    FrameThreads the pictures' threads share it). Over-aligned new/delete keep their defaults.

    Canary build (-DFASTALLOC_CANARY=1, for testing only): every block gets 16 guard bytes right after the requested
    size and a magic word in its header; fresh blocks are filled with 0xCB and freed blocks poisoned with 0xDD. Delete
    checks the magic (double free, foreign pointer, underrun) and the guard bytes (overrun); reuse of a freed block
    checks the poison (write after free). Violations abort with a message. A canary run that stays bit-exact also
    shows that no output depends on uninitialized or freed heap memory.
*/

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>

#ifndef FASTALLOC_CANARY
#define FASTALLOC_CANARY 0
#endif

namespace
{

constexpr size_t HEADER        = 16;
constexpr int    NUM_CLASSES   = 16;
constexpr size_t CLASS_SIZE[NUM_CLASSES] = { 16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024 };
constexpr size_t MAX_SMALL     = 1024;
constexpr size_t CHUNK_BYTES   = 1 << 20;   // carved into blocks of one class at a time
constexpr uint32_t LARGE_CLASS = 0xFFFFFFFFu;

#if FASTALLOC_CANARY
constexpr size_t   GUARD       = 16;
constexpr uint32_t MAGIC_ALIVE = 0xA110CA7Eu;
constexpr uint32_t MAGIC_FREED = 0xDEADF4EEu;
constexpr uint8_t  GUARD_BYTE  = 0xFD;
constexpr uint8_t  FRESH_BYTE  = 0xCB;
constexpr uint8_t  FREED_BYTE  = 0xDD;

struct Header
{
  uint32_t cls;
  uint32_t magic;
  uint64_t size;   // requested size
};
static_assert( sizeof( Header ) == HEADER, "header size" );

[[noreturn]] void canaryFail( const char* what, const void* ptr )
{
  fprintf( stderr, "FastAlloc canary: %s (block %p)\n", what, ptr );
  fflush( stderr );
  std::abort();
}

bool allBytes( const char* p, size_t n, uint8_t v )
{
  for( size_t i = 0; i < n; i++ )
  {
    if( (uint8_t) p[i] != v )
    {
      return false;
    }
  }
  return true;
}

// header + fill of a block handed out
void* canaryArm( char* p, uint32_t cls, size_t size )
{
  Header* h = (Header*) p;
  h->cls    = cls;
  h->magic  = MAGIC_ALIVE;
  h->size   = size;
  memset( p + HEADER, FRESH_BYTE, size );
  memset( p + HEADER + size, GUARD_BYTE, GUARD );
  return p + HEADER;
}

// checks of a block being freed; poisons it
void canaryDisarm( char* p )
{
  Header* h = (Header*) p;
  if( h->magic != MAGIC_ALIVE )
  {
    canaryFail( h->magic == MAGIC_FREED ? "double free" : "delete of a foreign pointer or header overwritten", p + HEADER );
  }
  if( !allBytes( p + HEADER + h->size, GUARD, GUARD_BYTE ) )
  {
    canaryFail( "write past the end of the block", p + HEADER );
  }
  h->magic = MAGIC_FREED;
  memset( p + HEADER, FREED_BYTE, h->size + GUARD );
}
#else
constexpr size_t GUARD = 0;
#endif

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
  const size_t inner = size + GUARD;   // bytes the block must hold
  if( inner > MAX_SMALL )
  {
    char* p = (char*) std::malloc( inner + HEADER );
    if( !p )
    {
      return nullptr;
    }
#if FASTALLOC_CANARY
    return canaryArm( p, LARGE_CLASS, size );
#else
    *(uint32_t*) p = LARGE_CLASS;
    return p + HEADER;
#endif
  }
  Allocator&  a   = allocator();
  const int   c   = a.classOf[( inner + 15 ) >> 4];
  LockGuard   guard( a.lock );
  FreeBlock*  blk = a.freeList[c];
  if( blk )
  {
    a.freeList[c] = blk->next;
#if FASTALLOC_CANARY
    char*         p     = (char*) blk - HEADER;
    const Header* h     = (const Header*) p;
    const size_t  freed = h->size + GUARD;   // poisoned when freed; the first bytes held the free-list link
    if( h->magic != MAGIC_FREED || h->cls != (uint32_t) c
        || !allBytes( (const char*) blk + sizeof( FreeBlock ), freed - sizeof( FreeBlock ), FREED_BYTE ) )
    {
      canaryFail( "write after free (or corrupted free list)", blk );
    }
    return canaryArm( p, (uint32_t) c, size );
#else
    return blk;
#endif
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
#if FASTALLOC_CANARY
  return canaryArm( p, (uint32_t) c, size );
#else
  *(uint32_t*) p = (uint32_t) c;
  return p + HEADER;
#endif
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
#if FASTALLOC_CANARY
    canaryDisarm( p );
#endif
    std::free( p );
    return;
  }
  Allocator& a = allocator();
  LockGuard  guard( a.lock );
#if FASTALLOC_CANARY
  if( c >= (uint32_t) NUM_CLASSES )
  {
    canaryFail( "delete of a foreign pointer or header overwritten", ptr );
  }
  canaryDisarm( p );
#endif
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
