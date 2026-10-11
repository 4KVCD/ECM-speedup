/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2025, ITU/ISO/IEC
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

/** \file     EncCuWorkers.cpp
    \brief    threads that run the parallel loops of one picture's mode search (CuThreads)
*/

#include "EncCuWorkers.h"

#include <chrono>
#include <thread>
#include <immintrin.h>

struct EncCuWorkers::Shared
{
  std::atomic<unsigned> generation{ 0 };   ///< counts the jobs started
  std::atomic<int>      running{ 0 };      ///< workers still in the current job
  Job                   job = nullptr;
  void*                 arg = nullptr;
  int                   numActive = 0;     ///< workers 1..numActive run the current job
  std::mutex            mutex;
  std::condition_variable cv;
  std::atomic<int>      sleeping{ 0 };     ///< workers waiting on cv
  unsigned              wakeups = 0;       ///< counts the wake-ups without a job (under mutex)
};

namespace
{
// how long a worker spins for the next job before it sleeps
constexpr auto SPIN_TIME = std::chrono::milliseconds( 2 );
}   // namespace

EncCuWorkers::EncCuWorkers( int numWorkers )
  : m_shared( new Shared )
  , m_numWorkers( numWorkers )
{
  for( int w = 1; w <= numWorkers; w++ )
  {
    Shared* s = m_shared;
    std::thread( [s, w]()
    {
      unsigned seen = 0;
      while( true )
      {
        // spin, then sleep until a new job starts
        auto spinEnd = std::chrono::steady_clock::now() + SPIN_TIME;
        int  n       = 0;
        while( s->generation.load( std::memory_order_acquire ) == seen )
        {
          _mm_pause();
          if( ( ++n & 255 ) == 0 && std::chrono::steady_clock::now() > spinEnd )
          {
            std::unique_lock<std::mutex> lock( s->mutex );
            const unsigned wakeups = s->wakeups;
            s->sleeping.fetch_add( 1, std::memory_order_relaxed );
            s->cv.wait( lock, [&]() { return s->generation.load( std::memory_order_acquire ) != seen || s->wakeups != wakeups; } );
            s->sleeping.fetch_sub( 1, std::memory_order_relaxed );
            spinEnd = std::chrono::steady_clock::now() + SPIN_TIME;
          }
        }
        seen = s->generation.load( std::memory_order_acquire );
        if( w <= s->numActive )
        {
          s->job( s->arg, w );
        }
        s->running.fetch_sub( 1, std::memory_order_acq_rel );
      }
    } ).detach();
  }
}

void EncCuWorkers::start( Job job, void* arg, int numActive )
{
  Shared* s    = m_shared;
  s->job       = job;
  s->arg       = arg;
  s->numActive = numActive;
  s->running.store( m_numWorkers, std::memory_order_relaxed );
  {
    std::lock_guard<std::mutex> lock( s->mutex );
    s->generation.fetch_add( 1, std::memory_order_acq_rel );
    if( s->sleeping.load( std::memory_order_relaxed ) )
    {
      s->cv.notify_all();
    }
  }
}

void EncCuWorkers::wake()
{
  Shared* s = m_shared;
  if( s->sleeping.load( std::memory_order_relaxed ) )
  {
    std::lock_guard<std::mutex> lock( s->mutex );
    s->wakeups++;
    s->cv.notify_all();
  }
}

void EncCuWorkers::wait()
{
  while( m_shared->running.load( std::memory_order_acquire ) )
  {
    _mm_pause();
  }
}
