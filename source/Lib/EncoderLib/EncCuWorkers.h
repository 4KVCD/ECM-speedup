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

/** \file     EncCuWorkers.h
    \brief    threads that run the parallel loops of one picture's mode search (CuThreads)
*/

#ifndef __ENCCUWORKERS__
#define __ENCCUWORKERS__

#include <atomic>
#include <condition_variable>
#include <mutex>

/// The worker threads of one search stack. The searching thread starts a job, the workers run it with their index
/// (1..numWorkers) and the searching thread waits for them. The loops are short (tens to hundreds of microseconds) and
/// waking a sleeping thread takes longer, so a worker spins for a while after a job before it sleeps, and the searching
/// thread wakes the workers ahead of a job. The threads never end: a thread that exits runs the destructors of its
/// thread_local objects, which crashes with MinGW's emulated TLS.
class EncCuWorkers
{
public:
  typedef void ( *Job )( void* arg, int worker );

  explicit EncCuWorkers( int numWorkers );
  int  getNumWorkers() const { return m_numWorkers; }
  /// Starts job( arg, w ) on the workers w = 1..numActive and returns at once.
  void start( Job job, void* arg, int numActive );
  /// Waits until every worker has returned from the job.
  void wait();
  /// Wakes the sleeping workers, so that they spin when the next job starts.
  void wake();

private:
  struct Shared;
  Shared* m_shared;   ///< shared with the threads, which outlive the object
  int     m_numWorkers;
};

#endif
