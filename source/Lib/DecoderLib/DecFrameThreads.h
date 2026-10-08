/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2023, ITU/ISO/IEC
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

/** \file     DecFrameThreads.h
    \brief    frame-level threading of the decoder

    The main thread parses the high-level syntax, manages the picture buffers and the output, and walks the
    decode-order state (DecPicState.h). At the end of a picture it hands the picture's slices to a worker context
    (another DecLib with its own prediction, transform, CABAC and loop filter objects), which waits for the
    reference pictures and the decode-order state it reads, decodes the CTUs, runs the loop filters and the
    picture padding. Pictures with more than one slice and pictures using tools not decoded in threads are decoded
    by the main thread as before.
*/

#ifndef __DECFRAMETHREADS__
#define __DECFRAMETHREADS__

#include "DecPicState.h"
#include "CommonLib/BitStream.h"
#include "CommonLib/Slice.h"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class DecLib;
struct Picture;

//! \ingroup DecoderLib
//! \{

/// the decoding of one picture
struct DecPicJob
{
  Picture*                                     pic = nullptr;
  std::vector<Slice*>                          slices;
  std::vector<std::unique_ptr<InputBitstream>> bitstreams;            ///< slice data (copies of the NAL units)
  std::vector<DecSliceState>                   sliceStates;
  std::vector<std::shared_ptr<DecPicJob>>      refJobs;               ///< reference pictures not decoded at dispatch
  std::vector<const Picture*>                  usedPics;              ///< reference pictures, kept until the report
  DecLfCtrlState                               lf;
  std::unique_ptr<PicHeader>                   picHeader;             ///< copy of the decoder's picture header
  std::unique_ptr<PPS>                         pps;                   ///< copy of the PPS (shares its PreCalcValues)
  std::vector<std::unique_ptr<APS>>            apss;                  ///< copies of the APS the picture uses
  bool                                         sync       = false;    ///< decoded by the main thread
  bool                                         referenced = false;    ///< at the end of the picture (POC line)
  DecSlot                                      done;
  std::string                                  error;                 ///< what a worker threw

  ~DecPicJob()
  {
    if( pps )
    {
      pps->pcv = nullptr;   // owned by the PPS of the parameter set manager
    }
  }
};

/// worker threads with their decoding contexts
class DecFrameThreads
{
public:
  DecFrameThreads( DecLib& main, int numThreads, bool serial );
  ~DecFrameThreads();

  void enqueue( const std::shared_ptr<DecPicJob>& job );
  int  numThreads() const { return (int) m_threads.size(); }
  bool isSerial()   const { return m_serial; }

private:
  void workerLoop( DecLib* ctx );

  std::vector<std::unique_ptr<DecLib>>   m_ctx;
  std::vector<std::thread>               m_threads;
  std::mutex                             m_mutex;
  std::condition_variable                m_cond;
  std::deque<std::shared_ptr<DecPicJob>> m_queue;
  bool                                   m_stop = false;
  bool                                   m_serial;    ///< one picture at a time (tests the hand-over of state)
};

//! \}

#endif
