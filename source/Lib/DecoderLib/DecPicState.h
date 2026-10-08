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

/** \file     DecPicState.h
    \brief    decode-order state that one picture hands to later pictures

    Besides the reference pictures, ECM carries state from picture to picture in decoding order: the temporal CABAC
    contexts, the AMVP-SbTMVP area statistics, the CCSAO and chroma ALF / CCALF CTU control of earlier pictures.
    The stock decoder keeps it in global or per-decoder stores that are read and updated while a picture is decoded,
    which only works when pictures are decoded one after the other. Here the main thread walks the stores in decoding
    order when it sets up a slice, with the same insert / evict rules, and gives the slice the slots it reads and the
    slots it fills. A slot is filled by the picture that produces the value; a reader waits until it is ready. With one
    thread every slot is ready before it is read, so the result is that of the stock decoder.
*/

#ifndef __DECPICSTATE__
#define __DECPICSTATE__

#include "CommonLib/CommonDef.h"
#include "CommonLib/Contexts.h"

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

//! \ingroup DecoderLib
//! \{

/// a value that one picture (or slice) produces and later ones read
struct DecSlot
{
  std::atomic<bool> ready{ false };
};

#if JVET_Z0135_TEMP_CABAC_WIN_WEIGHT
/// temporal CABAC contexts stored at the last CTU of a picture
struct DecCtxSlot : DecSlot
{
  CtxStateBuf state;
  SliceType   sliceType = I_SLICE;
};
#endif

/// AMVP-SbTMVP area of a slice (g_picAmvpSbTmvpEnabledArea at its end)
struct DecAreaSlot : DecSlot
{
  uint32_t area = 0;
};

/// CTU control flags of a picture (CCSAO control; chroma ALF flags and alternatives; CCALF filter indices)
struct DecCtrlSlot : DecSlot
{
  std::vector<uint8_t> a;
  std::vector<uint8_t> b;
};

/// one mutex and condition variable for every wait of a decoder; waits are rare (a few per picture)
class DecWaitHub
{
public:
  void wait( const DecSlot& slot )
  {
    if( slot.ready.load( std::memory_order_acquire ) )
    {
      return;
    }
    CHECK( !m_threaded, "decode-order state read before it was produced" );
    std::unique_lock<std::mutex> lock( m_mutex );
    m_cond.wait( lock, [&] { return slot.ready.load( std::memory_order_acquire ); } );
  }
  void publish( DecSlot& slot )
  {
    {
      std::lock_guard<std::mutex> lock( m_mutex );
      slot.ready.store( true, std::memory_order_release );
    }
    m_cond.notify_all();
  }
  void setThreaded( bool b ) { m_threaded = b; }
  bool isThreaded() const    { return m_threaded; }
  std::mutex&              mutex() { return m_mutex; }
  std::condition_variable& cond()  { return m_cond; }

private:
  std::mutex              m_mutex;
  std::condition_variable m_cond;
  bool                    m_threaded = false;
};

/// the slots of one slice
struct DecSliceState
{
#if JVET_Z0135_TEMP_CABAC_WIN_WEIGHT
  std::shared_ptr<DecCtxSlot>  ctxLoad;    ///< stored contexts for this slice's key (null: none stored)
  std::shared_ptr<DecCtxSlot>  ctxStore;   ///< filled by this slice (null: it stores nothing)
#endif
#if JVET_AG0098_AMVP_WITH_SBTMVP
  bool                         sbTmvpEnabled = false;  ///< AMVP-SbTMVP enabled (as derived in DecSlice::decompressSlice)
  std::shared_ptr<DecAreaSlot> sbTmvpLoad;             ///< stored area for this slice's temporal layer (null: none)
  std::shared_ptr<DecAreaSlot> sbTmvpArea;             ///< filled by this slice when enabled
#endif
#if JVET_AL0142_CCSAO_REUSE_CTU
  std::shared_ptr<DecCtrlSlot> ccSaoWrite[MAX_NUM_COMPONENT];   ///< CCSAO CTU control this slice parses (entry it added)
  std::shared_ptr<DecCtrlSlot> ccSaoRead [MAX_NUM_COMPONENT];   ///< CCSAO CTU control this slice reuses
#endif
};

/// decode-order state as the next slice will see it (main thread)
struct DecOrderState
{
#if JVET_Z0135_TEMP_CABAC_WIN_WEIGHT
  std::map<std::pair<int, int>, std::shared_ptr<DecCtxSlot>> ctx[2];   ///< CtxStateStore::m_stateBuf
#endif
#if JVET_AG0098_AMVP_WITH_SBTMVP
  std::map<int, std::shared_ptr<DecAreaSlot>> sbTmvp;      ///< DecSlice::m_amvpSbTmvpArea
  std::shared_ptr<DecAreaSlot>                sbTmvpLast;  ///< g_picAmvpSbTmvpEnabledArea after the previous slice
#endif
#if JVET_AM0209_CHROMA_ALF_CCALF_REUSE_CTU
  std::shared_ptr<DecCtrlSlot> alfCtrl  [ALF_CTB_MAX_NUM_APS][MAX_NUM_COMPONENT];   ///< g_pictureControlInfo flags + filters
  std::shared_ptr<DecCtrlSlot> ccAlfCtrl[ALF_CTB_MAX_NUM_APS][MAX_NUM_COMPONENT];   ///< g_pictureControlInfo CCALF filters
#endif
};

//! \}

#endif
