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

/** \file     VkLoopFilter.h
    \brief    In-loop filters of the GPU decoder
*/

#pragma once

#include "VkCore.h"

#include "CommonLib/Unit.h"
#include "CommonLib/Buffer.h"
#include "CommonLib/LoopFilter.h"

#include <string>
#include <vector>

class CodingStructure;
class SampleAdaptiveOffset;

namespace vkd
{

/// A picture in GPU memory: the planes of one picture with a margin around each, in a single buffer.
class GpuPicture
{
public:
  struct Plane
  {
    int origin = 0;   ///< index of sample (0, 0)
    int stride = 0;
    int width  = 0;
    int height = 0;
  };

  void create( Core& core, ChromaFormat format, int width, int height, int margin );
  void destroy( Core& core ) { core.destroyBuffer( m_buf ); }
  bool matches( ChromaFormat format, int width, int height ) const
  {
    return m_buf.buffer && format == m_format && width == m_plane[0].width && height == m_plane[0].height;
  }

  void upload( const CPelUnitBuf& src );
  void download( PelUnitBuf& dst ) const;
  /// Compares with ref; on a difference, describes the first one in diff, sets its component and position and
  /// returns false.
  bool compare( const CPelUnitBuf& ref, std::string& diff, int& comp, int& x, int& y ) const;

  const Buffer& buffer() const { return m_buf; }
  const Plane&  plane( int c ) const { return m_plane[c]; }
  int           numComp() const { return m_numComp; }
  ChromaFormat  format() const { return m_format; }

private:
  Buffer       m_buf;
  Plane        m_plane[MAX_NUM_COMPONENT];
  int          m_numComp = 0;
  ChromaFormat m_format  = CHROMA_420;
};

/// Runs in-loop filter stages of a picture on the GPU: deblocking, and SAO with the bilateral filter and CCSAO.
/// The CPU decoder runs the other stages and hands over the parameters each GPU stage needs.
///
/// In check mode the CPU also runs every GPU stage. Each GPU stage then starts from the CPU's input and its result
/// must equal the CPU's (checkAgainstCpu). Otherwise the picture stays on the GPU between GPU stages, and
/// syncToCpu copies it back before a stage that runs on the CPU.
class VkLoopFilter
{
public:
  void init( Core* core, bool check );
  void destroy();
  bool checking() const { return m_check; }

  /// Starts a picture whose reconstruction is on the CPU.
  void beginPicture( CodingStructure& cs );
  /// Makes the CPU picture current, before a stage that runs on the CPU. The GPU copy is stale afterwards.
  void syncToCpu( CodingStructure& cs );

  /// The edge maps for LoopFilter::setEdgeMaps, cleared, before LoopFilter::loopFilterPic.
  DeblockEdgeMaps& deblockPrepare( CodingStructure& cs );
  /// Deblocks the picture with the recorded edge parameters.
  void deblock( CodingStructure& cs );

  /// SAO, bilateral filter, CCSAO and their joint clipping, as SAOProcess, CCSAOProcess and jointClipSaoBifCcSao.
  void sao( CodingStructure& cs, SampleAdaptiveOffset& sao );

  /// Check mode: compares the GPU result of the last stage with the CPU picture; throws on a difference.
  void checkAgainstCpu( CodingStructure& cs, const char* stage );

  double gpuMs() const { return m_gpuMs; }

  /// Check mode: how much work each GPU stage did, so that a check run shows which paths it covered.
  struct Counts
  {
    uint64_t pictures       = 0;
    uint64_t deblockSegments = 0;   ///< luma and chroma edge segments with a boundary strength
    uint64_t saoCtus[5]     = {};  ///< CTU components with SAO on, by type: edge 0, 90, 135, 45 degrees, band
    uint64_t bifBlocks      = 0;
    uint64_t ccSaoBand[9]   = {};  ///< CTU components with CCSAO on, by band candidate position
    uint64_t ccSaoEdge[4]   = {};  ///< and by edge direction
  };
  const Counts& counts() const { return m_counts; }

private:
  void ensureOnGpu( CodingStructure& cs );
  void ensureBuffer( Buffer& buf, size_t size, Mem mem );
  std::string describeLuma( int x, int y, const CPelUnitBuf& cpu ) const;

  Core*            m_core  = nullptr;
  bool             m_check = false;
  bool             m_cpuCurrent = true;
  bool             m_gpuCurrent = false;
  int              m_lastStage  = 0;     ///< 1 after deblocking, for describeLuma
  GpuPicture       m_pic;
  GpuPicture       m_tmp;               ///< scratch picture, e.g. the input of SAO
  std::vector<Pel> m_input;             ///< check mode: the input of the last stage, for describeLuma

  Buffer           m_edgeBuf;           ///< deblocking: luma maps (vertical, horizontal), then chroma maps
  DeblockEdgeMaps  m_maps;
  size_t           m_lumaMapBytes   = 0;
  size_t           m_chromaMapBytes = 0;

  Buffer           m_saoParams;         ///< SaoCtu per CTU and component
  Buffer           m_bifBlocks;
  Buffer           m_bifTables;
  Buffer           m_ccSaoSets;
  Buffer           m_ctuBytes;          ///< CCSAO set per CTU and component, CTU availability, clip flags
  Buffer           m_ccSaoTables;
  std::vector<uint8_t> m_scratch;

  double           m_gpuMs = 0;
  Counts           m_counts;
};

}   // namespace vkd
