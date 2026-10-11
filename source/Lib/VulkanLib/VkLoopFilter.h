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

/// Runs the deblocking filter of a picture on the GPU. The CPU decoder hands over the reconstruction and the edge
/// parameters that LoopFilter records while it walks the coding units.
class VkLoopFilter
{
public:
  void init( Core* core ) { m_core = core; }
  void destroy();

  /// Prepares the GPU buffers for the picture of cs and uploads its reconstruction.
  void beginPicture( const CodingStructure& cs, bool filterOnCpu );
  /// The edge maps for LoopFilter::setEdgeMaps, cleared for the current picture.
  DeblockEdgeMaps& edgeMaps() { return m_maps; }
  /// Deblocks the uploaded picture with the recorded edge parameters.
  void deblock( const CodingStructure& cs );
  GpuPicture& picture() { return m_pic; }
  double lastGpuMs() const { return m_lastGpuMs; }
  /// Describes the edge parameters and samples around a luma position, for a difference found by the check mode.
  std::string describeLuma( int x, int y, const CPelUnitBuf& cpu ) const;

private:
  std::vector<Pel> m_input;   ///< the picture before deblocking, kept in check mode for describeLuma
  Core*           m_core = nullptr;
  GpuPicture      m_pic;
  Buffer          m_edgeBuf;   ///< luma maps (vertical, horizontal), then chroma maps (vertical, horizontal)
  DeblockEdgeMaps m_maps;
  size_t          m_lumaMapBytes   = 0;
  size_t          m_chromaMapBytes = 0;
  double          m_lastGpuMs      = 0;
};

}   // namespace vkd
