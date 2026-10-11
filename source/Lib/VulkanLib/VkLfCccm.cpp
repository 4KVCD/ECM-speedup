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

/** \file     VkLfCccm.cpp
    \brief    The CCCM loop filter (LF-CCCM) of the GPU decoder
*/

#include "VkLoopFilter.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Slice.h"

#include <cstddef>
#include <cstring>

namespace vkd
{

// Layouts shared with lfcccm_down.comp, lfcccm_window.comp and lfcccm_apply.comp.
struct LfCccmDownPush
{
  uint64_t luma, dst;
  int32_t  lumaOrigin, lumaStride, dstOrigin, dstStride, width, height, type, lumaOffset;
};
struct GpuLfCccmWindow
{
  int32_t wx, wy, ww, wh, rx, ry, rw, rh, modelType, skip;
};
struct LfCccmWindowPush
{
  uint64_t down, pic, update, windows;
  int32_t  downOrigin, downStride, picOrigin[2], picStride[2], updOrigin[2], updStride;
  int32_t  bitDepthC, lumaOffset, clpMax[2], clpMin[2], impMode, badThr[3];
  uint64_t stats;
};
struct LfCccmApplyPush
{
  uint64_t pic, corr;
  int32_t  picOrigin[2], picStride[2], corrOrigin[2], corrStride, width, height, clpMin[2], clpMax[2], mode;
};
static_assert( offsetof( LfCccmWindowPush, badThr ) == 96 && offsetof( LfCccmWindowPush, stats ) == 112,
               "lfcccm_window.comp push constant layout" );
static_assert( offsetof( LfCccmApplyPush, mode ) == 68, "lfcccm_apply.comp push constant layout" );

static LfCccmApplyPush applyPush( const CodingStructure& cs, const GpuPicture& pic, const Buffer& corr, int mode )
{
  LfCccmApplyPush ap;
  memset( &ap, 0, sizeof( ap ) );
  ap.pic  = pic.buffer().address;
  ap.corr = corr.address;
  for( int c = 0; c < 2; c++ )
  {
    ap.picOrigin[c]  = pic.plane( c + 1 ).origin;
    ap.picStride[c]  = pic.plane( c + 1 ).stride;
    ap.corrOrigin[c] = pic.plane( c + 1 ).origin;   // the correction has the picture's layout
    const ClpRng& rng = cs.slice->clpRng( ComponentID( c + 1 ) );
    ap.clpMin[c] = rng.min;
    ap.clpMax[c] = rng.max;
  }
  ap.corrStride = pic.plane( 1 ).stride;
  ap.width      = pic.plane( 1 ).width;
  ap.height     = pic.plane( 1 ).height;
  ap.mode       = mode;
  return ap;
}

void VkLoopFilter::lfCccm( CodingStructure& cs, int impMode )
{
  const PreCalcValues& pcv = *cs.pcv;
  const SPS&           sps = *cs.sps;
  const Slice&         slice = *cs.slice;
  ensureOnGpu( cs );
  const GpuPicture::Plane& cb = m_pic.plane( 1 );
  const int widthC = cb.width, heightC = cb.height;
  // the updated chroma starts as the chroma after SAO: a copy of the picture, of which the chroma planes are used
  if( m_lfCorr.size < m_pic.buffer().size )
  {
    m_core->destroyBuffer( m_lfCorr );
    m_lfCorr = m_core->createBuffer( m_pic.buffer().size, Mem::HostCached );
  }
  const int downStride = widthC + 2;
  ensureBuffer( m_lfDown, size_t( downStride ) * ( heightC + 2 ) * sizeof( Pel ), Mem::GpuOnly );

  // the windows of the CTUs with LF-CCCM on, without those that lfCccmWindowProcess skips
  static const Size windowSizes[8] = { Size( 4, 4 ), Size( 8, 2 ), Size( 2, 8 ), Size( 8, 8 ), Size( 16, 16 ), Size( 32, 32 ), Size( 64, 64 ), Size( 128, 128 ) };
  static const int  modelSizes[8]  = { 7, 2, 3, 6, 5, 5, 11, 10 };
  std::vector<GpuLfCccmWindow> windows;
  for( int ctu = 0; ctu < pcv.sizeInCtus; ctu++ )
  {
    if( !slice.m_lfCccmEnabled.at( ctu ) )
    {
      continue;
    }
    const int sizeIdx = slice.m_lfCccmWindowSizeIndex.at( ctu );
    const int type    = slice.m_lfCccmModelType.at( ctu );
    if( m_check )
    {
      m_counts.lfCccmModelTypes[type]++;
      m_counts.lfCccmWindowSizes[sizeIdx]++;
    }
    // the CTU's chroma area, clipped at the right and bottom picture edges (clipArea)
    const int px = ( ctu % pcv.widthInCtus ) * ( pcv.maxCUWidth >> 1 ), py = ( ctu / pcv.widthInCtus ) * ( pcv.maxCUHeight >> 1 );
    const int pw = std::min<int>( pcv.maxCUWidth >> 1, widthC - px ), ph = std::min<int>( pcv.maxCUWidth >> 1, heightC - py );
    const Size win = windowSizes[sizeIdx];
    for( int yi = py; yi < py + ph; yi += win.height )
    {
      for( int xi = px; xi < px + pw; xi += win.width )
      {
        GpuLfCccmWindow w;
        w.wx = xi;
        w.wy = yi;
        w.ww = std::min<int>( win.width, px + pw - xi );
        w.wh = std::min<int>( win.height, py + ph - yi );
        // lfCccmSetWindows: one sample more on each side, within the CTU; at its left and top edge the window
        // keeps its start and still grows by 2
        w.rx = std::max( px, w.wx - 1 );
        w.ry = std::max( py, w.wy - 1 );
        w.rw = std::min( w.ww + 2, px + pw - w.rx );
        w.rh = std::min( w.wh + 2, py + ph - w.ry );
        w.modelType = type;
        w.skip      = sizeIdx > 4;
        if( w.rw * w.rh < modelSizes[type] )
        {
          continue;
        }
        windows.push_back( w );
      }
    }
  }
  ensureBuffer( m_lfWindows, std::max<size_t>( windows.size(), 1 ) * sizeof( GpuLfCccmWindow ), Mem::HostWrite );
  if( !windows.empty() )
  {
    memcpy( m_lfWindows.mapped, windows.data(), windows.size() * sizeof( GpuLfCccmWindow ) );
  }

  const int bdL = sps.getBitDepth( CHANNEL_TYPE_LUMA ), bdC = sps.getBitDepth( CHANNEL_TYPE_CHROMA );
  const int ver = sps.getCclmVerCollocatedChromaFlag(), hor = sps.getCclmHorCollocatedChromaFlag();
  LfCccmDownPush down;
  down.luma       = m_pic.buffer().address;
  down.dst        = m_lfDown.address;
  down.lumaOrigin = m_pic.plane( 0 ).origin;
  down.lumaStride = m_pic.plane( 0 ).stride;
  down.dstOrigin  = downStride + 1;
  down.dstStride  = downStride;
  down.width      = widthC;
  down.height     = heightC;
  down.type       = ver && hor ? 2 : !ver && !hor ? 1 : 0;
  down.lumaOffset = 1 << ( bdL - 1 );

  LfCccmWindowPush wp;
  memset( &wp, 0, sizeof( wp ) );
  wp.down       = m_lfDown.address;
  wp.pic        = m_pic.buffer().address;
  wp.update     = m_lfCorr.address;
  wp.windows    = m_lfWindows.address;
  wp.downOrigin = down.dstOrigin;
  wp.downStride = downStride;
  for( int c = 0; c < 2; c++ )
  {
    wp.picOrigin[c] = m_pic.plane( c + 1 ).origin;
    wp.picStride[c] = m_pic.plane( c + 1 ).stride;
    wp.updOrigin[c] = m_pic.plane( c + 1 ).origin;
    const ClpRng& rng = slice.clpRng( ComponentID( c + 1 ) );
    wp.clpMin[c] = rng.min;
    wp.clpMax[c] = rng.max;
  }
  wp.updStride  = cb.stride;
  wp.bitDepthC  = bdC;
  wp.lumaOffset = 1 << ( bdL - 1 );
  wp.impMode    = impMode;
  // lfCccmInitBd
  wp.badThr[0]  = 32 << std::max( 0, bdL - 10 );
  wp.badThr[1]  = 16 << std::max( 0, bdL - 10 );
  wp.badThr[2]  = 28 << std::max( 0, bdL - 10 );
  if( m_check )
  {
    ensureBuffer( m_lfStats, 6 * sizeof( uint32_t ), Mem::HostCached );
    memset( m_lfStats.mapped, 0, 6 * sizeof( uint32_t ) );
    wp.stats = m_lfStats.address;
  }

  const LfCccmApplyPush ap = applyPush( cs, m_pic, m_lfCorr, 0 );

  m_core->begin();
  const int t0 = m_core->timestamp();
  m_core->copy( m_pic.buffer(), m_lfCorr, 0, 0, m_pic.buffer().size );
  m_core->dispatch( "lfcccm_down", uint32_t( ( widthC + 2 + 15 ) / 16 ), uint32_t( ( heightC + 2 + 15 ) / 16 ), 1, &down, sizeof( down ) );
  m_core->barrier();
  if( !windows.empty() )
  {
    m_core->dispatch( "lfcccm_window", uint32_t( windows.size() ), 1, 1, &wp, sizeof( wp ) );
    m_core->barrier();
  }
  m_core->dispatch( "lfcccm_apply", uint32_t( ( widthC + 15 ) / 16 ), uint32_t( ( heightC + 15 ) / 16 ), 2, &ap, sizeof( ap ) );
  m_core->barrier();
  const int t1 = m_core->timestamp();
  m_core->submit();
  m_gpuMs += m_core->elapsedMs( t0, t1 );
  if( m_check )
  {
    m_counts.lfCccmPictures++;
    m_counts.lfCccmWindows += windows.size();
    for( int i = 0; i < 6; i++ )
    {
      m_counts.lfCccmBranches[i] += m_lfStats.ptr<uint32_t>()[i];
    }
    m_counts.lfCccmImpModes[impMode]++;
  }
}

void VkLoopFilter::lfCccmApply( CodingStructure& cs )
{
  ensureOnGpu( cs );
  const LfCccmApplyPush ap = applyPush( cs, m_pic, m_lfCorr, 1 );
  m_core->begin();
  const int t0 = m_core->timestamp();
  m_core->dispatch( "lfcccm_apply", uint32_t( ( ap.width + 15 ) / 16 ), uint32_t( ( ap.height + 15 ) / 16 ), 2, &ap, sizeof( ap ) );
  m_core->barrier();
  const int t1 = m_core->timestamp();
  m_core->submit();
  m_gpuMs += m_core->elapsedMs( t0, t1 );
  m_cpuCurrent = m_check;
  m_lastStage  = 5;
}

void VkLoopFilter::lfCccmCheck( CodingStructure& cs, const PelUnitBuf& cpuCorrection )
{
  if( !m_check )
  {
    return;
  }
  for( int c = 0; c < 2; c++ )
  {
    const GpuPicture::Plane& p = m_pic.plane( c + 1 );
    const CPelBuf&           r = cpuCorrection.bufs[c + 1];
    const Pel*               g = m_lfCorr.ptr<Pel>() + p.origin;
    int numDiff = 0, firstX = 0, firstY = 0;
    for( int y = 0; y < p.height; y++ )
    {
      for( int x = 0; x < p.width; x++ )
      {
        if( g[y * p.stride + x] != r.buf[y * r.stride + x] && !numDiff++ )
        {
          firstX = x;
          firstY = y;
        }
      }
    }
    if( numDiff )
    {
      THROW( "GPU LF-CCCM correction differs from the CPU in POC " << cs.slice->getPOC() << ": component " << c + 1 << ", "
             << numDiff << " samples, first at (" << firstX << ", " << firstY << "): CPU " << r.buf[firstY * r.stride + firstX]
             << ", GPU " << g[firstY * p.stride + firstX] );
    }
  }
}

}   // namespace vkd
