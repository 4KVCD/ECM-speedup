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

/** \file     VkLoopFilter.cpp
    \brief    In-loop filters of the GPU decoder
*/

#include "VkLoopFilter.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Slice.h"

#include <cstddef>
#include <cstring>

namespace vkd
{

// ---- GpuPicture ------------------------------------------------------------------------------------------------------

void GpuPicture::create( Core& core, ChromaFormat format, int width, int height, int margin )
{
  core.destroyBuffer( m_buf );
  m_format  = format;
  m_numComp = getNumberValidComponents( format );
  size_t total = 0;
  for( int c = 0; c < m_numComp; c++ )
  {
    const ComponentID comp = ComponentID( c );
    Plane& p  = m_plane[c];
    p.width   = width >> getComponentScaleX( comp, format );
    p.height  = height >> getComponentScaleY( comp, format );
    const int m = margin >> getComponentScaleX( comp, format );
    p.stride  = ( p.width + 2 * m + 15 ) & ~15;
    const int rows = p.height + 2 * m;
    p.origin  = int( total ) + m * p.stride + m;
    total += size_t( p.stride ) * rows;
  }
  m_buf = core.createBuffer( total * sizeof( Pel ), Mem::HostCached );
  memset( m_buf.mapped, 0, total * sizeof( Pel ) );
}

void GpuPicture::upload( const CPelUnitBuf& src )
{
  for( int c = 0; c < m_numComp; c++ )
  {
    const Plane&   p = m_plane[c];
    const CPelBuf& s = src.bufs[c];
    Pel*           d = m_buf.ptr<Pel>() + p.origin;
    for( int y = 0; y < p.height; y++ )
    {
      memcpy( d + y * p.stride, s.buf + y * s.stride, p.width * sizeof( Pel ) );
    }
  }
}

void GpuPicture::download( PelUnitBuf& dst ) const
{
  for( int c = 0; c < m_numComp; c++ )
  {
    const Plane& p = m_plane[c];
    PelBuf&      d = dst.bufs[c];
    const Pel*   s = m_buf.ptr<Pel>() + p.origin;
    for( int y = 0; y < p.height; y++ )
    {
      memcpy( d.buf + y * d.stride, s + y * p.stride, p.width * sizeof( Pel ) );
    }
  }
}

bool GpuPicture::compare( const CPelUnitBuf& ref, std::string& diff, int& comp, int& x0, int& y0 ) const
{
  for( int c = 0; c < m_numComp; c++ )
  {
    const Plane&   p = m_plane[c];
    const CPelBuf& r = ref.bufs[c];
    const Pel*     g = m_buf.ptr<Pel>() + p.origin;
    int numDiff = 0, firstX = -1, firstY = -1;
    for( int y = 0; y < p.height; y++ )
    {
      if( memcmp( g + y * p.stride, r.buf + y * r.stride, p.width * sizeof( Pel ) ) == 0 )
      {
        continue;
      }
      for( int x = 0; x < p.width; x++ )
      {
        if( g[y * p.stride + x] != r.buf[y * r.stride + x] )
        {
          if( !numDiff )
          {
            firstX = x;
            firstY = y;
          }
          numDiff++;
        }
      }
    }
    if( numDiff )
    {
      diff = "component " + std::to_string( c ) + ": " + std::to_string( numDiff ) + " samples differ, first at ("
             + std::to_string( firstX ) + ", " + std::to_string( firstY ) + "): CPU "
             + std::to_string( r.buf[firstY * r.stride + firstX] ) + ", GPU "
             + std::to_string( g[firstY * p.stride + firstX] );
      comp = c;
      x0   = firstX;
      y0   = firstY;
      return false;
    }
  }
  return true;
}

// ---- VkLoopFilter --------------------------------------------------------------------------------------------------

void VkLoopFilter::destroy()
{
  if( m_core )
  {
    m_pic.destroy( *m_core );
    m_core->destroyBuffer( m_edgeBuf );
  }
}

void VkLoopFilter::beginPicture( const CodingStructure& cs, bool filterOnCpu )
{
  const ChromaFormat format = cs.pcv->chrFormat;
  const int width = cs.pcv->lumaWidth, height = cs.pcv->lumaHeight;
  if( !m_pic.matches( format, width, height ) )
  {
    // Deblocking reads at most 8 samples beyond an edge, all inside the picture; the margin is for later filters.
    m_pic.create( *m_core, format, width, height, 16 );
    m_maps.unitsW    = ( width + 3 ) / 4;
    m_maps.unitsH    = ( height + 3 ) / 4;
    m_lumaMapBytes   = size_t( m_maps.unitsW ) * m_maps.unitsH * sizeof( DeblockLumaEdge );
    m_chromaMapBytes = size_t( m_maps.unitsW ) * m_maps.unitsH * sizeof( DeblockChromaEdge );
    m_core->destroyBuffer( m_edgeBuf );
    m_edgeBuf = m_core->createBuffer( 2 * ( m_lumaMapBytes + m_chromaMapBytes ), Mem::HostWrite );
    uint8_t* base    = m_edgeBuf.ptr<uint8_t>();
    m_maps.luma[0]   = reinterpret_cast<DeblockLumaEdge*>( base );
    m_maps.luma[1]   = reinterpret_cast<DeblockLumaEdge*>( base + m_lumaMapBytes );
    m_maps.chroma[0] = reinterpret_cast<DeblockChromaEdge*>( base + 2 * m_lumaMapBytes );
    m_maps.chroma[1] = reinterpret_cast<DeblockChromaEdge*>( base + 2 * m_lumaMapBytes + m_chromaMapBytes );
  }
  memset( m_edgeBuf.mapped, 0, m_edgeBuf.size );
  m_maps.filterSamples = filterOnCpu;
  m_pic.upload( cs.picture->getRecoBuf() );
  if( filterOnCpu )
  {
    const Pel* p = m_pic.buffer().ptr<Pel>();
    m_input.assign( p, p + m_pic.buffer().size / sizeof( Pel ) );
  }
}

std::string VkLoopFilter::describeLuma( int x, int y, const CPelUnitBuf& cpu ) const
{
  std::string s;
  char        line[256];
  auto edgeText = [&]( const DeblockLumaEdge& e ) {
    snprintf( line, sizeof( line ), "bs %u lenP %u lenQ %u largeP %u largeQ %u noP %u noQ %u asym %u qp %d tcOff %d betaOff %d",
              e.flags & 3, ( e.flags >> 2 ) & 7, ( e.flags >> 5 ) & 7, ( e.flags >> 8 ) & 1, ( e.flags >> 9 ) & 1,
              ( e.flags >> 10 ) & 1, ( e.flags >> 11 ) & 1, ( e.flags >> 12 ) & 1, e.qp, e.tcOffsetDiv2, e.betaOffsetDiv2 );
    return std::string( line );
  };
  const int ux = x >> 2, uy = y >> 2;
  for( int dx = -2; dx <= 2; dx++ )
  {
    if( ux + dx >= 0 && ux + dx < m_maps.unitsW && m_maps.luma[EDGE_VER][uy * m_maps.unitsW + ux + dx].flags )
    {
      s += "\n  vertical edge x=" + std::to_string( ( ux + dx ) * 4 ) + " y=" + std::to_string( uy * 4 ) + ": "
           + edgeText( m_maps.luma[EDGE_VER][uy * m_maps.unitsW + ux + dx] );
    }
  }
  for( int dy = -2; dy <= 2; dy++ )
  {
    if( uy + dy >= 0 && uy + dy < m_maps.unitsH && m_maps.luma[EDGE_HOR][( uy + dy ) * m_maps.unitsW + ux].flags )
    {
      s += "\n  horizontal edge x=" + std::to_string( ux * 4 ) + " y=" + std::to_string( ( uy + dy ) * 4 ) + ": "
           + edgeText( m_maps.luma[EDGE_HOR][( uy + dy ) * m_maps.unitsW + ux] );
    }
  }
  const GpuPicture::Plane& p = m_pic.plane( 0 );
  for( int r = y - 4; r <= y + 4; r++ )
  {
    if( r < 0 || r >= p.height )
    {
      continue;
    }
    std::string in = "\n  row " + std::to_string( r ) + " in: ", c = "\n          cpu:", g = "\n          gpu:";
    for( int col = std::max( 0, x - 8 ); col <= std::min( p.width - 1, x + 8 ); col++ )
    {
      in += " " + ( m_input.empty() ? std::string( "?" ) : std::to_string( m_input[p.origin + r * p.stride + col] ) );
      c += " " + std::to_string( cpu.bufs[0].buf[r * cpu.bufs[0].stride + col] );
      g += " " + std::to_string( m_pic.buffer().ptr<Pel>()[p.origin + r * p.stride + col] );
    }
    s += in + c + g;
  }
  return s;
}

// Push constants of deblock.comp. The member order and natural alignment match the shader's scalar layout.
struct DeblockPush
{
  uint64_t buf;
  int32_t  origin[3];
  int32_t  stride[3];
  uint64_t lumaEdges;
  uint64_t chromaEdges;
  int32_t  unitsW, unitsH, vertical, chroma;
  int32_t  bitDepth[2];
  int32_t  shiftX, shiftY;
  int32_t  ladfNumIntervals;
  int32_t  ladfQpOffset[5];
  int32_t  ladfLowerBound[5];
};
static_assert( offsetof( DeblockPush, lumaEdges ) == 32 && sizeof( DeblockPush ) == 128, "deblock.comp push constant layout" );

void VkLoopFilter::deblock( const CodingStructure& cs )
{
  const SPS&  sps = *cs.sps;
  DeblockPush push;
  memset( &push, 0, sizeof( push ) );
  push.buf = m_pic.buffer().address;
  for( int c = 0; c < m_pic.numComp(); c++ )
  {
    push.origin[c] = m_pic.plane( c ).origin;
    push.stride[c] = m_pic.plane( c ).stride;
  }
  push.unitsW      = m_maps.unitsW;
  push.unitsH      = m_maps.unitsH;
  push.bitDepth[0] = sps.getBitDepth( CHANNEL_TYPE_LUMA );
  push.bitDepth[1] = sps.getBitDepth( CHANNEL_TYPE_CHROMA );
  push.shiftX      = getComponentScaleX( COMPONENT_Cb, m_pic.format() );
  push.shiftY      = getComponentScaleY( COMPONENT_Cb, m_pic.format() );
#if LUMA_ADAPTIVE_DEBLOCKING_FILTER_QP_OFFSET
  if( sps.getLadfEnabled() )
  {
    push.ladfNumIntervals = sps.getLadfNumIntervals();
    for( int k = 0; k < sps.getLadfNumIntervals() && k < 5; k++ )
    {
      push.ladfQpOffset[k]   = sps.getLadfQpOffset( k );
      push.ladfLowerBound[k] = sps.getLadfIntervalLowerBound( k );
    }
  }
#endif
  const bool hasChroma = m_pic.numComp() > 1;
  m_core->begin();
  const int t0 = m_core->timestamp();
  for( int dir = 0; dir < 2; dir++ )
  {
    // one invocation per strip of 4 lines (vertical edges) or 4 columns (horizontal edges)
    const uint32_t groups = uint32_t( ( ( dir == EDGE_VER ? m_maps.unitsH : m_maps.unitsW ) + 63 ) / 64 );
    push.vertical    = dir == EDGE_VER;
    push.lumaEdges   = m_edgeBuf.address + dir * m_lumaMapBytes;
    push.chromaEdges = m_edgeBuf.address + 2 * m_lumaMapBytes + dir * m_chromaMapBytes;
    push.chroma      = 0;
    m_core->dispatch( "deblock", groups, 1, 1, &push, sizeof( push ) );
    if( hasChroma )
    {
      push.chroma = 1;
      m_core->dispatch( "deblock", groups, 1, 1, &push, sizeof( push ) );
    }
    m_core->barrier();
  }
  const int t1 = m_core->timestamp();
  m_core->submit();
  m_lastGpuMs = m_core->elapsedMs( t0, t1 );
}

}   // namespace vkd
