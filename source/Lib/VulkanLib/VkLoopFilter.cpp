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
#include "CommonLib/SampleAdaptiveOffset.h"
#include "CommonLib/Slice.h"
#include "CommonLib/UnitTools.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>

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

// ---- VkLoopFilter: picture state -----------------------------------------------------------------------------------

void VkLoopFilter::init( Core* core, bool check )
{
  m_core  = core;
  m_check = check;
}

void VkLoopFilter::destroy()
{
  if( !m_core )
  {
    return;
  }
  submitStages( true );
  m_pic.destroy( *m_core );
  m_tmp.destroy( *m_core );
  m_db.destroy( *m_core );
  m_src.destroy( *m_core );
  m_resi.destroy( *m_core );
  for( Buffer* b: { &m_edgeBuf, &m_saoParams, &m_bifBlocks, &m_bifTables, &m_ccSaoSets, &m_ctuBytes, &m_ccSaoTables,
                    &m_codingInfo, &m_refineCtus, &m_alfRes, &m_alfCls, &m_alfTables, &m_alfParams,
                    &m_lfCorr, &m_lfDown, &m_lfWindows, &m_lfStats } )
  {
    m_core->destroyBuffer( *b );
  }
#if JVET_AK0065_TALF
  m_core->destroyBuffer( m_talfParams );
  for( TalfRef& r: m_talfRefs )
  {
    m_core->destroyBuffer( r.buf );
  }
  m_talfRefs.clear();
#endif
}

const char* VkLoopFilter::stageName( int stage )
{
  static const char* names[NUM_STAGES] = { "deblocking", "SAO/BIF/CCSAO", "offset refinement", "LF-CCCM", "ALF", "temporal ALF", "clipping" };
  return names[stage];
}

int VkLoopFilter::stageBegin()
{
  if( !m_recording )
  {
    waitStages();
    m_core->begin();
    m_recording = true;
  }
  return m_core->timestamp();
}

void VkLoopFilter::stageEnd( Stage stage, int t0 )
{
  m_stageTimes.push_back( { stage, t0, m_core->timestamp() } );
  if( m_check )
  {
    submitStages( true );
  }
}

void VkLoopFilter::submitStages( bool wait )
{
  if( m_recording )
  {
    m_core->submit( false );
    m_recording = false;
  }
  if( wait )
  {
    waitStages();
  }
}

void VkLoopFilter::waitStages()
{
  if( m_core->inFlight() || m_recording )
  {
    submitStages( false );
    m_core->wait();
  }
  for( const StageTime& t: m_stageTimes )
  {
    addGpuTime( t.stage, t.t0, t.t1 );
  }
  m_stageTimes.clear();
}

void VkLoopFilter::endPicture()
{
  submitStages( m_check );
  m_downloadPending = !m_cpuCurrent;
}

void VkLoopFilter::completePicture( PelUnitBuf reco )
{
  waitStages();
  if( m_downloadPending )
  {
    m_pic.download( reco );
    m_downloadPending = false;
    m_cpuCurrent      = true;
  }
}

// Push constants of clip_luma.comp.
struct ClipLumaPush
{
  uint64_t buf;
  int32_t  origin, stride, width, height, minVal, maxVal;
};

bool VkLoopFilter::clipLuma( CodingStructure& cs, int minVal, int maxVal )
{
  if( !m_check && ( m_cpuCurrent || !m_gpuCurrent ) )
  {
    return false;
  }
  ensureOnGpu( cs );
  const GpuPicture::Plane& pl = m_pic.plane( 0 );
  ClipLumaPush push = { m_pic.buffer().address, pl.origin, pl.stride, pl.width, pl.height, minVal, maxVal };
  const int t0 = stageBegin();
  m_core->dispatch( "clip_luma", uint32_t( ( pl.width + 15 ) / 16 ), uint32_t( ( pl.height + 15 ) / 16 ), 1, &push, sizeof( push ) );
  m_core->barrier();
  stageEnd( STAGE_CLIP, t0 );
  m_cpuCurrent = m_check;
  m_lastStage  = 7;
  return true;
}

void VkLoopFilter::addGpuTime( Stage stage, int t0, int t1 )
{
  const double ms = m_core->elapsedMs( t0, t1 );
  m_gpuMs += ms;
  m_stageMs[stage] += ms;
}

void VkLoopFilter::ensureBuffer( Buffer& buf, size_t size, Mem mem )
{
  if( buf.size < size )
  {
    m_core->destroyBuffer( buf );
    buf = m_core->createBuffer( std::max<size_t>( size, 256 ), mem );
  }
}

void VkLoopFilter::beginPicture( CodingStructure& cs )
{
  const ChromaFormat format = cs.pcv->chrFormat;
  const int width = cs.pcv->lumaWidth, height = cs.pcv->lumaHeight;
  if( !m_pic.matches( format, width, height ) )
  {
    // Deblocking and SAO read at most 8 samples beyond an edge, inside the picture; CCSAO clamps its reads.
    m_pic.create( *m_core, format, width, height, 16 );
    m_tmp.create( *m_core, format, width, height, 16 );
    m_db.create( *m_core, format, width, height, 16 );
  }
  CHECK( m_downloadPending || m_recording, "the previous picture is not complete" );
  waitStages();
  m_cpuCurrent = true;
  m_gpuCurrent = false;
  m_lastStage  = 0;
  m_alfInputOnGpu = false;
  m_counts.pictures++;
#if JVET_AK0065_TALF
  talfForget( cs.picture );
#endif
}

void VkLoopFilter::ensureOnGpu( CodingStructure& cs )
{
  if( m_check || !m_gpuCurrent )
  {
    // In check mode every GPU stage starts from the CPU's input, so that a difference points at that stage. The
    // upload must not overtake recorded commands that read the picture.
    submitStages( true );
    m_pic.upload( cs.picture->getRecoBuf() );
    m_gpuCurrent = true;
  }
  if( m_check )
  {
    const Pel* p = m_pic.buffer().ptr<Pel>();
    m_input.assign( p, p + m_pic.buffer().size / sizeof( Pel ) );
  }
}

void VkLoopFilter::syncToCpu( CodingStructure& cs, bool cpuWrites )
{
  if( !m_cpuCurrent )
  {
    submitStages( true );
    PelUnitBuf reco = cs.picture->getRecoBuf();
    m_pic.download( reco );
    m_cpuCurrent = true;
  }
  if( !m_check && cpuWrites )
  {
    m_gpuCurrent = false;
  }
}

void VkLoopFilter::checkAgainstCpu( CodingStructure& cs, const char* stage )
{
  if( !m_check )
  {
    return;
  }
  PelUnitBuf  reco = cs.picture->getRecoBuf();
  std::string diff;
  int         comp, x, y;
  if( !m_pic.compare( reco, diff, comp, x, y ) )
  {
    THROW( "GPU " << stage << " differs from the CPU in POC " << cs.slice->getPOC() << ": " << diff
                  << ( comp == 0 && m_lastStage == 1 ? describeLuma( x, y, reco ) : std::string() ) );
  }
}

// ---- deblocking ----------------------------------------------------------------------------------------------------

DeblockEdgeMaps& VkLoopFilter::deblockPrepare( CodingStructure& cs )
{
  const int unitsW = ( cs.pcv->lumaWidth + 3 ) / 4, unitsH = ( cs.pcv->lumaHeight + 3 ) / 4;
  if( unitsW != m_maps.unitsW || unitsH != m_maps.unitsH || !m_edgeBuf.buffer )
  {
    m_maps.unitsW    = unitsW;
    m_maps.unitsH    = unitsH;
    m_lumaMapBytes   = size_t( unitsW ) * unitsH * sizeof( DeblockLumaEdge );
    m_chromaMapBytes = size_t( unitsW ) * unitsH * sizeof( DeblockChromaEdge );
    m_core->destroyBuffer( m_edgeBuf );
    m_edgeBuf = m_core->createBuffer( 2 * ( m_lumaMapBytes + m_chromaMapBytes ), Mem::HostWrite );
    uint8_t* base    = m_edgeBuf.ptr<uint8_t>();
    m_maps.luma[0]   = reinterpret_cast<DeblockLumaEdge*>( base );
    m_maps.luma[1]   = reinterpret_cast<DeblockLumaEdge*>( base + m_lumaMapBytes );
    m_maps.chroma[0] = reinterpret_cast<DeblockChromaEdge*>( base + 2 * m_lumaMapBytes );
    m_maps.chroma[1] = reinterpret_cast<DeblockChromaEdge*>( base + 2 * m_lumaMapBytes + m_chromaMapBytes );
  }
  memset( m_edgeBuf.mapped, 0, m_edgeBuf.size );
  m_maps.filterSamples = m_check;
  ensureOnGpu( cs );
  return m_maps;
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

void VkLoopFilter::deblock( CodingStructure& cs )
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
  for( int dir = 0; dir < 2 && m_check; dir++ )
  {
    for( int i = 0; i < m_maps.unitsW * m_maps.unitsH; i++ )
    {
      m_counts.deblockSegments += ( m_maps.luma[dir][i].flags & 3 ) != 0;
      m_counts.deblockSegments += ( m_maps.chroma[dir][i].flags & 15 ) != 0;
    }
  }
  const int t0 = stageBegin();
  // ALF reads the picture before deblocking
  m_core->copy( m_pic.buffer(), m_db.buffer(), 0, 0, m_pic.buffer().size );
  m_core->barrier();
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
  stageEnd( STAGE_DEBLOCK, t0 );
  m_cpuCurrent = m_check;
  m_lastStage  = 1;
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

// ---- SAO, bilateral filter, CCSAO ----------------------------------------------------------------------------------

// Layouts shared with sao.comp, bif.comp, ccsao.comp and saoclip.comp.
struct GpuSaoCtu
{
  int32_t  type;
  uint32_t avail;
  int16_t  offset[32];
};
static_assert( sizeof( GpuSaoCtu ) == 72, "sao.comp SaoCtu layout" );

struct GpuBifBlock
{
  int32_t  x, y, w, h;
  int32_t  comp;
  int32_t  qp;
  int32_t  strengthLog2W, strengthLog2H;
  uint32_t flags;
};

struct GpuCcSaoSet
{
  int32_t type;
  int32_t candPos[3];
  int32_t bandNum[3];
  int16_t offset[64];
};
static_assert( sizeof( GpuCcSaoSet ) == 156, "ccsao.comp CcSaoSet layout" );

struct SaoPush
{
  uint64_t src, dst, params;
  int32_t  origin, stride, width, height, ctuW, ctuH, widthInCtus, numComp, comp, bitDepth, scaleX, scaleY;
  int32_t  numHorVb, numVerVb, horVb[3], verVb[3];
};
static_assert( sizeof( SaoPush ) == 104, "sao.comp push constant layout" );

struct BifPush
{
  uint64_t src, dst, blocks, tables;
  int32_t  origin[3], stride[3], bitDepth, maxVal[3], roundAdd, roundShift;
};
static_assert( sizeof( BifPush ) == 80, "bif.comp push constant layout" );

struct CcSaoPush
{
  uint64_t src, dst, sets, setIdc, avail;
  int32_t  origin[3], stride[3], width[3], height[3];
  int32_t  comp, numCtus, widthInCtus, ctuSize, scaleX, scaleY, bitDepth, offsetStepLog2;
  int32_t  numHorVb, numVerVb, horVb[3], verVb[3];
  uint64_t tabs;
};
static_assert( offsetof( CcSaoPush, tabs ) == 152 && sizeof( CcSaoPush ) == 160, "ccsao.comp push constant layout" );

struct ClipPush
{
  uint64_t dst, blocks, ctuClip;
  int32_t  origin[3], stride[3], maxVal[3];
  int32_t  mode, comp, width, height, ctuW, ctuH, widthInCtus, numCtus;
};

// The bilateral filter blocks of SAOProcess: every transform block it filters, split at virtual boundaries as
// BilateralFilter::bilateralFilterDiamond5x5 does, with the border availability that decides the padding.
static void addBifBlocks( std::vector<GpuBifBlock>& out, CodingStructure& cs, TransformUnit& tu,
                          ComponentID compID, int qp, bool crossedByVb, const int horVb[], const int verVb[], int numHorVb,
                          int numVerVb, bool clipTop, bool clipBottom, bool clipLeft, bool clipRight )
{
  const ChromaFormat fmt    = cs.pcv->chrFormat;
  const int          scaleX = getChannelTypeScaleX( toChannelType( compID ), fmt );
  const int          scaleY = getChannelTypeScaleY( toChannelType( compID ), fmt );
  const int          picW   = cs.pps->getPicWidthInLumaSamples() >> scaleX;
  const int          picH   = cs.pps->getPicHeightInLumaSamples() >> scaleY;
  const int          ctuSize = cs.sps->getCTUSize();
  const bool         noPadFromPicture = isChroma( compID ) && fmt != CHROMA_420;
  const bool         inter  = tu.cu->predMode == MODE_INTER;
  int strengthW = 0, strengthH = 0;
  if( isChroma( compID ) )
  {
    const Size sz = tu.blocks[COMPONENT_Y].valid() ? tu.blocks[COMPONENT_Y].size() : tu.blocks[compID].size();
    strengthW = floorLog2( sz.width );
    strengthH = floorLog2( sz.height );
  }
  auto add = [&]( int x, int y, int w, int h, bool availT, bool availB, bool availL, bool availR, bool ctbTop, bool ctbBottom, bool clipNow ) {
    if( noPadFromPicture )
    {
      availT = availB = availL = availR = false;
    }
    GpuBifBlock b;
    b.x = x;
    b.y = y;
    b.w = w;
    b.h = h;
    b.comp = compID;
    b.qp   = qp;
    b.strengthLog2W = isLuma( compID ) ? floorLog2( w ) : strengthW;
    b.strengthLog2H = isLuma( compID ) ? floorLog2( h ) : strengthH;
    b.flags = ( isLuma( compID ) ? 1u : 0u ) | ( inter ? 2u : 0u ) | ( availT ? 4u : 0u ) | ( availB ? 8u : 0u ) | ( availL ? 16u : 0u )
              | ( availR ? 32u : 0u ) | ( ctbTop ? 64u : 0u ) | ( ctbBottom ? 128u : 0u ) | ( clipNow ? 256u : 0u );
    out.push_back( b );
  };
  const CompArea& area = tu.blocks[compID];
  if( crossedByVb )
  {
    int yStart = area.y;
    for( int i = 0; i <= numHorVb; i++ )
    {
      const int  yEnd  = i == numHorVb ? area.y + area.height : ( horVb[i] >> scaleY );
      const int  h     = yEnd - yStart;
      const bool clipT = ( i == 0 && clipTop ) || i > 0 || yStart - 2 < 0;
      const bool clipB = ( i == numHorVb && clipBottom ) || i < numHorVb || yEnd + 2 >= picH;
      int xStart = area.x;
      for( int j = 0; j <= numVerVb; j++ )
      {
        const int  xEnd  = j == numVerVb ? area.x + area.width : ( verVb[j] >> scaleX );
        const int  w     = xEnd - xStart;
        const bool clipL = ( j == 0 && clipLeft ) || j > 0 || xStart - 2 < 0;
        const bool clipR = ( j == numVerVb && clipRight ) || j < numVerVb || xEnd + 2 >= picW;
        add( xStart, yStart, w, h, !clipT && yStart - 2 >= 0, !clipB && yStart + h + 1 < picH, !clipL && xStart - 2 >= 0,
             !clipR && xStart + w + 1 < picW, yStart % ( ctuSize >> scaleY ) == 0, ( yStart + h ) % ( ctuSize >> scaleY ) == 0, true );
        xStart = xEnd;
      }
      yStart = yEnd;
    }
  }
  else
  {
    // The CPU tests the top CTU boundary with the horizontal scale; that only matters for 4:2:2.
    add( area.x, area.y, area.width, area.height, area.y - 2 >= 0, int( area.y + area.height + 1 ) < picH, area.x - 2 >= 0,
         int( area.x + area.width + 1 ) < picW, area.y % ( ctuSize >> scaleX ) == 0, ( area.y + area.height ) % ( ctuSize >> scaleY ) == 0,
         false );
  }
}

void VkLoopFilter::sao( CodingStructure& cs, SampleAdaptiveOffset& saoCpu )
{
  const PreCalcValues& pcv = *cs.pcv;
  const SPS&           sps = *cs.sps;
  const PPS&           pps = *cs.pps;
  const int  numComp   = m_pic.numComp();
  const int  numCtus   = pcv.sizeInCtus;
  const bool saoOn     = sps.getSAOEnabledFlag();
  const bool ccSaoOn   = sps.getCCSAOEnabledFlag();
  const bool bifOn     = pps.getUseBIF() || pps.getUseChromaBIF();
  if( !saoOn && !bifOn && !ccSaoOn )
  {
    return;
  }
  ensureOnGpu( cs );
  BilateralFilter& bif = saoCpu.m_bilateralFilter;
  const PicHeader& ph  = *cs.picHeader;
  int numHorVb = 0, numVerVb = 0, horVb[3] = { 0, 0, 0 }, verVb[3] = { 0, 0, 0 };
  if( ph.getVirtualBoundariesPresentFlag() )
  {
    numHorVb = ph.getNumHorVirtualBoundaries();
    numVerVb = ph.getNumVerVirtualBoundaries();
    for( int i = 0; i < numHorVb; i++ )
    {
      horVb[i] = ph.getVirtualBoundariesPosY( i );
    }
    for( int i = 0; i < numVerVb; i++ )
    {
      verVb[i] = ph.getVirtualBoundariesPosX( i );
    }
  }

  // Per CTU: availability, SAO parameters, CCSAO set and clipping flag.
  ensureBuffer( m_ctuBytes, size_t( numCtus ) * 7, Mem::HostWrite );
  uint8_t* ccSetIdc = m_ctuBytes.ptr<uint8_t>();
  uint8_t* avail    = ccSetIdc + 3 * numCtus;
  uint8_t* ctuClip  = avail + numCtus;
  memset( m_ctuBytes.mapped, 0, size_t( numCtus ) * 7 );
  ensureBuffer( m_saoParams, size_t( numCtus ) * numComp * sizeof( GpuSaoCtu ), Mem::HostWrite );
  GpuSaoCtu* saoParams = m_saoParams.ptr<GpuSaoCtu>();
  std::vector<SAOBlkParam> recon( numCtus );
  if( saoOn )
  {
    saoCpu.reconstructSaoParams( cs, recon.data() );
  }
  const CcSaoComParam& ccParam = cs.slice->m_ccSaoComParam;
  bool anySao = false, anyCcSao = false;
  for( int ctu = 0; ctu < numCtus; ctu++ )
  {
    const Position pos( ( ctu % pcv.widthInCtus ) * pcv.maxCUWidth, ( ctu / pcv.widthInCtus ) * pcv.maxCUHeight );
    avail[ctu] = saoCpu.ctuAvailability( cs, pos );
    for( int c = 0; c < numComp; c++ )
    {
      GpuSaoCtu& p = saoParams[ctu * numComp + c];
      memset( &p, 0, sizeof( p ) );
      p.type  = -1;
      p.avail = avail[ctu];
      const SAOOffset& o = recon[ctu][c];
      if( saoOn && o.modeIdc != SAO_MODE_OFF )
      {
        p.type = o.typeIdc;
        for( int k = 0; k < 32; k++ )
        {
          p.offset[k] = int16_t( k < MAX_NUM_SAO_CLASSES ? o.offset[k] : 0 );
        }
        anySao       = true;
        ctuClip[c * numCtus + ctu] = 1;
      }
      if( ccSaoOn && ccParam.enabled[c] )
      {
        const uint8_t set = saoCpu.getCcSaoControlIdc( ComponentID( c ) )[ctu];
        ccSetIdc[c * numCtus + ctu] = set;
        if( set )
        {
          anyCcSao = true;
          ctuClip[c * numCtus + ctu] = 1;
        }
      }
    }
  }

  // The bilateral filter blocks, in the order SAOProcess filters them.
  std::vector<GpuBifBlock> blocks;
  if( bifOn )
  {
    int ctuRsAddr = 0;
    for( uint32_t yPos = 0; yPos < pcv.lumaHeight; yPos += pcv.maxCUHeight )
    {
      for( uint32_t xPos = 0; xPos < pcv.lumaWidth; xPos += pcv.maxCUWidth )
      {
        const uint32_t width  = std::min( pcv.maxCUWidth, pcv.lumaWidth - xPos );
        const uint32_t height = std::min( pcv.maxCUHeight, pcv.lumaHeight - yPos );
        const UnitArea area( cs.area.chromaFormat, Area( xPos, yPos, width, height ) );
        if( pps.getUseBIF() )
        {
          BifParams& bifParams = cs.picture->getBifParam( COMPONENT_Y );
          for( auto& cu: cs.traverseCUs( CS::getArea( cs, area, CH_L ), CH_L ) )
          {
            for( auto& tu: CU::traverseTUs( cu ) )
            {
              if( bifParams.ctuOn[ctuRsAddr] && bif.getApplyBIF( tu, COMPONENT_Y ) )
              {
                bool clipTop, clipBottom, clipLeft, clipRight;
                int  nh = 0, nv = 0, hp[3] = { 0, 0, 0 }, vp[3] = { 0, 0, 0 };
                const bool crossed = bif.isCrossedByVirtualBoundaries( cs, tu.Y().x, tu.Y().y, tu.lumaSize().width, tu.lumaSize().height,
                                                                       clipTop, clipBottom, clipLeft, clipRight, nh, nv, hp, vp );
                addBifBlocks( blocks, cs, tu, COMPONENT_Y, tu.cu->qp + pps.getBIFQPOffset(), crossed, hp, vp, nh, nv, clipTop,
                              clipBottom, clipLeft, clipRight );
              }
            }
          }
        }
        if( pps.getUseChromaBIF() )
        {
          const ChannelType chType = CS::isDualITree( cs ) ? CH_C : CH_L;
          const int scaleX = getChannelTypeScaleX( CHANNEL_TYPE_CHROMA, pcv.chrFormat );
          const int scaleY = getChannelTypeScaleY( CHANNEL_TYPE_CHROMA, pcv.chrFormat );
          for( auto& cu: cs.traverseCUs( CS::getArea( cs, area, chType ), chType ) )
          {
            if( !( cu.Cb().valid() && cu.Cr().valid() ) )
            {
              continue;
            }
            for( auto& tu: CU::traverseTUs( cu ) )
            {
              for( int c = COMPONENT_Cb; c < MAX_NUM_COMPONENT; c++ )
              {
                const ComponentID compID = ComponentID( c );
                if( cs.picture->getBifParam( compID ).ctuOn[ctuRsAddr] && bif.getApplyBIF( tu, compID ) )
                {
                  bool clipTop, clipBottom, clipLeft, clipRight;
                  int  nh = 0, nv = 0, hp[3] = { 0, 0, 0 }, vp[3] = { 0, 0, 0 };
                  const CompArea& a = tu.block( compID );
                  const bool crossed = bif.isCrossedByVirtualBoundaries( cs, a.x << scaleX, a.y << scaleY, a.width << scaleX,
                                                                         a.height << scaleY, clipTop, clipBottom, clipLeft, clipRight,
                                                                         nh, nv, hp, vp );
                  addBifBlocks( blocks, cs, tu, compID, tu.cu->qp + pps.getChromaBIFQPOffset(), crossed, hp, vp, nh, nv,
                                clipTop, clipBottom, clipLeft, clipRight );
                }
              }
            }
          }
        }
        ctuRsAddr++;
      }
    }
  }
  if( !blocks.empty() )
  {
    ensureBuffer( m_bifBlocks, blocks.size() * sizeof( GpuBifBlock ), Mem::HostWrite );
    memcpy( m_bifBlocks.mapped, blocks.data(), blocks.size() * sizeof( GpuBifBlock ) );
    if( !m_bifTables.buffer )
    {
      ensureBuffer( m_bifTables, 2816, Mem::HostWrite );
      int8_t* t = m_bifTables.ptr<int8_t>();
      memcpy( t, bif.m_lut, 26 * 48 );
      memcpy( t + 1248, bif.m_lutChroma, 26 * 48 );
      memcpy( t + 2496, bif.m_tuSizeFactor, 128 );
      memcpy( t + 2624, bif.m_tuMADFactor, 32 );
      memcpy( t + 2656, bif.m_tuSizeFactorChroma, 128 );
      memcpy( t + 2784, bif.m_tuMADFactorChroma, 32 );
    }
  }

  // CCSAO sets of the slice, as CCSAOProcess uses them.
  if( anyCcSao )
  {
    ensureBuffer( m_ccSaoSets, 3 * MAX_CCSAO_SET_NUM * sizeof( GpuCcSaoSet ), Mem::HostWrite );
    GpuCcSaoSet* sets = m_ccSaoSets.ptr<GpuCcSaoSet>();
    memset( sets, 0, 3 * MAX_CCSAO_SET_NUM * sizeof( GpuCcSaoSet ) );
    for( int c = 0; c < numComp; c++ )
    {
      for( int s = 0; s < MAX_CCSAO_SET_NUM; s++ )
      {
        GpuCcSaoSet& g = sets[c * MAX_CCSAO_SET_NUM + s];
        g.type = ccParam.setType[c][s] == CCSAO_SET_TYPE_EDGE ? 1 : 0;
        for( int k = 0; k < 3; k++ )
        {
          g.candPos[k] = ccParam.candPos[c][s][k];
          g.bandNum[k] = ccParam.bandNum[c][s][k];
        }
        for( int k = 0; k < MAX_CCSAO_CLASS_NUM; k++ )
        {
          g.offset[k] = ccParam.offset[c][s][k];
        }
      }
    }
    if( !m_ccSaoTables.buffer )
    {
      ensureBuffer( m_ccSaoTables, 64 * sizeof( int16_t ), Mem::HostWrite );
      int16_t* t = m_ccSaoTables.ptr<int16_t>();
      for( int i = 0; i < MAX_CCSAO_EDGE_IDC; i++ )
      {
        for( int j = 0; j < MAX_CCSAO_EDGE_THR; j++ )
        {
          t[i * 16 + j] = g_ccSaoEdgeThr[i][j];
        }
      }
      for( int i = 0; i < MAX_CCSAO_BAND_IDC; i++ )
      {
        t[32 + 2 * i]     = g_ccSaoBandTab[i][0];
        t[32 + 2 * i + 1] = g_ccSaoBandTab[i][1];
      }
    }
  }

  const int t0 = stageBegin();
  m_core->copy( m_pic.buffer(), m_tmp.buffer(), 0, 0, m_pic.buffer().size );
  m_core->barrier();
  const int scaleXc = getComponentScaleX( COMPONENT_Cb, m_pic.format() ), scaleYc = getComponentScaleY( COMPONENT_Cb, m_pic.format() );
  if( anySao )
  {
    for( int c = 0; c < numComp; c++ )
    {
      const GpuPicture::Plane& pl = m_pic.plane( c );
      SaoPush p;
      memset( &p, 0, sizeof( p ) );
      p.src = m_tmp.buffer().address;
      p.dst = m_pic.buffer().address;
      p.params = m_saoParams.address;
      p.origin = pl.origin;
      p.stride = pl.stride;
      p.width  = pl.width;
      p.height = pl.height;
      p.scaleX = c ? scaleXc : 0;
      p.scaleY = c ? scaleYc : 0;
      p.ctuW   = pcv.maxCUWidth >> p.scaleX;
      p.ctuH   = pcv.maxCUHeight >> p.scaleY;
      p.widthInCtus = pcv.widthInCtus;
      p.numComp  = numComp;
      p.comp     = c;
      p.bitDepth = sps.getBitDepth( toChannelType( ComponentID( c ) ) );
      p.numHorVb = numHorVb;
      p.numVerVb = numVerVb;
      memcpy( p.horVb, horVb, sizeof( horVb ) );
      memcpy( p.verVb, verVb, sizeof( verVb ) );
      m_core->dispatch( "sao", ( pl.width + 15 ) / 16, ( pl.height + 15 ) / 16, 1, &p, sizeof( p ) );
    }
    m_core->barrier();
  }
  int maxVal[3] = { 0, 0, 0 };
  for( int c = 0; c < numComp; c++ )
  {
    maxVal[c] = ( 1 << sps.getBitDepth( toChannelType( ComponentID( c ) ) ) ) - 1;
  }
  if( !blocks.empty() )
  {
    BifPush p;
    memset( &p, 0, sizeof( p ) );
    p.src    = m_tmp.buffer().address;
    p.dst    = m_pic.buffer().address;
    p.blocks = m_bifBlocks.address;
    p.tables = m_bifTables.address;
    for( int c = 0; c < numComp; c++ )
    {
      p.origin[c] = m_pic.plane( c ).origin;
      p.stride[c] = m_pic.plane( c ).stride;
      p.maxVal[c] = maxVal[c];
    }
    p.bitDepth   = bif.internalBitDepth;
    p.roundAdd   = BIF_ROUND_ADD >> pps.getBIFStrength();
    p.roundShift = BIF_ROUND_SHIFT - pps.getBIFStrength();
    m_core->dispatch( "bif", uint32_t( blocks.size() ), 1, 1, &p, sizeof( p ) );
    m_core->barrier();
  }
  if( anyCcSao )
  {
    for( int c = 0; c < numComp; c++ )
    {
      if( !ccParam.enabled[c] )
      {
        continue;
      }
      CcSaoPush p;
      memset( &p, 0, sizeof( p ) );
      p.src    = m_tmp.buffer().address;
      p.dst    = m_pic.buffer().address;
      p.sets   = m_ccSaoSets.address;
      p.setIdc = m_ctuBytes.address;
      p.avail  = m_ctuBytes.address + 3 * numCtus;
      for( int k = 0; k < numComp; k++ )
      {
        p.origin[k] = m_pic.plane( k ).origin;
        p.stride[k] = m_pic.plane( k ).stride;
        p.width[k]  = m_pic.plane( k ).width;
        p.height[k] = m_pic.plane( k ).height;
      }
      p.comp           = c;
      p.numCtus        = numCtus;
      p.widthInCtus    = pcv.widthInCtus;
      p.ctuSize        = pcv.maxCUWidth;
      p.scaleX         = scaleXc;
      p.scaleY         = scaleYc;
      p.bitDepth       = sps.getBitDepth( toChannelType( ComponentID( c ) ) );
      p.offsetStepLog2 = saoCpu.offsetStepLog2( ComponentID( c ) );
      p.numHorVb       = numHorVb;
      p.numVerVb       = numVerVb;
      memcpy( p.horVb, horVb, sizeof( horVb ) );
      memcpy( p.verVb, verVb, sizeof( verVb ) );
      p.tabs = m_ccSaoTables.address;
      // one invocation per step of the CPU's line sequence: at most the CTU height plus a first and a last line
      const uint32_t slots = uint32_t( ( pcv.maxCUHeight >> ( c ? scaleYc : 0 ) ) + 2 );
      m_core->dispatch( "ccsao", ( slots + 63 ) / 64, uint32_t( numCtus ), 1, &p, sizeof( p ) );
      // Each component reads the picture before SAO and writes only its own plane: no barrier between them.
    }
    m_core->barrier();
  }
  {
    ClipPush p;
    memset( &p, 0, sizeof( p ) );
    p.dst     = m_pic.buffer().address;
    p.blocks  = m_bifBlocks.address;
    p.ctuClip = m_ctuBytes.address + 4 * numCtus;
    for( int c = 0; c < numComp; c++ )
    {
      p.origin[c] = m_pic.plane( c ).origin;
      p.stride[c] = m_pic.plane( c ).stride;
      p.maxVal[c] = maxVal[c];
    }
    p.widthInCtus = pcv.widthInCtus;
    p.numCtus     = numCtus;
    if( anySao || anyCcSao )
    {
      p.mode = 0;
      for( int c = 0; c < numComp; c++ )
      {
        const GpuPicture::Plane& pl = m_pic.plane( c );
        p.comp   = c;
        p.width  = pl.width;
        p.height = pl.height;
        p.ctuW   = pcv.maxCUWidth >> ( c ? scaleXc : 0 );
        p.ctuH   = pcv.maxCUHeight >> ( c ? scaleYc : 0 );
        m_core->dispatch( "saoclip", uint32_t( ( pl.width * pl.height + 255 ) / 256 ), 1, 1, &p, sizeof( p ) );
      }
    }
    if( !blocks.empty() )
    {
      // after the per-CTU clipping: both clip the same samples in some CTUs
      m_core->barrier();
      p.mode = 1;
      m_core->dispatch( "saoclip", uint32_t( blocks.size() ), 1, 1, &p, sizeof( p ) );
    }
  }
  m_core->barrier();
  stageEnd( STAGE_SAO, t0 );
  m_cpuCurrent = m_check;
  m_lastStage  = 2;
  if( m_check )
  {
    m_counts.bifBlocks += blocks.size();
    for( int i = 0; i < numCtus * numComp; i++ )
    {
      if( saoParams[i].type >= 0 )
      {
        m_counts.saoCtus[saoParams[i].type]++;
      }
    }
    for( int c = 0; c < numComp; c++ )
    {
      for( int ctu = 0; ctu < numCtus; ctu++ )
      {
        const int set = ccSetIdc[c * numCtus + ctu];
        if( set )
        {
          const int pos = ccParam.candPos[c][set - 1][0];
          if( ccParam.setType[c][set - 1] == CCSAO_SET_TYPE_EDGE )
          {
            m_counts.ccSaoEdge[pos & 3]++;
          }
          else
          {
            m_counts.ccSaoBand[pos < 9 ? pos : 8]++;
          }
        }
      }
    }
  }
}

}   // namespace vkd
