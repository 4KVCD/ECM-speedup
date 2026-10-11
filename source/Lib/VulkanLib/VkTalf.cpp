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

/** \file     VkTalf.cpp
    \brief    The temporal ALF of the GPU decoder
*/

#include "VkLoopFilter.h"

#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Slice.h"

#include <cstddef>
#include <cstring>

namespace vkd
{

#if JVET_AK0065_TALF
// Layouts shared with talf.comp.
struct GpuTalfFilter
{
  int32_t coeff[NUM_TALF_COEFF], clip[NUM_TALF_COEFF], shift, offset, shape;
};
struct GpuTalfRef
{
  uint64_t buf;
  int32_t  origin, stride;
};
struct TalfPush
{
  uint64_t dst, src, blocks, filters, refs;
  int32_t  dstOrigin, dstStride, srcOrigin, srcStride, numBlocks, bi;
};
static_assert( sizeof( TalfGpuPlan::Block ) == 16 && sizeof( GpuTalfFilter ) == 116 && sizeof( GpuTalfRef ) == 16,
               "talf.comp buffer layouts" );
static_assert( offsetof( TalfPush, bi ) == 60, "talf.comp push constant layout" );

// The reference samples that talf.comp reads lie within TALF_EXT_SIZE of the picture plus the template's reach.
static const int TALF_REF_MARGIN = 32;

const VkLoopFilter::TalfRef& VkLoopFilter::talfRef( const Picture* pic )
{
  for( TalfRef& r: m_talfRefs )
  {
    if( r.pic == pic )
    {
      r.lastUse = m_talfUse;
      return r;
    }
  }
  // Not on the GPU yet: take the least recently used slot that this picture does not use, or a new one.
  static const size_t maxRefs = 8;
  TalfRef* slot = nullptr;
  for( TalfRef& r: m_talfRefs )
  {
    if( m_talfRefs.size() >= maxRefs && r.lastUse < m_talfUse && ( !slot || r.lastUse < slot->lastUse ) )
    {
      slot = &r;
    }
  }
  if( !slot )
  {
    m_talfRefs.emplace_back();
    slot = &m_talfRefs.back();
  }
  // the CPU picture with the border that finishPicture padded, as filterBlkTAlf reads it
  const CPelBuf luma   = pic->getRecoBuf( COMPONENT_Y );
  const int     stride = int( luma.width ) + 2 * TALF_REF_MARGIN;
  const int     rows   = int( luma.height ) + 2 * TALF_REF_MARGIN;
  ensureBuffer( slot->buf, size_t( stride ) * rows * sizeof( Pel ), Mem::HostWrite );
  Pel* dst = slot->buf.ptr<Pel>();
  for( int y = -TALF_REF_MARGIN; y < rows - TALF_REF_MARGIN; y++, dst += stride )
  {
    memcpy( dst, luma.buf + y * luma.stride - TALF_REF_MARGIN, stride * sizeof( Pel ) );
  }
  slot->pic     = pic;
  slot->origin  = TALF_REF_MARGIN * stride + TALF_REF_MARGIN;
  slot->stride  = stride;
  slot->lastUse = m_talfUse;
  return *slot;
}

void VkLoopFilter::talfForget( const Picture* pic )
{
  for( TalfRef& r: m_talfRefs )
  {
    if( r.pic == pic )
    {
      r.pic = nullptr;
    }
  }
}

bool VkLoopFilter::talf( CodingStructure& cs, AdaptiveLoopFilter& alf )
{
  TalfGpuPlan& plan = m_talfPlan;
  if( !m_alfInputOnGpu )
  {
    plan.unsupported( "ALF ran on the CPU" );
  }
  else
  {
    alf.talfGpuPlan( cs, plan );
  }
  const int width = cs.pcv->lumaWidth, height = cs.pcv->lumaHeight;
  for( size_t i = 0; i < plan.refs.size() && plan.gpu; i++ )
  {
    const CPelBuf luma = plan.refs[i]->getRecoBuf( COMPONENT_Y );
    if( luma.width != width || luma.height != height )
    {
      plan.unsupported( "reference picture of another size" );
    }
    else if( int( plan.refs[i]->margin ) < TALF_REF_MARGIN )
    {
      plan.unsupported( "picture margin below 32" );
    }
  }
  if( !plan.gpu )
  {
    m_talfCpuReasons[plan.cpuReason ? plan.cpuReason : "?"]++;
    m_counts.talfCpuPictures++;
    return false;
  }
  if( m_check )
  {
    m_counts.talfPictures++;
    m_counts.talfBlocks += plan.blocks.size();
    m_counts.talfModes[plan.mode] += plan.blocks.empty() ? 0 : 1;
    for( const TalfGpuPlan::Block& b: plan.blocks )
    {
      m_counts.talfShapes[plan.filters[b.filter].shape]++;
    }
  }
  if( plan.blocks.empty() )
  {
    return true;
  }

  // blocks, filters and references, each 8-byte aligned
  const size_t offFilters = ( plan.blocks.size() * sizeof( TalfGpuPlan::Block ) + 7 ) & ~size_t( 7 );
  const size_t offRefs    = ( offFilters + plan.filters.size() * sizeof( GpuTalfFilter ) + 7 ) & ~size_t( 7 );
  ensureBuffer( m_talfParams, offRefs + plan.refs.size() * sizeof( GpuTalfRef ), Mem::HostWrite );
  uint8_t* base = m_talfParams.ptr<uint8_t>();
  memcpy( base, plan.blocks.data(), plan.blocks.size() * sizeof( TalfGpuPlan::Block ) );
  GpuTalfFilter* filters = reinterpret_cast<GpuTalfFilter*>( base + offFilters );
  for( size_t i = 0; i < plan.filters.size(); i++ )
  {
    const TalfGpuPlan::Filter& f = plan.filters[i];
    memcpy( filters[i].coeff, f.coeff, sizeof( f.coeff ) );
    memcpy( filters[i].clip, f.clip, sizeof( f.clip ) );
    filters[i].shift = f.shift;
    // filterBlkTAlf's 1 << ( shift - 1 ); x86 takes the shift count modulo 32
    filters[i].offset = int32_t( 1u << ( ( f.shift - 1 ) & 31 ) );
    filters[i].shape  = f.shape;
  }
  GpuTalfRef* refs = reinterpret_cast<GpuTalfRef*>( base + offRefs );
  m_talfUse++;
  for( size_t i = 0; i < plan.refs.size(); i++ )
  {
    const TalfRef& r = talfRef( plan.refs[i] );
    refs[i] = { r.buf.address, r.origin, r.stride };
  }

  ensureOnGpu( cs );
  const GpuPicture::Plane& dst = m_pic.plane( 0 );
  const GpuPicture::Plane& src = m_src.plane( 0 );
  TalfPush push = { m_pic.buffer().address, m_src.buffer().address, m_talfParams.address, m_talfParams.address + offFilters,
                    m_talfParams.address + offRefs, dst.origin, dst.stride, src.origin, src.stride,
                    int32_t( plan.blocks.size() ), plan.bi ? 1 : 0 };
  const int t0 = stageBegin();
  m_core->dispatch( "talf", uint32_t( ( plan.blocks.size() * 16 + 63 ) / 64 ), 1, 1, &push, sizeof( push ) );
  m_core->barrier();
  stageEnd( STAGE_TALF, t0 );
  m_cpuCurrent = m_check;
  m_lastStage  = 6;
  return true;
}
#endif

}   // namespace vkd
