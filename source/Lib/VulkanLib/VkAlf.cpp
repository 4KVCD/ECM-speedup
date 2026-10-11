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

/** \file     VkAlf.cpp
    \brief    The adaptive loop filter of the GPU decoder
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

// The layouts shared with alf_common.glsl. Members follow the shader's order; the scalar layout aligns each member
// naturally, as C++ does.
struct AlfGpuParams
{
  uint64_t src, db, dst, resi, ci, res, cls, ccls, tables, ctus, lumaFilters, chromaFilters, ccAlfFilters;
  int32_t  srcOrigin[3], srcStride[3], dbOrigin[3], dbStride[3], dstOrigin[3], dstStride[3];
  int32_t  resiOrigin, resiStride, ciOrigin, ciStride;
  int32_t  resOrigin[AlfGpuPlan::NUM_PLANES], resStride[AlfGpuPlan::NUM_PLANES];
  int32_t  clsStride, clsPlane, cclsStride, cclsPlane;
  int32_t  idn9, coeff9, clip9, idn13, coeff13, clip13;
  int32_t  width, height, widthC, heightC, ctuSizeLog2, ctusW;
  int32_t  bitDepth[2], clpMin[3], clpMax[3];
  int32_t  intraSlice, fixedFilterAdjust, highRes;
  int32_t  fixedSetIdx[3], fixedSet[6], clipValues[8], chromaScaleCorr[2], lumaScaleMode;
};

struct GpuAlfLumaFilter
{
  int32_t type, classifier, coeffBits, unused;
  int16_t coeff[MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF];
  int16_t clip[MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF];
  int32_t scaleIdx[MAX_NUM_ALF_CLASSES];
  int32_t scaleCorr[MAX_NUM_ALF_CLASSES];
};
struct GpuAlfChromaFilter
{
  int32_t type, scaleIdx;
  int16_t coeff[MAX_NUM_ALF_CHROMA_COEFF];
  int16_t clip[MAX_NUM_ALF_CHROMA_COEFF];
};
struct GpuAlfCcFilter
{
  int32_t coeffPrec;
  int16_t coeff[MAX_NUM_CC_ALF_CHROMA_COEFF];
};
struct GpuAlfCtu
{
  int16_t lumaMode, lumaFilter, chromaFilter[2], ccAlfFilter[2];
};
struct GpuAlfTile
{
  int32_t x, y, w, h, plane, arg;
};
struct AlfPush
{
  uint64_t params, tiles;
};
static_assert( MAX_NUM_ALF_CLASSES == 24 && MAX_NUM_ALF_LUMA_COEFF == 47 && MAX_NUM_ALF_CHROMA_COEFF == 26
                 && MAX_NUM_CC_ALF_CHROMA_COEFF == 34, "ALF constants of alf_common.glsl" );
static_assert( ALF_FILTER_9 == 3 && ALF_FILTER_13_EXT_DB_RESI_DIRECT == 9 && ALF_FILTER_13_EXT_DB_RESI == 10
                 && ALF_FILTER_13_DB_RESI_DIRECT == 11 && ALF_FILTER_13_DB_RESI == 12 && ALF_FILTER_9_NO_FIX == 13,
               "AlfFilterType values of alf_common.glsl" );
static_assert( sizeof( GpuAlfLumaFilter ) == 4720 && sizeof( GpuAlfChromaFilter ) == 112 && sizeof( GpuAlfCcFilter ) == 72
                 && sizeof( GpuAlfCtu ) == 12 && sizeof( GpuAlfTile ) == 24, "alf_common.glsl struct layouts" );

// fixed filter tables: element offsets in m_alfTables
static const int TAB_SETS     = NUM_SETS_FIXED_FILTERS;
static const int TAB_IDN      = TAB_SETS * NUM_CLASSES_FIX;
static const int TAB_COEFF9   = TAB_SETS * NUM_FIXED_FILTERS * FIX_FILTER_NUM_COEFF_9_DB_9;
static const int TAB_COEFF13  = TAB_SETS * NUM_FIXED_FILTERS * FIX_FILTER_NUM_COEFF_13_DB_9;
static const int OFF_IDN9     = 0;
static const int OFF_COEFF9   = OFF_IDN9 + TAB_IDN;
static const int OFF_CLIP9    = OFF_COEFF9 + TAB_COEFF9;
static const int OFF_IDN13    = OFF_CLIP9 + TAB_COEFF9;
static const int OFF_COEFF13  = OFF_IDN13 + TAB_IDN;
static const int OFF_CLIP13   = OFF_COEFF13 + TAB_COEFF13;
static const int TAB_TOTAL    = OFF_CLIP13 + TAB_COEFF13;
static_assert( NUM_CLASSES_FIX == 7168 && NUM_FIXED_FILTERS == 512, "fixed filter table sizes of alf_fixed.comp" );

// Splits regions into tiles of up to 16x16 samples, one workgroup each.
static void addTiles( std::vector<GpuAlfTile>& tiles, int x, int y, int w, int h, int plane, int arg )
{
  for( int ty = y; ty < y + h; ty += 16 )
  {
    for( int tx = x; tx < x + w; tx += 16 )
    {
      tiles.push_back( { tx, ty, std::min( 16, x + w - tx ), std::min( 16, y + h - ty ), plane, arg } );
    }
  }
}

void VkLoopFilter::alfBuffers( CodingStructure& cs, AdaptiveLoopFilter& alf )
{
  const ChromaFormat format = cs.pcv->chrFormat;
  const int width = cs.pcv->lumaWidth, height = cs.pcv->lumaHeight;
  if( !m_src.matches( format, width, height ) )
  {
    m_src.create( *m_core, format, width, height, 16 );
    m_resi.create( *m_core, CHROMA_400, width, height, 16 );
    m_core->destroyBuffer( m_alfCls );
    const size_t lumaBlocks = size_t( width / 2 ) * ( height / 2 );
    m_alfCls = m_core->createBuffer( ( 6 * lumaBlocks + 2 * lumaBlocks / 4 + 64 ) * sizeof( int16_t ), Mem::HostCached );
  }
  AdaptiveLoopFilter::ResultPlane planes[AlfGpuPlan::NUM_PLANES];
  alf.resultPlanes( planes );
  size_t total = 0;
  bool   same  = m_alfRes.buffer != VK_NULL_HANDLE;
  for( int p = 0; p < AlfGpuPlan::NUM_PLANES; p++ )
  {
    same = same && planes[p].width == m_alfResWidth[p] && planes[p].height == m_alfResHeight[p];
    m_alfResOffset[p] = total;
    m_alfResWidth[p]  = planes[p].width;
    m_alfResHeight[p] = planes[p].height;
    total += size_t( planes[p].width ) * planes[p].height * sizeof( Pel );
  }
  if( !same )
  {
    m_core->destroyBuffer( m_alfRes );
    m_alfRes = m_core->createBuffer( std::max<size_t>( total, 256 ), Mem::HostCached );
    memset( m_alfRes.mapped, 0, m_alfRes.size );
  }
  if( !m_alfTables.buffer )
  {
    m_alfTables = m_core->createBuffer( TAB_TOTAL * sizeof( int16_t ), Mem::HostWrite );
    const AdaptiveLoopFilter::FixedFilterTables t = AdaptiveLoopFilter::fixedFilterTables();
    int16_t* d = m_alfTables.ptr<int16_t>();
    memcpy( d + OFF_IDN9, t.classIdn9Db9, TAB_IDN * sizeof( int16_t ) );
    memcpy( d + OFF_COEFF9, t.coeff9Db9, TAB_COEFF9 * sizeof( int16_t ) );
    memcpy( d + OFF_CLIP9, t.clip9Db9, TAB_COEFF9 * sizeof( int16_t ) );
    memcpy( d + OFF_IDN13, t.classIdn13Db9, TAB_IDN * sizeof( int16_t ) );
    memcpy( d + OFF_COEFF13, t.coeff13Db9, TAB_COEFF13 * sizeof( int16_t ) );
    memcpy( d + OFF_CLIP13, t.clip13Db9, TAB_COEFF13 * sizeof( int16_t ) );
  }
}

bool VkLoopFilter::alfComparePlanes( AdaptiveLoopFilter& alf, std::string& diff )
{
  static const char* names[AlfGpuPlan::NUM_PLANES] = { "luma fixed filter stage 1 set 0", "luma fixed filter stage 1 set 1",
                                                      "luma fixed filter stage 2 set 0", "luma fixed filter stage 2 set 1",
                                                      "Cb fixed filter set 0",           "Cb fixed filter set 1",
                                                      "Cr fixed filter set 0",           "Cr fixed filter set 1",
                                                      "residual fixed filter 0",         "residual fixed filter 1",
                                                      "gauss results",                   "laplacian results" };
  // Only the samples that the picture wrote: the CPU allocates the planes anew for every picture.
  AdaptiveLoopFilter::ResultPlane planes[AlfGpuPlan::NUM_PLANES];
  alf.resultPlanes( planes );
  for( const std::vector<AlfGpuPlan::Region>* regions: { &m_alfPlan.writes, &m_alfPlan.mirrors } )
  {
    for( const AlfGpuPlan::Region& r: *regions )
    {
      const int  p   = r.plane;
      const int  pad = AlfGpuPlan::planePadding( p );
      const Pel* g   = reinterpret_cast<const Pel*>( m_alfRes.ptr<uint8_t>() + m_alfResOffset[p] );
      int numDiff = 0, firstX = 0, firstY = 0;
      for( int y = r.y; y < r.y + r.h; y++ )
      {
        const Pel* gr = g + size_t( y + pad ) * planes[p].width + pad;
        const Pel* cr = planes[p].rows[y + pad] + pad;
        for( int x = r.x; x < r.x + r.w; x++ )
        {
          if( gr[x] != cr[x] && !numDiff++ )
          {
            firstX = x;
            firstY = y;
          }
        }
      }
      if( numDiff )
      {
        const Pel* gr = g + size_t( firstY + pad ) * planes[p].width + pad;
        diff = std::string( names[p] ) + ": " + std::to_string( numDiff ) + " samples differ in the "
               + ( regions == &m_alfPlan.writes ? "region (" : "mirrored region (" ) + std::to_string( r.x ) + ", "
               + std::to_string( r.y ) + ", " + std::to_string( r.w ) + "x" + std::to_string( r.h ) + "), first at ("
               + std::to_string( firstX ) + ", " + std::to_string( firstY ) + "): CPU "
               + std::to_string( planes[p].rows[firstY + pad][firstX + pad] ) + ", GPU " + std::to_string( gr[firstX] );
        return false;
      }
    }
  }
  return true;
}

// Check mode, after a difference: compares the GPU's luma classes with the CPU's where the CPU's are still those of
// the regions ALFProcess filtered last.
static std::string alfDiagnose( const AlfGpuPlan& plan, const AdaptiveLoopFilter& alf, const int16_t* cls, int width, int height, int ctuSize )
{
  const int clsStride = width / 2, clsPlane = ( width / 2 ) * ( height / 2 );
  std::string out;
  auto compare = [&]( const char* what, int gpuPlane, int cpuIdx, int x0, int y0, int w, int h, int& num, std::string& first ) {
    const AlfClassifier* const* rows = alf.classifierRows( cpuIdx );
    for( int y = std::max( 0, y0 ); y < std::min( height, y0 + h ); y += 2 )
    {
      for( int x = std::max( 0, x0 ); x < std::min( width, x0 + w ); x += 2 )
      {
        const int g = cls[gpuPlane * clsPlane + ( y >> 1 ) * clsStride + ( x >> 1 )];
        if( g != rows[y][x] )
        {
          if( !num++ )
          {
            first = std::string( what ) + " class at (" + std::to_string( x ) + ", " + std::to_string( y ) + "): CPU "
                    + std::to_string( rows[y][x] ) + ", GPU " + std::to_string( g );
          }
        }
      }
    }
  };
  int numCi = 0, numFix1 = 0, numFix2 = 0, numFinal = 0;
  std::string firstCi, firstFix1, firstFix2, firstFinal;
  const int ctusW = ( width + ctuSize - 1 ) / ctuSize;
  for( int i = 0; i < int( plan.ctus.size() ); i++ )
  {
    const AlfGpuPlan::Ctu& c = plan.ctus[i];
    const int x = ( i % ctusW ) * ctuSize, y = ( i / ctusW ) * ctuSize;
    if( c.lumaMode == AlfGpuPlan::LUMA_OFF )
    {
      continue;
    }
    compare( "coding information", 0, ALF_NUM_CLASSIFIER + 2, x, y, ctuSize, ctuSize, numCi, firstCi );
    if( c.lumaMode != AlfGpuPlan::LUMA_FILTER )
    {
      compare( "first-stage fixed filter", 1, 0, x, y, ctuSize, ctuSize, numFix1, firstFix1 );
    }
    else
    {
      const int classifier = plan.lumaFilters[c.lumaFilter].classifier;
      compare( ( "final, classifier " + std::to_string( classifier ) ).c_str(), 3 + classifier, classifier, x, y, ctuSize, ctuSize, numFinal, firstFinal );
    }
  }
  for( const AlfGpuPlan::Region& r: plan.writes )
  {
    if( r.plane == AlfGpuPlan::PLANE_FIX_LUMA + 2 || r.plane == AlfGpuPlan::PLANE_FIX_LUMA + 3 )
    {
      compare( "second-stage fixed filter", 2, ALF_NUM_CLASSIFIER, r.x, r.y, r.w, r.h, numFix2, firstFix2 );
    }
  }
  for( const std::string* f: { &firstCi, &firstFix1, &firstFix2, &firstFinal } )
  {
    if( !f->empty() )
    {
      out += "\n  " + *f;
    }
  }
  out += "\n  class differences: coding information " + std::to_string( numCi ) + ", first stage " + std::to_string( numFix1 )
         + ", second stage " + std::to_string( numFix2 ) + ", final " + std::to_string( numFinal );
  return out;
}

void VkLoopFilter::alfPrepare( CodingStructure& cs, AdaptiveLoopFilter& alf )
{
  alfBuffers( cs, alf );
  m_alfPlan.skipSamples   = !m_check;
  m_alfPlan.keepCpuInputs = true;
  alf.setGpuPlan( &m_alfPlan );
}

bool VkLoopFilter::alf( CodingStructure& cs, AdaptiveLoopFilter& alf )
{
  alf.setGpuPlan( nullptr );
  const AlfGpuPlan& plan = m_alfPlan;
  if( !plan.gpu )
  {
    // ALFProcess filtered the picture on the CPU
    m_alfCpuReasons[plan.cpuReason ? plan.cpuReason : "?"]++;
    m_counts.alfCpuPictures++;
    m_gpuCurrent = false;
    return false;
  }
  alfRun( cs, alf );
  if( m_check )
  {
    std::string diff;
    if( !alfComparePlanes( alf, diff ) )
    {
      THROW( "GPU ALF result planes differ from the CPU in POC " << cs.slice->getPOC() << ": " << diff
             << alfDiagnose( m_alfPlan, alf, m_alfCls.ptr<int16_t>(), cs.pcv->lumaWidth, cs.pcv->lumaHeight, cs.pcv->maxCUWidth ) );
    }
  }
  return true;
}

void VkLoopFilter::alfRun( CodingStructure& cs, AdaptiveLoopFilter& alf )
{
  const AlfGpuPlan&    plan = m_alfPlan;
  const PreCalcValues& pcv  = *cs.pcv;
  const int numComp = m_pic.numComp();
  const int width = pcv.lumaWidth, height = pcv.lumaHeight;
  const int widthC = numComp > 1 ? m_pic.plane( 1 ).width : 0, heightC = numComp > 1 ? m_pic.plane( 1 ).height : 0;

  // inputs: the picture before ALF and the residual; the coding information arrived after deblocking
  if( m_check )
  {
    // The CPU picture already holds ALFProcess's output; its input is in the ALF's copy.
    m_src.upload( alf.callRecBeforeAlfBuf() );
  }
  else
  {
    ensureOnGpu( cs );
  }
  m_resi.upload( alf.callRecResiBuf() );

  // per picture parameters, then CTUs, filters and tiles, each 8-byte aligned
  std::vector<GpuAlfTile> fixA, fixB, mirrorA[2], mirrorB[2], lumaTiles, chromaTiles;
  for( const AlfGpuPlan::Region& r: plan.writes )
  {
    const bool stage2 = r.plane == AlfGpuPlan::PLANE_FIX_LUMA + 2 || r.plane == AlfGpuPlan::PLANE_FIX_LUMA + 3;
    addTiles( stage2 ? fixB : fixA, r.x, r.y, r.w, r.h, r.plane, 0 );
  }
  for( const AlfGpuPlan::Region& r: plan.mirrors )
  {
    const bool stage2 = r.plane == AlfGpuPlan::PLANE_FIX_LUMA + 2 || r.plane == AlfGpuPlan::PLANE_FIX_LUMA + 3;
    addTiles( ( stage2 ? mirrorB : mirrorA )[r.axis], r.x, r.y, r.w, r.h, r.plane, r.axis );
  }
  const int ctuSize = pcv.maxCUWidth;
  const int ctusW   = pcv.widthInCtus;
  bool needChromaClasses = false;
  for( const AlfGpuPlan::Region& r: plan.writes )
  {
    needChromaClasses |= AlfGpuPlan::planeIsChroma( r.plane );
  }
  for( int ctu = 0; ctu < int( plan.ctus.size() ); ctu++ )
  {
    const AlfGpuPlan::Ctu& c = plan.ctus[ctu];
    const int x = ( ctu % ctusW ) * ctuSize, y = ( ctu / ctusW ) * ctuSize;
    const int w = std::min( ctuSize, width - x ), h = std::min( ctuSize, height - y );
    if( c.lumaMode != AlfGpuPlan::LUMA_OFF )
    {
      addTiles( lumaTiles, x, y, w, h, ctu, 0 );
    }
    for( int comp = 1; comp < numComp; comp++ )
    {
      if( c.chromaFilter[comp - 1] >= 0 || c.ccAlfFilter[comp - 1] >= 0 || plan.chromaScaleCorr[comp - 1] != 0 )
      {
        addTiles( chromaTiles, x >> 1, y >> 1, w >> 1, h >> 1, ctu, comp );
      }
    }
  }

  const size_t numCtus = plan.ctus.size();
  auto align8 = []( size_t v ) { return ( v + 7 ) & ~size_t( 7 ); };
  size_t off = 0;
  const size_t offParams = off;   off = align8( off + sizeof( AlfGpuParams ) );
  const size_t offCtus   = off;   off = align8( off + numCtus * sizeof( GpuAlfCtu ) );
  const size_t offLuma   = off;   off = align8( off + plan.lumaFilters.size() * sizeof( GpuAlfLumaFilter ) );
  const size_t offChroma = off;   off = align8( off + plan.chromaFilters.size() * sizeof( GpuAlfChromaFilter ) );
  const size_t offCc     = off;   off = align8( off + plan.ccAlfFilters.size() * sizeof( GpuAlfCcFilter ) );
  std::vector<GpuAlfTile>* lists[] = { &fixA, &fixB, &mirrorA[0], &mirrorA[1], &mirrorB[0], &mirrorB[1], &lumaTiles, &chromaTiles };
  size_t offList[8];
  for( int i = 0; i < 8; i++ )
  {
    offList[i] = off;
    off = align8( off + lists[i]->size() * sizeof( GpuAlfTile ) );
  }
  ensureBuffer( m_alfParams, off, Mem::HostWrite );
  uint8_t* base = m_alfParams.ptr<uint8_t>();
  const uint64_t addr = m_alfParams.address;

  AlfGpuParams& P = *reinterpret_cast<AlfGpuParams*>( base + offParams );
  memset( &P, 0, sizeof( P ) );
  P.src    = m_src.buffer().address;
  P.db     = m_db.buffer().address;
  P.dst    = m_pic.buffer().address;
  P.resi   = m_resi.buffer().address;
  P.ci     = m_codingInfo.address;
  P.res    = m_alfRes.address;
  P.cls    = m_alfCls.address;
  P.clsStride = width / 2;
  P.clsPlane  = ( width / 2 ) * ( height / 2 );
  P.ccls      = m_alfCls.address + 6 * size_t( P.clsPlane ) * sizeof( int16_t );
  P.cclsStride = widthC / 2;
  P.cclsPlane  = ( widthC / 2 ) * ( heightC / 2 );
  P.tables = m_alfTables.address;
  P.ctus   = addr + offCtus;
  P.lumaFilters   = addr + offLuma;
  P.chromaFilters = addr + offChroma;
  P.ccAlfFilters  = addr + offCc;
  for( int c = 0; c < numComp; c++ )
  {
    P.srcOrigin[c] = m_src.plane( c ).origin;
    P.srcStride[c] = m_src.plane( c ).stride;
    P.dbOrigin[c]  = m_db.plane( c ).origin;
    P.dbStride[c]  = m_db.plane( c ).stride;
    P.dstOrigin[c] = m_pic.plane( c ).origin;
    P.dstStride[c] = m_pic.plane( c ).stride;
    const ClpRng& rng = cs.slice->clpRng( ComponentID( c ) );
    P.clpMin[c] = rng.min;
    P.clpMax[c] = rng.max;
  }
  P.resiOrigin = m_resi.plane( 0 ).origin;
  P.resiStride = m_resi.plane( 0 ).stride;
  P.ciOrigin   = 0;
  P.ciStride   = width;
  for( int p = 0; p < AlfGpuPlan::NUM_PLANES; p++ )
  {
    const int pad = AlfGpuPlan::planePadding( p );
    P.resOrigin[p] = int( m_alfResOffset[p] / sizeof( Pel ) ) + pad * m_alfResWidth[p] + pad;
    P.resStride[p] = m_alfResWidth[p];
  }
  P.idn9 = OFF_IDN9;
  P.coeff9 = OFF_COEFF9;
  P.clip9 = OFF_CLIP9;
  P.idn13 = OFF_IDN13;
  P.coeff13 = OFF_COEFF13;
  P.clip13 = OFF_CLIP13;
  P.width = width;
  P.height = height;
  P.widthC = widthC;
  P.heightC = heightC;
  P.ctuSizeLog2 = floorLog2( ctuSize );
  P.ctusW = ctusW;
  P.bitDepth[0] = plan.bitDepth[CHANNEL_TYPE_LUMA];
  P.bitDepth[1] = plan.bitDepth[CHANNEL_TYPE_CHROMA];
  P.intraSlice = plan.intraSlice;
  P.fixedFilterAdjust = plan.fixedFilterAdjust;
  P.highRes = width > 1280 && height > 720;
  for( int c = 0; c < MAX_NUM_COMPONENT; c++ )
  {
    P.fixedSetIdx[c] = plan.fixedSetIdx[c];
    P.fixedSet[c * 2] = plan.fixedSet[c][0];
    P.fixedSet[c * 2 + 1] = plan.fixedSet[c][1];
  }
  for( int ch = 0; ch < MAX_NUM_CHANNEL_TYPE; ch++ )
  {
    for( int i = 0; i < 4; i++ )
    {
      P.clipValues[ch * 4 + i] = plan.clipValues[ch][i];
    }
  }
  P.chromaScaleCorr[0] = plan.chromaScaleCorr[0];
  P.chromaScaleCorr[1] = plan.chromaScaleCorr[1];
  P.lumaScaleMode = plan.lumaScaleMode;

  GpuAlfCtu* ctus = reinterpret_cast<GpuAlfCtu*>( base + offCtus );
  for( size_t i = 0; i < numCtus; i++ )
  {
    const AlfGpuPlan::Ctu& c = plan.ctus[i];
    ctus[i] = { int16_t( c.lumaMode ), c.lumaFilter, { c.chromaFilter[0], c.chromaFilter[1] }, { c.ccAlfFilter[0], c.ccAlfFilter[1] } };
  }
  GpuAlfLumaFilter* lf = reinterpret_cast<GpuAlfLumaFilter*>( base + offLuma );
  for( size_t i = 0; i < plan.lumaFilters.size(); i++ )
  {
    const AlfGpuPlan::LumaFilter& f = plan.lumaFilters[i];
    lf[i].type       = f.type;
    lf[i].classifier = f.classifier;
    lf[i].coeffBits  = f.coeffBits;
    lf[i].unused     = 0;
    memcpy( lf[i].coeff, f.coeff, sizeof( lf[i].coeff ) );
    memcpy( lf[i].clip, f.clip, sizeof( lf[i].clip ) );
    for( int c = 0; c < MAX_NUM_ALF_CLASSES; c++ )
    {
      lf[i].scaleIdx[c]  = f.scaleIdx[c];
      lf[i].scaleCorr[c] = f.scaleCorr[c];
    }
  }
  GpuAlfChromaFilter* cf = reinterpret_cast<GpuAlfChromaFilter*>( base + offChroma );
  for( size_t i = 0; i < plan.chromaFilters.size(); i++ )
  {
    const AlfGpuPlan::ChromaFilter& f = plan.chromaFilters[i];
    cf[i].type     = f.type;
    cf[i].scaleIdx = f.scaleIdx;
    memcpy( cf[i].coeff, f.coeff, sizeof( cf[i].coeff ) );
    memcpy( cf[i].clip, f.clip, sizeof( cf[i].clip ) );
  }
  GpuAlfCcFilter* ccf = reinterpret_cast<GpuAlfCcFilter*>( base + offCc );
  for( size_t i = 0; i < plan.ccAlfFilters.size(); i++ )
  {
    ccf[i].coeffPrec = plan.ccAlfFilters[i].coeffPrec;
    memcpy( ccf[i].coeff, plan.ccAlfFilters[i].coeff, sizeof( ccf[i].coeff ) );
  }
  for( int i = 0; i < 8; i++ )
  {
    if( !lists[i]->empty() )
    {
      memcpy( base + offList[i], lists[i]->data(), lists[i]->size() * sizeof( GpuAlfTile ) );
    }
  }

  if( m_check )
  {
    for( size_t i = 0; i < numCtus; i++ )
    {
      const AlfGpuPlan::Ctu& c = plan.ctus[i];
      if( c.lumaMode == AlfGpuPlan::LUMA_COPY_STAGE1 || c.lumaMode == AlfGpuPlan::LUMA_COPY_STAGE2 )
      {
        m_counts.alfLumaCtus[c.lumaMode - 1]++;
      }
      else if( c.lumaMode == AlfGpuPlan::LUMA_FILTER )
      {
        const AlfGpuPlan::LumaFilter& f = plan.lumaFilters[c.lumaFilter];
        m_counts.alfLumaCtus[2 + f.type - ALF_FILTER_13_EXT_DB_RESI_DIRECT]++;
        m_counts.alfClassifier[f.classifier]++;
        m_counts.alfScaleCtus += plan.lumaScaleMode;
      }
      for( int comp = 0; comp < 2; comp++ )
      {
        if( c.chromaFilter[comp] >= 0 )
        {
          m_counts.alfChromaCtus[plan.chromaFilters[c.chromaFilter[comp]].type == ALF_FILTER_9 ? 0 : 1]++;
        }
        m_counts.ccAlfCtus += c.ccAlfFilter[comp] >= 0;
      }
    }
    m_counts.alfChromaScalePictures += plan.chromaScaleCorr[0] || plan.chromaScaleCorr[1];
    m_counts.alfMirrors += plan.mirrors.size();
  }
  m_counts.alfPictures++;

  // dispatches
  m_core->begin();
  const int t0 = m_core->timestamp();
  if( !m_check )
  {
    // the ALF input; in check mode the CPU's copy was uploaded instead
    m_core->copy( m_pic.buffer(), m_src.buffer(), 0, 0, m_pic.buffer().size );
    m_core->barrier();
  }
  else
  {
    // the output starts as the input
    m_core->copy( m_src.buffer(), m_pic.buffer(), 0, 0, m_pic.buffer().size );
    m_core->barrier();
  }
  struct PadPush
  {
    uint64_t buf;
    int32_t  origin, stride, width, height, pad;
  };
  auto pad = [&]( const GpuPicture& pic, int c, int margin ) {
    const GpuPicture::Plane& pl = pic.plane( c );
    const int m = c ? margin >> getComponentScaleX( ComponentID( c ), pic.format() ) : margin;
    PadPush push = { pic.buffer().address, pl.origin, pl.stride, pl.width, pl.height, m };
    m_core->dispatch( "alf_pad", uint32_t( ( pl.width + 2 * m + 15 ) / 16 ), uint32_t( ( pl.height + 2 * m + 15 ) / 16 ), 1, &push, sizeof( push ) );
  };
  for( int c = 0; c < numComp; c++ )
  {
    pad( m_src, c, 16 );
    pad( m_db, c, 16 );
  }
  pad( m_resi, 0, 16 );
  m_core->barrier();

  AlfPush push = { addr + offParams, 0 };
  m_core->dispatch( "alf_class", uint32_t( ( width + 31 ) / 32 ), uint32_t( ( height + 31 ) / 32 ), 1, &push, sizeof( push ) );
  if( needChromaClasses )
  {
    m_core->dispatch( "alf_cclass", uint32_t( ( widthC / 2 + 15 ) / 16 ), uint32_t( ( heightC / 2 + 15 ) / 16 ), 2, &push, sizeof( push ) );
  }
  m_core->barrier();
  auto run = [&]( const char* shader, int list ) {
    if( !lists[list]->empty() )
    {
      push.tiles = addr + offList[list];
      m_core->dispatch( shader, uint32_t( lists[list]->size() ), 1, 1, &push, sizeof( push ) );
      m_core->barrier();
    }
  };
  run( "alf_fixed", 0 );    // first-stage, residual, chroma fixed filters, gauss and laplacian results
  run( "alf_mirror", 2 );
  run( "alf_mirror", 3 );
  run( "alf_fixed", 1 );    // second-stage fixed filter
  run( "alf_mirror", 4 );
  run( "alf_mirror", 5 );
  if( !lumaTiles.empty() )
  {
    push.tiles = addr + offList[6];
    m_core->dispatch( "alf_luma", uint32_t( lumaTiles.size() ), 1, 1, &push, sizeof( push ) );
  }
  if( !chromaTiles.empty() )
  {
    push.tiles = addr + offList[7];
    m_core->dispatch( "alf_chroma", uint32_t( chromaTiles.size() ), 1, 1, &push, sizeof( push ) );
  }
  m_core->barrier();
  const int t1 = m_core->timestamp();
  m_core->submit();
  m_gpuMs += m_core->elapsedMs( t0, t1 );
  m_cpuCurrent = m_check;
  m_gpuCurrent = true;
  m_lastStage  = 3;
  m_alfInputOnGpu = true;
}

// Push constants of offset_refine.comp
struct RefinePush
{
  uint64_t src, dst, ci, ctuOn;
  int32_t  srcOrigin, srcStride, dstOrigin, dstStride, ciStride, width, height, ctuSizeLog2, ctusW;
  int32_t  maxVal, bdScale, paramA, paramB, paramC, boundTh;
};
static_assert( offsetof( RefinePush, boundTh ) == 88, "offset_refine.comp push constant layout" );

void VkLoopFilter::offsetRefinement( CodingStructure& cs, int stageIdx, int refineIdx, AdaptiveLoopFilter& alf )
{
  const SPS& sps = *cs.sps;
  const int  width = cs.pcv->lumaWidth, height = cs.pcv->lumaHeight;
  ensureOnGpu( cs );
  // the stage's input: the picture before deblocking, or the ALF input
  const GpuPicture* src = &m_db;
  if( stageIdx > 0 )
  {
    if( !m_alfInputOnGpu || m_check )
    {
      if( !m_src.matches( m_pic.format(), width, height ) )
      {
        m_src.create( *m_core, m_pic.format(), width, height, 16 );
      }
      m_src.upload( alf.callRecAfterSaoBuf() );
    }
    src = &m_src;
  }
  // after ALF, only the CTUs with luma ALF on, if the slice has luma ALF
  const int numCtus = cs.pcv->sizeInCtus;
  ensureBuffer( m_refineCtus, size_t( numCtus ), Mem::HostWrite );
  const bool perCtu = stageIdx > 0 && cs.slice->getTileGroupAlfEnabledFlag( COMPONENT_Y );
  if( perCtu )
  {
    memcpy( m_refineCtus.mapped, cs.picture->getAlfCtuEnableFlag( COMPONENT_Y ), numCtus );
  }

  const int bitDepth = sps.getBitDepth( CHANNEL_TYPE_LUMA );
  const bool useDiffTh = !sps.getInloopOffsetRefineFlag() && !cs.slice->isIntra();
  RefinePush push;
  push.src       = src->buffer().address;
  push.dst       = m_pic.buffer().address;
  push.ci        = m_codingInfo.address;
  push.ctuOn     = perCtu ? m_refineCtus.address : 0;
  push.srcOrigin = src->plane( 0 ).origin;
  push.srcStride = src->plane( 0 ).stride;
  push.dstOrigin = m_pic.plane( 0 ).origin;
  push.dstStride = m_pic.plane( 0 ).stride;
  push.ciStride  = width;
  push.width     = width;
  push.height    = height;
  push.ctuSizeLog2 = floorLog2( cs.pcv->maxCUWidth );
  push.ctusW     = cs.pcv->widthInCtus;
  push.maxVal    = ( 1 << bitDepth ) - 1;
  push.bdScale   = std::max( 0, bitDepth - 10 );
  push.paramA    = sps.getInloopOffsetRefineFunc() ? 5 : 6;
  push.paramB    = 31;
  push.paramC    = 5;
  // the thresholds of calcOffsetRefinementBlk for samples at block boundaries
  push.boundTh   = refineIdx == 1 ? ( useDiffTh ? ( stageIdx ? 16 : 9 ) : ( stageIdx ? 15 : 12 ) ) : ( useDiffTh ? 3 : 1 );

  m_core->begin();
  const int t0 = m_core->timestamp();
  m_core->dispatch( "offset_refine", uint32_t( ( width + 15 ) / 16 ), uint32_t( ( height + 15 ) / 16 ), 1, &push, sizeof( push ) );
  m_core->barrier();
  const int t1 = m_core->timestamp();
  m_core->submit();
  m_gpuMs += m_core->elapsedMs( t0, t1 );
  m_cpuCurrent = m_check;
  m_lastStage  = 4;
  m_counts.refinePictures[stageIdx > 0 ? 1 : 0]++;
}

void VkLoopFilter::uploadCodingInfo( const CPelBuf& ci )
{
  const int width = m_pic.plane( 0 ).width, height = m_pic.plane( 0 ).height;
  ensureBuffer( m_codingInfo, size_t( width ) * height * sizeof( Pel ), Mem::HostWrite );
  Pel* d = m_codingInfo.ptr<Pel>();
  for( int y = 0; y < height; y++ )
  {
    memcpy( d + size_t( y ) * width, ci.buf + y * ci.stride, width * sizeof( Pel ) );
  }
}

}   // namespace vkd
