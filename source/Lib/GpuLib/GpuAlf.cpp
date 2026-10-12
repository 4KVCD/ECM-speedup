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

/** \file     GpuAlf.cpp
    \brief    adaptive loop filter (ALFProcess) on a Vulkan compute device, bit-exact with the CPU version

    The CPU filters CTU by CTU; the intermediate results it derives for a CTU and its surroundings (classes, fixed-
    filter outputs, Gaussian and Laplacian images) depend only on the sample position, so the kernels derive each of
    them for the whole picture, with the CPU's mirror padding at the picture boundary, and then filter every CTU.
*/

#include "GpuAlf.h"

#include "CommonLib/BilateralFilter.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Slice.h"
#include "CommonLib/UnitTools.h"

#include "alf_chclass.spv.h"
#include "alf_chroma.spv.h"
#include "alf_ciclass.spv.h"
#include "alf_class12.spv.h"
#include "alf_fix0.spv.h"
#include "alf_fix1.spv.h"
#include "alf_fixclass.spv.h"
#include "alf_fixresi.spv.h"
#include "alf_gauss.spv.h"
#include "alf_luma.spv.h"
#include "lfcccm.spv.h"
#include "lfcccm_ds.spv.h"
#include "offset_refine.spv.h"
#include "sao_apply.spv.h"
#include "sao_bif.spv.h"
#include "sao_pad.spv.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

namespace
{
struct GpuPlane
{
  uint64_t addr;
  int32_t  origin;
  int32_t  stride;
};
static_assert( sizeof( GpuPlane ) == 16, "GLSL struct Plane" );

struct PushCiClass
{
  GpuPlane resi;
  GpuPlane ci;
  uint64_t out;
  int32_t  outStride, wBlk, hBlk, calcResi, bitDepth;
};

struct PushFixClass
{
  GpuPlane rec;
  uint64_t ciClass, fixClass0, fixClass1, mainClass0;
  int32_t  blkStride, wBlk, hBlk, bitDepth;
};
static_assert( sizeof( PushFixClass ) == 64, "push constants" );

struct PushClass12
{
  GpuPlane rec, resi;
  uint64_t ciClass, mainClass1, mainClass2;
  int32_t  blkStride, wBlk, hBlk, bitDepth;
};
static_assert( sizeof( PushClass12 ) == 72, "push constants" );

struct PushFixFilter
{
  GpuPlane src, db, fix0, dst;
  int32_t  clipValues[4];
  uint64_t fixClass, ciClass, classIdn, coeff, clip;
  int32_t  width, height, blkStride, useBound, useResi, bsFactor, resiFactor, offsetClip, outMin, outMax;
};
static_assert( sizeof( PushFixFilter ) == 160, "push constants" );

struct PushGauss
{
  GpuPlane db, rec, ci, gauss, lapl;
  int32_t  clipValues[4];
  uint64_t ciClass;
  int32_t  width, height, blkStride, useBound, bsFactor, offsetClip, diffGauss, diffLapl, diffLocal, outMin, outMax;
};
static_assert( sizeof( PushGauss ) == 152, "push constants" );

struct PushLuma
{
  GpuPlane rec, db, resi, fix0, fix1, fixResi, gauss, lapl, dst;
  uint64_t ctuInfo, params, mainClass0, mainClass1, mainClass2;
  int32_t  width, height, blkStride, ctuSizeLog2, ctusPerRow, outMin, outMax, currBase;
};
static_assert( sizeof( PushLuma ) == 216, "push constants" );

struct PushChClass
{
  GpuPlane rec;
  uint64_t fixClass;
  int32_t  blkStride, wBlk, hBlk, bitDepth;
};
static_assert( sizeof( PushChClass ) == 40, "push constants" );

struct PushChroma
{
  GpuPlane rec, fix, lumaRec, lumaResi, crossDb, dst, lfCccm;
  uint64_t ctuInfo, chromaCoeff, ccalfCoeff;
  int32_t  comp, width, height, ctuSizeLog2, ctusPerRow, withFixed, ccalfPrec, ccalfClip, scaleCorr, outMin, outMax,
    bitDepth;
};
static_assert( sizeof( PushChroma ) == 184, "push constants" );

struct PushLfDs
{
  GpuPlane luma, ds;
  int32_t  width, height, lumaOffset, filterType;
};
static_assert( sizeof( PushLfDs ) == 48, "push constants" );

struct PushLfCccm
{
  GpuPlane ds, cb, cr, corrCb, corrCr;
  uint64_t ctuList;
  int32_t  numCtus, numWindows, groupsX, ctusPerRow, ctuSize, width, height, impMode, lumaOffset, bitDepth, badThr,
    badThrReg, minCb, maxCb, minCr, maxCr, lumaOffsetLog2, pad0;
};
static_assert( sizeof( PushLfCccm ) == 160, "push constants" );

struct PushSaoBif
{
  GpuPlane src[3], dst[3];
  uint64_t blocks, tables;
  int32_t  numBlocks, groupsX, width, height, ctuSize, roundAdd, roundShift, pad0;
};
static_assert( sizeof( PushSaoBif ) == 144, "push constants" );

struct PushSaoApply
{
  GpuPlane srcY, srcCb, srcCr, bif, dst;
  uint64_t ctuInfo, saoOffsets, ccsao;
  int32_t  comp, width, height, ctuSizeLog2, ctusPerRow, bitDepth, outMin, outMax, ccsaoShift, pad0;
};
static_assert( sizeof( PushSaoApply ) == 144, "push constants" );

struct PushOffsetRefine
{
  GpuPlane pic, before, codingInfo;
  int32_t  width, height, bitDepth, refineIdx, useDiffTh, paramA, pad0, pad1;
};
static_assert( sizeof( PushOffsetRefine ) == 80, "push constants" );

struct PushSaoPad
{
  GpuPlane plane;
  int32_t  width, height, pad, pad0;
};
static_assert( sizeof( PushSaoPad ) == 32, "push constants" );

// the bilateral filter's tables (sao_bif.comp): LUT [luma, chroma][26][48], MAD factor [luma, chroma][2][16]
constexpr int BIF_LUT_BYTES   = 26 * 48;
constexpr int BIF_TABLE_BYTES = 2 * BIF_LUT_BYTES + 2 * 2 * 16;
// CCSAO set parameters and offsets (sao_apply.comp)
constexpr int CCSAO_STRIDE = 72;

// coefficient-set layouts, as in alf_luma.comp and alf_chroma.comp
constexpr int NUM_COEFF       = MAX_NUM_ALF_LUMA_COEFF;
constexpr int SET_CLIP        = MAX_NUM_ALF_CLASSES * NUM_COEFF;
constexpr int SET_SCALE       = 2 * MAX_NUM_ALF_CLASSES * NUM_COEFF;
constexpr int SET_CORR        = SET_SCALE + MAX_NUM_ALF_CLASSES;
constexpr int SET_SHIFT       = SET_CORR + MAX_NUM_ALF_CLASSES;
constexpr int SET_SCALING     = SET_SHIFT + 1;
constexpr int LUMA_SET_STRIDE = SET_SCALING + 7;
constexpr int MAX_LUMA_SETS   = ALF_CTB_MAX_NUM_APS * MAX_NUM_ALF_ALTERNATIVES_LUMA;
constexpr int CHROMA_STRIDE   = 64;
constexpr int CCALF_STRIDE    = 34;
static_assert( NUM_COEFF == 47 && MAX_NUM_ALF_CLASSES == 24, "alf_luma.comp" );
static_assert( MAX_NUM_ALF_CHROMA_COEFF == 26 && MAX_NUM_CC_ALF_CHROMA_COEFF <= CCALF_STRIDE, "alf_chroma.comp" );

uint32_t divUp( int a, int b )
{
  return uint32_t( ( a + b - 1 ) / b );
}

VkDeviceSize align256( VkDeviceSize v )
{
  return ( v + 255 ) & ~VkDeviceSize( 255 );
}

int log2Of( int v )
{
  int l = 0;
  while( ( 1 << l ) < v )
  {
    l++;
  }
  return l;
}

/// the fixed-filter sets of a QP: AdaptiveLoopFilter::deriveClassificationAndFixFilterResultsBlk
void fixedFilterQpSets( int qp, int& minSet, int& maxSet )
{
  minSet = ( qp - 22 ) / 4;
  maxSet = ( qp - 18 ) / 4;
  if( minSet <= 0 )
  {
    minSet = 0;
    maxSet = 1;
  }
  if( maxSet > NUM_SETS_FIXED_FILTERS - 1 )
  {
    minSet = NUM_SETS_FIXED_FILTERS - 2;
    maxSet = NUM_SETS_FIXED_FILTERS - 1;
  }
}
}   // namespace

// ====================================================================================================================
// set-up
// ====================================================================================================================

struct GpuAlf::Shared
{
  gpu::Context& ctx;
  gpu::Kernel   ciClass, fixClass, class12, fix0, fix1, fixResi, gauss, luma, chClass, chroma, lfDs, lfCccm;
  gpu::Kernel   saoBif, saoApply, saoPad, offsetRefine;
  // the fixed-filter tables (AlfFixedFilters.h), the bilateral filter's tables
  gpu::Buffer   tables;
  VkDeviceSize  classIdn9 = 0, coeff9 = 0, clip9 = 0, classIdn13 = 0, coeff13 = 0, clip13 = 0, bif = 0;

  explicit Shared( gpu::Context& c ) : ctx( c )
  {
#define GPU_KERNEL( k, name, push ) k.create( ctx, spv_##name, sizeof( spv_##name ), sizeof( push ), #name )
    GPU_KERNEL( ciClass, alf_ciclass, PushCiClass );
    GPU_KERNEL( fixClass, alf_fixclass, PushFixClass );
    GPU_KERNEL( class12, alf_class12, PushClass12 );
    GPU_KERNEL( fix0, alf_fix0, PushFixFilter );
    GPU_KERNEL( fix1, alf_fix1, PushFixFilter );
    GPU_KERNEL( fixResi, alf_fixresi, PushFixFilter );
    GPU_KERNEL( gauss, alf_gauss, PushGauss );
    GPU_KERNEL( luma, alf_luma, PushLuma );
    GPU_KERNEL( chClass, alf_chclass, PushChClass );
    GPU_KERNEL( chroma, alf_chroma, PushChroma );
    GPU_KERNEL( lfDs, lfcccm_ds, PushLfDs );
    GPU_KERNEL( lfCccm, lfcccm, PushLfCccm );
    GPU_KERNEL( saoBif, sao_bif, PushSaoBif );
    GPU_KERNEL( saoApply, sao_apply, PushSaoApply );
    GPU_KERNEL( saoPad, sao_pad, PushSaoPad );
    GPU_KERNEL( offsetRefine, offset_refine, PushOffsetRefine );
#undef GPU_KERNEL

    const AdaptiveLoopFilter::FixedFilterTables t = AdaptiveLoopFilter::fixedFilterTables();
    const VkDeviceSize classIdnBytes = sizeof( short ) * NUM_SETS_FIXED_FILTERS * NUM_CLASSES_FIX;
    const VkDeviceSize coeff9Bytes = sizeof( short ) * NUM_SETS_FIXED_FILTERS * NUM_FIXED_FILTERS * FIX_FILTER_NUM_COEFF_9_DB_9;
    const VkDeviceSize coeff13Bytes = sizeof( short ) * NUM_SETS_FIXED_FILTERS * NUM_FIXED_FILTERS * FIX_FILTER_NUM_COEFF_13_DB_9;
    VkDeviceSize       bytes        = 0;
    auto               place        = [&bytes]( VkDeviceSize& offset, VkDeviceSize size ) {
      offset = bytes;
      bytes += align256( size );
    };
    place( classIdn9, classIdnBytes );
    place( coeff9, coeff9Bytes );
    place( clip9, coeff9Bytes );
    place( classIdn13, classIdnBytes );
    place( coeff13, coeff13Bytes );
    place( clip13, coeff13Bytes );
    place( bif, BIF_TABLE_BYTES );
    gpu::Buffer staging;
    staging.create( ctx, bytes, gpu::Buffer::UPLOAD );
    uint8_t* p = (uint8_t*) staging.mapped;
    memcpy( p + classIdn9, t.classIdn9Db9, classIdnBytes );
    memcpy( p + coeff9, t.coeff9Db9, coeff9Bytes );
    memcpy( p + clip9, t.clip9Db9, coeff9Bytes );
    memcpy( p + classIdn13, t.classIdn13Db9, classIdnBytes );
    memcpy( p + coeff13, t.coeff13Db9, coeff13Bytes );
    memcpy( p + clip13, t.clip13Db9, coeff13Bytes );
    std::unique_ptr<BilateralFilter> bf( new BilateralFilter );   // BilateralFilter::create derives the LUTs
    bf->create();
    memcpy( p + bif, bf->m_lut, BIF_LUT_BYTES );
    memcpy( p + bif + BIF_LUT_BYTES, bf->m_lutChroma, BIF_LUT_BYTES );
    memcpy( p + bif + 2 * BIF_LUT_BYTES, bf->m_tuMADFactor, 2 * 16 );
    memcpy( p + bif + 2 * BIF_LUT_BYTES + 2 * 16, bf->m_tuMADFactorChroma, 2 * 16 );
    static_assert( sizeof( bf->m_lut ) == BIF_LUT_BYTES && sizeof( bf->m_tuMADFactor ) == 2 * 16, "sao_bif.comp" );
    tables.create( ctx, bytes, gpu::Buffer::DEVICE );
    gpu::Stream s( ctx, 0 );
    s.begin();
    s.copy( staging, 0, tables, 0, bytes );
    s.submitAndWait();
    staging.destroy( ctx );
  }
  ~Shared() { tables.destroy( ctx ); }

  /// the 9x9 fixed-filter tables of a QP-dependent filter set
  void tables9( PushFixFilter& f, int set ) const
  {
    f.classIdn = tables.addr + classIdn9 + sizeof( short ) * NUM_CLASSES_FIX * set;
    f.coeff    = tables.addr + coeff9 + sizeof( short ) * NUM_FIXED_FILTERS * FIX_FILTER_NUM_COEFF_9_DB_9 * set;
    f.clip     = tables.addr + clip9 + sizeof( short ) * NUM_FIXED_FILTERS * FIX_FILTER_NUM_COEFF_9_DB_9 * set;
  }
  void tables13( PushFixFilter& f, int set ) const
  {
    f.classIdn = tables.addr + classIdn13 + sizeof( short ) * NUM_CLASSES_FIX * set;
    f.coeff    = tables.addr + coeff13 + sizeof( short ) * NUM_FIXED_FILTERS * FIX_FILTER_NUM_COEFF_13_DB_9 * set;
    f.clip     = tables.addr + clip13 + sizeof( short ) * NUM_FIXED_FILTERS * FIX_FILTER_NUM_COEFF_13_DB_9 * set;
  }
};

static std::once_flag s_alfSharedOnce;
GpuAlf::Shared*       GpuAlf::s_shared = nullptr;   // destroyed by GpuAlf::releaseShared, before the device

const GpuAlf::Shared& GpuAlf::shared( gpu::Context& ctx )
{
  std::call_once( s_alfSharedOnce, [&ctx] { s_shared = new Shared( ctx ); } );
  return *s_shared;
}

// pictures compared with the CPU (GpuAlf::VERIFY), reported when the process ends
static std::atomic<long> s_verifiedPictures( 0 ), s_verifiedLfCccm( 0 ), s_verifiedSao( 0 ), s_failedPictures( 0 );
static struct VerifyReport
{
  ~VerifyReport()
  {
    if( s_verifiedPictures )
    {
      fprintf( stderr, "GPU loop filter verification: %ld pictures (%ld with SAO, %ld with LF-CCCM), %ld with differences\n",
               s_verifiedPictures.load(), s_verifiedSao.load(), s_verifiedLfCccm.load(), s_failedPictures.load() );
    }
  }
} s_verifyReport;

/// the GpuAlf objects of the process, used by one picture at a time; destroyed by GpuAlf::releaseShared
struct GpuAlfPool
{
  static constexpr size_t              maxSize = 2;   ///< a picture holds one for a few ms, decoding it takes far longer
  std::mutex                           mutex;
  std::condition_variable              freed;
  std::vector<std::unique_ptr<GpuAlf>> all;
  std::vector<GpuAlf*>                 available;

  static GpuAlfPool& get()
  {
    static GpuAlfPool* pool = new GpuAlfPool;
    return *pool;
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock( mutex );
    available.clear();
    all.clear();
  }

  GpuAlf* acquire( gpu::Context& ctx, int mode )
  {
    std::unique_lock<std::mutex> lock( mutex );
    for( ;; )
    {
      if( !available.empty() )
      {
        GpuAlf* alf = available.back();
        available.pop_back();
        alf->m_mode = mode;
        return alf;
      }
      if( all.size() < maxSize )
      {
        all.emplace_back( new GpuAlf( ctx, GpuAlf::shared( ctx ), mode ) );
        return all.back().get();
      }
      freed.wait( lock );
    }
  }

  void release( GpuAlf* alf )
  {
    {
      std::lock_guard<std::mutex> lock( mutex );
      available.push_back( alf );
    }
    freed.notify_one();
  }
};

void GpuAlf::prefetch( int mode, int width, int height, int ctuSize )
{
  gpu::Context::prefetch( [=]( gpu::Context& ctx ) {
    shared( ctx );
    if( width > 0 )
    {
      // the buffers of the first picture
      GpuAlfPool& pool = GpuAlfPool::get();
      GpuAlf*     alf  = pool.acquire( ctx, mode );
      alf->layout( width, height, ctuSize );
      pool.release( alf );
    }
  } );
}

bool GpuAlf::ready()
{
  return gpu::Context::prefetchDone();
}

void GpuAlf::releaseShared()
{
  GpuAlfPool::get().clear();
  delete s_shared;   // its tables and kernels
  s_shared = nullptr;
}

GpuAlf::GpuAlf( gpu::Context& ctx, const Shared& shared, int mode ) : m_ctx( ctx ), m_k( shared ), m_mode( mode )
{
  m_stream.reset( new gpu::Stream( ctx ) );
}

GpuAlf::~GpuAlf()
{
  m_stream.reset();
  for( gpu::Buffer* b: { &m_in, &m_inUpload, &m_inDownload, &m_bifBlocks, &m_bifBlocksUpload, &m_work, &m_outDev,
                         &m_download, &m_workDownload, &m_saoDownload } )
  {
    b->destroy( m_ctx );
  }
}

GpuAlfHook* GpuAlfHook::create( int mode )
{
  gpu::Context*         ctx = gpu::Context::get();
  static std::once_flag reported;
  if( !ctx )
  {
    std::call_once( reported, [] {
      fprintf( stderr, "GPU loop filter not available: %s\n", gpu::Context::lastError().c_str() );
    } );
    return nullptr;
  }
  try
  {
    GpuAlf::shared( *ctx );
    std::call_once( reported, [ctx] { fprintf( stderr, "GPU loop filter: %s\n", ctx->deviceName() ); } );
    return new GpuAlfHook( *ctx, mode );
  }
  catch( const std::exception& e )
  {
    std::call_once( reported, [&e] { fprintf( stderr, "GPU loop filter not available: %s\n", e.what() ); } );
    return nullptr;
  }
}

bool GpuAlfHook::process( AdaptiveLoopFilter& alf, CodingStructure& cs )
{
  GpuAlfPool& pool = GpuAlfPool::get();
  GpuAlf*       gpu    = pool.acquire( m_ctx, m_mode );
  GpuSaoJob*    sao    = m_sao;
  GpuLfCccmJob* lfCccm = m_lfCccm;
  m_sao                = nullptr;
  m_lfCccm             = nullptr;
  bool          done;
  try
  {
    done = gpu->process( alf, cs, sao, lfCccm );
  }
  catch( ... )
  {
    pool.release( gpu );
    throw;
  }
  pool.release( gpu );
  return done;
}

// ====================================================================================================================
// pictures
// ====================================================================================================================

bool GpuAlf::accept( const AdaptiveLoopFilter& alf, const CodingStructure& cs ) const
{
  // what the GPU version does not implement: virtual boundaries, more than one slice or tile, sub-pictures,
  // per-CU QP classification, other chroma formats
  if( cs.picHeader->getVirtualBoundariesPresentFlag() || cs.pps->getNumTiles() != 1 || cs.pps->getNumSubPics() > 1
      || cs.slice->getCuQpDeltaSubdiv() || cs.area.chromaFormat != CHROMA_420 )
  {
    return false;
  }
  const PreCalcValues& pcv = *cs.pcv;
  if( pcv.maxCUWidth != pcv.maxCUHeight || ( pcv.maxCUWidth & ( pcv.maxCUWidth - 1 ) ) || ( pcv.lumaWidth & 7 )
      || ( pcv.lumaHeight & 7 ) )
  {
    return false;
  }
  for( int y = 0; y < pcv.lumaHeight; y += pcv.maxCUHeight )
  {
    for( int x = 0; x < pcv.lumaWidth; x += pcv.maxCUWidth )
    {
      if( cs.getCU( Position( x, y ), CHANNEL_TYPE_LUMA )->slice != cs.slice )
      {
        return false;
      }
    }
  }
  // the filter types of the APSs: the residual luma filters and the 9x9 chroma filters
  Slice& slice = *cs.slice;
  APS**  aps   = slice.getAlfAPSs();
  for( int i = 0; i < slice.getTileGroupNumAps(); i++ )
  {
    const APS* a = aps[slice.getTileGroupApsIdLuma()[i]];
    if( !a )
    {
      return false;
    }
    const int type = const_cast<APS*>( a )->getAlfAPSParam().filterType[CHANNEL_TYPE_LUMA];
    if( type != ALF_FILTER_13_EXT_DB_RESI_DIRECT && type != ALF_FILTER_13_EXT_DB_RESI )
    {
      return false;
    }
  }
  if( slice.getTileGroupAlfEnabledFlag( COMPONENT_Cb ) || slice.getTileGroupAlfEnabledFlag( COMPONENT_Cr ) )
  {
    APS* a = aps[slice.getTileGroupApsIdChroma()];
    if( !a )
    {
      return false;
    }
    const int type = a->getAlfAPSParam().filterType[CHANNEL_TYPE_CHROMA];
    if( type != ALF_FILTER_9 && type != ALF_FILTER_9_NO_FIX )
    {
      return false;
    }
  }
  return true;
}

void GpuAlf::stateBefore( AdaptiveLoopFilter& alf, CodingStructure& cs ) const
{
  // AdaptiveLoopFilter::ALFProcess up to the filtering of the CTUs (the path without virtual boundaries)
  alf.m_clpRngs = cs.slice->getClpRngs();
  for( int compIdx = 0; compIdx < MAX_NUM_COMPONENT; compIdx++ )
  {
    alf.m_ctuEnableFlag[compIdx]  = cs.picture->getAlfCtuEnableFlag( compIdx );
    alf.m_ctuAlternative[compIdx] = cs.picture->getAlfCtuAlternativeData( compIdx );
  }
#if JVET_AM0209_CHROMA_ALF_CCALF_REUSE_CTU
  for( int compIdx = 1; compIdx < MAX_NUM_COMPONENT; compIdx++ )
  {
    ComponentID compID = (ComponentID) compIdx;
    if( cs.slice->getTileGroupAlfEnabledFlag( compID ) && cs.slice->getTileGroupAlfReuseFlag( compID ) )
    {
      int apsIdxChroma = cs.slice->getTileGroupApsIdChroma();
      memcpy( alf.m_ctuEnableFlag[compID], alf.m_pictureControlInfo[apsIdxChroma].bufAlfFlag[compID].data(),
              sizeof( uint8_t ) * alf.m_numCTUsInPic );
      std::copy_n( alf.m_pictureControlInfo[apsIdxChroma].bufAlfFilter[compID].data(), alf.m_numCTUsInPic,
                   alf.m_ctuAlternative[compID] );
    }
    if( cs.slice->m_ccAlfFilterParam.ccAlfFilterEnabled[compIdx - 1] && cs.slice->getTileGroupCcalfReuseFlag( compID ) )
    {
      int apsIdxCcalf = compID == COMPONENT_Cb ? cs.slice->getTileGroupCcAlfCbApsId() : cs.slice->getTileGroupCcAlfCrApsId();
      memcpy( alf.m_ccAlfFilterControl[compIdx - 1], alf.m_pictureControlInfo[apsIdxCcalf].bufCcalfFilter[compIdx].data(),
              sizeof( uint8_t ) * alf.m_numCTUsInPic );
    }
  }
#endif
  // the padding of the inputs of the kernels (copiesBefore: the copies of the picture)
  PelUnitBuf tmpYuvBeforeDb = alf.m_tempBufBeforeDb.getBuf( cs.area );
  alf.mirroredPaddingForAlf( cs, tmpYuvBeforeDb, NUM_DB_PAD, true, true );
  PelUnitBuf tmpYuvResi = alf.m_tempBufResi.getBuf( cs.area );
  alf.mirroredPaddingForAlf( cs, tmpYuvResi, MAX_FILTER_LENGTH_FIXED >> 1, true, false );
  alf.m_isFixedFilterPaddedPerCtu = cs.picHeader->getVirtualBoundariesPresentFlag() || cs.slice->getCuQpDeltaSubdiv();
  memset( alf.m_ctuEnableOnlineLumaFlag, 0, sizeof( uint8_t ) * alf.m_numCTUsInPic );
  memset( alf.m_ctuPadFlag, 0, sizeof( uint8_t ) * alf.m_numCTUsInPic );
  alf.setAlfScaleMode( cs.sps->getAlfScaleMode() );
  if( !alf.m_alfScalePrevExternal )
  {
    alf.resetAlfScalePrev( *cs.slice );
    alf.getAlfScalePrev( cs, *cs.slice );
  }

  // the CTU loop: the APSs of the slice and the residual scales of the luma filters
  const PreCalcValues& pcv               = *cs.pcv;
  short*               alfCtuFilterIndex = nullptr;
  uint32_t             lastSliceIdx      = 0xFFFFFFFF;
  int                  ctuIdx            = 0;
  for( int yPos = 0; yPos < pcv.lumaHeight; yPos += pcv.maxCUHeight )
  {
    for( int xPos = 0; xPos < pcv.lumaWidth; xPos += pcv.maxCUWidth, ctuIdx++ )
    {
      const CodingUnit* cu = cs.getCU( Position( xPos, yPos ), CHANNEL_TYPE_LUMA );
      if( !cu->slice->getTileGroupAlfEnabledFlag( COMPONENT_Y ) && !cu->slice->getTileGroupAlfEnabledFlag( COMPONENT_Cb )
          && !cu->slice->getTileGroupAlfEnabledFlag( COMPONENT_Cr ) )
      {
        continue;
      }
      if( ctuIdx == 0 || lastSliceIdx != cu->slice->getSliceID() || alfCtuFilterIndex == nullptr )
      {
        cs.slice = cu->slice;
        alf.reconstructCoeffAPSs( cs, true,
                                  cu->slice->getTileGroupAlfEnabledFlag( COMPONENT_Cb )
                                    || cu->slice->getTileGroupAlfEnabledFlag( COMPONENT_Cr ),
                                  false );
        alfCtuFilterIndex      = cu->slice->getPic()->getAlfCtbFilterIndex();
        alf.m_ccAlfFilterParam = cu->slice->m_ccAlfFilterParam;
      }
      lastSliceIdx = cu->slice->getSliceID();
      const short filterSetIndex = alfCtuFilterIndex[ctuIdx];
      if( alf.m_ctuEnableFlag[COMPONENT_Y][ctuIdx] && filterSetIndex >= NUM_FIXED_FILTER_SETS && cs.sps->getAlfScaleMode() )
      {
        ScaleAlf& scale =
          cs.slice->getAlfScale( filterSetIndex - NUM_FIXED_FILTER_SETS, alf.m_ctuAlternative[COMPONENT_Y][ctuIdx] );
        if( !scale.initMinMaxDone )
        {
          scale.setMinMax( cs.slice->getLumaPelMin(), cs.slice->getLumaPelMax() );
          scale.setGroupSize( scale.groupShift );
          scale.fillIdxCorr();
        }
      }
    }
  }
}

void GpuAlf::copiesBefore( AdaptiveLoopFilter& alf, CodingStructure& cs ) const
{
  // ALFProcess's copies of the picture before ALF: the kernels' input, later stages read some of them
  if( cs.slice->getAlfScaleChroma( COMPONENT_Cb ) || cs.slice->getAlfScaleChroma( COMPONENT_Cr ) )
  {
    alf.m_tempBuf3.copyFrom( cs.getRecoBuf(), false, true );
  }
  PelUnitBuf recYuv = cs.getRecoBuf();
  alf.m_tempBuf.copyFrom( recYuv );
  PelUnitBuf tmpYuv = alf.m_tempBuf.getBuf( cs.area );
  alf.mirroredPaddingForAlf( cs, tmpYuv, MAX_FILTER_LENGTH_FIXED >> 1, true, true );
  alf.m_tempBufSAO.copyFrom( recYuv );
  PelUnitBuf tmpYuvSAO = alf.m_tempBufSAO.getBuf( cs.area );
  alf.mirroredPaddingForAlf( cs, tmpYuvSAO, MAX_FILTER_LENGTH_FIXED >> 1, false, true );
}

void GpuAlf::stateAfter( AdaptiveLoopFilter& alf, CodingStructure& cs ) const
{
  // AdaptiveLoopFilter::ALFProcess after the filtering of the CTUs (alfAddCorrectChroma is done by the kernels)
  if( cs.slice->getUseAlfScale() && cs.slice->getTileGroupAlfEnabledFlag( COMPONENT_Y ) && !alf.m_alfScalePrevExternal )
  {
    alf.storeAlfScalePrev( cs, *cs.slice );
  }
#if JVET_AM0209_CHROMA_ALF_CCALF_REUSE_CTU
  for( int compIdx = 1; compIdx < MAX_NUM_COMPONENT; compIdx++ )
  {
    ComponentID compID = (ComponentID) compIdx;
    if( cs.slice->getTileGroupAlfEnabledFlag( compID ) && !cs.slice->getTileGroupAlfReuseFlag( compID ) )
    {
      int apsIdxChroma = cs.slice->getTileGroupApsIdChroma();
      memcpy( alf.m_pictureControlInfo[apsIdxChroma].bufAlfFlag[compID].data(), alf.m_ctuEnableFlag[compID],
              sizeof( uint8_t ) * alf.m_numCTUsInPic );
      memcpy( alf.m_pictureControlInfo[apsIdxChroma].bufAlfFilter[compID].data(), alf.m_ctuAlternative[compID],
              sizeof( uint8_t ) * alf.m_numCTUsInPic );
      alf.m_pictureControlInfo[apsIdxChroma].alfTemporalLayer[compID] = cs.slice->getTLayer();
    }
  }
  for( int compIdx = 1; compIdx < MAX_NUM_COMPONENT; compIdx++ )
  {
    ComponentID compID = (ComponentID) compIdx;
    if( cs.slice->m_ccAlfFilterParam.ccAlfFilterEnabled[compIdx - 1] && !cs.slice->getTileGroupCcalfReuseFlag( compID ) )
    {
      int apsIdxCcalf = compID == COMPONENT_Cb ? cs.slice->getTileGroupCcAlfCbApsId() : cs.slice->getTileGroupCcAlfCrApsId();
      memcpy( alf.m_pictureControlInfo[apsIdxCcalf].bufCcalfFilter[compIdx].data(), alf.m_ccAlfFilterControl[compIdx - 1],
              sizeof( uint8_t ) * alf.m_numCTUsInPic );
      alf.m_pictureControlInfo[apsIdxCcalf].ccalfTemporalLayer[compID] = cs.slice->getTLayer();
    }
  }
#endif
}

bool GpuAlf::gather( const AdaptiveLoopFilter& alf, const CodingStructure& cs )
{
  Params& p = m_params;
  p.ctuInfo.assign( 4 * m_numCtus, 0 );
  p.lumaSets.clear();
  p.chroma.assign( MAX_NUM_ALF_ALTERNATIVES_CHROMA * CHROMA_STRIDE, 0 );
  p.luma = p.lumaFixed = p.chromaFixed = false;
  for( int c = 0; c < 2; c++ )
  {
    p.ccalfCoeff[c].assign( MAX_NUM_CC_ALF_FILTERS * CCALF_STRIDE, 0 );
    p.chromaAlf[c] = p.ccalf[c] = false;
    p.ccalfPrec[c] = p.scaleCorr[c] = 0;
  }

  const Slice& slice = *cs.slice;
  if( !slice.getTileGroupAlfEnabledFlag( COMPONENT_Y ) && !slice.getTileGroupAlfEnabledFlag( COMPONENT_Cb )
      && !slice.getTileGroupAlfEnabledFlag( COMPONENT_Cr ) )
  {
    return true;   // ALFProcess skips every CTU
  }
  const short*        filterSet = cs.picture->getAlfCtbFilterIndex();
  const bool          scaling   = cs.sps->getAlfScaleMode() != 0;
  std::map<int, int>  setOfKey;
  for( int ctu = 0; ctu < m_numCtus; ctu++ )
  {
    if( alf.m_ctuEnableFlag[COMPONENT_Y][ctu] )
    {
      p.luma         = true;
      p.lumaFixed    = true;
      const int fsi  = filterSet[ctu];
      uint32_t  info = 0;
      if( fsi < NUM_FIXED_FILTER_SETS )
      {
        info = fsi == 0 ? 1 : 2;
      }
      else
      {
        const int aps  = fsi - NUM_FIXED_FILTER_SETS;
        const int alt  = alf.m_ctuAlternative[COMPONENT_Y][ctu];
        const int type = alf.m_filterTypeApsLuma[aps];
        const int key  = aps * MAX_NUM_ALF_ALTERNATIVES_LUMA + alt;
        auto      it   = setOfKey.find( key );
        if( it == setOfKey.end() )
        {
          it = setOfKey.emplace( key, (int) setOfKey.size() ).first;
          p.lumaSets.resize( p.lumaSets.size() + LUMA_SET_STRIDE, 0 );
          int16_t* s = &p.lumaSets[p.lumaSets.size() - LUMA_SET_STRIDE];
          for( int i = 0; i < MAX_NUM_ALF_CLASSES * NUM_COEFF; i++ )
          {
            s[i]            = alf.m_coeffApsLuma[aps][alt][i];
            s[SET_CLIP + i] = alf.m_clippApsLuma[aps][alt][i];
          }
          const ScaleAlf& scale = cs.slice->getAlfScale( aps, alt );
          if( scaling && scale.idxCorr.size() < MAX_NUM_ALF_CLASSES )
          {
            return false;   // alfAddCorrect would read beyond the scales
          }
          for( int cls = 0; cls < MAX_NUM_ALF_CLASSES; cls++ )
          {
            s[SET_SCALE + cls] = AdaptiveLoopFilter::m_SCALE_FACTOR[(int) alf.m_scaleIdxApsLuma[aps][alt][cls]];
            if( scaling )
            {
              const int idx = scale.idxCorr[cls];
              if( idx < 0 || idx >= (int) alf.m_scaleCorr.size() )
              {
                return false;   // getScaleCorrInt() fails
              }
              s[SET_CORR + cls] = alf.m_scaleCorr[idx];
            }
          }
          s[SET_SHIFT]   = alf.m_coeffBitsApsLuma[aps][alt] - 1 - ( scaling ? shiftPrecis : 0 ) + AdaptiveLoopFilter::m_SCALE_SHIFT;
          s[SET_SCALING] = scaling;
        }
        info = ( type == ALF_FILTER_13_EXT_DB_RESI_DIRECT ? 3 : 4 ) | ( alf.m_classifierIdxApsLuma[aps][alt] << 4 )
               | ( it->second << 8 );
      }
      p.ctuInfo[4 * ctu] = info;
    }
    for( int c = 0; c < 2; c++ )
    {
      const ComponentID comp = ComponentID( COMPONENT_Cb + c );
      uint32_t          info = 0;
      if( alf.m_ctuEnableFlag[comp][ctu] )
      {
        info |= 1 | ( alf.m_ctuAlternative[comp][ctu] << 4 );
        p.chromaAlf[c] = true;
      }
      if( slice.m_ccAlfFilterParam.ccAlfFilterEnabled[c] && alf.m_ccAlfFilterControl[c][ctu] )
      {
        info |= alf.m_ccAlfFilterControl[c][ctu] << 8;
        p.ccalf[c] = true;
      }
      p.ctuInfo[4 * ctu + 1 + c] = info;
    }
  }
  p.chromaFixed = alf.m_filterTypeApsChroma == ALF_FILTER_9;
  if( p.chromaAlf[0] || p.chromaAlf[1] )
  {
    // Only the alternatives of the slice's chroma APS are filled, because a CTU selects no other. Past them
    // m_chromaScaleIdxFinal holds no scale index, and the m_SCALE_FACTOR lookup would read out of bounds.
    APS*        chromaAps       = cs.slice->getAlfAPSs()[slice.getTileGroupApsIdChroma()];
    const int   numAlternatives = chromaAps->getAlfAPSParam().numAlternativesChroma;
    const char* scaleIdx        = &alf.m_chromaScaleIdxFinal[0][0];   // indexed by alternatives up to 7, as ALFProcess does
    for( int alt = 0; alt < numAlternatives; alt++ )
    {
      int16_t* s = &p.chroma[alt * CHROMA_STRIDE];
      for( int k = 0; k < MAX_NUM_ALF_CHROMA_COEFF; k++ )
      {
        s[k]                            = alf.m_chromaCoeffFinal[alt][k];
        s[MAX_NUM_ALF_CHROMA_COEFF + k] = alf.m_chromaClippFinal[alt][k];
      }
      s[2 * MAX_NUM_ALF_CHROMA_COEFF] = AdaptiveLoopFilter::m_SCALE_FACTOR[(int) scaleIdx[alt]];
    }
  }
  for( int c = 0; c < 2; c++ )
  {
    if( slice.m_ccAlfFilterParam.ccAlfFilterEnabled[c] )
    {
      for( int f = 0; f < MAX_NUM_CC_ALF_FILTERS; f++ )
      {
        for( int k = 0; k < MAX_NUM_CC_ALF_CHROMA_COEFF; k++ )
        {
          p.ccalfCoeff[c][f * CCALF_STRIDE + k] = alf.m_ccAlfFilterParam.ccAlfCoeff[c][f][k];
        }
      }
      p.ccalfPrec[c] = alf.m_ccAlfFilterParam.ccAlfCoeffPrec[c];
    }
  }
  // alfAddCorrectChroma
  const bool ccalfUsed[2] = { cs.slice->getTileGroupCcAlfCbEnabledFlag(), cs.slice->getTileGroupCcAlfCrEnabledFlag() };
  for( int c = 0; c < 2; c++ )
  {
    const ComponentID comp  = ComponentID( COMPONENT_Cb + c );
    const int         sBest = slice.getAlfScaleChroma( comp );
    if( sBest && ( slice.getTileGroupAlfEnabledFlag( comp ) || ccalfUsed[c] ) )
    {
      p.scaleCorr[c] = alf.m_scaleCorrChroma[sBest];
    }
  }
  return true;
}

bool GpuAlf::gatherLfCccm( const CodingStructure& cs )
{
  Params& p = m_params;
  p.lfCccm  = false;
  p.lfCtus.clear();
  p.lfWindows = 0;
  // the kernel implements the decimal and matrix precision of bit depths up to 10 (DECIM_BITS, solve3)
  const Slice& slice     = *cs.slice;
  const int    impFactor = slice.getLfCccmImpEnabledFlag() ? slice.getLfCccmImpFactorIdx() : 0;
  if( cs.sps->getBitDepth( CHANNEL_TYPE_LUMA ) > 10 || cs.sps->getBitDepth( CHANNEL_TYPE_CHROMA ) > 10 || impFactor < 0
      || impFactor > 3 )
  {
    return false;
  }
  // LoopFilterCccm::m_lfCccmWindowSizes, in chroma samples
  static const int winW[8] = { 4, 8, 2, 8, 16, 32, 64, 128 };
  static const int winH[8] = { 4, 2, 8, 8, 16, 32, 64, 128 };
  const int        ctuSizeC = m_ctuSize / 2, wC = m_width / 2, hC = m_height / 2;
  for( int ctu = 0; ctu < m_numCtus; ctu++ )
  {
    if( !slice.m_lfCccmEnabled.at( ctu ) )
    {
      continue;
    }
    const int type = slice.m_lfCccmModelType.at( ctu );
    const int win  = slice.m_lfCccmWindowSizeIndex.at( ctu );
    if( type < 0 || type > 7 || win < 0 || win > 7 )
    {
      return false;
    }
    const int px = ( ctu % m_ctusPerRow ) * ctuSizeC, py = ( ctu / m_ctusPerRow ) * ctuSizeC;
    const int pw = std::min( ctuSizeC, wC - px ), ph = std::min( ctuSizeC, hC - py );
    p.lfCtus.insert( p.lfCtus.end(),
                     { uint32_t( ctu ), uint32_t( type | ( win << 8 ) ), uint32_t( p.lfWindows ), 0u } );
    p.lfWindows += ( ( pw + winW[win] - 1 ) / winW[win] ) * ( ( ph + winH[win] - 1 ) / winH[win] );
  }
  p.lfCccm = true;
  return true;
}

bool GpuAlf::gatherSao( CodingStructure& cs, const GpuSaoJob& job, bool reconstruct )
{
  SampleAdaptiveOffset& sao = *job.sao;
  Params&    p         = m_params;
  const SPS& sps       = *cs.sps;
  const PPS& pps       = *cs.pps;
  const bool saoOn     = sps.getSAOEnabledFlag();
  const bool bifLuma   = pps.getUseBIF();
  const bool bifChroma = pps.getUseChromaBIF();
  const bool ccsaoOn   = sps.getCCSAOEnabledFlag();
  p.sao                = false;
  p.offsetRefinement   = job.offsetRefinement;
  // the offset steps, the bilateral filter and the CCSAO thresholds of bit depths up to 10
  if( ( !saoOn && !bifLuma && !bifChroma && !ccsaoOn && !job.offsetRefinement )
      || sps.getBitDepth( CHANNEL_TYPE_LUMA ) > 10 || sps.getBitDepth( CHANNEL_TYPE_CHROMA ) > 10 )
  {
    return false;
  }

  // CCSAO (CCSAOProcess): the sets of each component and the set of each CTU
  p.saoInfo.assign( 4 * m_numCtus, 0 );
  p.ccsao.assign( 3 * MAX_CCSAO_SET_NUM * CCSAO_STRIDE, 0 );
  if( ccsaoOn )
  {
    const CcSaoComParam& cc = cs.slice->m_ccSaoComParam;
    for( int comp = 0; comp < 3; comp++ )
    {
      if( !cc.enabled[comp] )
      {
        continue;
      }
      for( int s = 0; s < MAX_CCSAO_SET_NUM; s++ )
      {
        int16_t* d = &p.ccsao[( comp * MAX_CCSAO_SET_NUM + s ) * CCSAO_STRIDE];
        int      classes;
        if( cc.setType[comp][s] == CCSAO_SET_TYPE_EDGE )
        {
          // offsetBlockCcSaoNoClipEdge
          const int edgeDir = cc.candPos[comp][s][COMPONENT_Y], edgeCmp = cc.candPos[comp][s][COMPONENT_Cb];
          const int bandIdc = cc.bandNum[comp][s][COMPONENT_Y], edgeIdc = cc.bandNum[comp][s][COMPONENT_Cr];
          const int edgeThr = cc.bandNum[comp][s][COMPONENT_Cb];
          if( edgeDir >= MAX_CCSAO_EDGE_DIR || edgeCmp > 2 || bandIdc >= MAX_CCSAO_BAND_IDC || edgeIdc >= MAX_CCSAO_EDGE_IDC
              || edgeThr >= MAX_CCSAO_EDGE_THR )
          {
            return false;
          }
          d[0]    = 1;
          d[1]    = edgeDir;
          d[2]    = edgeCmp;
          d[3]    = g_ccSaoBandTab[bandIdc][0];
          d[4]    = g_ccSaoBandTab[bandIdc][1];
          d[5]    = edgeIdc;
          d[6]    = g_ccSaoEdgeThr[edgeIdc][edgeThr];   // << max(0, bit depth - 10)
          classes = g_ccSaoBandTab[bandIdc][1] * g_ccSaoEdgeNum[edgeIdc][0];
        }
        else
        {
          // offsetBlockCcSaoNoClip
          d[0]    = 0;
          d[1]    = cc.candPos[comp][s][COMPONENT_Y];
          d[2]    = cc.bandNum[comp][s][COMPONENT_Y];
          d[3]    = cc.bandNum[comp][s][COMPONENT_Cb];
          d[4]    = cc.bandNum[comp][s][COMPONENT_Cr];
          classes = d[2] * d[3] * d[4];
          if( d[1] >= MAX_CCSAO_CAND_POS_Y )
          {
            return false;
          }
        }
        if( classes > MAX_CCSAO_CLASS_NUM )
        {
          return false;   // the CPU would read beyond the offsets
        }
        for( int i = 0; i < MAX_CCSAO_CLASS_NUM; i++ )
        {
          d[8 + i] = cc.offset[comp][s][i];
        }
      }
      const uint8_t* control = sao.m_ccSaoControl[comp];
      for( int ctu = 0; ctu < m_numCtus; ctu++ )
      {
        if( control[ctu] > MAX_CCSAO_SET_NUM )
        {
          return false;
        }
        p.saoInfo[4 * ctu + comp] |= uint32_t( control[ctu] ) << 4;
      }
    }
  }

  // the bilateral filter (SAOProcess): the transform blocks it filters
  p.bifBlocks.clear();
  p.bif[0] = p.bif[1] = p.bif[2] = false;
  p.bifRoundAdd   = BIF_ROUND_ADD >> pps.getBIFStrength();
  p.bifRoundShift = BIF_ROUND_SHIFT - pps.getBIFStrength();
  BilateralFilter& bf = sao.m_bilateralFilter;
  auto block = [&]( const TransformUnit& tu, ComponentID comp, int qp ) {
    const CompArea& a  = tu.blocks[comp];
    int             ws = a.width, hs = a.height;   // the size the strength derives from
    if( comp != COMPONENT_Y && tu.blocks[COMPONENT_Y].valid() )
    {
      ws = tu.blocks[COMPONENT_Y].width;
      hs = tu.blocks[COMPONENT_Y].height;
    }
    const int  inter  = tu.cu->predMode == MODE_INTER;
    const int  sizeIx = std::min( floorLog2( ws ), 7 ) * 8 + std::min( floorLog2( hs ), 7 );
    const int  factor = comp == COMPONENT_Y ? bf.m_tuSizeFactor[inter][sizeIx] : bf.m_tuSizeFactorChroma[inter][sizeIx];
    const int  sqp    = std::min( std::max( qp - 17, 0 ), 25 );
    p.bifBlocks.insert( p.bifBlocks.end(), { uint32_t( a.x | ( a.y << 16 ) ), uint32_t( a.width | ( a.height << 16 ) ),
                                             uint32_t( comp | ( inter << 2 ) | ( sqp << 3 ) | ( factor << 8 ) ), 0u } );
    p.bif[comp] = true;
  };
  if( bifLuma || bifChroma )
  {
    const PreCalcValues& pcv        = *cs.pcv;
    const bool           dualTree   = CS::isDualITree( cs );
    const ChannelType    chromaType = dualTree ? CH_C : CH_L;
    int                  ctu        = 0;
    for( int yPos = 0; yPos < pcv.lumaHeight; yPos += pcv.maxCUHeight )
    {
      for( int xPos = 0; xPos < pcv.lumaWidth; xPos += pcv.maxCUWidth, ctu++ )
      {
        const UnitArea area( cs.area.chromaFormat, Area( xPos, yPos, std::min<int>( pcv.maxCUWidth, pcv.lumaWidth - xPos ),
                                                         std::min<int>( pcv.maxCUHeight, pcv.lumaHeight - yPos ) ) );
        if( bifLuma && cs.picture->getBifParam( COMPONENT_Y ).ctuOn[ctu] )
        {
          for( auto& cu: cs.traverseCUs( CS::getArea( cs, area, CH_L ), CH_L ) )
          {
            for( auto& tu: CU::traverseTUs( cu ) )
            {
              if( bf.getApplyBIF( tu, COMPONENT_Y ) )
              {
                block( tu, COMPONENT_Y, tu.cu->qp + pps.getBIFQPOffset() );
              }
            }
          }
        }
        if( bifChroma && ( cs.picture->getBifParam( COMPONENT_Cb ).ctuOn[ctu] || cs.picture->getBifParam( COMPONENT_Cr ).ctuOn[ctu] ) )
        {
          for( auto& cu: cs.traverseCUs( CS::getArea( cs, area, chromaType ), chromaType ) )
          {
            if( !cu.Cb().valid() || !cu.Cr().valid() )
            {
              continue;
            }
            for( auto& tu: CU::traverseTUs( cu ) )
            {
              for( int comp = COMPONENT_Cb; comp <= COMPONENT_Cr; comp++ )
              {
                const ComponentID compID = ComponentID( comp );
                if( cs.picture->getBifParam( compID ).ctuOn[ctu] && bf.getApplyBIF( tu, compID ) )
                {
                  block( tu, compID, tu.cu->qp + pps.getChromaBIFQPOffset() );
                }
              }
            }
          }
        }
      }
    }
  }

  // SAO (SAOProcess): the type and offsets of each CTU, after resolving the merges
  p.saoOffsets.assign( m_numCtus * 3 * 32, 0 );
  if( saoOn )
  {
    if( reconstruct )
    {
      sao.xReconstructBlkSAOParams( cs, cs.picture->getSAO() );
    }
    const SAOBlkParam* prm = cs.picture->getSAO();
    for( int ctu = 0; ctu < m_numCtus; ctu++ )
    {
      for( int comp = 0; comp < 3; comp++ )
      {
        const SAOOffset& o = prm[ctu][comp];
        if( o.modeIdc == SAO_MODE_OFF )
        {
          continue;
        }
        p.saoInfo[4 * ctu + comp] |= uint32_t( o.typeIdc + 1 );   // EO 1..4, BO 5
        for( int i = 0; i < 32; i++ )
        {
          p.saoOffsets[( ctu * 3 + comp ) * 32 + i] = int16_t( o.offset[i] );
        }
      }
    }
  }
  p.sao = true;
  return true;
}

void GpuAlf::layout( int width, int height, int ctuSize )
{
  if( width == m_width && height == m_height && ctuSize == m_ctuSize )
  {
    return;
  }
  m_width      = width;
  m_height     = height;
  m_ctuSize    = ctuSize;
  m_ctusPerRow = ( m_width + m_ctuSize - 1 ) / m_ctuSize;
  m_numCtus    = m_ctusPerRow * ( ( m_height + m_ctuSize - 1 ) / m_ctuSize );
  m_wBlk       = m_width / 2;
  m_hBlk       = m_height / 2;
  m_wBlkC      = m_width / 4;
  m_hBlkC      = m_height / 4;
  const int wC = m_width / 2, hC = m_height / 2;

  auto plane = []( PlaneDesc& p, int w, int h, int pad, VkDeviceSize& offset ) {
    p.width  = w;
    p.height = h;
    p.pad    = pad;
    p.stride = ( w + 2 * pad + 7 ) & ~7;
    p.offset = offset;
    offset += align256( p.bytes() );
  };
  auto place = []( VkDeviceSize& offset, VkDeviceSize bytes, VkDeviceSize& total ) {
    offset = total;
    total += align256( bytes );
  };

  // the deblocked picture first and then the picture after SAO, so that either is left out of the upload
  VkDeviceSize in = 0;
  for( int c = 0; c < 3; c++ )
  {
    plane( m_inDeblocked[c], c ? wC : m_width, c ? hC : m_height, 0, in );
  }
  for( int c = 0; c < 3; c++ )
  {
    plane( m_inRec[c], c ? wC : m_width, c ? hC : m_height, 8, in );
  }
  for( int c = 0; c < 3; c++ )
  {
    plane( m_inDb[c], c ? wC : m_width, c ? hC : m_height, 8, in );
  }
  plane( m_inResi, m_width, m_height, 8, in );
  plane( m_inCi, m_width, m_height, 0, in );
  place( m_inCtuInfo, sizeof( uint32_t ) * 4 * m_numCtus, in );
  place( m_inLumaSets, sizeof( int16_t ) * LUMA_SET_STRIDE * MAX_LUMA_SETS, in );
  place( m_inChroma, sizeof( int16_t ) * CHROMA_STRIDE * MAX_NUM_ALF_ALTERNATIVES_CHROMA, in );
  place( m_inCcalf[0], sizeof( int16_t ) * CCALF_STRIDE * MAX_NUM_CC_ALF_FILTERS, in );
  place( m_inCcalf[1], sizeof( int16_t ) * CCALF_STRIDE * MAX_NUM_CC_ALF_FILTERS, in );
  place( m_inLfCtus, sizeof( uint32_t ) * 4 * m_numCtus, in );
  place( m_inSaoInfo, sizeof( uint32_t ) * 4 * m_numCtus, in );
  place( m_inSaoOffsets, sizeof( int16_t ) * m_numCtus * 3 * 32, in );
  place( m_inCcsao, sizeof( int16_t ) * 3 * MAX_CCSAO_SET_NUM * CCSAO_STRIDE, in );
  m_inBytes = in;
  m_in.ensure( m_ctx, in, gpu::Buffer::DEVICE );
  m_inUpload.ensure( m_ctx, in, gpu::Buffer::UPLOAD );
  if( m_mode == VERIFY )
  {
    m_inDownload.ensure( m_ctx, in, gpu::Buffer::DOWNLOAD );
  }

  const VkDeviceSize blocks = VkDeviceSize( m_wBlk ) * m_hBlk, blocksC = VkDeviceSize( m_wBlkC ) * m_hBlkC;
  VkDeviceSize       work   = 0;
  place( m_ciClass, blocks, work );
  place( m_fixClass0, blocks * 2, work );
  place( m_fixClass1, blocks * 2, work );
  for( int c = 0; c < 3; c++ )
  {
    place( m_mainClass[c], blocks * 2, work );
  }
  plane( m_fix0, m_width, m_height, 0, work );
  plane( m_fix1, m_width, m_height, 0, work );
  plane( m_fixResi, m_width, m_height, 0, work );
  plane( m_gauss, m_width, m_height, 0, work );
  plane( m_lapl, m_width, m_height, 0, work );
  for( int c = 0; c < 2; c++ )
  {
    place( m_chClass[c], blocksC * 2, work );
    plane( m_chFix[c], wC, hC, 0, work );
  }
  plane( m_lfDs, wC, hC, 1, work );
  for( int c = 0; c < 2; c++ )
  {
    plane( m_lfCorr[c], wC, hC, 0, work );
  }
  for( int c = 0; c < 3; c++ )
  {
    plane( m_bifOffsets[c], c ? wC : m_width, c ? hC : m_height, 0, work );
  }
  m_work.ensure( m_ctx, work, gpu::Buffer::DEVICE );
  if( m_mode == VERIFY )
  {
    m_workDownload.ensure( m_ctx, work, gpu::Buffer::DOWNLOAD );
  }

  VkDeviceSize out = 0;
  for( int c = 0; c < 3; c++ )
  {
    plane( m_out[c], c ? wC : m_width, c ? hC : m_height, 0, out );
  }
  m_outDev.ensure( m_ctx, out, gpu::Buffer::DEVICE );
  m_download.ensure( m_ctx, out, gpu::Buffer::DOWNLOAD );
  m_saoDownload.ensure( m_ctx, m_inRec[0].bytes(), gpu::Buffer::DOWNLOAD );
}

void GpuAlf::upload( const AdaptiveLoopFilter& alf, const CodingStructure& cs, const CPelUnitBuf& deblocked )
{
  uint8_t* base = (uint8_t*) m_inUpload.mapped;
  // a plane with its margin, from a buffer whose margin is at least as wide
  auto put = [base]( const PlaneDesc& p, const CPelBuf& src ) {
    int16_t* dst = (int16_t*) ( base + p.offset );
    for( int y = -p.pad; y < p.height + p.pad; y++ )
    {
      memcpy( dst + p.origin() + y * p.stride - p.pad, src.buf + y * src.stride - p.pad,
              sizeof( int16_t ) * ( p.width + 2 * p.pad ) );
    }
  };
  const Params& p = m_params;
  for( int c = 0; c < 3; c++ )
  {
    const ComponentID comp = ComponentID( c );
    if( p.sao )
    {
      put( m_inDeblocked[c], deblocked.get( comp ) );
    }
    else
    {
      put( m_inRec[c], alf.m_tempBuf.get( comp ) );
    }
    put( m_inDb[c], alf.m_tempBufBeforeDb.get( comp ) );
  }
  put( m_inResi, alf.m_tempBufResi.get( COMPONENT_Y ) );
  put( m_inCi, alf.m_tempBufCodingInfo.get( COMPONENT_Y ) );

  memcpy( base + m_inCtuInfo, p.ctuInfo.data(), sizeof( uint32_t ) * p.ctuInfo.size() );
  memcpy( base + m_inLumaSets, p.lumaSets.data(), sizeof( int16_t ) * p.lumaSets.size() );
  memcpy( base + m_inChroma, p.chroma.data(), sizeof( int16_t ) * p.chroma.size() );
  for( int c = 0; c < 2; c++ )
  {
    memcpy( base + m_inCcalf[c], p.ccalfCoeff[c].data(), sizeof( int16_t ) * p.ccalfCoeff[c].size() );
  }
  memcpy( base + m_inLfCtus, p.lfCtus.data(), sizeof( uint32_t ) * p.lfCtus.size() );
  if( p.sao )
  {
    memcpy( base + m_inSaoInfo, p.saoInfo.data(), sizeof( uint32_t ) * p.saoInfo.size() );
    memcpy( base + m_inSaoOffsets, p.saoOffsets.data(), sizeof( int16_t ) * p.saoOffsets.size() );
    memcpy( base + m_inCcsao, p.ccsao.data(), sizeof( int16_t ) * p.ccsao.size() );
    if( !p.bifBlocks.empty() )
    {
      const VkDeviceSize bytes = sizeof( uint32_t ) * p.bifBlocks.size();
      m_bifBlocks.ensure( m_ctx, bytes, gpu::Buffer::DEVICE );
      m_bifBlocksUpload.ensure( m_ctx, bytes, gpu::Buffer::UPLOAD );
      memcpy( m_bifBlocksUpload.mapped, p.bifBlocks.data(), bytes );
    }
  }
}

void GpuAlf::run( const AdaptiveLoopFilter& alf, const CodingStructure& cs )
{
  auto gp = []( const gpu::Buffer& b, const PlaneDesc& p ) {
    GpuPlane g;
    g.addr   = b.addr + p.offset;
    g.origin = p.origin();
    g.stride = p.stride;
    return g;
  };
  const Params& prm      = m_params;
  const int     bitDepth = cs.sps->getBitDepth( CHANNEL_TYPE_LUMA );
  const int     bdChroma = cs.sps->getBitDepth( CHANNEL_TYPE_CHROMA );
  const ClpRng& clpRng   = cs.slice->clpRng( COMPONENT_Y );
  const bool    intra    = cs.slice->isIntra();
  const bool    adjust   = cs.sps->getAlfLumaFixedFilterAdjust();
  const int     fixedSet = cs.slice->getTileGroupAlfFixedFilterSetIdx( COMPONENT_Y );
  int           qpSetMin, qpSetMax;
  fixedFilterQpSets( cs.slice->getSliceQp(), qpSetMin, qpSetMax );
  const int qpSet        = fixedSet == 0 ? qpSetMin : qpSetMax;
  const int qpSetReverse = fixedSet == 0 ? qpSetMax : qpSetMin;
  const int wC = m_width / 2, hC = m_height / 2;

  gpu::Stream& s = *m_stream;
  s.begin();
  if( prm.sao )
  {
    // the deblocked picture instead of the one after SAO
    s.copy( m_inUpload, m_inDeblocked[0].offset, m_in, m_inDeblocked[0].offset, m_inRec[0].offset - m_inDeblocked[0].offset );
    s.copy( m_inUpload, m_inDb[0].offset, m_in, m_inDb[0].offset, m_inBytes - m_inDb[0].offset );
    s.copy( m_bifBlocksUpload, 0, m_bifBlocks, 0, sizeof( uint32_t ) * prm.bifBlocks.size() );
    for( int c = 0; c < 3; c++ )
    {
      if( prm.bif[c] )
      {
        s.fill( m_work, m_bifOffsets[c].offset, m_bifOffsets[c].bytes(), 0 );   // no offset outside the blocks
      }
    }
  }
  else
  {
    s.copy( m_inUpload, m_inRec[0].offset, m_in, m_inRec[0].offset, m_inBytes - m_inRec[0].offset );
  }
  if( prm.lfCccm )
  {
    for( int c = 0; c < 2; c++ )
    {
      s.fill( m_work, m_lfCorr[c].offset, m_lfCorr[c].bytes(), 0 );   // no correction outside the windows
    }
  }
  s.barrier();

  // the offset refinement after deblocking, SAO, the bilateral filter and CCSAO: the picture after SAO in m_inRec,
  // mirror-padded as ALFProcess pads it
  if( prm.sao )
  {
#if JVET_AK0121_LOOPFILTER_OFFSET_REFINEMENT
    if( prm.offsetRefinement )
    {
      PushOffsetRefine r;
      r.pic        = gp( m_in, m_inDeblocked[0] );
      r.before     = gp( m_in, m_inDb[0] );
      r.codingInfo = gp( m_in, m_inCi );
      r.width      = m_width;
      r.height     = m_height;
      r.bitDepth   = bitDepth;
      r.refineIdx  = cs.slice->getOffsetRefinementDbfIdx();
      r.useDiffTh  = !cs.sps->getInloopOffsetRefineFlag() && !intra;
      r.paramA     = cs.sps->getInloopOffsetRefineFunc() ? 5 : 6;
      r.pad0 = r.pad1 = 0;
      s.dispatch( m_k.offsetRefine, &r, divUp( m_width, 16 ), divUp( m_height, 8 ) );
      s.barrier();
    }
#endif
    if( !prm.bifBlocks.empty() )
    {
      PushSaoBif bif;
      memset( &bif, 0, sizeof( bif ) );
      for( int c = 0; c < 3; c++ )
      {
        bif.src[c] = gp( m_in, m_inDeblocked[c] );
        bif.dst[c] = gp( m_work, m_bifOffsets[c] );
      }
      bif.blocks     = m_bifBlocks.addr;
      bif.tables     = m_k.tables.addr + m_k.bif;
      bif.numBlocks  = int( prm.bifBlocks.size() / 4 );
      bif.groupsX    = std::min( bif.numBlocks, 65535 );
      bif.width      = m_width;
      bif.height     = m_height;
      bif.ctuSize    = m_ctuSize;
      bif.roundAdd   = prm.bifRoundAdd;
      bif.roundShift = prm.bifRoundShift;
      s.dispatch( m_k.saoBif, &bif, bif.groupsX, divUp( bif.numBlocks, bif.groupsX ) );
      s.barrier();
    }
    for( int c = 0; c < 3; c++ )
    {
      const ComponentID comp = ComponentID( c );
      const ClpRng&     rng  = cs.slice->clpRng( comp );
      PushSaoApply      sa;
      memset( &sa, 0, sizeof( sa ) );
      sa.srcY        = gp( m_in, m_inDeblocked[0] );
      sa.srcCb       = gp( m_in, m_inDeblocked[1] );
      sa.srcCr       = gp( m_in, m_inDeblocked[2] );
      sa.bif         = prm.bif[c] ? gp( m_work, m_bifOffsets[c] ) : GpuPlane{ 0, 0, 0 };
      sa.dst         = gp( m_in, m_inRec[c] );
      sa.ctuInfo     = m_in.addr + m_inSaoInfo;
      sa.saoOffsets  = m_in.addr + m_inSaoOffsets;
      sa.ccsao       = m_in.addr + m_inCcsao;
      sa.comp        = c;
      sa.width       = m_width;
      sa.height      = m_height;
      sa.ctuSizeLog2 = log2Of( m_ctuSize );
      sa.ctusPerRow  = m_ctusPerRow;
      sa.bitDepth    = cs.sps->getBitDepth( toChannelType( comp ) );
      sa.outMin      = rng.min;
      sa.outMax      = rng.max;
      sa.ccsaoShift  = std::max( 0, sa.bitDepth - MAX_SAO_TRUNCATED_BITDEPTH );   // m_offsetStepLog2
      s.dispatch( m_k.saoApply, &sa, divUp( c ? wC : m_width, 16 ), divUp( c ? hC : m_height, 8 ) );
    }
    s.barrier();
    for( int c = 0; c < 3; c++ )
    {
      PushSaoPad pad;
      pad.plane  = gp( m_in, m_inRec[c] );
      pad.width  = c ? wC : m_width;
      pad.height = c ? hC : m_height;
      pad.pad    = MAX_FILTER_LENGTH_FIXED >> 1;
      pad.pad0   = 0;
      s.dispatch( m_k.saoPad, &pad, divUp( pad.width + 2 * pad.pad, 16 ), divUp( pad.height + 2 * pad.pad, 8 ) );
    }
    s.barrier();
  }

  // classes
  PushCiClass ciClass;
  ciClass.resi      = gp( m_in, m_inResi );
  ciClass.ci        = gp( m_in, m_inCi );
  ciClass.out       = m_work.addr + m_ciClass;
  ciClass.outStride = m_wBlk;
  ciClass.wBlk      = m_wBlk;
  ciClass.hBlk      = m_hBlk;
  ciClass.calcResi  = !intra;
  ciClass.bitDepth  = bitDepth;
  s.dispatch( m_k.ciClass, &ciClass, divUp( m_wBlk, 16 ), divUp( m_hBlk, 8 ) );
  s.barrier();

  PushFixClass fixClass;
  fixClass.rec        = gp( m_in, m_inRec[0] );
  fixClass.ciClass    = m_work.addr + m_ciClass;
  fixClass.fixClass0  = m_work.addr + m_fixClass0;
  fixClass.fixClass1  = m_work.addr + m_fixClass1;
  fixClass.mainClass0 = m_work.addr + m_mainClass[0];
  fixClass.blkStride  = m_wBlk;
  fixClass.wBlk       = m_wBlk;
  fixClass.hBlk       = m_hBlk;
  fixClass.bitDepth   = bitDepth;
  s.dispatch( m_k.fixClass, &fixClass, divUp( m_wBlk, 16 ), divUp( m_hBlk, 8 ) );

  PushClass12 class12;
  class12.rec        = gp( m_in, m_inRec[0] );
  class12.resi       = gp( m_in, m_inResi );
  class12.ciClass    = m_work.addr + m_ciClass;
  class12.mainClass1 = m_work.addr + m_mainClass[1];
  class12.mainClass2 = m_work.addr + m_mainClass[2];
  class12.blkStride  = m_wBlk;
  class12.wBlk       = m_wBlk;
  class12.hBlk       = m_hBlk;
  class12.bitDepth   = bitDepth;
  s.dispatch( m_k.class12, &class12, divUp( m_wBlk, 16 ), divUp( m_hBlk, 8 ) );

  for( int c = 0; c < 2; c++ )
  {
    PushChClass chClass;
    chClass.rec       = gp( m_in, m_inRec[1 + c] );
    chClass.fixClass  = m_work.addr + m_chClass[c];
    chClass.blkStride = m_wBlkC;
    chClass.wBlk      = m_wBlkC;
    chClass.hBlk      = m_hBlkC;
    chClass.bitDepth  = bdChroma;
    s.dispatch( m_k.chClass, &chClass, divUp( m_wBlkC, 16 ), divUp( m_hBlkC, 8 ) );
  }

  // LF-CCCM: the downsampled luma (LoopFilterCccm::lfCccmFillDownsampledLumaBuffers)
  const SPS& sps = *cs.sps;
  if( prm.lfCccm )
  {
    PushLfDs ds;
    ds.luma       = gp( m_in, m_inRec[0] );
    ds.ds         = gp( m_work, m_lfDs );
    ds.width      = wC;
    ds.height     = hC;
    ds.lumaOffset = 1 << ( bitDepth - 1 );
    ds.filterType = sps.getCclmVerCollocatedChromaFlag() && sps.getCclmHorCollocatedChromaFlag()     ? 2
                    : !sps.getCclmVerCollocatedChromaFlag() && !sps.getCclmHorCollocatedChromaFlag() ? 1
                                                                                                       : 0;
    s.dispatch( m_k.lfDs, &ds, divUp( wC + 2, 16 ), divUp( hC + 2, 8 ) );
  }
  s.barrier();

  // the fixed filters: AdaptiveLoopFilter::fixedFilterBlk with the coding information of the luma ALF
  PushFixFilter fix;
  memset( &fix, 0, sizeof( fix ) );
  fix.src = gp( m_in, m_inRec[0] );
  fix.db  = gp( m_in, m_inDb[0] );
  for( int i = 0; i < 4; i++ )
  {
    fix.clipValues[i] = alf.m_alfClippingValues[CHANNEL_TYPE_LUMA][i];
  }
  fix.ciClass    = m_work.addr + m_ciClass;
  fix.width      = m_width;
  fix.height     = m_height;
  fix.blkStride  = m_wBlk;
  fix.useBound   = !( !adjust && intra );
  fix.useResi    = !intra;
  fix.bsFactor   = intra ? 4 : 3;
  fix.resiFactor = intra ? 0 : ( 3 >> ( !adjust ? 1 : 0 ) );
  fix.offsetClip = 1 << ( clpRng.bd - 1 );
  fix.outMin     = clpRng.min;
  fix.outMax     = clpRng.max;
  auto tables9   = [this]( PushFixFilter& f, int set ) { m_k.tables9( f, set ); };

  PushFixFilter fix0 = fix;
  fix0.dst           = gp( m_work, m_fix0 );
  fix0.fixClass      = m_work.addr + m_fixClass0;
  tables9( fix0, qpSet );
  s.dispatch( m_k.fix0, &fix0, divUp( m_width, 16 ), divUp( m_height, 8 ) );

  // AdaptiveLoopFilter::fixedFilteringResi: the other QP set, coding information if adjusted
  PushFixFilter resi = fix;
  resi.src           = gp( m_in, m_inResi );
  resi.dst           = gp( m_work, m_fixResi );
  resi.fixClass      = m_work.addr + m_fixClass0;
  tables9( resi, qpSetReverse );
  resi.useBound = adjust;
  resi.useResi  = adjust && !intra;
  resi.outMin   = -clpRng.max;
  resi.outMax   = clpRng.max;
  s.dispatch( m_k.fixResi, &resi, divUp( m_width, 16 ), divUp( m_height, 8 ) );

  // Gaussian image before deblocking (z = 0) and Laplacian image (z = 1)
  const bool highRes = m_width > 1280 && m_height > 720;
  const int  bdUp    = std::max( 0, bitDepth - 10 );
  PushGauss  gauss;
  gauss.db    = gp( m_in, m_inDb[0] );
  gauss.rec   = gp( m_in, m_inRec[0] );
  gauss.ci    = gp( m_in, m_inCi );
  gauss.gauss = gp( m_work, m_gauss );
  gauss.lapl  = gp( m_work, m_lapl );
  for( int i = 0; i < 4; i++ )
  {
    gauss.clipValues[i] = alf.m_alfClippingValues[CHANNEL_TYPE_LUMA][i];
  }
  gauss.ciClass    = m_work.addr + m_ciClass;
  gauss.width      = m_width;
  gauss.height     = m_height;
  gauss.blkStride  = m_wBlk;
  gauss.useBound   = !( !adjust && intra );
  gauss.bsFactor   = intra ? 4 + 2 : 3 + 2;
  gauss.offsetClip = 1 << ( clpRng.bd - 1 );
  gauss.diffGauss  = 32 << bdUp;
  gauss.diffLapl   = ( highRes ? 128 : 96 ) << bdUp;
  gauss.diffLocal  = ( highRes ? 1024 : 96 ) << bdUp;
  gauss.outMin     = clpRng.min;
  gauss.outMax     = clpRng.max;
  s.dispatch( m_k.gauss, &gauss, divUp( m_width, 16 ), divUp( m_height, 8 ), 2 );

  // chroma fixed filter: 9x9 + 9x9 before deblocking, no coding information (deriveFixFilterResultsBlkChroma)
  for( int c = 0; c < 2; c++ )
  {
    const ComponentID comp     = ComponentID( COMPONENT_Cb + c );
    const ClpRng&     clpRngC  = cs.slice->clpRng( comp );
    PushFixFilter     chFix;
    memset( &chFix, 0, sizeof( chFix ) );
    chFix.src = gp( m_in, m_inRec[1 + c] );
    chFix.db  = gp( m_in, m_inDb[1 + c] );
    chFix.dst = gp( m_work, m_chFix[c] );
    for( int i = 0; i < 4; i++ )
    {
      chFix.clipValues[i] = alf.m_alfClippingValues[CHANNEL_TYPE_CHROMA][i];
    }
    chFix.fixClass  = m_work.addr + m_chClass[c];
    chFix.ciClass   = m_work.addr + m_ciClass;   // not read
    chFix.width     = wC;
    chFix.height    = hC;
    chFix.blkStride = m_wBlkC;
    chFix.outMin    = clpRngC.min;
    chFix.outMax    = clpRngC.max;
    int setMin, setMax;
    fixedFilterQpSets( cs.slice->getSliceQp() + cs.slice->getSliceChromaQpDelta( comp ), setMin, setMax );
    tables9( chFix, cs.slice->getTileGroupAlfFixedFilterSetIdx( comp ) == 0 ? setMin : setMax );
    s.dispatch( m_k.fix0, &chFix, divUp( wC, 16 ), divUp( hC, 8 ) );
  }

  // LF-CCCM: the corrections of the windows (LoopFilterCccm::lfCccmCtuProcess)
  if( prm.lfCccm && prm.lfWindows )
  {
    const Slice& slice = *cs.slice;
    PushLfCccm   lf;
    memset( &lf, 0, sizeof( lf ) );
    lf.ds             = gp( m_work, m_lfDs );
    lf.cb             = gp( m_in, m_inRec[1] );
    lf.cr             = gp( m_in, m_inRec[2] );
    lf.corrCb         = gp( m_work, m_lfCorr[0] );
    lf.corrCr         = gp( m_work, m_lfCorr[1] );
    lf.ctuList        = m_in.addr + m_inLfCtus;
    lf.numCtus        = int( prm.lfCtus.size() / 4 );
    lf.numWindows     = prm.lfWindows;
    lf.groupsX        = std::min( prm.lfWindows, 65535 );
    lf.ctusPerRow     = m_ctusPerRow;
    lf.ctuSize        = m_ctuSize / 2;
    lf.width          = wC;
    lf.height         = hC;
    lf.impMode        = slice.getLfCccmImpEnabledFlag() ? 1 + slice.getLfCccmImpFactorIdx() : 0;
    lf.lumaOffset     = 1 << ( bitDepth - 1 );
    lf.lumaOffsetLog2 = bitDepth - 1;
    lf.bitDepth       = bdChroma;
    lf.badThr         = 32 << std::max( 0, bitDepth - 10 );   // LoopFilterCccm::lfCccmInitBd
    lf.badThrReg      = 28 << std::max( 0, bitDepth - 10 );
    lf.minCb          = slice.clpRng( COMPONENT_Cb ).min;
    lf.maxCb          = slice.clpRng( COMPONENT_Cb ).max;
    lf.minCr          = slice.clpRng( COMPONENT_Cr ).min;
    lf.maxCr          = slice.clpRng( COMPONENT_Cr ).max;
    s.dispatch( m_k.lfCccm, &lf, lf.groupsX, divUp( prm.lfWindows, lf.groupsX ) );
  }
  s.barrier();

  PushFixFilter fix1 = fix;
  fix1.fix0          = gp( m_work, m_fix0 );
  fix1.dst           = gp( m_work, m_fix1 );
  fix1.fixClass      = m_work.addr + m_fixClass1;
  m_k.tables13( fix1, qpSet );
  s.dispatch( m_k.fix1, &fix1, divUp( m_width, 16 ), divUp( m_height, 8 ) );
  s.barrier();

  // the filters
  PushLuma luma;
  luma.rec        = gp( m_in, m_inRec[0] );
  luma.db         = gp( m_in, m_inDb[0] );
  luma.resi       = gp( m_in, m_inResi );
  luma.fix0       = gp( m_work, m_fix0 );
  luma.fix1       = gp( m_work, m_fix1 );
  luma.fixResi    = gp( m_work, m_fixResi );
  luma.gauss      = gp( m_work, m_gauss );
  luma.lapl       = gp( m_work, m_lapl );
  luma.dst        = gp( m_outDev, m_out[0] );
  luma.ctuInfo    = m_in.addr + m_inCtuInfo;
  luma.params     = m_in.addr + m_inLumaSets;
  luma.mainClass0 = m_work.addr + m_mainClass[0];
  luma.mainClass1 = m_work.addr + m_mainClass[1];
  luma.mainClass2 = m_work.addr + m_mainClass[2];
  luma.width      = m_width;
  luma.height     = m_height;
  luma.blkStride  = m_wBlk;
  luma.ctuSizeLog2 = log2Of( m_ctuSize );
  luma.ctusPerRow = m_ctusPerRow;
  luma.outMin     = clpRng.min;
  luma.outMax     = clpRng.max;
  luma.currBase   = 1 << ( clpRng.bd - 1 );
  s.dispatch( m_k.luma, &luma, divUp( m_width, 16 ), divUp( m_height, 8 ) );

  for( int c = 0; c < 2; c++ )
  {
    const ComponentID comp    = ComponentID( COMPONENT_Cb + c );
    const ClpRng&     clpRngC = cs.slice->clpRng( comp );
    PushChroma        ch;
    ch.rec         = gp( m_in, m_inRec[1 + c] );
    ch.fix         = gp( m_work, m_chFix[c] );
    ch.lumaRec     = gp( m_in, m_inRec[0] );
    ch.lumaResi    = gp( m_in, m_inResi );
    ch.crossDb     = gp( m_in, m_inDb[2 - c] );
    ch.dst         = gp( m_outDev, m_out[1 + c] );
    // the verification compares the correction separately
    ch.lfCccm      = prm.lfCccm && m_mode != VERIFY ? gp( m_work, m_lfCorr[c] ) : GpuPlane{ 0, 0, 0 };
    ch.ctuInfo     = m_in.addr + m_inCtuInfo;
    ch.chromaCoeff = m_in.addr + m_inChroma;
    ch.ccalfCoeff  = m_in.addr + m_inCcalf[c];
    ch.comp        = c;
    ch.width       = wC;
    ch.height      = hC;
    ch.ctuSizeLog2 = log2Of( m_ctuSize ) - 1;
    ch.ctusPerRow  = m_ctusPerRow;
    ch.withFixed   = prm.chromaFixed;
    ch.ccalfPrec   = prm.ccalfPrec[c];
    ch.ccalfClip   = alf.m_alfClippingValues[CHANNEL_TYPE_LUMA][1];
    ch.scaleCorr   = prm.scaleCorr[c];
    ch.outMin      = clpRngC.min;
    ch.outMax      = clpRngC.max;
    ch.bitDepth    = clpRngC.bd;
    s.dispatch( m_k.chroma, &ch, divUp( wC, 16 ), divUp( hC, 8 ) );
  }
  s.barrier();

  s.copy( m_outDev, 0, m_download, 0, m_outDev.size );
  if( m_mode == VERIFY )
  {
    s.copy( m_work, 0, m_workDownload, 0, m_work.size );
    s.copy( m_in, 0, m_inDownload, 0, m_in.size );
  }
  // the luma after SAO for the CPU stages after ALF that read it: TAlfProcess, the offset refinement after ALF
  m_saoLumaDownloaded = false;
#if JVET_AK0065_TALF
  m_saoLumaDownloaded |= cs.sps->getUseTAlf();
#endif
#if JVET_AK0121_LOOPFILTER_OFFSET_REFINEMENT
  m_saoLumaDownloaded |= cs.slice->getOffsetRefinementAlf() && !cs.slice->isIntra();
#endif
  m_saoLumaDownloaded = m_saoLumaDownloaded && prm.sao;
  if( m_saoLumaDownloaded )
  {
    s.copy( m_in, m_inRec[0].offset, m_saoDownload, 0, m_inRec[0].bytes() );
  }
  s.submitAndWait();
}

void GpuAlf::output( AdaptiveLoopFilter& alf, CodingStructure& cs ) const
{
  auto get = []( const gpu::Buffer& b, const PlaneDesc& p, PelBuf dst ) {
    const int16_t* src = (const int16_t*) ( (const uint8_t*) b.mapped + p.offset ) + p.origin();
    for( int y = 0; y < p.height; y++ )
    {
      memcpy( dst.buf + y * dst.stride, src + y * p.stride, sizeof( int16_t ) * p.width );
    }
  };
  PelUnitBuf rec = cs.getRecoBuf();
  for( int c = 0; c < 3; c++ )
  {
    get( m_download, m_out[c], rec.get( ComponentID( c ) ) );
  }
  if( m_saoLumaDownloaded )
  {
    // where ALFProcess leaves the picture before ALF (m_saoDownload holds m_inRec[0] alone)
    PlaneDesc p = m_inRec[0];
    p.offset    = 0;
    get( m_saoDownload, p, alf.m_tempBuf.get( COMPONENT_Y ) );
    get( m_saoDownload, p, alf.m_tempBufSAO.get( COMPONENT_Y ) );
  }
}

bool GpuAlf::verify( const AdaptiveLoopFilter& alf, const CodingStructure& cs, const GpuLfCccmJob* lfCccm )
{
  const uint8_t*  dl        = (const uint8_t*) m_workDownload.mapped;
  const uint8_t*  ciClass   = dl + m_ciClass;
  const uint16_t* fixClass1 = (const uint16_t*) ( dl + m_fixClass1 );
  auto            plane     = [dl]( const PlaneDesc& p ) { return (const int16_t*) ( dl + p.offset ) + p.origin(); };
  const int16_t*  fix0      = plane( m_fix0 );
  const int16_t*  fix1      = plane( m_fix1 );
  const int16_t*  fixResi   = plane( m_fixResi );
  const int16_t*  gauss     = plane( m_gauss );
  const int16_t*  lapl      = plane( m_lapl );
  const int       fixedSet  = cs.slice->getTileGroupAlfFixedFilterSetIdx( COMPONENT_Y );
  const short*    filterSet = cs.picture->getAlfCtbFilterIndex();
  const int       pad       = ALF_PADDING_SIZE_FIXED_RESULTS;
  const int       padG      = ALF_PADDING_SIZE_GAUSS_RESULTS;
  const int       padL      = ALF_PADDING_SIZE_LAPLACIAN_RESULTS;

  struct Check
  {
    const char* name;
    long        compared = 0, differ = 0;
  } checks[] = { { "coding-info class" },  { "fixed filter 1" },       { "fixed filter 2 class" },
                 { "fixed filter 2" },     { "residual filter" },      { "classifier class" },
                 { "Gauss image" },        { "Laplacian image" },      { "chroma fixed filter" },
                 { "luma output" },        { "Cb output" },            { "Cr output" },
                 { "LF-CCCM Cb correction" }, { "LF-CCCM Cr correction" },
                 { "luma after SAO" },     { "Cb after SAO" },         { "Cr after SAO" } };
  auto check = [&cs]( Check& c, int x, int y, int cpu, int gpu ) {
    c.compared++;
    if( cpu != gpu && c.differ++ < 3 )
    {
      fprintf( stderr, "GPU ALF POC %d: %s at (%d,%d): CPU %d, GPU %d\n", cs.slice->getPOC(), c.name, x, y, cpu, gpu );
    }
  };

  // intermediate results, where the CPU derived them (the CTUs with luma / chroma ALF)
  int ctuIdx = 0;
  for( int yCtu = 0; yCtu < m_height; yCtu += m_ctuSize )
  {
    for( int xCtu = 0; xCtu < m_width; xCtu += m_ctuSize, ctuIdx++ )
    {
      const bool anyAlf = cs.slice->getTileGroupAlfEnabledFlag( COMPONENT_Y )
                          || cs.slice->getTileGroupAlfEnabledFlag( COMPONENT_Cb )
                          || cs.slice->getTileGroupAlfEnabledFlag( COMPONENT_Cr );
      if( !anyAlf )
      {
        continue;
      }
      for( int c = 0; c < 2 && m_params.chromaFixed; c++ )
      {
        if( !alf.m_ctuEnableFlag[COMPONENT_Cb + c][ctuIdx] )
        {
          continue;
        }
        const int16_t* gpuFix   = plane( m_chFix[c] );
        const int      fixedSetC = cs.slice->getTileGroupAlfFixedFilterSetIdx( ComponentID( COMPONENT_Cb + c ) );
        for( int y = yCtu / 2; y < std::min( yCtu + m_ctuSize, m_height ) / 2; y++ )
        {
          for( int x = xCtu / 2; x < std::min( xCtu + m_ctuSize, m_width ) / 2; x++ )
          {
            check( checks[8], x, y, alf.m_fixFilterResult[COMPONENT_Cb + c][fixedSetC][y + pad][x + pad],
                   gpuFix[y * m_chFix[c].stride + x] );
          }
        }
      }
      if( !alf.m_ctuEnableFlag[COMPONENT_Y][ctuIdx] )
      {
        continue;
      }
      const int  fsi        = filterSet[ctuIdx];
      const bool aps        = fsi >= NUM_FIXED_FILTER_SETS;
      const int  alt        = alf.m_ctuAlternative[COMPONENT_Y][ctuIdx];
      const int  type       = aps ? alf.m_filterTypeApsLuma[fsi - NUM_FIXED_FILTER_SETS] : -1;
      const int  classifier = aps ? alf.m_classifierIdxApsLuma[fsi - NUM_FIXED_FILTER_SETS][alt] : -1;
      const bool useFixed   = !aps || type == ALF_FILTER_13_EXT_DB_RESI || type == ALF_FILTER_13_EXT_DB_RESI_DIRECT;
      const uint16_t* mainClass = classifier >= 0 ? (const uint16_t*) ( dl + m_mainClass[classifier] ) : nullptr;
      for( int y = yCtu; y < std::min( yCtu + m_ctuSize, m_height ); y++ )
      {
        for( int x = xCtu; x < std::min( xCtu + m_ctuSize, m_width ); x++ )
        {
          const int b = ( y / 2 ) * m_wBlk + x / 2;
          check( checks[0], x, y, alf.m_classifierCodingInfo[0][y][x], ciClass[b] );
          if( useFixed )
          {
            check( checks[1], x, y, alf.m_fixFilterResult[COMPONENT_Y][fixedSet][y + pad][x + pad],
                   fix0[y * m_fix0.stride + x] );
            if( fsi != 0 )
            {
              check( checks[2], x, y, alf.m_classifier[ALF_NUM_CLASSIFIER][y][x], fixClass1[b] );
              check( checks[3], x, y, alf.m_fixFilterResult[COMPONENT_Y][2 + fixedSet][y + pad][x + pad],
                     fix1[y * m_fix1.stride + x] );
            }
          }
          if( aps )
          {
            check( checks[5], x, y, alf.m_classifier[classifier][y][x], mainClass[b] );
            check( checks[6], x, y, alf.m_gaussPic[0][y + padG][x + padG], gauss[y * m_gauss.stride + x] );
          }
          if( type == ALF_FILTER_13_EXT_DB_RESI )
          {
            check( checks[4], x, y, alf.m_fixFilterResiResult[1 - fixedSet][y][x], fixResi[y * m_fixResi.stride + x] );
            check( checks[7], x, y, alf.m_laplacianPic[0][y + padL][x + padL], lapl[y * m_lapl.stride + x] );
          }
        }
      }
    }
  }

  // the filtered picture
  const CPelUnitBuf rec = cs.getRecoBuf();
  for( int c = 0; c < 3; c++ )
  {
    const PlaneDesc& p   = m_out[c];
    const int16_t*   gpu = (const int16_t*) ( (const uint8_t*) m_download.mapped + p.offset );
    const CPelBuf    cpu = rec.get( ComponentID( c ) );
    for( int y = 0; y < p.height; y++ )
    {
      for( int x = 0; x < p.width; x++ )
      {
        check( checks[9 + c], x, y, cpu.at( x, y ), gpu[y * p.stride + x] );
      }
    }
  }

  // the picture after SAO: the CPU's in ALFProcess's copy (the margin the ALF kernels read included)
  if( m_params.sao )
  {
    s_verifiedSao++;
    const uint8_t* dlIn = (const uint8_t*) m_inDownload.mapped;
    for( int c = 0; c < 3; c++ )
    {
      const PlaneDesc& p   = m_inRec[c];
      const int16_t*   gpu = (const int16_t*) ( dlIn + p.offset ) + p.origin();
      const CPelBuf    cpu = alf.m_tempBuf.get( ComponentID( c ) );
      const int        m   = MAX_FILTER_LENGTH_FIXED >> 1;
      for( int y = -m; y < p.height + m; y++ )
      {
        for( int x = -m; x < p.width + m; x++ )
        {
          check( checks[14 + c], x, y, cpu.at( x, y ), gpu[y * p.stride + x] );
        }
      }
    }
  }

  // the LF-CCCM correction the decoder adds after ALFProcess
  if( lfCccm && m_params.lfCccm )
  {
    s_verifiedLfCccm++;
    for( int c = 0; c < 2; c++ )
    {
      const PlaneDesc& p   = m_lfCorr[c];
      const int16_t*   gpu = plane( p );
      const CPelBuf    cpu = lfCccm->correction->get( ComponentID( COMPONENT_Cb + c ) );
      for( int y = 0; y < p.height; y++ )
      {
        for( int x = 0; x < p.width; x++ )
        {
          check( checks[12 + c], x, y, cpu.at( x, y ), gpu[y * p.stride + x] );
        }
      }
    }
  }

  s_verifiedPictures++;
  bool ok = true;
  for( const Check& c: checks )
  {
    if( c.differ )
    {
      ok = false;
      fprintf( stderr, "GPU ALF POC %d: %ld of %ld %s values differ\n", cs.slice->getPOC(), c.differ, c.compared, c.name );
    }
  }
  if( !ok )
  {
    s_failedPictures++;
  }
  return ok;
}

bool GpuAlf::process( AdaptiveLoopFilter& alf, CodingStructure& cs, GpuSaoJob* sao, GpuLfCccmJob* lfCccm )
{
  // on the CPU, SAO comes first, then the LF-CCCM correction from the picture after SAO, then ALF
  auto cpuSao = [sao]() {
    if( sao )
    {
      sao->cpu();
    }
  };
  auto cpuLfCccm = [lfCccm]() {
    if( lfCccm )
    {
      lfCccm->cpu();
    }
  };
  m_params.lfCccm = false;
  m_params.sao    = false;
  if( !accept( alf, cs ) )
  {
    cpuSao();
    cpuLfCccm();
    return false;
  }
  layout( cs.pcv->lumaWidth, cs.pcv->lumaHeight, cs.pcv->maxCUWidth );
  // ALFProcess without the hook: all of it on the CPU
  auto cpuAlf = [&alf, &cs]() {
    AlfProcessHook* hook = alf.m_processHook;
    alf.m_processHook    = nullptr;
    alf.ALFProcess( cs );
    alf.m_processHook = hook;
  };
  if( m_mode == VERIFY )
  {
    // the CPU results are kept; the GPU works on the inputs the CPU stages prepared and is compared with them
    if( sao )
    {
      if( m_verifyDeblocked.bufs.empty() || m_verifyDeblocked.Y().width != cs.area.lwidth()
          || m_verifyDeblocked.Y().height != cs.area.lheight() )
      {
        m_verifyDeblocked.destroy();
        m_verifyDeblocked.create( cs.area );
      }
      m_verifyDeblocked.copyFrom( cs.getRecoBuf() );
    }
    cpuSao();
    cpuLfCccm();
    cpuAlf();
    if( gather( alf, cs ) )
    {
      if( lfCccm )
      {
        gatherLfCccm( cs );
      }
      if( sao )
      {
        gatherSao( cs, *sao, false );   // SAOProcess resolved the merges
      }
      upload( alf, cs, m_verifyDeblocked );
      run( alf, cs );
      verify( alf, cs, lfCccm );
    }
    return true;
  }

  stateBefore( alf, cs );
  if( !gather( alf, cs ) )
  {
    // something the kernels do not implement: ALFProcess does it all (its state work again, which is idempotent)
    cpuSao();
    cpuLfCccm();
    cpuAlf();
    return true;
  }
  // SAO on the GPU when the LF-CCCM correction is too: the CPU derives it from the picture after SAO
  const bool lfCccmGpu = lfCccm && gatherLfCccm( cs );
  if( sao && !( ( !lfCccm || lfCccmGpu ) && gatherSao( cs, *sao, true ) ) )
  {
    cpuSao();
  }
  if( lfCccm && !lfCccmGpu )
  {
    cpuLfCccm();
  }
  const Params& p = m_params;
  if( !p.sao )
  {
    copiesBefore( alf, cs );
  }
  if( p.luma || p.chromaAlf[0] || p.chromaAlf[1] || p.ccalf[0] || p.ccalf[1] || p.scaleCorr[0] || p.scaleCorr[1]
      || p.lfCccm || p.sao )
  {
    upload( alf, cs, cs.getRecoBuf() );
    run( alf, cs );
    output( alf, cs );
  }
  if( p.lfCccm )
  {
    lfCccm->applied = true;
  }
  stateAfter( alf, cs );
  return true;
}
