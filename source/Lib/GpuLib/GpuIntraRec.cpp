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

/** \file     GpuIntraRec.cpp
    \brief    intra reconstruction of a picture on the GPU (verification of the recorded CPU reconstruction)
*/

#include "GpuIntraRec.h"
#include "VkCompute.h"

#include <mutex>

#include "CommonLib/CodingStructure.h"
#include "CommonLib/BilateralFilter.h"
#include "CommonLib/InterpolationFilter.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Slice.h"
#include "CommonLib/TrQuant.h"
#include "CommonLib/UnitTools.h"

#include "intra_rec.spv.h"
#include "intra_resi.spv.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace
{
// layout shared with intra_rec.comp (struct Tu): 52 int32
struct Tu
{
  int32_t comp, x, y, w, h;
  int32_t dirMode;
  int32_t isModeVer, intraPredAngle, absInvAngle, useCubic;
  int32_t unitW, unitH;
  int32_t bitDepth;
  int32_t cbf, coeffOffset;
  int32_t dqScale, dqRightShift, dqInMin, dqInMax;
  int32_t trMin, trMax, shift1, shift2;
  int32_t recMin, recMax;
  int32_t dbgOffset;
  int32_t plIdx;
  int32_t refFilter, applyPDPC, gradPDPC, angularScale, fusion, disableRefFilter;
  int32_t avail[2][5];
  int32_t readMax[2][2];
  int32_t nzRows, nzCols;
  int32_t bf;
  int32_t qp;
  int32_t pad[1];
};
static_assert( sizeof( Tu ) == 52 * 4, "Tu layout" );

// layout shared with intra_rec.comp (struct Params, std430)
struct Params
{
  uint64_t tus;
  uint64_t coeffs;
  uint64_t plane[3];
  uint64_t owner[3];
  uint64_t filters;
  uint64_t matrices;
  uint64_t scratch;
  uint64_t dbgPred;
  uint64_t dbgResi;
  uint64_t depCount;
  uint64_t depOffset;
  uint64_t depList;
  uint64_t queue;
  uint64_t counter;
  int32_t  origin[3];
  int32_t  stride[3];
  int32_t  width[3];
  int32_t  height[3];
  int32_t  ownerW[2];
  int32_t  ownerH[2];
  int32_t  unitLog2X[2];
  int32_t  unitLog2Y[2];
  int32_t  matrixOffset[8];
};
static_assert( offsetof( Params, origin ) == 144 && offsetof( Params, matrixOffset ) == 224 && sizeof( Params ) == 256,
               "Params layout" );

struct Push
{
  uint64_t params;
  int32_t  numTus;
  int32_t  verify;
};

constexpr int FILTER_TAPS[5] = { 6, 8, 10, 12, 16 };
constexpr int SCRATCH_INTS   = 16384;   // per workgroup of intra_resi
constexpr int RESI_GROUPS    = 256;

// the kernels and the constant tables, created once and shared by every instance (the decoder contexts of the frame
// threads); never destroyed, like the device
struct Shared
{
  gpu::Kernel                      kernel, kernelResi;
  gpu::Buffer                      filters, matrices;
  int32_t                          matrixOffset[8] = {};
  std::unique_ptr<BilateralFilter> bif;   // the intra prediction bilateral filter's tables
};

Shared* createShared( gpu::Context& ctx )
{
  Shared* sh = new Shared;
  sh->kernel.create( ctx, spv_intra_rec, sizeof( spv_intra_rec ), sizeof( Push ), "intra_rec" );
  sh->kernelResi.create( ctx, spv_intra_resi, sizeof( spv_intra_resi ), sizeof( Push ), "intra_resi" );

  // tables, from the C++ tables themselves
  sh->bif.reset( new BilateralFilter );   // BilateralFilter::create derives the LUTs
  sh->bif->create();
  std::vector<int32_t> filt( 5 * 32 * 16 + 32 * 4 + 26 * 48 + 64 + 16, 0 );
  for( int t = 0; t < 5; t++ )
  {
    for( int frac = 0; frac < 32; frac++ )
    {
      const TFilterCoeff* f = InterpolationFilter::getIntraNtapNonSmoothingFilter( false, frac, FILTER_TAPS[t] );
      for( int i = 0; i < FILTER_TAPS[t]; i++ )
      {
        filt[( t * 32 + frac ) * 16 + i] = f[i];
      }
    }
  }
  for( int frac = 0; frac < 32; frac++ )   // weak 4-tap filter of the negative-angle reference extension
  {
    const TFilterCoeff* f = InterpolationFilter::getWeak4TapFilterTable( frac );
    for( int i = 0; i < 4; i++ )
    {
      filt[5 * 32 * 16 + frac * 4 + i] = f[i];
    }
  }
  {
    int32_t* bt = filt.data() + 5 * 32 * 16 + 32 * 4;   // intra_rec's BF_LUT, BF_SIZE, BF_MAD
    for( int q = 0; q < 26; q++ ) for( int i = 0; i < 48; i++ ) bt[q * 48 + i] = sh->bif->m_lut[q][i];
    for( int i = 0; i < 64; i++ ) bt[26 * 48 + i] = sh->bif->m_tuSizeFactor[0][i];
    for( int i = 0; i < 16; i++ ) bt[26 * 48 + 64 + i] = sh->bif->m_tuMADFactor[0][i];
  }
  std::vector<int32_t> mat;
  const TMatrixCoeff* tables[7] = { g_trCoreDCT2P2[TRANSFORM_INVERSE][0],  g_trCoreDCT2P4[TRANSFORM_INVERSE][0],
                                    g_trCoreDCT2P8[TRANSFORM_INVERSE][0],  g_trCoreDCT2P16[TRANSFORM_INVERSE][0],
                                    g_trCoreDCT2P32[TRANSFORM_INVERSE][0], g_trCoreDCT2P64[TRANSFORM_INVERSE][0],
                                    g_trCoreDCT2P128[TRANSFORM_INVERSE][0] };
  for( int l = 0; l < 7; l++ )
  {
    const int n          = 2 << l;
    sh->matrixOffset[l]  = (int32_t) mat.size();
    mat.insert( mat.end(), tables[l], tables[l] + n * n );
  }
  sh->filters.create( ctx, filt.size() * 4, gpu::Buffer::DEVICE );
  sh->matrices.create( ctx, mat.size() * 4, gpu::Buffer::DEVICE );
  gpu::Buffer upload;
  upload.create( ctx, ( filt.size() + mat.size() ) * 4, gpu::Buffer::UPLOAD );
  memcpy( upload.mapped, filt.data(), filt.size() * 4 );
  memcpy( (char*) upload.mapped + filt.size() * 4, mat.data(), mat.size() * 4 );
  {
    gpu::Stream stream( ctx );
    stream.begin();
    stream.copy( upload, 0, sh->filters, 0, filt.size() * 4 );
    stream.copy( upload, filt.size() * 4, sh->matrices, 0, mat.size() * 4 );
    stream.submitAndWait();
  }
  upload.destroy( ctx );
  return sh;
}

Shared* shared( gpu::Context& ctx )
{
  static std::once_flag once;
  static Shared*        sh = nullptr;
  std::call_once( once, [&] { sh = createShared( ctx ); } );
  return sh;
}
}   // namespace

struct GpuIntraRec::Impl
{
  gpu::Context*               ctx = nullptr;
  const Shared*               sh  = nullptr;
  std::unique_ptr<gpu::Stream> stream;
  gpu::Buffer                 scratch;
  gpu::Buffer                 params, tus, coeffs, planes[3], owner[3], sched, dbgPred, dbgResi;
  int                         numGroups = 0;
  gpu::Buffer                 upload, download;

  // the picture being recorded
  std::vector<Tu>             rec;
  std::vector<int32_t>        coef;
  std::vector<int16_t>        cpuPred, cpuResi;   // verification only
  size_t                      numSamples = 0;     // samples of the recorded blocks
  std::string                 unsupportedWhy;
  int                         poc = 0;
  bool                        on = false;
  int                         numOn = 0, numFallback = 0;

  // totals
  int                         numPictures = 0, numVerified = 0, numMismatch = 0, numSkipped = 0;
  std::string                 firstSkipReason;
  double                      gpuMs = 0;

  ~Impl()
  {
    if( !ctx )
    {
      return;
    }
    for( gpu::Buffer* b: { &scratch, &params, &tus, &coeffs, &planes[0], &planes[1], &planes[2],
                           &owner[0], &owner[1], &owner[2], &sched, &dbgPred, &dbgResi, &upload, &download } )
    {
      b->destroy( *ctx );
    }
  }
};

GpuIntraRec* GpuIntraRec::create()
{
  gpu::Context* ctx = gpu::Context::get();
  if( !ctx || !ctx->vulkanMemoryModel() )   // the blocks of a picture synchronize through the Vulkan memory model
  {
    return nullptr;
  }
  GpuIntraRec* r = new GpuIntraRec;
  r->m_impl.reset( new Impl );
  Impl& m = *r->m_impl;
  m.ctx = ctx;
  m.sh  = shared( *ctx );
  m.stream.reset( new gpu::Stream( *ctx ) );
  // persistent workgroups of intra_rec (ECM_GPU_GROUPS overrides)
  m.numGroups = getenv( "ECM_GPU_GROUPS" ) ? std::max( 1, atoi( getenv( "ECM_GPU_GROUPS" ) ) ) : 32;
  m.scratch.create( *ctx, size_t( SCRATCH_INTS ) * 4 * RESI_GROUPS, gpu::Buffer::DEVICE );
  m.params.create( *ctx, sizeof( Params ), gpu::Buffer::DEVICE );
  return r;
}

GpuIntraRec::~GpuIntraRec()
{
  if( m_impl )
  {
    const Impl& m = *m_impl;
    if( m.on )
    {
      printf( "\n[gpu-intra] %d pictures: %d reconstructed on the GPU, %d by the CPU%s%s; GPU %.1f ms\n", m.numPictures,
              m.numOn, m.numPictures - m.numOn, m.firstSkipReason.empty() ? "" : ", first reason: ", m.firstSkipReason.c_str(),
              m.gpuMs );
    }
    else
    {
      printf( "\n[gpu-intra] %d pictures: %d verified (%d with differences), %d skipped%s%s; GPU %.1f ms\n", m.numPictures,
              m.numVerified, m.numMismatch, m.numSkipped, m.firstSkipReason.empty() ? "" : ", first reason: ",
              m.firstSkipReason.c_str(), m.gpuMs );
    }
  }
}

void GpuIntraRec::setOn( bool on )
{
  m_impl->on = on;
}

bool GpuIntraRec::on() const
{
  return m_impl->on;
}

bool GpuIntraRec::failed() const
{
  return !m_impl->unsupportedWhy.empty();
}

bool GpuIntraRec::supported( const Slice& slice, std::string& why )
{
  // every tool whose mode derivation or prediction reads reconstructed samples, or that the kernels do not implement,
  // must be off: with the GPU reconstruction on, the CPU reconstructs nothing the derivations could read
  const SPS& sps = *slice.getSPS();
  why.clear();
  if( !slice.isIntra() )                                                                  why = "not an intra slice";
  else if( sps.getChromaFormatIdc() != CHROMA_420 && sps.getChromaFormatIdc() != CHROMA_400 ) why = "chroma format";
  else if( sps.getUseDualITree() )                                                        why = "dual tree";
  else if( sps.getLog2MaxTbSize() > 7 )                                                   why = "transform blocks above 128";
  else if( sps.getUseLmcs() )                                                             why = "LMCS";
  else if( sps.getUseQcs() )                                                              why = "QCS";
  else if( sps.getScalingListFlag() )                                                     why = "scaling lists";
  else if( sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag() )               why = "extended precision";
  else if( sps.getUseColorTrans() )                                                       why = "ACT";
  else if( sps.getNumPredSigns() > 0 )                                                    why = "sign prediction";
  else if( sps.getUseMIP() )                                                              why = "MIP";
  else if( sps.getUseISP() )                                                              why = "ISP";
  else if( sps.getUseMRL() || sps.getUseTmrl() )                                          why = "MRL / TMRL";
  else if( sps.getUseIntraLFNSTISlice() || sps.getUseIntraLFNSTPBSlice() || sps.getUseNSPT() ) why = "LFNST / NSPT";
  else if( sps.getUseIntraMTS() || sps.getUseImplicitMTS() )                              why = "MTS";
  else if( sps.getTransformSkipEnabledFlag() || sps.getBDPCMEnabledFlag() )               why = "transform skip / BDPCM";
  else if( sps.getJointCbCrEnabledFlag() )                                                why = "joint CbCr";
  else if( sps.getUseLMChroma() || sps.getUseCcpMerge() || sps.getUseDdCcpFusion() || sps.getUseBvgCccm() ) why = "cross-component prediction";
  else if( sps.getUseChromaReordering() || sps.getUseIntraDBV() )                         why = "chroma reordering / DBV";
  else if( sps.getUseDimd() || sps.getUseBvgDimd() || sps.getUseTimd() || sps.getUseTimdMrg() ) why = "DIMD / TIMD";
  else if( sps.getUseMdip() || sps.getUseMpmSorting() )                                   why = "MDIP / MPM sorting";
  else if( sps.getUseIntraTMP() || sps.getIBCFlag() )                                     why = "IntraTMP / IBC";
  else if( sps.getUseEip() || sps.getUseSgpm() )                                          why = "EIP / SGPM";
  else if( sps.getUsePDP() )                                                              why = "PDP";
  else if( slice.getPnnMode() )                                                           why = "NN intra prediction";
  return why.empty();
}

void GpuIntraRec::beginPicture( const CodingStructure& cs )
{
  Impl& m = *m_impl;
  m.rec.clear();
  m.coef.clear();
  m.cpuPred.clear();
  m.cpuResi.clear();
  m.numSamples = 0;
  m.unsupportedWhy.clear();
  m.poc = cs.slice->getPOC();
  std::string why;
  if( !supported( *cs.slice, why ) )
  {
    unsupported( why.c_str() );
  }
}

void GpuIntraRec::unsupported( const char* why )
{
  if( m_impl->unsupportedWhy.empty() )
  {
    m_impl->unsupportedWhy = why;
  }
}

void GpuIntraRec::record( const TransformUnit& tu, const ComponentID compID, const IntraPrediction::IpaSnapshot& ipa,
                          const QpParam& qp, const CPelBuf* pred, const CPelBuf* resi )
{
  Impl& m = *m_impl;
  if( !m.unsupportedWhy.empty() )
  {
    return;
  }
  const CodingUnit&      cu  = *tu.cu;
  const PredictionUnit&  pu  = *cu.firstPU;
  const CodingStructure& cs  = *tu.cs;
  const SPS&             sps = *cs.sps;
  const Slice&           slice = *cs.slice;
  const ChannelType      chType = toChannelType( compID );
  const CompArea&        area   = tu.blocks[compID];
  const int              dirMode = PU::getFinalIntraMode( pu, chType );

  // what the GPU path does not do yet
  const char* why = nullptr;
  if( !CU::isIntra( cu ) )                                          why = "not intra";
  else if( cu.bdpcmMode || cu.bdpcmModeChroma )                     why = "BDPCM";
  else if( cu.mipFlag )                                             why = "MIP";
  else if( cu.ispMode )                                             why = "ISP";
  else if( cu.lfnstIdx )                                            why = "LFNST";
  else if( tu.mtsIdx[compID] != MTS_DCT2_DCT2 )                     why = "MTS / transform skip";
  else if( tu.jointCbCr )                                           why = "joint CbCr";
  else if( cu.tmpFlag )                                             why = "IntraTMP";
  else if( cu.dimd )                                                why = "DIMD";
  else if( cu.timd )                                                why = "TIMD";
  else if( cu.sgpm )                                                why = "SGPM";
  else if( cu.tmrlFlag )                                            why = "TMRL";
  else if( cu.eipFlag )                                             why = "EIP";
  else if( pu.multiRefIdx || ipa.multiRefIndex )                    why = "MRL";
  else if( dirMode >= NUM_LUMA_MODE )                               why = "intra mode beyond 66 (LM / NN / ...)";
  else if( sps.getUseLmcs() )                                       why = "LMCS";
  else if( sps.getUseQcs() )                                        why = "QCS";
  else if( sps.getScalingListFlag() )                               why = "scaling lists";
  else if( sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag() ) why = "extended precision";
  else if( sps.getNumPredSigns() > 0 )                              why = "sign prediction";
  else if( area.width < 2 || area.height < 2 || area.width > 128 || area.height > 128 ) why = "block size";
  if( why )
  {
    unsupported( why );
    return;
  }

  Tu t{};
  t.comp           = compID;
  t.x              = area.x;
  t.y              = area.y;
  t.w              = area.width;
  t.h              = area.height;
  t.dirMode        = dirMode;
  t.isModeVer      = ipa.isModeVer;
  t.intraPredAngle = ipa.intraPredAngle;
  t.absInvAngle    = ipa.absInvAngle;
  t.useCubic       = !ipa.interpolationFlag;
  t.plIdx          = isLuma( compID ) && dirMode == PLANAR_IDX ? cu.plIdx : 0;
  t.refFilter        = ipa.refFilterFlag;
  t.applyPDPC        = ipa.applyPDPC;
  t.gradPDPC         = ipa.useGradPDPC;
  t.angularScale     = ipa.angularScale;
  t.fusion           = ipa.applyFusion;
  t.disableRefFilter = sps.getDisableRefFilter();
  t.qp = cu.qp;
  t.bf = 0;
  if( compID == COMPONENT_Y && sps.getUseIntraPredBf() && slice.isIntra() )
  {
    const int W = area.width, H = area.height, qpY = cu.qp;
    bool apply = qpY > 17 && W <= 128 && H <= 128 && ( W & ( W - 1 ) ) == 0 && ( H & ( H - 1 ) ) == 0 && W >= 4 && H >= 4;
    apply = apply && W * H > 16 && W * H < 1024 && std::abs( floorLog2( W ) - floorLog2( H ) ) <= 2;
    if( dirMode > DC_IDX )
    {
      const int md = dirMode;
      apply = apply && !( ( md >= 2 && md <= 3 ) || ( md >= 17 && md <= 19 ) || ( md >= 33 && md <= 35 ) || ( md >= 49 && md <= 51 ) || ( md >= 65 && md <= 66 ) );
    }
    apply = apply && m.sh->bif->m_tuSizeFactor[0][std::min( floorLog2( W ), 7 ) * 8 + std::min( floorLog2( H ), 7 )] != 0;
    if( apply )
    {
      if( area.pos() != cu.blocks[COMPONENT_Y].pos() || area.size() != cu.blocks[COMPONENT_Y].size() )
      {
        unsupported( "intra bilateral filter of a split CU" );
        return;
      }
      t.bf = dirMode <= DC_IDX ? 1 : 2;
    }
  }
  const PreCalcValues& pcv = *cs.pcv;
  const bool noShift = pcv.noChroma2x2 && area.width == 4;
  t.unitW    = pcv.minCUWidth >> ( noShift ? 0 : getComponentScaleX( compID, sps.getChromaFormatIdc() ) );
  t.unitH    = pcv.minCUHeight >> ( noShift ? 0 : getComponentScaleY( compID, sps.getChromaFormatIdc() ) );
  t.bitDepth = sps.getBitDepth( chType );
  t.cbf      = TU::getCbf( tu, compID ) ? 1 : 0;

  // Quant::dequant without scaling lists, transform skip, extended precision or QCS
  const int  maxLog2TrDynamicRange = sps.getMaxLog2TrDynamicRange( chType );
  const bool needSqrtAdjustment    = TU::needsBlockSizeTrafoScale( tu, compID );
  const int  iTransformShift = getTransformShift( t.bitDepth, area.size(), maxLog2TrDynamicRange ) + ( needSqrtAdjustment ? -1 : 0 );
  const int  rightShift      = IQUANT_SHIFT - ( iTransformShift + qp.per( false ) );
  const int  scaleBits       = IQUANT_SHIFT + 1;
  const int  targetInputBitDepth = std::min<int>( maxLog2TrDynamicRange + 1, 32 + rightShift - scaleBits );
  t.dqScale      = g_invQuantScales[needSqrtAdjustment ? 1 : 0][qp.rem( false )];
  t.dqRightShift = rightShift;
  t.dqInMin      = -( 1 << ( targetInputBitDepth - 1 ) );
  t.dqInMax      = ( 1 << ( targetInputBitDepth - 1 ) ) - 1;
  t.trMin        = -( 1 << maxLog2TrDynamicRange );
  t.trMax        = ( 1 << maxLog2TrDynamicRange ) - 1;
  // TrQuant::xIT
  const int transformMatrixShift = g_transformMatrixShift[TRANSFORM_INVERSE];
  t.shift1 = transformMatrixShift + 1 + COM16_C806_TRANS_PREC;
  t.shift2 = ( transformMatrixShift + maxLog2TrDynamicRange - 1 ) - t.bitDepth + COM16_C806_TRANS_PREC;
  // DecCu::xIntraRecBlk reconstruction clipping (adaptive clipping for luma)
  if( compID == COMPONENT_Y )
  {
    t.recMin = slice.getLumaPelMin();
    t.recMax = slice.getLumaPelMax();
  }
  else
  {
    t.recMin = slice.clpRng( compID ).min;
    t.recMax = slice.clpRng( compID ).max;
  }

  // the levels: only the region holding the nonzero ones (rows < nzRows, columns < nzCols)
  t.coeffOffset = (int32_t) m.coef.size();
  t.dbgOffset   = (int32_t) m.numSamples;
  m.numSamples += size_t( area.width ) * area.height;
  t.nzRows = t.nzCols = 0;
  if( t.cbf )
  {
    const CCoeffBuf c = tu.getCoeffs( compID );
    for( int y = 0; y < (int) area.height; y++ )
    {
      const TCoeff* row = c.bufAt( 0, y );
      for( int x = (int) area.width - 1; x >= t.nzCols; x-- )
      {
        if( row[x] != 0 )
        {
          t.nzRows = y + 1;
          t.nzCols = x + 1;
          break;
        }
      }
      if( t.nzRows < y + 1 )
      {
        for( int x = 0; x < t.nzCols; x++ )
        {
          if( row[x] != 0 )
          {
            t.nzRows = y + 1;
            break;
          }
        }
      }
    }
    for( int y = 0; y < t.nzRows; y++ )
    {
      const TCoeff* row = c.bufAt( 0, y );
      m.coef.insert( m.coef.end(), row, row + t.nzCols );
    }
  }
  if( pred && resi )   // the CPU's prediction and residual, for the comparison
  {
    for( int y = 0; y < (int) area.height; y++ )
    {
      for( int x = 0; x < (int) area.width; x++ )
      {
        m.cpuPred.push_back( pred->at( x, y ) );
        m.cpuResi.push_back( resi->at( x, y ) );
      }
    }
  }
  m.rec.push_back( t );
}

void GpuIntraRec::endPicture( const CodingStructure& cs )
{
  Impl& m = *m_impl;
  m.numPictures++;
  if( m.on && !m.unsupportedWhy.empty() )
  {
    m.numFallback++;
  }
  if( !m.unsupportedWhy.empty() || m.rec.empty() )
  {
    m.numSkipped++;
    if( m.firstSkipReason.empty() )
    {
      m.firstSkipReason = m.unsupportedWhy.empty() ? "no intra blocks" : m.unsupportedWhy;
    }
    return;
  }
  gpu::Context& ctx = *m.ctx;
  const int numComp = cs.pcv->chrFormat == CHROMA_400 ? 1 : 3;

  Params P{};
  size_t planeBytes[3] = {};
  for( int c = 0; c < numComp; c++ )
  {
    const CPelBuf reco = cs.picture->getRecoBuf( ComponentID( c ) );
    P.width[c]  = reco.width;
    P.height[c] = reco.height;
    P.stride[c] = reco.width;
    P.origin[c] = 0;
    planeBytes[c] = size_t( reco.width ) * reco.height * 2;
    m.planes[c].ensure( ctx, planeBytes[c], gpu::Buffer::DEVICE );
    P.plane[c] = m.planes[c].addr;
  }
  // owner maps: per unit and component the block covering it (CodingStructure unit scales, 4:2:0); -1 elsewhere
  std::vector<int32_t> ownerMap[3];
  size_t ownerBytes[3] = {};
  for( int ch = 0; ch < ( numComp > 1 ? 2 : 1 ); ch++ )
  {
    const int sx = ch == 0 ? 2 : 1, sy = ch == 0 ? 2 : 1;   // UnitScaleArray (4:2:0)
    P.unitLog2X[ch] = sx;
    P.unitLog2Y[ch] = sy;
    P.ownerW[ch]    = ( P.width[ch] + ( 1 << sx ) - 1 ) >> sx;
    P.ownerH[ch]    = ( P.height[ch] + ( 1 << sy ) - 1 ) >> sy;
  }
  for( int c = 0; c < numComp; c++ )
  {
    const int ch = c == 0 ? 0 : 1;
    ownerMap[c].assign( size_t( P.ownerW[ch] ) * P.ownerH[ch], -1 );
  }
  for( size_t r = 0; r < m.rec.size(); r++ )
  {
    const Tu& t  = m.rec[r];
    const int ch = t.comp == 0 ? 0 : 1;
    for( int uy = t.y >> P.unitLog2Y[ch]; uy <= ( t.y + t.h - 1 ) >> P.unitLog2Y[ch]; uy++ )
    {
      for( int ux = t.x >> P.unitLog2X[ch]; ux <= ( t.x + t.w - 1 ) >> P.unitLog2X[ch]; ux++ )
      {
        ownerMap[t.comp][size_t( uy ) * P.ownerW[ch] + ux] = (int32_t) r;
      }
    }
  }
  for( int c = 0; c < numComp; c++ )
  {
    ownerBytes[c] = ownerMap[c].size() * 4;
    m.owner[c].ensure( ctx, ownerBytes[c], gpu::Buffer::DEVICE );
    P.owner[c] = m.owner[c].addr;
  }
  // dependencies: the blocks owning a sample a block reads - geometry, decoding order and syntax only. Per reference
  // line: the availability the CPU sees (corner flag, available prefix of each unit group), the highest top / left
  // index the prediction can read (a superset), then for each sample in that range its owner, or for an unavailable
  // sample the owner of its padding source (the same rule as the kernel's fill)
  const int numTus = (int) m.rec.size();
  std::vector<std::vector<int>> deps( numTus );
  {
    constexpr int PAD = 6;   // interpFilterNumPadded
    std::vector<int> mark( numTus, -1 );
    for( int i = 0; i < numTus; i++ )
    {
      Tu&       t  = m.rec[i];
      const int ch = t.comp == 0 ? 0 : 1;
      auto own = [&]( int c, int x, int y ) -> int
      {
        if( x < 0 || y < 0 || ( x >> P.unitLog2X[ch] ) >= P.ownerW[ch] || ( y >> P.unitLog2Y[ch] ) >= P.ownerH[ch] )
        {
          return -1;
        }
        return ownerMap[c][size_t( y >> P.unitLog2Y[ch] ) * P.ownerW[ch] + ( x >> P.unitLog2X[ch] )];
      };
      const int availComp = t.comp == 0 ? 0 : 1;
      auto avail = [&]( int x, int y ) { const int a = own( availComp, x, y ); return a >= 0 && a < i; };
      const int numAbove = std::max( t.w / t.unitW, 1 ), numLeft = std::max( t.h / t.unitH, 1 );
      const int tAU = ( t.w << 3 ) / t.unitW, tLU = ( t.h << 3 ) / t.unitH, totalUnits = tAU + tLU + 1;
      memset( t.avail, 0, sizeof( t.avail ) );
      memset( t.readMax, 0, sizeof( t.readMax ) );
      for( int mrl = 0; mrl <= ( t.fusion ? 1 : 0 ); mrl++ )
      {
        // the read range
        const int W = t.w, H = t.h, rf = t.refFilter ? 1 : 0;
        int topMax, leftMax;
        if( t.dirMode == PLANAR_IDX )
        {
          topMax  = W + 1 + rf;
          leftMax = H + 1 + rf;
        }
        else if( t.dirMode == DC_IDX )
        {
          topMax  = W + 1;
          leftMax = H + 1;
        }
        else
        {
          const bool ver = t.isModeVer != 0;
          const int  w = ver ? W : H, h = ver ? H : W, refLength = 8 * w, sideLength = 8 * h;
          const int  angle = t.intraPredAngle;
          int mainMax, sideMax;
          if( mrl == 0 )
          {
            if( angle == 0 )
            {
              mainMax = w + 1;
              sideMax = t.applyPDPC ? h + 1 : 1;
            }
            else if( angle < 0 )
            {
              mainMax = w + PAD + 2;   // the copied part (the extension reads the side up to sizeSide + 2)
              sideMax = h + PAD + 2;
            }
            else
            {
              mainMax = std::min( refLength, ( ( angle * h ) >> 5 ) + w + 10 );   // beyond refLength: its replicated value
              sideMax = 1;
              if( t.applyPDPC )
              {
                const int limit = std::min( 3 << t.angularScale, w );
                sideMax = t.gradPDPC ? h + 1 : std::min( sideLength, h + ( ( 256 + limit * t.absInvAngle ) >> 9 ) + 1 );
              }
            }
            mainMax += rf;
            sideMax += rf;
          }
          else   // the fusion line (non-integer slopes)
          {
            if( angle < 0 )
            {
              mainMax = w + PAD + 10;
              sideMax = std::min( sideLength + 1, ( ( ( h + 1 + PAD ) * t.absInvAngle + 8 ) >> 4 >> 5 ) + 3 );
            }
            else
            {
              mainMax = std::min( refLength + 1, ( ( angle * ( h + 1 ) ) >> 5 ) + w + 12 );
              sideMax = 1;
            }
          }
          topMax  = ver ? mainMax : sideMax;
          leftMax = ver ? sideMax : mainMax;
        }
        if( t.bf == 2 && mrl == 0 )   // the filter's gate: the texture of the whole reference arrays
        {
          topMax  = 8 * W;
          leftMax = 8 * H;
        }
        t.readMax[mrl][0] = std::max( mrl, std::min( topMax, 8 * W + mrl ) );
        t.readMax[mrl][1] = std::max( mrl, std::min( leftMax, 8 * H + mrl ) );

        const int x0 = t.x - 1 - mrl, y0 = t.y - 1 - mrl;
        int* av = t.avail[mrl];
        // availability: the corner flag and the available prefix of each unit group, scanned only over the units the
        // read range overlaps (at least one: a group's first unit decides whether any of it is available). Units
        // beyond the range change neither a sample in range nor its padding source: those come from the nearest
        // available unit before it in the CPU's unit order (an interval end, or a group's fixed start) or from the
        // first available unit, whose position only depends on whether its group's prefix is 0.
        const int topMaxL = t.readMax[mrl][0], leftMaxL = t.readMax[mrl][1];
        const int unitsTop  = topMaxL >= 1 + mrl ? ( topMaxL - 1 - mrl ) / t.unitW + 1 : 0;
        const int unitsLeft = leftMaxL >= 1 + mrl ? ( leftMaxL - 1 - mrl ) / t.unitH + 1 : 0;
        const int limA  = std::max( 1, std::min( numAbove, unitsTop ) );
        const int limAR = std::max( 1, std::min( tAU - numAbove, unitsTop - numAbove ) );
        const int limL  = std::max( 1, std::min( numLeft, unitsLeft ) );
        const int limBL = std::max( 1, std::min( tLU - numLeft, unitsLeft - numLeft ) );
        av[0] = avail( x0, y0 ) ? 1 : 0;
        int k;
        for( k = 0; k < limA && avail( t.x + k * t.unitW, y0 ); k++ ) {}
        av[1] = k;
        for( k = 0; k < limAR && avail( t.x + t.w - 1 + t.unitW + k * t.unitW, y0 ); k++ ) {}
        av[2] = k;
        for( k = 0; k < limL && avail( x0, t.y + k * t.unitH ); k++ ) {}
        av[3] = k;
        for( k = 0; k < limBL && avail( x0, t.y + t.h - 1 + t.unitH + k * t.unitH ); k++ ) {}
        av[4] = k;

        if( t.bf && mrl == 0 )   // the bilateral filter's neighbours (rows y-2, y-1; columns x-2, x-1)
        {
          auto addOwner = [&]( int sx, int sy )
          {
            const int o = own( t.comp, sx, sy );
            if( o >= 0 && o < i && mark[o] != i )
            {
              mark[o] = i;
              deps[i].push_back( o );
            }
          };
          if( av[0] )
          {
            addOwner( t.x - 2, t.y - 2 ); addOwner( t.x - 1, t.y - 2 ); addOwner( t.x - 2, t.y - 1 ); addOwner( t.x - 1, t.y - 1 );
          }
          for( int x = 0; x < av[1] * t.unitW && x < t.w; x++ )
          {
            addOwner( t.x + x, t.y - 2 );
            addOwner( t.x + x, t.y - 1 );
          }
          for( int y = 0; y < av[3] * t.unitH && y < t.h; y++ )
          {
            addOwner( t.x - 2, t.y + y );
            addOwner( t.x - 1, t.y + y );
          }
        }
        // the owners of the samples in range (the kernel's fill rule)
        const int corner = av[0], pA = av[1], pAR = av[2], pL = av[3], pBL = av[4];
        const int blS = tLU - numLeft - pBL, blE = tLU - 1 - numLeft, lS = tLU - pL, lE = tLU - 1;
        const int aS = tLU + 1, aE = tLU + pA, arS = tLU + 1 + numAbove, arE = tLU + numAbove + pAR;
        int firstAvail = totalUnits;
        if( pAR > 0 ) firstAvail = arS;
        if( pA > 0 ) firstAvail = aS;
        if( corner ) firstAvail = tLU;
        if( pL > 0 ) firstAvail = lS;
        if( pBL > 0 ) firstAvail = blS;
        if( firstAvail >= totalUnits )
        {
          continue;   // no neighbour: the DC value
        }
        for( int side = 0; side < 2; side++ )
        {
          const bool top = side == 0;
          int lastKey = -1;   // the owner unit (available sample) or the availability unit (padding) just added
          for( int idx = 0; idx <= t.readMax[mrl][side]; idx++ )
          {
            const int u = idx <= mrl ? tLU : top ? tLU + 1 + ( idx - 1 - mrl ) / t.unitW : tLU - 1 - ( idx - 1 - mrl ) / t.unitH;
            int pm = -1;
            if( pBL > 0 && blS <= u ) pm = std::max( pm, std::min( u, blE ) );
            if( pL > 0 && lS <= u ) pm = std::max( pm, std::min( u, lE ) );
            if( corner && tLU <= u ) pm = std::max( pm, tLU );
            if( pA > 0 && aS <= u ) pm = std::max( pm, std::min( u, aE ) );
            if( pAR > 0 && arS <= u ) pm = std::max( pm, std::min( u, arE ) );
            int sx, sy;
            if( pm == u )
            {
              sx = top ? x0 + idx : x0;
              sy = top ? y0 : y0 + idx;
              const int key = ( ( sy >> P.unitLog2Y[ch] ) * P.ownerW[ch] + ( sx >> P.unitLog2X[ch] ) ) * 2;
              if( sx >= 0 && sy >= 0 && key == lastKey )
              {
                continue;
              }
              lastKey = sx >= 0 && sy >= 0 ? key : -1;
            }
            else
            {
              if( lastKey == u * 2 + 1 )   // the same padding source as the sample before
              {
                continue;
              }
              lastKey = u * 2 + 1;
              const bool before = pm >= 0;
              const int  s = before ? pm : firstAvail;
              if( s == tLU )
              {
                sx = before ? x0 + mrl : x0;
                sy = before ? y0 : y0 + mrl;
              }
              else if( s > tLU )
              {
                sx = x0 + ( before ? ( s - tLU ) * t.unitW + mrl : ( s - tLU - 1 ) * t.unitW + 1 + mrl );
                sy = y0;
              }
              else
              {
                sx = x0;
                sy = y0 + ( before ? ( tLU - s - 1 ) * t.unitH + 1 + mrl : ( tLU - s ) * t.unitH + mrl );
              }
            }
            const int o = own( t.comp, sx, sy );
            if( o >= 0 && o < i && mark[o] != i )
            {
              mark[o] = i;
              deps[i].push_back( o );
            }
          }
        }
      }
    }
  }
  // the scheduling data: pending counts, the dependents of each block (CSR), the queue seeded with the blocks
  // without dependencies, the queue counters and the count of reconstructed blocks
  std::vector<int32_t> sched;
  {
    std::vector<int32_t> numDependents( numTus + 1, 0 );
    for( int i = 0; i < numTus; i++ ) for( int d: deps[i] ) numDependents[d + 1]++;
    sched.resize( size_t( numTus ) * 3 + 1 + numDependents.size() + 3 );
    int32_t* depCount = sched.data();
    int32_t* depOffset = depCount + numTus;
    for( int i = 0; i < numTus; i++ ) depCount[i] = (int32_t) deps[i].size();
    depOffset[0] = 0;
    for( int i = 0; i < numTus; i++ ) depOffset[i + 1] = depOffset[i] + numDependents[i + 1];
    const int numEdges = depOffset[numTus];
    sched.resize( size_t( numTus ) * 2 + 1 + numEdges + numTus + 3 );
    depCount  = sched.data();
    depOffset = depCount + numTus;
    int32_t* depList = depOffset + numTus + 1;
    int32_t* queue   = depList + numEdges;
    int32_t* counter = queue + numTus;
    std::vector<int32_t> fillPos( depOffset, depOffset + numTus );
    for( int i = 0; i < numTus; i++ ) for( int d: deps[i] ) depList[fillPos[d]++] = i;
    int numReady = 0;
    for( int i = 0; i < numTus; i++ ) queue[i] = -1;
    for( int i = 0; i < numTus; i++ ) if( depCount[i] == 0 ) queue[numReady++] = i;
    counter[0] = 0;
    counter[1] = numReady;
    counter[2] = 0;
  }
  const size_t schedBytes = sched.size() * 4;
  const size_t tuBytes = m.rec.size() * sizeof( Tu ), coefBytes = m.coef.size() * 4, dbgBytes = m.numSamples * 2;
  m.tus.ensure( ctx, tuBytes, gpu::Buffer::DEVICE );
  m.coeffs.ensure( ctx, std::max<size_t>( coefBytes, 4 ), gpu::Buffer::DEVICE );
  m.sched.ensure( ctx, schedBytes, gpu::Buffer::DEVICE );
  m.dbgPred.ensure( ctx, dbgBytes, gpu::Buffer::DEVICE );
  m.dbgResi.ensure( ctx, dbgBytes, gpu::Buffer::DEVICE );
  P.tus      = m.tus.addr;
  P.coeffs   = m.coeffs.addr;
  P.filters  = m.sh->filters.addr;
  P.matrices = m.sh->matrices.addr;
  memcpy( P.matrixOffset, m.sh->matrixOffset, sizeof( P.matrixOffset ) );
  P.scratch  = m.scratch.addr;
  P.dbgPred  = m.dbgPred.addr;
  P.dbgResi  = m.dbgResi.addr;
  {
    size_t numEdges = sched.size() - ( size_t( numTus ) * 3 + 1 + 3 );
    P.depCount  = m.sched.addr;
    P.depOffset = P.depCount + size_t( numTus ) * 4;
    P.depList   = P.depOffset + ( size_t( numTus ) + 1 ) * 4;
    P.queue     = P.depList + numEdges * 4;
    P.counter   = P.queue + size_t( numTus ) * 4;
  }

  size_t upBytes = sizeof( Params ) + tuBytes + coefBytes + schedBytes;
  for( int c = 0; c < numComp; c++ )
  {
    upBytes += ownerBytes[c];
  }
  m.upload.ensure( ctx, upBytes, gpu::Buffer::UPLOAD );
  size_t up = 0;
  memcpy( (char*) m.upload.mapped + up, &P, sizeof( Params ) );
  up += sizeof( Params );
  memcpy( (char*) m.upload.mapped + up, m.rec.data(), tuBytes );
  up += tuBytes;
  memcpy( (char*) m.upload.mapped + up, m.coef.data(), coefBytes );
  up += coefBytes;
  for( int c = 0; c < numComp; c++ )
  {
    memcpy( (char*) m.upload.mapped + up, ownerMap[c].data(), ownerBytes[c] );
    up += ownerBytes[c];
  }
  memcpy( (char*) m.upload.mapped + up, sched.data(), schedBytes );
  size_t downBytes = m.on ? 0 : 2 * dbgBytes;
  for( int c = 0; c < numComp; c++ )
  {
    downBytes += planeBytes[c];
  }
  m.download.ensure( ctx, downBytes, gpu::Buffer::DOWNLOAD );

  gpu::Stream& s = *m.stream;
  s.begin();
  up = 0;
  s.copy( m.upload, up, m.params, 0, sizeof( Params ) );
  up += sizeof( Params );
  s.copy( m.upload, up, m.tus, 0, tuBytes );
  up += tuBytes;
  if( coefBytes )
  {
    s.copy( m.upload, up, m.coeffs, 0, coefBytes );
  }
  up += coefBytes;
  for( int c = 0; c < numComp; c++ )
  {
    s.copy( m.upload, up, m.owner[c], 0, ownerBytes[c] );
    up += ownerBytes[c];
    s.fill( m.planes[c], 0, ( planeBytes[c] + 3 ) & ~size_t( 3 ), 0 );
  }
  s.copy( m.upload, up, m.sched, 0, schedBytes );
  s.barrier();
  const uint32_t t0 = s.timestamp();
  Push push{ m.params.addr, (int32_t) m.rec.size(), m.on ? 0 : 1 };
  s.dispatch( m.sh->kernelResi, &push, (uint32_t) std::min<size_t>( RESI_GROUPS, m.rec.size() ) );   // the residuals
  s.barrier();
  s.dispatch( m.sh->kernel, &push, (uint32_t) std::min<size_t>( m.numGroups, m.rec.size() ) );
  const uint32_t t1 = s.timestamp();
  s.barrier();
  size_t off = 0;
  if( !m.on )
  {
    s.copy( m.dbgPred, 0, m.download, off, dbgBytes );
    off += dbgBytes;
    s.copy( m.dbgResi, 0, m.download, off, dbgBytes );
    off += dbgBytes;
  }
  for( int c = 0; c < numComp; c++ )
  {
    s.copy( m.planes[c], 0, m.download, off, planeBytes[c] );
    off += planeBytes[c];
  }
  s.submitAndWait();
  const double ms = s.elapsedNs( t0, t1 ) * 1e-6;
  m.gpuMs += ms;

  if( m.on )   // the GPU's picture is the reconstruction
  {
    const int16_t* gPlane = (const int16_t*) m.download.mapped;
    for( int c = 0; c < numComp; c++ )
    {
      PelBuf reco = cs.picture->getRecoBuf( ComponentID( c ) );
      for( int y = 0; y < P.height[c]; y++ )
      {
        memcpy( reco.bufAt( 0, y ), gPlane + size_t( y ) * P.width[c], P.width[c] * sizeof( Pel ) );
      }
      gPlane += size_t( P.width[c] ) * P.height[c];
    }
    m.numOn++;
    return;
  }
  // compare: per block prediction and residual, then the whole picture
  const int16_t* gPred = (const int16_t*) m.download.mapped;
  const int16_t* gResi = gPred + m.cpuPred.size();
  std::string first;
  int numBad = 0;
  for( size_t i = 0; i < m.rec.size() && first.empty(); i++ )
  {
    const Tu& t = m.rec[i];
    for( int k = 0; k < t.w * t.h; k++ )
    {
      const int o = t.dbgOffset + k;
      const bool predBad = gPred[o] != m.cpuPred[o], resiBad = gResi[o] != m.cpuResi[o];
      if( predBad || resiBad )
      {
        char buf[320];
        snprintf( buf, sizeof( buf ),
                  "block %d (comp %d at %d,%d %dx%d, mode %d, angle %d, ver %d, cubic %d, cbf %d): %s at (%d,%d) CPU %d GPU %d",
                  (int) i, t.comp, t.x, t.y, t.w, t.h, t.dirMode, t.intraPredAngle, t.isModeVer, t.useCubic, t.cbf,
                  predBad ? "prediction" : "residual", k % t.w, k / t.w, predBad ? m.cpuPred[o] : m.cpuResi[o],
                  predBad ? gPred[o] : gResi[o] );
        first = buf;
        // the blocks owning the reference row above (x - 1 ... x + 2 w) and the column left (... y + 2 h)
        const int ch = t.comp == 0 ? 0 : 1;
        auto own = [&]( int x, int y ) -> int
        {
          if( x < 0 || y < 0 || ( x >> P.unitLog2X[ch] ) >= P.ownerW[ch] || ( y >> P.unitLog2Y[ch] ) >= P.ownerH[ch] )
          {
            return -1;
          }
          return ownerMap[t.comp][size_t( y >> P.unitLog2Y[ch] ) * P.ownerW[ch] + ( x >> P.unitLog2X[ch] )];
        };
        std::string above = "; owners above:", left = "; left:";
        for( int x = t.x - 1; x < t.x + 2 * t.w; x += 1 << P.unitLog2X[ch] )
        {
          above += " " + std::to_string( own( x, t.y - 1 ) );
        }
        for( int y = t.y - 1; y < t.y + 2 * t.h; y += 1 << P.unitLog2Y[ch] )
        {
          left += " " + std::to_string( own( t.x - 1, y ) );
        }
        first += above + left;
        break;
      }
    }
  }
  const int16_t* gPlane = gResi + m.cpuPred.size();
  for( int c = 0; c < numComp; c++ )
  {
    const CPelBuf reco = cs.picture->getRecoBuf( ComponentID( c ) );
    for( int y = 0; y < P.height[c]; y++ )
    {
      for( int x = 0; x < P.width[c]; x++ )
      {
        if( gPlane[y * P.width[c] + x] != reco.at( x, y ) )
        {
          if( numBad == 0 && first.empty() )
          {
            char buf[160];
            snprintf( buf, sizeof( buf ), "picture comp %d at (%d,%d): CPU %d GPU %d", c, x, y, reco.at( x, y ), gPlane[y * P.width[c] + x] );
            first = buf;
          }
          numBad++;
        }
      }
    }
    gPlane += P.width[c] * P.height[c];
  }
  m.numVerified++;
  const bool bad = numBad > 0 || !first.empty();
  m.numMismatch += bad ? 1 : 0;
  // the longest chain of blocks that read each other's samples
  std::vector<int> depth( numTus, 1 );
  int maxDepth = 0;
  for( int i = 0; i < numTus; i++ )
  {
    for( int d: deps[i] ) depth[i] = std::max( depth[i], depth[d] + 1 );
    maxDepth = std::max( maxDepth, depth[i] );
  }
  int numFilter = 0, numPdpc = 0, numGrad = 0, numFusion = 0, numBf = 0;
  for( const Tu& t: m.rec )
  {
    numFilter += t.refFilter;
    numPdpc += t.applyPDPC;
    numGrad += t.applyPDPC && t.gradPDPC;
    numFusion += t.fusion;
    numBf += t.bf != 0;
  }
  printf( "\n[gpu-intra] POC %d: %d blocks, chain %d (ref filter %d, PDPC %d, gradient PDPC %d, fusion %d, bilateral filter candidates %d), GPU %.2f ms: %s%s%s", m.poc,
          (int) m.rec.size(), maxDepth, numFilter, numPdpc, numGrad, numFusion, numBf, ms, bad ? "MISMATCH, " : "identical",
          bad ? first.c_str() : "", numBad ? ( ", " + std::to_string( numBad ) + " picture samples differ" ).c_str() : "" );
}
