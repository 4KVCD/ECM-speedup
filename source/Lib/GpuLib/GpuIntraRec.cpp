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

#include <algorithm>
#include <mutex>

#include "CommonLib/CodingStructure.h"
#include "CommonLib/BilateralFilter.h"
#include "CommonLib/InterpolationFilter.h"
#include "CommonLib/MipData.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Quant.h"
#include "CommonLib/Reshape.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Slice.h"
#include "CommonLib/TrQuant.h"
#include "CommonLib/UnitTools.h"

#include "intra_rec.spv.h"
#include "intra_resi.spv.h"
#include "intra_deps.spv.h"
#include "scan.spv.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace
{
// layout shared with intra_rec.comp (struct Tu): 67 int32
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
  int32_t lmcs;
  int32_t crsX, crsY, crsW, crsH;
  int32_t crsAvail;
  int32_t quant;
  int32_t jccr, jccrOffset;
  int32_t mtxV, mtxH;
  int32_t transformSkip;
  int32_t mipMatrix, mipTransposed, mipSizeId;
  int32_t precomputed;
};
static_assert( sizeof( Tu ) == 67 * 4, "Tu layout" );

// Tu::lmcs
constexpr int LMCS_BF_INVERSE  = 1;   // the intra bilateral filter works on inverse-mapped samples
constexpr int LMCS_CHROMA_SCALE = 2;   // the chroma residual is scaled by the CU's neighbouring luma

// Tu::quant
constexpr int QUANT_DEP = 1;   // dependent quantization: the coefficients are quantization indices (DepQuant)
constexpr int QUANT_QCS = 2;   // quantization center shift (JVET-AN0095 / JVET-AE0125)

// Tu::jccr (joint coding of the chroma residuals): the Cb block of a TU carries the ICT mode (TU::getICTMode, +-1..+-3)
// and computes both residuals; its Cr block, recorded right after it, is marked JCCR_DERIVED
constexpr int JCCR_DERIVED = 4;

// Tu::precomputed: the host ran a step of the CPU's inverse transform on the coefficients
constexpr int PRECOMPUTED_COEFFS   = 1;   // dequantized and inverse-LFNST'ed: the input of the primary transform
constexpr int PRECOMPUTED_RESIDUAL = 2;   // the residual itself (NSPT, which replaces the primary transform)

// layout shared with intra_rec.comp (struct Params, std430)
struct Params
{
  uint64_t tus;
  uint64_t coeffs;
  uint64_t plane[3];
  uint64_t owner[3];
  uint64_t filters;
  uint64_t matrices;
  uint64_t dbgPred;
  uint64_t dbgResi;
  uint64_t depCount;
  uint64_t depOffset;
  uint64_t depList;
  uint64_t queue;
  uint64_t counter;
  uint64_t depStart;
  uint64_t depTmp;
  uint64_t numDep;
  uint64_t lmcs;
  int32_t  origin[3];
  int32_t  stride[3];
  int32_t  width[3];
  int32_t  height[3];
  int32_t  ownerW[2];
  int32_t  ownerH[2];
  int32_t  unitLog2X[2];
  int32_t  unitLog2Y[2];
  int32_t  edgeCap;
  int32_t  lmcsLutSize;
  int32_t  lumaClipMin, lumaClipMax;
  int32_t  qcsOffset;
  int32_t  pad;
};
static_assert( offsetof( Params, origin ) == 168 && offsetof( Params, edgeCap ) == 248 && sizeof( Params ) == 272,
               "Params layout" );

struct Push
{
  uint64_t params;
  int32_t  numTus;
  int32_t  verify;   // intra_deps: the pass
};

struct PushScan
{
  uint64_t src, dst;
  int32_t  n;
  int32_t  pad;
};

constexpr int FILTER_TAPS[5] = { 6, 8, 10, 12, 16 };
constexpr int RESI_GROUPS    = 256;   // persistent workgroups of intra_resi

// the kernels and the constant tables, created once and shared by every instance (the decoder contexts of the frame
// threads); never destroyed, like the device
struct Shared
{
  gpu::Kernel                      kernel, kernelResi, kernelDeps, kernelScan;
  gpu::Buffer                      filters, matrices;
  int32_t                          trOffset[NUM_TRANS_TYPE][7];   // of the inverse transform matrices of sizes 2..128 in matrices (-1: none)
  int32_t                          qcsOffset = 0;                 // of the QCS weights in matrices
  int32_t                          mipOffset[3] = {};             // of the MIP matrices of each size class in matrices
  std::unique_ptr<BilateralFilter> bif;   // the intra prediction bilateral filter's tables
};

Shared* createShared( gpu::Context& ctx )
{
  Shared* sh = new Shared;
  sh->kernel.create( ctx, spv_intra_rec, sizeof( spv_intra_rec ), sizeof( Push ), "intra_rec" );
  sh->kernelResi.create( ctx, spv_intra_resi, sizeof( spv_intra_resi ), sizeof( Push ), "intra_resi" );
  sh->kernelDeps.create( ctx, spv_intra_deps, sizeof( spv_intra_deps ), sizeof( Push ), "intra_deps" );
  sh->kernelScan.create( ctx, spv_scan, sizeof( spv_scan ), sizeof( PushScan ), "scan" );

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
  // the inverse transform matrices of every type and size the CPU has (TrQuant::xIT's transform functions)
  std::vector<int32_t> mat;
  {
    // Don't put the TrQuant on the stack: it holds coefficient buffers of the largest transform, beyond the stack of
    // the thread that creates the device
    std::unique_ptr<TrQuant> trQuant( new TrQuant );
    trQuant->init( nullptr, MAX_TB_SIZEY, false, false, false, false );
    std::vector<TCoeff> m;
    for( int type = 0; type < NUM_TRANS_TYPE; type++ )
    {
      for( int l = 0; l < 7; l++ )
      {
        sh->trOffset[type][l] = -1;
        if( trQuant->getInvTransformMatrix( type, l + 1, m ) )
        {
          sh->trOffset[type][l] = (int32_t) mat.size();
          mat.insert( mat.end(), m.begin(), m.end() );
        }
      }
    }
  }
  sh->qcsOffset = (int32_t) mat.size();   // read by intra_resi's dequantization with QCS
  mat.insert( mat.end(), m_coeffShift64, m_coeffShift64 + 64 );
  // MIP (MatrixIntraPrediction::getMatrixData): per size class, the modes' matrices, each a row of weights per
  // reduced prediction sample (the large class without the first input's weight)
  sh->mipOffset[0] = (int32_t) mat.size();
  mat.insert( mat.end(), &mipMatrix4x4[0][0][0], &mipMatrix4x4[0][0][0] + sizeof( mipMatrix4x4 ) );
  sh->mipOffset[1] = (int32_t) mat.size();
  mat.insert( mat.end(), &mipMatrix8x8[0][0][0], &mipMatrix8x8[0][0][0] + sizeof( mipMatrix8x8 ) );
  sh->mipOffset[2] = (int32_t) mat.size();
  mat.insert( mat.end(), &mipMatrix16x16[0][0][0], &mipMatrix16x16[0][0][0] + sizeof( mipMatrix16x16 ) );
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
  gpu::Buffer                 params, tus, coeffs, planes[3], owner[3], sched, dbgPred, dbgResi, lmcs;
  Reshape*                    reshape = nullptr;   // the decoder's reshaper (LMCS state of the picture being decoded)
  TrQuant*                    trQuant = nullptr;   // the decoder's TrQuant (transform types of the blocks)
  int                         numGroups = 0;
  size_t                      edgeCap   = 0;   // dependency edges per picture the scheduling buffer holds
  gpu::Buffer                 upload, download;

  // the picture being recorded
  std::vector<Tu>             rec;
  std::vector<int32_t>        coef;
  std::vector<int16_t>        cpuPred, cpuResi;   // verification only
  size_t                      numSamples = 0;     // samples of the recorded blocks
  bool                        lmcsUsed = false;   // a block of the picture reads the LMCS tables
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
    for( gpu::Buffer* b: { &params, &tus, &coeffs, &planes[0], &planes[1], &planes[2],
                           &owner[0], &owner[1], &owner[2], &sched, &dbgPred, &dbgResi, &lmcs, &upload, &download } )
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

void GpuIntraRec::setReshape( Reshape* reshape )
{
  m_impl->reshape = reshape;
}

void GpuIntraRec::setTrQuant( TrQuant* trQuant )
{
  m_impl->trQuant = trQuant;
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
  else if( sps.getLog2MaxTbSize() > 7 )                                                   why = "transform blocks above 128";
  // implicit MTS selects by modes that only DIMD derives (TrQuant::getTrTypes, candModeListForTransform): with DIMD
  // off the stock encoder stops at a CHECK, so it can be neither supported nor tested before DIMD
  else if( sps.getUseImplicitMTS() )                                                      why = "implicit MTS";
  else if( sps.getScalingListFlag() )                                                     why = "scaling lists";
  else if( sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag() )               why = "extended precision";
  else if( sps.getUseColorTrans() )                                                       why = "ACT";
  else if( sps.getNumPredSigns() > 0 )                                                    why = "sign prediction";
  else if( sps.getUseISP() )                                                              why = "ISP";
  else if( sps.getUseMRL() || sps.getUseTmrl() )                                          why = "MRL / TMRL";
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
  m.lmcsUsed   = false;
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
  // MIP blocks (luma; IntraPrediction::predIntraMip) are recorded with the DC mode: they read the references DC reads
  // (a superset), and their bilateral filter, like DC's, has no angular texture gate; Tu::mipMatrix selects MIP
  const bool             mip     = isLuma( compID ) && cu.mipFlag;
  const int              dirMode = mip ? DC_IDX : PU::getFinalIntraMode( pu, chType );

  // what the GPU path does not do yet
  const char* why = nullptr;
  if( !CU::isIntra( cu ) )                                          why = "not intra";
  else if( !isLuma( compID ) && PU::isMIP( pu, chType ) )           why = "chroma MIP (4:4:4)";
  else if( cu.ispMode )                                             why = "ISP";
  // LFNST / NSPT of MIP blocks select their kernel by a DIMD mode the CPU derives from the MIP prediction
  else if( cu.mipFlag && cu.lfnstIdx )                              why = "LFNST of MIP";
  // transform skip with the regular residual coding is dequantized like a transform block (DepQuant::dequant)
  else if( ( tu.mtsIdx[compID] == MTS_SKIP || ( tu.jointCbCr && tu.mtsIdx[COMPONENT_Cr] == MTS_SKIP ) )
           && slice.getTSResidualCodingDisabledFlag() )             why = "transform skip with regular residual coding";
  else if( cu.tmpFlag )                                             why = "IntraTMP";
  else if( cu.dimd )                                                why = "DIMD";
  else if( cu.timd )                                                why = "TIMD";
  else if( cu.sgpm )                                                why = "SGPM";
  else if( cu.tmrlFlag )                                            why = "TMRL";
  else if( cu.eipFlag )                                             why = "EIP";
  else if( pu.multiRefIdx || ipa.multiRefIndex )                    why = "MRL";
  else if( dirMode >= NUM_LUMA_MODE )                               why = "intra mode beyond 66 (LM / NN / ...)";
  else if( sps.getUseLmcs() && !m.reshape )                         why = "LMCS without the decoder's reshaper";
  else if( sps.getScalingListFlag() )                               why = "scaling lists";
  else if( sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag() ) why = "extended precision";
  else if( sps.getNumPredSigns() > 0 )                              why = "sign prediction";
  else if( area.width < 2 || area.height < 2 || area.width > 128 || area.height > 128 ) why = "block size";
  else if( ( area.width & ( area.width - 1 ) ) || ( area.height & ( area.height - 1 ) ) ) why = "block size not a power of 2";
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
  // MIP (MatrixIntraPrediction::predBlock): the mode's matrix, transposition and size class; the prediction reads the
  // unfiltered references and uses none of the angular parameters
  t.mipMatrix = -1;
  if( mip )
  {
    const int    sizeId = getMipSizeId( area.size() );
    const int    rows[3] = { 16, 16, 64 }, rowLen[3] = { 4, 8, 7 };   // reduced samples, weights per sample
    const int    modeIdx = pu.intraDir[CHANNEL_TYPE_LUMA];
    if( modeIdx < 0 || modeIdx >= getNumModesMip( area.size() ) )
    {
      unsupported( "MIP mode index out of range" );
      return;
    }
    t.mipMatrix     = m.sh->mipOffset[sizeId] + modeIdx * rows[sizeId] * rowLen[sizeId];
    t.mipTransposed = pu.mipTransposedFlag ? 1 : 0;
    t.mipSizeId     = sizeId;
    t.refFilter = t.applyPDPC = t.gradPDPC = t.fusion = 0;
  }
  // BDPCM (IntraPrediction::xPredIntraBDPCM): the unfiltered left column (1, horizontal) or above row (2, vertical)
  // copied, i.e. the angular prediction of mode 18 or 50 without reference filter, PDPC, fusion or bilateral filter
  const int bdpcmPred = isLuma( compID ) ? cu.bdpcmMode : cu.bdpcmModeChroma;
  if( bdpcmPred )
  {
    if( dirMode != ( bdpcmPred == 1 ? HOR_IDX : VER_IDX ) )
    {
      unsupported( "BDPCM with an intra mode other than its direction" );
      return;
    }
    t.refFilter = t.applyPDPC = t.gradPDPC = t.fusion = t.bf = 0;
  }
  const PreCalcValues& pcv = *cs.pcv;
  const bool noShift = pcv.noChroma2x2 && area.width == 4;
  t.unitW    = pcv.minCUWidth >> ( noShift ? 0 : getComponentScaleX( compID, sps.getChromaFormatIdc() ) );
  t.unitH    = pcv.minCUHeight >> ( noShift ? 0 : getComponentScaleY( compID, sps.getChromaFormatIdc() ) );
  t.bitDepth = sps.getBitDepth( chType );

  // the component whose coefficients give the block's residual: with joint CbCr (DecCu::xIntraRecBlk at Cb), the
  // coded one of the pair (Cb for jointCbCr 2 and 3, else Cr with its own QP); the Cb block computes both residuals
  // (TrQuant::invTransformICT), the Cr block that follows it gets its residual from there
  ComponentID resiComp = compID;
  t.jccr       = 0;
  t.jccrOffset = 0;
  if( tu.jointCbCr && compID == COMPONENT_Cb )
  {
    t.jccr   = TU::getICTMode( tu );
    resiComp = ( tu.jointCbCr >> 1 ) ? COMPONENT_Cb : COMPONENT_Cr;
  }
  else if( tu.jointCbCr && compID == COMPONENT_Cr )
  {
    // DecCu::xIntraRecQT records a TU's Cb and Cr blocks one after the other, so the Cr samples follow the Cb ones
    const Tu* cb = m.rec.empty() ? nullptr : &m.rec.back();
    if( !cb || cb->comp != COMPONENT_Cb || cb->x != area.x || cb->y != area.y || cb->jccr == 0
        || cb->jccrOffset != (int32_t) m.numSamples )
    {
      unsupported( "joint CbCr: a Cr block not right after its Cb block" );
      return;
    }
    t.jccr = JCCR_DERIVED;
  }
  const QpParam resiQp = resiComp == compID ? qp : QpParam( tu, resiComp );
  t.cbf = t.jccr != JCCR_DERIVED && TU::getCbf( tu, resiComp ) ? 1 : 0;
  // transform skip (TrQuant::xITransformSkip): the dequantized levels are the residual
  const bool ts   = tu.mtsIdx[resiComp] == MTS_SKIP;
  t.transformSkip = ts ? 1 : 0;

  // TrQuant::xIT's transform types (DCT-II, or those of explicit MTS) and their matrices. With LFNST / NSPT (lfnstIdx)
  // the host runs the CPU's own dequantization and inverse LFNST and the GPU the primary transform; where NSPT replaces
  // the primary transform, the host computes the residual itself (both syntax only)
  t.mtxV = t.mtxH = 0;
  t.precomputed   = 0;
  if( t.cbf && !ts )
  {
    if( !m.trQuant )
    {
      unsupported( "transform types without the decoder's TrQuant" );
      return;
    }
    int trTypeHor, trTypeVer;
    if( !m.trQuant->getInvTransformTypes( tu, resiComp, trTypeHor, trTypeVer ) )
    {
      if( !cu.lfnstIdx )
      {
        unsupported( "a 1-D transform" );
        return;
      }
      t.precomputed = PRECOMPUTED_RESIDUAL;
    }
    else
    {
      t.mtxV = m.sh->trOffset[trTypeVer][floorLog2( area.height ) - 1];
      t.mtxH = m.sh->trOffset[trTypeHor][floorLog2( area.width ) - 1];
      if( t.mtxV < 0 || t.mtxH < 0 )
      {
        unsupported( "a transform type without a matrix of this size" );
        return;
      }
      t.precomputed = cu.lfnstIdx ? PRECOMPUTED_COEFFS : 0;
    }
  }

  // dequantization without scaling lists or extended precision: DepQuant::dequant (dequantBlock) with dependent
  // quantization (not for transform skip, which uses its own residual coding), else Quant::dequant; QCS blends either
  // with the dequantization of the next larger index
  const int maxLog2TrDynamicRange = sps.getMaxLog2TrDynamicRange( chType );
  t.quant = ( slice.getDepQuantEnabledIdc() && !ts ? QUANT_DEP : 0 ) | ( sps.getUseQcs() ? QUANT_QCS : 0 );
  if( t.precomputed )   // the host dequantized
  {
    t.quant = 0;
  }
  if( t.quant & QUANT_DEP )
  {
    // the index of a level depends on the quantizer state (the indices are derived with the levels below); QP + 1,
    // and the scale is shifted left instead of a negative right shift, without rounding
    const bool sqrt2          = TU::needsSqrt2Scale( tu, resiComp );
    const int  qpDQ           = resiQp.Qp( false ) + 1;
    const int  qpPer          = qpDQ / 6;
    const int  qpRem          = qpDQ - 6 * qpPer;
    const int  transformShift = getTransformShift( t.bitDepth, area.size(), maxLog2TrDynamicRange ) + ( sqrt2 ? -1 : 0 );
    const int  shift          = IQUANT_SHIFT + 1 - qpPer - transformShift;
    const int  scale          = g_invQuantScales[sqrt2 ? 1 : 0][qpRem];
    t.dqScale      = shift < 0 ? scale << -shift : scale;
    t.dqRightShift = std::max( shift, 0 );
    t.dqInMin      = INT32_MIN;   // the indices are not clipped
    t.dqInMax      = INT32_MAX;
  }
  else
  {
    // transform skip: the QP of transform skip, and no transform shift
    const bool needSqrtAdjustment  = TU::needsBlockSizeTrafoScale( tu, resiComp );
    const int  iTransformShift     = getTransformShift( t.bitDepth, area.size(), maxLog2TrDynamicRange ) + ( needSqrtAdjustment ? -1 : 0 );
    const int  rightShift          = IQUANT_SHIFT - ( ( ts ? 0 : iTransformShift ) + resiQp.per( ts ) );
    const int  scaleBits           = IQUANT_SHIFT + 1;
    const int  targetInputBitDepth = std::min<int>( maxLog2TrDynamicRange + 1, 32 + rightShift - scaleBits );
    t.dqScale      = g_invQuantScales[needSqrtAdjustment ? 1 : 0][resiQp.rem( ts )];
    t.dqRightShift = rightShift;
    t.dqInMin      = -( 1 << ( targetInputBitDepth - 1 ) );
    t.dqInMax      = ( 1 << ( targetInputBitDepth - 1 ) ) - 1;
  }
  t.trMin        = -( 1 << maxLog2TrDynamicRange );
  t.trMax        = ( 1 << maxLog2TrDynamicRange ) - 1;
  if( t.trMin < INT16_MIN || t.trMax > INT16_MAX )   // intra_resi keeps the first transform stage in 16 bits
  {
    unsupported( "transform range above 16 bits" );
    return;
  }
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
  // LMCS (the reconstruction stays in the mapped domain): the luma clipping range is mapped (DecCu::xIntraRecBlk),
  // the intra bilateral filter works on inverse-mapped samples (IntraPrediction::checkBfLmcsApplication), and the
  // chroma residual is scaled by the average luma next to the CU (DecCu::xIntraRecBlk, calculateChromaAdjVpduNei)
  t.lmcs = 0;
  if( sps.getUseLmcs() )
  {
    Reshape&                rsp   = *m.reshape;
    const SliceReshapeInfo& rInfo = rsp.getSliceReshaperInfo();
    if( compID == COMPONENT_Y && cs.picHeader->getLmcsEnabledFlag() )
    {
      t.recMin = rsp.getFwdLUT()[slice.getLumaPelMin()];
      t.recMax = rsp.getFwdLUT()[slice.getLumaPelMax()];
    }
    const bool bfLmcs = cs.picHeader->getLmcsEnabledFlag() && rInfo.getUseSliceReshaper()
                        && ( slice.getSliceType() == I_SLICE ? rInfo.getUseSliceReshaper() : rsp.getCTUFlag() );
    if( t.bf && bfLmcs )
    {
      t.lmcs |= LMCS_BF_INVERSE;
    }
    const bool lmcsSlice = slice.getLmcsEnabledFlag() && ( slice.isIntra() || rsp.getCTUFlag() );
    if( lmcsSlice && isChroma( compID ) && area.width * area.height > 4 && ( TU::getCbf( tu, compID ) || tu.jointCbCr )
        && cs.picHeader->getLmcsChromaResidualScaleFlag() )
    {
      t.lmcs |= LMCS_CHROMA_SCALE;
      // the CU's luma, or the chroma CU's area in luma samples (DecCu::xIntraRecBlk with LMCS_CHROMA_CALC_CU)
      const Area areaY = cu.Y().valid() ? Area( cu.Y() )
                                        : Area( recalcPosition( tu.chromaFormat, tu.chType, CHANNEL_TYPE_LUMA, cu.blocks[tu.chType].pos() ),
                                                recalcSize( tu.chromaFormat, tu.chType, CHANNEL_TYPE_LUMA, cu.blocks[tu.chType].size() ) );
      const CodingUnit *cuLeft, *cuAbove;
      Reshape::getChromaAdjNeighbours( tu, areaY.pos(), cuLeft, cuAbove );
      t.crsX     = areaY.x;
      t.crsY     = areaY.y;
      t.crsW     = areaY.width;
      t.crsH     = areaY.height;
      t.crsAvail = ( cuLeft ? 1 : 0 ) | ( cuAbove ? 2 : 0 );
    }
    m.lmcsUsed |= t.lmcs != 0;
  }

  // the levels: only the region holding the nonzero ones (rows < nzRows, columns < nzCols)
  t.coeffOffset = (int32_t) m.coef.size();
  t.dbgOffset   = (int32_t) m.numSamples;
  m.numSamples += size_t( area.width ) * area.height;
  if( t.jccr != 0 && t.jccr != JCCR_DERIVED )
  {
    t.jccrOffset = (int32_t) m.numSamples;   // the Cr block's samples
  }
  t.nzRows = t.nzCols = 0;
  if( t.cbf )
  {
    // BDPCM (Quant::dequant, with transform skip): the levels accumulated along the rows or columns
    const bool          bdpcm = isLuma( resiComp ) ? cu.bdpcmMode != 0 : cu.bdpcmModeChroma != 0;
    std::vector<TCoeff> accumulated;
    CCoeffBuf           c = tu.getCoeffs( resiComp );
    if( bdpcm )
    {
      accumulated.resize( area.area() );
      CoeffBuf acc( accumulated.data(), area.width, area.height );
      invResDPCM( tu, resiComp, acc );
      c = CCoeffBuf( accumulated.data(), area.width, area.height );
    }
    else if( t.precomputed == PRECOMPUTED_COEFFS )
    {
      c = m.trQuant->getInvLfnstOutput( tu, resiComp, resiQp );
    }
    else if( t.precomputed == PRECOMPUTED_RESIDUAL )
    {
      // invTransformNxN takes the TU non-const but only reads it
      std::vector<Pel> resiPel( area.area() );
      PelBuf           resiBuf( resiPel.data(), area.width, area.height );
      m.trQuant->invTransformNxN( const_cast<TransformUnit&>( tu ), resiComp, resiBuf, resiQp );
      accumulated.assign( resiPel.begin(), resiPel.end() );
      c = CCoeffBuf( accumulated.data(), area.width, area.height );
    }
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
    if( t.quant & QUANT_DEP )
    {
      // DepQuant's dequantBlock: from the last nonzero level down the diagonal scan, the quantizer state (a function
      // of the levels' parities only) turns each level into its quantization index
      CHECK( c.stride != area.width, "coefficients of a TU are contiguous" );
      const ScanElement* scan = g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][gp_sizeIdxInfo->idxFrom( area.width )]
                                           [gp_sizeIdxInfo->idxFrom( area.height )];
      const uint64_t     stateTransTab = g_stateTransTab[slice.getDepQuantEnabledIdc()];
      int                lastScanIdx   = (int) area.area() - 1;
      while( lastScanIdx >= 0 && c.buf[scan[lastScanIdx].idx] == 0 )
      {
        lastScanIdx--;
      }
      for( int state = 0, scanIdx = lastScanIdx; scanIdx >= 0; scanIdx-- )
      {
        const unsigned rasterPos = scan[scanIdx].idx;
        const TCoeff   level     = c.buf[rasterPos];
        if( level )
        {
          m.coef[t.coeffOffset + ( rasterPos / area.width ) * t.nzCols + rasterPos % area.width] =
            ( level << 1 ) + ( level > 0 ? -( state & 1 ) : ( state & 1 ) );
        }
        state = int( ( stateTransTab >> ( ( state << 3 ) + ( ( level & 1 ) << 2 ) ) ) & 15 );
      }
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
  // with the reconstruction on, intra_deps computes the owner maps and the scheduling data on the GPU; verification
  // computes them here too and compares
  const bool hostDeps = !m.on;
  for( int c = 0; c < numComp && hostDeps; c++ )
  {
    const int ch = c == 0 ? 0 : 1;
    ownerMap[c].assign( size_t( P.ownerW[ch] ) * P.ownerH[ch], -1 );
  }
  for( size_t r = 0; r < m.rec.size() && hostDeps; r++ )
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
    const int ch  = c == 0 ? 0 : 1;
    ownerBytes[c] = size_t( P.ownerW[ch] ) * P.ownerH[ch] * 4;
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
    for( int i = 0; i < numTus && hostDeps; i++ )
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
      if( t.lmcs & LMCS_CHROMA_SCALE )   // the luma column left of and the row above the CU, clipped to the picture
      {
        auto addLuma = [&]( int sx, int sy )
        {
          const int o = ownerMap[COMPONENT_Y][size_t( sy >> P.unitLog2Y[0] ) * P.ownerW[0] + ( sx >> P.unitLog2X[0] )];
          if( o >= 0 && o < i && mark[o] != i )
          {
            mark[o] = i;
            deps[i].push_back( o );
          }
        };
        if( t.crsAvail & 1 )
        {
          for( int k = 0; k < t.crsH; k++ ) addLuma( t.crsX - 1, std::min( t.crsY + k, P.height[0] - 1 ) );
        }
        if( t.crsAvail & 2 )
        {
          for( int k = 0; k < t.crsW; k++ ) addLuma( std::min( t.crsX + k, P.width[0] - 1 ), t.crsY - 1 );
        }
      }
    }
  }
  double         ms      = 0;
  const int16_t* gPlanes = nullptr;   // the downloaded results
  const int16_t* gPredDl = nullptr;
  const Tu*      gTusDl  = nullptr;
  const int32_t* gOffDl  = nullptr;
  const int32_t* gListDl = nullptr;
  // the scheduling buffer (intra_deps fills it): pending counts, the dependents of each block (CSR, edgeCap entries),
  // the queue, the counters, then intra_deps' own arrays
  const size_t n = size_t( numTus );
  m.edgeCap      = std::max( m.edgeCap, n * 8 );
  const size_t tuBytes = m.rec.size() * sizeof( Tu ), coefBytes = m.coef.size() * 4, dbgBytes = m.numSamples * 2;
  m.tus.ensure( ctx, tuBytes, gpu::Buffer::DEVICE );
  m.coeffs.ensure( ctx, std::max<size_t>( coefBytes, 4 ), gpu::Buffer::DEVICE );
  m.dbgPred.ensure( ctx, dbgBytes, gpu::Buffer::DEVICE );
  m.dbgResi.ensure( ctx, dbgBytes, gpu::Buffer::DEVICE );
  P.tus      = m.tus.addr;
  P.coeffs   = m.coeffs.addr;
  P.filters  = m.sh->filters.addr;
  P.matrices = m.sh->matrices.addr;
  P.dbgPred  = m.dbgPred.addr;
  P.dbgResi  = m.dbgResi.addr;
  P.lumaClipMin = cs.slice->clpRng( COMPONENT_Y ).min;
  P.lumaClipMax = cs.slice->clpRng( COMPONENT_Y ).max;
  P.qcsOffset   = m.sh->qcsOffset;
  // the picture's LMCS tables: the forward and inverse LUTs and the chroma scale of every average luma value
  // (Reshape::calculateChromaAdj itself)
  std::vector<int32_t> lmcsTable;
  if( m.lmcsUsed )
  {
    Reshape&  rsp = *m.reshape;
    const int lutSize = (int) rsp.getFwdLUT().size();
    lmcsTable.resize( 3 * size_t( lutSize ) );
    for( int v = 0; v < lutSize; v++ )
    {
      lmcsTable[v]               = rsp.getFwdLUT()[v];
      lmcsTable[lutSize + v]     = rsp.getInvLUT()[v];
      lmcsTable[2 * lutSize + v] = rsp.calculateChromaAdj( Pel( v ) );
    }
    m.lmcs.ensure( ctx, lmcsTable.size() * 4, gpu::Buffer::DEVICE );
    P.lmcs        = m.lmcs.addr;
    P.lmcsLutSize = lutSize;
  }
  const size_t lmcsBytes = lmcsTable.size() * 4;

  for( ;; )   // once, again with more room when the dependencies do not fit
  {
    const size_t cap = m.edgeCap;
    size_t       schedInts = 0;
    auto take = [&]( size_t ints ) { const size_t o = schedInts; schedInts += ints; return o; };
    const size_t oCount = take( n ), oOffset = take( n + 1 ), oList = take( cap ), oQueue = take( n ), oCounter = take( 4 );
    const size_t oStart = take( n + 1 ), oTmp = take( cap ), oNumDep = take( n );
    m.sched.ensure( ctx, schedInts * 4, gpu::Buffer::DEVICE );
    P.depCount  = m.sched.addr + oCount * 4;
    P.depOffset = m.sched.addr + oOffset * 4;
    P.depList   = m.sched.addr + oList * 4;
    P.queue     = m.sched.addr + oQueue * 4;
    P.counter   = m.sched.addr + oCounter * 4;
    P.depStart  = m.sched.addr + oStart * 4;
    P.depTmp    = m.sched.addr + oTmp * 4;
    P.numDep    = m.sched.addr + oNumDep * 4;
    P.edgeCap   = (int32_t) cap;

    const size_t upBytes = sizeof( Params ) + tuBytes + coefBytes + lmcsBytes;
    m.upload.ensure( ctx, upBytes, gpu::Buffer::UPLOAD );
    memcpy( m.upload.mapped, &P, sizeof( Params ) );
    memcpy( (char*) m.upload.mapped + sizeof( Params ), m.rec.data(), tuBytes );
    memcpy( (char*) m.upload.mapped + sizeof( Params ) + tuBytes, m.coef.data(), coefBytes );
    memcpy( (char*) m.upload.mapped + sizeof( Params ) + tuBytes + coefBytes, lmcsTable.data(), lmcsBytes );
    // downloaded: the dependency count (overflow check), with verification the CPU comparison data and the GPU's
    // scheduling data, then the planes
    size_t downBytes = 8;
    const size_t dPred = downBytes, dResi = dPred + dbgBytes, dTus = dResi + dbgBytes, dOffset = dTus + tuBytes;
    const size_t dList = dOffset + ( n + 1 ) * 4;
    if( !m.on )
    {
      downBytes = dList + cap * 4;
    }
    size_t dPlane[3] = {};
    for( int c = 0; c < numComp; c++ )
    {
      dPlane[c] = downBytes;
      downBytes += planeBytes[c];
    }
    m.download.ensure( ctx, downBytes, gpu::Buffer::DOWNLOAD );

    gpu::Stream& s = *m.stream;
    s.begin();
    s.copy( m.upload, 0, m.params, 0, sizeof( Params ) );
    s.copy( m.upload, sizeof( Params ), m.tus, 0, tuBytes );
    if( coefBytes )
    {
      s.copy( m.upload, sizeof( Params ) + tuBytes, m.coeffs, 0, coefBytes );
    }
    if( lmcsBytes )
    {
      s.copy( m.upload, sizeof( Params ) + tuBytes + coefBytes, m.lmcs, 0, lmcsBytes );
    }
    for( int c = 0; c < numComp; c++ )
    {
      s.fill( m.owner[c], 0, ownerBytes[c], 0xFFFFFFFFu );
      s.fill( m.planes[c], 0, ( planeBytes[c] + 3 ) & ~size_t( 3 ), 0 );
    }
    s.fill( m.sched, oQueue * 4, n * 4, 0xFFFFFFFFu );
    s.fill( m.sched, oCounter * 4, 4 * 4, 0 );
    s.fill( m.sched, oNumDep * 4, n * 4, 0 );
    s.barrier();
    const uint32_t t0 = s.timestamp();
    const uint32_t depGroups = (uint32_t) ( ( n + 63 ) / 64 );
    Push push{ m.params.addr, (int32_t) n, m.on ? 0 : 1 };
    s.dispatch( m.sh->kernelResi, &push, (uint32_t) std::min<size_t>( RESI_GROUPS, n ) );   // the residuals
    Push pass{ m.params.addr, (int32_t) n, 0 };
    s.dispatch( m.sh->kernelDeps, &pass, depGroups );   // the owner maps
    s.barrier();
    pass.verify = 1;
    s.dispatch( m.sh->kernelDeps, &pass, depGroups );   // availability, read ranges, dependency counts
    s.barrier();
    PushScan scan{ P.depCount, P.depStart, (int32_t) n, 0 };
    s.dispatch( m.sh->kernelScan, &scan, 1 );
    s.barrier();
    pass.verify = 2;
    s.dispatch( m.sh->kernelDeps, &pass, depGroups );   // the dependencies, the dependents' counts
    s.barrier();
    scan = PushScan{ P.numDep, P.depOffset, (int32_t) n, 0 };
    s.dispatch( m.sh->kernelScan, &scan, 1 );
    s.barrier();
    s.fill( m.sched, oNumDep * 4, n * 4, 0 );
    s.barrier();
    pass.verify = 3;
    s.dispatch( m.sh->kernelDeps, &pass, depGroups );   // the dependents lists, the queue
    s.barrier();
    s.dispatch( m.sh->kernel, &push, (uint32_t) std::min<size_t>( m.numGroups, n ) );
    const uint32_t t1 = s.timestamp();
    s.barrier();
    s.copy( m.sched, ( oStart + n ) * 4, m.download, 0, 4 );
    if( !m.on )
    {
      s.copy( m.dbgPred, 0, m.download, dPred, dbgBytes );
      s.copy( m.dbgResi, 0, m.download, dResi, dbgBytes );
      s.copy( m.tus, 0, m.download, dTus, tuBytes );
      s.copy( m.sched, oOffset * 4, m.download, dOffset, ( n + 1 ) * 4 );
      s.copy( m.sched, oList * 4, m.download, dList, cap * 4 );
    }
    for( int c = 0; c < numComp; c++ )
    {
      s.copy( m.planes[c], 0, m.download, dPlane[c], planeBytes[c] );
    }
    s.submitAndWait();
    const size_t numEdges = (size_t) *(const int32_t*) m.download.mapped;
    if( numEdges > cap )   // nothing was reconstructed: again with room for them
    {
      m.edgeCap = numEdges + numEdges / 4;
      continue;
    }
    if( numEdges > cap - cap / 4 )
    {
      m.edgeCap = numEdges + numEdges / 4;   // the next pictures
    }
    ms = s.elapsedNs( t0, t1 ) * 1e-6;
    gPlanes  = (const int16_t*) ( (const char*) m.download.mapped + dPlane[0] );
    gPredDl  = (const int16_t*) ( (const char*) m.download.mapped + dPred );
    gTusDl   = (const Tu*) ( (const char*) m.download.mapped + dTus );
    gOffDl   = (const int32_t*) ( (const char*) m.download.mapped + dOffset );
    gListDl  = (const int32_t*) ( (const char*) m.download.mapped + dList );
    break;
  }
  m.gpuMs += ms;

  if( m.on )   // the GPU's picture is the reconstruction
  {
    const int16_t* gPlane = gPlanes;
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
  const int16_t* gPred = gPredDl;
  const int16_t* gResi = gPred + m.cpuPred.size();
  std::string first;
  int numBad = 0;
  // the GPU's scheduling data against the host's: availability, read ranges, the dependency graph
  {
    std::vector<std::vector<int>> dependents( numTus );
    for( int i = 0; i < numTus; i++ ) for( int d: deps[i] ) dependents[d].push_back( i );
    for( int i = 0; i < numTus && first.empty(); i++ )
    {
      const Tu& h = m.rec[i];
      const Tu& g = gTusDl[i];
      std::vector<int> gd( gListDl + gOffDl[i], gListDl + gOffDl[i + 1] );
      std::sort( gd.begin(), gd.end() );
      gd.erase( std::unique( gd.begin(), gd.end() ), gd.end() );
      std::sort( dependents[i].begin(), dependents[i].end() );
      if( memcmp( h.avail, g.avail, sizeof( h.avail ) ) || memcmp( h.readMax, g.readMax, sizeof( h.readMax ) ) || gd != dependents[i] )
      {
        first = "scheduling data of block " + std::to_string( i ) + " differ (availability, read ranges or dependents)";
      }
    }
  }
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
  const int16_t* gPlane = gPlanes;
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
