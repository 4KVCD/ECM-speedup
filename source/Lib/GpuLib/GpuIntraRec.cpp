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

#include "CommonLib/CodingStructure.h"
#include "CommonLib/InterpolationFilter.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Slice.h"
#include "CommonLib/TrQuant.h"
#include "CommonLib/UnitTools.h"

#include "intra_rec.spv.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace
{
// layout shared with intra_rec.comp (struct Tu): 32 int32
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
  int32_t pad[5];
};
static_assert( sizeof( Tu ) == 32 * 4, "Tu layout" );

// layout shared with intra_rec.comp (struct Params, std430)
struct Params
{
  uint64_t tus;
  uint64_t coeffs;
  uint64_t plane[3];
  int32_t  origin[3];
  int32_t  stride[3];
  int32_t  width[3];
  int32_t  height[3];
  uint64_t decomp[2];
  int32_t  decompW[2];
  int32_t  decompH[2];
  int32_t  unitLog2X[2];
  int32_t  unitLog2Y[2];
  uint64_t filters;
  uint64_t matrices;
  int32_t  matrixOffset[8];
  uint64_t scratch;
  uint64_t dbgPred;
  uint64_t dbgResi;
};
static_assert( offsetof( Params, decomp ) == 88 && offsetof( Params, filters ) == 136 && offsetof( Params, scratch ) == 184
                 && sizeof( Params ) == 208, "Params layout" );

struct Push
{
  uint64_t params;
  int32_t  numTus;
  int32_t  pad;
};

constexpr int FILTER_TAPS[5] = { 6, 8, 10, 12, 16 };
constexpr int SCRATCH_INTS   = 4 * 16384;
}   // namespace

struct GpuIntraRec::Impl
{
  gpu::Context*               ctx = nullptr;
  gpu::Kernel                 kernel;
  std::unique_ptr<gpu::Stream> stream;
  gpu::Buffer                 filters, matrices, scratch;
  gpu::Buffer                 params, tus, coeffs, planes[3], decomp[2], dbgPred, dbgResi;
  gpu::Buffer                 upload, download;
  int32_t                     matrixOffset[8] = {};

  // the picture being recorded
  std::vector<Tu>             rec;
  std::vector<int32_t>        coef;
  std::vector<int16_t>        cpuPred, cpuResi;
  std::string                 unsupportedWhy;
  int                         poc = 0;

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
    for( gpu::Buffer* b: { &filters, &matrices, &scratch, &params, &tus, &coeffs, &planes[0], &planes[1], &planes[2],
                           &decomp[0], &decomp[1], &dbgPred, &dbgResi, &upload, &download } )
    {
      b->destroy( *ctx );
    }
  }
};

GpuIntraRec* GpuIntraRec::create()
{
  gpu::Context* ctx = gpu::Context::get();
  if( !ctx )
  {
    return nullptr;
  }
  GpuIntraRec* r = new GpuIntraRec;
  r->m_impl.reset( new Impl );
  Impl& m = *r->m_impl;
  m.ctx = ctx;
  m.kernel.create( *ctx, spv_intra_rec, sizeof( spv_intra_rec ), sizeof( Push ), "intra_rec" );
  m.stream.reset( new gpu::Stream( *ctx ) );

  // tables, from the C++ tables themselves
  std::vector<int32_t> filt( 5 * 32 * 16, 0 );
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
  std::vector<int32_t> mat;
  const TMatrixCoeff* tables[7] = { g_trCoreDCT2P2[TRANSFORM_INVERSE][0],  g_trCoreDCT2P4[TRANSFORM_INVERSE][0],
                                    g_trCoreDCT2P8[TRANSFORM_INVERSE][0],  g_trCoreDCT2P16[TRANSFORM_INVERSE][0],
                                    g_trCoreDCT2P32[TRANSFORM_INVERSE][0], g_trCoreDCT2P64[TRANSFORM_INVERSE][0],
                                    g_trCoreDCT2P128[TRANSFORM_INVERSE][0] };
  for( int l = 0; l < 7; l++ )
  {
    const int n          = 2 << l;
    m.matrixOffset[l]    = (int32_t) mat.size();
    mat.insert( mat.end(), tables[l], tables[l] + n * n );
  }
  m.filters.create( *ctx, filt.size() * 4, gpu::Buffer::DEVICE );
  m.matrices.create( *ctx, mat.size() * 4, gpu::Buffer::DEVICE );
  m.scratch.create( *ctx, SCRATCH_INTS * 4, gpu::Buffer::DEVICE );
  m.params.create( *ctx, sizeof( Params ), gpu::Buffer::DEVICE );
  m.upload.create( *ctx, ( filt.size() + mat.size() ) * 4, gpu::Buffer::UPLOAD );
  memcpy( m.upload.mapped, filt.data(), filt.size() * 4 );
  memcpy( (char*) m.upload.mapped + filt.size() * 4, mat.data(), mat.size() * 4 );
  m.stream->begin();
  m.stream->copy( m.upload, 0, m.filters, 0, filt.size() * 4 );
  m.stream->copy( m.upload, filt.size() * 4, m.matrices, 0, mat.size() * 4 );
  m.stream->submitAndWait();
  return r;
}

GpuIntraRec::~GpuIntraRec()
{
  if( m_impl )
  {
    const Impl& m = *m_impl;
    printf( "\n[gpu-intra] %d pictures: %d verified (%d with differences), %d skipped%s%s; GPU %.1f ms\n", m.numPictures,
            m.numVerified, m.numMismatch, m.numSkipped, m.firstSkipReason.empty() ? "" : ", first reason: ",
            m.firstSkipReason.c_str(), m.gpuMs );
  }
}

void GpuIntraRec::beginPicture( const CodingStructure& cs )
{
  Impl& m = *m_impl;
  m.rec.clear();
  m.coef.clear();
  m.cpuPred.clear();
  m.cpuResi.clear();
  m.unsupportedWhy.clear();
  m.poc = cs.slice->getPOC();
  if( cs.pcv->chrFormat != CHROMA_420 && cs.pcv->chrFormat != CHROMA_400 )
  {
    unsupported( "chroma format" );
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
                          const QpParam& qp, const CPelBuf& pred, const CPelBuf& resi )
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
  else if( ipa.refFilterFlag || ipa.applyPDPC || ipa.applyFusion )  why = "reference filter / PDPC / intra fusion";
  else if( sps.getUseIntraPredBf() )                                why = "intra bilateral filter";
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

  const int n   = area.width * area.height;
  t.coeffOffset = (int32_t) m.coef.size();
  t.dbgOffset   = (int32_t) m.cpuPred.size();
  const CCoeffBuf c = tu.getCoeffs( compID );
  for( int y = 0; y < (int) area.height; y++ )
  {
    for( int x = 0; x < (int) area.width; x++ )
    {
      m.coef.push_back( c.at( x, y ) );
      m.cpuPred.push_back( pred.at( x, y ) );
      m.cpuResi.push_back( resi.at( x, y ) );
    }
  }
  CHECK( (int) m.coef.size() != t.coeffOffset + n, "coefficient count" );
  m.rec.push_back( t );
}

void GpuIntraRec::endPicture( const CodingStructure& cs )
{
  Impl& m = *m_impl;
  m.numPictures++;
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
  size_t mapBytes[2] = {};
  for( int ch = 0; ch < ( numComp > 1 ? 2 : 1 ); ch++ )
  {
    const int sx = ch == 0 ? 2 : 1, sy = ch == 0 ? 2 : 1;   // UnitScaleArray (4:2:0)
    P.unitLog2X[ch] = sx;
    P.unitLog2Y[ch] = sy;
    P.decompW[ch]   = ( P.width[ch] + ( 1 << sx ) - 1 ) >> sx;
    P.decompH[ch]   = ( P.height[ch] + ( 1 << sy ) - 1 ) >> sy;
    mapBytes[ch]    = size_t( P.decompW[ch] ) * P.decompH[ch];
    m.decomp[ch].ensure( ctx, ( mapBytes[ch] + 3 ) & ~size_t( 3 ), gpu::Buffer::DEVICE );
    P.decomp[ch] = m.decomp[ch].addr;
  }
  const size_t tuBytes = m.rec.size() * sizeof( Tu ), coefBytes = m.coef.size() * 4, dbgBytes = m.cpuPred.size() * 2;
  m.tus.ensure( ctx, tuBytes, gpu::Buffer::DEVICE );
  m.coeffs.ensure( ctx, std::max<size_t>( coefBytes, 4 ), gpu::Buffer::DEVICE );
  m.dbgPred.ensure( ctx, dbgBytes, gpu::Buffer::DEVICE );
  m.dbgResi.ensure( ctx, dbgBytes, gpu::Buffer::DEVICE );
  P.tus      = m.tus.addr;
  P.coeffs   = m.coeffs.addr;
  P.filters  = m.filters.addr;
  P.matrices = m.matrices.addr;
  memcpy( P.matrixOffset, m.matrixOffset, sizeof( P.matrixOffset ) );
  P.scratch  = m.scratch.addr;
  P.dbgPred  = m.dbgPred.addr;
  P.dbgResi  = m.dbgResi.addr;

  const size_t upBytes = sizeof( Params ) + tuBytes + coefBytes;
  m.upload.ensure( ctx, upBytes, gpu::Buffer::UPLOAD );
  memcpy( m.upload.mapped, &P, sizeof( Params ) );
  memcpy( (char*) m.upload.mapped + sizeof( Params ), m.rec.data(), tuBytes );
  memcpy( (char*) m.upload.mapped + sizeof( Params ) + tuBytes, m.coef.data(), coefBytes );
  size_t downBytes = 2 * dbgBytes;
  for( int c = 0; c < numComp; c++ )
  {
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
  for( int c = 0; c < numComp; c++ )
  {
    s.fill( m.planes[c], 0, ( planeBytes[c] + 3 ) & ~size_t( 3 ), 0 );
  }
  for( int ch = 0; ch < ( numComp > 1 ? 2 : 1 ); ch++ )
  {
    s.fill( m.decomp[ch], 0, ( mapBytes[ch] + 3 ) & ~size_t( 3 ), 0 );
  }
  s.barrier();
  const uint32_t t0 = s.timestamp();
  Push push{ m.params.addr, (int32_t) m.rec.size(), 0 };
  s.dispatch( m.kernel, &push, 1 );
  const uint32_t t1 = s.timestamp();
  s.barrier();
  size_t off = 0;
  s.copy( m.dbgPred, 0, m.download, off, dbgBytes );
  off += dbgBytes;
  s.copy( m.dbgResi, 0, m.download, off, dbgBytes );
  off += dbgBytes;
  for( int c = 0; c < numComp; c++ )
  {
    s.copy( m.planes[c], 0, m.download, off, planeBytes[c] );
    off += planeBytes[c];
  }
  s.submitAndWait();
  const double ms = s.elapsedNs( t0, t1 ) * 1e-6;
  m.gpuMs += ms;

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
  printf( "\n[gpu-intra] POC %d: %d blocks, GPU %.2f ms: %s%s%s", m.poc, (int) m.rec.size(), ms, bad ? "MISMATCH, " : "identical",
          bad ? first.c_str() : "", numBad ? ( ", " + std::to_string( numBad ) + " picture samples differ" ).c_str() : "" );
}
