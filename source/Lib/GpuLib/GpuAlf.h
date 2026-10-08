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

/** \file     GpuAlf.h
    \brief    adaptive loop filter (ALFProcess) on a Vulkan compute device, bit-exact with the CPU version
*/

#ifndef __GPUALF__
#define __GPUALF__

#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/SampleAdaptiveOffset.h"
#include "VkCompute.h"

#include <functional>
#include <memory>
#include <vector>

/// SAO, the bilateral filter and CCSAO (SampleAdaptiveOffset::SAOProcess, CCSAOProcess, jointClipSaoBifCcSao) of the
/// picture that ALFProcess filters next: the decoder defers them to ALFProcess, which runs them on the GPU from the
/// deblocked picture, or on the CPU first
struct GpuSaoJob
{
  std::function<void()> cpu;              ///< runs them on the CPU, on the picture
  SampleAdaptiveOffset* sao = nullptr;
};

/// LF-CCCM (JVET-AL0153) of the picture that ALFProcess filters next: the chroma correction that the decoder derives
/// from the picture before ALF and adds to the picture after ALF
struct GpuLfCccmJob
{
  std::function<void()> cpu;                    ///< derives the correction on the CPU, into `correction`
  const PelStorage*     correction = nullptr;   ///< Cb, Cr: the correction cpu() derived
  bool                  applied    = false;     ///< the GPU added the correction to the filtered picture
};

/// the GPU resources to filter one picture at a time (command stream, buffers); see GpuAlfHook
class GpuAlf
{
public:
  enum Mode
  {
    OFF    = 0,
    ON     = 1,   ///< the GPU filters the pictures it supports
    VERIFY = 2    ///< the CPU filters every picture, the GPU result is compared with it
  };

  /// smaller pictures are filtered on the CPU: the fixed costs of the device exceed what it saves
  static bool    worthwhile( int width, int height ) { return width * height >= 640 * 360; }
  /// starts creating the device, the kernels and (width > 0) a first GpuAlf with buffers for pictures of this size
  /// in a background thread
  static void    prefetch( int mode, int width = 0, int height = 0, int ctuSize = 0 );
  /// prefetch() has finished: GpuAlfHook::create() does not wait
  static bool    ready();

  ~GpuAlf();
  /// with `sao`, SAO first; with `lfCccm`, also the LF-CCCM correction (the GPU adds it, or lfCccm->cpu() derives it
  /// before the CPU filters)
  bool process( AdaptiveLoopFilter& alf, CodingStructure& cs, GpuSaoJob* sao, GpuLfCccmJob* lfCccm );

private:
  friend class GpuAlfHook;
  friend struct GpuAlfPool;

  /// a 16-bit plane in a buffer: width x height samples and `pad` samples of margin on every side
  struct PlaneDesc
  {
    int          width = 0, height = 0, pad = 0, stride = 0;
    VkDeviceSize offset = 0;
    int          origin() const { return pad * stride + pad; }
    VkDeviceSize bytes() const { return VkDeviceSize( stride ) * ( height + 2 * pad ) * sizeof( int16_t ); }
  };

  /// what the kernels need of the filter state of a picture
  struct Params
  {
    std::vector<uint32_t> ctuInfo;       ///< [4] per CTU: luma mode, Cb, Cr (see alf_luma.comp, alf_chroma.comp)
    std::vector<int16_t>  lumaSets;      ///< coefficient sets of the luma CTUs
    std::vector<int16_t>  chroma;        ///< [8 alternatives][64]
    std::vector<int16_t>  ccalfCoeff[2]; ///< [16 filters][34] per chroma component
    bool                  luma       = false;   ///< any luma CTU filtered
    bool                  lumaFixed  = false;   ///< fixed-filter outputs needed
    bool                  chromaAlf[2] = { false, false };
    bool                  ccalf[2]   = { false, false };   ///< any CTU with CCALF
    bool                  chromaFixed = false;  ///< chroma ALF with the taps on the fixed-filter output
    int                   ccalfPrec[2] = { 0, 0 };
    int                   scaleCorr[2] = { 0, 0 };
    bool                  lfCccm     = false;   ///< the kernels derive the LF-CCCM correction
    /// [4] per CTU with LF-CCCM: address, model type | window size index << 8, first window
    std::vector<uint32_t> lfCtus;
    int                   lfWindows  = 0;       ///< windows of all these CTUs
    bool                  sao        = false;   ///< the kernels do SAO, the bilateral filter and CCSAO
    std::vector<uint32_t> saoInfo;              ///< [4] per CTU: word c: SAO type + 1 | CCSAO set << 4 (sao_apply.comp)
    std::vector<int16_t>  saoOffsets;           ///< [CTU][3][32]
    std::vector<int16_t>  ccsao;                ///< [3][4 sets][72] (sao_apply.comp)
    std::vector<uint32_t> bifBlocks;            ///< [4] per block with the bilateral filter (sao_bif.comp)
    bool                  bif[3]     = { false, false, false };   ///< blocks of the component
    int                   bifRoundAdd = 0, bifRoundShift = 0;
  };

  struct Shared;   ///< kernels and fixed-filter tables, one set for every GpuAlf of the process

  GpuAlf( gpu::Context& ctx, const Shared& shared, int mode );
  static const Shared& shared( gpu::Context& ctx );

  bool accept( const AdaptiveLoopFilter& alf, const CodingStructure& cs ) const;
  /// what ALFProcess does besides filtering, before and after the filtering of the CTUs; copies: the picture copies
  /// (not needed when the kernels do SAO)
  void stateBefore( AdaptiveLoopFilter& alf, CodingStructure& cs ) const;
  void copiesBefore( AdaptiveLoopFilter& alf, CodingStructure& cs ) const;
  void stateAfter( AdaptiveLoopFilter& alf, CodingStructure& cs ) const;
  /// false when the filter state uses what the kernels do not implement
  bool gather( const AdaptiveLoopFilter& alf, const CodingStructure& cs );
  /// the windows of the CTUs with LF-CCCM; false when the picture uses what the kernels do not implement
  bool gatherLfCccm( const CodingStructure& cs );
  /// the SAO, bilateral filter and CCSAO parameters (reconstruct: the SAO merges, which SAOProcess resolves); false
  /// when the picture uses none of them or what the kernels do not implement
  bool gatherSao( CodingStructure& cs, SampleAdaptiveOffset& sao, bool reconstruct );
  void layout( int width, int height, int ctuSize );
  /// deblocked: the picture before SAO when the kernels do it
  void upload( const AdaptiveLoopFilter& alf, const CodingStructure& cs, const CPelUnitBuf& deblocked );
  void run( const AdaptiveLoopFilter& alf, const CodingStructure& cs );
  void output( AdaptiveLoopFilter& alf, CodingStructure& cs ) const;
  bool verify( const AdaptiveLoopFilter& alf, const CodingStructure& cs, const GpuLfCccmJob* lfCccm );

  gpu::Context&                m_ctx;
  const Shared&                m_k;
  int                          m_mode;
  std::unique_ptr<gpu::Stream> m_stream;

  // picture geometry of the current layout
  int                          m_width = 0, m_height = 0, m_ctuSize = 0, m_ctusPerRow = 0, m_numCtus = 0;
  int                          m_wBlk = 0, m_hBlk = 0;   ///< 2x2 luma blocks
  int                          m_wBlkC = 0, m_hBlkC = 0; ///< 2x2 chroma blocks

  Params                       m_params;

  // inputs and parameters: device buffer and an upload buffer with the same layout; m_inRec (the picture after SAO,
  // mirror-padded) is uploaded, or the SAO kernels write it from m_inDeblocked
  PlaneDesc                    m_inDeblocked[3], m_inRec[3], m_inDb[3], m_inResi, m_inCi;
  VkDeviceSize                 m_inCtuInfo = 0, m_inLumaSets = 0, m_inChroma = 0, m_inCcalf[2] = { 0, 0 };
  VkDeviceSize                 m_inLfCtus = 0, m_inSaoInfo = 0, m_inSaoOffsets = 0, m_inCcsao = 0;
  VkDeviceSize                 m_inBytes = 0;
  gpu::Buffer                  m_in, m_inUpload, m_inDownload;
  gpu::Buffer                  m_bifBlocks, m_bifBlocksUpload;   ///< the bilateral filter blocks (size varies)
  // intermediate results: classes per 2x2 block, filter outputs (picture size, mirrored when read)
  VkDeviceSize                 m_ciClass = 0, m_fixClass0 = 0, m_fixClass1 = 0, m_mainClass[3] = { 0, 0, 0 };
  VkDeviceSize                 m_chClass[2] = { 0, 0 };
  PlaneDesc                    m_fix0, m_fix1, m_fixResi, m_gauss, m_lapl, m_chFix[2];
  PlaneDesc                    m_lfDs, m_lfCorr[2];   ///< LF-CCCM: downsampled luma, Cb and Cr corrections
  PlaneDesc                    m_bifOffsets[3];       ///< the bilateral filter's offsets
  gpu::Buffer                  m_work;
  // outputs, read back; m_saoDownload: the luma after SAO, for the CPU stages after ALF that read it
  PlaneDesc                    m_out[3];
  gpu::Buffer                  m_outDev, m_download, m_workDownload, m_saoDownload;
  bool                         m_saoLumaDownloaded = false;
  PelStorage                   m_verifyDeblocked;   ///< the verification's copy of the picture before SAO
};

/// ALFProcess on the GPU for a decoder context: every picture is filtered with a GpuAlf of a pool that all decoder
/// contexts of the process share (a picture holds one for a few milliseconds)
class GpuAlfHook : public AlfProcessHook
{
public:
  /// nullptr when there is no usable Vulkan device (the reason is reported once); waits for GpuAlf::prefetch()
  static GpuAlfHook* create( int mode );

  /// the deferred SAO and the LF-CCCM correction of the next picture, nullptr when it has none (process() takes them)
  void setSao( GpuSaoJob* job ) { m_sao = job; }
  void setLfCccm( GpuLfCccmJob* job ) { m_lfCccm = job; }
  bool process( AdaptiveLoopFilter& alf, CodingStructure& cs ) override;

private:
  GpuAlfHook( gpu::Context& ctx, int mode ) : m_ctx( ctx ), m_mode( mode ) {}

  gpu::Context& m_ctx;
  int           m_mode;
  GpuSaoJob*    m_sao    = nullptr;
  GpuLfCccmJob* m_lfCccm = nullptr;
};

#endif
