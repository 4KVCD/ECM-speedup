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

/** \file     VkLoopFilter.h
    \brief    In-loop filters of the GPU decoder
*/

#pragma once

#include "VkCore.h"

#include "CommonLib/Unit.h"
#include "CommonLib/Buffer.h"
#include "CommonLib/LoopFilter.h"
#include "CommonLib/AlfGpuPlan.h"

#include <map>

#include <string>
#include <vector>

class AdaptiveLoopFilter;
class CodingStructure;
class Picture;
class SampleAdaptiveOffset;

namespace vkd
{

/// A picture in GPU memory: the planes of one picture with a margin around each, in a single buffer.
class GpuPicture
{
public:
  struct Plane
  {
    int origin = 0;   ///< index of sample (0, 0)
    int stride = 0;
    int width  = 0;
    int height = 0;
  };

  void create( Core& core, ChromaFormat format, int width, int height, int margin );
  void destroy( Core& core ) { core.destroyBuffer( m_buf ); }
  bool matches( ChromaFormat format, int width, int height ) const
  {
    return m_buf.buffer && format == m_format && width == m_plane[0].width && height == m_plane[0].height;
  }

  void upload( const CPelUnitBuf& src );
  void download( PelUnitBuf& dst ) const;
  /// Compares with ref; on a difference, describes the first one in diff, sets its component and position and
  /// returns false.
  bool compare( const CPelUnitBuf& ref, std::string& diff, int& comp, int& x, int& y ) const;

  const Buffer& buffer() const { return m_buf; }
  const Plane&  plane( int c ) const { return m_plane[c]; }
  int           numComp() const { return m_numComp; }
  ChromaFormat  format() const { return m_format; }

private:
  Buffer       m_buf;
  Plane        m_plane[MAX_NUM_COMPONENT];
  int          m_numComp = 0;
  ChromaFormat m_format  = CHROMA_420;
};

/// Runs the in-loop filters of a picture on the GPU: deblocking, SAO with the bilateral filter and CCSAO, the offset
/// refinements, LF-CCCM, ALF and the temporal ALF. The CPU decoder hands over the parameters each GPU stage needs and
/// runs a stage itself when the GPU does not cover a picture.
///
/// In check mode the CPU also runs every GPU stage. Each GPU stage then starts from the CPU's input and its result
/// must equal the CPU's (checkAgainstCpu). Otherwise the picture stays on the GPU between GPU stages, and
/// syncToCpu copies it back before a stage that runs on the CPU.
class VkLoopFilter
{
public:
  void init( Core* core, bool check );
  void destroy();
  bool checking() const { return m_check; }

  /// Starts a picture whose reconstruction is on the CPU.
  void beginPicture( CodingStructure& cs );
  /// Makes the CPU picture current, before a stage that runs on the CPU. Unless cpuWrites is false, the GPU copy is
  /// stale afterwards outside check mode.
  void syncToCpu( CodingStructure& cs, bool cpuWrites = true );

  /// The edge maps for LoopFilter::setEdgeMaps, cleared, before LoopFilter::loopFilterPic.
  DeblockEdgeMaps& deblockPrepare( CodingStructure& cs );
  /// Deblocks the picture with the recorded edge parameters.
  void deblock( CodingStructure& cs );

  /// SAO, bilateral filter, CCSAO and their joint clipping, as SAOProcess, CCSAOProcess and jointClipSaoBifCcSao.
  void sao( CodingStructure& cs, SampleAdaptiveOffset& sao );

  /// ALF, as AdaptiveLoopFilter::ALFProcess. alfPrepare lets ALFProcess record the picture into a plan (and, outside
  /// check mode, skip the sample work); after ALFProcess, alf runs the plan on the GPU and returns true. Pictures that
  /// the plan does not cover keep the CPU's result, and alf returns false. The CPU picture must be current before
  /// ALFProcess.
  void alfPrepare( CodingStructure& cs, AdaptiveLoopFilter& alf );
  bool alf( CodingStructure& cs, AdaptiveLoopFilter& alf );

  /// LF-CCCM (LoopFilterCccm::lfCccmCtuProcess for every CTU) on the picture after SAO: the chroma correction that
  /// lfCccmApply adds after ALF. lfCccmCheck compares the correction with the CPU's in check mode.
  void lfCccm( CodingStructure& cs, int impMode );
  void lfCccmApply( CodingStructure& cs );
  void lfCccmCheck( CodingStructure& cs, const PelUnitBuf& cpuCorrection );

#if JVET_AK0065_TALF
  /// The temporal ALF, as AdaptiveLoopFilter::TAlfProcess, after ALF and the LF-CCCM correction. Returns false if
  /// TAlfProcess has to run on the CPU (talfCpuReasons says why), e.g. after an ALF on the CPU.
  bool talf( CodingStructure& cs, AdaptiveLoopFilter& alf );
#endif

  /// The deblocking's coding information (luma), which the offset refinement and ALF read; after loopFilterPic.
  void uploadCodingInfo( const CPelBuf& ci );
  /// The offset refinement of AdaptiveLoopFilter::calcOffsetRefinement and copyOffsetRefinement, after deblocking
  /// (stageIdx 0) or after ALF (1).
  void offsetRefinement( CodingStructure& cs, int stageIdx, int refineIdx, AdaptiveLoopFilter& alf );

  /// Check mode: compares the GPU result of the last stage with the CPU picture; throws on a difference.
  void checkAgainstCpu( CodingStructure& cs, const char* stage );

  double gpuMs() const { return m_gpuMs; }
  /// GPU time per stage, in the order of stageName.
  enum Stage
  {
    STAGE_DEBLOCK,
    STAGE_SAO,
    STAGE_REFINE,
    STAGE_LFCCCM,
    STAGE_ALF,
    STAGE_TALF,
    NUM_STAGES
  };
  static const char* stageName( int stage );
  double             stageMs( int stage ) const { return m_stageMs[stage]; }

  /// Check mode: how much work each GPU stage did, so that a check run shows which paths it covered.
  struct Counts
  {
    uint64_t pictures       = 0;
    uint64_t deblockSegments = 0;   ///< luma and chroma edge segments with a boundary strength
    uint64_t saoCtus[5]     = {};  ///< CTU components with SAO on, by type: edge 0, 90, 135, 45 degrees, band
    uint64_t bifBlocks      = 0;
    uint64_t ccSaoBand[9]   = {};  ///< CTU components with CCSAO on, by band candidate position
    uint64_t ccSaoEdge[4]   = {};  ///< and by edge direction
    uint64_t alfPictures    = 0;   ///< pictures whose ALF ran on the GPU
    uint64_t alfCpuPictures = 0;   ///< pictures with ALF that the plan does not cover (alfCpuReasons)
    uint64_t alfLumaCtus[6] = {};  ///< luma CTUs: fixed filter set 0 or 1 only, APS filter types 9-12
    uint64_t alfClassifier[3] = {};   ///< luma CTUs with an APS filter, by classifier
    uint64_t alfScaleCtus   = 0;   ///< luma CTUs with the residual scaling correction
    uint64_t alfChromaCtus[2] = {};   ///< chroma CTUs with ALF_FILTER_9, ALF_FILTER_9_NO_FIX
    uint64_t ccAlfCtus      = 0;
    uint64_t alfChromaScalePictures = 0;
    uint64_t alfMirrors     = 0;   ///< mirrored padding regions of the result planes
    uint64_t refinePictures[2] = {};  ///< pictures with the offset refinement after deblocking, after ALF
    uint64_t lfCccmPictures = 0;
    uint64_t lfCccmWindows  = 0;
    uint64_t lfCccmModelTypes[8]  = {};   ///< CTUs by model type
    uint64_t lfCccmWindowSizes[8] = {};   ///< CTUs by window size index
    uint64_t lfCccmImpModes[5]    = {};   ///< pictures by update factor (impMode)
    uint64_t lfCccmBranches[6]    = {};   ///< steps with a smaller model, too few samples, an invalid model, a stop at
                                          ///< a deviating sample, halves with sub-models, all steps
    uint64_t talfPictures   = 0;   ///< pictures whose temporal ALF ran on the GPU
    uint64_t talfCpuPictures = 0;  ///< pictures whose temporal ALF ran on the CPU (talfCpuReasons)
    uint64_t talfBlocks     = 0;   ///< filtered 4x4 blocks
    uint64_t talfModes[6]   = {};  ///< pictures with filtered blocks, by TAlfFilterMode
    uint64_t talfShapes[2]  = {};  ///< filtered blocks by template shape
  };
  const Counts& counts() const { return m_counts; }
  /// Why ALF stayed on the CPU, with the number of pictures.
  const std::map<std::string, uint64_t>& alfCpuReasons() const { return m_alfCpuReasons; }
  /// Why the temporal ALF stayed on the CPU, with the number of pictures.
  const std::map<std::string, uint64_t>& talfCpuReasons() const { return m_talfCpuReasons; }

private:
  void ensureOnGpu( CodingStructure& cs );
  void ensureBuffer( Buffer& buf, size_t size, Mem mem );
  std::string describeLuma( int x, int y, const CPelUnitBuf& cpu ) const;

  Core*            m_core  = nullptr;
  bool             m_check = false;
  bool             m_cpuCurrent = true;
  bool             m_gpuCurrent = false;
  int              m_lastStage  = 0;     ///< 1 after deblocking, for describeLuma
  GpuPicture       m_pic;
  GpuPicture       m_tmp;               ///< scratch picture, e.g. the input of SAO
  std::vector<Pel> m_input;             ///< check mode: the input of the last stage, for describeLuma

  Buffer           m_edgeBuf;           ///< deblocking: luma maps (vertical, horizontal), then chroma maps
  DeblockEdgeMaps  m_maps;
  size_t           m_lumaMapBytes   = 0;
  size_t           m_chromaMapBytes = 0;

  Buffer           m_saoParams;         ///< SaoCtu per CTU and component
  Buffer           m_bifBlocks;
  Buffer           m_bifTables;
  Buffer           m_ccSaoSets;
  Buffer           m_ctuBytes;          ///< CCSAO set per CTU and component, CTU availability, clip flags
  Buffer           m_ccSaoTables;
  std::vector<uint8_t> m_scratch;

  // ALF
  void alfBuffers( CodingStructure& cs, AdaptiveLoopFilter& alf );
  bool alfComparePlanes( AdaptiveLoopFilter& alf, std::string& diff );
  void alfRun( CodingStructure& cs, AdaptiveLoopFilter& alf );

  AlfGpuPlan       m_alfPlan;
  GpuPicture       m_db;                ///< the picture before deblocking
  GpuPicture       m_src;               ///< the ALF input
  GpuPicture       m_resi;              ///< the luma residual
  Buffer           m_codingInfo;        ///< the deblocking's coding information, luma
  Buffer           m_lfCorr;            ///< LF-CCCM: the updated chroma, then the correction; the picture's layout
  Buffer           m_lfDown;            ///< LF-CCCM: the downsampled luma, padded by 1
  Buffer           m_lfWindows;
  Buffer           m_lfStats;           ///< check mode: lfcccm_window.comp branch counters
  Buffer           m_refineCtus;        ///< offset refinement after ALF: CTUs with luma ALF on
  bool             m_alfInputOnGpu = false;   ///< m_src holds this picture's ALF input
  Buffer           m_alfRes;            ///< the result planes of AlfGpuPlan::Plane
  size_t           m_alfResOffset[AlfGpuPlan::NUM_PLANES] = {};
  int              m_alfResWidth[AlfGpuPlan::NUM_PLANES]  = {};
  int              m_alfResHeight[AlfGpuPlan::NUM_PLANES] = {};
  Buffer           m_alfCls;            ///< luma and chroma classes
  Buffer           m_alfTables;         ///< the fixed filter tables
  Buffer           m_alfParams;         ///< per picture: parameters, CTUs, filters, tiles
  std::map<std::string, uint64_t> m_alfCpuReasons;

#if JVET_AK0065_TALF
  // The temporal ALF reads the luma of reference pictures with the border that finishPicture padded. Copies of the
  // last few on the GPU, by picture; beginPicture drops the copy of a picture object that holds a new picture.
  struct TalfRef
  {
    const Picture* pic     = nullptr;
    Buffer         buf;
    int            origin  = 0;
    int            stride  = 0;
    uint64_t       lastUse = 0;
  };
  const TalfRef& talfRef( const Picture* pic );
  void           talfForget( const Picture* pic );
  TalfGpuPlan          m_talfPlan;
  std::vector<TalfRef> m_talfRefs;
  uint64_t             m_talfUse = 0;
  Buffer               m_talfParams;   ///< per picture: blocks, filters, references
#endif
  std::map<std::string, uint64_t> m_talfCpuReasons;

  void             addGpuTime( Stage stage, int t0, int t1 );
  double           m_gpuMs = 0;
  double           m_stageMs[NUM_STAGES] = {};
  Counts           m_counts;
};

}   // namespace vkd
