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

/** \file     AlfGpuPlan.h
    \brief    What ALFProcess computes for one picture, recorded for the GPU decoder
*/

#pragma once

#include "CommonDef.h"

#include <cstdint>
#include <vector>

/// What ALFProcess computes for one picture, recorded so that the GPU decoder can do the sample work.
///
/// ALFProcess walks the CTUs as usual and records the parameters of each CTU and each region it filters; with
/// skipSamples set it leaves the samples alone. Its intermediate result planes (fixed filter, gauss and laplacian
/// results) are allocated anew for every picture, and in some layouts of CTU filter types ALFProcess reads samples
/// of them that the picture has not written, i.e. uninitialized memory. The plan tracks which samples the picture has
/// written so far and marks the picture stale at any other read; ALFProcess then processes the picture on the CPU
/// after all, as the stock decoder does.
///
/// The plan covers pictures in 4:2:0 or 4:0:0 with one slice, no virtual boundaries and no CU-level QP for the
/// fixed filters; ALFProcess processes other pictures on the CPU (cpuReason says why).
struct AlfGpuPlan
{
  /// The result planes, in the order of AdaptiveLoopFilter::resultPlanes.
  enum Plane
  {
    PLANE_FIX_LUMA  = 0,    ///< 4 planes, index classifierIdx * 2 + fixedFilterSetIdx: first stage 0-1, second 2-3
    PLANE_FIX_CB    = 4,    ///< 2 planes, index fixedFilterSetIdx
    PLANE_FIX_CR    = 6,
    PLANE_FIX_RESI  = 8,    ///< 2 planes, the fixed filter of the residual
    PLANE_GAUSS     = 10,
    PLANE_LAPLACIAN = 11,
    NUM_PLANES      = 12
  };
  static bool planeIsChroma( int plane ) { return plane >= PLANE_FIX_CB && plane < PLANE_FIX_RESI; }
  static int  planePadding( int plane )
  {
    return plane < PLANE_FIX_RESI ? ALF_PADDING_SIZE_FIXED_RESULTS : plane < PLANE_GAUSS ? 0 : ALF_PADDING_SIZE_GAUSS_RESULTS;
  }

  /// A region of a plane in the plane's picture coordinates. Mirror regions lie in the padding beyond one picture
  /// edge; axis 0 mirrors at the left or right edge, axis 1 at the top or bottom edge.
  struct Region
  {
    int16_t plane;
    int16_t axis;
    int32_t x, y, w, h;
  };

  enum LumaMode : int8_t
  {
    LUMA_OFF,
    LUMA_COPY_STAGE1,   ///< fixed filter set 0: the first-stage fixed filter result
    LUMA_COPY_STAGE2,   ///< fixed filter set 1: the second-stage fixed filter result
    LUMA_FILTER         ///< an APS filter, lumaFilter
  };
  struct Ctu
  {
    int8_t  lumaMode        = LUMA_OFF;
    int16_t lumaFilter      = -1;
    int16_t chromaFilter[2] = { -1, -1 };   ///< -1: chroma ALF off
    int16_t ccAlfFilter[2]  = { -1, -1 };   ///< -1: CCALF off
  };

  /// One alternative of an APS luma filter as ALFProcess applies it.
  struct LumaFilter
  {
    int     type;         ///< AlfFilterType
    int     classifier;   ///< 0-2
    int     coeffBits;
    short   coeff[MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF];
    short   clip[MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF];   ///< clipping values, not indices
    int     scaleIdx[MAX_NUM_ALF_CLASSES];
    int     scaleCorr[MAX_NUM_ALF_CLASSES];   ///< factor of alfAddCorrect per class; 0 without the scale mode
  };
  struct ChromaFilter
  {
    int   type;   ///< ALF_FILTER_9 or ALF_FILTER_9_NO_FIX
    int   scaleIdx;
    short coeff[MAX_NUM_ALF_CHROMA_COEFF];
    short clip[MAX_NUM_ALF_CHROMA_COEFF];
  };
  struct CcAlfFilter
  {
    int   coeffPrec;
    short coeff[MAX_NUM_CC_ALF_CHROMA_COEFF];
  };

  // ---- set by the owner ----
  bool                  skipSamples = false;
  /// With skipSamples: ALFProcess still copies its inputs (the picture before ALF, with padding), which later CPU
  /// stages such as the temporal ALF read.
  bool                  keepCpuInputs = true;

  // ---- recorded by ALFProcess ----
  bool        gpu       = false;     ///< the GPU processes this picture
  const char* cpuReason = nullptr;   ///< why the CPU processes it, when gpu is false
  bool        stale     = false;     ///< ALFProcess read result samples the picture had not written

  int  width = 0, height = 0;        ///< luma picture size
  bool hasChroma = false;
  int  bitDepth[MAX_NUM_CHANNEL_TYPE] = {};
  bool intraSlice        = false;
  bool fixedFilterAdjust = false;    ///< SPS ALF luma fixed filter adjustment
  int  fixedSetIdx[MAX_NUM_COMPONENT] = {};          ///< slice fixed filter set flag per component
  int  fixedSet[MAX_NUM_COMPONENT][2]     = {};      ///< fixed filter set (0-7) for flag 0 and 1, from the QP
  int  clipValues[MAX_NUM_CHANNEL_TYPE][4] = {};
  int  chromaScaleCorr[2] = {};      ///< alfAddCorrectChroma factor of Cb and Cr; 0: off
  bool lumaScaleMode = false;         ///< alfAddCorrect after the luma APS filters (SPS ALF scale mode)

  std::vector<Ctu>          ctus;
  std::vector<LumaFilter>   lumaFilters;
  std::vector<ChromaFilter> chromaFilters;
  std::vector<CcAlfFilter>  ccAlfFilters;
  std::vector<Region>       writes;    ///< computed regions, in the CPU's order
  std::vector<Region>       mirrors;   ///< mirrored padding regions, in the CPU's order

  /// Starts recording a picture.
  void begin( int picWidth, int picHeight, ChromaFormat format, int numCtus );
  void unsupported( const char* reason ) { gpu = false; cpuReason = reason; }

  /// Records a computed region (inside the picture).
  void write( int plane, int x, int y, int w, int h );
  /// Records a mirrored padding region; mirror regions of axis 1 can read the padding that axis 0 regions wrote.
  void mirror( int plane, int axis, int x, int y, int w, int h );
  /// Read checks. The picture becomes stale if a sample read was not written in this picture.
  void readRect( int plane, int x, int y, int w, int h );
  /// Samples within the given distance along a row or column of the region, and within crossDist in both directions.
  void readCross( int plane, int x, int y, int w, int h, int axisDist, int crossDist );
  /// Samples within the given city-block distance of the region.
  void readDiamond( int plane, int x, int y, int w, int h, int dist );

private:
  /// Which samples of a plane the picture has written, per 2x2 cell of the padded plane: every region that
  /// ALFProcess computes or mirrors starts and ends on even coordinates. A cell holds the epoch of the picture that
  /// last wrote it, so that the masks need no clearing between pictures.
  struct Mask
  {
    int                  pad = 0, cellsW = 0, cellsH = 0;
    std::vector<uint8_t> epoch;
  };
  void markCells( int plane, int x, int y, int w, int h );
  void checkRow( int plane, int y, int x0, int x1 );   ///< samples [x0, x1) of row y

  Mask    m_mask[NUM_PLANES];
  int     m_maskW = 0, m_maskH = 0;
  bool    m_maskChroma = false;
  uint8_t m_epoch = 0;
};

#if JVET_AK0065_TALF
class Picture;

/// What TAlfProcess filters in one picture, recorded by AdaptiveLoopFilter::talfGpuPlan so that the GPU decoder can
/// filter the samples: the luma 4x4 blocks that getMotionOffset accepts, with their reference blocks and filters.
struct TalfGpuPlan
{
  struct Filter
  {
    int32_t coeff[NUM_TALF_COEFF];
    int32_t clip[NUM_TALF_COEFF];   ///< the clipping value of each coefficient
    int32_t shift;
    int32_t shape;                  ///< 0: templateShape0, 1: templateShape1
  };
  struct Block
  {
    int16_t x, y;           ///< luma position
    int16_t mv[2][2];       ///< integer offsets of the reference blocks
    uint8_t ref[2];         ///< indices into refs; ref[1] only with bi
    uint8_t filter;         ///< index into filters
    uint8_t pad;
  };

  bool                        gpu = false;      ///< false: TAlfProcess filters on the CPU (cpuReason says why)
  const char*                 cpuReason = nullptr;
  bool                        bi = false;       ///< the average of two reference blocks, else one
  int                         mode = 0;         ///< TAlfFilterMode
  std::vector<const Picture*> refs;             ///< the unscaled reference pictures that the blocks read
  std::vector<Filter>         filters;
  std::vector<Block>          blocks;

  void unsupported( const char* reason ) { gpu = false; cpuReason = reason; }
};
#endif
