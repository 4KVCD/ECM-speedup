/*/* The copyright in this software is being made available under the BSD
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


#ifndef INTRA_PREDICTION_NEURAL_NETWORK_H
#define INTRA_PREDICTION_NEURAL_NETWORK_H

#include <array>
#include <map>
#include "CommonDef.h"
#if JVET_AK0118_BF_FOR_INTRA_PRED
class IntraPrediction;
#endif

struct CompArea;

template <typename T> struct AreaBuf;
typedef AreaBuf<      Pel>  PelBuf;
typedef AreaBuf<const Pel> CPelBuf;
using WidthHeight = std::pair<int16_t, int16_t>;

class IntraPredictionNN
{
private:
  struct InternalData;
  InternalData* m_predModel;
  bool m_enable = true;
  std::array<std::array<uint32_t, NUM_INDICES_REP>, MAX_NUM_COMPONENT> m_argmaxs;
  WidthHeight m_aboveLeftAvailableContext;
  WidthHeight m_aboveLeftAvailableNbLines;

public:
  IntraPredictionNN();
  ~IntraPredictionNN();
  IntraPredictionNN(const IntraPredictionNN&) = delete;
  IntraPredictionNN(IntraPredictionNN&&) = delete;
  IntraPredictionNN& operator=(const IntraPredictionNN&) = delete;
  IntraPredictionNN& operator=(IntraPredictionNN&&) = delete;
  void init(const int bitdepth);
  void setEnable(const bool b) { m_enable = b; }
  void resetCuData();
  void setAvailableContext(const int aboveAvailableCtx, const int leftAvailableCtx, const int aboveAvailableNbLines, const int leftAvailableNbLines)
  {
    m_aboveLeftAvailableContext = WidthHeight(aboveAvailableCtx, leftAvailableCtx);
    m_aboveLeftAvailableNbLines = WidthHeight(aboveAvailableNbLines, leftAvailableNbLines);
  }
  void predictPnn(PelBuf& pDst, const CPelBuf& srcBuf, const int inputBitDepth, const CompArea& area, bool& isPredictionRun);
#if JVET_AK0118_BF_FOR_INTRA_PRED
  Pel* getNnIntraPredPtr ();
#endif
  static WidthHeight getNbLines(const int blockWidth, const int blockHeight);
  static WidthHeight getPnnCtxLength(const int blockWidth, const int blockHeight);
  static bool hasPnnPrediction(const CodingUnit& cu);
  static bool isUpsamplingNeeded(const CompArea& area);
  bool isContextCollectionNeeded(const CompArea& area);
  void setEquivalentIntraDir(const ComponentID compId, const int idxDir0, const int idxDir1) { m_argmaxs[compId][0] = idxDir0; m_argmaxs[compId][1] = idxDir1; }
  const std::array<uint32_t, NUM_INDICES_REP>& getEquivalentIntraDir(const ComponentID compId) { return m_argmaxs[compId]; }
  void collectContextWidthHeightMasksInPixels(const CompArea &area, const int16_t* const piRoiOrigin, const int iPicStride, const int inputBitDepth, const CodingUnit& cu, const ComponentID compID);

  /// the decoder's GPU reconstruction: how predictPnn predicts luma block area of CU cu - its model, the context's
  /// transformation (downsampling, transposition), the context available (collectContextWidthHeightMasksInPixels;
  /// the lengths and lines as extractContextPortions takes them, downsampled and transposed) and the upsampling's
  /// boundaries
  struct GpuBlock
  {
    int  model;                         // the model's index (modelData)
    int  heightTr, widthTr;             // the model's target size (the downsampled block, transposed)
    int  factorVer, factorHor;          // the downsampling of the context, the upsampling of the prediction
    bool transposed;
    int  maxNbLines[2], maxCtxLength[2];   // the input's layout: lines above / left, the above / left arm's length
    int  ctxNbLines[2], ctxLength[2];      // the context available
    int  availAbove, availLeft, nbLinesAbove, nbLinesLeft;   // the same in samples of the block (setAvailableContext's)
    bool leftBoundary, topBoundary;     // the upsampling starts from the reference column left / row above
  };
  static GpuBlock gpuBlock(const CompArea& area, const CodingUnit& cu);
  /// the SADL model data of model `model` of numModels()
  static const unsigned int* modelData(const int model, size_t& bytes);
  static int numModels();

private:
  void xPredIntraPnnContext(const CompArea& area, const CodingUnit& cu, const int unitHeight, const int unitWidth);
  static void contextAvailability(const CompArea& area, const CodingUnit& cu, const int unitHeight, const int unitWidth,
                                  int& availAbove, int& availLeft, int& nbLinesAbove, int& nbLinesLeft);
  static int getNbLinesIndividualDimension(const unsigned int dimension);
  static bool decideShapeHandledPnn(const uint32_t height, const uint32_t width);
};
#endif


