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

/** \file     GpuRecon.h
    \brief    Hooks through which the GPU decoder takes over reconstruction functions of CommonLib
*/

#pragma once

#include "CommonDef.h"
#include "Buffer.h"

#include <cstdint>
#include <vector>

class TrQuant;
class QpParam;
struct TransformUnit;

/// What TrQuant::invTransformNxN computes for one transform block, recorded by TrQuant::recordInvTransform with
/// every decision taken on the CPU, so that the GPU does the arithmetic: dequantization, LFNST, then the separable
/// transform, NSPT or transform skip. The matrices are pointers into the CPU tables, which the GPU decoder holds too.
struct GpuInvTransformJob
{
  int width  = 0;
  int height = 0;

  // ---- dequantization (Quant::dequant or the dependent quantizer's dequantBlock) ----
  bool                 depQuant    = false;
  int                  bdpcm       = 0;       ///< Quant::dequant: 1 horizontal, 2 vertical accumulation of the levels
  bool                 scalingList = false;   ///< dequantCoef holds a factor per coefficient
  bool                 qcs         = false;   ///< the latent shift of SPS QCS
  int                  shift       = 0;       ///< depQuant: shift, may be negative; else rightShift
  int                  scale       = 0;       ///< depQuant: invQScale; else the scale without scaling list
  int                  add         = 0;       ///< depQuant only
  int                  inputMin    = 0;       ///< Quant::dequant: clipping of the levels
  int                  inputMax    = 0;
  int                  coefMin     = 0;       ///< clipping of the dequantized coefficients and of transform outputs
  int                  coefMax     = 0;
  uint64_t             stateTrans  = 0;       ///< depQuant: the packed state transition table
  const int*           dequantCoef = nullptr;
  const int*           qcsTable    = nullptr; ///< 64 entries

  // ---- LFNST ----
  bool                 lfnst          = false;
  int                  lfnstSbSize    = 0;
  int                  lfnstNumIn     = 0;    ///< coefficients gathered along the scan
  int                  lfnstZeroOut   = 0;    ///< inputs of the matrix product
  int                  lfnstTrSize    = 0;    ///< outputs
  bool                 lfnstTranspose = false;
  bool                 lfnstExt       = false;
  const int8_t*        lfnstMat       = nullptr;
  std::vector<int>     lfnstScan;             ///< raster positions of the inputs

  // ---- transform ----
  enum Mode
  {
    MODE_SKIP,        ///< xITransformSkip
    MODE_2D,          ///< two passes of fastInvTrans
    MODE_1D_VER,      ///< width 1
    MODE_1D_HOR,      ///< height 1
    MODE_NSPT
  };
  Mode                 mode       = MODE_SKIP;
  const int16_t*       matHor     = nullptr;  ///< inverse kernel of width x width
  const int16_t*       matVer     = nullptr;  ///< height x height
  int                  skipWidth  = 0;
  int                  skipHeight = 0;
  int                  shift1     = 0;        ///< MODE_2D: first pass; 1D: the only pass
  int                  shift2     = 0;

  // ---- NSPT ----
  int                  nsptW         = 0;     ///< the dimensions passed to computeInvNspt (transposed with nsptTranspose)
  int                  nsptH         = 0;
  bool                 nsptTranspose = false;
  int                  nsptNumIn     = 0;
  int                  nsptShift     = 0;
  const int8_t*        nsptMat       = nullptr;
  std::vector<int>     nsptScan;

  const TCoeff*        coeffs = nullptr;      ///< the parsed levels, width x height
  std::vector<int>     dqScan;                ///< depQuant: raster positions in scan order
};

/// Reconstruction functions that the GPU decoder takes over. CommonLib calls a hook first; when it returns true the
/// GPU did the work. Only the decoder installs hooks; the encoder never does.
class GpuReconHooks
{
public:
  virtual ~GpuReconHooks() {}
  virtual bool invTransformNxN( TrQuant& trQuant, TransformUnit& tu, ComponentID compID, PelBuf& resi, const QpParam& qp ) = 0;
};

extern GpuReconHooks* g_gpuReconHooks;
