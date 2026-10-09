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

/** \file     GpuIntraRec.h
    \brief    intra reconstruction of a picture on the GPU (first stage of the GPU reconstruction: verification only)

    The decoder records every intra transform block while it reconstructs a picture on the CPU (the CPU's prediction
    and residual too); at the end of the picture the GPU reconstructs the picture from the records alone (reference
    samples, prediction, dequantization, inverse transform, reconstruction) and the results are compared block by
    block. Supported so far (M1): planar / DC / angular intra prediction without reference filtering, PDPC, intra
    fusion, the intra bilateral filter, MRL, ISP, MIP and the decoder-side mode derivations; DCT-II without LFNST, MTS
    or transform skip; scalar dequantization; 4:2:0 or 4:0:0. A picture with anything else is skipped and counted.
*/

#ifndef __GPUINTRAREC__
#define __GPUINTRAREC__

#include "CommonLib/CommonDef.h"
#include "CommonLib/IntraPrediction.h"
#include "CommonLib/Unit.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class CodingStructure;
class QpParam;

class GpuIntraRec
{
public:
  /// nullptr without a usable Vulkan device
  static GpuIntraRec* create();
  ~GpuIntraRec();

  /// on: the GPU reconstructs the supported pictures (the CPU only derives the block parameters); off: the CPU
  /// reconstructs every picture and the GPU's reconstruction of the supported ones is compared (verification)
  void setOn( bool on );
  bool on() const;
  /// whether the GPU path supports the slice's tools (sequence- and slice-level settings; per block record() checks)
  static bool supported( const Slice& slice, std::string& why );

  void beginPicture( const CodingStructure& cs );
  /// one transform block of one component (with the CPU's prediction and residual when verifying, else nullptr)
  void record( const TransformUnit& tu, const ComponentID compID, const IntraPrediction::IpaSnapshot& ipa,
               const QpParam& qp, const CPelBuf* pred, const CPelBuf* resi );
  /// the picture has blocks the GPU path cannot reconstruct (on: the CPU reconstructs it instead)
  bool failed() const;
  /// the picture uses something the GPU path does not support yet (the first reason is reported)
  void unsupported( const char* why );
  /// reconstructs the recorded picture on the GPU; on: into the picture's reconstruction buffer, off: compared with
  /// the CPU's reconstruction
  void endPicture( const CodingStructure& cs );

  struct Impl;

private:
  GpuIntraRec() = default;
  std::unique_ptr<Impl> m_impl;
};

#endif
