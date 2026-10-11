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

/** \file     VkRecon.h
    \brief    Reconstruction functions of the GPU decoder
*/

#pragma once

#include "VkCore.h"

#include "CommonLib/GpuRecon.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace vkd
{

/// Takes over reconstruction functions of CommonLib through GpuReconHooks: each call becomes a GPU dispatch whose
/// result replaces the CPU's. The port grows function by function; in check mode the CPU runs every function too
/// and the results must be identical.
class VkRecon : public GpuReconHooks
{
public:
  void init( Core* core, bool check );
  void destroy();

  bool invTransformNxN( TrQuant& trQuant, TransformUnit& tu, ComponentID compID, PelBuf& resi, const QpParam& qp ) override;

  /// How many calls each GPU function served, by path, for the coverage report.
  struct Counts
  {
    uint64_t invTransformModes[5] = {};   ///< by GpuInvTransformJob::Mode
    uint64_t invTransformDepQuant = 0;
    uint64_t invTransformScaling  = 0;
    uint64_t invTransformBdpcm    = 0;
    uint64_t invTransformQcs      = 0;
    uint64_t invTransformLfnst[2] = {};   ///< VVC LFNST, extended LFNST
    uint64_t invTransformCpu      = 0;    ///< calls left to the CPU
  };
  const Counts& counts() const { return m_counts; }

private:
  /// An append-only table of the CPU's constant matrices on the GPU, by address of the matrix.
  struct Table
  {
    Buffer                                     buf;
    size_t                                     used = 0;
    std::unordered_map<const void*, uint32_t>  offsets;   ///< in elements
  };
  uint32_t tableOffset( Table& t, const void* data, size_t bytes, size_t elementSize );
  void     ensureBuffer( Buffer& buf, size_t size, Mem mem );

  Core*              m_core  = nullptr;
  bool               m_check = false;
  Table              m_mat8;      ///< LFNST and NSPT matrices
  Table              m_mat16;     ///< inverse transform kernels
  Buffer             m_job;
  Buffer             m_data;
  Buffer             m_coef;
  Buffer             m_tmp;
  Buffer             m_resi;
  GpuInvTransformJob m_trJob;
  std::vector<int>   m_dataCpu;
  Counts             m_counts;
};

}   // namespace vkd
