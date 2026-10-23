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
    \brief    intra reconstruction of a picture on the GPU

    The decoder records every intra transform block of a picture; at the end of the picture the GPU reconstructs the
    picture from the records alone (reference samples, prediction, dequantization, inverse transform, reconstruction).
    With the reconstruction on, the CPU only derives the records; when verifying, it reconstructs the picture too
    (its prediction and residual are recorded) and the results are compared block by block. Supported so far: planar / DC / angular intra prediction with reference filtering, PDPC, intra fusion, the
    intra bilateral filter, MRL, ISP and MIP, PDP (also MIP's, and in DIMD / OBIC / intra merge blends), NN intra
    prediction; DCT-II and explicit MTS, LFNST (with the extended kernels) and NSPT,
    transform skip and BDPCM; scalar and dependent quantization, QCS; joint CbCr; LMCS; sign prediction; the dual tree;
    4:2:0 or 4:0:0. A picture with anything else is skipped and counted. The GPU derives the intra modes itself (the
    MPM lists, chroma DM and candidates, in a node per CU before its blocks) where they depend on reconstructed
    samples, i.e. with DIMD or TIMD: DIMD CUs (their blended prediction), DIMD's modes in every MPM list, DIMD chroma,
    OBIC CUs (the modes of their neighbours, blended), intra merge CUs (a neighbour's parameters, ranked by template
    costs), TIMD CUs (modes from template costs, blended); and the transforms that depend on a derived mode, also those
    of DIMD, OBIC, intra merge and TIMD CUs (with MTSS, from a DIMD mode list of their own) and LFNST / NSPT of MIP
    blocks (from a DIMD of their prediction). ECM_GPU_MODES=1 makes it derive every MPM mode on any stream.
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
class Reshape;
class TrQuant;

class GpuIntraRec
{
public:
  /// nullptr without a usable Vulkan device
  static GpuIntraRec* create();
  ~GpuIntraRec();
  /// destroys the kernels and tables every instance shares, after the last instance (before gpu::Context::release)
  static void releaseShared();

  /// on: the GPU reconstructs the supported pictures (the CPU only derives the block parameters); off: the CPU
  /// reconstructs every picture and the GPU's reconstruction of the supported ones is compared (verification)
  void setOn( bool on );
  bool on() const;
  /// the decoder's reshaper, whose LMCS state (LUTs, chroma scales, slice flags) is that of the picture being decoded
  void setReshape( Reshape* reshape );
  /// the decoder's TrQuant, which selects the transform types of each block (TrQuant::getInvTransformTypes)
  void setTrQuant( TrQuant* trQuant );
  /// the decoder's intra prediction: verification compares its template costs of each intra merge CU's ranking
  void setIntraPrediction( const IntraPrediction* intraPred );
  /// whether the GPU path supports the slice's tools (sequence- and slice-level settings; per block record() checks);
  /// partial: partial verification's (the tool families it checks per CU left out)
  static bool supported( const Slice& slice, std::string& why, bool partial = false );

  void beginPicture( const CodingStructure& cs );
  /// sign prediction (TrQuant::predCoeffSigns): takes the block's predicted levels and their coded signs for the next
  /// record() of it, before the CPU decides the signs (the TU stays as it is); call right before record() or, when
  /// the CPU reconstructs too, right before predCoeffSigns
  void prepareSignPred( TransformUnit& tu, const ComponentID compID );
  /// one transform block of one component (with the CPU's prediction and residual when verifying, else nullptr)
  void record( const TransformUnit& tu, const ComponentID compID, const IntraPrediction::IpaSnapshot& ipa,
               const QpParam& qp, const CPelBuf* pred, const CPelBuf* resi );
  /// the picture has blocks the GPU path cannot reconstruct (on: the CPU reconstructs it instead)
  bool failed() const;
  /// the picture uses something the GPU path does not support yet (the first reason is reported)
  void unsupported( const char* why );
  /// CU cu is one the GPU path does not reconstruct (an inter, IBC or palette CU): with partial verification its
  /// samples are taken from the CPU, else as unsupported()
  void cpuCu( const CodingUnit& cu, const char* why );
  /// partial verification: the CCP models of non-intra CU cu, which the CPU has reconstructed (JVET-AF0073: each TU's
  /// a candidate of the merge CUs next to it), with history those CU::saveModelsInHCCP puts into the CTU row's history
  /// (called before CU::saveProCcpInfo, whose models of the TUs without one are candidates only: again without)
  /// (an intra CU: where it was taken from the CPU at its luma in a single tree, its chroma's record, now decoded)
  void cpuCuModels( const CodingUnit& cu, bool history = true );
  /// non-intra CU cu, before the CPU decodes it: with partial verification an inter CU the GPU reconstructs from the
  /// CPU's prediction (recordInterCu records its blocks), else one taken from the CPU (cpuCu: the reason)
  void interCu( const CodingUnit& cu );
  /// with an inter CU interCu took: block compID's predicted signs, before the CPU decides them (prepareSignPred)
  void prepareInterSignPred( TransformUnit& tu, const ComponentID compID );
  /// inter CU cu, which the CPU has reconstructed: the records of its blocks (interCu's CUs)
  void recordInterCu( const CodingUnit& cu );
  /// IBC GPM CU cu, which the CPU has predicted: its parts' final vectors (recordInterCu's)
  void ibcGpmVectors( const CodingUnit& cu, const Mv& mv0, const Mv& mv1 );
  /// inter CU cu of multi-pass DMVR (bdmvrRefine), which the CPU has predicted: DMVR's refined vectors per 16 x 16
  /// sub-PU (InterPrediction's m_bdmvrSubPuMvBuf, stride DMVR_SUBPU_STRIDE), recordInterCu's
  void dmvrVectors( const CodingUnit& cu, const Mv* list0, const Mv* list1 );
  /// GPM CU cu, which the CPU has predicted (motionCompensationGeo): the blend it took (1 the plain one, 2 the rounded
  /// one after each part's own OBMC, 0 another), the split direction, the blending index, each part's motion
  /// (InterPrediction::m_gpuGeoPart) and its motion and flags at its OBMC (m_gpuGeoMi, m_gpuGeoObmc), recordInterCu's
  void gpmParts( const CodingUnit& cu, int kind, int splitDir, int bldIdx, const int32_t parts[2][22],
                 const MotionInfo partMi[2], const int32_t partObmc[2] );
  /// TU tu of an inter CCCM or inter CCP merge CU, before DecCu blends its chroma prediction
  /// (deriveInterCccmPrediction, deriveInterCcpMergePrediction): the inter one
  void interChromaPred( const TransformUnit& tu );
  /// chroma reordering (JVET-AH0136): the index of PU pu's chroma mode in its reordered list as DecCu parsed it
  /// (chromaIdx), which the GPU's derivation of the list takes
  void chromaReorderIndex( const PredictionUnit& pu, int idx );
  /// reconstructs the recorded picture on the GPU; on: into the picture's reconstruction buffer, off: compared with
  /// the CPU's reconstruction
  void endPicture( const CodingStructure& cs );

  struct Impl;

private:
  GpuIntraRec() = default;
  std::unique_ptr<Impl> m_impl;
};

#endif
