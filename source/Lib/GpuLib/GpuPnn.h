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

/** \file     GpuPnn.h
    \brief    the models of the NN intra prediction (JVET-AJ0249, IntraPredictionNN) as intra_rec reads them

    Each model is a chain of SADL MatMul layers (int16 weights, int32 sums shifted and saturated), all but the last
    followed by a leaky ReLU. Its tables (intra_nn.glsl's PNN_*): a header (the number of layers, the input's size, the
    output's quantizer), per layer its outputs, shift, the leaky ReLU's slope and shift (-1: none) and where its columns
    start; per column (output) its weights in packs of 8 consecutive inputs - the input index of the first, then the 8
    int16 weights two per word - as SADL's sparse layers store them (a dense layer: every pack of the column). The sums
    are those of SADL's every kernel: int32 arithmetic wraps, so neither the order nor the zero weights matter.
*/

#ifndef __GPUPNN__
#define __GPUPNN__

#include <cstdint>
#include <vector>

/// appends the tables of IntraPredictionNN's models to words (offsets absolute in it, so its start must be where the
/// buffer starts), the offset of each model's header to modelOffset; false where a model has what the kernels lack
bool appendPnnModels( std::vector<int32_t>& words, std::vector<int32_t>& modelOffset );

#endif
