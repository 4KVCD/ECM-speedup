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

/** \file     SegmentParallel.h
    \brief    Bit-exact parallel encoding of intra-period segments
*/

#ifndef __SEGMENTPARALLEL__
#define __SEGMENTPARALLEL__

#include <string>

//! \ingroup EncoderApp
//! \{

// The reference encoder resets all cross-picture state at each CRA with pending RAS init (CABAC
// storage, ALF APSs, SAO/CCSAO history, ...), so that chunks [s, s+L] encoded independently can be
// joined into the bitstream of one full-sequence encode (see EncGOP: "bit-equal").  This runs the
// chunks as concurrent child processes of the same executable and joins them in-process: the
// leading IDR access unit of every chunk after the first is dropped (it duplicates the CRA that
// ends the previous chunk) and picture order count LSBs are rebased.
// All-intra sequences (every picture a CRA, parameter sets repeated) are split into disjoint chunks
// instead: each worker starts at the chunk's own POC (SegmentPOCOffset) with the state the full encode
// has at that CRA, so the chunk bitstreams are simply concatenated.
struct SegmentParallelCfg
{
  int         numWorkers        = 1;    // concurrent child encoders
  int         segmentLength     = 0;    // frames per chunk (multiple of the intra period); 0: automatic
  int         intraPeriod       = -1;
  int         framesToBeEncoded = 0;   // coded frames (after temporal subsampling)
  int         frameSkip         = 0;   // input frames
  int         temporalSubsampleRatio = 1;
  double      frameRate         = 0;   // input frame rate
  bool        pocOffsetChunks   = false;   // disjoint chunks started at their own POC, joined by concatenation
  std::string bitstreamFile;
  std::string reconFile;
};

// returns true if the configuration can be split into independently coded chunks; reason gets why not
bool segmentParallelSupported( const SegmentParallelCfg& cfg, int decodingRefreshType, int maxLayers, std::string& reason );

// runs the whole encode; returns the process exit code
int  runSegmentParallel( int argc, char* argv[], const SegmentParallelCfg& cfg );

//! \}

#endif // __SEGMENTPARALLEL__
