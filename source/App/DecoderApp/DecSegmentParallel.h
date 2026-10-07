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

/** \file     DecSegmentParallel.h
    \brief    Bit-exact parallel decoding of random-access segments
*/

#ifndef __DECSEGMENTPARALLEL__
#define __DECSEGMENTPARALLEL__

#include <string>

//! \ingroup DecoderApp
//! \{

// The stream is split at CRA pictures into segments that are decoded by concurrent child processes of the same
// executable; their outputs are concatenated. A child gets the stream up to the end of its segment and starts with
// the standard random-access process (SkipFrames): it parses the slice headers of the pictures before its start
// (so parameter sets, APSs and the POC derivation are those of a sequential decode) and decodes from a CRA. When the
// segment's first CRA has RASL pictures, the child starts at the previous CRA instead, so that the RASL pictures find
// their reference pictures; the pictures decoded before the segment's first CRA are references only and are not
// written (SegmentOverlapPictures). A child keeps the POC MSB at its random-access point and treats a missing
// reference picture as an error. If a stream cannot be split, or a child fails, the caller decodes sequentially.
struct DecSegmentParallelCfg
{
  int         numWorkers = 1;   // concurrent child decoders
  std::string bitstreamFile;
  std::string reconFile;
  bool        outputBitDepthGiven = false;   // -d: else each child writes the bit depth of its first picture
};

// returns the exit code of the segment-parallel decode, or -1 when it was not done (reason says why; nothing was
// written to the output), in which case the caller decodes sequentially
int runDecSegmentParallel( int argc, char* argv[], const DecSegmentParallelCfg& cfg, std::string& reason );

//! \}

#endif // __DECSEGMENTPARALLEL__
