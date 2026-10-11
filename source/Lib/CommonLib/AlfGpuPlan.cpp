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

/** \file     AlfGpuPlan.cpp
    \brief    What ALFProcess computes for one picture, recorded for the GPU decoder
*/

#include "AlfGpuPlan.h"

#include <algorithm>
#include <cstring>

void AlfGpuPlan::begin( int picWidth, int picHeight, ChromaFormat format, int numCtus )
{
  gpu       = true;
  cpuReason = nullptr;
  stale     = false;
  width     = picWidth;
  height    = picHeight;
  hasChroma = format != CHROMA_400;
  chromaScaleCorr[0] = chromaScaleCorr[1] = 0;
  lumaScaleMode      = false;
  ctus.assign( numCtus, Ctu() );
  lumaFilters.clear();
  chromaFilters.clear();
  ccAlfFilters.clear();
  writes.clear();
  mirrors.clear();

  if( picWidth != m_maskW || picHeight != m_maskH || hasChroma != m_maskChroma )
  {
    m_maskW      = picWidth;
    m_maskH      = picHeight;
    m_maskChroma = hasChroma;
    for( int p = 0; p < NUM_PLANES; p++ )
    {
      Mask&     m      = m_mask[p];
      const int chroma = planeIsChroma( p ) ? 1 : 0;
      m.pad            = planePadding( p );
      m.cellsW         = ( ( picWidth >> chroma ) + 2 * m.pad ) / 2;
      m.cellsH         = ( ( picHeight >> chroma ) + 2 * m.pad ) / 2;
      m.epoch.assign( chroma && !hasChroma ? 0 : size_t( m.cellsW ) * m.cellsH, 0 );
    }
    m_epoch = 0;
  }
  if( ++m_epoch == 0 )
  {
    // the epochs wrapped around: forget all earlier pictures
    for( Mask& m: m_mask )
    {
      std::fill( m.epoch.begin(), m.epoch.end(), 0 );
    }
    m_epoch = 1;
  }
}

void AlfGpuPlan::markCells( int plane, int x, int y, int w, int h )
{
  Mask& m = m_mask[plane];
  CHECK( ( x | y | w | h ) & 1, "ALF GPU plan: region not on even coordinates" );
  const int cx0 = ( x + m.pad ) >> 1, cy0 = ( y + m.pad ) >> 1;
  const int cx1 = ( x + w + m.pad ) >> 1, cy1 = ( y + h + m.pad ) >> 1;
  CHECK( cx0 < 0 || cy0 < 0 || cx1 > m.cellsW || cy1 > m.cellsH, "ALF GPU plan: region outside the plane" );
  for( int cy = cy0; cy < cy1; cy++ )
  {
    memset( &m.epoch[size_t( cy ) * m.cellsW + cx0], m_epoch, cx1 - cx0 );
  }
}

void AlfGpuPlan::write( int plane, int x, int y, int w, int h )
{
  if( w <= 0 || h <= 0 )
  {
    return;
  }
  writes.push_back( { int16_t( plane ), 0, x, y, w, h } );
  markCells( plane, x, y, w, h );
}

void AlfGpuPlan::mirror( int plane, int axis, int x, int y, int w, int h )
{
  if( w <= 0 || h <= 0 )
  {
    return;
  }
  // The mirror reads the samples that it copies: the same rows (axis 0) or columns (axis 1) on the other side of
  // the picture edge.
  const int chroma = planeIsChroma( plane ) ? 1 : 0;
  const int pw = width >> chroma, ph = height >> chroma;
  if( axis == 0 )
  {
    const int sx0 = x < 0 ? -x - w : 2 * pw - x - w;
    readRect( plane, sx0, y, w, h );
  }
  else
  {
    const int sy0 = y < 0 ? -y - h : 2 * ph - y - h;
    readRect( plane, x, sy0, w, h );
  }
  mirrors.push_back( { int16_t( plane ), int16_t( axis ), x, y, w, h } );
  markCells( plane, x, y, w, h );
}

void AlfGpuPlan::checkRow( int plane, int y, int x0, int x1 )
{
  if( stale || x1 <= x0 )
  {
    return;
  }
  Mask& m = m_mask[plane];
  const int cy = ( y + m.pad ) >> 1, cx0 = ( x0 + m.pad ) >> 1, cx1 = ( x1 - 1 + m.pad ) >> 1;
  if( y + m.pad < 0 || x0 + m.pad < 0 || cy >= m.cellsH || cx1 >= m.cellsW )
  {
    stale = true;   // beyond the padding: whatever the CPU reads there is not a result of this picture
    return;
  }
  const uint8_t* e = &m.epoch[size_t( cy ) * m.cellsW];
  for( int cx = cx0; cx <= cx1; cx++ )
  {
    if( e[cx] != m_epoch )
    {
      stale = true;
      return;
    }
  }
}

void AlfGpuPlan::readRect( int plane, int x, int y, int w, int h )
{
  for( int r = y; r < y + h; r++ )
  {
    checkRow( plane, r, x, x + w );
  }
}

void AlfGpuPlan::readCross( int plane, int x, int y, int w, int h, int axisDist, int crossDist )
{
  readRect( plane, x - axisDist, y, w + 2 * axisDist, h );
  readRect( plane, x, y - axisDist, w, axisDist );
  readRect( plane, x, y + h, w, axisDist );
  readRect( plane, x - crossDist, y - crossDist, w + 2 * crossDist, h + 2 * crossDist );
}

void AlfGpuPlan::readDiamond( int plane, int x, int y, int w, int h, int dist )
{
  readRect( plane, x - dist, y, w + 2 * dist, h );
  for( int d = 1; d <= dist; d++ )
  {
    const int ext = dist - d;
    checkRow( plane, y - d, x - ext, x + w + ext );
    checkRow( plane, y + h - 1 + d, x - ext, x + w + ext );
  }
}
