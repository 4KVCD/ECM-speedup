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

#include "DepQuant.h"
#include "TrQuant.h"
#include "CodingStructure.h"
#include "UnitTools.h"

#include <bitset>
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace DQIntern
{
  /*================================================================================*/
  /*=====                                                                      =====*/
  /*=====   R A T E   E S T I M A T O R                                        =====*/
  /*=====                                                                      =====*/
  /*================================================================================*/

  struct NbInfoSbb
  {
#if JVET_AE0102_LFNST_CTX
    uint8_t inPos[2][5];   // 0-tpl for normal ctx, 1-for lfnst
    uint8_t num[2];
#else
    uint8_t  num;
    uint8_t  inPos[5];
#endif
  };
  struct NbInfoOut
  {
#if JVET_AE0102_LFNST_CTX
    uint16_t outPos[2][5];   // 0-tpl for normal ctx, 1-for lfnst
    uint16_t num[2];
    uint16_t maxDist[2];
#else
    uint16_t maxDist;
    uint16_t num;
    uint16_t outPos[5];
#endif
  };
  struct CoeffFracBits
  {
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
    int32_t   bits[GTN + 3];
#else
    int32_t   bits[6];
#endif
  };

  enum ScanPosType
  {
    SCAN_ISCSBB = 0,
    SCAN_SOCSBB = 1,
    SCAN_EOCSBB = 2
  };

  struct ScanInfo
  {
    ScanInfo() {}
    int         sbbSize;
    int         numSbb;
    int         scanIdx;
    int         rasterPos;
    int         sbbPos;
    int         insidePos;
    bool        eosbb;
    ScanPosType spt;
    unsigned    sigCtxOffsetNext;
    unsigned    gtxCtxOffsetNext;
    int         nextInsidePos;
    NbInfoSbb   nextNbInfoSbb;
#if JVET_AE0102_LFNST_CTX
    uint8_t     nextNbShuffle[2][16];   // nextNbInfoSbb.inPos[ctxType][0..num-1] as byte-shuffle indices, 0x80 = unused
    uint8_t     nextNbRows[2][5];       // nextNbInfoSbb.inPos[ctxType][0..num-1], 16 (an all-zero level row) = unused
#else
    uint8_t     nextNbShuffle[1][16];
    uint8_t     nextNbRows[1][5];
#endif
    int         nextSbbRight;
    int         nextSbbBelow;
    int         posX;
    int         posY;
    ChannelType chType;
    int         sbtInfo;
    int         tuWidth;
    int         tuHeight;
  };

  class Rom;
  struct TUParameters
  {
    TUParameters(const Rom &rom, const unsigned width, const unsigned height, const ChannelType chType);
    ~TUParameters() { delete[] m_scanInfo; }

    ChannelType        m_chType;
    unsigned           m_width;
    unsigned           m_height;
    unsigned           m_numCoeff;
    unsigned           m_numSbb;
    unsigned           m_log2SbbWidth;
    unsigned           m_log2SbbHeight;
    unsigned           m_log2SbbSize;
    unsigned           m_sbbSize;
    unsigned           m_sbbMask;
    unsigned           m_widthInSbb;
    unsigned           m_heightInSbb;
    CoeffScanType      m_scanType;
    const ScanElement *m_scanSbbId2SbbPos;
    const ScanElement *m_scanId2BlkPos;
    const NbInfoSbb *  m_scanId2NbInfoSbb;
    const NbInfoOut *  m_scanId2NbInfoOut;
    ScanInfo *         m_scanInfo;

  private:
    void xSetScanInfo(ScanInfo &scanInfo, int scanIdx);
  };

  class Rom
  {
  public:
    Rom() : m_scansInitialized(false) {}
    ~Rom() { xUninitScanArrays(); }
    void                init() { xInitScanArrays(); }
    const NbInfoSbb *   getNbInfoSbb(int hd, int vd) const { return m_scanId2NbInfoSbbArray[hd][vd]; }
    const NbInfoOut *   getNbInfoOut(int hd, int vd) const { return m_scanId2NbInfoOutArray[hd][vd]; }
    const TUParameters *getTUPars(const CompArea &area, const ComponentID compID) const
    {
      return m_tuParameters[floorLog2(area.width)][floorLog2(area.height)][toChannelType(compID)];
    }

  private:
    void xInitScanArrays();
    void xUninitScanArrays();

  private:
    bool          m_scansInitialized;
    NbInfoSbb *   m_scanId2NbInfoSbbArray[MAX_CU_DEPTH + 1][MAX_CU_DEPTH + 1];
    NbInfoOut *   m_scanId2NbInfoOutArray[MAX_CU_DEPTH + 1][MAX_CU_DEPTH + 1];
    TUParameters *m_tuParameters[MAX_CU_DEPTH + 1][MAX_CU_DEPTH + 1][MAX_NUM_CHANNEL_TYPE];
  };

  void Rom::xInitScanArrays()
  {
    if (m_scansInitialized)
    {
      return;
    }
    ::memset(m_scanId2NbInfoSbbArray, 0, sizeof(m_scanId2NbInfoSbbArray));
    ::memset(m_scanId2NbInfoOutArray, 0, sizeof(m_scanId2NbInfoOutArray));
    ::memset(m_tuParameters, 0, sizeof(m_tuParameters));

    uint32_t raster2id[MAX_CU_SIZE * MAX_CU_SIZE];
    ::memset(raster2id, 0, sizeof(raster2id));

    for (int hd = 0; hd <= MAX_CU_DEPTH; hd++)
    {
      for (int vd = 0; vd <= MAX_CU_DEPTH; vd++)
      {
        if ((hd == 0 && vd <= 1) || (hd <= 1 && vd == 0))
        {
          continue;
        }
        const uint32_t      blockWidth   = (1 << hd);
        const uint32_t      blockHeight  = (1 << vd);
        const uint32_t      log2CGWidth  = g_log2SbbSize[hd][vd][0];
        const uint32_t      log2CGHeight = g_log2SbbSize[hd][vd][1];
        const uint32_t      groupWidth   = 1 << log2CGWidth;
        const uint32_t      groupHeight  = 1 << log2CGHeight;
        const uint32_t      groupSize    = groupWidth * groupHeight;
        const CoeffScanType scanType     = SCAN_DIAG;
        const SizeType      blkWidthIdx  = gp_sizeIdxInfo->idxFrom(blockWidth);
        const SizeType      blkHeightIdx = gp_sizeIdxInfo->idxFrom(blockHeight);
        const ScanElement * scanId2RP    = g_scanOrder[SCAN_GROUPED_4x4][scanType][blkWidthIdx][blkHeightIdx];
        NbInfoSbb *&        sId2NbSbb    = m_scanId2NbInfoSbbArray[hd][vd];
        NbInfoOut *&        sId2NbOut    = m_scanId2NbInfoOutArray[hd][vd];
        // consider only non-zero-out region
        const uint32_t blkWidthNZOut  = std::min<unsigned>(JVET_C0024_ZERO_OUT_TH, blockWidth);
        const uint32_t blkHeightNZOut = std::min<unsigned>(JVET_C0024_ZERO_OUT_TH, blockHeight);
        const uint32_t totalValues    = blkWidthNZOut * blkHeightNZOut;

        sId2NbSbb = new NbInfoSbb[totalValues];
        sId2NbOut = new NbInfoOut[totalValues];

        for (uint32_t scanId = 0; scanId < totalValues; scanId++)
        {
          raster2id[scanId2RP[scanId].idx] = scanId;
        }

        for (unsigned scanId = 0; scanId < totalValues; scanId++)
        {
          const int posX = scanId2RP[scanId].x;
          const int posY = scanId2RP[scanId].y;
          const int rpos = scanId2RP[scanId].idx;
          {
            //===== inside subband neighbours =====
            NbInfoSbb &nbSbb  = sId2NbSbb[scanId];
            const int  begSbb = scanId - (scanId & (groupSize - 1));   // first pos in current subblock
            int        cpos[5];

            cpos[0] =
              (posX + 1 < blkWidthNZOut ? (raster2id[rpos + 1] < groupSize + begSbb ? raster2id[rpos + 1] - begSbb : 0)
                                        : 0);
            cpos[1] =
              (posX + 2 < blkWidthNZOut ? (raster2id[rpos + 2] < groupSize + begSbb ? raster2id[rpos + 2] - begSbb : 0)
                                        : 0);
            cpos[2] =
              (posX + 1 < blkWidthNZOut && posY + 1 < blkHeightNZOut ? (
                 raster2id[rpos + 1 + blockWidth] < groupSize + begSbb ? raster2id[rpos + 1 + blockWidth] - begSbb : 0)
                                                                     : 0);
            cpos[3] =
              (posY + 1 < blkHeightNZOut
                 ? (raster2id[rpos + blockWidth] < groupSize + begSbb ? raster2id[rpos + blockWidth] - begSbb : 0)
                 : 0);
            cpos[4] =
              (posY + 2 < blkHeightNZOut ? (
                 raster2id[rpos + 2 * blockWidth] < groupSize + begSbb ? raster2id[rpos + 2 * blockWidth] - begSbb : 0)
                                         : 0);

#if JVET_AE0102_LFNST_CTX
            int lpos[5];
            for (int i = 0; i < 5; i++)
            {
              lpos[i] = ((scanId + i + 1 < totalValues) && scanId2RP[scanId + i + 1].x < blkWidthNZOut
                         && scanId2RP[scanId + i + 1].y < blkHeightNZOut)
                          ? (scanId + i + 1 < groupSize + begSbb ? scanId + i + 1 - begSbb : 0)
                          : 0;
            }

            for (nbSbb.num[0] = 0; true;)
            {
              int nk = -1;

              for (int k = 0; k < 5; k++)
              {
                if (cpos[k] != 0 && (nk < 0 || cpos[k] < cpos[nk]))
                {
                  nk = k;
                }
              }
              if (nk < 0)
              {
                break;
              }
              nbSbb.inPos[0][nbSbb.num[0]++] = uint8_t(cpos[nk]);
              cpos[nk]                       = 0;
            }
            for (int k = nbSbb.num[0]; k < 5; k++)
            {
              nbSbb.inPos[0][k] = 0;
            }
            nbSbb.num[1] = 0;
            for (int k = 0; k < 5; k++)
            {
              if (lpos[k] > 0)
              {
                nbSbb.num[1]++;
              }
              nbSbb.inPos[1][k] = lpos[k];
            }
#undef CHECK_POS
#else
            for (nbSbb.num = 0; true;)
            {
              int nk = -1;
              for (int k = 0; k < 5; k++)
              {
                if (cpos[k] != 0 && (nk < 0 || cpos[k] < cpos[nk]))
                {
                  nk = k;
                }
              }
              if (nk < 0)
              {
                break;
              }
              nbSbb.inPos[nbSbb.num++] = uint8_t(cpos[nk]);
              cpos[nk]                 = 0;
            }
            for (int k = nbSbb.num; k < 5; k++)
            {
              nbSbb.inPos[k] = 0;
            }
#endif
          }
          {
            //===== outside subband neighbours =====
            NbInfoOut &nbOut  = sId2NbOut[scanId];
            const int  begSbb = scanId - (scanId & (groupSize - 1));   // first pos in current subblock
            int        cpos[5];

            cpos[0] =
              (posX + 1 < blkWidthNZOut ? (raster2id[rpos + 1] >= groupSize + begSbb ? raster2id[rpos + 1] : 0) : 0);
            cpos[1] =
              (posX + 2 < blkWidthNZOut ? (raster2id[rpos + 2] >= groupSize + begSbb ? raster2id[rpos + 2] : 0) : 0);
            cpos[2] =
              (posX + 1 < blkWidthNZOut && posY + 1 < blkHeightNZOut
                 ? (raster2id[rpos + 1 + blockWidth] >= groupSize + begSbb ? raster2id[rpos + 1 + blockWidth] : 0)
                 : 0);
            cpos[3] = (posY + 1 < blkHeightNZOut
                         ? (raster2id[rpos + blockWidth] >= groupSize + begSbb ? raster2id[rpos + blockWidth] : 0)
                         : 0);
            cpos[4] =
              (posY + 2 < blkHeightNZOut
                 ? (raster2id[rpos + 2 * blockWidth] >= groupSize + begSbb ? raster2id[rpos + 2 * blockWidth] : 0)
                 : 0);

#if JVET_AE0102_LFNST_CTX
            int lpos[5];
            for (int i = 0; i < 5; i++)
            {
              lpos[i] = ((scanId + i + 1) < totalValues && scanId2RP[scanId + i + 1].x < blkWidthNZOut
                         && scanId2RP[scanId + i + 1].y < blkHeightNZOut)
                          ? ((scanId + i + 1) >= groupSize + begSbb ? scanId + i + 1 : 0)
                          : 0;
            }

            for (nbOut.num[0] = 0; true;)
            {
              int nk = -1;
              for (int k = 0; k < 5; k++)
              {
                if (cpos[k] != 0 && (nk < 0 || cpos[k] < cpos[nk]))
                {
                  nk = k;
                }
              }
              if (nk < 0)
              {
                break;
              }
              nbOut.outPos[0][nbOut.num[0]++] = uint16_t(cpos[nk]);
              cpos[nk]                        = 0;
            }
            for (int k = nbOut.num[0]; k < 5; k++)
            {
              nbOut.outPos[0][k] = 0;
            }

            // needs checking
            for (nbOut.num[1] = 0; true;)
            {
              int nk = -1;
              for (int k = 0; k < 5; k++)
              {
                if (lpos[k] != 0 && (nk < 0 || lpos[k] < lpos[nk]))
                {
                  nk = k;
                }
              }
              if (nk < 0)
              {
                break;
              }
              nbOut.outPos[1][nbOut.num[1]++] = uint16_t(lpos[nk]);
              lpos[nk]                        = 0;
            }
            for (int k = nbOut.num[1]; k < 5; k++)
            {
              nbOut.outPos[1][k] = 0;
            }

            nbOut.maxDist[0] = (scanId == 0 ? 0 : sId2NbOut[scanId - 1].maxDist[0]);
            for (int k = 0; k < nbOut.num[0]; k++)
            {
              if (nbOut.outPos[0][k] > nbOut.maxDist[0])
              {
                nbOut.maxDist[0] = nbOut.outPos[0][k];
              }
            }

            // PN TODO add maxdist for lfnst
            nbOut.maxDist[1] = (scanId == 0 ? 0 : sId2NbOut[scanId - 1].maxDist[1]);
            for (int k = 0; k < nbOut.num[1]; k++)
            {
              if (nbOut.outPos[1][k] > nbOut.maxDist[1])
              {
                nbOut.maxDist[1] = nbOut.outPos[1][k];
              }
            }
#else
            for (nbOut.num = 0; true;)
            {
              int nk = -1;
              for (int k = 0; k < 5; k++)
              {
                if (cpos[k] != 0 && (nk < 0 || cpos[k] < cpos[nk]))
                {
                  nk = k;
                }
              }
              if (nk < 0)
              {
                break;
              }
              nbOut.outPos[nbOut.num++] = uint16_t(cpos[nk]);
              cpos[nk]                  = 0;
            }
            for (int k = nbOut.num; k < 5; k++)
            {
              nbOut.outPos[k] = 0;
            }
            nbOut.maxDist = (scanId == 0 ? 0 : sId2NbOut[scanId - 1].maxDist);
            for (int k = 0; k < nbOut.num; k++)
            {
              if (nbOut.outPos[k] > nbOut.maxDist)
              {
                nbOut.maxDist = nbOut.outPos[k];
              }
            }
#endif
          }
        }

#if JVET_AE0102_LFNST_CTX
        // make it relative
        for (unsigned scanId = 0; scanId < totalValues; scanId++)
        {
          NbInfoOut &nbOut  = sId2NbOut[scanId];
          const int  begSbb = scanId - (scanId & (groupSize - 1));   // first pos in current subblock
          for (int k = 0; k < nbOut.num[0]; k++)
          {
            CHECK(begSbb > nbOut.outPos[0][k], "Position must be past sub block begin");
            nbOut.outPos[0][k] -= begSbb;
          }
          nbOut.maxDist[0] -= scanId;
        }

        for (unsigned scanId = 0; scanId < totalValues; scanId++)
        {
          NbInfoOut &nbOut  = sId2NbOut[scanId];
          const int  begSbb = scanId - (scanId & (groupSize - 1));   // first pos in current subblock
          for (int k = 0; k < nbOut.num[1]; k++)
          {
            // CHECK(begSbb > nbOut.outPos[1][k], "Position must be past sub block begin");
            nbOut.outPos[1][k] -= begSbb;
          }
          nbOut.maxDist[1] -= scanId;
        }
#else
        // make it relative
        for (unsigned scanId = 0; scanId < totalValues; scanId++)
        {
          NbInfoOut &nbOut  = sId2NbOut[scanId];
          const int  begSbb = scanId - (scanId & (groupSize - 1));   // first pos in current subblock
          for (int k = 0; k < nbOut.num; k++)
          {
            CHECK(begSbb > nbOut.outPos[k], "Position must be past sub block begin");
            nbOut.outPos[k] -= begSbb;
          }
          nbOut.maxDist -= scanId;
        }
#endif

        for (int chId = 0; chId < MAX_NUM_CHANNEL_TYPE; chId++)
        {
          m_tuParameters[hd][vd][chId] = new TUParameters(*this, blockWidth, blockHeight, ChannelType(chId));
        }
      }
    }
    m_scansInitialized = true;
  }

  void Rom::xUninitScanArrays()
  {
    if (!m_scansInitialized)
    {
      return;
    }
    for (int hd = 0; hd <= MAX_CU_DEPTH; hd++)
    {
      for (int vd = 0; vd <= MAX_CU_DEPTH; vd++)
      {
        NbInfoSbb *&sId2NbSbb = m_scanId2NbInfoSbbArray[hd][vd];
        NbInfoOut *&sId2NbOut = m_scanId2NbInfoOutArray[hd][vd];
        if (sId2NbSbb)
        {
          delete[] sId2NbSbb;
        }
        if (sId2NbOut)
        {
          delete[] sId2NbOut;
        }
        for (int chId = 0; chId < MAX_NUM_CHANNEL_TYPE; chId++)
        {
          TUParameters *&tuPars = m_tuParameters[hd][vd][chId];
          if (tuPars)
          {
            delete tuPars;
          }
        }
      }
    }
    m_scansInitialized = false;
  }

  static Rom g_Rom;

  TUParameters::TUParameters(const Rom &rom, const unsigned width, const unsigned height, const ChannelType chType)
  {
    m_chType                     = chType;
    m_width                      = width;
    m_height                     = height;
    const uint32_t nonzeroWidth  = std::min<uint32_t>(JVET_C0024_ZERO_OUT_TH, m_width);
    const uint32_t nonzeroHeight = std::min<uint32_t>(JVET_C0024_ZERO_OUT_TH, m_height);
    m_numCoeff                   = nonzeroWidth * nonzeroHeight;
    const int log2W              = floorLog2(m_width);
    const int log2H              = floorLog2(m_height);
    m_log2SbbWidth               = g_log2SbbSize[log2W][log2H][0];
    m_log2SbbHeight              = g_log2SbbSize[log2W][log2H][1];
    m_log2SbbSize                = m_log2SbbWidth + m_log2SbbHeight;
    m_sbbSize                    = (1 << m_log2SbbSize);
    m_sbbMask                    = m_sbbSize - 1;
    m_widthInSbb                 = nonzeroWidth >> m_log2SbbWidth;
    m_heightInSbb                = nonzeroHeight >> m_log2SbbHeight;
    m_numSbb                     = m_widthInSbb * m_heightInSbb;
    m_scanType                   = SCAN_DIAG;
    SizeType hsbb                = gp_sizeIdxInfo->idxFrom(m_widthInSbb);
    SizeType vsbb                = gp_sizeIdxInfo->idxFrom(m_heightInSbb);
    SizeType hsId                = gp_sizeIdxInfo->idxFrom(m_width);
    SizeType vsId                = gp_sizeIdxInfo->idxFrom(m_height);
    m_scanSbbId2SbbPos           = g_scanOrder[SCAN_UNGROUPED][m_scanType][hsbb][vsbb];
    m_scanId2BlkPos              = g_scanOrder[SCAN_GROUPED_4x4][m_scanType][hsId][vsId];
    m_scanId2NbInfoSbb           = rom.getNbInfoSbb(log2W, log2H);
    m_scanId2NbInfoOut           = rom.getNbInfoOut(log2W, log2H);
    m_scanInfo                   = new ScanInfo[m_numCoeff];
    for (int scanIdx = 0; scanIdx < m_numCoeff; scanIdx++)
    {
      xSetScanInfo(m_scanInfo[scanIdx], scanIdx);
    }
  }

  void TUParameters::xSetScanInfo(ScanInfo &scanInfo, int scanIdx)
  {
    scanInfo.chType    = m_chType;
    scanInfo.tuWidth   = m_width;
    scanInfo.tuHeight  = m_height;
    scanInfo.sbbSize   = m_sbbSize;
    scanInfo.numSbb    = m_numSbb;
    scanInfo.scanIdx   = scanIdx;
    scanInfo.rasterPos = m_scanId2BlkPos[scanIdx].idx;
    scanInfo.sbbPos    = m_scanSbbId2SbbPos[scanIdx >> m_log2SbbSize].idx;
    scanInfo.insidePos = scanIdx & m_sbbMask;
    scanInfo.eosbb     = (scanInfo.insidePos == 0);
    scanInfo.spt       = SCAN_ISCSBB;
    if (scanInfo.insidePos == m_sbbMask && scanIdx > scanInfo.sbbSize && scanIdx < m_numCoeff - 1)
      scanInfo.spt = SCAN_SOCSBB;
    else if (scanInfo.eosbb && scanIdx > 0 && scanIdx < m_numCoeff - m_sbbSize)
      scanInfo.spt = SCAN_EOCSBB;
    scanInfo.posX = m_scanId2BlkPos[scanIdx].x;
    scanInfo.posY = m_scanId2BlkPos[scanIdx].y;
    ::memset(scanInfo.nextNbShuffle, 0x80, sizeof(scanInfo.nextNbShuffle));
    ::memset(scanInfo.nextNbRows, 16, sizeof(scanInfo.nextNbRows));
    if (scanIdx)
    {
      const int nextScanIdx = scanIdx - 1;
      const int diag        = m_scanId2BlkPos[nextScanIdx].x + m_scanId2BlkPos[nextScanIdx].y;
      if (m_chType == CHANNEL_TYPE_LUMA)
      {
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        scanInfo.sigCtxOffsetNext = (diag < 2 ? NSIGCTX*2 : diag < 5 ? NSIGCTX : 0);
        scanInfo.gtxCtxOffsetNext = (diag < 1 ? 1 + (NGTXCTX * 3) : diag < 3 ? 1 + (NGTXCTX * 2) : diag < 10 ? 1 + NGTXCTX : 1);
#else
        scanInfo.sigCtxOffsetNext = ( diag < 2 ? 8 : diag < 5 ?  4 : 0 );
        scanInfo.gtxCtxOffsetNext = (diag < 1 ? 16 : diag < 3 ? 11 : diag < 10 ? 6 : 1);
#endif
      }
      else
      {
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        scanInfo.sigCtxOffsetNext = (diag < 2 ? NSIGCTX : 0);
        scanInfo.gtxCtxOffsetNext = (diag < 1 ? 1 + NGTXCTX : 1);
#else
        scanInfo.sigCtxOffsetNext = ( diag < 2 ? 4 : 0 );
        scanInfo.gtxCtxOffsetNext = (diag < 1 ? 6 : 1);
#endif
      }
      scanInfo.nextInsidePos = nextScanIdx & m_sbbMask;
      scanInfo.nextNbInfoSbb = m_scanId2NbInfoSbb[nextScanIdx];
#if JVET_AE0102_LFNST_CTX
      for (int ctxType = 0; ctxType < 2; ctxType++)
      {
        for (int k = 0; k < std::min<int>(5, scanInfo.nextNbInfoSbb.num[ctxType]); k++)
        {
          scanInfo.nextNbShuffle[ctxType][k] = scanInfo.nextNbInfoSbb.inPos[ctxType][k];
          scanInfo.nextNbRows[ctxType][k]    = scanInfo.nextNbInfoSbb.inPos[ctxType][k];
        }
      }
#else
      for (int k = 0; k < std::min<int>(5, scanInfo.nextNbInfoSbb.num); k++)
      {
        scanInfo.nextNbShuffle[0][k] = scanInfo.nextNbInfoSbb.inPos[k];
        scanInfo.nextNbRows[0][k]    = scanInfo.nextNbInfoSbb.inPos[k];
      }
#endif
      if (scanInfo.eosbb)
      {
        const int nextSbbPos  = m_scanSbbId2SbbPos[nextScanIdx >> m_log2SbbSize].idx;
        const int nextSbbPosY = nextSbbPos / m_widthInSbb;
        const int nextSbbPosX = nextSbbPos - nextSbbPosY * m_widthInSbb;
        scanInfo.nextSbbRight = (nextSbbPosX < m_widthInSbb - 1 ? nextSbbPos + 1 : 0);
        scanInfo.nextSbbBelow = (nextSbbPosY < m_heightInSbb - 1 ? nextSbbPos + m_widthInSbb : 0);
      }
    }
  }

  // non-virtual access to the rate tables of the context models (the per-TU setup reads ~200 of them)
  struct StdFracBitsAccess
  {
    const CtxStore<BinProbModel_Std> &ctxStore;
    BinFracBits getFracBitsArray(unsigned ctxId) const { return ctxStore[ctxId].getFracBitsArray(); }
  };

  class RateEstimator
  {
  public:
    RateEstimator() {}
    ~RateEstimator() {}
    template<class FracBitsAccessT>
    void initCtx(const TUParameters &tuPars, const TransformUnit &tu, const ComponentID compID,
                 const FracBitsAccessT &fracBitsAccess
#if JVET_AE0102_LFNST_CTX
                 ,
                 const uint32_t lfnstIdx = 0
#endif
    );

    inline const BinFracBits *sigSbbFracBits() const { return m_sigSbbFracBits; }
    inline const BinFracBits *sigFlagBits(unsigned stateId) const
    {
#if TCQ_8STATES
      return m_sigFracBits[(stateId & 1) ? 1 + ((stateId & 3) >> 1) : 0];
#else
      return m_sigFracBits[std::max(((int) stateId) - 1, 0)];
#endif
    }
    inline const CoeffFracBits *gtxFracBits(unsigned stateId) const { return m_gtxFracBits; }
    inline int32_t              lastOffset(unsigned scanIdx) const
    {
      return m_lastBitsX[m_scanId2Pos[scanIdx].x] + m_lastBitsY[m_scanId2Pos[scanIdx].y];
    }

  private:
    template<class FracBitsAccessT>
    void xSetLastCoeffOffset(const FracBitsAccessT &fracBitsAccess, const TUParameters &tuPars, const TransformUnit &tu,
                             const ComponentID compID);
    template<class FracBitsAccessT>
    void xSetSigSbbFracBits(const FracBitsAccessT &fracBitsAccess, ChannelType chType);
#if JVET_AE0102_LFNST_CTX
    template<class FracBitsAccessT>
    void xSetSigFlagBits(const FracBitsAccessT &fracBitsAccess, ChannelType chType, const bool isLfnst);
    template<class FracBitsAccessT>
    void xSetGtxFlagBits(const FracBitsAccessT &fracBitsAccess, ChannelType chType, const bool isLfnst);
#else
    template<class FracBitsAccessT>
    void xSetSigFlagBits(const FracBitsAccessT &fracBitsAccess, ChannelType chType);
    template<class FracBitsAccessT>
    void xSetGtxFlagBits(const FracBitsAccessT &fracBitsAccess, ChannelType chType);
#endif

  private:
    static const unsigned sm_numCtxSetsSig    = 3;
    static const unsigned sm_numCtxSetsGtx    = 2;
    static const unsigned sm_maxNumSigSbbCtx  = 2;
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
    static const unsigned sm_maxNumSigCtx = NSIGCTX * 3;
    static const unsigned sm_maxNumGtxCtx = 1 + NGTXCTX * 4;
#else
    static const unsigned sm_maxNumSigCtx     = 12;
    static const unsigned sm_maxNumGtxCtx = 21;
#endif
#if JVET_AG0143_INTER_INTRA
    bool      m_condition = false;
    SliceType m_sliceType = NUMBER_OF_SLICE_TYPES;
#endif
  private:
    const ScanElement *m_scanId2Pos;
    int32_t            m_lastBitsX[MAX_TB_SIZEY];
    int32_t            m_lastBitsY[MAX_TB_SIZEY];
    BinFracBits        m_sigSbbFracBits[sm_maxNumSigSbbCtx];
    BinFracBits        m_sigFracBits[sm_numCtxSetsSig][sm_maxNumSigCtx];
    CoeffFracBits      m_gtxFracBits[sm_maxNumGtxCtx];
  };

  template<class FracBitsAccessT>
  void RateEstimator::initCtx(const TUParameters &tuPars, const TransformUnit &tu, const ComponentID compID,
                              const FracBitsAccessT &fracBitsAccess
#if JVET_AE0102_LFNST_CTX
                              ,
                              const uint32_t lfnstIdx
#endif
  )
  {
#if JVET_AG0143_INTER_INTRA
    assert(tuPars.m_chType == toChannelType(compID));
    m_condition = CoeffCodingContext::getSwitchCondition(*tu.cu,tuPars.m_chType);
    m_sliceType = tu.cu->slice->getSliceType();
#endif
    m_scanId2Pos = tuPars.m_scanId2BlkPos;
    xSetSigSbbFracBits(fracBitsAccess, tuPars.m_chType);
#if JVET_AE0102_LFNST_CTX
    xSetSigFlagBits(fracBitsAccess, tuPars.m_chType, lfnstIdx > 0);
    xSetGtxFlagBits(fracBitsAccess, tuPars.m_chType, lfnstIdx > 0);
#else
    xSetSigFlagBits(fracBitsAccess, tuPars.m_chType);
    xSetGtxFlagBits(fracBitsAccess, tuPars.m_chType);
#endif
    xSetLastCoeffOffset(fracBitsAccess, tuPars, tu, compID);
  }

  template<class FracBitsAccessT>
  void RateEstimator::xSetLastCoeffOffset(const FracBitsAccessT &fracBitsAccess, const TUParameters &tuPars,
                                          const TransformUnit &tu, const ComponentID compID)
  {
    const ChannelType chType       = (compID == COMPONENT_Y ? CHANNEL_TYPE_LUMA : CHANNEL_TYPE_CHROMA);
    int32_t           cbfDeltaBits = 0;
    if (compID == COMPONENT_Y && !CU::isIntra(*tu.cu) && !tu.depth)
    {
      const BinFracBits bits = fracBitsAccess.getFracBitsArray(Ctx::QtRootCbf());
      cbfDeltaBits           = int32_t(bits.intBits[1]) - int32_t(bits.intBits[0]);
    }
    else
    {
      BinFracBits bits;
      bool        prevLumaCbf           = false;
      bool        lastCbfIsInferred     = false;
      bool        useIntraSubPartitions = tu.cu->ispMode && isLuma(chType);
      if (useIntraSubPartitions)
      {
        bool     rootCbfSoFar       = false;
        bool     isLastSubPartition = CU::isISPLast(*tu.cu, tu.Y(), compID);
        uint32_t nTus = tu.cu->ispMode == HOR_INTRA_SUBPARTITIONS ? tu.cu->lheight() >> floorLog2(tu.lheight())
                                                                  : tu.cu->lwidth() >> floorLog2(tu.lwidth());
        if (isLastSubPartition)
        {
          TransformUnit *tuPointer = tu.cu->firstTU;
          for (int tuIdx = 0; tuIdx < nTus - 1; tuIdx++)
          {
            rootCbfSoFar |= TU::getCbfAtDepth(*tuPointer, COMPONENT_Y, tu.depth);
            tuPointer = tuPointer->next;
          }
          if (!rootCbfSoFar)
          {
            lastCbfIsInferred = true;
          }
        }
        if (!lastCbfIsInferred)
        {
          prevLumaCbf = TU::getPrevTuCbfAtDepth(tu, compID, tu.depth);
        }
        bits = fracBitsAccess.getFracBitsArray(Ctx::QtCbf[compID](DeriveCtx::CtxQtCbf(compID, prevLumaCbf, true)));
      }
      else
      {
        bits = fracBitsAccess.getFracBitsArray(Ctx::QtCbf[compID](DeriveCtx::CtxQtCbf(compID, tu.cbf[COMPONENT_Cb])));
      }
      cbfDeltaBits = lastCbfIsInferred ? 0 : int32_t(bits.intBits[1]) - int32_t(bits.intBits[0]);
    }

    static const unsigned prefixCtx[] = { 0, 0, 0, 3, 6, 10, 15, 21 };
    uint32_t              ctxBits[LAST_SIGNIFICANT_GROUPS];
    for (unsigned xy = 0; xy < 2; xy++)
    {
      int32_t        bitOffset = (xy ? cbfDeltaBits : 0);
      int32_t *      lastBits  = (xy ? m_lastBitsY : m_lastBitsX);
      const unsigned size      = (xy ? tuPars.m_height : tuPars.m_width);
      const unsigned log2Size  = ceilLog2(size);
      const bool     useYCtx   = (xy != 0);
#if JVET_AG0143_INTER_INTRA
      const CtxSet &ctxSetLast =
        m_condition ? ((useYCtx ? Ctx::LastYCtxSetSwitch : Ctx::LastXCtxSetSwitch)[chType])
                    : ((useYCtx ? Ctx::LastY : Ctx::LastX)[chType]);
#else
      const CtxSet &ctxSetLast = (useYCtx ? Ctx::LastY : Ctx::LastX)[chType];
#endif

      const unsigned lastShift  = (compID == COMPONENT_Y ? (log2Size + 1) >> 2 : Clip3<unsigned>(0, 2, size >> 3));
      const unsigned lastOffset = (compID == COMPONENT_Y ? (prefixCtx[log2Size]) : 0);
      uint32_t       sumFBits   = 0;
      unsigned       maxCtxId   = g_uiGroupIdx[std::min<unsigned>(JVET_C0024_ZERO_OUT_TH, size) - 1];
      for (unsigned ctxId = 0; ctxId < maxCtxId; ctxId++)
      {
        const BinFracBits bits = fracBitsAccess.getFracBitsArray(ctxSetLast(lastOffset + (ctxId >> lastShift)));
        ctxBits[ctxId] = sumFBits + bits.intBits[0] + (ctxId > 3 ? ((ctxId - 2) >> 1) << SCALE_BITS : 0) + bitOffset;
        sumFBits += bits.intBits[1];
      }
      ctxBits[maxCtxId] = sumFBits + (maxCtxId > 3 ? ((maxCtxId - 2) >> 1) << SCALE_BITS : 0) + bitOffset;
      for (unsigned pos = 0; pos < std::min<unsigned>(JVET_C0024_ZERO_OUT_TH, size); pos++)
      {
        lastBits[pos] = ctxBits[g_uiGroupIdx[pos]];
      }
    }
  }

  template<class FracBitsAccessT>
  void RateEstimator::xSetSigSbbFracBits(const FracBitsAccessT &fracBitsAccess, ChannelType chType)
  {
#if JVET_AG0143_INTER_INTRA
    const CtxSet &ctxSet =
      m_condition ?  Ctx::SigCoeffGroupCtxSetSwitch[chType]
                  : Ctx::SigCoeffGroup[chType];
#else
    const CtxSet &ctxSet = Ctx::SigCoeffGroup[chType];
#endif
    for (unsigned ctxId = 0; ctxId < sm_maxNumSigSbbCtx; ctxId++)
    {
      m_sigSbbFracBits[ctxId] = fracBitsAccess.getFracBitsArray(ctxSet(ctxId));
    }
  }
  template<class FracBitsAccessT>
  void RateEstimator::xSetSigFlagBits(const FracBitsAccessT &fracBitsAccess, ChannelType chType
#if JVET_AE0102_LFNST_CTX
                                      ,
                                      const bool isLfnst
#endif
  )
  {
    for (unsigned ctxSetId = 0; ctxSetId < sm_numCtxSetsSig; ctxSetId++)
    {
      BinFracBits *bits = m_sigFracBits[ctxSetId];

#if JVET_AG0143_INTER_INTRA 
#if JVET_AE0102_LFNST_CTX
      const CtxSet &ctxSet =
        m_condition
          ? (isLfnst ? 
                 Ctx::SigFlagL[chType + 2 * ctxSetId] 
                 : 
                 Ctx::SigFlagCtxSetSwitch[chType + 2 * ctxSetId]
                 )
          : (isLfnst ? Ctx::SigFlagL[chType + 2 * ctxSetId] : Ctx::SigFlag[chType + 2 * ctxSetId]);
#else
      const CtxSet &ctxSet = m_condition ? Ctx::SigFlagCtxSetSwitch[chType + 2 * ctxSetId]
                                         : Ctx::SigFlag[chType + 2 * ctxSetId];
#endif

#else
#if JVET_AE0102_LFNST_CTX
      const CtxSet &ctxSet =
        isLfnst ? Ctx::SigFlagL[chType + 2 * ctxSetId] : Ctx::SigFlag[chType + 2 * ctxSetId];
#else
      const CtxSet &ctxSet = Ctx::SigFlag[chType + 2 * ctxSetId];
#endif
#endif

#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
      const unsigned  numCtx = (chType == CHANNEL_TYPE_LUMA ? NSIGCTX * 3 : NSIGCTX * 2);
#else
      const unsigned  numCtx  = ( chType == CHANNEL_TYPE_LUMA ? 12 : 8 );
#endif
      for( unsigned ctxId = 0; ctxId < numCtx; ctxId++ )
      {
        bits[ctxId] = fracBitsAccess.getFracBitsArray(ctxSet(ctxId));
      }
    }
  }

  template<class FracBitsAccessT>
  void RateEstimator::xSetGtxFlagBits(const FracBitsAccessT &fracBitsAccess, ChannelType chType
#if JVET_AE0102_LFNST_CTX
                                      ,
                                      const bool isLfnst
#endif
  )
  {
#if JVET_AG0143_INTER_INTRA
#if JVET_AE0102_LFNST_CTX
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING 
    // ctxSetPar was missing in this case, added it back
    const CtxSet &ctxSetGt1 =
        m_condition
        ?  
        (isLfnst ?
            Ctx::GtxFlagL[2 + chType] 
            : 
            Ctx::GtxFlagCtxSetSwitch[2 + chType])
        : (isLfnst ? Ctx::GtxFlagL[2 + chType] : Ctx::GtxFlag[2 + chType]);
    const CtxSet &ctxSetGt2 =
        m_condition ? (isLfnst ? 
            Ctx::GtxFlagL[4 + chType] 
            : 
            Ctx::GtxFlagCtxSetSwitch[4 + chType])
        : (isLfnst ? Ctx::GtxFlagL[4 + chType] : Ctx::GtxFlag[4 + chType]);

    const CtxSet &ctxSetGt3 =
        m_condition
        ?  
        (isLfnst ?
            Ctx::GtxFlagL[chType] 
            : 
            Ctx::GtxFlagCtxSetSwitch[chType])
        : (isLfnst ? Ctx::GtxFlagL[chType] : Ctx::GtxFlag[chType]);
    const CtxSet &ctxSetGt4 =
        m_condition ? (isLfnst ? 
            Ctx::GtxFlagL[6 + chType] 
            : 
            Ctx::GtxFlagCtxSetSwitch[6 + chType])
        : (isLfnst ? Ctx::GtxFlagL[6 + chType] : Ctx::GtxFlag[6 + chType]);



#else
    const CtxSet &ctxSetPar =
      m_condition ?  (isLfnst ? 
        Ctx::ParFlagL[chType] 
        : 
        Ctx::ParFlagCtxSetSwitch[chType]
        )
      : (isLfnst ? Ctx::ParFlagL[chType] : Ctx::ParFlag[chType]);

    const CtxSet &ctxSetGt1 =
        m_condition
        ?  
        (isLfnst ?
            Ctx::GtxFlagL[2 + chType] 
            : 
            Ctx::GtxFlagCtxSetSwitch[2 + chType])
        : (isLfnst ? Ctx::GtxFlagL[2 + chType] : Ctx::GtxFlag[2 + chType]);
    const CtxSet &ctxSetGt2 =
        m_condition ? (isLfnst ? 
            Ctx::GtxFlagL[chType] 
            : 
            Ctx::GtxFlagCtxSetSwitch[chType])
        : (isLfnst ? Ctx::GtxFlagL[chType] : Ctx::GtxFlag[chType]);
#endif


#else
    const CtxSet &ctxSetGt1 =
      m_condition ? Ctx::GtxFlagCtxSetSwitch[2 + chType]
                  : Ctx::GtxFlag[2 + chType];
    const CtxSet &ctxSetGt2 = m_condition
                                ? Ctx::GtxFlagCtxSetSwitch[chType]
                                : Ctx::GtxFlag[chType];
    const CtxSet &ctxSetPar = m_condition
        ?  Ctx::ParFlagCtxSetSwitch[chType]
        : Ctx::ParFlag[chType];

#endif
#else 


#if JVET_AE0102_LFNST_CTX
const CtxSet &ctxSetPar = isLfnst ? Ctx::ParFlagL[chType] : Ctx::ParFlag[chType];

#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
const CtxSet &ctxSetGt1 = isLfnst ? Ctx::GtxFlagL[2 + chType] : Ctx::GtxFlag[2 + chType];
const CtxSet &ctxSetGt2 = isLfnst ? Ctx::GtxFlagL[4 + chType] : Ctx::GtxFlag[4 + chType];
const CtxSet &ctxSetGt3 = isLfnst ? Ctx::GtxFlagL[chType] : Ctx::GtxFlag[chType];
const CtxSet &ctxSetGt4 = isLfnst ? Ctx::GtxFlagL[6 + chType] : Ctx::GtxFlag[6 + chType];
#else
    const CtxSet &ctxSetGt1 = isLfnst ? Ctx::GtxFlagL[2 + chType] : Ctx::GtxFlag[2 + chType];
    const CtxSet &ctxSetGt2 = isLfnst ? Ctx::GtxFlagL[chType] : Ctx::GtxFlag[chType];
#endif
#else
    const CtxSet &ctxSetPar = Ctx::ParFlag[chType];
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
    const CtxSet &ctxSetGt1 = Ctx::GtxFlag[2 + chType];
    const CtxSet &ctxSetGt2 = Ctx::GtxFlag[4 + chType];
    const CtxSet &ctxSetGt3 = Ctx::GtxFlag[chType];
    const CtxSet &ctxSetGt4 = Ctx::GtxFlag[6 + chType];
#else
    const CtxSet &ctxSetGt1 = Ctx::GtxFlag[2 + chType];
    const CtxSet &ctxSetGt2 = Ctx::GtxFlag[chType];
#endif
#endif
#endif
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
    const unsigned  numCtx = (chType == CHANNEL_TYPE_LUMA ? 1+NGTXCTX * 4 : 1 + NGTXCTX * 2);
#else
    const unsigned  numCtx      = ( chType == CHANNEL_TYPE_LUMA ? 21 : 11 );
#endif
    for( unsigned ctxId = 0; ctxId < numCtx; ctxId++ )
    {
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
      BinFracBits     fbGt1 = fracBitsAccess.getFracBitsArray(ctxSetGt1(ctxId));
      BinFracBits     fbGt2 = fracBitsAccess.getFracBitsArray(ctxSetGt2(ctxId));
      BinFracBits     fbGt3 = fracBitsAccess.getFracBitsArray(ctxSetGt3(ctxId));
      BinFracBits     fbGtN = fracBitsAccess.getFracBitsArray(ctxSetGt4(ctxId));

      CoeffFracBits& cb = m_gtxFracBits[ctxId];
      cb.bits[0] = 0;
      cb.bits[1] = fbGt1.intBits[0] + (1 << SCALE_BITS);
      int32_t         gt1 = (1 << SCALE_BITS) + int32_t(fbGt1.intBits[1]);
      int32_t         gt2 = gt1 + fbGt2.intBits[1];
      int32_t         gt3 = gt2 + fbGt3.intBits[1];

      int32_t         c0 = fbGtN.intBits[0];
      int32_t         c1 = fbGtN.intBits[1];

      cb.bits[2] = gt1 + fbGt2.intBits[0];
      cb.bits[3] = gt2 + fbGt3.intBits[0];
      cb.bits[4] = gt3 + c0;
      cb.bits[GTN + 2] = gt3 + c1 * (GTN - 3) + (1 << SCALE_BITS);
      cb.bits[GTN + 1] = gt3 + c1 * (GTN - 3) + (1 << SCALE_BITS);

      int diff = GTN - 4;
      for (int i = 0; diff > 0; i++, diff--)
      {
        cb.bits[GTN - i] = gt3 + c1 * diff + c0;
      }


#else
      BinFracBits     fbPar = fracBitsAccess.getFracBitsArray(ctxSetPar(ctxId));
      BinFracBits     fbGt1 = fracBitsAccess.getFracBitsArray(ctxSetGt1(ctxId));
      BinFracBits     fbGt2 = fracBitsAccess.getFracBitsArray(ctxSetGt2(ctxId));
      CoeffFracBits& cb = m_gtxFracBits[ctxId];
      int32_t         par0 = (1 << SCALE_BITS) + int32_t(fbPar.intBits[0]);
      int32_t         par1 = (1 << SCALE_BITS) + int32_t(fbPar.intBits[1]);
      cb.bits[0] = 0;
      cb.bits[1] = fbGt1.intBits[0] + (1 << SCALE_BITS);
      cb.bits[2] = fbGt1.intBits[1] + par0 + fbGt2.intBits[0];
      cb.bits[3] = fbGt1.intBits[1] + par1 + fbGt2.intBits[0];
      cb.bits[4] = fbGt1.intBits[1] + par0 + fbGt2.intBits[1];
      cb.bits[5] = fbGt1.intBits[1] + par1 + fbGt2.intBits[1];
#endif
    }
  }

  /*================================================================================*/
  /*=====                                                                      =====*/
  /*=====   D A T A   S T R U C T U R E S                                      =====*/
  /*=====                                                                      =====*/
  /*================================================================================*/

  struct PQData
  {
    TCoeff  absLevel;
    int64_t deltaDist;
  };

  struct Decision
  {
    int64_t rdCost;
    TCoeff  absLevel;
    int     prevId;
  };

  /*================================================================================*/
  /*=====                                                                      =====*/
  /*=====   P R E - Q U A N T I Z E R                                          =====*/
  /*=====                                                                      =====*/
  /*================================================================================*/

  class Quantizer
  {
  public:
    Quantizer() {}
#if TCQ_8STATES
    void dequantBlock(const TransformUnit &tu, const ComponentID compID, const QpParam &cQP, CoeffBuf &recCoeff,
                      bool enableScalingLists, int *piDequantCoef, const uint64_t stateTransTab) const;
#else
    void        dequantBlock(const TransformUnit &tu, const ComponentID compID, const QpParam &cQP, CoeffBuf &recCoeff,
                             bool enableScalingLists, int *piDequantCoef) const;
#endif

    void initQuantBlock(const TransformUnit &tu, const ComponentID compID, const QpParam &cQP, const double lambda,
                        int gValue);
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
    inline void preQuantCoeff(const TCoeff absCoeff, PQData *pqData, TCoeff quanCoeff) const;
#else
    inline void preQuantCoeff(const TCoeff absCoeff, PQData *pqData, int quanCoeff) const;
#endif
    inline TCoeff getLastThreshold() const { return m_thresLast; }
    inline TCoeff getSSbbThreshold() const { return m_thresSSbb; }

    inline int64_t getQScale() const { return m_QScale; }

#if JVET_AE0125_SHIFT_QUANTIZATION_CENTER
    const int m_coeffShift[64] = { 0, 63, 31, 21, 15, 12, 10, 9, 7, 7, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3,
                                   2, 2,  2,  2,  2,  2,  2,  2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
#endif

  private:
    // quantization
    int     m_QShift;
    int64_t m_QAdd;
    int64_t m_QScale;
    TCoeff  m_maxQIdx;
    TCoeff  m_thresLast;
    TCoeff  m_thresSSbb;
    // distortion normalization
    int     m_DistShift;
    int64_t m_DistAdd;
    int64_t m_DistStepAdd;
    int64_t m_DistOrgFact;
  };

  inline int ceil_log2(uint64_t x)
  {
    static const uint64_t t[6] = { 0xFFFFFFFF00000000ull, 0x00000000FFFF0000ull, 0x000000000000FF00ull,
                                   0x00000000000000F0ull, 0x000000000000000Cull, 0x0000000000000002ull };
    int                   y    = (((x & (x - 1)) == 0) ? 0 : 1);
    int                   j    = 32;
    for (int i = 0; i < 6; i++)
    {
      int k = (((x & t[i]) == 0) ? 0 : j);
      y += k;
      x >>= k;
      j >>= 1;
    }
    return y;
  }
  void Quantizer::initQuantBlock(const TransformUnit &tu, const ComponentID compID, const QpParam &cQP,
                                 const double lambda, int gValue = -1)
  {
    CHECKD(lambda <= 0.0, "Lambda must be greater than 0");

    const int         qpDQ                  = cQP.Qp(tu.mtsIdx[compID] == MTS_SKIP) + 1;
    const int         qpPer                 = qpDQ / 6;
    const int         qpRem                 = qpDQ - 6 * qpPer;
    const SPS &       sps                   = *tu.cs->sps;
#if JVET_AL0181_ASBT
    const CompArea&   area                  = tu.blocksResidual[compID];
#else
    const CompArea &  area                  = tu.blocks[compID];
#endif
    const ChannelType chType                = toChannelType(compID);
    const int         channelBitDepth       = sps.getBitDepth(chType);
    const int         maxLog2TrDynamicRange = sps.getMaxLog2TrDynamicRange(chType);
    const int         nomTransformShift     = getTransformShift(channelBitDepth, area.size(), maxLog2TrDynamicRange);
    const bool        clipTransformShift =
      (tu.mtsIdx[compID] == MTS_SKIP && sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag());
    const bool needsSqrt2ScaleAdjustment = TU::needsSqrt2Scale(tu, compID);
    const int  transformShift = (clipTransformShift ? std::max<int>(0, nomTransformShift) : nomTransformShift)
                               + (needsSqrt2ScaleAdjustment ? -1 : 0);
    // quant parameters
    m_QShift = QUANT_SHIFT - 1 + qpPer + transformShift;
    m_QAdd   = -((3 << m_QShift) >> 1);
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
    int invShift = IQUANT_SHIFT + 1 - qpPer - transformShift;
#else
    Intermediate_Int invShift = IQUANT_SHIFT + 1 - qpPer - transformShift;
#endif
    m_QScale = g_quantScales[needsSqrt2ScaleAdjustment ? 1 : 0][qpRem];
    const unsigned qIdxBD =
      std::min<unsigned>(maxLog2TrDynamicRange + 1, 8 * sizeof(Intermediate_Int) + invShift - IQUANT_SHIFT - 1);
    m_maxQIdx   = (1 << (qIdxBD - 1)) - 4;
    m_thresLast = TCoeff((int64_t(4) << m_QShift));
    m_thresSSbb = TCoeff((int64_t(3) << m_QShift));
    // distortion calculation parameters
    const int64_t qScale    = (gValue == -1) ? m_QScale : gValue;
    const int     nomDShift = SCALE_BITS - 2 * (nomTransformShift + DISTORTION_PRECISION_ADJUSTMENT(channelBitDepth))
                          + m_QShift + (needsSqrt2ScaleAdjustment ? 1 : 0);
    const double  qScale2       = double(qScale * qScale);
    const double  nomDistFactor = (nomDShift < 0 ? 1.0 / (double(int64_t(1) << (-nomDShift)) * qScale2 * lambda)
                                                 : double(int64_t(1) << nomDShift) / (qScale2 * lambda));
    const int64_t pow2dfShift   = (int64_t) (nomDistFactor * qScale2) + 1;
    const int     dfShift       = ceil_log2(pow2dfShift);
    m_DistShift                 = 62 + m_QShift - 2 * maxLog2TrDynamicRange - dfShift;
    m_DistAdd                   = (int64_t(1) << m_DistShift) >> 1;
    m_DistStepAdd               = (int64_t) (nomDistFactor * double(int64_t(1) << (m_DistShift + m_QShift)) + .5);
    m_DistOrgFact               = (int64_t) (nomDistFactor * double(int64_t(1) << (m_DistShift + 1)) + .5);
  }

#if TCQ_8STATES
  void Quantizer::dequantBlock(const TransformUnit &tu, const ComponentID compID, const QpParam &cQP,
                               CoeffBuf &recCoeff, bool enableScalingLists, int *piDequantCoef,
                               const uint64_t stateTransTab) const
#else
  void Quantizer::dequantBlock(const TransformUnit &tu, const ComponentID compID, const QpParam &cQP,
                               CoeffBuf &recCoeff, bool enableScalingLists, int *piDequantCoef) const
#endif
  {
    //----- set basic parameters -----
#if JVET_AL0181_ASBT
    const CompArea&     area     = tu.blocksResidual[compID];
#else
    const CompArea &    area     = tu.blocks[compID];
#endif
    const int           numCoeff = area.area();
    const SizeType      hsId     = gp_sizeIdxInfo->idxFrom(area.width);
    const SizeType      vsId     = gp_sizeIdxInfo->idxFrom(area.height);
    const CoeffScanType scanType = SCAN_DIAG;
    const ScanElement * scan     = g_scanOrder[SCAN_GROUPED_4x4][scanType][hsId][vsId];
    const TCoeff *      qCoeff   = tu.getCoeffs(compID).buf;
    TCoeff *            tCoeff   = recCoeff.buf;

    //----- reset coefficients and get last scan index -----
    ::memset(tCoeff, 0, numCoeff * sizeof(TCoeff));
    int lastScanIdx = -1;
    for (int scanIdx = numCoeff - 1; scanIdx >= 0; scanIdx--)
    {
      if (qCoeff[scan[scanIdx].idx])
      {
        lastScanIdx = scanIdx;
        break;
      }
    }
    if (lastScanIdx < 0)
    {
      return;
    }

    //----- set dequant parameters -----
    const int         qpDQ                  = cQP.Qp(tu.mtsIdx[compID] == MTS_SKIP) + 1;
    const int         qpPer                 = qpDQ / 6;
    const int         qpRem                 = qpDQ - 6 * qpPer;
    const SPS &       sps                   = *tu.cs->sps;
    const ChannelType chType                = toChannelType(compID);
    const int         channelBitDepth       = sps.getBitDepth(chType);
    const int         maxLog2TrDynamicRange = sps.getMaxLog2TrDynamicRange(chType);
    const TCoeff      minTCoeff             = -(1 << maxLog2TrDynamicRange);
    const TCoeff      maxTCoeff             = (1 << maxLog2TrDynamicRange) - 1;
    const int         nomTransformShift     = getTransformShift(channelBitDepth, area.size(), maxLog2TrDynamicRange);
    const bool        clipTransformShift =
      (tu.mtsIdx[compID] == MTS_SKIP && sps.getSpsRangeExtension().getExtendedPrecisionProcessingFlag());
    const bool needsSqrt2ScaleAdjustment = TU::needsSqrt2Scale(tu, compID);
    const int  transformShift = (clipTransformShift ? std::max<int>(0, nomTransformShift) : nomTransformShift)
                               + (needsSqrt2ScaleAdjustment ? -1 : 0);
    Intermediate_Int shift =
      IQUANT_SHIFT + 1 - qpPer - transformShift + (enableScalingLists ? LOG2_SCALING_LIST_NEUTRAL_VALUE : 0);
    Intermediate_Int invQScale = g_invQuantScales[needsSqrt2ScaleAdjustment ? 1 : 0][qpRem];
    Intermediate_Int add       = (shift < 0) ? 0 : ((1 << shift) >> 1);
    //----- dequant coefficients -----
    for (int state = 0, scanIdx = lastScanIdx; scanIdx >= 0; scanIdx--)
    {
      const unsigned rasterPos = scan[scanIdx].idx;
      const TCoeff & level     = qCoeff[rasterPos];
      if (level)
      {
        if( enableScalingLists )
        {
          invQScale = piDequantCoef[ rasterPos ];   // scalingfactor*levelScale
        }
        if (shift < 0 && (enableScalingLists || scanIdx == lastScanIdx))
        {
          invQScale <<= -shift;
        }
#if TCQ_8STATES
        Intermediate_Int qIdx = (level << 1) + (level > 0 ? -(state & 1) : (state & 1));
#else
        Intermediate_Int qIdx = (level << 1) + (level > 0 ? -(state >> 1) : (state >> 1));
#endif
        int64_t nomTCoeff = ((int64_t) qIdx * (int64_t) invQScale + add) >> ((shift < 0) ? 0 : shift);

#if JVET_AE0125_SHIFT_QUANTIZATION_CENTER
#if JVET_AN0095_QUANTIZATION_CENTER_SHIFT
        if (sps.getUseQcs())
        {
#endif
          // Latent Shift
          int qIdx2 = qIdx;
          qIdx2 += (qIdx > 0 ? 1 : -1);
          int64_t nomTCoeff2 = ((int64_t)qIdx2 * (int64_t)invQScale + add) >> ((shift < 0) ? 0 : shift);
          int     aqIdx = abs(qIdx);
          int     coef = 0;
          if (aqIdx < 64)
          {
            coef = m_coeffShift[aqIdx];
          }
          nomTCoeff = (int64_t)((1024 - coef) * nomTCoeff + coef * nomTCoeff2) >> 10;
          // Latent Shift
#if JVET_AN0095_QUANTIZATION_CENTER_SHIFT
        }
#endif
#endif

        tCoeff[rasterPos] = (TCoeff) Clip3<int64_t>(minTCoeff, maxTCoeff, nomTCoeff);
      }
#if TCQ_8STATES
      state = int((stateTransTab >> ((state << 3) + ((level & 1) << 2))) & 15);
#else
      state = (32040 >> ((state << 2) + ((level & 1) << 1)))
              & 3;   // the 16-bit value "32040" represent the state transition table
#endif
    }
  }

#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
  inline void Quantizer::preQuantCoeff(const TCoeff absCoeff, PQData *pqData, TCoeff quanCoeff) const
#else
  inline void Quantizer::preQuantCoeff(const TCoeff absCoeff, PQData *pqData, int quanCoeff) const
#endif
  {
    int64_t scaledOrg = int64_t(absCoeff) * quanCoeff;
    TCoeff  qIdx      = std::max<TCoeff>(1, std::min<TCoeff>(m_maxQIdx, TCoeff((scaledOrg + m_QAdd) >> m_QShift)));
    int64_t scaledAdd = qIdx * m_DistStepAdd - scaledOrg * m_DistOrgFact;
    PQData &pq_a      = pqData[qIdx & 3];
    pq_a.deltaDist    = (scaledAdd * qIdx + m_DistAdd) >> m_DistShift;
    pq_a.absLevel     = (++qIdx) >> 1;
    scaledAdd += m_DistStepAdd;
    PQData &pq_b   = pqData[qIdx & 3];
    pq_b.deltaDist = (scaledAdd * qIdx + m_DistAdd) >> m_DistShift;
    pq_b.absLevel  = (++qIdx) >> 1;
    scaledAdd += m_DistStepAdd;
    PQData &pq_c   = pqData[qIdx & 3];
    pq_c.deltaDist = (scaledAdd * qIdx + m_DistAdd) >> m_DistShift;
    pq_c.absLevel  = (++qIdx) >> 1;
    scaledAdd += m_DistStepAdd;
    PQData &pq_d   = pqData[qIdx & 3];
    pq_d.deltaDist = (scaledAdd * qIdx + m_DistAdd) >> m_DistShift;
    pq_d.absLevel  = (++qIdx) >> 1;
  }

  /*================================================================================*/
  /*=====                                                                      =====*/
  /*=====   T C Q   S T A T E                                                  =====*/
  /*=====                                                                      =====*/
  /*================================================================================*/

  class State;

  struct SbbCtx
  {
    uint8_t *sbbFlags;
    uint8_t *levels;
  };

  // index of the all-zero template init (paths that started inside the current sub-block)
  static const int CTX_INIT_ZERO = 8;

  class CommonCtx
  {
  public:
#if TCQ_8STATES
    CommonCtx() : m_currSbbCtx(m_allSbbCtx), m_prevSbbCtx(m_currSbbCtx + 8) { ::memset(m_ctxInitT, 0, sizeof(m_ctxInitT)); }
#else
    CommonCtx() : m_currSbbCtx(m_allSbbCtx), m_prevSbbCtx(m_currSbbCtx + 4) { ::memset(m_ctxInitT, 0, sizeof(m_ctxInitT)); }
#endif

    inline void swap() { std::swap(m_currSbbCtx, m_prevSbbCtx); }

    // template sums contributed by the neighbours outside the current sub-block, for each position inside it
    // (sumNum + (sumAbs1 << 3) + (sumAbs << 8)); computed per state at the end of the previous sub-block and shared by
    // all states that descend from it (a state only keeps the index, instead of a copy per state and scan position).
    // Stored as [position][state]; the columns >= 8 stay zero (CTX_INIT_ZERO).
    inline uint16_t        ctxInit(int idx, int pos) const { return m_ctxInitT[pos][idx]; }
    inline const uint16_t *ctxInitRow(int pos) const { return m_ctxInitT[pos]; }
    inline const BinFracBits &sbbFlagBits(int idx) const { return m_sbbFlagBits[idx]; }

    inline void reset(const TUParameters &tuPars, const RateEstimator &rateEst)
    {
      m_nbInfo = tuPars.m_scanId2NbInfoOut;
      ::memcpy(m_sbbFlagBits, rateEst.sigSbbFracBits(), 2 * sizeof(BinFracBits));
      const int numSbb    = tuPars.m_numSbb;
      const int chunkSize = numSbb + tuPars.m_numCoeff;
      uint8_t * nextMem   = m_memory;
#if TCQ_8STATES
      for (int k = 0; k < 16; k++, nextMem += chunkSize)
#else
      for (int k = 0; k < 8; k++, nextMem += chunkSize)
#endif
      {
        m_allSbbCtx[k].sbbFlags = nextMem;
        m_allSbbCtx[k].levels   = nextMem + numSbb;
      }
    }
#if JVET_AE0102_LFNST_CTX
    inline void update(const ScanInfo &scanInfo, const State *prevState, State &currState, bool lfnstidx);
#else
    inline void update(const ScanInfo &scanInfo, const State *prevState, State &currState);
#endif
    // end of a sub-block for state stateId (sub-block context of its predecessor: prevRefSbbCtxId, -1 = none):
    // sub-block flags and levels memory of the state, template init of the next sub-block (ctxInit(stateId, .));
    // returns the context of the next sub-block's flag (index for sbbFlagBits)
    inline int updateSbb(const ScanInfo &scanInfo, const int prevRefSbbCtxId, const int stateId, const bool sigSbb,
                         const uint8_t *absLevels, const bool lfnstidx);

  private:
    const NbInfoOut *m_nbInfo;
    BinFracBits      m_sbbFlagBits[2];
    alignas(16) uint16_t m_ctxInitT[16][16];
#if TCQ_8STATES
    SbbCtx m_allSbbCtx[16];
#else
    SbbCtx m_allSbbCtx[8];
#endif
    SbbCtx *m_currSbbCtx;
    SbbCtx *m_prevSbbCtx;
#if TCQ_8STATES
    uint8_t m_memory[16 * (MAX_TB_SIZEY * MAX_TB_SIZEY + MLS_GRP_NUM)];
#else
    uint8_t m_memory[8 * (MAX_TB_SIZEY * MAX_TB_SIZEY + MLS_GRP_NUM)];
#endif
  };

#define RICEMAX 32
  const int32_t g_goRiceBits[4][RICEMAX] = {
    { 32768,  65536,  98304,  131072, 163840, 196608, 262144, 262144, 327680, 327680, 327680,
      327680, 393216, 393216, 393216, 393216, 393216, 393216, 393216, 393216, 458752, 458752,
      458752, 458752, 458752, 458752, 458752, 458752, 458752, 458752, 458752, 458752 },
    { 65536,  65536,  98304,  98304,  131072, 131072, 163840, 163840, 196608, 196608, 229376,
      229376, 294912, 294912, 294912, 294912, 360448, 360448, 360448, 360448, 360448, 360448,
      360448, 360448, 425984, 425984, 425984, 425984, 425984, 425984, 425984, 425984 },
    { 98304,  98304,  98304,  98304,  131072, 131072, 131072, 131072, 163840, 163840, 163840,
      163840, 196608, 196608, 196608, 196608, 229376, 229376, 229376, 229376, 262144, 262144,
      262144, 262144, 327680, 327680, 327680, 327680, 327680, 327680, 327680, 327680 },
    { 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 163840, 163840, 163840,
      163840, 163840, 163840, 163840, 163840, 196608, 196608, 196608, 196608, 196608, 196608,
      196608, 196608, 229376, 229376, 229376, 229376, 229376, 229376, 229376, 229376 }
  };

#if defined(__GNUC__) && !defined(__clang__)
#define DQ_UNROLL_STATES _Pragma("GCC unroll 8")   // loops over the trellis states: fully unrolled (constant indices)
#else
#define DQ_UNROLL_STATES
#endif
#if defined(__SSE4_1__)
#define DQ_SSE41 1
#else
#define DQ_SSE41 0
#endif
// trellis of the active configuration: per-TU loop templated on the number of states, states as a structure of
// arrays updated with SSE4.1 (other configurations keep the State based implementation)
#define DQ_TRELLIS_LOOP_TEMPLATE                                                                                       \
  (TCQ_8STATES && TU_256 && !EXTENDED_LFNST && JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS && JVET_AE0102_LFNST_CTX           \
   && JVET_AG0100_TRANSFORM_COEFFICIENT_CODING && DQ_SSE41)

#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
  static const int TPL_LEVEL_CLIP = GTN_LEVEL;   // a template neighbour adds min(TPL_LEVEL_CLIP + (t & 1), t) to sumAbs1
#else
  static const int TPL_LEVEL_CLIP = 4;
#endif
  alignas(16) static const uint8_t g_zeroAbsLevels[16] = { 0 };

  // dst = src with dst[insidePos] = val (16 levels of a sub-block; written as one vector, so the next stage's load of
  // the whole vector is forwarded from the store instead of waiting for a byte store to retire)
  static inline void setAbsLevels(uint8_t *dst, const uint8_t *src, const int insidePos, const uint8_t val)
  {
#if defined(__SSE4_1__)
    const __m128i lv  = _mm_load_si128((const __m128i *) src);
    const __m128i msk = _mm_cmpeq_epi8(_mm_set1_epi8((char) insidePos),
                                       _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    _mm_store_si128((__m128i *) dst, _mm_blendv_epi8(lv, _mm_set1_epi8((char) val), msk));
#else
    ::memcpy(dst, src, 16);
    dst[insidePos] = val;
#endif
  }

  // template sums over the neighbours inside the sub-block: levels[shuffle[k]] for the leading shuffle entries < 16
  static inline void tplSumsInside(const uint8_t *levels, const uint8_t *shuffle, int &sumAbs1, int &sumNum,
                                   int &sumAbs)
  {
#if defined(__SSE4_1__)
    const __m128i zero = _mm_setzero_si128();
    const __m128i one  = _mm_set1_epi8(1);
    const __m128i nb   = _mm_shuffle_epi8(_mm_load_si128((const __m128i *) levels),
                                          _mm_loadu_si128((const __m128i *) shuffle));   // unused lanes are 0
    const __m128i abs1 = _mm_min_epu8(nb, _mm_add_epi8(_mm_and_si128(nb, one), _mm_set1_epi8(TPL_LEVEL_CLIP)));
    const __m128i num  = _mm_min_epu8(nb, one);
    const __m128i s12  = _mm_sad_epu8(_mm_unpacklo_epi64(abs1, num), zero);
    sumAbs1            = _mm_cvtsi128_si32(s12);
    sumNum             = _mm_extract_epi16(s12, 4);
    sumAbs             = _mm_cvtsi128_si32(_mm_sad_epu8(nb, zero));
#else
    sumAbs1 = sumNum = sumAbs = 0;
    for (int k = 0; k < 5 && shuffle[k] < 16; k++)
    {
      const int t = levels[shuffle[k]];
      sumAbs += t;
      sumAbs1 += std::min(TPL_LEVEL_CLIP + (t & 1), t);
      sumNum += !!t;
    }
#endif
  }

  class State
  {
    friend class CommonCtx;

  public:
    State(const RateEstimator &rateEst, CommonCtx &commonCtx, const int stateId);

#if JVET_AE0102_LFNST_CTX
    inline void updateState(const ScanInfo &scanInfo, const State *prevStates, const Decision &decision, int lfnstIdx);
#else
    inline void updateState(const ScanInfo &scanInfo, const State *prevStates, const Decision &decision);
#endif
#if TCQ_8STATES
#if JVET_AE0102_LFNST_CTX
    inline void updateStateEOS(const ScanInfo &scanInfo, const State *prevStates, const State *skipStates,
                               const Decision &decision, int skipOffsets, bool bLfnst);
#else
    inline void updateStateEOS(const ScanInfo &scanInfo, const State *prevStates, const State *skipStates,
                               const Decision &decision, int skipOffsets);
#endif

#else
    inline void updateStateEOS(const ScanInfo &scanInfo, const State *prevStates, const State *skipStates,
                               const Decision &decision);
#endif

#if TCQ_8STATES
    inline void init(int effectiveWidth, int effectiveHeight)
#else
    inline void init()
#endif
    {
      ::memset(m_absLevels, 0, sizeof(m_absLevels));
      m_ctxInitIdx    = CTX_INIT_ZERO;
      m_rdCost        = std::numeric_limits<int64_t>::max() >> 1;
      m_numSigSbb     = 0;
      m_remRegBins    = 4;   // just large enough for last scan pos
      m_refSbbCtxId   = -1;
      m_sigFracBits   = m_sigFracBitsArray;
      m_coeffFracBits = m_gtxFracBitsArray;
      m_goRicePar     = 0;
      m_goRiceZero    = 0;
#if TCQ_8STATES
      effWidth  = effectiveWidth;
      effHeight = effectiveHeight;
#endif
    }
    void checkRdCosts(const ScanPosType spt, const PQData &pqDataA, const PQData &pqDataB, Decision &decisionA,
                      Decision &decisionB) const
    {
      int64_t rdCostA, rdCostB, rdCostZ;
      calcRdCosts(spt, pqDataA, pqDataB, rdCostA, rdCostB, rdCostZ);
      if (rdCostA < decisionA.rdCost)
      {
        decisionA.rdCost   = rdCostA;
        decisionA.absLevel = pqDataA.absLevel;
        decisionA.prevId   = m_stateId;
      }
      if (rdCostZ < decisionA.rdCost)
      {
        decisionA.rdCost   = rdCostZ;
        decisionA.absLevel = 0;
        decisionA.prevId   = m_stateId;
      }
      if (rdCostB < decisionB.rdCost)
      {
        decisionB.rdCost   = rdCostB;
        decisionB.absLevel = pqDataB.absLevel;
        decisionB.prevId   = m_stateId;
      }
    }

    // costs of coding level A, level B or zero from this state; a zero that is not allowed (the last coefficient of a
    // coded sub-block without other significant coefficients) gets INT64_MAX, so it never wins (the stock code
    // compared a cost equal to the current decision, which never wins either)
    inline void calcRdCosts(const ScanPosType spt, const PQData &pqDataA, const PQData &pqDataB, int64_t &costA,
                            int64_t &costB, int64_t &costZ) const
    {
      const int32_t *goRiceTab = g_goRiceBits[m_goRicePar];
      int64_t        rdCostA   = m_rdCost + pqDataA.deltaDist;
      int64_t        rdCostB   = m_rdCost + pqDataB.deltaDist;
      int64_t        rdCostZ   = m_rdCost;

      const TCoeff   absLevelA = pqDataA.absLevel;
      const TCoeff   absLevelB = pqDataB.absLevel;
      const int32_t *bits      = m_coeffFracBits->bits;

      if (m_remRegBins >= 4)
      {
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        if (absLevelA < GTN_LEVEL)
#else
        if (absLevelA < 4)
#endif
        {
          rdCostA += bits[absLevelA];
        }
        else
        {
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
          const TCoeff value = (absLevelA - GTN_LEVEL) >> 1;
#else
          const TCoeff value = (absLevelA - 4) >> 1;
#endif
#else
          const int value = (absLevelA - 4) >> 1;
#endif
          rdCostA += bits[absLevelA - (value << 1)] + goRiceTab[value < RICEMAX ? value : RICEMAX - 1];
        }

#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        if (absLevelB < GTN_LEVEL)
#else
        if (absLevelB < 4)
#endif
        {
          rdCostB += bits[absLevelB];
        }
        else
        {
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
          const TCoeff value = ( absLevelB - GTN_LEVEL ) >> 1;
#else
          const TCoeff value = ( absLevelB - 4 ) >> 1;
#endif
#else
          const int value = (absLevelB - 4) >> 1;
#endif
          rdCostB += bits[absLevelB - (value << 1)] + goRiceTab[value < RICEMAX ? value : RICEMAX - 1];
        }
        const int sigBit1 = m_sigFracBits->intBits[1];

        if (spt == SCAN_ISCSBB)
        {
          rdCostA += sigBit1;
          rdCostB += sigBit1;
          rdCostZ += m_sigFracBits->intBits[0];
        }
        else if (spt == SCAN_SOCSBB)
        {
          const int sbbBit1 = m_sbbFracBits.intBits[1];

          rdCostA += sbbBit1 + sigBit1;
          rdCostB += sbbBit1 + sigBit1;
          rdCostZ += sbbBit1 + m_sigFracBits->intBits[0];
        }
        else if (m_numSigSbb)
        {
          rdCostA += sigBit1;
          rdCostB += sigBit1;
          rdCostZ += m_sigFracBits->intBits[0];
        }
        else
        {
          rdCostZ = std::numeric_limits<int64_t>::max();
        }
      }
      else
      {
        rdCostA +=
          (1 << SCALE_BITS)
          + goRiceTab[absLevelA <= m_goRiceZero ? absLevelA - 1 : (absLevelA < RICEMAX ? absLevelA : RICEMAX - 1)];
        rdCostB +=
          (1 << SCALE_BITS)
          + goRiceTab[absLevelB <= m_goRiceZero ? absLevelB - 1 : (absLevelB < RICEMAX ? absLevelB : RICEMAX - 1)];
        rdCostZ += goRiceTab[m_goRiceZero];
      }
      costA = rdCostA;
      costB = rdCostB;
      costZ = rdCostZ;
    }

    inline void checkRdCostStart(int32_t lastOffset, const PQData &pqData, Decision &decision) const
    {
      const int64_t rdCost = calcRdCostStart(lastOffset, pqData);
      if (rdCost < decision.rdCost)
      {
        decision.rdCost   = rdCost;
        decision.absLevel = pqData.absLevel;
        decision.prevId   = -1;
      }
    }

    inline int64_t rdCostSkipSbb() const { return m_rdCost + m_sbbFracBits.intBits[0]; }

    inline int64_t calcRdCostStart(int32_t lastOffset, const PQData &pqData) const
    {
      int64_t rdCost = pqData.deltaDist + lastOffset;
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
      if (pqData.absLevel < GTN_LEVEL)
#else
      if (pqData.absLevel < 4)
#endif
      {
        rdCost += m_coeffFracBits->bits[pqData.absLevel];
      }
      else
      {
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        const TCoeff value = (pqData.absLevel - GTN_LEVEL) >> 1;
#else
        const TCoeff value = (pqData.absLevel - 4) >> 1;
#endif
#else
        const unsigned value = (pqData.absLevel - 4) >> 1;
#endif
        rdCost += m_coeffFracBits->bits[pqData.absLevel - (value << 1)]
                  + g_goRiceBits[m_goRicePar][value < RICEMAX ? value : RICEMAX - 1];
      }
      return rdCost;
    }

#if TCQ_8STATES
    inline void checkRdCostSkipSbb(Decision &decision, int skipOffset) const
#else
    inline void checkRdCostSkipSbb(Decision &decision) const
#endif
    {
      int64_t rdCost = rdCostSkipSbb();
      if (rdCost < decision.rdCost)
      {
        decision.rdCost   = rdCost;
        decision.absLevel = 0;
#if TCQ_8STATES
        decision.prevId = skipOffset + m_stateId;
#else
        decision.prevId = 4 + m_stateId;
#endif
      }
    }
#if !TU_256 || EXTENDED_LFNST
#if TCQ_8STATES
    inline void checkRdCostSkipSbbZeroOut(Decision &decision, int skipOffset) const
#else
    inline void checkRdCostSkipSbbZeroOut(Decision &decision) const
#endif
    {
      int64_t rdCost    = m_rdCost + m_sbbFracBits.intBits[0];
      decision.rdCost   = rdCost;
      decision.absLevel = 0;
#if TCQ_8STATES
      decision.prevId = skipOffset + m_stateId;
#else
      decision.prevId = 4 + m_stateId;
#endif
    }
#endif

  private:
    int64_t                    m_rdCost;
    alignas(16) uint8_t        m_absLevels[16];   // levels of the current sub-block by position inside it (max 255)
    uint8_t                    m_ctxInitIdx;      // template sums from outside the sub-block: m_commonCtx.ctxInit(idx)
    int8_t                     m_numSigSbb;
    int                        m_remRegBins;
    int8_t                     m_refSbbCtxId;
    BinFracBits                m_sbbFracBits;
    const BinFracBits *        m_sigFracBits;     // points into the rate tables (fixed while a TU is quantized)
    const CoeffFracBits *      m_coeffFracBits;
    int8_t                     m_goRicePar;
    int8_t                     m_goRiceZero;
    const int8_t               m_stateId;
    const BinFracBits *const   m_sigFracBitsArray;
    const CoeffFracBits *const m_gtxFracBitsArray;
    CommonCtx &                m_commonCtx;

  public:
    unsigned effWidth;
    unsigned effHeight;
  };

  State::State(const RateEstimator &rateEst, CommonCtx &commonCtx, const int stateId)
    : m_sbbFracBits{ { 0, 0 } }
    , m_sigFracBits(rateEst.sigFlagBits(stateId))
    , m_coeffFracBits(rateEst.gtxFracBits(stateId))
    , m_stateId(stateId)
    , m_sigFracBitsArray(rateEst.sigFlagBits(stateId))
    , m_gtxFracBitsArray(rateEst.gtxFracBits(stateId))
    , m_commonCtx(commonCtx)
  {
  }

#if JVET_AE0102_LFNST_CTX
  inline void State::updateState(const ScanInfo &scanInfo, const State *prevStates, const Decision &decision,
                                 int lfnstIdx)
#else
  inline void State::updateState(const ScanInfo &scanInfo, const State *prevStates, const Decision &decision)
#endif
  {
    m_rdCost = decision.rdCost;
    if (decision.prevId > -2)
    {
      const uint8_t *prvLevels;
      if (decision.prevId >= 0)
      {
        const State *prvState = prevStates + decision.prevId;
        m_numSigSbb           = prvState->m_numSigSbb + !!decision.absLevel;
        m_refSbbCtxId         = prvState->m_refSbbCtxId;
        m_sbbFracBits         = prvState->m_sbbFracBits;
        m_remRegBins          = prvState->m_remRegBins - 1;
        m_goRicePar           = prvState->m_goRicePar;
        if (m_remRegBins >= 4)
        {
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
          m_remRegBins -= (decision.absLevel < (GTN_LEVEL - 1) ? (unsigned)decision.absLevel : (GTN_LEVEL - 1));
#else
          m_remRegBins -= (decision.absLevel < 2 ? (unsigned)decision.absLevel : 3);
#endif
#else
          m_remRegBins -= (decision.absLevel < 2 ? decision.absLevel : 3);
#endif
        }
        m_ctxInitIdx = prvState->m_ctxInitIdx;
        prvLevels    = prvState->m_absLevels;
      }
      else
      {
        m_numSigSbb           = 1;
        m_refSbbCtxId         = -1;
        int ctxBinSampleRatio = (scanInfo.chType == CHANNEL_TYPE_LUMA) ? MAX_TU_LEVEL_CTX_CODED_BIN_CONSTRAINT_LUMA
                                                                       : MAX_TU_LEVEL_CTX_CODED_BIN_CONSTRAINT_CHROMA;
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        m_remRegBins = (effWidth * effHeight * ctxBinSampleRatio) / 16 - (decision.absLevel < (GTN_LEVEL - 1) ? (unsigned)decision.absLevel : (GTN_LEVEL - 1));
#else
        m_remRegBins = (effWidth * effHeight * ctxBinSampleRatio) / 16 - (decision.absLevel < 2 ? (unsigned)decision.absLevel : 3);
#endif
#else
        m_remRegBins =
          (effWidth * effHeight * ctxBinSampleRatio) / 16 - (decision.absLevel < 2 ? decision.absLevel : 3);
#endif
        m_ctxInitIdx = CTX_INIT_ZERO;
        prvLevels    = g_zeroAbsLevels;
      }

      setAbsLevels(m_absLevels, prvLevels, scanInfo.insidePos, (uint8_t) std::min<TCoeff>(255, decision.absLevel));

      // template of the next scan position: neighbours outside (ctx init) and inside the sub-block
      const int tinit = m_commonCtx.ctxInit(m_ctxInitIdx, scanInfo.nextInsidePos);
#if JVET_AE0102_LFNST_CTX
      const uint8_t *nbShuffle = scanInfo.nextNbShuffle[lfnstIdx > 0 ? 1 : 0];
#else
      const uint8_t *nbShuffle = scanInfo.nextNbShuffle[0];
#endif
      int sumAbs1In, sumNumIn, sumAbsIn;
      tplSumsInside(m_absLevels, nbShuffle, sumAbs1In, sumNumIn, sumAbsIn);

      if (m_remRegBins >= 4)
      {
        const int sumAbs1 = ((tinit >> 3) & 31) + sumAbs1In;
        const int sumNum  = (tinit & 7) + sumNumIn;
        const int sumGt1  = sumAbs1 - sumNum;
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        m_sigFracBits   = m_sigFracBitsArray + (scanInfo.sigCtxOffsetNext + std::min((sumAbs1 + 1) >> 1, NSIGCTX - 1));
        m_coeffFracBits = m_gtxFracBitsArray + (scanInfo.gtxCtxOffsetNext + std::min(sumGt1, NGTXCTX - 1));
#else
        m_sigFracBits   = m_sigFracBitsArray + (scanInfo.sigCtxOffsetNext + std::min((sumAbs1 + 1) >> 1, 3));
        m_coeffFracBits = m_gtxFracBitsArray + (scanInfo.gtxCtxOffsetNext + std::min(sumGt1, 4));
#endif

        const int sumAbs = (tinit >> 8) + sumAbsIn;
#if !JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        int sumAll = std::max(std::min(31, sumAbs - 4 * 5), 0);
        m_goRicePar = g_auiGoRiceParsCoeff[sumAll];
#else
        int sumAll = std::max(std::min(GTN_MAXSUM - 1, sumAbs - sumNum), 0);
        m_goRicePar = g_auiGoRiceParsCoeffGTN[sumAll];
#endif
      }
      else
      {
        const int sumAbs = std::min(31, (tinit >> 8) + sumAbsIn);
        m_goRicePar      = g_auiGoRiceParsCoeff[sumAbs];
        m_goRiceZero     = g_auiGoRicePosCoeff0(m_stateId, m_goRicePar);
      }
    }
  }

#if TCQ_8STATES
#if JVET_AE0102_LFNST_CTX
  inline void State::updateStateEOS(const ScanInfo &scanInfo, const State *prevStates, const State *skipStates,
                                    const Decision &decision, int skipOffset, bool bLfnst)
#else
  inline void State::updateStateEOS(const ScanInfo &scanInfo, const State *prevStates, const State *skipStates,
                                    const Decision &decision, int skipOffset)
#endif
#else
  inline void State::updateStateEOS(const ScanInfo &scanInfo, const State *prevStates, const State *skipStates,
                                    const Decision &decision)
#endif
  {
    m_rdCost = decision.rdCost;
    if (decision.prevId > -2)
    {
      const State   *prvState  = 0;
      const uint8_t *prvLevels = g_zeroAbsLevels;
#if TCQ_8STATES
      if (decision.prevId >= skipOffset)
#else
      if (decision.prevId >= 4)
#endif
      {
        CHECK(decision.absLevel != 0, "cannot happen");
#if TCQ_8STATES
        prvState = skipStates + (decision.prevId - skipOffset);
#else
        prvState = skipStates + (decision.prevId - 4);
#endif
        m_numSigSbb = 0;
      }
      else if (decision.prevId >= 0)
      {
        prvState    = prevStates + decision.prevId;
        m_numSigSbb = prvState->m_numSigSbb + !!decision.absLevel;
        prvLevels   = prvState->m_absLevels;
      }
      else
      {
        m_numSigSbb = 1;
      }
      setAbsLevels(m_absLevels, prvLevels, scanInfo.insidePos, (uint8_t) std::min<TCoeff>(255, decision.absLevel));
#if JVET_AE0102_LFNST_CTX
      m_commonCtx.update(scanInfo, prvState, *this, bLfnst);
#else
      m_commonCtx.update(scanInfo, prvState, *this);
#endif

      TCoeff  tinit   = m_commonCtx.ctxInit(m_ctxInitIdx, scanInfo.nextInsidePos);
      TCoeff  sumNum  =   tinit        & 7;
      TCoeff  sumAbs1 = ( tinit >> 3 ) & 31;
      TCoeff  sumGt1  = sumAbs1        - sumNum;

#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
      m_sigFracBits = m_sigFracBitsArray + (scanInfo.sigCtxOffsetNext + std::min<TCoeff>((sumAbs1 + 1) >> 1, NSIGCTX-1));
#else
      m_sigFracBits   = m_sigFracBitsArray + ( scanInfo.sigCtxOffsetNext + std::min<TCoeff>( (sumAbs1+1)>>1, 3 ) );
#endif
#else
      m_sigFracBits = m_sigFracBitsArray + (scanInfo.sigCtxOffsetNext + std::min((sumAbs1 + 1) >> 1, 3));
#endif
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
      m_coeffFracBits = m_gtxFracBitsArray + (scanInfo.gtxCtxOffsetNext + (sumGt1 < (NGTXCTX-1) ? sumGt1 : (NGTXCTX - 1)));
#else
      m_coeffFracBits = m_gtxFracBitsArray + ( scanInfo.gtxCtxOffsetNext + ( sumGt1  < 4 ? sumGt1  : 4 ) );
#endif

    }
  }
  inline int CommonCtx::updateSbb(const ScanInfo &scanInfo, const int prevRefSbbCtxId, const int stateId,
                                  const bool sigSbb, const uint8_t *absLevels16, const bool lfnstidx)
  {
    uint8_t *sbbFlags = m_currSbbCtx[stateId].sbbFlags;
    uint8_t *levels   = m_currSbbCtx[stateId].levels;
#if JVET_AE0102_LFNST_CTX
    std::size_t setCpSize = m_nbInfo[scanInfo.scanIdx - 1].maxDist[lfnstidx] * sizeof(uint8_t);
#else
    std::size_t setCpSize = m_nbInfo[scanInfo.scanIdx - 1].maxDist * sizeof(uint8_t);
#endif
    if (prevRefSbbCtxId >= 0)
    {
      ::memcpy(sbbFlags, m_prevSbbCtx[prevRefSbbCtxId].sbbFlags, scanInfo.numSbb * sizeof(uint8_t));
      ::memcpy(levels + scanInfo.scanIdx, m_prevSbbCtx[prevRefSbbCtxId].levels + scanInfo.scanIdx, setCpSize);
    }
    else
    {
      ::memset(sbbFlags, 0, scanInfo.numSbb * sizeof(uint8_t));
      ::memset(levels + scanInfo.scanIdx, 0, setCpSize);
    }
    sbbFlags[scanInfo.sbbPos] = sigSbb;
    ::memcpy(levels + scanInfo.scanIdx, absLevels16, scanInfo.sbbSize * sizeof(uint8_t));

    const int sigNSbb = ((scanInfo.nextSbbRight ? sbbFlags[scanInfo.nextSbbRight] : false)
                             || (scanInfo.nextSbbBelow ? sbbFlags[scanInfo.nextSbbBelow] : false)
                           ? 1
                           : 0);

    const int        scanBeg   = scanInfo.scanIdx - scanInfo.sbbSize;
    const NbInfoOut *nbOut     = m_nbInfo + scanBeg;
    const uint8_t *  absLevels = levels + scanBeg;
    for (int id = 0; id < scanInfo.sbbSize; id++, nbOut++)
    {
#if JVET_AE0102_LFNST_CTX
      if (nbOut->num[lfnstidx])
      {
        TCoeff sumAbs = 0, sumAbs1 = 0, sumNum = 0;
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
#define UPDATE(k) {TCoeff t=absLevels[nbOut->outPos[lfnstidx][k]]; sumAbs+=t; sumAbs1+=std::min<TCoeff>(GTN_LEVEL+(t&1),t); sumNum+=!!t; }
#else
#define UPDATE(k) {TCoeff t=absLevels[nbOut->outPos[lfnstidx][k]]; sumAbs+=t; sumAbs1+=std::min<TCoeff>(4+(t&1),t); sumNum+=!!t; }
#endif
        UPDATE(0);
        if (nbOut->num[lfnstidx] > 1)
        {
          UPDATE(1);
          if (nbOut->num[lfnstidx] > 2)
          {
            UPDATE(2);
            if (nbOut->num[lfnstidx] > 3)
            {
              UPDATE(3);
              if (nbOut->num[lfnstidx] > 4)
              {
                UPDATE(4);
              }
            }
          }
        }
#else
      if (nbOut->num)
      {
        TCoeff sumAbs = 0, sumAbs1 = 0, sumNum = 0;
#define UPDATE(k)                                                                                                      \
  {                                                                                                                    \
    TCoeff t = absLevels[nbOut->outPos[k]];                                                                            \
    sumAbs += t;                                                                                                       \
    sumAbs1 += std::min<TCoeff>(4 + (t & 1), t);                                                                       \
    sumNum += !!t;                                                                                                     \
  }
        UPDATE(0);
        if (nbOut->num > 1)
        {
          UPDATE(1);
          if (nbOut->num > 2)
          {
            UPDATE(2);
            if (nbOut->num > 3)
            {
              UPDATE(3);
              if (nbOut->num > 4)
              {
                UPDATE(4);
              }
            }
          }
        }
#endif
#undef UPDATE
#if JVET_AG0100_TRANSFORM_COEFFICIENT_CODING
        m_ctxInitT[id][stateId] = uint16_t(sumNum) + (uint16_t(std::min<TCoeff>(31, sumAbs1)) << 3) + ((uint16_t)std::min<TCoeff>(127, sumAbs) << 8);
#else
        m_ctxInitT[id][stateId] = uint16_t(sumNum) + ( uint16_t(sumAbs1) << 3 ) + ( (uint16_t)std::min<TCoeff>( 127, sumAbs ) << 8 );
#endif

      }
      else
      {
        m_ctxInitT[id][stateId] = 0;
      }
    }
    for (int id = scanInfo.sbbSize; id < 16; id++)
    {
      m_ctxInitT[id][stateId] = 0;
    }
    return sigNSbb;
  }

#if JVET_AE0102_LFNST_CTX
  inline void CommonCtx::update(const ScanInfo &scanInfo, const State *prevState, State &currState, bool lfnstidx)
#else
  inline void CommonCtx::update(const ScanInfo &scanInfo, const State *prevState, State &currState)
#endif
  {
#if !JVET_AE0102_LFNST_CTX
    const bool lfnstidx = false;
#endif
    const int sigNSbb = updateSbb(scanInfo, prevState ? prevState->m_refSbbCtxId : -1, currState.m_stateId,
                                  currState.m_numSigSbb != 0, currState.m_absLevels, lfnstidx);
    currState.m_numSigSbb = 0;
    if (prevState)
    {
      currState.m_remRegBins = prevState->m_remRegBins;
    }
    else
    {
      int ctxBinSampleRatio  = (scanInfo.chType == CHANNEL_TYPE_LUMA) ? MAX_TU_LEVEL_CTX_CODED_BIN_CONSTRAINT_LUMA
                                                                      : MAX_TU_LEVEL_CTX_CODED_BIN_CONSTRAINT_CHROMA;
      currState.m_remRegBins = (currState.effWidth * currState.effHeight * ctxBinSampleRatio) / 16;
    }
    currState.m_goRicePar   = 0;
    currState.m_refSbbCtxId = currState.m_stateId;
    currState.m_sbbFracBits = m_sbbFlagBits[sigNSbb];
    ::memset(currState.m_absLevels, 0, sizeof(currState.m_absLevels));
    currState.m_ctxInitIdx = currState.m_stateId;
  }

#if DQ_TRELLIS_LOOP_TEMPLATE
  // The states of one trellis stage as a structure of arrays (lane = state id; 4 or 8 states), so that the state
  // update after each decision runs on all states at once. Relation to the members of State:
  //   m_sigFracBits = sigFlagBits(state) + sigCtx, m_coeffFracBits = gtxFracBits + gtxCtx,
  //   m_sbbFracBits = { sbbBits0, sbbBits1 }, m_absLevelsAndCtxInit = absLevels column + ctxInit(ctxInitIdx).
  // Every lane reproduces its State exactly, including states without a path (decision.prevId == -2, which keep
  // all members) and their descendants: their members only enter costs that never win against a real path, but the
  // slot values they leave behind (e.g. goRiceZero, which a sub-block end keeps) are read later by real paths.
  // numSigSbb, refSbbCtxId, ctxInitIdx, sbbBits, remRegBins and absLevels are taken over from the predecessor
  // (shuffles); goRicePar, goRiceZero, sigCtx and gtxCtx belong to the state slot (recomputed or kept as in State).
  struct alignas(64) StateArr
  {
    int64_t  rdCost[8];
    int32_t  remRegBins[8];
    uint32_t sbbBits0[8];   // m_sbbFracBits: kept by the start of a path and by State::init (i.e. across TUs)
    uint32_t sbbBits1[8];
    int8_t   numSigSbb[8];   // 16-byte pairs: { numSigSbb, refSbbCtxId }, { ctxInitIdx, (unused) },
    int8_t   refSbbCtxId[8];   //                { goRicePar, goRiceZero }, { sigCtx, gtxCtx }
    uint8_t  ctxInitIdx[8];
    uint8_t  unused[8];
    int8_t   goRicePar[8];
    int8_t   goRiceZero[8];
    uint8_t  sigCtx[8];
    uint8_t  gtxCtx[8];
    uint8_t  absLevels[17][8];   // levels of the current sub-block [position][state] (max 255); row 16 stays zero
  };

  // [position][state] -> [state][position] (16 positions, 8 states)
  static inline void transposeAbsLevels(const uint8_t src[17][8], uint8_t dst[8][16])
  {
#if defined(__SSE4_1__)
    const __m128i ilv = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    __m128i       t[8];
    for (int r = 0; r < 8; r++)
    {
      // 16-bit element s: levels of state s at positions 2r and 2r+1
      t[r] = _mm_shuffle_epi8(_mm_load_si128((const __m128i *) src[2 * r]), ilv);
    }
    const __m128i a0 = _mm_unpacklo_epi16(t[0], t[1]);
    const __m128i a1 = _mm_unpackhi_epi16(t[0], t[1]);
    const __m128i a2 = _mm_unpacklo_epi16(t[2], t[3]);
    const __m128i a3 = _mm_unpackhi_epi16(t[2], t[3]);
    const __m128i a4 = _mm_unpacklo_epi16(t[4], t[5]);
    const __m128i a5 = _mm_unpackhi_epi16(t[4], t[5]);
    const __m128i a6 = _mm_unpacklo_epi16(t[6], t[7]);
    const __m128i a7 = _mm_unpackhi_epi16(t[6], t[7]);
    const __m128i b0 = _mm_unpacklo_epi32(a0, a2);
    const __m128i b1 = _mm_unpackhi_epi32(a0, a2);
    const __m128i b2 = _mm_unpacklo_epi32(a1, a3);
    const __m128i b3 = _mm_unpackhi_epi32(a1, a3);
    const __m128i b4 = _mm_unpacklo_epi32(a4, a6);
    const __m128i b5 = _mm_unpackhi_epi32(a4, a6);
    const __m128i b6 = _mm_unpacklo_epi32(a5, a7);
    const __m128i b7 = _mm_unpackhi_epi32(a5, a7);
    _mm_store_si128((__m128i *) dst[0], _mm_unpacklo_epi64(b0, b4));
    _mm_store_si128((__m128i *) dst[1], _mm_unpackhi_epi64(b0, b4));
    _mm_store_si128((__m128i *) dst[2], _mm_unpacklo_epi64(b1, b5));
    _mm_store_si128((__m128i *) dst[3], _mm_unpackhi_epi64(b1, b5));
    _mm_store_si128((__m128i *) dst[4], _mm_unpacklo_epi64(b2, b6));
    _mm_store_si128((__m128i *) dst[5], _mm_unpackhi_epi64(b2, b6));
    _mm_store_si128((__m128i *) dst[6], _mm_unpacklo_epi64(b3, b7));
    _mm_store_si128((__m128i *) dst[7], _mm_unpackhi_epi64(b3, b7));
#else
    for (int s = 0; s < 8; s++)
    {
      for (int pos = 0; pos < 16; pos++)
      {
        dst[s][pos] = src[pos][s];
      }
    }
#endif
  }
#endif

  /*================================================================================*/
  /*=====                                                                      =====*/
  /*=====   T C Q                                                              =====*/
  /*=====                                                                      =====*/
  /*================================================================================*/
  class DepQuant : private RateEstimator
  {
  public:
    DepQuant();

    void quant(TransformUnit &tu, const CCoeffBuf &srcCoeff, const ComponentID compID, const QpParam &cQP,
               const double lambda, const Ctx &ctx, TCoeff &absSum, bool enableScalingLists, int *quantCoeff);
    void dequant(const TransformUnit &tu, CoeffBuf &recCoeff, const ComponentID compID, const QpParam &cQP,
                 bool enableScalingLists, int *quantCoeff);

  private:
#if DQ_TRELLIS_LOOP_TEMPLATE
    template<int numStates>
    void xQuantTrellis(const TUParameters &tuPars, const TCoeff *tCoeff, const int firstTestPos, const int lfnstIdx,
                       const TCoeff defaultQuantisationCoefficient, const bool enableScalingLists,
                       const int *quantCoeff, const TransformUnit &tu, const ComponentID compID, const QpParam &cQP,
                       const double lambda);
    void xInitStates(const unsigned effWidth, const unsigned effHeight, const ChannelType chType);
    template<int numStates>
    void xDecideSoA(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions,
                    const TCoeff quanCoeff);
    template<int numStates>
    void xUpdateStatesSoA(const ScanInfo &scanInfo, const Decision *decisions, const int ctxType);
    template<int numStates>
    void xUpdateStatesEosSoA(const ScanInfo &scanInfo, const Decision *decisions, const int ctxType);
    inline void calcRdCostsSoA(const StateArr &st, const int s, const ScanPosType spt, const PQData &pqDataA,
                               const PQData &pqDataB, int64_t &costA, int64_t &costB, int64_t &costZ) const;
    inline int64_t calcRdCostStartSoA(const int32_t lastOffset, const PQData &pqData) const;

    StateArr           m_stateArr[3];
    StateArr *         m_currArr;
    StateArr *         m_prevArr;
    StateArr *         m_skipArr;
    const BinFracBits *m_sigTab[8];         // sig flag rate table of each state id
    int32_t            m_initRemRegBins;    // budget of context coded bins of the current TU
    int16_t            m_riceThrGtn[3];     // g_auiGoRiceParsCoeffGTN[x] = number of thresholds <= x
    int16_t            m_riceThrCoeff[3];   // g_auiGoRiceParsCoeff[x]    = number of thresholds <= x
#endif
#if TU_256 && !EXTENDED_LFNST
#if TCQ_8STATES
    template<int numStates>
#endif
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
#if JVET_AE0102_LFNST_CTX
    void xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, TCoeff quantCoeff, int lfnstIdx);
#else
    void xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, TCoeff quantCoeff);
#endif
#else
    void xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, int quantCoeff);
#endif
#if TCQ_8STATES
    template<int numStates>
#endif
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
    void xDecide(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions,
                 TCoeff quantCoeff);
#else
    void xDecide(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions,
                 int quantCoeff);
#endif
#else
#if TCQ_8STATES
    template<int numStates>
#endif
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
    void xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, bool zeroOut, TCoeff quantCoeff);
#else
    void xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, bool zeroOut, int quantCoeff);
#endif
#if TCQ_8STATES
    template<int numStates>
#endif
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
    void xDecide(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions, bool zeroOut,
                 TCoeff quantCoeff);
#else
    void xDecide(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions, bool zeroOut,
                 int quantCoeff);
#endif
#endif

  private:
    CommonCtx m_commonCtx;
#if TCQ_8STATES
    State m_allStates[24];
#else
    State m_allStates[12];
#endif
    State *   m_currStates;
    State *   m_prevStates;
    State *   m_skipStates;
    State     m_startState;
    Quantizer m_quant;
#if TCQ_8STATES
    Decision m_trellis[MAX_TB_SIZEY * MAX_TB_SIZEY * 8];
#else
    Decision m_trellis[MAX_TB_SIZEY * MAX_TB_SIZEY][8];
#endif
  };

#define TINIT(x) { *this, m_commonCtx, x }
  DepQuant::DepQuant()
    : RateEstimator()
    , m_commonCtx()
#if TCQ_8STATES
    , m_allStates
  {
    TINIT(0), TINIT(1), TINIT(2), TINIT(3), TINIT(4), TINIT(5), TINIT(6), TINIT(7), TINIT(0), TINIT(1), TINIT(2),
      TINIT(3), TINIT(4), TINIT(5), TINIT(6), TINIT(7), TINIT(0), TINIT(1), TINIT(2), TINIT(3), TINIT(4), TINIT(5),
      TINIT(6), TINIT(7)
  }
#else
    , m_allStates
  {
    TINIT(0), TINIT(1), TINIT(2), TINIT(3), TINIT(0), TINIT(1), TINIT(2), TINIT(3), TINIT(0), TINIT(1), TINIT(2),
      TINIT(3)
  }
#endif
  , m_currStates(m_allStates)
#if TCQ_8STATES
      ,
    m_prevStates(m_currStates + 8), m_skipStates(m_prevStates + 8)
#else
      ,
    m_prevStates(m_currStates + 4), m_skipStates(m_prevStates + 4)
#endif
                                      ,
    m_startState TINIT(0)
  {
#if DQ_TRELLIS_LOOP_TEMPLATE
    ::memset(m_stateArr, 0, sizeof(m_stateArr));
    m_currArr = m_stateArr;
    m_prevArr = m_stateArr + 1;
    m_skipArr = m_stateArr + 2;
    for (int k = 0; k < 8; k++)
    {
      m_sigTab[k] = sigFlagBits(k);
    }
    m_initRemRegBins = 0;
    // the Rice parameter tables are non-decreasing with values 0..3: parameter = number of thresholds reached
    for (int i = 1; i < GTN_MAXSUM; i++)
    {
      CHECK(g_auiGoRiceParsCoeffGTN[i] < g_auiGoRiceParsCoeffGTN[i - 1] || g_auiGoRiceParsCoeffGTN[i] > 3,
            "unexpected Rice parameter table");
    }
    for (int i = 1; i < 32; i++)
    {
      CHECK(g_auiGoRiceParsCoeff[i] < g_auiGoRiceParsCoeff[i - 1] || g_auiGoRiceParsCoeff[i] > 3,
            "unexpected Rice parameter table");
    }
    for (int t = 0; t < 3; t++)
    {
      m_riceThrGtn[t]   = GTN_MAXSUM;
      m_riceThrCoeff[t] = 32;
      for (int i = GTN_MAXSUM - 1; i >= 0; i--)
      {
        m_riceThrGtn[t] = (g_auiGoRiceParsCoeffGTN[i] > (uint32_t) t ? i : m_riceThrGtn[t]);
      }
      for (int i = 31; i >= 0; i--)
      {
        m_riceThrCoeff[t] = (g_auiGoRiceParsCoeff[i] > (uint32_t) t ? i : m_riceThrCoeff[t]);
      }
    }
#endif
  }
#undef TINIT

  void DepQuant::dequant(const TransformUnit &tu, CoeffBuf &recCoeff, const ComponentID compID, const QpParam &cQP,
                         bool enableScalingLists, int *piDequantCoef)
  {
#if TCQ_8STATES
    m_quant.dequantBlock(tu, compID, cQP, recCoeff, enableScalingLists, piDequantCoef,
                         g_stateTransTab[tu.cs->slice->getDepQuantEnabledIdc()]);
#else
    m_quant.dequantBlock(tu, compID, cQP, recCoeff, enableScalingLists, piDequantCoef);
#endif
  }

#define DINIT(l, p)                                                                                                    \
  {                                                                                                                    \
    std::numeric_limits<int64_t>::max() >> 2, l, p                                                                     \
  }
#if TCQ_8STATES
  // A trellis stage holds one decision per state. A decision for a skipped sub-block refers to state
  // numStates + k; the backtracking resolves it directly to state k at the end of the next sub-block
  // (the original stored a chain of (0, numStates + k) entries for that, i.e. twice the decisions per stage).
  static const Decision startDec = DINIT(-1, -2);
#else
  static const Decision startDec[8] = { DINIT(-1, -2), DINIT(-1, -2), DINIT(-1, -2), DINIT(-1, -2),
                                        DINIT(0, 4),   DINIT(0, 5),   DINIT(0, 6),   DINIT(0, 7) };
#endif
#undef DINIT

#if TCQ_8STATES
  template<int numStates> ALWAYS_INLINE
#endif
#if TU_256 && !EXTENDED_LFNST
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
  void DepQuant::xDecide(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions,
                         TCoeff quanCoeff)
#else
  void DepQuant::xDecide(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions,
                         int quanCoeff)
#endif
#else
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
  void DepQuant::xDecide(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions,
                         bool zeroOut, TCoeff quanCoeff)
#else
  void DepQuant::xDecide(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset, Decision *decisions,
                         bool zeroOut, int quanCoeff)
#endif
#endif
  {
#if TCQ_8STATES
    // For destination state j the candidates are: level A and zero from state P[j] and level B from state Q[j]
    // (next state of P[j] for an even level / of Q[j] for an odd level), in the stock evaluation order (by source state),
    // then skipping the sub-block (from skip state j) and starting the path (j = 0: pqData[0], j = START_B: pqData[2]).
    // The selection is written branch-free; ties keep the earlier candidate as before (strict "<").
    static const int8_t P[2][8]    = { { 0, 2, 1, 3, -1, -1, -1, -1 }, { 0, 2, 4, 6, 5, 1, 3, 7 } };
    static const int8_t Q[2][8]    = { { 2, 0, 3, 1, -1, -1, -1, -1 }, { 4, 6, 0, 2, 3, 7, 5, 1 } };
    const int           START_B    = (numStates > 4 ? 2 : 1);
    const int8_t       *P_         = P[numStates > 4];
    const int8_t       *Q_         = Q[numStates > 4];
#endif
#if !TU_256 || EXTENDED_LFNST
#if TCQ_8STATES
    for (int k = 0; k < numStates; k++)
    {
      decisions[k] = startDec;
    }
#else
    ::memcpy(decisions, startDec, 8 * sizeof(Decision));
#endif
    if (zeroOut)
    {
      if (spt == SCAN_EOCSBB)
      {
#if TCQ_8STATES
        m_skipStates[0].checkRdCostSkipSbbZeroOut(decisions[0], numStates);
        m_skipStates[1].checkRdCostSkipSbbZeroOut(decisions[1], numStates);
        m_skipStates[2].checkRdCostSkipSbbZeroOut(decisions[2], numStates);
        m_skipStates[3].checkRdCostSkipSbbZeroOut(decisions[3], numStates);
        m_skipStates[4].checkRdCostSkipSbbZeroOut(decisions[4], numStates);
        m_skipStates[5].checkRdCostSkipSbbZeroOut(decisions[5], numStates);
        m_skipStates[6].checkRdCostSkipSbbZeroOut(decisions[6], numStates);
        m_skipStates[7].checkRdCostSkipSbbZeroOut(decisions[7], numStates);
#else
        m_skipStates[0].checkRdCostSkipSbbZeroOut(decisions[0]);
        m_skipStates[1].checkRdCostSkipSbbZeroOut(decisions[1]);
        m_skipStates[2].checkRdCostSkipSbbZeroOut(decisions[2]);
        m_skipStates[3].checkRdCostSkipSbbZeroOut(decisions[3]);
#endif
      }
      return;
    }
#endif

    PQData pqData[4];
    m_quant.preQuantCoeff(absCoeff, pqData, quanCoeff);
#if TCQ_8STATES
    int64_t costA[numStates], costB[numStates], costZ[numStates];
    DQ_UNROLL_STATES
    for (int s = 0; s < numStates; s++)
    {
      m_prevStates[s].calcRdCosts(spt, pqData[(s & 1) ? 3 : 0], pqData[(s & 1) ? 1 : 2], costA[s], costB[s], costZ[s]);
    }
    const int64_t startCostA = m_startState.calcRdCostStart(lastOffset, pqData[0]);
    const int64_t startCostB = m_startState.calcRdCostStart(lastOffset, pqData[2]);

    DQ_UNROLL_STATES
    for (int j = 0; j < numStates; j++)
    {
      const int p = P_[j];
      const int q = Q_[j];
      int64_t   cost     = startDec.rdCost;
      TCoeff    absLevel = startDec.absLevel;
      int       prevId   = startDec.prevId;
      auto      select   = [&](const int64_t c, const TCoeff l, const int id)
      {
        const bool better = c < cost;
        cost              = better ? c : cost;
        absLevel          = better ? l : absLevel;
        prevId            = better ? id : prevId;
      };
      if (p < q)
      {
        select(costA[p], pqData[(p & 1) ? 3 : 0].absLevel, p);
        select(costZ[p], 0, p);
        select(costB[q], pqData[(q & 1) ? 1 : 2].absLevel, q);
      }
      else
      {
        select(costB[q], pqData[(q & 1) ? 1 : 2].absLevel, q);
        select(costA[p], pqData[(p & 1) ? 3 : 0].absLevel, p);
        select(costZ[p], 0, p);
      }
      if (spt == SCAN_EOCSBB)
      {
        select(m_skipStates[j].rdCostSkipSbb(), 0, numStates + j);
      }
      if (j == 0)
      {
        select(startCostA, pqData[0].absLevel, -1);
      }
      if (j == START_B)
      {
        select(startCostB, pqData[2].absLevel, -1);
      }
      decisions[j].rdCost   = cost;
      decisions[j].absLevel = absLevel;
      decisions[j].prevId   = prevId;
    }
#else
    ::memcpy(decisions, startDec, 8 * sizeof(Decision));
    m_prevStates[0].checkRdCosts(spt, pqData[0], pqData[2], decisions[0], decisions[2]);
    m_prevStates[1].checkRdCosts(spt, pqData[0], pqData[2], decisions[2], decisions[0]);
    m_prevStates[2].checkRdCosts(spt, pqData[3], pqData[1], decisions[1], decisions[3]);
    m_prevStates[3].checkRdCosts(spt, pqData[3], pqData[1], decisions[3], decisions[1]);
    if (spt == SCAN_EOCSBB)
    {
      m_skipStates[0].checkRdCostSkipSbb(decisions[0]);
      m_skipStates[1].checkRdCostSkipSbb(decisions[1]);
      m_skipStates[2].checkRdCostSkipSbb(decisions[2]);
      m_skipStates[3].checkRdCostSkipSbb(decisions[3]);
    }
    m_startState.checkRdCostStart(lastOffset, pqData[0], decisions[0]);
    m_startState.checkRdCostStart(lastOffset, pqData[2], decisions[2]);
#endif
  }

#if TCQ_8STATES
  template<int numStates> ALWAYS_INLINE
#endif
#if TU_256 && !EXTENDED_LFNST
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
#if JVET_AE0102_LFNST_CTX
  void DepQuant::xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, TCoeff quantCoeff, int lfnstIdx)
#else
  void DepQuant::xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, TCoeff quantCoeff)
#endif
#else
  void DepQuant::xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, int quantCoeff)
#endif
#else
#if JVET_R0351_HIGH_BIT_DEPTH_SUPPORT_VS
  void DepQuant::xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, bool zeroOut, TCoeff quantCoeff)
#else
  void DepQuant::xDecideAndUpdate(const TCoeff absCoeff, const ScanInfo &scanInfo, bool zeroOut, int quantCoeff)
#endif
#endif
  {
#if TCQ_8STATES
    Decision *decisions = m_trellis + scanInfo.scanIdx * numStates;
#else
    Decision *decisions = m_trellis[scanInfo.scanIdx];
#endif

    std::swap(m_prevStates, m_currStates);
#if TU_256 && !EXTENDED_LFNST
#if TCQ_8STATES
    xDecide<numStates>(scanInfo.spt, absCoeff, lastOffset(scanInfo.scanIdx), decisions, quantCoeff);
#else
    xDecide(scanInfo.spt, absCoeff, lastOffset(scanInfo.scanIdx), decisions, quantCoeff);
#endif
#else
#if TCQ_8STATES
    xDecide<numStates>(scanInfo.spt, absCoeff, lastOffset(scanInfo.scanIdx), decisions, zeroOut, quantCoeff);
#else
    xDecide(scanInfo.spt, absCoeff, lastOffset(scanInfo.scanIdx), decisions, zeroOut, quantCoeff);
#endif
#endif
    if (scanInfo.scanIdx)
    {
      if (scanInfo.eosbb)
      {
        m_commonCtx.swap();
#if TCQ_8STATES
        // PN add propagation of lfnstidx
#if JVET_AE0102_LFNST_CTX
        m_currStates[0].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[0], numStates, lfnstIdx > 0);
        m_currStates[1].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[1], numStates, lfnstIdx > 0);
        m_currStates[2].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[2], numStates, lfnstIdx > 0);
        m_currStates[3].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[3], numStates, lfnstIdx > 0);

        if (numStates > 4)
        {
          m_currStates[4].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[4], numStates, lfnstIdx > 0);
          m_currStates[5].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[5], numStates, lfnstIdx > 0);
          m_currStates[6].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[6], numStates, lfnstIdx > 0);
          m_currStates[7].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[7], numStates, lfnstIdx > 0);
        }
#else
        m_currStates[0].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[0], numStates);
        m_currStates[1].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[1], numStates);
        m_currStates[2].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[2], numStates);
        m_currStates[3].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[3], numStates);

        if (numStates > 4)
        {
          m_currStates[4].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[4], numStates);
          m_currStates[5].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[5], numStates);
          m_currStates[6].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[6], numStates);
          m_currStates[7].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[7], numStates);
        }
#endif
#else
        m_currStates[0].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[0]);
        m_currStates[1].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[1]);
        m_currStates[2].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[2]);
        m_currStates[3].updateStateEOS(scanInfo, m_prevStates, m_skipStates, decisions[3]);
        ::memcpy(decisions + 4, decisions, 4 * sizeof(Decision));
#endif
      }
#if TU_256 && !EXTENDED_LFNST
      else
#else
      else if (!zeroOut)
#endif
      {
#if TCQ_8STATES
        for (int k = 0; k < numStates; k++)
#else
        for (int k = 0; k < 4; k++)
#endif
        {
#if JVET_AE0102_LFNST_CTX
          m_currStates[k].updateState(scanInfo, m_prevStates, decisions[k], lfnstIdx > 0 ? 1 : 0);
#else
          m_currStates[k].updateState(scanInfo, m_prevStates, decisions[k]);
#endif
        }
      }

      if (scanInfo.spt == SCAN_SOCSBB)
      {
        std::swap(m_prevStates, m_skipStates);
      }
    }
  }

#if DQ_TRELLIS_LOOP_TEMPLATE
  void DepQuant::xInitStates(const unsigned effWidth, const unsigned effHeight, const ChannelType chType)
  {
    // same values as State::init for all states of the three arrays (which keeps m_sbbFracBits)
    for (int a = 0; a < 3; a++)
    {
      StateArr &st = m_stateArr[a];
      for (int k = 0; k < 8; k++)
      {
        st.rdCost[k]      = std::numeric_limits<int64_t>::max() >> 1;
        st.remRegBins[k]  = 4;   // just large enough for last scan pos
        st.numSigSbb[k]   = 0;
        st.refSbbCtxId[k] = -1;
        st.ctxInitIdx[k]  = CTX_INIT_ZERO;
        st.goRicePar[k]   = 0;
        st.goRiceZero[k]  = 0;
        st.sigCtx[k]      = 0;
        st.gtxCtx[k]      = 0;
      }
      ::memset(st.absLevels, 0, sizeof(st.absLevels));
    }
    const int ctxBinSampleRatio = (chType == CHANNEL_TYPE_LUMA) ? MAX_TU_LEVEL_CTX_CODED_BIN_CONSTRAINT_LUMA
                                                                : MAX_TU_LEVEL_CTX_CODED_BIN_CONSTRAINT_CHROMA;
    m_initRemRegBins = (effWidth * effHeight * ctxBinSampleRatio) / 16;
  }

  // State::calcRdCosts for state s of a StateArr
  inline void DepQuant::calcRdCostsSoA(const StateArr &st, const int s, const ScanPosType spt, const PQData &pqDataA,
                                       const PQData &pqDataB, int64_t &costA, int64_t &costB, int64_t &costZ) const
  {
    const int32_t *goRiceTab = g_goRiceBits[st.goRicePar[s]];
    int64_t        rdCostA   = st.rdCost[s] + pqDataA.deltaDist;
    int64_t        rdCostB   = st.rdCost[s] + pqDataB.deltaDist;
    int64_t        rdCostZ   = st.rdCost[s];
    const TCoeff   absLevelA = pqDataA.absLevel;
    const TCoeff   absLevelB = pqDataB.absLevel;

    if (st.remRegBins[s] >= 4)
    {
      const int32_t *bits = gtxFracBits(0)[st.gtxCtx[s]].bits;
      if (absLevelA < GTN_LEVEL)
      {
        rdCostA += bits[absLevelA];
      }
      else
      {
        const TCoeff value = (absLevelA - GTN_LEVEL) >> 1;
        rdCostA += bits[absLevelA - (value << 1)] + goRiceTab[value < RICEMAX ? value : RICEMAX - 1];
      }
      if (absLevelB < GTN_LEVEL)
      {
        rdCostB += bits[absLevelB];
      }
      else
      {
        const TCoeff value = (absLevelB - GTN_LEVEL) >> 1;
        rdCostB += bits[absLevelB - (value << 1)] + goRiceTab[value < RICEMAX ? value : RICEMAX - 1];
      }
      const BinFracBits &sigBits = m_sigTab[s][st.sigCtx[s]];
      const int          sigBit1 = sigBits.intBits[1];
      if (spt == SCAN_ISCSBB)
      {
        rdCostA += sigBit1;
        rdCostB += sigBit1;
        rdCostZ += sigBits.intBits[0];
      }
      else if (spt == SCAN_SOCSBB)
      {
        const int sbbBit1 = st.sbbBits1[s];
        rdCostA += sbbBit1 + sigBit1;
        rdCostB += sbbBit1 + sigBit1;
        rdCostZ += sbbBit1 + sigBits.intBits[0];
      }
      else if (st.numSigSbb[s])
      {
        rdCostA += sigBit1;
        rdCostB += sigBit1;
        rdCostZ += sigBits.intBits[0];
      }
      else
      {
        rdCostZ = std::numeric_limits<int64_t>::max();   // last coefficient of a coded sub-block: zero not allowed
      }
    }
    else
    {
      const int goRiceZero = st.goRiceZero[s];
      rdCostA += (1 << SCALE_BITS)
                 + goRiceTab[absLevelA <= goRiceZero ? absLevelA - 1 : (absLevelA < RICEMAX ? absLevelA : RICEMAX - 1)];
      rdCostB += (1 << SCALE_BITS)
                 + goRiceTab[absLevelB <= goRiceZero ? absLevelB - 1 : (absLevelB < RICEMAX ? absLevelB : RICEMAX - 1)];
      rdCostZ += goRiceTab[goRiceZero];
    }
    costA = rdCostA;
    costB = rdCostB;
    costZ = rdCostZ;
  }

  // State::calcRdCostStart of the start state (gtx context 0, Rice parameter 0)
  inline int64_t DepQuant::calcRdCostStartSoA(const int32_t lastOffset, const PQData &pqData) const
  {
    const int32_t *bits   = gtxFracBits(0)[0].bits;
    int64_t        rdCost = pqData.deltaDist + lastOffset;
    if (pqData.absLevel < GTN_LEVEL)
    {
      rdCost += bits[pqData.absLevel];
    }
    else
    {
      const TCoeff value = (pqData.absLevel - GTN_LEVEL) >> 1;
      rdCost += bits[pqData.absLevel - (value << 1)] + g_goRiceBits[0][value < RICEMAX ? value : RICEMAX - 1];
    }
    return rdCost;
  }

  template<int numStates>
  ALWAYS_INLINE void DepQuant::xDecideSoA(const ScanPosType spt, const TCoeff absCoeff, const int lastOffset,
                                          Decision *decisions, const TCoeff quanCoeff)
  {
    // For destination state j the candidates are: level A and zero from state P[j] and level B from state Q[j]
    // (next state of P[j] for an even level / of Q[j] for an odd level), in the stock evaluation order (by source
    // state), then skipping the sub-block (from skip state j) and starting the path (j = 0: pqData[0], j = START_B:
    // pqData[2]). The selection is branch-free; ties keep the earlier candidate as before (strict "<").
    static const int8_t P[2][8] = { { 0, 2, 1, 3, -1, -1, -1, -1 }, { 0, 2, 4, 6, 5, 1, 3, 7 } };
    static const int8_t Q[2][8] = { { 2, 0, 3, 1, -1, -1, -1, -1 }, { 4, 6, 0, 2, 3, 7, 5, 1 } };
    const int           START_B = (numStates > 4 ? 2 : 1);
    const int8_t *      P_      = P[numStates > 4];
    const int8_t *      Q_      = Q[numStates > 4];

    PQData pqData[4];
    m_quant.preQuantCoeff(absCoeff, pqData, quanCoeff);

    const StateArr &prv = *m_prevArr;
    const StateArr &skp = *m_skipArr;
    StateArr &      cur = *m_currArr;
    int64_t         costA[numStates], costB[numStates], costZ[numStates];
    DQ_UNROLL_STATES
    for (int s = 0; s < numStates; s++)
    {
      calcRdCostsSoA(prv, s, spt, pqData[(s & 1) ? 3 : 0], pqData[(s & 1) ? 1 : 2], costA[s], costB[s], costZ[s]);
    }
    const int64_t startCostA = calcRdCostStartSoA(lastOffset, pqData[0]);
    const int64_t startCostB = calcRdCostStartSoA(lastOffset, pqData[2]);

    DQ_UNROLL_STATES
    for (int j = 0; j < numStates; j++)
    {
      const int p        = P_[j];
      const int q        = Q_[j];
      int64_t   cost     = startDec.rdCost;
      TCoeff    absLevel = startDec.absLevel;
      int       prevId   = startDec.prevId;
      auto      select   = [&](const int64_t c, const TCoeff l, const int id)
      {
        const bool better = c < cost;
        cost              = better ? c : cost;
        absLevel          = better ? l : absLevel;
        prevId            = better ? id : prevId;
      };
      if (p < q)
      {
        select(costA[p], pqData[(p & 1) ? 3 : 0].absLevel, p);
        select(costZ[p], 0, p);
        select(costB[q], pqData[(q & 1) ? 1 : 2].absLevel, q);
      }
      else
      {
        select(costB[q], pqData[(q & 1) ? 1 : 2].absLevel, q);
        select(costA[p], pqData[(p & 1) ? 3 : 0].absLevel, p);
        select(costZ[p], 0, p);
      }
      if (spt == SCAN_EOCSBB)
      {
        select(skp.rdCost[j] + skp.sbbBits0[j], 0, numStates + j);
      }
      if (j == 0)
      {
        select(startCostA, pqData[0].absLevel, -1);
      }
      if (j == START_B)
      {
        select(startCostB, pqData[2].absLevel, -1);
      }
      // one 16-byte store per decision (the state update reads the decisions as vectors)
      _mm_storeu_si128((__m128i *) &decisions[j],
                       _mm_set_epi64x((int64_t) (((uint64_t) (uint32_t) prevId << 32) | (uint32_t) absLevel), cost));
      cur.rdCost[j] = cost;
    }
  }

  // State::updateState for all states (stage inside a sub-block)
  template<int numStates>
  ALWAYS_INLINE void DepQuant::xUpdateStatesSoA(const ScanInfo &scanInfo, const Decision *decisions, const int ctxType)
  {
    StateArr &      cur  = *m_currArr;
    const StateArr &prv  = *m_prevArr;
    const __m128i   zero = _mm_setzero_si128();

    //----- decisions: absolute levels (int32) and predecessors (-2: none, -1: start of the path) -----
    __m128i abs03, abs47, prv03, prv47;
    {
      const __m128i h01 = _mm_unpackhi_epi64(_mm_loadu_si128((const __m128i *) &decisions[0]),
                                             _mm_loadu_si128((const __m128i *) &decisions[1]));
      const __m128i h23 = _mm_unpackhi_epi64(_mm_loadu_si128((const __m128i *) &decisions[2]),
                                             _mm_loadu_si128((const __m128i *) &decisions[3]));
      const __m128i s01 = _mm_shuffle_epi32(h01, _MM_SHUFFLE(3, 1, 2, 0));   // a0 a1 p0 p1
      const __m128i s23 = _mm_shuffle_epi32(h23, _MM_SHUFFLE(3, 1, 2, 0));   // a2 a3 p2 p3
      abs03             = _mm_unpacklo_epi64(s01, s23);
      prv03             = _mm_unpackhi_epi64(s01, s23);
      if (numStates > 4)
      {
        const __m128i h45 = _mm_unpackhi_epi64(_mm_loadu_si128((const __m128i *) &decisions[4]),
                                               _mm_loadu_si128((const __m128i *) &decisions[5]));
        const __m128i h67 = _mm_unpackhi_epi64(_mm_loadu_si128((const __m128i *) &decisions[6]),
                                               _mm_loadu_si128((const __m128i *) &decisions[7]));
        const __m128i s45 = _mm_shuffle_epi32(h45, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i s67 = _mm_shuffle_epi32(h67, _MM_SHUFFLE(3, 1, 2, 0));
        abs47             = _mm_unpacklo_epi64(s45, s67);
        prv47             = _mm_unpackhi_epi64(s45, s67);
      }
      else
      {
        abs47 = _mm_set1_epi32(-1);
        prv47 = _mm_set1_epi32(-2);   // unused lanes: like states without a path (kept as they are)
      }
    }
    const __m128i prev8 = _mm_packs_epi16(_mm_packs_epi32(prv03, prv47), zero);   // bytes 0..7
    const __m128i dead8 = _mm_cmpeq_epi8(prev8, _mm_set1_epi8(-2));
    const int     dead  = _mm_movemask_epi8(dead8) & 0xff;   // only in the first stages of a TU
    alignas(64) StateArr deadKeep;
    if (dead)
    {
      deadKeep = cur;
    }
    const __m128i start8  = _mm_cmpeq_epi8(prev8, _mm_set1_epi8(-1));
    const __m128i start03 = _mm_cmpeq_epi32(prv03, _mm_set1_epi32(-1));
    const __m128i start47 = _mm_cmpeq_epi32(prv47, _mm_set1_epi32(-1));
    // shuffle gathering two byte fields (2 x 8 bytes) of the predecessors; negative ids give 0
    const __m128i idx2 =
      _mm_unpacklo_epi64(prev8, _mm_blendv_epi8(_mm_add_epi8(prev8, _mm_set1_epi8(8)), prev8, prev8));
    const __m128i abs8 = _mm_packs_epi16(_mm_packs_epi32(abs03, abs47), zero);   // only compared to 0

    //----- numSigSbb (+ 1 for a non-zero level; start: 1), refSbbCtxId (start: -1) -----
    {
      __m128i v = _mm_shuffle_epi8(_mm_load_si128((const __m128i *) prv.numSigSbb), idx2);
      v         = _mm_add_epi8(v, _mm_add_epi8(_mm_set1_epi8(1), _mm_cmpeq_epi8(abs8, zero)));   // high half: + 0
      v         = _mm_or_si128(v, _mm_slli_si128(start8, 8));
      _mm_store_si128((__m128i *) cur.numSigSbb, v);
    }
    //----- ctxInitIdx (start: zero template init) -----
    {
      __m128i v = _mm_shuffle_epi8(_mm_load_si128((const __m128i *) prv.ctxInitIdx), idx2);
      v         = _mm_or_si128(v, _mm_and_si128(start8, _mm_set1_epi8(CTX_INIT_ZERO)));
      _mm_store_si128((__m128i *) cur.ctxInitIdx, v);
    }
    //----- int32 members of the predecessors: remRegBins, sbbBits -----
    __m128i rem03, rem47;
    {
      const __m128i byteOffs = _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
      // shuffles reading lane prevId of an int32 array held in two registers (lanes 0-3, lanes 4-7)
      auto gatherIdx = [&](const __m128i spread, __m128i &i0, __m128i &i1)
      {
        __m128i i = _mm_shuffle_epi8(prev8, spread);   // predecessor id in each byte of an int32 lane
        i         = _mm_add_epi8(i, i);
        i         = _mm_add_epi8(_mm_add_epi8(i, i), byteOffs);   // 4 * id + byte (negative ids stay negative)
        i0 = _mm_or_si128(i, _mm_and_si128(_mm_cmpgt_epi8(i, _mm_set1_epi8(15)), _mm_set1_epi8((char) 0x80)));
        i1 = _mm_sub_epi8(i, _mm_set1_epi8(16));
      };
      __m128i iLo0, iLo1, iHi0, iHi1;
      gatherIdx(_mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3), iLo0, iLo1);
      gatherIdx(_mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7), iHi0, iHi1);
      auto gather32 = [&](const void *src, __m128i &lo, __m128i &hi)
      {
        const __m128i r0 = _mm_load_si128((const __m128i *) src);
        const __m128i r1 = _mm_load_si128((const __m128i *) src + 1);
        lo = _mm_or_si128(_mm_shuffle_epi8(r0, iLo0), _mm_shuffle_epi8(r1, iLo1));
        hi = _mm_or_si128(_mm_shuffle_epi8(r0, iHi0), _mm_shuffle_epi8(r1, iHi1));
      };
      // remRegBins: predecessor - 1 - (>= 4 ? min(absLevel, 7) : 0); start: init - min(absLevel, 7)
      gather32(prv.remRegBins, rem03, rem47);
      const __m128i initRem1 = _mm_set1_epi32(m_initRemRegBins + 1);
      const __m128i three    = _mm_set1_epi32(3);
      const __m128i maxDec   = _mm_set1_epi32(GTN_LEVEL - 1);
      rem03 = _mm_sub_epi32(_mm_blendv_epi8(rem03, initRem1, start03), _mm_set1_epi32(1));
      rem47 = _mm_sub_epi32(_mm_blendv_epi8(rem47, initRem1, start47), _mm_set1_epi32(1));
      rem03 = _mm_sub_epi32(rem03, _mm_and_si128(_mm_or_si128(_mm_cmpgt_epi32(rem03, three), start03),
                                                 _mm_min_epi32(abs03, maxDec)));
      rem47 = _mm_sub_epi32(rem47, _mm_and_si128(_mm_or_si128(_mm_cmpgt_epi32(rem47, three), start47),
                                                 _mm_min_epi32(abs47, maxDec)));
      _mm_store_si128((__m128i *) &cur.remRegBins[0], rem03);
      _mm_store_si128((__m128i *) &cur.remRegBins[4], rem47);
      // sbbBits: from the predecessor; the start of a path keeps the value of the state
      const __m128i own03 = _mm_cmplt_epi32(prv03, zero);
      const __m128i own47 = _mm_cmplt_epi32(prv47, zero);
      __m128i       lo, hi;
      gather32(prv.sbbBits0, lo, hi);
      _mm_store_si128((__m128i *) &cur.sbbBits0[0],
                      _mm_blendv_epi8(lo, _mm_load_si128((const __m128i *) &cur.sbbBits0[0]), own03));
      _mm_store_si128((__m128i *) &cur.sbbBits0[4],
                      _mm_blendv_epi8(hi, _mm_load_si128((const __m128i *) &cur.sbbBits0[4]), own47));
      gather32(prv.sbbBits1, lo, hi);
      _mm_store_si128((__m128i *) &cur.sbbBits1[0],
                      _mm_blendv_epi8(lo, _mm_load_si128((const __m128i *) &cur.sbbBits1[0]), own03));
      _mm_store_si128((__m128i *) &cur.sbbBits1[4],
                      _mm_blendv_epi8(hi, _mm_load_si128((const __m128i *) &cur.sbbBits1[4]), own47));
    }
    const __m128i three = _mm_set1_epi32(3);
    const __m128i reg8  = _mm_packs_epi16(
      _mm_packs_epi32(_mm_cmpgt_epi32(rem03, three), _mm_cmpgt_epi32(rem47, three)), zero);   // ctx coded bins left

    //----- levels of the sub-block: predecessor's, with min(255, absLevel) at the current position -----
    {
      const __m128i *src = (const __m128i *) prv.absLevels;
      __m128i *      dst = (__m128i *) cur.absLevels;
      for (int r = 0; r < 8; r++)
      {
        _mm_store_si128(dst + r, _mm_shuffle_epi8(_mm_load_si128(src + r), idx2));
      }
      const __m128i lev8 = _mm_packus_epi16(_mm_packs_epi32(abs03, abs47), zero);
      const __m128i half = _mm_cmpeq_epi32(_mm_set1_epi32(scanInfo.insidePos & 1), _mm_setr_epi32(0, 0, 1, 1));
      __m128i *     row2 = dst + (scanInfo.insidePos >> 1);
      _mm_store_si128(row2, _mm_blendv_epi8(_mm_load_si128(row2), _mm_unpacklo_epi64(lev8, lev8), half));
    }

    //----- template of the next scan position -----
    const __m128i one16 = _mm_set1_epi16(1);
    __m128i       sAbs = zero, sAbs1 = zero, sNum = zero;
    {
      const uint8_t *nbRows = scanInfo.nextNbRows[ctxType];
      const __m128i  clip   = _mm_set1_epi16(TPL_LEVEL_CLIP);
      for (int k = 0; k < 5; k++)
      {
        const __m128i t = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *) cur.absLevels[nbRows[k]]));
        sAbs            = _mm_add_epi16(sAbs, t);
        sAbs1           = _mm_add_epi16(sAbs1, _mm_min_epi16(t, _mm_add_epi16(_mm_and_si128(t, one16), clip)));
        sNum            = _mm_add_epi16(sNum, _mm_min_epi16(t, one16));
      }
    }
    __m128i tinit;
    {
      const __m128i cii   = _mm_loadl_epi64((const __m128i *) cur.ctxInitIdx);
      const __m128i none  = _mm_cmpgt_epi8(cii, _mm_set1_epi8(7));   // CTX_INIT_ZERO
      const __m128i c2    = _mm_add_epi8(cii, cii);
      const __m128i ciIdx = _mm_unpacklo_epi8(_mm_or_si128(c2, none),
                                              _mm_or_si128(_mm_add_epi8(c2, _mm_set1_epi8(1)), none));
      tinit = _mm_shuffle_epi8(_mm_load_si128((const __m128i *) m_commonCtx.ctxInitRow(scanInfo.nextInsidePos)),
                               ciIdx);
    }
    const __m128i sumAbs1 = _mm_add_epi16(_mm_and_si128(_mm_srli_epi16(tinit, 3), _mm_set1_epi16(31)), sAbs1);
    const __m128i sumNum  = _mm_add_epi16(_mm_and_si128(tinit, _mm_set1_epi16(7)), sNum);
    const __m128i sumAbs  = _mm_add_epi16(_mm_srli_epi16(tinit, 8), sAbs);

    // context coded: sig / gtx contexts and Rice parameter from the template
    const __m128i sig8 = _mm_packus_epi16(
      _mm_add_epi16(_mm_set1_epi16(scanInfo.sigCtxOffsetNext),
                    _mm_min_epi16(_mm_srli_epi16(_mm_add_epi16(sumAbs1, one16), 1), _mm_set1_epi16(NSIGCTX - 1))),
      zero);
    const __m128i gtx8 = _mm_packus_epi16(
      _mm_add_epi16(_mm_set1_epi16(scanInfo.gtxCtxOffsetNext),
                    _mm_min_epi16(_mm_sub_epi16(sumAbs1, sumNum), _mm_set1_epi16(NGTXCTX - 1))),
      zero);
    const __m128i sumAll = _mm_max_epi16(
      _mm_min_epi16(_mm_sub_epi16(sumAbs, sumNum), _mm_set1_epi16(GTN_MAXSUM - 1)), zero);
    // bypass coded: Rice parameter from min(31, sumAbs), goRiceZero = (1 + (state & 1)) << parameter
    const __m128i sum31 = _mm_min_epi16(sumAbs, _mm_set1_epi16(31));
    __m128i       parReg = zero, parByp = zero;
    for (int t = 0; t < 3; t++)
    {
      parReg = _mm_sub_epi16(parReg, _mm_cmpgt_epi16(sumAll, _mm_set1_epi16(m_riceThrGtn[t] - 1)));
      parByp = _mm_sub_epi16(parByp, _mm_cmpgt_epi16(sum31, _mm_set1_epi16(m_riceThrCoeff[t] - 1)));
    }
    const __m128i parReg8  = _mm_packus_epi16(parReg, zero);
    const __m128i parByp8  = _mm_packus_epi16(parByp, zero);
    const __m128i zeroByp8 = _mm_shuffle_epi8(_mm_setr_epi8(1, 2, 4, 8, 2, 4, 8, 16, 0, 0, 0, 0, 0, 0, 0, 0),
                                              _mm_add_epi8(parByp8, _mm_setr_epi8(0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 0,
                                                                                  0, 0, 0, 0)));
    //----- { goRicePar, goRiceZero }: goRiceZero is only set when bypass coded (as in State::updateState) -----
    {
      const __m128i old   = _mm_load_si128((const __m128i *) cur.goRicePar);
      const __m128i par8  = _mm_blendv_epi8(parByp8, parReg8, reg8);
      const __m128i zero8 = _mm_blendv_epi8(_mm_srli_si128(old, 8), zeroByp8, _mm_andnot_si128(reg8, _mm_andnot_si128(dead8, _mm_set1_epi8(-1))));
      _mm_store_si128((__m128i *) cur.goRicePar, _mm_unpacklo_epi64(par8, zero8));
    }
    //----- { sigCtx, gtxCtx }: only set when context coded -----
    {
      const __m128i old  = _mm_load_si128((const __m128i *) cur.sigCtx);
      const __m128i reg2 = _mm_unpacklo_epi64(reg8, reg8);
      _mm_store_si128((__m128i *) cur.sigCtx, _mm_blendv_epi8(old, _mm_unpacklo_epi64(sig8, gtx8), reg2));
    }

    //----- states without a path keep all members (except rdCost) -----
    if (dead)
    {
      for (int j = 0; j < 8; j++)
      {
        if ((dead >> j) & 1)
        {
          cur.remRegBins[j]  = deadKeep.remRegBins[j];
          cur.sbbBits0[j]    = deadKeep.sbbBits0[j];
          cur.sbbBits1[j]    = deadKeep.sbbBits1[j];
          cur.numSigSbb[j]   = deadKeep.numSigSbb[j];
          cur.refSbbCtxId[j] = deadKeep.refSbbCtxId[j];
          cur.ctxInitIdx[j]  = deadKeep.ctxInitIdx[j];
          cur.goRicePar[j]   = deadKeep.goRicePar[j];
          cur.goRiceZero[j]  = deadKeep.goRiceZero[j];
          cur.sigCtx[j]      = deadKeep.sigCtx[j];
          cur.gtxCtx[j]      = deadKeep.gtxCtx[j];
          for (int pos = 0; pos < 16; pos++)
          {
            cur.absLevels[pos][j] = deadKeep.absLevels[pos][j];
          }
        }
      }
    }
  }

  // State::updateStateEOS for all states (last scan position of a sub-block)
  template<int numStates>
  void DepQuant::xUpdateStatesEosSoA(const ScanInfo &scanInfo, const Decision *decisions, const int ctxType)
  {
    StateArr &      cur = *m_currArr;
    const StateArr &prv = *m_prevArr;
    const StateArr &skp = *m_skipArr;
    alignas(16) uint8_t prvLevels[8][16];
    transposeAbsLevels(prv.absLevels, prvLevels);
    for (int j = 0; j < numStates; j++)
    {
      const Decision &decision = decisions[j];
      if (decision.prevId > -2)
      {
        int            prevRef   = -1;
        int            prevRem   = m_initRemRegBins;
        int            numSig    = 1;
        const uint8_t *srcLevels = g_zeroAbsLevels;
        if (decision.prevId >= numStates)
        {
          CHECKD(decision.absLevel != 0, "cannot happen");
          const int k = decision.prevId - numStates;
          prevRef     = skp.refSbbCtxId[k];
          prevRem     = skp.remRegBins[k];
          numSig      = 0;
        }
        else if (decision.prevId >= 0)
        {
          const int p = decision.prevId;
          prevRef     = prv.refSbbCtxId[p];
          prevRem     = prv.remRegBins[p];
          numSig      = prv.numSigSbb[p] + !!decision.absLevel;
          srcLevels   = prvLevels[p];
        }
        alignas(16) uint8_t levels[16];
        setAbsLevels(levels, srcLevels, scanInfo.insidePos, (uint8_t) std::min<TCoeff>(255, decision.absLevel));
        const int sigNSbb = m_commonCtx.updateSbb(scanInfo, prevRef, j, numSig != 0, levels, ctxType > 0);

        cur.numSigSbb[j]   = 0;
        cur.remRegBins[j]  = prevRem;
        cur.goRicePar[j]   = 0;
        cur.refSbbCtxId[j] = j;
        cur.sbbBits0[j]    = m_commonCtx.sbbFlagBits(sigNSbb).intBits[0];
        cur.sbbBits1[j]    = m_commonCtx.sbbFlagBits(sigNSbb).intBits[1];
        cur.ctxInitIdx[j]  = j;
        const int tinit    = m_commonCtx.ctxInit(j, scanInfo.nextInsidePos);
        const int sumNum   = tinit & 7;
        const int sumAbs1  = (tinit >> 3) & 31;
        const int sumGt1   = sumAbs1 - sumNum;
        cur.sigCtx[j]      = scanInfo.sigCtxOffsetNext + std::min((sumAbs1 + 1) >> 1, NSIGCTX - 1);
        cur.gtxCtx[j]      = scanInfo.gtxCtxOffsetNext + std::min(sumGt1, NGTXCTX - 1);
      }
    }
    // the levels of the new states are zero; states without a path keep theirs
    alignas(8) uint8_t keep[8];
    for (int j = 0; j < 8; j++)
    {
      keep[j] = (j >= numStates || decisions[j].prevId == -2 ? 0xff : 0);
    }
    const __m128i keep8 = _mm_loadl_epi64((const __m128i *) keep);
    const __m128i keep2 = _mm_unpacklo_epi64(keep8, keep8);
    __m128i      *rows  = (__m128i *) cur.absLevels;
    for (int r = 0; r < 8; r++)
    {
      _mm_store_si128(rows + r, _mm_and_si128(_mm_load_si128(rows + r), keep2));
    }
  }

  template<int numStates>
  void DepQuant::xQuantTrellis(const TUParameters &tuPars, const TCoeff *tCoeff, const int firstTestPos,
                               const int lfnstIdx, const TCoeff defaultQuantisationCoefficient,
                               const bool enableScalingLists, const int *quantCoeff, const TransformUnit &tu,
                               const ComponentID compID, const QpParam &cQP, const double lambda)
  {
    const int ctxType = lfnstIdx > 0 ? 1 : 0;
    for (int scanIdx = firstTestPos; scanIdx >= 0; scanIdx--)
    {
      const ScanInfo &scanInfo  = tuPars.m_scanInfo[scanIdx];
      TCoeff          quanCoeff = defaultQuantisationCoefficient;
      if (enableScalingLists)
      {
        m_quant.initQuantBlock(tu, compID, cQP, lambda, quantCoeff[scanInfo.rasterPos]);
        quanCoeff = quantCoeff[scanInfo.rasterPos];
      }
      Decision *decisions = m_trellis + scanIdx * numStates;
      std::swap(m_prevArr, m_currArr);
      xDecideSoA<numStates>(scanInfo.spt, abs(tCoeff[scanInfo.rasterPos]), lastOffset(scanIdx), decisions, quanCoeff);
      if (scanIdx)
      {
        if (scanInfo.eosbb)
        {
          m_commonCtx.swap();
          xUpdateStatesEosSoA<numStates>(scanInfo, decisions, ctxType);
        }
        else
        {
          xUpdateStatesSoA<numStates>(scanInfo, decisions, ctxType);
        }
        if (scanInfo.spt == SCAN_SOCSBB)
        {
          std::swap(m_prevArr, m_skipArr);
        }
      }
    }
  }
#endif

  void DepQuant::quant(TransformUnit &tu, const CCoeffBuf &srcCoeff, const ComponentID compID, const QpParam &cQP,
                       const double lambda, const Ctx &ctx, TCoeff &absSum, bool enableScalingLists, int *quantCoeff)
  {
    CHECKD(tu.cs->sps->getSpsRangeExtension().getExtendedPrecisionProcessingFlag(), "ext precision is not supported");
#if SIGN_PREDICTION
    tu.initSignBuffers(compID);
#endif

    //===== reset / pre-init =====
#if TCQ_8STATES
    const int numStates = 2 << tu.cs->slice->getDepQuantEnabledIdc();
#endif
#if JVET_AL0181_ASBT
    const TUParameters& tuPars = *g_Rom.getTUPars(tu.blocksResidual[compID], compID);
#else
    const TUParameters &tuPars = *g_Rom.getTUPars(tu.blocks[compID], compID);
#endif
    m_quant.initQuantBlock(tu, compID, cQP, lambda);
    TCoeff *      qCoeff   = tu.getCoeffs(compID).buf;
    const TCoeff *tCoeff   = srcCoeff.buf;
#if JVET_AL0181_ASBT
    const int     numCoeff = tu.blocksResidual[compID].area();
#else
    const int     numCoeff = tu.blocks[compID].area();
#endif
    ::memset(tu.getCoeffs(compID).buf, 0x00, numCoeff * sizeof(TCoeff));
    absSum = 0;

#if JVET_AL0181_ASBT
    const CompArea& area     = tu.blocksResidual[compID];
#else
    const CompArea &area     = tu.blocks[compID];
#endif
    const uint32_t  width    = area.width;
    const uint32_t  height   = area.height;
    const uint32_t  lfnstIdx = tu.cu->lfnstIdx;

    //===== scaling matrix ====
    // const int         qpDQ = cQP.Qp + 1;
    // const int         qpPer = qpDQ / 6;
    // const int         qpRem = qpDQ - 6 * qpPer;

    // TCoeff thresTmp = thres;
    bool zeroOut = false;
#if EXTENDED_LFNST || !TU_256
    int effWidth = tuPars.m_width, effHeight = tuPars.m_height;
#endif
#if !TU_256
    if ((tu.mtsIdx[compID] > MTS_SKIP
         || (tu.cs->sps->getUseMTS() && tu.cu->sbtInfo != 0 && tuPars.m_height <= 32 && tuPars.m_width <= 32))
        && compID == COMPONENT_Y)
    {
      effHeight = (tuPars.m_height == 32) ? 16 : tuPars.m_height;
      effWidth  = (tuPars.m_width == 32) ? 16 : tuPars.m_width;
      zeroOut   = (effHeight < tuPars.m_height || effWidth < tuPars.m_width);
    }
#endif
#if EXTENDED_LFNST
    int lfnst_threshold = 0;
    if (lfnstIdx > 0 && tu.mtsIdx[compID] != MTS_SKIP && width >= 4 && height >= 4)
    {
      const bool whge3 = width >= 8 && height >= 8;
      lfnst_threshold  = whge3 ? 8 : 4;
      zeroOut          = true;
      effWidth         = lfnst_threshold;
      effHeight        = lfnst_threshold;
    }
#endif
    bool zeroOutforThres = zeroOut;
#if !TU_256
    zeroOutforThres = zeroOut || (32 < tuPars.m_height || 32 < tuPars.m_width);
#endif

    //===== find first test position =====
    int firstTestPos = numCoeff - 1;
#if !EXTENDED_LFNST
    if (lfnstIdx > 0 && tu.mtsIdx[compID] != MTS_SKIP && width >= 4 && height >= 4)
    {
#if JVET_W0119_LFNST_EXTENSION
#if JVET_AC0130_NSPT
#if JVET_AH0103_LOW_DELAY_LFNST_NSPT
      bool spsIntraLfnstEnabled = ( ( tu.cu->slice->getSliceType() == I_SLICE && tu.cu->cs->sps->getUseIntraLFNSTISlice() ) ||
                                    ( tu.cu->slice->getSliceType() != I_SLICE && tu.cu->cs->sps->getUseIntraLFNSTPBSlice() ) );
      bool allowNSPT = CU::isNSPTAllowed( tu, compID, width, height, spsIntraLfnstEnabled && CU::isIntra( *( tu.cu ) ) );
#else
      bool allowNSPT = CU::isNSPTAllowed(tu, compID, width, height, CU::isIntra(*(tu.cu)));
#endif

      if (allowNSPT)
      {
        firstTestPos = PU::getNSPTMatrixDim(width, height) - 1;
      }
      else
      {
#if AHG7_LN_TOOLOFF_CFG
        firstTestPos = PU::getLFNSTMatrixDim( width, height, tu.cu->cs->sps->getUseLFNSTExt() ) - 1;
#else
        firstTestPos = PU::getLFNSTMatrixDim(width, height) - 1;
#endif
      }
#else
#if AHG7_LN_TOOLOFF_CFG
      firstTestPos = PU::getLFNSTMatrixDim( width, height, tu.cu->cs->sps->getUseLFNSTExt() ) - 1;
#else
      firstTestPos = PU::getLFNSTMatrixDim(width, height) - 1;
#endif
#endif
#else
#if JVET_AC0130_NSPT
#if JVET_AH0103_LOW_DELAY_LFNST_NSPT
      bool spsIntraLfnstEnabled = ( ( tu.cu->slice->getSliceType() == I_SLICE && tu.cu->cs->sps->getUseIntraLFNSTISlice() ) ||
                                    ( tu.cu->slice->getSliceType() != I_SLICE && tu.cu->cs->sps->getUseIntraLFNSTPBSlice() ) );
      bool allowNSPT = CU::isNSPTAllowed( tu, compID, width, height, spsIntraLfnstEnabled && CU::isIntra( *( tu.cu ) ) );
#else
      bool allowNSPT = CU::isNSPTAllowed(tu, compID, width, height, CU::isIntra(*(tu.cu)));
#endif

      if (allowNSPT)
      {
        firstTestPos = PU::getNSPTMatrixDim(width, height) - 1;
      }
      else
      {
        firstTestPos = ((width == 4 && height == 4) || (width == 8 && height == 8)) ? 7 : 15;
      }
#else
      firstTestPos = ((width == 4 && height == 4) || (width == 8 && height == 8)) ? 7 : 15;
#endif
#endif
    }
#endif
    const TCoeff defaultQuantisationCoefficient = (TCoeff) m_quant.getQScale();
    const TCoeff thres                          = m_quant.getLastThreshold();
    for (; firstTestPos >= 0; firstTestPos--)
    {
      if (zeroOutforThres
          && (tuPars.m_scanId2BlkPos[firstTestPos].x >= ((tuPars.m_width == 32 && zeroOut) ? 16 : 32)
              || tuPars.m_scanId2BlkPos[firstTestPos].y >= ((tuPars.m_height == 32 && zeroOut) ? 16 : 32)))
        continue;

#if EXTENDED_LFNST
      if (lfnst_threshold > 0
          && (tuPars.m_scanId2BlkPos[firstTestPos].x >= lfnst_threshold
              || tuPars.m_scanId2BlkPos[firstTestPos].y >= lfnst_threshold))
      {
        continue;
      }
#endif
      TCoeff thresTmp = (enableScalingLists)
                          ? TCoeff(thres / (4 * quantCoeff[tuPars.m_scanId2BlkPos[firstTestPos].idx]))
                          : TCoeff(thres / (4 * defaultQuantisationCoefficient));

      if (abs(tCoeff[tuPars.m_scanId2BlkPos[firstTestPos].idx]) > thresTmp)
      {
        break;
      }
    }
    if (firstTestPos < 0)
    {
      return;
    }

    //===== real init =====
    if (ctx.getBPMType() == BPM_Std)
    {
      RateEstimator::initCtx(tuPars, tu, compID,
                             StdFracBitsAccess{ static_cast<const CtxStore<BinProbModel_Std> &>(ctx) }
#if JVET_AE0102_LFNST_CTX
                             ,
                             lfnstIdx
#endif
      );
    }
    else
    {
      RateEstimator::initCtx(tuPars, tu, compID, ctx.getFracBitsAcess()
#if JVET_AE0102_LFNST_CTX
                                                   ,
                             lfnstIdx
#endif
      );
    }

    m_commonCtx.reset(tuPars, *this);
#if !TCQ_8STATES
    for (int k = 0; k < 12; k++)
    {
      m_allStates[k].init();
    }
    m_startState.init();
#endif
#if TU_256
    int effectWidth  = width;
    int effectHeight = height;
#else
    int effectWidth = std::min(32, effWidth);
    int effectHeight = std::min(32, effHeight);
#endif
#if EXTENDED_LFNST
    effectWidth  = (lfnst_threshold > 0) ? lfnst_threshold : effectWidth;
    effectHeight = (lfnst_threshold > 0) ? lfnst_threshold : effectHeight;
#endif
#if DQ_TRELLIS_LOOP_TEMPLATE
    xInitStates(effectWidth, effectHeight, toChannelType(compID));
#elif TCQ_8STATES
    for (int k = 0; k < numStates; k++)
    {
      m_allStates[k].init(effectWidth, effectHeight);
      m_allStates[k + 8].init(effectWidth, effectHeight);
      m_allStates[k + 16].init(effectWidth, effectHeight);
    }
    m_startState.init(effectWidth, effectHeight);
#else
    for (int k = 0; k < 12; k++)
    {
      m_allStates[k].effWidth = effectWidth;
      m_allStates[k].effHeight = effectHeight;
    }
    m_startState.effWidth = effectWidth;
    m_startState.effHeight = effectHeight;
#endif
    //===== populate trellis ===== //
#if DQ_TRELLIS_LOOP_TEMPLATE
    if (numStates > 4)
    {
      xQuantTrellis<8>(tuPars, tCoeff, firstTestPos, tu.cu->lfnstIdx, defaultQuantisationCoefficient,
                       enableScalingLists, quantCoeff, tu, compID, cQP, lambda);
    }
    else
    {
      xQuantTrellis<4>(tuPars, tCoeff, firstTestPos, tu.cu->lfnstIdx, defaultQuantisationCoefficient,
                       enableScalingLists, quantCoeff, tu, compID, cQP, lambda);
    }
#else
#if TCQ_8STATES
#if TU_256 && !EXTENDED_LFNST
#if JVET_AE0102_LFNST_CTX
    void (DepQuant::*fdecide)(const TCoeff, const ScanInfo &, int, int) =
      (numStates > 4 ? &DepQuant::xDecideAndUpdate<8> : &DepQuant::xDecideAndUpdate<4>);
#else
    void (DepQuant::*fdecide)(const TCoeff, const ScanInfo &, int) =
      (numStates > 4 ? &DepQuant::xDecideAndUpdate<8> : &DepQuant::xDecideAndUpdate<4>);
#endif
#else
    void (DepQuant::*fdecide)(const TCoeff, const ScanInfo &, bool, int) =
      (numStates > 4 ? &DepQuant::xDecideAndUpdate<8> : &DepQuant::xDecideAndUpdate<4>);
#endif
#endif
    for (int scanIdx = firstTestPos; scanIdx >= 0; scanIdx--)
    {
      const ScanInfo &scanInfo = tuPars.m_scanInfo[scanIdx];
#if JVET_AE0102_LFNST_CTX
      int lfnstIdx = tu.cu->lfnstIdx;
#endif

      if (enableScalingLists)
      {
        m_quant.initQuantBlock(tu, compID, cQP, lambda, quantCoeff[scanInfo.rasterPos]);
#if TU_256 && !EXTENDED_LFNST
#if TCQ_8STATES
#if JVET_AE0102_LFNST_CTX
        (this->*fdecide)(abs(tCoeff[scanInfo.rasterPos]), scanInfo, quantCoeff[scanInfo.rasterPos], lfnstIdx);
#else
        (this->*fdecide)(abs(tCoeff[scanInfo.rasterPos]), scanInfo, quantCoeff[scanInfo.rasterPos]);
#endif
#else
        xDecideAndUpdate(abs(tCoeff[scanInfo.rasterPos]), scanInfo, quantCoeff[scanInfo.rasterPos]);
#endif
#else
#if TCQ_8STATES
        (this->*fdecide)(abs(tCoeff[scanInfo.rasterPos]), scanInfo,
                         (zeroOut && (scanInfo.posX >= effWidth || scanInfo.posY >= effHeight)),
                         quantCoeff[scanInfo.rasterPos]);
#else
        xDecideAndUpdate(abs(tCoeff[scanInfo.rasterPos]), scanInfo,
                         (zeroOut && (scanInfo.posX >= effWidth || scanInfo.posY >= effHeight)),
                         quantCoeff[scanInfo.rasterPos]);
#endif
#endif
      }
      else
#if TU_256 && !EXTENDED_LFNST
#if TCQ_8STATES
#if JVET_AE0102_LFNST_CTX
        (this->*fdecide)(abs(tCoeff[scanInfo.rasterPos]), scanInfo, defaultQuantisationCoefficient, lfnstIdx);
#else
        (this->*fdecide)(abs(tCoeff[scanInfo.rasterPos]), scanInfo, defaultQuantisationCoefficient);
#endif
#else
        xDecideAndUpdate(abs(tCoeff[scanInfo.rasterPos]), scanInfo, defaultQuantisationCoefficient);
#endif
#else
#if TCQ_8STATES
        (this->*fdecide)(abs(tCoeff[scanInfo.rasterPos]), scanInfo,
                         (zeroOut && (scanInfo.posX >= effWidth || scanInfo.posY >= effHeight)),
                         defaultQuantisationCoefficient);
#else
        xDecideAndUpdate(abs(tCoeff[scanInfo.rasterPos]), scanInfo,
                         (zeroOut && (scanInfo.posX >= effWidth || scanInfo.posY >= effHeight)),
                         defaultQuantisationCoefficient);
#endif
#endif
    }
#endif

    //===== find best path =====
    Decision decision    = { std::numeric_limits<int64_t>::max(), -1, -2 };
    int64_t  minPathCost = 0;
#if TCQ_8STATES
    for (int8_t stateId = 0; stateId < numStates; stateId++)
#else
    for (int8_t stateId = 0; stateId < 4; stateId++)
#endif
    {
#if TCQ_8STATES
      int64_t pathCost = m_trellis[stateId].rdCost;
#else
      int64_t pathCost = m_trellis[0][stateId].rdCost;
#endif
      if (pathCost < minPathCost)
      {
        decision.prevId = stateId;
        minPathCost     = pathCost;
      }
    }

    //===== backward scanning =====
    int scanIdx = 0;
#if TCQ_8STATES
    for (const Decision *trellisStage = m_trellis; decision.prevId >= 0; scanIdx++, trellisStage += numStates)
#else
    for (; decision.prevId >= 0; scanIdx++)
#endif
    {
#if TCQ_8STATES
      if (decision.prevId >= numStates)
      {
        // skipped sub-block (decided at its first scan position): its coefficients stay zero; continue with
        // state prevId - numStates at the first scan position of the next sub-block
        const int skip = tuPars.m_sbbSize - 1;
        scanIdx += skip;
        trellisStage += skip * numStates;
        decision = trellisStage[decision.prevId - numStates];
      }
      else
      {
        decision = trellisStage[decision.prevId];
      }
#else
      decision = m_trellis[scanIdx][decision.prevId];
#endif
      int32_t blkpos = tuPars.m_scanId2BlkPos[scanIdx].idx;
      qCoeff[blkpos] = (tCoeff[blkpos] < 0 ? -decision.absLevel : decision.absLevel);
      absSum += decision.absLevel;
    }
  }

};   // namespace DQIntern

//===== interface class =====
DepQuant::DepQuant(const Quant *other, bool enc) : QuantRDOQ(other)
{
  const DepQuant *dq = dynamic_cast<const DepQuant *>(other);
  CHECK(other && !dq, "The DepQuant cast must be successfull!");
  p = new DQIntern::DepQuant();
  if (enc)
  {
    DQIntern::g_Rom.init();
  }
}

DepQuant::~DepQuant()
{
  delete static_cast<DQIntern::DepQuant *>(p);
}

void DepQuant::quant(TransformUnit &tu, const ComponentID &compID, const CCoeffBuf &pSrc, TCoeff &uiAbsSum,
                     const QpParam &cQP, const Ctx &ctx)
{
  const bool useRegularResidualCoding =
    tu.cu->slice->getTSResidualCodingDisabledFlag() || tu.mtsIdx[compID] != MTS_SKIP;

#if TCQ_8STATES
  if (tu.cs->slice->getDepQuantEnabledIdc() && useRegularResidualCoding)
#else
  if (tu.cs->slice->getDepQuantEnabledFlag() && useRegularResidualCoding)
#endif
  {
    //===== scaling matrix ====
    const int       qpDQ            = cQP.Qp(tu.mtsIdx[compID] == MTS_SKIP) + 1;
    const int       qpPer           = qpDQ / 6;
    const int       qpRem           = qpDQ - 6 * qpPer;
#if JVET_AL0181_ASBT
    const CompArea    &rect         = tu.blocksResidual[compID];
#else
    const CompArea &rect            = tu.blocks[compID];
#endif
    const int       width           = rect.width;
    const int       height          = rect.height;
    uint32_t        scalingListType = getScalingListType(tu.cu->predMode, compID);
    CHECK(scalingListType >= SCALING_LIST_NUM, "Invalid scaling list");
    const uint32_t log2TrWidth  = floorLog2(width);
    const uint32_t log2TrHeight = floorLog2(height);

    const bool disableSMForLFNST = tu.cs->slice->getExplicitScalingListUsed()
                                     ? tu.cs->slice->getSPS()->getDisableScalingMatrixForLfnstBlks()
                                     : false;
#if !INTRA_RM_SMALL_BLOCK_SIZE_CONSTRAINTS
    const bool isLfnstApplied = tu.cu->lfnstIdx > 0 && (tu.cu->isSepTree() ? true : isLuma(compID));
#else
    const bool isLfnstApplied = tu.cu->lfnstIdx > 0 && (CS::isDualITree(*tu.cs) ? true : isLuma(compID));
#endif
    const bool disableSMForACT =
      tu.cs->slice->getSPS()->getScalingMatrixForAlternativeColourSpaceDisabledFlag()
      && (tu.cs->slice->getSPS()->getScalingMatrixDesignatedColourSpaceFlag() == tu.cu->colorTransform);
    const bool enableScalingLists = getUseScalingList(width, height, (tu.mtsIdx[compID] == MTS_SKIP), isLfnstApplied,
                                                      disableSMForLFNST, disableSMForACT);
    static_cast<DQIntern::DepQuant *>(p)->quant(
      tu, pSrc, compID, cQP, Quant::m_dLambda, ctx, uiAbsSum, enableScalingLists,
      Quant::getQuantCoeff(scalingListType, qpRem, log2TrWidth, log2TrHeight));
  }
  else
  {
    QuantRDOQ::quant(tu, compID, pSrc, uiAbsSum, cQP, ctx);
  }
}

#if SIGN_PREDICTION
#if JVET_Y0141_SIGN_PRED_IMPROVE
void DepQuant::getPredictedSigns(TransformUnit &tu, const ComponentID compID,
                                 static_vector<Position, SIGN_PRED_MAX_NUM> &predSignsXY, uint8_t *signBuf,
                                 bool isDecoder)
{
  if (TU::getUseSignPred(tu, compID))
  {
    CoeffBuf        bufQCoeffs = tu.getCoeffs(compID);
    TCoeff         *coeff      = bufQCoeffs.buf;
    const ptrdiff_t strideC    = bufQCoeffs.stride;

#if JVET_AL0181_ASBT
    const CompArea &area     = tu.blocksResidual[compID];
#else
    const CompArea &area     = tu.blocks[compID];
#endif
    const int       numCoeff = area.area();

    const SizeType     hsId = gp_sizeIdxInfo->idxFrom(area.width);
    const SizeType     vsId = gp_sizeIdxInfo->idxFrom(area.height);
    const ScanElement *scan = g_scanOrder[SCAN_GROUPED_4x4][SCAN_DIAG][hsId][vsId];

    // Find last nonzero coefficient
    int lastScanIdx = -1;
    for (int scanIdx = numCoeff - 1; scanIdx >= 0; scanIdx--)
    {
      if (coeff[scan[scanIdx].idx])
      {
        lastScanIdx = scanIdx;
        break;
      }
    }

    // exit if all coefficients are 0
    if (lastScanIdx < 0)
    {
      return;
    }

    AreaBuf<SIGN_PRED_TYPE> bufSigns = tu.getCoeffSigns(compID);
    const SIGN_PRED_TYPE   *signs    = bufSigns.buf;
    const ptrdiff_t strideS  = bufSigns.stride;

#if JVET_AL0181_ASBT
    const bool lfnstEnabled = ((tu.checkLFNSTApplied(compID)) || ((tu.asbtDecimH[compID] || tu.asbtDecimW[compID])
                                && !((tu.asbtDecimH[compID] + tu.asbtDecimW[compID]) == 2 && (tu.blocks[compID].width <= 8 || tu.blocks[compID].height <= 8))
                                && !(tu.blocks[compID].width <= 4) && !(tu.blocks[compID].height <= 4)));
    uint32_t minAreaSize = ((tu.asbtDecimH[compID] || tu.asbtDecimW[compID]) && compID > 0) ?
                                std::min(tu.blocks[compID].height << (tu.asbtDecimH[compID] + tu.asbtDecimW[compID]), (tu.blocks[compID].width << (tu.asbtDecimH[compID] + tu.asbtDecimW[compID]))) : 4;
                                minAreaSize = std::min( minAreaSize , (uint32_t) 4 );     
    const uint32_t extAreaSize    = lfnstEnabled ? minAreaSize : tu.cs->sps->getSignPredArea();
    const uint32_t signPredHeight = std::min(tu.blocksResidual[compID].height, extAreaSize);
    const uint32_t signPredWidth  = std::min(tu.blocksResidual[compID].width, extAreaSize);
#else
    const bool lfnstEnabled = tu.checkLFNSTApplied(compID);

    const uint32_t extAreaSize    = lfnstEnabled ? 4 : tu.cs->sps->getSignPredArea();
    const uint32_t signPredHeight = std::min(tu.blocks[compID].height, extAreaSize);
    const uint32_t signPredWidth  = std::min(tu.blocks[compID].width, extAreaSize);
#endif

    uint32_t numAreaSigns = 0;

    if (isDecoder)
    {
      for (uint32_t y = 0; y < signPredHeight; y++)
      {
        for (uint32_t x = 0; x < signPredWidth; x++)
        {
          const TCoeff coef = coeff[y * strideC + x];
          if (coef != 0)
          {
            if (signs[y * strideS + x] != SIGN_PRED_HIDDEN)
            {
              signBuf[numAreaSigns] = coef < 0 ? 1 : 0;
              numAreaSigns++;
            }
          }
        }
      }
    }

    // This should be a 4KB data structure, which isn't too large for the stack
    // 32 is the maximum value of extAreaSize below
    static_vector<PositionWithLevel, 32 * 32> predSignsXYLevel;

    const uint64_t stateTransTab = g_stateTransTab[tu.cs->slice->getDepQuantEnabledIdc()];

    int state = 0;
    for (int scanIdx = lastScanIdx; scanIdx >= 0; scanIdx--)
    {
      const auto    &scanData  = scan[scanIdx];
      const uint32_t rasterPos = scanData.idx;
      const TCoeff   level     = coeff[rasterPos];
      if (level != 0 && scanData.y < signPredHeight && scanData.x < signPredWidth)
      {
        if (signs[rasterPos] != SIGN_PRED_HIDDEN)
        {
          Intermediate_Int qIdx = 2 * std::abs(level) - (state & 1);
          predSignsXYLevel.push_back({ scanData.x, scanData.y, qIdx });
        }
      }
      state = int((stateTransTab >> (4 * (2 * state + (level & 1)))) & 15);
    }
    stableSortSmall(predSignsXYLevel.begin(), predSignsXYLevel.end(),
                    [](const PositionWithLevel &a, const PositionWithLevel &b) { return a.level > b.level; });

    IdxBuf bufSignsScanIdx = tu.getCoeffSignsScanIdx(compID);

    if (isDecoder)
    {
      CHECK(numAreaSigns != predSignsXYLevel.size(), "sign prediction number error");
    }

#if JVET_AL0181_ASBT
    const int32_t maxNumPredSigns =
      lfnstEnabled ? std::min<int>(minAreaSize, tu.cs->sps->getNumPredSigns()) : tu.cs->sps->getNumPredSigns();
#else
    const int32_t maxNumPredSigns =
      lfnstEnabled ? std::min<int>(4, tu.cs->sps->getNumPredSigns()) : tu.cs->sps->getNumPredSigns();
#endif

    for (int idx = 0; idx < predSignsXYLevel.size(); idx++)
    {
      const PositionWithLevel &pos = predSignsXYLevel[idx];

      bufSignsScanIdx.at(pos.x, pos.y) = idx;

      if (isDecoder)
      {
        TCoeff &quantCoeff = bufQCoeffs.at(pos.x, pos.y);
        quantCoeff = std::abs(quantCoeff) * (signBuf[idx] ? -1 : 1);
      }

      if (idx < maxNumPredSigns)
      {
        predSignsXY.push_back({ pos.x, pos.y });
      }
    }
    CHECK(predSignsXY.size() > maxNumPredSigns, "");
  }
}
#else
void DepQuant::getPredictedSigns(TransformUnit &tu, const ComponentID compID,
                                 static_vector<Position, SIGN_PRED_MAX_NUM> &predSignsXY)
{
  if (TU::getUseSignPred(tu, compID))
  {
    const int32_t maxNumPredSigns = tu.cs->sps->getNumPredSigns();

    CoeffBuf bufQCoeffs = tu.getCoeffs( compID );
    AreaBuf<SIGN_PRED_TYPE> bufSigns = tu.getCoeffSigns(compID);
    const TCoeff *coeff = bufQCoeffs.buf;
    const TCoeff *signs = bufSigns.buf;
    const ptrdiff_t strideC = bufQCoeffs.stride;
    const ptrdiff_t strideS = bufSigns.stride;

    for (uint32_t y = 0; y < SIGN_PRED_FREQ_RANGE; y++)
    {
      for (uint32_t x = 0; x < SIGN_PRED_FREQ_RANGE; x++)
      {
        if (coeff[y * strideC + x] != 0)
        {
          if (signs[y * strideS + x] != SIGN_PRED_HIDDEN)
          {
            predSignsXY.push_back({ x, y });
            if (predSignsXY.size() >= maxNumPredSigns)
            {
              return;
            }
          }
        }
      }
    }
  }
}
#endif
#endif

void DepQuant::dequant(const TransformUnit &tu, CoeffBuf &dstCoeff, const ComponentID &compID, const QpParam &cQP)
{
  const bool useRegularResidualCoding =
    tu.cu->slice->getTSResidualCodingDisabledFlag() || tu.mtsIdx[compID] != MTS_SKIP;

#if TCQ_8STATES
  if (tu.cs->slice->getDepQuantEnabledIdc() && useRegularResidualCoding)
#else
  if (tu.cs->slice->getDepQuantEnabledFlag() && useRegularResidualCoding)
#endif
  {
    const int       qpDQ            = cQP.Qp(tu.mtsIdx[compID] == MTS_SKIP) + 1;
    const int       qpPer           = qpDQ / 6;
    const int       qpRem           = qpDQ - 6 * qpPer;
#if JVET_AL0181_ASBT
    const CompArea    &rect         = tu.blocksResidual[compID];
#else
    const CompArea &rect            = tu.blocks[compID];
#endif
    const int       width           = rect.width;
    const int       height          = rect.height;
    uint32_t        scalingListType = getScalingListType(tu.cu->predMode, compID);
    CHECK(scalingListType >= SCALING_LIST_NUM, "Invalid scaling list");
    const uint32_t log2TrWidth  = floorLog2(width);
    const uint32_t log2TrHeight = floorLog2(height);

    const bool disableSMForLFNST = tu.cs->slice->getExplicitScalingListUsed()
                                     ? tu.cs->slice->getSPS()->getDisableScalingMatrixForLfnstBlks()
                                     : false;
#if !INTRA_RM_SMALL_BLOCK_SIZE_CONSTRAINTS
    const bool isLfnstApplied = tu.cu->lfnstIdx > 0 && (tu.cu->isSepTree() ? true : isLuma(compID));
#else
#if JVET_AI0136_ADAPTIVE_DUAL_TREE
    const bool isLfnstApplied = tu.cu->lfnstIdx > 0 && (tu.cu->separateTree ? true : isLuma(compID));
#else
    const bool isLfnstApplied = tu.cu->lfnstIdx > 0 && (CS::isDualITree(*tu.cs) ? true : isLuma(compID));
#endif
#endif
    const bool disableSMForACT =
      tu.cs->slice->getSPS()->getScalingMatrixForAlternativeColourSpaceDisabledFlag()
      && (tu.cs->slice->getSPS()->getScalingMatrixDesignatedColourSpaceFlag() == tu.cu->colorTransform);
    const bool enableScalingLists = getUseScalingList(width, height, (tu.mtsIdx[compID] == MTS_SKIP), isLfnstApplied,
                                                      disableSMForLFNST, disableSMForACT);
    static_cast<DQIntern::DepQuant *>(p)->dequant(
      tu, dstCoeff, compID, cQP, enableScalingLists,
      Quant::getDequantCoeff(scalingListType, qpRem, log2TrWidth, log2TrHeight));
  }
  else
  {
    QuantRDOQ::dequant(tu, dstCoeff, compID, cQP);
  }
}
