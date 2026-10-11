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

/** \file     LoopFilter.h
    \brief    deblocking filter (header)
*/

#ifndef __LOOPFILTER__
#define __LOOPFILTER__

#include "CommonDef.h"
#include "Unit.h"
#include "Picture.h"

//! \ingroup CommonLib
//! \{

#define DEBLOCK_SMALLEST_BLOCK  8

/// Parameters of one luma edge segment of 4 samples, for a deblocking implementation outside LoopFilter (the GPU
/// decoder). flags: bs (bits 0-1), maxFilterLengthP (2-4), maxFilterLengthQ (5-7), sidePisLarge (8),
/// sideQisLarge (9), P not filtered (10), Q not filtered (11), asymmetric filter (12). qp is the average of the two
/// CU QPs before the LADF shift.
struct DeblockLumaEdge
{
  uint32_t flags;
  int8_t   qp;
  int8_t   tcOffsetDiv2;
  int8_t   betaOffsetDiv2;
  int8_t   pad;
};

/// Parameters of the chroma edge segment at the chroma samples of a 4-sample luma segment. flags: bs Cb (bits 0-1),
/// bs Cr (2-3), maxFilterLengthP (4-6), maxFilterLengthQ (7-9), P not filtered (10), Q not filtered (11),
/// asymmetric filter (12), horizontal CTB boundary (13).
struct DeblockChromaEdge
{
  uint32_t flags;
  int8_t   qp[2];
  int8_t   tcOffsetDiv2[2];
  int8_t   betaOffsetDiv2[2];
  int8_t   pad[2];
};

/// Edge maps that LoopFilter fills while it deblocks: one entry per 4x4 luma unit and edge direction, for the edge at
/// the left (vertical) or top (horizontal) of the unit. Entries of edges that are not filtered stay as they are, so
/// the owner clears the maps before each picture.
struct DeblockEdgeMaps
{
  int                unitsW        = 0;
  int                unitsH        = 0;
  DeblockLumaEdge*   luma[2]       = { nullptr, nullptr };   ///< [EDGE_VER, EDGE_HOR]
  DeblockChromaEdge* chroma[2]     = { nullptr, nullptr };
  bool               filterSamples = true;   ///< false: only record the parameters and leave the samples unchanged
};

// ====================================================================================================================
// Class definition
// ====================================================================================================================

/// deblocking filter class
class LoopFilter
{
  static constexpr int LOG_GRID_SIZE = 2;
  static constexpr int GRID_SIZE     = 1 << LOG_GRID_SIZE;

private:
  static_vector<char, MAX_NUM_PARTS_IN_CTU> m_aapucBS       [NUM_EDGE_DIR];         ///< Bs for [Ver/Hor][Y/U/V][Blk_Idx]
  static_vector<bool, MAX_NUM_PARTS_IN_CTU> m_aapbEdgeFilter[NUM_EDGE_DIR];
  LFCUParam m_stLFCUParam;                   ///< status structure
  int     m_ctuXLumaSamples, m_ctuYLumaSamples;                            // location of left-edge and top-edge of CTU
  int     m_shiftHor, m_shiftVer;                                          // shift values to convert location from luma sample units to chroma sample units
  uint8_t m_maxFilterLengthP[MAX_NUM_COMPONENT][MAX_CU_SIZE / GRID_SIZE]
                            [MAX_CU_SIZE
                             / GRID_SIZE];   // maxFilterLengthP for [component][luma/chroma sample distance from left
                                             // edge of CTU][luma/chroma sample distance from top edge of CTU]
  uint8_t m_maxFilterLengthQ[MAX_NUM_COMPONENT][MAX_CU_SIZE / GRID_SIZE]
                            [MAX_CU_SIZE
                             / GRID_SIZE];   // maxFilterLengthQ for [component][luma/chroma sample distance from left
                                             // edge of CTU][luma/chroma sample distance from top edge of CTU]
  bool
    m_transformEdge[MAX_NUM_COMPONENT][MAX_CU_SIZE / GRID_SIZE]
                   [MAX_CU_SIZE / GRID_SIZE];   // transform edge flag for [component][luma/chroma sample distance from
                                                // left edge of CTU][luma/chroma sample distance from top edge of CTU]
  PelStorage                   m_encPicYuvBuffer;
  bool                         m_enc;
private:
  void clearFilterLengthAndTransformEdge();

  // set / get functions
  void xSetLoopfilterParam        ( const CodingUnit& cu );

  // filtering functions
  unsigned
  xGetBoundaryStrengthSingle      ( const CodingUnit& cu, const DeblockEdgeDir edgeDir, const Position& localPos, const ChannelType chType  ) const;

  void xSetEdgefilterMultiple     ( const CodingUnit&     cu,
                                    const DeblockEdgeDir  edgeDir,
                                    const Area&           area,
                                    const bool            bValue,
                                    const bool            EdgeIdx = false );
  void xEdgeFilterLuma( const CodingUnit& cu, const DeblockEdgeDir edgeDir, const int iEdge );
  void xEdgeFilterChroma(const CodingUnit& cu, const DeblockEdgeDir edgeDir, const int iEdge);

#if LUMA_ADAPTIVE_DEBLOCKING_FILTER_QP_OFFSET
  void deriveLADFShift( const Pel* src, const int stride, int& shift, const DeblockEdgeDir edgeDir, const SPS sps );
#endif
  void xSetMaxFilterLengthPQFromTransformSizes( const DeblockEdgeDir edgeDir, const CodingUnit& cu, const TransformUnit& currTU );
  void xSetMaxFilterLengthPQForCodingSubBlocks( const DeblockEdgeDir edgeDir, const CodingUnit& cu, const PredictionUnit& currPU, const bool& mvSubBlocks, const int& subBlockSize, const Area& areaPu );

#if JVET_AB0171_ASYMMETRIC_DB_FOR_GDR  
  inline void xBilinearFilter     ( Pel* srcP, Pel* srcQ, int offset, int refMiddle, int refP, int refQ, int numberPSide, int numberQSide, const int* dbCoeffsP, const int* dbCoeffsQ, int tc, const ClpRng& clpRng) const;
  inline void xFilteringPandQ(Pel* src, int offset, int numberPSide, int numberQSide, int tc, const ClpRng& clpRng) const;
#else
  inline void xBilinearFilter     ( Pel* srcP, Pel* srcQ, int offset, int refMiddle, int refP, int refQ, int numberPSide, int numberQSide, const int* dbCoeffsP, const int* dbCoeffsQ, int tc ) const;
  inline void xFilteringPandQ     ( Pel* src, int offset, int numberPSide, int numberQSide, int tc ) const;
#endif
  
  inline void xPelFilterLuma      ( Pel* piSrc, const int iOffset, const int tc, const bool sw, const bool bPartPNoFilter, const bool bPartQNoFilter, const int iThrCut, const bool bFilterSecondP, const bool bFilterSecondQ, const ClpRng& clpRng, bool sidePisLarge = false, bool sideQisLarge = false, int maxFilterLengthP = 7, int maxFilterLengthQ = 7 ) const;
  inline void xPelFilterChroma(Pel* piSrc, const int iOffset, const int tc, const bool sw, const bool bPartPNoFilter, const bool bPartQNoFilter, const ClpRng& clpRng, const bool largeBoundary, const bool isChromaHorCTBBoundary) const;
  inline bool xUseStrongFiltering(Pel* piSrc, const int iOffset, const int d, const int beta, const int tc, bool sidePisLarge = false, bool sideQisLarge = false, int maxFilterLengthP = 7, int maxFilterLengthQ = 7, bool isChromaHorCTBBoundary = false) const;//move the computation outside the function
  inline unsigned BsSet(unsigned val, const ComponentID compIdx) const;
  inline unsigned BsGet(unsigned val, const ComponentID compIdx) const;

  inline bool isCrossedByVirtualBoundaries ( const int xPos, const int yPos, const int width, const int height, int& numHorVirBndry, int& numVerVirBndry, int horVirBndryPos[], int verVirBndryPos[], const PicHeader* picHeader );
  inline void xDeriveEdgefilterParam       ( const int xPos, const int yPos, const int numVerVirBndry, const int numHorVirBndry, const int verVirBndryPos[], const int horVirBndryPos[], bool &verEdgeFilter, bool &horEdgeFilter );

  inline int xCalcDP(Pel* piSrc, const int iOffset, const bool isChromaHorCTBBoundary = false) const;
  inline int xCalcDQ              ( Pel* piSrc, const int iOffset ) const;
  static const uint16_t sm_tcTable[MAX_QP + 3];
  static const uint8_t sm_betaTable[MAX_QP + 1];

public:

  LoopFilter();
  ~LoopFilter();

  /// CU-level deblocking function
#if JVET_AJ0188_CODING_INFO_CLASSIFICATION || JVET_AK0091_LAPLACIAN_INFO_IN_ALF
  void xDeblockCU(CodingUnit& cu, const DeblockEdgeDir edgeDir, bool storeInfo, PelUnitBuf& alfCodingInfo);
#else
  void xDeblockCU(CodingUnit& cu, const DeblockEdgeDir edgeDir);
#endif
  void  initEncPicYuvBuffer(ChromaFormat chromaFormat, const Size &size, const unsigned maxCUSize);
  PelStorage& getDbEncPicYuvBuffer() { return m_encPicYuvBuffer; }
  void  setEnc(bool b) { m_enc = b; }

  void  create                    ( const unsigned uiMaxCUDepth );
  void  destroy                   ();

  /// picture-level deblocking filter
  void loopFilterPic              ( CodingStructure& cs
#if JVET_AJ0188_CODING_INFO_CLASSIFICATION || JVET_AK0091_LAPLACIAN_INFO_IN_ALF
  , PelUnitBuf& alfCodingInfo, bool storeInfo
#endif
                                    );

  static int getBeta              ( const int qp )
  {
    const int indexB = Clip3( 0, MAX_QP, qp );
    return sm_betaTable[ indexB ];
  }

  void resetFilterLengths();

  /// While maps is set, every filtered edge segment is also recorded there.
  void setEdgeMaps( DeblockEdgeMaps* maps ) { m_edgeMaps = maps; }

private:
  DeblockEdgeMaps* m_edgeMaps = nullptr;

public:

#if JVET_AB0171_ASYMMETRIC_DB_FOR_GDR
private:
  bool m_applyAsymmetricDB;
  bool m_asymmetricDB;
public:
  bool isAsymmetricDB() const { return m_asymmetricDB; }
  void setAsymmetricDB(bool b) { m_asymmetricDB = b; }
#endif
};

//! \}

#endif
