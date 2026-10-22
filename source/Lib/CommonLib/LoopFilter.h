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

  /// the decoder's GPU deblocking (GpuLib's dbf.comp): per segment of an edge that xEdgeFilterLuma /
  /// xEdgeFilterChroma filter (4 luma lines or a chroma part), what the filtering needs besides the samples - all of it
  /// from syntax, geometry and motion, the decisions and LADF's luma level are the kernel's (see GpuDbfSeg's flags)
  struct GpuDbfSeg
  {
    int32_t pos;     ///< x | y << 16 of its first sample in the component: the q side's first sample on the edge
    int32_t flags;   ///< GPU_DBF_* fields
    int32_t tc;      ///< luma: tc's table index before LADF and its clipping; chroma: tc
    int32_t beta;    ///< luma: beta's table index before LADF and its clipping; chroma: beta (largeBoundary)
  };
  // GpuDbfSeg::flags (GpuLib/shaders/dbf.comp): the component; a horizontal edge; the lines (or columns) of the
  // segment; luma: maxFilterLengthP / Q (P after the affine limit), sidePisLarge / sideQisLarge; the palette's
  // bPartPNoFilter / bPartQNoFilter; the asymmetric filtering (m_applyAsymmetricDB); chroma: largeBoundary,
  // isChromaHorCTBBoundary; the second line of the decisions (3, or 1 for subsampled chroma); luma: LADF
  static constexpr int GPU_DBF_HOR         = 1 << 2;
  static constexpr int GPU_DBF_LINES_SHIFT = 3;
  static constexpr int GPU_DBF_MAXP_SHIFT  = 6;
  static constexpr int GPU_DBF_MAXQ_SHIFT  = 10;
  static constexpr int GPU_DBF_SIDE_P      = 1 << 14;
  static constexpr int GPU_DBF_SIDE_Q      = 1 << 15;
  static constexpr int GPU_DBF_NO_P        = 1 << 16;
  static constexpr int GPU_DBF_NO_Q        = 1 << 17;
  static constexpr int GPU_DBF_ASYM        = 1 << 18;
  static constexpr int GPU_DBF_LARGE       = 1 << 19;
  static constexpr int GPU_DBF_CTB         = 1 << 20;
  static constexpr int GPU_DBF_LINE3_SHIFT = 21;
  static constexpr int GPU_DBF_LADF        = 1 << 23;
  /// with rec, loopFilterPic appends the segments it filters in its order (in each pass, component and line of
  /// segments, increasing along the line); filter: also filters (else only derives, its boundary strengths kept as the
  /// filtering would keep them, the ALF coding information stored)
  void setGpuRecord( std::vector<GpuDbfSeg>* rec, bool filter ) { m_gpuRec = rec; m_gpuFilter = filter; m_gpuUnsupported = false; }
  /// something of the recorded picture the kernel does not do (a minimum CU size other than 4 with LADF)
  bool gpuRecordUnsupported() const { return m_gpuUnsupported; }
  static const uint16_t* tcTable() { return sm_tcTable; }
  static const uint8_t*  betaTable() { return sm_betaTable; }

private:
  std::vector<GpuDbfSeg>* m_gpuRec = nullptr;
  bool                    m_gpuFilter = true;
  bool                    m_gpuUnsupported = false;
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
