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

/** \file     EncGOP.h
    \brief    GOP encoder class (header)
*/

#ifndef __ENCGOP__
#define __ENCGOP__

#include <condition_variable>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>

#include <stdlib.h>

#include "CommonLib/Picture.h"
#include "CommonLib/LoopFilter.h"
#include "CommonLib/NAL.h"
#include "EncSampleAdaptiveOffset.h"
#include "EncAdaptiveLoopFilter.h"
#include "EncReshape.h"
#include "EncSlice.h"
#include "VLCWriter.h"
#include "CABACWriter.h"
#include "SEIwrite.h"
#include "SEIEncoder.h"
#if EXTENSION_360_VIDEO
#include "AppEncHelper360/TExt360EncGop.h"
#endif

#include "Analyze.h"
#include "RateCtrl.h"
#include <vector>
#if JVET_V0094_BILATERAL_FILTER || JVET_X0071_CHROMA_BILATERAL_FILTER
#include "BilateralFilter.h"
#endif
#include "EncHRD.h"

#if JVET_O0756_CALCULATE_HDRMETRICS
#include "HDRLib/inc/ConvertColorFormat.H"
#include "HDRLib/inc/Convert.H"
#include "HDRLib/inc/ColorTransform.H"
#include "HDRLib/inc/TransferFunction.H"
#include "HDRLib/inc/DistortionMetricDeltaE.H"
#include <chrono>
#endif

#if JVET_AL0153_ALF_CCCM
#include "EncLoopFilterCccm.h"
#endif

#if NN_LF_UNIFIED
#include "EncNNFilterUnified.h"
#endif

//! \ingroup EncoderLib
//! \{

class EncLib;
struct EncSearchStack;

/// What a picture takes from earlier pictures in coding order and leaves for later ones. EncGOP resolves the sources
/// in the picture's setup; the search then reads only them, not what other pictures leave in shared objects (the
/// search stacks keep nothing from picture to picture). Several search stacks use the records for everything listed;
/// one stack uses them for the AMaxBT statistics and the CABAC init table only.
struct EncFinishContext;

/// Frame threads with several periods: a picture's PSNR sums, added to EncGOP's in coding order (EncGOP::xFtOutput)
struct EncPicAnalyze
{
  Analyze all, i, p, b;
#if WCG_WPSNR
  Analyze wpsnr;
#endif
  std::vector<int> rvmBits;   ///< for EncGOP::m_vRVM_RP
};

struct EncPicOrder
{
  int64_t                       turn   = 0;                     ///< setup and finish turn: the picture's index in coding order
                                                                ///< over all GOPs (frame threads), else its index in the GOP
  double                        searchPriority = 0;             ///< frame threads: waiting searches take stacks in
                                                                ///< decreasing priority (xFtSearchPriority)
  const Picture*                pic    = nullptr;
  int                           stackIdx = 0;                   ///< the picture's search stack (frame threads: from the
                                                                ///< start of the search, xFtAcquireStack)
  bool                          finished = false;               ///< the picture is done, including the padding as a reference
  std::shared_ptr<EncPicOrder>  previous;                       ///< previous picture in coding order (until the finish)
  bool                          cabacTableIdxDeferred = false;  ///< the CABAC init table comes from the previous picture
  SliceType                     encCABACTableIdxAfter = I_SLICE;   ///< CABAC init table the slice writing leaves
#if JVET_Z0135_TEMP_CABAC_WIN_WEIGHT
  std::shared_ptr<EncPicOrder>  ctxSource;                      ///< picture whose stored CABAC contexts this one loads
  CtxStateArray                 ctxStored;                      ///< the contexts this picture stores (after its finish)
#endif
#if JVET_AG0098_AMVP_WITH_SBTMVP
  std::shared_ptr<EncPicOrder>  sbTmvpSource;                   ///< picture whose AMVP-SbTMVP area this one loads
  uint32_t                      sbTmvpStoredArea = 0;           ///< the area this picture stores (after its finish)
#endif
  int                           ctxKeyQp = 0;                   ///< slice QP when the sources were resolved
  bool                          amaxbtRead = false;             ///< reads the block statistics of amaxbtContributors
  std::vector<std::shared_ptr<EncPicOrder>> amaxbtContributors;
  uint32_t                      amaxbtBlkSize = 0;              ///< this picture's AMaxBT block statistics (after its finish)
  uint32_t                      amaxbtNumBlk  = 0;
  PicHeader*                    sharedPicHeader = nullptr;      ///< the picture header the pictures share (EncLib's)
  std::unique_ptr<PicHeader>    picHeader;                      ///< frame threads: the picture's copy of the header
  std::unique_ptr<EncSliceSetup> setup;                         ///< frame threads: what the setup left in the setup
                                                                ///< stack, until the search stack takes it over
  std::vector<const Picture*>   refs;                           ///< frame threads: the reference pictures (until done)
  std::unique_ptr<RdCost>       finishRdCost;                   ///< frame threads: the search stack's RdCost and reshaper
  std::unique_ptr<EncReshape>   finishReshaper;                 ///< at the end of the search, for the finish stack
#if JVET_AI0136_ADAPTIVE_DUAL_TREE
  std::unique_ptr<QTBTPartitioner> finishPartitionerSST;        ///< and the CABAC encoder's separate-tree partitioner
#endif
#if JVET_AG0098_AMVP_WITH_SBTMVP
  uint32_t                      finishAmvpSbTmvpArea = 0;       ///< and the slice encoder's AMVP-SbTMVP area count
#endif
  // frame threads: the finishes of an intra period (from a RAS-init IRAP on) run in coding order with the period's
  // finish context; periods in flight at the same time use different contexts (FramePeriods)
  int                           finishCtx = 0;                  ///< EncLib::getFinishContext
  std::shared_ptr<EncPicOrder>  finishPrev;                     ///< the picture that used the context before (until the
                                                                ///< finish): the previous one of the period, or the last of
                                                                ///< the period that had the context before
  bool                          first     = false;              ///< the first picture in coding order (m_bFirst)
  bool                          seqFirst  = false;              ///< writes the parameter sets as the first (m_bSeqFirst)
  std::string                   log;                            ///< several periods: the finish's log, its access unit and
  std::unique_ptr<AccessUnit>   au;                             ///< its PSNR sums, output in coding order (xFtOutput)
  std::unique_ptr<EncPicAnalyze> analyze;
  bool                          referencedAtSetup = false;      ///< Picture::referenced at the end of the setup (for the log)
};

// ====================================================================================================================
// Class definition
// ====================================================================================================================
class AUWriterIf
{
public:
  virtual void outputAU( const AccessUnit& ) = 0;
};


class EncGOP
{
  class DUData
  {
  public:
    DUData()
    :accumBitsDU(0)
    ,accumNalsDU(0) {};

    int accumBitsDU;
    int accumNalsDU;
  };

private:

  Analyze                 m_gcAnalyzeAll;
  Analyze                 m_gcAnalyzeI;
  Analyze                 m_gcAnalyzeP;
  Analyze                 m_gcAnalyzeB;
#if WCG_WPSNR
  Analyze                 m_gcAnalyzeWPSNR;
#endif
  Analyze                 m_gcAnalyzeAll_in;
#if EXTENSION_360_VIDEO
  TExt360EncGop           m_ext360;
public:
  TExt360EncGop &getExt360Data() { return m_ext360; }
private:
#endif

  //  Data
  bool                    m_bLongtermTestPictureHasBeenCoded;
  bool                    m_bLongtermTestPictureHasBeenCoded2;
  uint32_t                    m_numLongTermRefPicSPS;
  uint32_t                    m_ltRefPicPocLsbSps[MAX_NUM_LONG_TERM_REF_PICS];
  bool                    m_ltRefPicUsedByCurrPicFlag[MAX_NUM_LONG_TERM_REF_PICS];
  int                     m_iLastIDR;
  int                     m_iGopSize;
  static thread_local int m_iNumPicCoded;             ///< pictures coded by the running compressGOP call
  bool                    m_bFirst;
  int                     m_iLastRecoveryPicPOC;
  int                     m_latestDRAPPOC;
  int                     m_lastRasPoc;

  //  Access channel
  EncLib*                 m_pcEncLib;
  EncCfg*                 m_pcCfg;
  // the picture of the running compressGOP call (with frame threads, every picture has a thread)
  static thread_local EncSlice*       m_pcSliceEncoder;   ///< the slice encoder of the picture's search stack
  static thread_local EncSearchStack* m_pcSearchStack;    ///< search stack of the picture being encoded
  static thread_local EncPicOrder*    m_pcPicOrder;       ///< order record of the picture being encoded
  // the objects of the picture's setup that are not the slice encoder's: its search stack's, or the setup stack's
  static thread_local RdCost*          m_pcSetupRdCost;
  static thread_local EncReshape*      m_pcSetupReshaper;
  // the objects of the picture's finish: its search stack's, or with frame threads the finish stack of its context
  static thread_local CABACEncoder*    m_pcFinishCabac;
  static thread_local CtxCache*        m_pcFinishCtxCache;
  static thread_local IntraPrediction* m_pcFinishIntraPred;
  static thread_local RdCost*          m_pcFinishRdCost;
  static thread_local EncReshape*      m_pcFinishReshaper;
  int64_t                 m_numPicsStarted = 0;       ///< pictures started so far (they use the stacks in turn)
  // coding-order sources of the pictures (EncPicOrder), as the setup of the next picture sees them
  std::shared_ptr<EncPicOrder> m_lastPicOrder;
#if JVET_Z0135_TEMP_CABAC_WIN_WEIGHT
  std::map<std::pair<int, int>, std::shared_ptr<EncPicOrder>> m_ctxOrder[2];   ///< as CtxStateStore, by slice type
#endif
#if JVET_AG0098_AMVP_WITH_SBTMVP
  std::map<int, std::shared_ptr<EncPicOrder>> m_sbTmvpOrder;                   ///< as EncSlice's areas, by temporal layer
  uint32_t                m_sbTmvpLastArea = 0;       ///< AMVP-SbTMVP area left by the last picture that counted one
#endif
  PicList*                m_pcListPic;
#if NN_LF_UNIFIED
  EncNNFilterUnified      m_nnfilterUnified;
#endif

  static thread_local HLSWriter* m_HLSWriter;          ///< the finish context's (EncLib::getFinishContext)
#if !JVET_AJ0237_INTERNAL_12BIT
#if JVET_V0094_BILATERAL_FILTER || JVET_X0071_CHROMA_BILATERAL_FILTER
  BilateralFilter         m_cBilateralFilter;
#endif
#endif
  static thread_local LoopFilter* m_pcLoopFilter;      ///< the finish context's

  SEIWriter               m_seiWriter;

  Picture *               m_picBg;
  Picture *               m_picOrig;
  int                     m_bgPOC;
  bool                    m_isEncodedLTRef;
  bool                    m_isPrepareLTRef;
  bool                    m_isUseLTRef;
  int                     m_lastLTRefPoc;
#if JVET_AA0096_MC_BOUNDARY_PADDING
  static thread_local InterPrediction* m_pcFrameMcPadPrediction;   ///< the finish context's MC boundary padding
#endif
  //--Adaptive Loop filter
  static thread_local EncSampleAdaptiveOffset* m_pcSAO;   ///< the finish context's
  static thread_local EncAdaptiveLoopFilter*   m_pcALF;   ///< the finish context's
  static thread_local ParameterSetMap<APS>*    m_pcApsMap;   ///< the finish context's APSs (the setups use EncLib's)
  EncReshape*               m_pcReshaper;
  RdCost                    m_lmcsWeights;   ///< luma level weights of the weighted SSE in coding order (LMCS updates them per picture)
  RateCtrl*                 m_pcRateCtrl;
#if JVET_AL0153_ALF_CCCM
  static thread_local EncLoopFilterCccm* m_pcLoopFilterCccm;   ///< the finish context's
#endif
  // indicate sequence first
  bool                    m_bSeqFirst;
  static thread_local bool m_audIrapOrGdrAuFlag;      ///< set and used by the finish

  EncHRD*                 m_HRD;

  // clean decoding refresh
  bool                    m_bRefreshPending;
  int                     m_pocCRA;
  NalUnitType             m_associatedIRAPType[MAX_VPS_LAYERS];
  int                     m_associatedIRAPPOC[MAX_VPS_LAYERS];

  std::vector<int>        m_vRVM_RP;
  uint32_t                    m_lastBPSEI[MAX_TLAYER];
  uint32_t                    m_totalCoded[MAX_TLAYER];
  bool                        m_rapWithLeading;
  static thread_local bool m_bufferingPeriodSEIPresentInAU;   ///< set and used by the finish
  SEIEncoder              m_seiEncoder;
#if W0038_DB_OPT
  PelStorage*             m_pcDeblockingTempPicYuv;
  int                     m_DBParam[MAX_ENCODER_DEBLOCKING_QUALITY_LAYERS][4];   //[layer_id][0: available; 1: bDBDisabled; 2: Beta Offset Div2; 3: Tc Offset Div2;]
#endif

  // members needed for adaptive max BT size: per depth, the pictures whose statistics the next one of the depth reads
  std::vector<std::shared_ptr<EncPicOrder>> m_amaxbtContributors[10];
  uint32_t                    m_uiPrevISlicePOC;
  bool                    m_bInitAMaxBT;

  AUWriterIf*             m_AUWriterIf;
#if JVET_Z0118_GDR
  int                     m_lastGdrIntervalPoc;  
#endif

#if JVET_O0756_CALCULATE_HDRMETRICS

  hdrtoolslib::Frame **m_ppcFrameOrg;
  hdrtoolslib::Frame **m_ppcFrameRec;

  hdrtoolslib::ConvertColorFormat     *m_pcConvertFormat;
  hdrtoolslib::Convert                *m_pcConvertIQuantize;
  hdrtoolslib::ColorTransform         *m_pcColorTransform;
  hdrtoolslib::DistortionMetricDeltaE *m_pcDistortionDeltaE;
  hdrtoolslib::TransferFunction       *m_pcTransferFct;

  hdrtoolslib::ColorTransformParams   *m_pcColorTransformParams;
  hdrtoolslib::FrameFormat            *m_pcFrameFormat;

  std::chrono::duration<long long, ratio<1, 1000000000>> m_metricTime;
#endif

public:
#if JVET_AJ0237_INTERNAL_12BIT
#if JVET_V0094_BILATERAL_FILTER || JVET_X0071_CHROMA_BILATERAL_FILTER
  BilateralFilter         m_cBilateralFilter;
#endif
#endif

  EncGOP();
  virtual ~EncGOP();

  void  create      ();
  void  destroy     ();

  void  init        ( EncLib* pcEncLib );

  /// Starts the first numPics pictures of the GOP in threads (FrameThreads): each runs compressGOP for its picIdInGOP.
  /// Returns at once: the pictures of later GOPs may start before these are done (setups and finishes keep coding
  /// order over all GOPs). The recon output list is not used.
  void  compressGOPFrameThreads( int numPics, int iPOCLast, int iNumPicRcvd, PicList& rcListPic,
                                 std::list<PelUnitBuf*>& rcListPicYuvRec, bool isField, bool isTff,
                                 const InputColourSpaceConversion snr_conversion, const bool printFrameMSE,
#if MSSIM_UNIFORM_METRICS_LOG
                                 const bool printMSSSIM,
#endif
                                 bool isEncodeLtRef );
  /// Waits until at most maxPending of the started pictures are not done; throws the error of a failed picture.
  void  waitFrameThreads    ( int64_t maxPending );
  /// Waits until all started pictures are done; throws the error of a failed picture.
  void  finishFrameThreads  ();
  /// A picture is done (encoded, written, its reconstruction final); forgetPicDone before the buffer is reused.
  bool  isPicDone           ( const Picture* pic );
  void  forgetPicDone       ( const Picture* pic );
  /// The picture's finish has ended (its log and access unit may still wait for earlier pictures); until forgetPicDone.
  bool  isPicFinished       ( const Picture* pic );
  /// Frame threads: a picture set up and not done references the picture (its buffer must not be reused).
  bool  isPicReferencedInFlight( const Picture* pic );
  void  compressGOP ( int iPOCLast, int iNumPicRcvd, PicList& rcListPic, std::list<PelUnitBuf*>& rcListPicYuvRec,
                      bool isField, bool isTff, const InputColourSpaceConversion snr_conversion, const bool printFrameMSE,
#if MSSIM_UNIFORM_METRICS_LOG
                      const bool printMSSSIM,
#endif
                      bool isEncodeLtRef, const int picIdInGOP);
  void  xAttachSliceDataToNalUnit (OutputNALUnit& rNalu, OutputBitstream* pcBitstreamRedirect);


  int   getGOPSize()          { return  m_iGopSize;  }

  PicList*   getListPic()      { return m_pcListPic; }
  void      setPicBg(Picture* tmpPicBg) { m_picBg = tmpPicBg; }
  Picture*  getPicBg() const { return m_picBg; }
  void      setPicOrig(Picture* tmpPicBg) { m_picOrig = tmpPicBg; }
  Picture*  getPicOrig() { return m_picOrig; }
  void      setNewestBgPOC(int poc) { m_bgPOC = poc; }
  int       getNewestBgPOC() const { return m_bgPOC; }
  void      setEncodedLTRef(bool isEncodedLTRef) { m_isEncodedLTRef = isEncodedLTRef; }
  bool      getEncodedLTRef() { return m_isEncodedLTRef; }
  void      setUseLTRef(bool isUseLTRef) { m_isUseLTRef = isUseLTRef; }
  bool      getUseLTRef() { return m_isUseLTRef; }
  void      setPrepareLTRef(bool isPrepareLTRef) { m_isPrepareLTRef = isPrepareLTRef; }
  bool      getPrepareLTRef() { return m_isPrepareLTRef; }
  void      setLastLTRefPoc(int iLastLTRefPoc) { m_lastLTRefPoc = iLastLTRefPoc; }
  int       getLastLTRefPoc() const { return m_lastLTRefPoc; }
#if JVET_Z0118_GDR
  void      setLastGdrIntervalPoc(int p)  { m_lastGdrIntervalPoc = p; }
  int       getLastGdrIntervalPoc() const { return m_lastGdrIntervalPoc; }
#endif

  void printOutSummary(uint32_t uiNumAllPicCoded, bool isField, const bool printMSEBasedSNR,
                       const bool printSequenceMSE,
#if MSSIM_UNIFORM_METRICS_LOG
                       const bool printMSSSIM,
#endif
                       const bool printHexPsnr, const bool printRprPSNR, const BitDepths &bitDepths
#if JVET_W0134_UNIFORM_METRICS_LOG
                       ,
                       int layerId
#endif
  );
#if W0038_DB_OPT
  uint64_t  preLoopFilterPicAndCalcDist( Picture* pcPic );
#endif
  EncSlice*  getSliceEncoder()   { return m_pcSliceEncoder; }
  NalUnitType getNalUnitType( int pocCurr, int lastIdr, bool isField );
  void arrangeCompositeReference(Slice* pcSlice, PicList& rcListPic, int pocCurr);
  void updateCompositeReference(Slice* pcSlice, PicList& rcListPic, int pocCurr);

#if JVET_AG0145_ADAPTIVE_CLIPPING
  void adaptiveClipToRealRange(Picture* pcPic);
  void getRealRange(Picture* pcPic);
#endif

#if EXTENSION_360_VIDEO
  Analyze& getAnalyzeAllData() { return m_gcAnalyzeAll; }
  Analyze& getAnalyzeIData() { return m_gcAnalyzeI; }
  Analyze& getAnalyzePData() { return m_gcAnalyzeP; }
  Analyze& getAnalyzeBData() { return m_gcAnalyzeB; }
#endif
#if JVET_O0756_CALCULATE_HDRMETRICS
  std::chrono::duration<long long, ratio<1, 1000000000>> getMetricTime()    const { return m_metricTime; };
#endif

protected:
  RateCtrl* getRateCtrl()       { return m_pcRateCtrl;  }

protected:

  void  xInitGOP          ( int iPOCLast, int iNumPicRcvd, bool isField
    , bool isEncodeLtRef
  );
  void  xPicInitHashME( Picture *pic, const PPS *pps, PicList &rcListPic );
  void  xPicInitRateControl(int &estimatedBits, int gopId, double &lambda, Picture *pic, Slice *slice);
  void  xPicInitLMCS       (Picture *pic, PicHeader *picHeader, Slice *slice);
  void  xGetBuffer        ( PicList& rcListPic, std::list<PelUnitBuf*>& rcListPicYuvRecOut,
                            int iNumPicRcvd, int iTimeOffset, Picture*& rpcPic, int pocCurr, bool isField );

#if JVET_O0756_CALCULATE_HDRMETRICS
  void xCalculateHDRMetrics ( Picture* pcPic, double deltaE[hdrtoolslib::NB_REF_WHITE], double psnrL[hdrtoolslib::NB_REF_WHITE]);
  void copyBuftoFrame       ( Picture* pcPic );
#endif

  void xCalculateAddPSNRs(const bool isField, const bool isFieldTopFieldFirst, const int iGOPid, Picture *pcPic,
                          const AccessUnit &accessUnit, PicList &rcListPic, double dEncTime,
                          const InputColourSpaceConversion snr_conversion, const bool printFrameMSE,
#if MSSIM_UNIFORM_METRICS_LOG
                          const bool printMSSSIM,
#endif
                          double *PSNR_Y, bool isEncodeLtRef);
  void xCalculateAddPSNR(Picture *pcPic, PelUnitBuf cPicD, const AccessUnit &, double dEncTime,
                         const InputColourSpaceConversion snr_conversion, const bool printFrameMSE,
#if MSSIM_UNIFORM_METRICS_LOG
                         const bool printMSSSIM,
#endif
                         double *PSNR_Y, bool isEncodeLtRef);

  void  xCalculateInterlacedAddPSNR( Picture* pcPicOrgFirstField, Picture* pcPicOrgSecondField,
                                     PelUnitBuf cPicRecFirstField, PelUnitBuf cPicRecSecondField,
                                   const InputColourSpaceConversion snr_conversion, const bool printFrameMSE,
#if MSSIM_UNIFORM_METRICS_LOG
                                   const bool printMSSSIM,
#endif
                                   double *PSNR_Y, bool isEncodeLtRef);
#if MSSIM_UNIFORM_METRICS_LOG
  double xCalculateMSSSIM(const Pel *org, const int orgStride, const Pel *rec, const int recStride, const int width,
                          const int height, const uint32_t bitDepth);
#endif

  uint64_t xFindDistortionPlane(const CPelBuf& pic0, const CPelBuf& pic1, const uint32_t rshift
#if ENABLE_QPA
                            , const uint32_t chromaShiftHor = 0, const uint32_t chromaShiftVer = 0
#endif
                             );
#if WCG_WPSNR
  double xFindDistortionPlaneWPSNR(const CPelBuf& pic0, const CPelBuf& pic1, const uint32_t rshift, const CPelBuf& picLuma0, ComponentID compID, const ChromaFormat chfmt );
#endif
  double xCalculateRVM();

  void xUpdateRasInit(Slice* slice);
  // the coding-order record of a picture (EncPicOrder): sources resolved at the end of the setup, applied at the start
  // of the search and of the finish, results recorded at the end of the finish
  void xSetupPicOrder ( Slice* slice, const std::shared_ptr<EncPicOrder>& order );
  void xStartPicSearch( Picture* pic, PicHeader* picHeader, Slice* slice, EncPicOrder& order, bool decPic );
  double xFtSearchPriority( Picture& pic, const EncPicOrder& order ) const;
  bool xUseFilteredOrig() const;   // whether the pictures keep a filtered original (temporal filter or BIM)
  void xStartPicFinish( Slice* slice, EncPicOrder& order, std::unique_lock<std::mutex>& apsMapLock );
  void xEndPicFinish  ( Slice* slice, EncPicOrder& order, std::unique_lock<std::mutex>& apsMapLock );
  void xApplyAMaxBT   ( Picture* pic, PicHeader* picHeader, Slice* slice, EncPicOrder& order );
  PicHeader* xUsePicHeaderCopy( Picture* pic, EncPicOrder& order );
  void xEndPicture    ( EncPicOrder& order );

  // frame threads (compressGOPFrameThreads): the pictures of a GOP run compressGOP in threads of their own; setups
  // and finishes take turns in coding order, a search starts when the pictures it reads are done
  struct FtAbort {};                                   ///< thrown into the waiting threads when another one fails
  bool                     m_ftActive = false;
  std::mutex               m_ftMutex;
  std::condition_variable  m_ftCv;
  int64_t                  m_ftSetupTurn  = 0;         ///< turn (EncPicOrder::turn) whose setup may run
  bool                     m_ftInputWaits = false;     ///< the input side waits for pictures to be done (waitFrameThreads)
  int                      m_ftPeriods    = 1;         ///< finish contexts (FramePeriods)
  int64_t                  m_ftPeriod     = -1;        ///< the intra period of the last setup
  std::vector<std::shared_ptr<EncPicOrder>> m_ftCtxLast;   ///< per finish context: the last picture set up to use it
  int64_t                  m_ftFirstNotFinished = 0;   ///< the earliest turn not finished (xFtAcquireStack)
  std::set<int64_t>        m_ftFinishedAhead;          ///< the finished turns after it
  int64_t                  m_ftOutputTurn = 0;         ///< several periods: the turn whose output runs next (xFtOutput)
  std::mutex               m_ftParamSetMutex;          ///< the parameter sets' changed flags, set and written in a finish
  std::mutex               m_ftSeiMutex;               ///< the SEI writer
#if JVET_AI0136_ADAPTIVE_DUAL_TREE
  std::unique_ptr<QTBTPartitioner> m_ftWriterPartitionerSST;   ///< as the last slice writing left it, for the next search
#endif
  int64_t                  m_ftNumTurns   = 0;         ///< turns started (pictures and GOP indices without one)
  int64_t                  m_ftNumDone    = 0;         ///< turns whose thread has returned
  static thread_local int64_t m_ftTurn;                ///< the turn of the running compressGOP call
  bool                     m_ftError      = false;
  std::exception_ptr       m_ftFirstError;
  std::set<const Picture*> m_ftPicsDone;               ///< done and not yet forgotten (forgetPicDone)
  std::set<const Picture*> m_ftPicsFinished;           ///< finish ended and not yet forgotten (forgetPicDone)
  std::map<const Picture*, int> m_ftRefsInFlight;      ///< per picture: references to it by pictures set up and not done
  std::vector<bool>        m_ftStackBusy;
  std::set<std::pair<double, int64_t>> m_ftStackWaiters;   ///< (-search priority, turn) of the searches waiting for a
                                                          ///< stack (xFtAcquireStack): the first takes the next one
  std::set<int64_t>        m_ftStackHolders;           ///< turns holding a stack
  std::set<const Picture*> m_ftPicsInFlight;           ///< set up and not done (not usable as a reference yet)
  std::mutex               m_ftApsMapMutex;            ///< EncLib's APS map: held by the finish, taken by a setup to use it
  std::mutex               m_ftSaoStatsMutex;          ///< several periods: held by the finish that has SAO statistics
  template<class Pred> void xFtWait( Pred pred );      ///< waits until pred() holds (pred runs with m_ftMutex locked)
  void xFtRunInPool       ( std::function<void()> job );   ///< runs job on a worker thread
  void xFtSkip            ( int64_t turn );            ///< a GOP index without a picture passes both turns
  void xFtAcquireStack    ( EncPicOrder& order );
  void xFtReleaseStack    ( EncPicOrder& order );     ///< at the end of the search (the finish has a stack of its own)
  void xLoadCodingOrderState( Slice* slice, EncPicOrder& order );   ///< CABAC store and AMVP-SbTMVP areas into the stack
  void xFtMarkFinished    ( int64_t turn );            ///< m_ftMutex held
  void xFtOutput          ( EncPicOrder& order );
  void xUseFinishContext  ( const EncFinishContext& ctx );
  void xFtEndSetup        ( EncPicOrder& order );
  void xFtWaitSearchInputs( Slice* slice, EncPicOrder& order );

  void xWriteAccessUnitDelimiter (AccessUnit &accessUnit, Slice *slice);

  void xCreateIRAPLeadingSEIMessages (SEIMessages& seiMessages, const SPS *sps, const PPS *pps);
  void xCreatePerPictureSEIMessages (int picInGOP, SEIMessages& seiMessages, SEIMessages& nestedSeiMessages, Slice *slice);
  void xCreateFrameFieldInfoSEI (SEIMessages& seiMessages, Slice *slice, bool isField);
  void xCreatePictureTimingSEI  (int IRAPGOPid, SEIMessages& seiMessages, SEIMessages& nestedSeiMessages, SEIMessages& duInfoSeiMessages, Slice *slice, bool isField, std::deque<DUData> &duData);
  void xUpdateDuData(AccessUnit &testAU, std::deque<DUData> &duData);
  void xUpdateTimingSEI(SEIPictureTiming *pictureTimingSEI, std::deque<DUData> &duData, const SPS *sps);
  void xUpdateDuInfoSEI(SEIMessages &duInfoSeiMessages, SEIPictureTiming *pictureTimingSEI, int maxSubLayers);
#if JVET_R0294_SUBPIC_HASH
  void xCreateScalableNestingSEI(SEIMessages& seiMessages, SEIMessages& nestedSeiMessages, const std::vector<int> &targetOLSs, const std::vector<int> &targetLayers, const std::vector<uint16_t>& subpicIDs);
#else
  void xCreateScalableNestingSEI(SEIMessages& seiMessages, SEIMessages& nestedSeiMessages, const std::vector<uint16_t>& subpicIDs);
#endif
  void xWriteSEI (NalUnitType naluType, SEIMessages& seiMessages, AccessUnit &accessUnit, AccessUnit::iterator &auPos, int temporalId);
  void xWriteSEISeparately (NalUnitType naluType, SEIMessages& seiMessages, AccessUnit &accessUnit, AccessUnit::iterator &auPos, int temporalId);
  void xClearSEIs(SEIMessages& seiMessages, bool deleteMessages);
  void xWriteLeadingSEIOrdered (SEIMessages& seiMessages, SEIMessages& duInfoSeiMessages, AccessUnit &accessUnit, int temporalId, bool testWrite);
  void xWriteLeadingSEIMessages  (SEIMessages& seiMessages, SEIMessages& duInfoSeiMessages, AccessUnit &accessUnit, int temporalId, const SPS *sps, std::deque<DUData> &duData);
  void xWriteTrailingSEIMessages (SEIMessages& seiMessages, AccessUnit &accessUnit, int temporalId);
  void xWriteDuSEIMessages       (SEIMessages& duInfoSeiMessages, AccessUnit &accessUnit, int temporalId, std::deque<DUData> &duData);

  int xWriteVPS (AccessUnit &accessUnit, const VPS *vps);
  int xWriteDCI (AccessUnit &accessUnit, const DCI *dci);
  int xWriteSPS( AccessUnit &accessUnit, const SPS *sps, const int layerId = 0 );
  int xWritePPS( AccessUnit &accessUnit, const PPS *pps, const int layerId = 0 );
  int xWriteAPS( AccessUnit &accessUnit, APS *aps, const int layerId, const bool isPrefixNUT );
  int xWriteParameterSets(AccessUnit &accessUnit, Slice *slice, const bool bSeqFirst, const int layerIdx);
  int xWritePicHeader( AccessUnit &accessUnit, PicHeader *picHeader );

  void applyDeblockingFilterMetric( Picture* pcPic, uint32_t uiNumSlices );
#if DB_PARAM_TID
  void applyDeblockingFilterParameterSelection( Picture* pcPic, Slice* pcSlice, const uint32_t numSlices, const int gopID );
#endif
#if W0038_DB_OPT
  void applyDeblockingFilterParameterSelection( Picture* pcPic, const uint32_t numSlices, const int gopID );
#endif
  void xCreateExplicitReferencePictureSetFromReference( Slice* slice, PicList& rcListPic, const ReferencePictureList *rpl0, const ReferencePictureList *rpl1 );
  bool xCheckMaxTidILRefPics(Picture* refPic, bool currentPicIsIRAP);

#if EMBEDDED_APS
  static thread_local std::vector<APS> m_aps;   ///< the APSs the picture header embeds, collected in the finish
#endif
};// END CLASS DEFINITION EncGOP

//! \}

#if JVET_Q0406_CABAC_ZERO
class EncBitstreamParams
{
public:
  EncBitstreamParams()
  : numBinsWritten(0)
  , numBytesInVclNalUnits(0)
  {};

  std::size_t numBinsWritten;
  std::size_t numBytesInVclNalUnits;
};
#endif

#endif // __ENCGOP__

