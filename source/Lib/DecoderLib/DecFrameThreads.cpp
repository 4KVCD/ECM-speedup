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

/** \file     DecFrameThreads.cpp
    \brief    frame-level threading of the decoder (see DecFrameThreads.h)
*/

#include "DecFrameThreads.h"
#include "DecLib.h"

#include "CommonLib/UnitTools.h"

#include <algorithm>
#include <map>

#if JVET_AG0196_CABAC_RETRAIN
namespace CabacRetrain
{
  extern void endFrame( int poc, int qp, bool switchBp, SliceType st );
}
#endif

//! \ingroup DecoderLib
//! \{

// ====================================================================================================================
// worker threads
// ====================================================================================================================

DecFrameThreads::DecFrameThreads( DecLib& main, int numThreads, bool serial )
  : m_serial( serial )
{
  // the contexts are set up before any thread runs (their constructors write global function tables)
  for( int i = 0; i < numThreads; i++ )
  {
    m_ctx.emplace_back( new DecLib );
    DecLib& ctx = *m_ctx.back();
    ctx.create();
#if JVET_J0090_MEMORY_BANDWITH_MEASURE
    ctx.init( "" );
#else
    ctx.init();
#endif
    ctx.initScalingList();
    ctx.m_waitHub         = main.m_waitHub;
    ctx.m_targetSubPicIdx = main.m_targetSubPicIdx;
#if ECM_VULKAN
    ctx.m_gpuLoopFilter   = main.m_gpuLoopFilter;
    if( main.m_gpuIntraRec )   // every context reconstructs its own pictures on the GPU (own queue and buffers)
    {
      ctx.setGpuRecon( main.m_gpuIntraRec->on() ? 1 : 2 );
    }
#endif
#if JVET_AL0142_CCSAO_REUSE_CTU
    ctx.m_CABACDecoder.getCABACReader( 0 )->setCcSaoPrvMirror( false );
#endif
  }
  for( auto& ctx: m_ctx )
  {
    m_threads.emplace_back( &DecFrameThreads::workerLoop, this, ctx.get() );
  }
}

DecFrameThreads::~DecFrameThreads()
{
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    m_stop = true;
  }
  m_cond.notify_all();
  for( auto& t: m_threads )
  {
    t.join();
  }
  for( auto& ctx: m_ctx )
  {
    ctx->deletePicBuffer();
    ctx->destroy();
  }
}

void DecFrameThreads::enqueue( const std::shared_ptr<DecPicJob>& job )
{
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    m_queue.push_back( job );
  }
  m_cond.notify_one();
}

void DecFrameThreads::workerLoop( DecLib* ctx )
{
  for( ;; )
  {
    std::shared_ptr<DecPicJob> job;
    {
      std::unique_lock<std::mutex> lock( m_mutex );
      m_cond.wait( lock, [&] { return m_stop || !m_queue.empty(); } );
      // stopping: a normal end has drained the queue; after an error the queued pictures are not decoded (no job
      // being decoded waits for a queued one: they are taken in order)
      if( m_stop || m_queue.empty() )
      {
        return;
      }
      job = m_queue.front();
      m_queue.pop_front();
    }
    ctx->xRunPicJob( *job );
  }
}

// ====================================================================================================================
// worker context: decoding of a picture
// ====================================================================================================================

void DecLib::xRunPicJob( DecPicJob& job )
{
  try
  {
    for( const auto& ref: job.refJobs )
    {
      m_waitHub->wait( ref->done );
    }
    m_pcPic = job.pic;
    xInitPictureTools( job.slices[0]->getSPS(), job.slices[0]->getPPS() );
    for( size_t i = 0; i < job.slices.size(); i++ )
    {
      xBindSliceToTools( job.slices[i] );
      xDecodeSliceData( job.slices[i], job.bitstreams[i].get(), job.sliceStates[i] );
    }
    xExecuteLoopFilters( &job.lf );
#if JVET_AG0145_ADAPTIVE_CLIPPING
    xClipToRealRange();
#endif
    xFinishPictureData();
  }
  catch( std::exception& e )
  {
    job.error = e.what();
  }
  catch( ... )
  {
    job.error = "unknown exception";
  }
  if( !job.error.empty() )
  {
    // what later pictures wait for: they are not decoded correctly, the main thread stops at this picture's report
    for( auto& st: job.sliceStates )
    {
#if JVET_Z0135_TEMP_CABAC_WIN_WEIGHT
      if( st.ctxStore && !st.ctxStore->ready )
      {
        m_waitHub->publish( *st.ctxStore );
      }
#endif
#if JVET_AG0098_AMVP_WITH_SBTMVP
      if( st.sbTmvpArea && !st.sbTmvpArea->ready )
      {
        m_waitHub->publish( *st.sbTmvpArea );
      }
#endif
#if JVET_AL0142_CCSAO_REUSE_CTU
      for( auto& w: st.ccSaoWrite )
      {
        if( w && !w->ready )
        {
          m_waitHub->publish( *w );
        }
      }
#endif
    }
#if JVET_AM0209_CHROMA_ALF_CCALF_REUSE_CTU
    for( int comp = 0; comp < MAX_NUM_COMPONENT; comp++ )
    {
      for( auto* s: { job.lf.alfStore[comp].get(), job.lf.ccAlfStore[comp].get() } )
      {
        if( s && !s->ready )
        {
          m_waitHub->publish( *s );
        }
      }
    }
#endif
#if JVET_AI0084_ALF_RESIDUALS_SCALING
    for( auto& st: job.lf.alfScaleStores )
    {
      if( !st.slot->ready )
      {
        m_waitHub->publish( *st.slot );
      }
    }
#endif
  }
  m_waitHub->publish( job.done );
}

#if JVET_AG0145_ADAPTIVE_CLIPPING
void DecLib::xClipToRealRange()
{
  // adaptiveClipToRealRange without setting the picture's clipping range (set when the picture was dispatched)
  ClpRng clpRng;
  clpRng.min = m_pcPic->cs->slice->getLumaPelMin();
  clpRng.max = m_pcPic->cs->slice->getLumaPelMax();
  int compIdx = 0;
  ComponentID compID = ComponentID(compIdx);
  int width = m_pcPic->cs->pps->getPicWidthInLumaSamples();
  int height = m_pcPic->cs->pps->getPicHeightInLumaSamples();
  Pel* reconPel = m_pcPic->getRecoBuf().get(compID).buf;
  int stride = m_pcPic->getRecoBuf().get(compID).stride;

  for (uint32_t yPos = 0; yPos < height; yPos++)
  {
    for (uint32_t xPos = 0; xPos < width; xPos++)
    {
      reconPel[xPos] = ClipPel(reconPel[xPos], clpRng);
    }
    reconPel += stride;
  }
}
#endif

void DecLib::xFinishPictureData()
{
  // the parts of finishPicture that work on the picture's data
#if JVET_Z0118_GDR
  m_pcPic->setCleanDirty(false);
  m_pcPic->copyCleanCurPicture();
#endif
  Slice* pcSlice = m_pcPic->cs->slice;

  m_pcPic->destroyTempBuffers();
#if JVET_AH0135_TEMPORAL_PARTITIONING
  m_pcPic->cs->destroyCoeffs();
  if (!(!pcSlice->isIntra() && !((m_pcPic->temporalId == 0) || (pcSlice->getSPS()->getNumReorderPics(m_pcPic->temporalId) != m_pcPic->temporalId))))
  {
    m_pcPic->cs->setSplitPred();
  }
#endif
  m_pcPic->cs->destroyTemporaryCsData();
#if JVET_AA0096_MC_BOUNDARY_PADDING
#if JVET_AK0085_TM_BOUNDARY_PADDING
  if(!(m_pcPic->getUseTMBP() && m_pcPic->cs->sps->getTMBP()))
#endif
  {
    m_cFrameMcPadPrediction.init(&m_cRdCost, pcSlice->getSPS()->getChromaFormatIdc(), pcSlice->getSPS()->getMaxCUHeight(),
#if JVET_AJ0172_IBC_ITMP_ALIGN_REF_AREA
#if JVET_AJ0237_INTERNAL_12BIT
                                 NULL, m_pcPic->getPicWidthInLumaSamples(),m_pcPic->getPicHeightInLumaSamples(), pcSlice->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA));
#else
                                 NULL, m_pcPic->getPicWidthInLumaSamples(),m_pcPic->getPicHeightInLumaSamples());
#endif
#else
#if JVET_AJ0237_INTERNAL_12BIT
                                 NULL, m_pcPic->getPicWidthInLumaSamples(), pcSlice->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA));
#else
                                 NULL, m_pcPic->getPicWidthInLumaSamples());
#endif
#endif
    m_cFrameMcPadPrediction.mcFramePad(m_pcPic, *(m_pcPic->slices[0]));
  }
#if JVET_AK0085_TM_BOUNDARY_PADDING
  else
  {
    m_pcPic->extendPicBorder(pcSlice->getPPS());
  }
#endif
#endif
}

// ====================================================================================================================
// main thread
// ====================================================================================================================

void DecLib::setFrameThreads( int numThreads, bool serial )
{
  if( numThreads > 1 )
  {
    m_ownWaitHub.setThreaded( true );
    m_frameThreads.reset( new DecFrameThreads( *this, numThreads, serial ) );
#if JVET_AL0142_CCSAO_REUSE_CTU
    m_CABACDecoder.getCABACReader( 0 )->setCcSaoPrvMirror( false );   // the shared entries belong to later pictures
#endif
  }
}

bool DecLib::xPictureNeedsDrain( const SPS* sps, const PPS* pps ) const
{
  // decoded by the main thread after the pictures in flight: tools that change other pictures while a picture is
  // decoded (long-term marking, subpicture borders, GDR clean / dirty buffers) or that are not set up in the
  // worker contexts
  bool drain = sps->getLongTermRefsPresent() || pps->getNumSubPics() > 1 || m_targetSubPicIdx != 0 || m_debugPOC >= 0;
#if JVET_Z0118_GDR
  drain = drain || sps->getGDREnabledFlag();
#endif
#if NN_LF_UNIFIED
  drain = drain || sps->getNnlfUnifiedEnabledFlag();
#endif
  drain = drain || ( m_vps != nullptr && m_vps->getMaxLayers() > 1 );
  return drain;
}

void DecLib::xWaitRefPictures( const Slice* slice )
{
  for( int list = 0; list < NUM_REF_PIC_LIST_01; list++ )
  {
    for( int idx = 0; idx < slice->getNumRefIdx( RefPicList( list ) ); idx++ )
    {
      const auto it = m_jobOfPic.find( slice->getRefPic( RefPicList( list ), idx ) );
      if( it != m_jobOfPic.end() && it->second )
      {
        m_waitHub->wait( it->second->done );
      }
    }
  }
}

void DecLib::xMakeSyncPicture()
{
  // a picture with more than one slice: the main thread decodes it, each slice before the next one is set up
  m_syncPicture = true;
  if( !m_pendingJob )
  {
    return;
  }
  m_pendingJob->sync = true;
  for( size_t i = 0; i < m_pendingJob->slices.size(); i++ )
  {
    if( m_pendingJob->bitstreams[i] )
    {
      xWaitRefPictures( m_pendingJob->slices[i] );
      xDecodeSliceData( m_pendingJob->slices[i], m_pendingJob->bitstreams[i].get(), m_pendingJob->sliceStates[i] );
      m_pendingJob->bitstreams[i].reset();
    }
  }
}

void DecLib::xSnapshotParameters( DecPicJob& job )
{
  // the picture header and the APS a picture uses may be replaced by those of the next pictures while it is decoded,
  // and every picture header rebuilds the tile, slice and subpicture maps of the PPS (parsePictureHeader)
  CodingStructure& cs  = *job.pic->cs;
  const PPS*       pps = job.slices[0]->getPPS();
  job.pps.reset( new PPS( *pps ) );
  for( auto* slice: job.slices )
  {
    CHECK( slice->getPPS() != pps, "slices of a picture with different PPS" );
    slice->setPPS( job.pps.get() );
  }
  cs.pps = job.pps.get();
  PicHeader*       ph = const_cast<PicHeader*>( job.slices[0]->getPicHeader() );
  if( ph == &m_picHeader )
  {
    job.picHeader.reset( new PicHeader( m_picHeader ) );
    ph = job.picHeader.get();
    for( auto* slice: job.slices )
    {
      slice->setPicHeader( ph );
    }
    if( cs.picHeader == &m_picHeader )
    {
      cs.picHeader = ph;
    }
  }
  std::map<const APS*, APS*> copies;
  auto copyOf = [&]( APS* aps ) -> APS*
  {
    if( aps == nullptr )
    {
      return nullptr;
    }
    auto it = copies.find( aps );
    if( it != copies.end() )
    {
      return it->second;
    }
    job.apss.emplace_back( new APS( *aps ) );
    copies[aps] = job.apss.back().get();
    return job.apss.back().get();
  };
  for( auto* slice: job.slices )
  {
    APS** alfApss = slice->getAlfAPSs();
    for( int i = 0; i < ALF_CTB_MAX_NUM_APS; i++ )
    {
      alfApss[i] = copyOf( alfApss[i] );
    }
#if JVET_AK0065_TALF
    APS** talfApss = slice->getTAlfAPSs();
    for( int i = 0; i < ALF_CTB_MAX_NUM_APS; i++ )
    {
      talfApss[i] = copyOf( talfApss[i] );
    }
#endif
  }
  for( int i = 0; i < ALF_CTB_MAX_NUM_APS; i++ )
  {
    cs.alfApss[i] = copyOf( cs.alfApss[i] );
#if JVET_AK0065_TALF
    cs.talfApss[i] = copyOf( cs.talfApss[i] );
#endif
  }
  cs.lmcsAps        = copyOf( cs.lmcsAps );
  cs.scalinglistAps = copyOf( cs.scalinglistAps );
  if( ph->getLmcsAPS() )
  {
    ph->setLmcsAPS( copyOf( ph->getLmcsAPS() ) );
  }
  if( ph->getScalingListAPS() )
  {
    ph->setScalingListAPS( copyOf( ph->getScalingListAPS() ) );
  }
}

void DecLib::dispatchPicture( int& poc, PicList*& rpcListPic, bool associatedWithNewClvs )
{
  CHECK( !m_pendingJob || m_pendingJob->slices.empty(), "no picture to dispatch" );
  std::shared_ptr<DecPicJob> job = std::move( m_pendingJob );
  m_pendingJob.reset();
  Picture* pic   = m_pcPic;
  Slice*   slice = pic->cs->slice;   // the last slice, as executeLoopFilters / finishPicture see it
  job->pic       = pic;
  for( auto* s: job->slices )
  {
    for( int list = 0; list < NUM_REF_PIC_LIST_01; list++ )
    {
      for( int idx = 0; idx < s->getNumRefIdx( RefPicList( list ) ); idx++ )
      {
        const Picture* ref = s->getRefPic( RefPicList( list ), idx );
        if( std::find( job->usedPics.begin(), job->usedPics.end(), ref ) == job->usedPics.end() )
        {
          job->usedPics.push_back( ref );
          const auto it = m_jobOfPic.find( ref );
          if( it != m_jobOfPic.end() && it->second && !it->second->done.ready.load( std::memory_order_acquire ) )
          {
            job->refJobs.push_back( it->second );
          }
        }
      }
    }
  }

  // decode-order state of the loop filters, in the order executeLoopFilters / ALFProcess handle it
  job->lf.slice = slice;
  if( slice->getSPS()->getALFEnabledFlag() )
  {
#if JVET_AM0209_CHROMA_ALF_CCALF_REUSE_CTU
    xSetupLfReads( slice, job->lf );
    xSetupLfStores( slice, job->lf );
#endif
#if JVET_AI0084_ALF_RESIDUALS_SCALING
    xSetupAlfScale( slice, job->lf );
#endif
  }

  if( job->sync )
  {
    // decoded by the main thread: the rest the stock way
    xExecuteLoopFilters( &job->lf );
#if JVET_AG0145_ADAPTIVE_CLIPPING
    adaptiveClipToRealRange();
#endif
    xFinishPictureData();
    job->bitstreams.clear();
    xSnapshotParameters( *job );   // its POC line is printed later, after the next picture headers are parsed
    m_waitHub->publish( job->done );
  }
  else
  {
    // what executeLoopFilters leaves for the next slice headers / pictures
    if( slice->getSPS()->getALFEnabledFlag() )
    {
      m_cALF.getCcAlfFilterParam() = slice->m_ccAlfFilterParam;
    }
#if JVET_W0066_CCSAO
    if( slice->getSPS()->getCCSAOEnabledFlag() )
    {
      m_cSAO.getCcSaoComParam() = slice->m_ccSaoComParam;
    }
#endif
#if JVET_AG0145_ADAPTIVE_CLIPPING
    pic->lumaClpRng.max = slice->getLumaPelMax();   // read by the slice set-up of later pictures (collocated picture)
    pic->lumaClpRng.min = slice->getLumaPelMin();
#endif
    xSnapshotParameters( *job );
  }

  // finishPicture: the picture state the main thread works with
  m_prevPicPOC    = slice->getPOC();
  job->referenced = pic->referenced;
  pic->neededForOutput = (slice->getPicHeader()->getPicOutputFlag() ? true : false);
#if JVET_R0270
  if (associatedWithNewClvs && pic->neededForOutput)
  {
    if (!slice->getPPS()->getMixedNaluTypesInPicFlag() && slice->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL)
    {
      pic->neededForOutput = false;
    }
    else if (slice->getPPS()->getMixedNaluTypesInPicFlag())
    {
      bool isRaslPic = true;
      for (int i = 0; isRaslPic && i < pic->numSlices; i++)
      {
        if (!(slice->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL || slice->getNalUnitType() == NAL_UNIT_CODED_SLICE_RADL))
        {
          isRaslPic = false;
        }
      }
      if (isRaslPic)
      {
        pic->neededForOutput = false;
      }
    }
  }
#endif
  pic->reconstructed = true;
  Slice::sortPicList( m_cListPic ); // sorting for application output
  poc                     = slice->getPOC();
  rpcListPic              = &m_cListPic;
  m_bFirstSliceInPicture  = true;
  m_maxDecSubPicIdx       = 0;
  m_maxDecSliceAddrInSubPic = -1;
  m_puCounter++;
  m_syncPicture           = false;

  // the picture and its references are in use until the picture is reported
  m_picUse[pic]++;
  for( const auto* ref: job->usedPics )
  {
    m_picUse[ref]++;
  }
  m_jobOfPic[pic] = job;
  m_jobs.push_back( job );
  if( !job->sync )
  {
    m_frameThreads->enqueue( job );
    if( m_frameThreads->isSerial() )
    {
      m_waitHub->wait( job->done );
    }
  }

  // pictures in flight: enough to keep the workers busy, a bounded number of picture buffers
  const int maxInFlight = 2 * m_frameThreads->numThreads();
  for( ;; )
  {
    int inFlight = 0;
    std::shared_ptr<DecPicJob> oldest;
    for( const auto& j: m_jobs )
    {
      if( !j->done.ready.load( std::memory_order_acquire ) )
      {
        inFlight++;
        oldest = oldest ? oldest : j;
      }
    }
    if( inFlight <= maxInFlight )
    {
      break;
    }
    m_waitHub->wait( oldest->done );
  }
  processFinishedPictures( false );
}

void DecLib::xReportPicture( DecPicJob& job )
{
  // the POC line of finishPicture
  const MsgLevel msgl    = INFO;
  Picture*       pic     = job.pic;
  Slice*         pcSlice = pic->cs->slice;

  char c = (pcSlice->isIntra() ? 'I' : pcSlice->isInterP() ? 'P' : 'B');
  if (!job.referenced)
  {
    c += 32;  // tolower
  }
#if JVET_AG0196_CABAC_RETRAIN
#if JVET_AH0176_LOW_DELAY_B_CTX
  CabacRetrain::endFrame(pcSlice->getPOC(), pcSlice->getSliceQp(), pcSlice->getCabacInitFlag(), pcSlice->isIntra() ? I_SLICE : pcSlice->isInterP() ? P_SLICE : pcSlice->isInterB() && pcSlice->getCheckLDC() ? L_SLICE : B_SLICE);
#else
  CabacRetrain::endFrame(pcSlice->getPOC(),pcSlice->getSliceQp(),pcSlice->getCabacInitFlag(),pcSlice->isIntra()?I_SLICE:pcSlice->isInterP()?P_SLICE:B_SLICE);
#endif
#endif
  if (pcSlice->isDRAP()) c = 'D';

  //-- For time output for each slice
  msg( msgl, "POC %4d LId: %2d TId: %1d ( %s, %c-SLICE, QP%3d ) ", pcSlice->getPOC(), pcSlice->getPic()->layerId,
         pcSlice->getTLayer(),
         nalUnitTypeToString(pcSlice->getNalUnitType()),
         c,
         pcSlice->getSliceQp() );
  msg( msgl, "[DT %6.3f] ", pcSlice->getProcessingTime() );

  for (int iRefList = 0; iRefList < 2; iRefList++)
  {
    msg( msgl, "[L%d", iRefList);
    for (int iRefIndex = 0; iRefIndex < pcSlice->getNumRefIdx(RefPicList(iRefList)); iRefIndex++)
    {
      const std::pair<int, int>& scaleRatio = pcSlice->getScalingRatio( RefPicList( iRefList ), iRefIndex );

      if( pcSlice->getPicHeader()->getEnableTMVPFlag() && pcSlice->getColFromL0Flag() == bool(1 - iRefList) && pcSlice->getColRefIdx() == iRefIndex )
      {
        if( scaleRatio.first != 1 << SCALE_RATIO_BITS || scaleRatio.second != 1 << SCALE_RATIO_BITS )
        {
          msg( msgl, " %dc(%1.2lfx, %1.2lfx)", pcSlice->getRefPOC( RefPicList( iRefList ), iRefIndex ), double( scaleRatio.first ) / ( 1 << SCALE_RATIO_BITS ), double( scaleRatio.second ) / ( 1 << SCALE_RATIO_BITS ) );
        }
        else
        {
          msg( msgl, " %dc", pcSlice->getRefPOC( RefPicList( iRefList ), iRefIndex ) );
        }
      }
      else
      {
        if( scaleRatio.first != 1 << SCALE_RATIO_BITS || scaleRatio.second != 1 << SCALE_RATIO_BITS )
        {
          msg( msgl, " %d(%1.2lfx, %1.2lfx)", pcSlice->getRefPOC( RefPicList( iRefList ), iRefIndex ), double( scaleRatio.first ) / ( 1 << SCALE_RATIO_BITS ), double( scaleRatio.second ) / ( 1 << SCALE_RATIO_BITS ) );
        }
        else
        {
          msg( msgl, " %d", pcSlice->getRefPOC( RefPicList( iRefList ), iRefIndex ) );
        }
      }

      if( pcSlice->getRefPOC( RefPicList( iRefList ), iRefIndex ) == pcSlice->getPOC() )
      {
        msg( msgl, ".%d", pcSlice->getRefPic( RefPicList( iRefList ), iRefIndex )->layerId );
      }
    }
    msg( msgl, "] ");
  }
  if (m_decodedPictureHashSEIEnabled)
  {
    SEIMessages pictureHashes = getSeisByType(pic->SEIs, SEI::DECODED_PICTURE_HASH );
    const SEIDecodedPictureHash *hash = ( pictureHashes.size() > 0 ) ? (SEIDecodedPictureHash*) *(pictureHashes.begin()) : NULL;
    if (pictureHashes.size() > 1)
    {
      msg( WARNING, "Warning: Got multiple decoded picture hash SEI messages. Using first.");
    }
    m_numberOfChecksumErrorsDetected += calcAndPrintHashStatus(((const Picture*) pic)->getRecoBuf(), hash, pcSlice->getSPS()->getBitDepths(), msgl);
  }

  msg( msgl, "\n");
}

void DecLib::processFinishedPictures( bool waitAll )
{
  while( !m_jobs.empty() )
  {
    std::shared_ptr<DecPicJob> job = m_jobs.front();
    if( !job->done.ready.load( std::memory_order_acquire ) )
    {
      if( !waitAll )
      {
        break;
      }
      m_waitHub->wait( job->done );
    }
    CHECK( !job->error.empty(), "decoding POC " << job->pic->getPOC() << " failed: " << job->error );
    xReportPicture( *job );
    job->bitstreams.clear();
    job->refJobs.clear();
    releasePicture( job->pic );
    for( const auto* ref: job->usedPics )
    {
      releasePicture( ref );
    }
    job->usedPics.clear();
    m_jobs.pop_front();
  }
  xFreeRetired();
}

bool DecLib::isPictureDecoded( const Picture* pic )
{
  const auto it = m_jobOfPic.find( pic );
  if( it == m_jobOfPic.end() || !it->second )
  {
    return true;
  }
  const bool done = it->second->done.ready.load( std::memory_order_acquire );
  // a picture whose decoding failed is not output: the error is reported, as sequential decoding stops there
  CHECK( done && !it->second->error.empty(), "decoding POC " << pic->getPOC() << " failed: " << it->second->error );
  return done;
}

void DecLib::waitPictureDecoded( const Picture* pic )
{
  const auto it = m_jobOfPic.find( pic );
  if( it != m_jobOfPic.end() && it->second )
  {
    m_waitHub->wait( it->second->done );
  }
}

void DecLib::releasePicture( const Picture* pic )
{
  auto it = m_picUse.find( pic );
  CHECK( it == m_picUse.end() || it->second <= 0, "picture released more often than held" );
  if( --it->second == 0 )
  {
    m_picUse.erase( it );
  }
}

void DecLib::retirePicture( Picture* pic )
{
  // out of the picture list (the output flushes it); deleted when no job or pending output uses it
  m_retiredPics.push_back( pic );
  xFreeRetired();
}

void DecLib::xFreeRetired()
{
  for( size_t i = 0; i < m_retiredPics.size(); )
  {
    Picture* pic = m_retiredPics[i];
    if( m_picUse.find( pic ) == m_picUse.end() )
    {
      m_jobOfPic.erase( pic );
      pic->destroy();
      delete pic;
      m_retiredPics.erase( m_retiredPics.begin() + i );
    }
    else
    {
      i++;
    }
  }
}

void DecLib::drainFrameThreads()
{
  if( m_frameThreads )
  {
    processFinishedPictures( true );
  }
}

//! \}
