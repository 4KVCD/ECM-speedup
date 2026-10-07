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

/** \file     SegmentParallel.cpp
    \brief    Bit-exact parallel encoding of intra-period segments
*/

#include "SegmentParallel.h"

#include "CommonLib/CommonDef.h"
#include "DecoderLib/NALread.h"
#include "DecoderLib/VLCReader.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
extern char** environ;
#endif

//! \ingroup EncoderApp
//! \{

namespace
{

struct Segment
{
  int         start;   // first frame, relative to the first encoded frame of the full run
  int         frames;  // frames coded by this chunk (the first one duplicates the previous chunk's last)
  std::string bitstream;
  std::string recon;
  std::string log;
  int         exitCode = -1;
};

// ---------------------------------------------------------------------------------------------------------------------
// child processes

std::string selfExecutable( const char* argv0 )
{
#ifdef _WIN32
  char buf[MAX_PATH * 4];
  DWORD n = GetModuleFileNameA( nullptr, buf, sizeof( buf ) );
  if( n > 0 && n < sizeof( buf ) )
  {
    return std::string( buf, n );
  }
#else
  char buf[4096];
  ssize_t n = readlink( "/proc/self/exe", buf, sizeof( buf ) - 1 );
  if( n > 0 )
  {
    return std::string( buf, n );
  }
#endif
  return argv0;
}

#ifdef _WIN32
// quoting per the MSVC/CommandLineToArgvW rules
std::string quoteWinArg( const std::string& a )
{
  if( !a.empty() && a.find_first_of( " \t\n\v\"" ) == std::string::npos )
  {
    return a;
  }
  std::string r = "\"";
  for( size_t i = 0; ; i++ )
  {
    size_t numBackslashes = 0;
    while( i < a.size() && a[i] == '\\' )
    {
      i++;
      numBackslashes++;
    }
    if( i == a.size() )
    {
      r.append( numBackslashes * 2, '\\' );
      break;
    }
    if( a[i] == '"' )
    {
      r.append( numBackslashes * 2 + 1, '\\' );
    }
    else
    {
      r.append( numBackslashes, '\\' );
    }
    r += a[i];
  }
  return r + "\"";
}
#endif

int runProcess( const std::vector<std::string>& args, const std::string& logFile )
{
#ifdef _WIN32
  std::string cmd;
  for( const auto& a : args )
  {
    cmd += ( cmd.empty() ? "" : " " ) + quoteWinArg( a );
  }
  SECURITY_ATTRIBUTES sa = { sizeof( sa ), nullptr, TRUE };
  HANDLE hLog = CreateFileA( logFile.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr );
  if( hLog == INVALID_HANDLE_VALUE )
  {
    return -1;
  }
  STARTUPINFOA si = {};
  si.cb         = sizeof( si );
  si.dwFlags    = STARTF_USESTDHANDLES;
  si.hStdInput  = GetStdHandle( STD_INPUT_HANDLE );
  si.hStdOutput = hLog;
  si.hStdError  = hLog;
  PROCESS_INFORMATION pi = {};
  std::vector<char> cmdBuf( cmd.begin(), cmd.end() );
  cmdBuf.push_back( 0 );
  BOOL ok = CreateProcessA( nullptr, cmdBuf.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi );
  CloseHandle( hLog );
  if( !ok )
  {
    return -1;
  }
  WaitForSingleObject( pi.hProcess, INFINITE );
  DWORD code = 1;
  GetExitCodeProcess( pi.hProcess, &code );
  CloseHandle( pi.hThread );
  CloseHandle( pi.hProcess );
  return (int) code;
#else
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init( &fa );
  posix_spawn_file_actions_addopen( &fa, 1, logFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644 );
  posix_spawn_file_actions_adddup2( &fa, 1, 2 );
  std::vector<char*> cargs;
  for( const auto& a : args )
  {
    cargs.push_back( const_cast<char*>( a.c_str() ) );
  }
  cargs.push_back( nullptr );
  pid_t pid;
  int err = posix_spawn( &pid, cargs[0], &fa, nullptr, cargs.data(), environ );
  posix_spawn_file_actions_destroy( &fa );
  if( err )
  {
    return -1;
  }
  int status = 0;
  waitpid( pid, &status, 0 );
  return WIFEXITED( status ) ? WEXITSTATUS( status ) : -1;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// joining chunk bitstreams (derived from App/Parcat, with POC rewriting done on the RBSP so that emulation
// prevention is re-derived exactly as NALwrite.cpp does, and the POC LSB length taken from the SPS)

class PocHeaderReader : public VLCReader
{
public:
  bool pictureHeaderInSliceHeader()
  {
    uint32_t uiCode;
    READ_FLAG( uiCode, "picture_header_in_slice_header_flag" );
    return uiCode == 1;
  }
  // parses the picture header up to (excluding) ph_pic_order_cnt_lsb, returns the SPS
  const SPS* pictureHeaderUpToPoc( ParameterSetManager& psm )
  {
    uint32_t uiCode;
#if JVET_S0076_ASPECT1
    uint32_t uiTmp;
    READ_FLAG( uiTmp, "gdr_or_irap_pic_flag" );
    READ_FLAG( uiCode, "ph_non_ref_pic_flag" );
    if( uiTmp )
    {
      READ_FLAG( uiCode, "gdr_pic_flag" );
    }
#else
    READ_FLAG( uiCode, "gdr_or_irap_pic_flag" );
    if( uiCode )
    {
      READ_FLAG( uiCode, "gdr_pic_flag" );
    }
#endif
    READ_FLAG( uiCode, "ph_inter_slice_allowed_flag" );
    if( uiCode )
    {
      READ_FLAG( uiCode, "ph_intra_slice_allowed_flag" );
    }
#if !JVET_S0076_ASPECT1
    READ_FLAG( uiCode, "non_reference_picture_flag" );
#endif
    READ_UVLC( uiCode, "ph_pic_parameter_set_id" );
    const PPS* pps = psm.getPPS( uiCode );
    CHECK( pps == nullptr, "Invalid PPS" );
    const SPS* sps = psm.getSPS( pps->getSPSId() );
    CHECK( sps == nullptr, "Invalid SPS" );
    return sps;
  }
};

// NAL unit bytes (header + payload, emulation prevention included) -> header + RBSP
std::vector<uint8_t> unescapeNal( const std::vector<uint8_t>& nal )
{
  std::vector<uint8_t> out;
  out.reserve( nal.size() );
  int zeros = 0;
  for( size_t i = 0; i < nal.size(); i++ )
  {
    const uint8_t v = nal[i];
    if( i >= 2 && zeros == 2 && v == 3 )
    {
      zeros = 0;
      continue;
    }
    zeros = ( i >= 2 && v == 0 ) ? zeros + 1 : 0;
    out.push_back( v );
  }
  return out;
}

// header + RBSP -> NAL unit bytes, identical to NALwrite.cpp
std::vector<uint8_t> escapeNal( const std::vector<uint8_t>& rbsp )
{
  std::vector<uint8_t> out( rbsp.begin(), rbsp.begin() + 2 );
  out.reserve( rbsp.size() * 2 + 1 );
  int zeros = 0;
  for( size_t i = 2; i < rbsp.size(); i++ )
  {
    const uint8_t v = rbsp[i];
    if( zeros == 2 && v <= 3 )
    {
      out.push_back( 3 );
      zeros = 0;
    }
    zeros = v == 0 ? zeros + 1 : 0;
    out.push_back( v );
  }
  if( zeros > 0 )
  {
    out.push_back( 3 );
  }
  return out;
}

void writeBits( std::vector<uint8_t>& buf, int bitPos, int numBits, uint32_t value )
{
  for( int i = 0; i < numBits; i++ )
  {
    const int      b   = bitPos + i;
    const uint8_t  bit = ( value >> ( numBits - 1 - i ) ) & 1;
    const uint8_t  msk = 0x80 >> ( b & 7 );
    buf[b >> 3]        = bit ? ( buf[b >> 3] | msk ) : ( buf[b >> 3] & ~msk );
  }
}

uint32_t readBits( const std::vector<uint8_t>& buf, int bitPos, int numBits )
{
  uint32_t v = 0;
  for( int i = 0; i < numBits; i++ )
  {
    const int b = bitPos + i;
    v           = ( v << 1 ) | ( ( buf[b >> 3] >> ( 7 - ( b & 7 ) ) ) & 1 );
  }
  return v;
}

// Appends chunk 'idx' (POC base 'pocBase') to 'out'.  For idx > 0 the leading IDR access unit is dropped
// (parameter sets, picture header and APSs before the IDR slice, the IDR slice and its suffix SEI).
bool appendSegment( const std::vector<uint8_t>& in, int idx, int pocBase, std::vector<uint8_t>& out, std::string& err )
{
  ParameterSetManager psm;
  bool                idrFound = false;
  bool                skipNextSuffixSei = false;
  const size_t        n        = in.size();

  auto findStart = [&]( size_t from, size_t& prefixStart, size_t& payloadStart ) {
    for( size_t i = from; i + 3 <= n; i++ )
    {
      if( in[i] == 0 && in[i + 1] == 0 && in[i + 2] == 1 )
      {
        prefixStart  = ( i > from && in[i - 1] == 0 ) ? i - 1 : i;
        payloadStart = i + 3;
        return true;
      }
    }
    return false;
  };

  size_t prefixStart, payloadStart;
  if( !findStart( 0, prefixStart, payloadStart ) )
  {
    err = "no NAL unit found";
    return false;
  }
  while( true )
  {
    // NAL ends at the next 0x000000 / 0x000001 (trailing zero bytes belong to the next start code)
    size_t end = payloadStart;
    while( end + 3 <= n && !( in[end] == 0 && in[end + 1] == 0 && ( in[end + 2] == 0 || in[end + 2] == 1 ) ) )
    {
      end++;
    }
    if( end + 3 > n )
    {
      end = n;
    }
    std::vector<uint8_t> nal( in.begin() + payloadStart, in.begin() + end );
    const std::vector<uint8_t> prefix( in.begin() + prefixStart, in.begin() + payloadStart );
    const int nalType = nal.size() > 1 ? nal[1] >> 3 : -1;

    InputNALUnit inp;
    inp.getBitstream().getFifo() = nal;
    read( inp );

    if( inp.m_nalUnitType == NAL_UNIT_SPS || inp.m_nalUnitType == NAL_UNIT_PPS )
    {
      HLSyntaxReader hls;
      hls.setBitstream( &inp.getBitstream() );
      if( inp.m_nalUnitType == NAL_UNIT_SPS )
      {
        SPS* sps = new SPS();
        hls.parseSPS( sps );
        psm.storeSPS( sps, inp.getBitstream().getFifo() );
      }
      else
      {
        PPS* pps = new PPS();
        hls.parsePPS( pps );
        psm.storePPS( pps, inp.getBitstream().getFifo() );
      }
    }

    const bool isIdr = nalType == NAL_UNIT_CODED_SLICE_IDR_W_RADL || nalType == NAL_UNIT_CODED_SLICE_IDR_N_LP;
    const bool isOtherVcl = nalType >= 0 && ( nalType < NAL_UNIT_CODED_SLICE_IDR_W_RADL
                                              || ( nalType > NAL_UNIT_CODED_SLICE_IDR_N_LP && nalType <= NAL_UNIT_RESERVED_IRAP_VCL_12 ) );
    if( inp.m_nalUnitType == NAL_UNIT_PH || isOtherVcl )
    {
      PocHeaderReader rd;
      rd.setBitstream( &inp.getBitstream() );
      if( inp.m_nalUnitType == NAL_UNIT_PH || rd.pictureHeaderInSliceHeader() )
      {
        const SPS* sps     = rd.pictureHeaderUpToPoc( psm );
        const int  bitsPoc = sps->getBitsForPOC();
        const int  bitPos  = rd.getBitstream()->getNumBitsRead();   // counted from the start of the NAL header
        std::vector<uint8_t> rbsp = unescapeNal( nal );
        if( escapeNal( rbsp ) != nal )
        {
          err = "emulation prevention round trip failed";
          return false;
        }
        const uint32_t mask   = ( 1u << bitsPoc ) - 1;
        const uint32_t oldLsb = readBits( rbsp, bitPos, bitsPoc );
        writeBits( rbsp, bitPos, bitsPoc, ( oldLsb + (uint32_t) pocBase ) & mask );
        nal = escapeNal( rbsp );
      }
    }

    bool drop = false;
    if( idx > 0 )
    {
      if( isIdr )
      {
        drop = idrFound = skipNextSuffixSei = true;
      }
      else if( !idrFound && ( nalType == NAL_UNIT_DCI || nalType == NAL_UNIT_VPS || nalType == NAL_UNIT_SPS || nalType == NAL_UNIT_PPS
                              || nalType == NAL_UNIT_PREFIX_APS || nalType == NAL_UNIT_SUFFIX_APS || nalType == NAL_UNIT_PH
                              || nalType == NAL_UNIT_ACCESS_UNIT_DELIMITER ) )
      {
        drop = true;
      }
      else if( nalType == NAL_UNIT_SUFFIX_SEI && skipNextSuffixSei )
      {
        drop              = true;
        skipNextSuffixSei = false;
      }
    }
    if( !drop )
    {
      out.insert( out.end(), prefix.begin(), prefix.end() );
      out.insert( out.end(), nal.begin(), nal.end() );
    }

    if( end >= n || !findStart( end, prefixStart, payloadStart ) )
    {
      break;
    }
  }
  if( idx > 0 && !idrFound )
  {
    err = "chunk does not start with an IDR picture";
    return false;
  }
  return true;
}

bool readFile( const std::string& path, std::vector<uint8_t>& data )
{
  std::ifstream f( path, std::ios::binary );
  if( !f )
  {
    return false;
  }
  data.assign( std::istreambuf_iterator<char>( f ), std::istreambuf_iterator<char>() );
  return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// log merging: re-number the per-picture lines of each chunk and accumulate a summary

struct PicStats
{
  int    frames = 0;
  double bits   = 0;
  double psnr[3] = { 0, 0, 0 };
};

std::string rebasePocLine( const std::string& line, int base, int& poc, bool& ok, double& bits, double psnr[3] )
{
  ok = false;
  int p;
  if( sscanf( line.c_str(), "POC %d", &p ) != 1 )
  {
    return line;
  }
  poc = p + base;
  ok  = true;
  std::ostringstream os;
  char head[32];
  snprintf( head, sizeof( head ), "POC %4d", poc );
  os << head;
  size_t i = line.find_first_not_of( ' ', 3 );
  i        = line.find_first_of( ' ', i );   // skip the number
  std::string rest = line.substr( i );
  // rebase the reference picture lists "[L0 a b c] [L1 ...]"
  for( const char* lst : { "[L0", "[L1" } )
  {
    size_t s = rest.find( lst );
    if( s == std::string::npos )
    {
      continue;
    }
    size_t e = rest.find( ']', s );
    std::istringstream is( rest.substr( s + 3, e - s - 3 ) );
    std::string tok, rebuilt = lst;
    while( is >> tok )
    {
      bool c = !tok.empty() && tok.back() == 'c';
      int  v = atoi( tok.c_str() );
      rebuilt += " " + std::to_string( v + base ) + ( c ? "c" : "" );
    }
    rest.replace( s, e - s, rebuilt );
  }
  os << rest;
  bits = 0;
  sscanf( rest.c_str() + rest.find( ')' ) + 1, "%lf bits", &bits );
  size_t y = rest.find( "[Y " );
  if( y != std::string::npos )
  {
    sscanf( rest.c_str() + y, "[Y %lf dB U %lf dB V %lf dB", &psnr[0], &psnr[1], &psnr[2] );
  }
  return os.str();
}

}   // namespace

// ---------------------------------------------------------------------------------------------------------------------

bool segmentParallelSupported( const SegmentParallelCfg& cfg, int decodingRefreshType, int maxLayers, std::string& reason )
{
  if( cfg.intraPeriod <= 0 )
  {
    reason = "needs a periodic intra refresh (IntraPeriod > 0)";
  }
  else if( decodingRefreshType != 1 )
  {
    reason = "needs CRA random access points (DecodingRefreshType = 1)";
  }
  else if( maxLayers != 1 )
  {
    reason = "single-layer coding only";
  }
  else if( cfg.bitstreamFile.empty() )
  {
    reason = "needs a bitstream file";
  }
  else if( cfg.framesToBeEncoded <= cfg.intraPeriod + 1 && cfg.segmentLength == 0 )
  {
    reason = "sequence has only one intra period";
  }
  else
  {
    return true;
  }
  return false;
}

int runSegmentParallel( int argc, char* argv[], const SegmentParallelCfg& cfg )
{
  const auto t0 = std::chrono::steady_clock::now();
  const int  P  = cfg.intraPeriod;
  const int  F  = cfg.framesToBeEncoded;

  int L = cfg.segmentLength;
  if( L <= 0 )
  {
    const int numPeriods = ( F - 1 + P - 1 ) / P;
    L                    = P * std::max( 1, ( numPeriods + cfg.numWorkers - 1 ) / cfg.numWorkers );
  }
  CHECK( L % P != 0, "SegmentLength must be a multiple of the intra period" );

  std::vector<Segment> segs;
  for( int s = 0; s < F - 1 || ( s == 0 && F == 1 ); s += L )
  {
    Segment seg;
    seg.start     = s;
    seg.frames    = std::min( L + 1, F - s );
    const std::string tag = cfg.bitstreamFile + ".seg" + std::to_string( segs.size() );
    seg.bitstream = tag + ".bin";
    seg.log       = tag + ".log";
    seg.recon     = cfg.reconFile.empty() ? "" : tag + ".yuv";
    segs.push_back( seg );
  }

  printf( "\nSegment-parallel encoding: %d frames, intra period %d, %d chunks of up to %d frames, %d workers\n", F, P,
          (int) segs.size(), L + 1, cfg.numWorkers );
  fflush( stdout );

  const std::string exe = selfExecutable( argv[0] );
  std::atomic<int>  next( 0 );
  std::mutex        ioMutex;
  auto worker = [&]() {
    for( int k = next++; k < (int) segs.size(); k = next++ )
    {
      Segment&                 seg = segs[k];
      std::vector<std::string> args( argv, argv + argc );
      args[0] = exe;
      // coded frame i is input frame FrameSkip + i * TemporalSubsampleRatio; FramesToBeEncoded counts input frames
      const int tsr = cfg.temporalSubsampleRatio;
      args.push_back( "--ParallelSegments=1" );
      args.push_back( "--FrameSkip=" + std::to_string( cfg.frameSkip + seg.start * tsr ) );
      args.push_back( "--FramesToBeEncoded=" + std::to_string( seg.frames * tsr ) );
      args.push_back( "--BitstreamFile=" + seg.bitstream );
      args.push_back( "--ReconFile=" + seg.recon );
      seg.exitCode = runProcess( args, seg.log );
      std::lock_guard<std::mutex> lock( ioMutex );
      printf( "  chunk %2d: frames %4d..%4d  %s\n", k, seg.start, seg.start + seg.frames - 1,
              seg.exitCode == 0 ? "done" : "FAILED" );
      fflush( stdout );
    }
  };
  std::vector<std::thread> pool;
  for( int i = 0; i < std::min<int>( cfg.numWorkers, (int) segs.size() ); i++ )
  {
    pool.emplace_back( worker );
  }
  for( auto& t : pool )
  {
    t.join();
  }
  for( size_t k = 0; k < segs.size(); k++ )
  {
    if( segs[k].exitCode != 0 )
    {
      fprintf( stderr, "chunk %d failed with exit code %d, see %s\n", (int) k, segs[k].exitCode, segs[k].log.c_str() );
      return 1;
    }
  }

  // join bitstreams
  std::vector<uint8_t> joined;
  for( size_t k = 0; k < segs.size(); k++ )
  {
    std::vector<uint8_t> data;
    std::string          err;
    if( !readFile( segs[k].bitstream, data ) || !appendSegment( data, (int) k, segs[k].start, joined, err ) )
    {
      fprintf( stderr, "joining chunk %d failed: %s\n", (int) k, err.empty() ? "cannot read bitstream" : err.c_str() );
      return 1;
    }
  }
  {
    std::ofstream f( cfg.bitstreamFile, std::ios::binary );
    f.write( reinterpret_cast<const char*>( joined.data() ), joined.size() );
    if( !f )
    {
      fprintf( stderr, "cannot write %s\n", cfg.bitstreamFile.c_str() );
      return 1;
    }
  }

  // join reconstructions (every chunk after the first repeats the previous chunk's last frame)
  if( !cfg.reconFile.empty() )
  {
    std::ofstream f( cfg.reconFile, std::ios::binary );
    for( size_t k = 0; k < segs.size(); k++ )
    {
      std::vector<uint8_t> data;
      readFile( segs[k].recon, data );
      const size_t frameBytes = data.size() / segs[k].frames;
      const size_t skip       = k ? frameBytes : 0;
      f.write( reinterpret_cast<const char*>( data.data() ) + skip, data.size() - skip );
    }
  }

  // per-picture lines in coding order, summary
  PicStats stats;
  printf( "\n" );
  for( size_t k = 0; k < segs.size(); k++ )
  {
    std::ifstream f( segs[k].log );
    std::string   line;
    bool          first = true;
    while( std::getline( f, line ) )
    {
      int    poc;
      bool   ok;
      double bits, psnr[3] = { 0, 0, 0 };
      std::string l = rebasePocLine( line, segs[k].start, poc, ok, bits, psnr );
      if( !ok )
      {
        continue;
      }
      if( k > 0 && first )   // the duplicated leading IRAP
      {
        first = false;
        continue;
      }
      first = false;
      printf( "%s\n", l.c_str() );
      stats.frames++;
      stats.bits += bits;
      for( int c = 0; c < 3; c++ )
      {
        stats.psnr[c] += psnr[c];
      }
    }
  }
  const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
  if( stats.frames )
  {
    printf( "\n\tTotal Frames |  Bitrate      Y-PSNR   U-PSNR   V-PSNR\n" );
    printf( "\t%-12d a %10.4f  %8.4f %8.4f %8.4f\n", stats.frames,
            stats.bits * cfg.frameRate / cfg.temporalSubsampleRatio / stats.frames / 1000.0,
            stats.psnr[0] / stats.frames, stats.psnr[1] / stats.frames, stats.psnr[2] / stats.frames );
  }
  printf( "\nBitstream: %s (%zu bytes)\n", cfg.bitstreamFile.c_str(), joined.size() );
  printf( " Total Time: %12.3f sec. [elapsed, %d chunks on %d workers]\n", secs, (int) segs.size(), cfg.numWorkers );

  for( auto& seg : segs )
  {
    remove( seg.bitstream.c_str() );
    remove( seg.log.c_str() );
    if( !seg.recon.empty() )
    {
      remove( seg.recon.c_str() );
    }
  }
  return 0;
}

//! \}
