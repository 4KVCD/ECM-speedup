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

/** \file     DecSegmentParallel.cpp
    \brief    Bit-exact parallel decoding of random-access segments
*/

#include "DecSegmentParallel.h"

#include "CommonLib/CommonDef.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
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

//! \ingroup DecoderApp
//! \{

namespace
{

// ---------------------------------------------------------------------------------------------------------------------
// child processes (same as in EncoderApp/SegmentParallel.cpp)

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
// stream structure from the NAL unit headers

struct Nal
{
  size_t start   = 0;       // first byte of the start code, including the zero bytes before it
  size_t payload = 0;       // first byte of the NAL unit header
  int    type    = -1;
  int    layerId = 0;
  bool   phInSh  = false;   // slices: sh_picture_header_in_slice_header_flag
};

struct Pic
{
  int  puStart = 0;   // first NAL unit of its picture unit
  int  type    = 0;   // NAL unit type of its slices
  bool mixed   = false;
};

// a segment start: decoding starts at picture `skip` (SkipFrames) and the first `overlap` decoded pictures, those
// before picture `pic`, are references only
struct SplitPoint
{
  int pic;
  int skip;
  int overlap;
};

bool isVcl( int t )
{
  return t >= NAL_UNIT_CODED_SLICE_TRAIL && t <= NAL_UNIT_RESERVED_IRAP_VCL_11;
}

bool isIrap( const Pic& p )
{
  return !p.mixed && ( p.type == NAL_UNIT_CODED_SLICE_IDR_W_RADL || p.type == NAL_UNIT_CODED_SLICE_IDR_N_LP
                       || p.type == NAL_UNIT_CODED_SLICE_CRA );
}

std::vector<Nal> scanNals( const std::vector<uint8_t>& d )
{
  std::vector<Nal> nals;
  for( size_t i = 2; i < d.size(); i++ )
  {
    if( d[i] == 1 && d[i - 1] == 0 && d[i - 2] == 0 )
    {
      // NAL units do not end with a zero byte (cabac_zero_words get an 0x03 appended), so zero bytes before a start
      // code prefix belong to it
      Nal nal;
      nal.start = i - 2;
      while( nal.start > ( nals.empty() ? 0 : nals.back().payload ) && d[nal.start - 1] == 0 )
      {
        nal.start--;
      }
      nal.payload = i + 1;
      nals.push_back( nal );
      i += 2;
    }
  }
  for( size_t k = 0; k < nals.size(); k++ )
  {
    Nal&         n   = nals[k];
    const size_t end = k + 1 < nals.size() ? nals[k + 1].start : d.size();
    if( end < n.payload + 2 )
    {
      continue;   // type -1: damaged
    }
    n.layerId = d[n.payload] & 0x3f;
    n.type    = d[n.payload + 1] >> 3;
    n.phInSh  = isVcl( n.type ) && end > n.payload + 2 && ( d[n.payload + 2] & 0x80 ) != 0;
  }
  return nals;
}

// pictures in decoding order; false (with the reason) for streams that are not split
bool scanPictures( const std::vector<Nal>& nals, std::vector<Pic>& pics, std::string& reason )
{
  int  pendingPuStart = 0;   // first NAL unit of the next picture unit, -1: none since the last slice
  bool phPending      = false;
  int  lastVcl = -1, firstEndOfStream = -1;
  for( int k = 0; k < (int) nals.size(); k++ )
  {
    const Nal& n = nals[k];
    if( n.type < 0 || n.layerId != 0 )
    {
      reason = n.type < 0 ? "damaged NAL unit" : "more than one layer";
      return false;
    }
    if( !isVcl( n.type ) )
    {
      const bool suffix = n.type == NAL_UNIT_SUFFIX_APS || n.type == NAL_UNIT_SUFFIX_SEI || n.type == NAL_UNIT_FD
                          || n.type == NAL_UNIT_EOS || n.type == NAL_UNIT_EOB;
      if( !suffix && pendingPuStart < 0 )
      {
        pendingPuStart = k;
      }
      if( ( n.type == NAL_UNIT_EOS || n.type == NAL_UNIT_EOB ) && firstEndOfStream < 0 )
      {
        firstEndOfStream = k;
      }
      phPending |= n.type == NAL_UNIT_PH;
      continue;
    }
    if( n.type == NAL_UNIT_CODED_SLICE_GDR || ( n.type >= NAL_UNIT_RESERVED_VCL_4 && n.type <= NAL_UNIT_RESERVED_VCL_6 )
        || n.type == NAL_UNIT_RESERVED_IRAP_VCL_11 )
    {
      reason = n.type == NAL_UNIT_CODED_SLICE_GDR ? "GDR pictures" : "reserved NAL unit types";
      return false;
    }
    lastVcl = k;
    // a picture unit starts with the first non-suffix NAL unit after the last slice of the previous picture: one between
    // two slices of a picture (a prefix SEI or APS there is legal) belongs to that picture
    const int puStart = pendingPuStart >= 0 ? pendingPuStart : k;
    pendingPuStart    = -1;
    if( n.phInSh || phPending )
    {
      Pic p;
      p.puStart = puStart;
      p.type    = n.type;
      pics.push_back( p );
      phPending = false;
    }
    else if( pics.empty() )
    {
      reason = "slice without picture header";
      return false;
    }
    else if( n.type != pics.back().type )
    {
      pics.back().mixed = true;
    }
  }
  if( firstEndOfStream >= 0 && firstEndOfStream < lastVcl )
  {
    reason = "end of sequence or bitstream NAL units inside the stream";
    return false;
  }
  if( pics.empty() )
  {
    reason = "no pictures";
    return false;
  }
  return true;
}

// pictures where a segment can start: CRAs. A CRA without RASL pictures starts its own segment; one with RASL
// pictures is reached from the previous IRAP, which must be the first picture or a CRA (the random-access process
// skips the RASL pictures of that CRA). IDRs after the first picture are not used: SkipFrames does not resume at them.
std::vector<SplitPoint> splitPoints( const std::vector<Pic>& pics )
{
  std::vector<SplitPoint> points;
  const int               n        = (int) pics.size();
  int                     prevIrap = isIrap( pics[0] ) ? 0 : -1;
  for( int p = 1; p < n; p++ )
  {
    if( !isIrap( pics[p] ) )
    {
      continue;
    }
    // leading pictures: RASL ones need the pictures before the CRA; RADL ones too in ECM, which resets decode-order
    // state (temporal CABAC contexts, SbTMVP area, CCSAO and chroma ALF history) only at the first trailing picture
    bool rasl = false;
    for( int q = p + 1; q < n && !isIrap( pics[q] ); q++ )
    {
      rasl |= pics[q].type == NAL_UNIT_CODED_SLICE_RASL || pics[q].type == NAL_UNIT_CODED_SLICE_RADL || pics[q].mixed;
    }
    if( pics[p].type == NAL_UNIT_CODED_SLICE_CRA )
    {
      if( !rasl )
      {
        points.push_back( { p, p, 0 } );
      }
      else if( prevIrap == 0 || ( prevIrap > 0 && pics[prevIrap].type == NAL_UNIT_CODED_SLICE_CRA ) )
      {
        int overlap = 0;
        for( int q = prevIrap; q < p; q++ )
        {
          overlap += pics[q].type != NAL_UNIT_CODED_SLICE_RASL;   // the RASL pictures of prevIrap are skipped
        }
        points.push_back( { p, prevIrap, overlap } );
      }
    }
    prevIrap = p;
  }
  return points;
}

struct Segment
{
  int         first = 0, end = 0;   // pictures [first, end) in decoding order
  int         skip = 0, overlap = 0;
  std::string bitstream, recon, log;
  bool        tempBitstream = false;
  int         exitCode      = -1;
};

} // namespace

int runDecSegmentParallel( int argc, char* argv[], const DecSegmentParallelCfg& cfg, std::string& reason )
{
  const auto t0 = std::chrono::steady_clock::now();

  std::vector<uint8_t> data;
  if( !readFile( cfg.bitstreamFile, data ) )
  {
    reason = "cannot read " + cfg.bitstreamFile;
    return -1;
  }
  const std::vector<Nal> nals = scanNals( data );
  std::vector<Pic>       pics;
  if( !scanPictures( nals, pics, reason ) )
  {
    return -1;
  }
  if( !cfg.reconFile.empty() && !cfg.outputBitDepthGiven )
  {
    // without -d every child writes the bit depth of its first picture: the outputs join only if all SPSs are the same
    std::vector<uint8_t> firstSps;
    for( size_t k = 0; k < nals.size(); k++ )
    {
      if( nals[k].type != NAL_UNIT_SPS )
      {
        continue;
      }
      const size_t         end = k + 1 < nals.size() ? nals[k + 1].start : data.size();
      std::vector<uint8_t> sps( data.begin() + nals[k].payload, data.begin() + end );
      if( firstSps.empty() )
      {
        firstSps = sps;
      }
      else if( sps != firstSps )
      {
        reason = "SPSs that differ (the output bit depth may change; set it with -d)";
        return -1;
      }
    }
  }
  const std::vector<SplitPoint> points = splitPoints( pics );

  // segment starts: the split points closest to an even division of the pictures
  const int               numPics = (int) pics.size();
  const int               numSeg  = std::min<int>( cfg.numWorkers, (int) points.size() + 1 );
  std::vector<SplitPoint> chosen;
  size_t                  from = 0;
  for( int k = 1; k < numSeg; k++ )
  {
    const double ideal = (double) k * numPics / numSeg;
    size_t       best  = points.size();
    for( size_t i = from; i < points.size(); i++ )
    {
      if( best < points.size() && std::abs( points[i].pic - ideal ) >= std::abs( points[best].pic - ideal ) )
      {
        break;   // the distance only grows from here
      }
      best = i;
    }
    if( best == points.size() )
    {
      break;
    }
    chosen.push_back( points[best] );
    from = best + 1;
  }
  if( chosen.empty() )
  {
    reason = "no picture to start a segment at (one random access point, or IDRs only)";
    return -1;
  }

  // the temporary files: names with the process ID, so that concurrent decodes of a stream do not share them
#ifdef _WIN32
  const unsigned long pid = GetCurrentProcessId();
#else
  const unsigned long pid = (unsigned long) getpid();
#endif
  const std::string    base = ( cfg.reconFile.empty() ? cfg.bitstreamFile : cfg.reconFile ) + "." + std::to_string( pid );
  std::vector<Segment> segs( chosen.size() + 1 );
  auto cleanUp = [&]( bool keepLogs ) {
    for( auto& seg : segs )
    {
      if( seg.tempBitstream )
      {
        remove( seg.bitstream.c_str() );
      }
      if( !seg.recon.empty() )
      {
        remove( seg.recon.c_str() );
      }
      if( !keepLogs && !seg.log.empty() )
      {
        remove( seg.log.c_str() );
      }
    }
  };
  for( size_t k = 0; k < segs.size(); k++ )
  {
    Segment& seg = segs[k];
    seg.first    = k ? chosen[k - 1].pic : 0;
    seg.end      = k < chosen.size() ? chosen[k].pic : numPics;
    seg.skip     = k ? chosen[k - 1].skip : 0;
    seg.overlap  = k ? chosen[k - 1].overlap : 0;
    const std::string tag = base + ".seg" + std::to_string( k );
    seg.log   = tag + ".log";
    seg.recon = cfg.reconFile.empty() ? "" : tag + ".yuv";
    if( seg.end < numPics )
    {
      // the stream up to the picture unit of the next segment's first picture
      seg.bitstream     = tag + ".bin";
      seg.tempBitstream = true;
      std::ofstream f( seg.bitstream, std::ios::binary );
      f.write( reinterpret_cast<const char*>( data.data() ), nals[pics[seg.end].puStart].start );
      f.close();
      if( !f )
      {
        reason = "cannot write " + seg.bitstream;
        cleanUp( false );
        return -1;
      }
    }
    else
    {
      seg.bitstream = cfg.bitstreamFile;
    }
  }

  printf( "\nSegment-parallel decoding: %d pictures, %d segments on %d workers\n", numPics, (int) segs.size(),
          std::min<int>( cfg.numWorkers, (int) segs.size() ) );
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
      args.push_back( "--ParallelSegments=0" );
      args.push_back( "--BitstreamFile=" + seg.bitstream );
      if( !seg.recon.empty() )
      {
        args.push_back( "--ReconFile=" + seg.recon );
      }
      args.push_back( "--SkipFrames=" + std::to_string( seg.skip ) );
      args.push_back( "--SegmentOverlapPictures=" + std::to_string( seg.overlap ) );
      seg.exitCode = runProcess( args, seg.log );
      std::lock_guard<std::mutex> lock( ioMutex );
      printf( "  segment %2d: pictures %5d..%5d (decoding order), %d reference-only  %s\n", k, seg.first, seg.end - 1,
              seg.overlap, seg.exitCode == 0 ? "done" : "FAILED" );
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
      reason = "segment " + std::to_string( k ) + " failed (exit code " + std::to_string( segs[k].exitCode )
               + "), see " + segs[k].log;
      cleanUp( true );
      return -1;
    }
  }

  // join the outputs
  if( !cfg.reconFile.empty() )
  {
    std::ofstream f( cfg.reconFile, std::ios::binary );
    for( auto& seg : segs )
    {
      std::ifstream in( seg.recon, std::ios::binary );
      if( in.peek() != std::ifstream::traits_type::eof() )
      {
        f << in.rdbuf();
      }
    }
    if( !f )
    {
      reason = "cannot write " + cfg.reconFile;
      cleanUp( true );
      return -1;
    }
  }

  // per-picture lines in decoding order, without those of the reference-only pictures
  int numDecoded = 0;
  printf( "\n" );
  for( auto& seg : segs )
  {
    std::ifstream f( seg.log );
    std::string   line;
    int           skipLines = seg.overlap;
    while( std::getline( f, line ) )
    {
      if( line.compare( 0, 4, "POC " ) == 0 && skipLines-- <= 0 )
      {
        printf( "%s\n", line.c_str() );
        numDecoded++;
      }
    }
  }
  cleanUp( false );
  const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
  printf( "\n%d pictures decoded in %.3f s (elapsed) by %d segments\n", numDecoded, secs, (int) segs.size() );
  return 0;
}

//! \}
