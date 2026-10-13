/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2025, ITU/ISO/IEC
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

/** \file     CrashReport.cpp
    \brief    on Windows, a crash prints the faulting thread's call stack
*/

#include "CrashReport.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>

namespace
{
DWORD64 g_linkBase = 0;   // the image base the linker chose (the loaded image is relocated)

// the executable file's image base: the loader overwrites the one in the loaded headers
DWORD64 readLinkBase()
{
  wchar_t path[MAX_PATH];
  if( !GetModuleFileNameW( nullptr, path, MAX_PATH ) )
  {
    return 0;
  }
  FILE* file = _wfopen( path, L"rb" );
  if( !file )
  {
    return 0;
  }
  IMAGE_DOS_HEADER dos;
  IMAGE_NT_HEADERS nt;
  const bool ok = fread( &dos, sizeof( dos ), 1, file ) == 1 && fseek( file, dos.e_lfanew, SEEK_SET ) == 0
                  && fread( &nt, sizeof( nt ), 1, file ) == 1;
  fclose( file );
  return ok ? nt.OptionalHeader.ImageBase : 0;
}

LONG WINAPI printCrashReport( EXCEPTION_POINTERS* ep )
{
  // addresses as the linker placed them, so that addr2line on the -g build of the same source resolves them
  const DWORD64           base      = (DWORD64) GetModuleHandleW( nullptr );
  const IMAGE_NT_HEADERS* nt        = (const IMAGE_NT_HEADERS*) ( base + ( (const IMAGE_DOS_HEADER*) base )->e_lfanew );
  const DWORD64           linkBase  = g_linkBase;
  const DWORD64           imageSize = nt->OptionalHeader.SizeOfImage;
  const EXCEPTION_RECORD& rec       = *ep->ExceptionRecord;
  fprintf( stderr, "\nEncoderApp crashed: exception 0x%08lx", rec.ExceptionCode );
  if( rec.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && rec.NumberParameters >= 2 )
  {
    fprintf( stderr, " (%s of address 0x%llx)", rec.ExceptionInformation[0] == 1 ? "write" : "read",
             (unsigned long long) rec.ExceptionInformation[1] );
  }
  fprintf( stderr, " in thread %lu\ncall stack (addr2line -f -C -i -e EncoderApp.exe ADDRESS on a -g build):\n",
           GetCurrentThreadId() );
  CONTEXT ctx = *ep->ContextRecord;
  for( int i = 0; i < 64 && ctx.Rip; i++ )
  {
    if( ctx.Rip - base < imageSize )
    {
      fprintf( stderr, "  #%-2d 0x%llx\n", i, (unsigned long long) ( ctx.Rip - base + linkBase ) );
    }
    else
    {
      fprintf( stderr, "  #%-2d 0x%llx (outside EncoderApp.exe)\n", i, (unsigned long long) ctx.Rip );
    }
    DWORD64           imageBase = 0;
    PRUNTIME_FUNCTION function  = RtlLookupFunctionEntry( ctx.Rip, &imageBase, nullptr );
    if( function )
    {
      PVOID   handlerData = nullptr;
      DWORD64 frame       = 0;
      RtlVirtualUnwind( UNW_FLAG_NHANDLER, imageBase, ctx.Rip, function, &ctx, &handlerData, &frame, nullptr );
    }
    else
    {
      ctx.Rip = *(const DWORD64*) ctx.Rsp;   // a leaf function: the return address is on top of the stack
      ctx.Rsp += 8;
    }
  }
  fflush( stderr );
  return EXCEPTION_CONTINUE_SEARCH;
}
}   // namespace

void installCrashReport()
{
  g_linkBase = readLinkBase();
  SetUnhandledExceptionFilter( printCrashReport );
}
#else
void installCrashReport()
{
}
#endif
