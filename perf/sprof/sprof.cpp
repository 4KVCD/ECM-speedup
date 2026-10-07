// sprof: minimal sampling profiler for Windows x64.
// Launches a command, samples all its threads every ~1 ms and unwinds each with DbgHelp's StackWalk64
// (driven by the image's .pdata unwind tables, so no frame pointers are needed).
// Writes aggregated stacks: addresses inside the main exe as RVAs, addresses in DLLs as "x:<name>".
// Symbolize/summarize with sprof_report.py.
//
// usage: sprof.exe out.txt <command line...>

#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
#include <tlhelp32.h>

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

static std::string quoteArg( const char* a )
{
  std::string s = a;
  if( !s.empty() && s.find_first_of( " \t\"" ) == std::string::npos )
  {
    return s;
  }
  std::string r = "\"";
  for( char c : s )
  {
    if( c == '"' )
    {
      r += '\\';
    }
    r += c;
  }
  return r + "\"";
}

int main( int argc, char** argv )
{
  if( argc < 3 )
  {
    fprintf( stderr, "usage: sprof out.txt command [args...]\n" );
    return 1;
  }
  std::string cmd;
  for( int i = 2; i < argc; i++ )
  {
    cmd += ( i > 2 ? " " : "" ) + quoteArg( argv[i] );
  }

  STARTUPINFOA si = { sizeof( si ) };
  PROCESS_INFORMATION pi = {};
  std::vector<char> cmdBuf( cmd.begin(), cmd.end() );
  cmdBuf.push_back( 0 );
  if( !CreateProcessA( nullptr, cmdBuf.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi ) )
  {
    fprintf( stderr, "CreateProcess failed: %lu\n", GetLastError() );
    return 1;
  }

  uint64_t base = 0, size = 0;
  for( int tries = 0; tries < 2000 && !base; tries++ )
  {
    HMODULE mods[1];
    DWORD needed = 0;
    if( EnumProcessModulesEx( pi.hProcess, mods, sizeof( mods ), &needed, LIST_MODULES_ALL ) && needed )
    {
      MODULEINFO mi;
      if( GetModuleInformation( pi.hProcess, mods[0], &mi, sizeof( mi ) ) )
      {
        base = (uint64_t) mi.lpBaseOfDll;
        size = mi.SizeOfImage;
      }
    }
    if( !base )
    {
      Sleep( 1 );
    }
  }
  auto inModule = [&]( uint64_t a ) { return a >= base && a < base + size; };

  SymSetOptions( SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS );
  bool symInit = false;

  timeBeginPeriod( 1 );
  std::map<std::vector<std::string>, uint64_t> stacks;
  std::map<uint64_t, std::string> extNames;
  uint64_t numSamples = 0;
  std::map<DWORD, HANDLE> threads;

  auto extName = [&]( uint64_t a ) -> const std::string& {
    auto it = extNames.find( a );
    if( it != extNames.end() )
    {
      return it->second;
    }
    char buf[sizeof( SYMBOL_INFO ) + 256];
    SYMBOL_INFO* s = (SYMBOL_INFO*) buf;
    s->SizeOfStruct = sizeof( SYMBOL_INFO );
    s->MaxNameLen = 255;
    DWORD64 disp = 0;
    std::string name;
    IMAGEHLP_MODULE64 mi = { sizeof( mi ) };
    if( SymGetModuleInfo64( pi.hProcess, a, &mi ) )
    {
      name = mi.ModuleName;
    }
    else
    {
      name = "?";
    }
    if( SymFromAddr( pi.hProcess, a, &disp, s ) )
    {
      name += std::string( "!" ) + s->Name;
    }
    return extNames[a] = "x:" + name;
  };

  int tick = 0;
  while( WaitForSingleObject( pi.hProcess, 1 ) == WAIT_TIMEOUT )
  {
    if( ( tick++ & 63 ) == 0 )
    {
      if( symInit )
      {
        SymRefreshModuleList( pi.hProcess );
      }
      else
      {
        symInit = SymInitialize( pi.hProcess, nullptr, TRUE );
      }
      HANDLE snap = CreateToolhelp32Snapshot( TH32CS_SNAPTHREAD, 0 );
      THREADENTRY32 te = { sizeof( te ) };
      if( Thread32First( snap, &te ) )
      {
        do
        {
          if( te.th32OwnerProcessID == pi.dwProcessId && !threads.count( te.th32ThreadID ) )
          {
            HANDLE h = OpenThread( THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
                                   te.th32ThreadID );
            if( h )
            {
              threads[te.th32ThreadID] = h;
            }
          }
        } while( Thread32Next( snap, &te ) );
      }
      CloseHandle( snap );
    }
    if( !symInit )
    {
      continue;
    }

    for( auto it = threads.begin(); it != threads.end(); )
    {
      HANDLE h = it->second;
      if( SuspendThread( h ) == (DWORD) -1 )
      {
        CloseHandle( h );
        it = threads.erase( it );
        continue;
      }
      alignas( 16 ) CONTEXT ctx;
      ctx.ContextFlags = CONTEXT_FULL;
      std::vector<uint64_t> pcs;
      if( GetThreadContext( h, &ctx ) )
      {
        STACKFRAME64 sf = {};
        sf.AddrPC.Offset = ctx.Rip;
        sf.AddrPC.Mode = AddrModeFlat;
        sf.AddrFrame.Offset = ctx.Rbp;
        sf.AddrFrame.Mode = AddrModeFlat;
        sf.AddrStack.Offset = ctx.Rsp;
        sf.AddrStack.Mode = AddrModeFlat;
        for( int d = 0; d < 96; d++ )
        {
          if( !StackWalk64( IMAGE_FILE_MACHINE_AMD64, pi.hProcess, h, &sf, &ctx, nullptr, SymFunctionTableAccess64,
                            SymGetModuleBase64, nullptr ) || !sf.AddrPC.Offset )
          {
            break;
          }
          pcs.push_back( sf.AddrPC.Offset );
        }
      }
      ResumeThread( h );
      ++it;
      // skip idle threads (sitting in a wait inside ntdll with no exe frames)
      bool anyExe = false;
      for( auto a : pcs )
      {
        anyExe |= inModule( a );
      }
      if( pcs.empty() || !anyExe )
      {
        continue;
      }
      std::vector<std::string> st;
      for( size_t i = 0; i < pcs.size(); i++ )
      {
        uint64_t a = pcs[i];
        if( inModule( a ) )
        {
          // return addresses point after the call; step back one byte for callers
          char b[32];
          snprintf( b, sizeof( b ), "%llx", (unsigned long long) ( a - base - ( i ? 1 : 0 ) ) );
          st.push_back( b );
        }
        else if( i == 0 || st.empty() || st.back().rfind( "x:", 0 ) != 0 )
        {
          st.push_back( extName( a ) );
        }
      }
      numSamples++;
      stacks[st]++;
    }
  }
  timeEndPeriod( 1 );

  DWORD exitCode = 0;
  GetExitCodeProcess( pi.hProcess, &exitCode );

  FILE* f = fopen( argv[1], "w" );
  fprintf( f, "# samples %llu\n", (unsigned long long) numSamples );
  for( auto& kv : stacks )
  {
    fprintf( f, "%llu", (unsigned long long) kv.second );
    for( auto& a : kv.first )
    {
      fprintf( f, " %s", a.c_str() );
    }
    fprintf( f, "\n" );
  }
  fclose( f );
  fprintf( stderr, "sprof: %llu samples, exit code %lu\n", (unsigned long long) numSamples, exitCode );
  return (int) exitCode;
}
