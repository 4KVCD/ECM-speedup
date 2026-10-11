// Check tool, not part of the encoder: reports which bytes of the executable's .data and .bss sections each phase of a
// picture writes (setup, mode search, finish). insert_global_watch.py copies this file to source/Lib/CommonLib of a
// throwaway tree and adds the calls; global_watch_report.py maps the reported addresses to symbols.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" char __data_start__[], __data_end__[], __bss_start__[], __bss_end__[], __ImageBase;

struct GlobalWatch
{
  struct Range
  {
    char* begin;
    char* end;
  };
  std::vector<Range>    ranges;
  std::vector<uint8_t>  snap;
  uintptr_t             base = uintptr_t( &__ImageBase );
  FILE*                 out  = nullptr;

  static GlobalWatch* get()
  {
    static GlobalWatch* g = new GlobalWatch;   // on the heap, so the tool's own state stays out of the watched sections
    return g;
  }

  // The watched sections without the given ranges (big arrays whose writers are known).
  void init( std::vector<Range> skip )
  {
    out = fopen( "global_watch.txt", "w" );
    std::sort( skip.begin(), skip.end(), []( const Range& a, const Range& b ) { return a.begin < b.begin; } );
    std::vector<Range> all = { { __data_start__, __data_end__ }, { __bss_start__, __bss_end__ } };
    for( Range r: all )
    {
      char* p = r.begin;
      for( const Range& s: skip )
      {
        if( s.begin >= p && s.end <= r.end )
        {
          ranges.push_back( { p, s.begin } );
          p = s.end;
        }
      }
      ranges.push_back( { p, r.end } );
    }
    take();
  }

  void take()
  {
    snap.clear();
    for( const Range& r: ranges )
    {
      snap.insert( snap.end(), r.begin, r.end );
    }
  }

  // Prints every changed run of bytes since the last snapshot as "phase poc rva length", then takes a new snapshot.
  void diff( const char* phase, int poc )
  {
    size_t off = 0;
    for( const Range& r: ranges )
    {
      const size_t n   = r.end - r.begin;
      const uint8_t* s = snap.data() + off;
      const uint8_t* c = reinterpret_cast<const uint8_t*>( r.begin );
      size_t i = 0;
      while( i < n )
      {
        // skip equal 64-byte blocks quickly
        if( i + 64 <= n && !memcmp( s + i, c + i, 64 ) )
        {
          i += 64;
          continue;
        }
        if( s[i] == c[i] )
        {
          i++;
          continue;
        }
        size_t j = i + 1;
        while( j < n && ( s[j] != c[j] || ( j + 16 < n && memcmp( s + j, c + j, 16 ) ) ) )
        {
          j++;
        }
        fprintf( out, "%s %d %llx %llu\n", phase, poc, (unsigned long long) ( uintptr_t( r.begin + i ) - base ),
                 (unsigned long long) ( j - i ) );
        i = j;
      }
      off += n;
    }
    fflush( out );
    take();
  }
};
