#!/usr/bin/env python3
"""Add the global-memory watch to a throwaway source tree (never commit the result).

The watch reports, per picture, which bytes of .data and .bss the setup (compressGOP up to the mode search), the mode
search (precompressSlice + compressSlice) and the finish (the rest of compressGOP) write, and what runs between two
pictures. In older trees where g_reusedUniMVs and g_reusedUniMVsLIC are still 175 MB globals, pass them to init() as
skip ranges (their writer, the motion search, is known); here they are heap members of InterSearch.
Output: global_watch.txt in the working directory; global_watch_report.py maps it to symbols.

usage: insert_global_watch.py <tree>
"""
import os
import shutil
import sys

TREE = sys.argv[1]
SRC = os.path.join(TREE, "source", "Lib")


def patch(path, pairs):
    full = os.path.join(SRC, path)
    raw = open(full, encoding="utf-8", newline="").read()
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            sys.exit(f"{path}: anchor found {n} times: {old[:80]!r}")
        s = s.replace(old, new)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(full, "w", encoding="utf-8", newline="").write(s)


shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "GlobalWatch.h"),
            os.path.join(SRC, "CommonLib", "GlobalWatch.h"))

patch("EncoderLib/EncGOP.cpp", [
    ('#include "EncGOP.h"\n', '#include "EncGOP.h"\n#include "CommonLib/GlobalWatch.h"   // watch\n'),
    ("""    //-- For time output for each slice
    auto beforeTime = std::chrono::steady_clock::now();
""", """    //-- For time output for each slice
    auto beforeTime = std::chrono::steady_clock::now();
    {   // watch
      static bool watchInit = false;
      if( !watchInit )
      {
        watchInit = true;
        GlobalWatch::get()->init( {} );
      }
      GlobalWatch::get()->diff( "between", -1 );
    }
"""),
    ("""        m_pcSliceEncoder->precompressSlice( pcPic );
""", """        GlobalWatch::get()->diff( "setup", pcSlice->getPOC() );   // watch
        m_pcSliceEncoder->precompressSlice( pcPic );
"""),
    ("""        m_pcSliceEncoder->compressSlice   ( pcPic, false, false );
""", """        m_pcSliceEncoder->compressSlice   ( pcPic, false, false );
        GlobalWatch::get()->diff( "search", pcSlice->getPOC() );   // watch
"""),
    ("""    pcPic->reconstructed = true;
    m_bFirst = false;
    m_iNumPicCoded++;
""", """    pcPic->reconstructed = true;
    m_bFirst = false;
    m_iNumPicCoded++;
    GlobalWatch::get()->diff( "finish", pcSlice->getPOC() );   // watch
"""),
])
print("global watch inserted into", TREE)
