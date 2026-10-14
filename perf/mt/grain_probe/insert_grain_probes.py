#!/usr/bin/env python3
"""Add the loop-grain probes to a throwaway source tree (never commit the result).

The probes time every candidate of the main loops of the mode search (intra luma RD candidates per pass, split into
ISP and non-ISP; merge and GPM full-RD candidates; GPM per-candidate MC, split-mode SAD and blend-SATD loops; uni-
and bi-pred motion search per reference) and the phases whose candidates are only counted (intra rough mode
decision, merge SATD pass). At exit the encoder prints, per loop, the number of loop instances and tasks, the total
time, the speed-up a 2/4/8/16-worker pool with 2 us overhead per task would reach, and how the time splits by task
and loop size.

The anchor line numbers are those of branch dec at 2912a239 (= enc-mt); every anchor is checked against its text and
the script stops on a mismatch.

usage: insert_grain_probes.py <tree>      e.g. a worktree made with: git worktree add --detach <tree> 2912a239
"""
import os
import shutil
import sys

TREE = sys.argv[1]
SRC = os.path.join(TREE, "source", "Lib")


def edit(path, edits):
    """edits: (line number, expected substring, 'before' | 'after', text); applied bottom-up so numbers stay valid."""
    full = os.path.join(SRC, path)
    with open(full, "rb") as f:
        data = f.read().decode("utf-8", errors="surrogateescape")
    nl = "\r\n" if "\r\n" in data else "\n"
    lines = data.split(nl)
    for num, expect, where, text in sorted(edits, key=lambda e: -e[0]):
        if expect not in lines[num - 1]:
            sys.exit(f"{path}:{num}: expected {expect!r}, found {lines[num - 1]!r}")
        idx = num if where == "after" else num - 1
        lines[idx:idx] = text.split("\n")
    with open(full, "wb") as f:
        f.write(nl.join(lines).encode("utf-8", errors="surrogateescape"))


shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProbeTimer.h"),
            os.path.join(SRC, "CommonLib", "ProbeTimer.h"))

COUNT = "  g_probeCandInserts++;   // probe"
edit("CommonLib/UnitTools.h", [
    (44, '#include "InterPrediction.h"', "after", '#include "ProbeTimer.h"'),
    (1451, "{", "after", COUNT), (1499, "{", "after", COUNT), (1549, "{", "after", COUNT),
    (1599, "{", "after", COUNT), (1638, "{", "after", COUNT), (1682, "{", "after", COUNT),
])

PASS = 'probePass[std::min( int( cu.lfnstIdx ), 3 ) * 2 + ( cu.mtsFlag ? 1 : 0 )]'
edit("EncoderLib/IntraSearch.cpp", [
    (913, "LumaModeFlagsMemoScope", "after", '  ProbePhaseCount probeRmd( "intraRMD" );   // probe'),
    (6129, "for (int mode = isSecondColorSpace", "before",
     '  probeRmd.finish();   // probe\n'
     '  static const char* probePass[] = { "L0M0", "L0M1", "L1", "L1", "L2", "L2", "L3", "L3" };\n'
     f'  ProbeLoop probeRdNonIsp( std::string( "intraRD ") + {PASS} );\n'
     f'  ProbeLoop probeRdIsp( std::string( "intraRD ISP ") + {PASS} );'),
    (6130, "{", "after",
     '      auto probeTask = makeProbeTask( [&]() -> ProbeLoop& { return cu.ispMode ? probeRdIsp : probeRdNonIsp; } );'),
])

edit("EncoderLib/EncCu.cpp", [
    (9350, "{", "after", '      ProbePhaseCount probeMergeSatd( "mergeSATD" );   // probe'),
    (9997, "for( uint32_t uiMrgHADIdx = 0; uiMrgHADIdx < uiNumMrgSATDCand", "before",
     '    ProbeLoop probeMergeRd( "mergeRD" );   // probe'),
    (9998, "{", "after", '      auto probeTask = makeProbeTask( [&]() -> ProbeLoop& { return probeMergeRd; } );'),
    (12209, "for (uint8_t mergeCand = 0; mergeCand < maxNumMergeCandidates", "before",
     '  ProbeLoop probeGeoMc( "geoMC" );   // probe'),
    (12210, "{", "after", '    auto probeTask = makeProbeTask( [&]() -> ProbeLoop& { return probeGeoMc; } );'),
    (12772, "for (int splitDirIdx = 0; splitDirIdx < GEO_NUM_PARTITION_MODE", "before",
     '  ProbeLoop probeGeoSad( "geoSplitSAD" );   // probe'),
    (12773, "{", "after", '    auto probeTask = makeProbeTask( [&]() -> ProbeLoop& { return probeGeoSad; } );'),
    (13147, "for (uint8_t candidateIdx = 0; candidateIdx < numberGeoCandChecked", "before",
     '  ProbeLoop probeGeoBlend( "geoBlendSATD" );   // probe'),
    (13148, "{", "after", '    auto probeTask = makeProbeTask( [&]() -> ProbeLoop& { return probeGeoBlend; } );'),
    (15398, "{", "after", '    ProbeLoop probeGeoRd( "geoRD" );   // probe'),
    (15406, "{", "after", '      auto probeTask = makeProbeTask( [&]() -> ProbeLoop& { return probeGeoRd; } );'),
])

edit("EncoderLib/InterSearch.cpp", [
    (6155, "for ( int iRefList = 0; iRefList < iNumPredDir; iRefList++ )", "before",
     '      ProbeLoop probeUniMe( "uniME" );   // probe'),
    (6159, "{", "after", '          auto probeTask = makeProbeTask( [&]() -> ProbeLoop& { return probeUniMe; } );'),
    (6441, "for (int iRefIdxTemp = iRefStart; iRefIdxTemp <= iRefEnd; iRefIdxTemp++)", "before",
     '          ProbeLoop probeBiMe( "biME" );   // probe'),
    (6442, "{", "after", '            auto probeTask = makeProbeTask( [&]() -> ProbeLoop& { return probeBiMe; } );'),
])
print("probes inserted into", TREE)
