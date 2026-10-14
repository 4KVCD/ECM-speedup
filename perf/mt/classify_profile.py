#!/usr/bin/env python3
"""Split an sprof profile of the encoder into the parallelisation classes of perf/HANDOFF_MULTITHREADING.md.

Each sample goes to the first class whose functions appear on its stack:
  filter             loop-filter encoders, temporal pre-filter, slice writing (serial finish stage)
  rd_*               full-RD candidate work (transform, quantisation, rate); parallel only as speculation, except
                     the first (DCT2) pass of the intra luma candidates
  me                 motion and block-vector search
  cand_*             candidate prediction / SATD / refinement passes; deterministic parallel-for
  cu_other           the rest of the CU search (mode control, CS copies, split bookkeeping)
  other              everything else

usage: classify_profile.py samples.txt EncoderApp.exe   (the exe the samples were taken with; mingw bin on PATH)
"""
import bisect
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "sprof"))
from sprof_report import load_symbols, short  # noqa: E402

RULES = [
    ("filter", ["EncSampleAdaptiveOffset::", "EncAdaptiveLoopFilter::", "LoopFilter::loopFilterPic",
                "EncLoopFilterCccm::", "EncTemporalFilter::", "EncGOP::xCalculateAddPSNR", "EncSlice::encodeSlice"]),
    ("rd_intra_luma", ["IntraSearch::xRecurIntraCodingLumaQT", "IntraSearch::xIntraCodingLumaISP"]),
    ("rd_intra_chroma", ["IntraSearch::xRecurIntraChromaCodingQT"]),
    ("rd_inter", ["EncCu::xEncodeInterResidual", "InterSearch::encodeResAndCalcRdInterCU"]),
    ("rd_obmc_in_rd", ["InterPrediction::subBlockOBMC"]),   # OBMC is applied per full-RD candidate
    ("me", ["InterSearch::xMotionEstimation", "InterSearch::xPredAffineInterSearch",
            "InterSearch::xEstimateMvPredAMVP", "InterSearch::predInterSearch", "InterSearch::predIBCSearch"]),
    ("cand_geo", ["EncCu::xCheckRDCostMergeGeoComb2Nx2N"]),
    ("cand_merge", ["EncCu::xCheckRDCostMerge2Nx2N"]),
    ("cand_intra_chroma", ["IntraSearch::estIntraPredChromaQT"]),
    ("cand_intra_luma", ["IntraSearch::estIntraPredLumaQT"]),
    ("cand_intra_derive", ["EncCu::xCheckRDCostIntra"]),
    ("cu_other", ["EncSlice::compressSlice"]),
]


def main():
    samples, exe = sys.argv[1], sys.argv[2]
    _, addrs, names = load_symbols(exe)
    cache = {}

    def sym(frame):
        if frame.startswith("x:"):
            return frame[2:]
        if frame not in cache:
            i = bisect.bisect_right(addrs, int(frame, 16)) - 1
            cache[frame] = short(names[i]) if i >= 0 else "?"
        return cache[frame]

    total = 0
    by_class = collections.Counter()
    for line in open(samples):
        if line.startswith("#"):
            continue
        parts = line.split()
        n = int(parts[0])
        funcs = [sym(f) for f in parts[1:]]
        total += n
        cls = next((name for name, keys in RULES if any(k in f for f in funcs for k in keys)), "other")
        by_class[cls] += n
    print(f"{os.path.basename(samples)}: {total} samples")
    for name, _ in RULES + [("other", None)]:
        print(f"  {name:18s} {100.0 * by_class[name] / total:6.2f}%")


if __name__ == "__main__":
    main()
