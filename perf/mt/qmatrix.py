#!/usr/bin/env python3
"""Quick bit-exactness matrix: the suite + ld_cif + ra_cif with the given option groups, every job on its own CPU mask
from a pool, as many at a time as the pool has masks (cu_matrix.py runs a group at a time). The bitstream md5s are
compared with runs/results/ref.json; ld_cif and ra_cif, which it lacks, with the reference exe's default encodes,
kept in runs/results/qmatrix_ref.json. Correctness only: jobs share the machine, so their times are not comparable.
usage: qmatrix.py EXE TAG REFEXE [groups, default "default,ft8"]   (QM_MASKS=hex,hex,... replaces the mask pool)"""
import ctypes
import json
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import bench  # noqa: E402

bench.TESTS["ld_cif"] = ('encoder_lowdelay_ecm.cfg', 'foreman_352x288_300.yuv', 352, 288, 30, 3, 32, [])
bench.TESTS["ra_cif"] = ('encoder_randomaccess_ecm.cfg', 'foreman_352x288_300.yuv', 352, 288, 30, 3, 32, [])
TESTS = ["ld_cif", "ra_cif", "ai_foreman", "ai_parkjoy", "ra_foreman", "ra_parkjoy", "ld_foreman"]   # longest first
OPTIONS = {"default": [], "ft2": ["--FrameThreads=2"], "ft4": ["--FrameThreads=4"], "ft8": ["--FrameThreads=8"],
           "ft4cu2": ["--FrameThreads=4", "--CuThreads=2"], "cu2": ["--CuThreads=2"]}
# two logical CPUs a job: the P-core and E-core pairs of the Core Ultra 7 265K
DEFAULT_POOL = "3,C0,300,C0000,C,30,C00,3000,C000,30000"
REF_CACHE = os.path.join(bench.RESULTS, "qmatrix_ref.json")


def pin(proc, mask):
    if not ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), ctypes.c_size_t(mask)):
        raise OSError("SetProcessAffinityMask failed")


def main():
    exe, tag, refexe = sys.argv[1:4]
    groups = (sys.argv[4] if len(sys.argv) > 4 else "default,ft8").split(",")
    bench.pin = pin   # run_test's fixed_core is a CPU mask here
    ref = {k: v["md5"] for k, v in json.load(open(os.path.join(bench.RESULTS, "ref.json"))).items()}
    cached = json.load(open(REF_CACHE)) if os.path.exists(REF_CACHE) else {}
    ref.update(cached)
    masks = queue.Queue()
    for m in os.environ.get("QM_MASKS", DEFAULT_POOL).split(","):
        masks.put(int(m, 16))
    lock = threading.Lock()
    results = []

    def run(test, args, name, exe_):
        mask = masks.get()
        try:
            out = os.path.join(bench.ROOT, "runs", "matrix", tag, f"{test}_{name}")
            os.makedirs(out, exist_ok=True)
            _, r = bench.run_test(exe_, out, test, args, fixed_core=mask)
        finally:
            masks.put(mask)
        with lock:
            results.append((test, name, r["md5"]))
            state = "REF" if name == "REF" else ("MATCH" if r["md5"] == ref.get(test) else "*** MISMATCH ***")
            print(f"{test:12s} {name:10s} {r['time']:8.2f} s  {r['md5']}  {state}", flush=True)

    def run_all(jobs):
        threads = [threading.Thread(target=run, args=job) for job in jobs]
        [t.start() for t in threads]
        [t.join() for t in threads]

    run_all([(t, [], "REF", refexe) for t in ("ld_cif", "ra_cif") if t not in ref])
    for test, name, md5 in results:
        ref[test] = cached[test] = md5
    with open(REF_CACHE, "w") as f:
        json.dump(cached, f, indent=1)
    results.clear()
    run_all([(t, OPTIONS[g], g, exe) for t in TESTS for g in groups])
    bad = [(t, n) for t, n, md5 in results if md5 != ref.get(t)]
    print("ALL MATCH" if not bad else f"*** {len(bad)} MISMATCHES: {bad}")


if __name__ == "__main__":
    main()
