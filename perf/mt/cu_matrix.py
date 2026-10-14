#!/usr/bin/env python3
"""Bit-exactness matrix for the thread options: runs the tests with each option set, the jobs of a group at the same
time on their own P-cores, and compares the bitstream md5s with runs/results/ref.json. Tests ref.json lacks (ld_cif,
ra_cif) are first encoded by the reference exe with default options.

usage: cu_matrix.py EXE TAG REFEXE      (MATRIX_GROUPS=default,cu2,cu2check,cu4,cu8,cu2ft2,ft2,ft4,ft8,ft4cu2
                                        selects groups)
"""
import ctypes
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import bench  # noqa: E402

bench.TESTS["ld_cif"] = ('encoder_lowdelay_ecm.cfg', 'foreman_352x288_300.yuv', 352, 288, 30, 3, 32, [])
bench.TESTS["ra_cif"] = ('encoder_randomaccess_ecm.cfg', 'foreman_352x288_300.yuv', 352, 288, 30, 3, 32, [])
exe, tag, refexe = sys.argv[1], sys.argv[2], sys.argv[3]
ref = {k: v["md5"] for k, v in json.load(open(os.path.join(bench.RESULTS, "ref.json"))).items()}
SUITE = ["ai_foreman", "ai_parkjoy", "ra_foreman", "ra_parkjoy", "ld_foreman"]
EXTRA = ["ld_cif", "ra_cif"]
# CPU masks of the Core Ultra 7 265K's P-cores (logical CPUs 0, 1, 6, 7, 8, 9, 18, 19)
P1 = [0x1, 0x2, 0x40, 0x80, 0x100, 0x200, 0x40000, 0x80000]
P2 = [0x3, 0xC0, 0x300, 0xC0000]
P4 = [0xC3, 0xC0300]
P8 = [0xC03C3]

lock = threading.Lock()
results = []


def pin(proc, mask):
    if not ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), ctypes.c_size_t(mask)):
        raise OSError("SetProcessAffinityMask failed")


bench.pin = pin   # run_test's fixed_core is a CPU mask here


def run(test, args, mask, name, exe_=None):
    out = os.path.join(bench.ROOT, "runs", "matrix", tag, f"{test}_{name}")
    os.makedirs(out, exist_ok=True)
    _, r = bench.run_test(exe_ or exe, out, test, args, fixed_core=mask)
    with lock:
        results.append((test, name, r))
        exp = ref.get(test)
        ok = "MATCH" if r["md5"] == exp else ("NO REF" if exp is None else "*** MISMATCH ***")
        print(f"{test:12s} {name:22s} {r['time']:8.2f} s  cpu {r['cpu']}  {r['md5']}  {ok}", flush=True)


def group(jobs, masks):
    """jobs: (test, args, name); len(masks) at a time"""
    pending = list(jobs)
    while pending:
        batch, pending = pending[:len(masks)], pending[len(masks):]
        ts = [threading.Thread(target=run, args=(t, a, m, n)) for (t, a, n), m in zip(batch, masks)]
        [t.start() for t in ts]
        [t.join() for t in ts]


ts = [threading.Thread(target=run, args=(t, [], m, "REF", refexe)) for t, m in zip(EXTRA, P1)]
[t.start() for t in ts]
[t.join() for t in ts]
for t, name, r in results:
    if name == "REF":
        ref[t] = r["md5"]
GROUPS = os.environ.get("MATRIX_GROUPS", "default,cu2,cu2check,cu4,cu8,cu2ft2").split(",")
if "default" in GROUPS:
    group([(t, [], "default") for t in SUITE + EXTRA], P1)
if "cu2" in GROUPS:
    group([(t, ["--CuThreads=2"], "cu2") for t in SUITE + EXTRA], P2)
if "cu2check" in GROUPS:
    group([(t, ["--CuThreads=2", "--CuThreadsCheck=1"], "cu2check") for t in SUITE + EXTRA], P2)
if "cu4" in GROUPS:
    group([(t, ["--CuThreads=4"], "cu4") for t in SUITE + EXTRA], P4)
if "cu8" in GROUPS:
    group([(t, ["--CuThreads=8"], "cu8") for t in SUITE], P8)
for n in (2, 4, 8):
    if f"ft{n}" in GROUPS:
        group([(t, [f"--FrameThreads={n}"], f"ft{n}") for t in SUITE + EXTRA], P4 if n <= 4 else P8)
if "ft4cu2" in GROUPS:
    group([(t, ["--FrameThreads=4", "--CuThreads=2"], "ft4cu2") for t in SUITE + EXTRA], P8)
if "cu2ft2" in GROUPS:
    group([(t, ["--CuThreads=2", "--FrameThreads=2"], "cu2ft2") for t in ["ra_foreman", "ra_parkjoy", "ld_foreman", "ra_cif"]], P4)
bad = [(t, n) for t, n, r in results if r["md5"] != ref.get(t)]
print("ALL MATCH" if not bad else f"*** {len(bad)} MISMATCHES: {bad}")
