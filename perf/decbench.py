#!/usr/bin/env python3
"""Decoder check and timing: decodes bitstreams with two DecoderApp builds at the same time (pinned to a pair of
CPUs, swapped every repetition), compares the md5 of the decoded YUV and reports the CPU-time ratio.

usage: decbench.py A.exe B.exe outdir bitstream[,bitstream...] [--reps N] [--cpus a,b]
"""
import argparse
import concurrent.futures as cf
import ctypes
import hashlib
import os
import subprocess
import time
from ctypes import wintypes


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_seconds(proc):
    ft = [wintypes.FILETIME() for _ in range(4)]
    ctypes.windll.kernel32.GetProcessTimes(wintypes.HANDLE(int(proc._handle)), *[ctypes.byref(f) for f in ft])
    val = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7
    return val(ft[2]) + val(ft[3])


def decode(exe, bs, out, cpu):
    t0 = time.perf_counter()
    with open(out + ".log", "w") as lf:
        p = subprocess.Popen([exe, "-b", bs, "-o", out], stdout=lf, stderr=subprocess.STDOUT)
        ctypes.windll.kernel32.SetProcessAffinityMask(int(p._handle), ctypes.c_size_t(1 << cpu))
        p.wait()
        c = cpu_seconds(p)
    dec = None
    for line in open(out + ".log", errors="replace"):
        if "Total Time" in line:
            dec = float(line.split()[2])   # the decoder's own timer (decode() only, excludes process start-up)
    return p.returncode, time.perf_counter() - t0, c, md5(out) if p.returncode == 0 else None, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("outdir")
    ap.add_argument("bitstreams")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cpus", default="0,1")
    args = ap.parse_args()
    ca, cb = (int(x) for x in args.cpus.split(","))
    os.makedirs(args.outdir, exist_ok=True)
    tot_a = tot_b = 0.0
    ok = True
    for bs in args.bitstreams.split(","):
        name = os.path.splitext(os.path.basename(bs))[0]
        ta, tb, da, db, md = [], [], [], [], set()
        for rep in range(args.reps):
            pa, pb = (cb, ca) if rep % 2 else (ca, cb)
            oa = os.path.join(args.outdir, f"{name}_a.yuv")
            ob = os.path.join(args.outdir, f"{name}_b.yuv")
            with cf.ThreadPoolExecutor(2) as ex:
                fa = ex.submit(decode, args.a, bs, oa, pa)
                fb = ex.submit(decode, args.b, bs, ob, pb)
                ra, rb = fa.result(), fb.result()
            if ra[0] or rb[0]:
                print(f"{name}: decoder failed (rc {ra[0]} / {rb[0]})")
                ok = False
                break
            ta.append(ra[2])
            tb.append(rb[2])
            da.append(ra[4] or 0)
            db.append(rb[4] or 0)
            md.update([ra[3], rb[3]])
            os.remove(oa)
            os.remove(ob)
        if len(ta) < args.reps:
            continue
        same = len(md) == 1
        ok &= same
        ma, mb = sum(ta) / len(ta), sum(tb) / len(tb)
        tot_a += ma
        tot_b += mb
        sda, sdb = sum(da), sum(db)
        print(f"{name:14s} CPU A {ma:8.3f}s  B {mb:8.3f}s  speedup {ma / mb:5.3f}x   "
              f"decode timer A {sda / len(da):7.3f}s  B {sdb / len(db):7.3f}s  speedup {sda / max(sdb, 1e-9):5.3f}x   "
              f"yuv md5 {'IDENTICAL ' + md.pop() if same else 'DIFFERENT'}", flush=True)
    if tot_b:
        print(f"{'TOTAL':14s} A {tot_a:8.2f}s  B {tot_b:8.2f}s  speedup {tot_a / tot_b:5.3f}x  "
              f"{'ALL IDENTICAL' if ok else '*** MISMATCH ***'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
