#!/usr/bin/env python3
"""Bit-exactness + timing harness for the ECM encoder.

  bench.py run  <name> [--exe PATH] [--jobs N] [--tests a,b]   encode all tests, save results/<name>.json
  bench.py cmp  <base> <new>                                   compare md5s and times of two result sets

A test passes when its bitstream md5 equals the reference ("ref") result.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = "C:/Code"
CFG = f"{ROOT}/ECM/cfg"
SEQ = f"{ROOT}/seq"
RESULTS = f"{ROOT}/runs/results"

# name: (cfg, yuv, w, h, fps, frames, qp, extra args)
TESTS = {
    "ai_foreman":   ("encoder_intra_ecm.cfg",        "foreman_352x288.yuv", 352, 288, 30, 1, 32, []),
    "ai_parkjoy":   ("encoder_intra_ecm.cfg",        "ParkJoy_256x256.yuv", 256, 256, 50, 2, 27, []),
    "ra_foreman":   ("encoder_randomaccess_ecm.cfg", "foreman_176x144.yuv", 176, 144, 30, 9, 32, []),
    "ra_parkjoy":   ("encoder_randomaccess_ecm.cfg", "ParkJoy_256x256.yuv", 256, 256, 50, 5, 37, []),
    "ld_foreman":   ("encoder_lowdelay_ecm.cfg",     "foreman_176x144.yuv", 176, 144, 30, 5, 32, []),
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_cpu_seconds(proc):
    """user+kernel CPU time of a finished (not yet reaped) child process; less load-sensitive than wall time."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    ft = [wintypes.FILETIME() for _ in range(4)]
    ok = ctypes.windll.kernel32.GetProcessTimes(wintypes.HANDLE(int(proc._handle)), *[ctypes.byref(f) for f in ft])
    if not ok:
        return None
    val = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7
    return val(ft[2]) + val(ft[3])


def run_test(exe, outdir, name):
    cfg, yuv, w, h, fps, frames, qp, extra = TESTS[name]
    bs = os.path.join(outdir, name + ".bin")
    log = os.path.join(outdir, name + ".log")
    cmd = [exe, "-c", f"{CFG}/{cfg}", "-i", f"{SEQ}/{yuv}", "-wdt", str(w), "-hgt", str(h), "-fr", str(fps),
           "--InputBitDepth=8", "-f", str(frames), "-q", str(qp), "-b", bs, "-o", "NUL"] + extra
    t0 = time.perf_counter()
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        proc.wait()
        cpu = process_cpu_seconds(proc)
        rc = proc.returncode
    dt = time.perf_counter() - t0
    return name, {"time": round(dt, 2), "cpu": round(cpu, 2) if cpu else None, "rc": rc,
                  "md5": md5(bs) if rc == 0 and os.path.exists(bs) else None}


def cmd_run(args):
    exe = args.exe or f"{ROOT}/bin/{args.name}/EncoderApp.exe"
    names = args.tests.split(",") if args.tests else list(TESTS)
    outdir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    res = {}
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_test, exe, outdir, n) for n in names]
        for fu in cf.as_completed(futs):
            n, r = fu.result()
            res[n] = r
            print(f"{n:14s} {r['time']:9.2f}s  rc={r['rc']}  {r['md5']}", flush=True)
    path = os.path.join(RESULTS, args.name + ".json")
    old = json.load(open(path)) if os.path.exists(path) and args.tests else {}
    old.update(res)
    json.dump(old, open(path, "w"), indent=1, sort_keys=True)
    ref_path = os.path.join(RESULTS, "ref.json")
    if args.name != "ref" and os.path.exists(ref_path):
        return compare(json.load(open(ref_path)), old, "ref", args.name)
    return 0


def compare(a, b, an, bn):
    ok = True
    ta = tb = 0.0
    print(f"\n{'test':14s} {an:>10s} {bn:>10s} {'speedup':>8s}  bit-exact")
    for n in sorted(set(a) & set(b)):
        same = a[n]["md5"] is not None and a[n]["md5"] == b[n]["md5"]
        ok &= same
        ta += a[n]["time"]
        tb += b[n]["time"]
        print(f"{n:14s} {a[n]['time']:10.1f} {b[n]['time']:10.1f} {a[n]['time'] / b[n]['time']:7.2f}x  "
              f"{'YES' if same else '*** NO ***'}")
    print(f"{'TOTAL':14s} {ta:10.1f} {tb:10.1f} {ta / tb:7.2f}x  {'ALL MATCH' if ok else '*** MISMATCH ***'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--exe")
    r.add_argument("--jobs", type=int, default=5)
    r.add_argument("--tests")
    c = sub.add_parser("cmp")
    c.add_argument("base")
    c.add_argument("new")
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    load = lambda n: json.load(open(os.path.join(RESULTS, n + ".json")))
    return compare(load(args.base), load(args.new), args.base, args.new)


if __name__ == "__main__":
    sys.exit(main())
