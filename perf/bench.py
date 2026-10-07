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
DEFAULT_TESTS = list(TESTS)

# long sequences for segment-parallel (wall-clock) measurements; select with --tests
TESTS.update({
    "ra_long":      ("encoder_randomaccess_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 30, 257, 32, []),
    # AI CTC uses TemporalSubsampleRatio 8: 257 input frames -> 33 coded frames
    "ai_long":      ("encoder_intra_ecm.cfg",        "foreman_128x128_257.yuv", 128, 128, 30, 257, 32, []),
})


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


def run_test(exe, outdir, name, more_args=()):
    cfg, yuv, w, h, fps, frames, qp, extra = TESTS[name]
    extra = list(extra) + list(more_args)
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
    names = args.tests.split(",") if args.tests else DEFAULT_TESTS
    outdir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    res = {}
    more = args.extra.split() if args.extra else []
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_test, exe, outdir, n, more) for n in names]
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


def cmd_ab(args):
    """Paired measurement: for every test, build A and build B run at the same time, so both see the same load."""
    names = args.tests.split(",") if args.tests else DEFAULT_TESTS
    res = {args.a: {}, args.b: {}}
    jobs = []
    for _ in range(args.reps):
        for n in names:
            for build in (args.a, args.b):
                jobs.append((build, n))
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {}
        for build, n in jobs:
            outdir = os.path.join(ROOT, "runs", "ab_" + build)
            os.makedirs(outdir, exist_ok=True)
            futs[ex.submit(run_test, f"{ROOT}/bin/{build}/EncoderApp.exe", outdir, n)] = build
        for fu in cf.as_completed(futs):
            build = futs[fu]
            n, r = fu.result()
            prev = res[build].get(n)
            if prev:  # keep the minimum over repetitions; md5 must agree
                r["md5"] = r["md5"] if r["md5"] == prev["md5"] else None
                r["time"] = min(r["time"], prev["time"])
                r["cpu"] = min(r["cpu"], prev["cpu"]) if r.get("cpu") and prev.get("cpu") else None
            res[build][n] = r
            print(f"{build:6s} {n:14s} {r['time']:9.2f}s  {r['md5']}", flush=True)
    ref = json.load(open(os.path.join(RESULTS, "ref.json")))
    ok = True
    for build in (args.a, args.b):
        for n, r in res[build].items():
            ok &= r["md5"] is not None and r["md5"] == ref[n]["md5"]
    rc = compare(res[args.a], res[args.b], args.a, args.b)
    print("bit-exact vs ref:", "ALL MATCH" if ok else "*** MISMATCH ***")
    return 0 if ok and rc == 0 else 1


def compare(a, b, an, bn):
    ok = True
    ta = tb = 0.0
    ca = cb = 0.0
    have_cpu = True
    print(f"\n{'test':14s} {an + ' wall':>12s} {bn + ' wall':>12s} {'speedup':>8s}  {'cpu speedup':>11s}  bit-exact")
    for n in sorted(set(a) & set(b)):
        same = a[n]["md5"] is not None and a[n]["md5"] == b[n]["md5"]
        ok &= same
        ta += a[n]["time"]
        tb += b[n]["time"]
        cpu = ""
        if a[n].get("cpu") and b[n].get("cpu"):
            ca += a[n]["cpu"]
            cb += b[n]["cpu"]
            cpu = f"{a[n]['cpu'] / b[n]['cpu']:10.2f}x"
        else:
            have_cpu = False
        print(f"{n:14s} {a[n]['time']:12.1f} {b[n]['time']:12.1f} {a[n]['time'] / b[n]['time']:7.2f}x  {cpu:>11s}  "
              f"{'YES' if same else '*** NO ***'}")
    cpu = f"{ca / cb:10.2f}x" if have_cpu and cb else ""
    print(f"{'TOTAL':14s} {ta:12.1f} {tb:12.1f} {ta / tb:7.2f}x  {cpu:>11s}  {'ALL MATCH' if ok else '*** MISMATCH ***'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--exe")
    r.add_argument("--jobs", type=int, default=5)
    r.add_argument("--tests")
    r.add_argument("--extra", help="extra encoder arguments, e.g. \"--ParallelSegments=8\"")
    c = sub.add_parser("cmp")
    c.add_argument("base")
    c.add_argument("new")
    p = sub.add_parser("ab")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--tests")
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "ab":
        return cmd_ab(args)
    load = lambda n: json.load(open(os.path.join(RESULTS, n + ".json")))
    return compare(load(args.base), load(args.new), args.base, args.new)


if __name__ == "__main__":
    sys.exit(main())
