#!/usr/bin/env python3
"""Bit-exactness + timing harness for the ECM decoder.

  decbench.py run <name> [--exe PATH] [--jobs N] [--streams a,b] [--extra ARGS]
      decode every stream, save runs/results/dec_<name>.json; unless <name> is "dref", compare every output frame
      with the reference decoder
  decbench.py ab <A> <B> [--jobs N] [--reps R] [--streams a,b] [--extra-a ARGS] [--extra-b ARGS]
      paired timing: for each stream, builds A and B decode at the same time (same load); every frame checked

Streams are C:/Code/ECM/streams/<name>.bin (ECM bitstreams). The reference "dref" is the stock ECM-20.0 decoder
(run once: decbench.py run dref --exe C:/Code/ECM/bin/ref/DecoderApp.exe); its per-frame md5s are the expected output.
Builds are C:/Code/ECM/bin/<build>/DecoderApp.exe. CPU time counts every process of a run (segment-parallel children too).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import CpuJob, ROOT  # noqa: E402

STREAMS = f"{ROOT}/streams"
RESULTS = f"{ROOT}/runs/results"


def stream_names(sel):
    names = sorted(f[:-4] for f in os.listdir(STREAMS) if f.endswith(".bin"))
    if sel:
        want = sel.split(",")
        missing = [w for w in want if w not in names]
        if missing:
            sys.exit(f"unknown streams: {missing}")
        return want
    return names


def load_ref():
    path = os.path.join(RESULTS, "dec_dref.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def frame_md5s(path, frame_bytes):
    out = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(frame_bytes)
            if not buf:
                break
            out.append(hashlib.md5(buf).hexdigest() if len(buf) == frame_bytes else "partial")
    return out


def decode(exe, outdir, name, extra, frame_bytes=None):
    """decode one stream; returns (name, result)"""
    os.makedirs(outdir, exist_ok=True)
    yuv = os.path.join(outdir, name + ".yuv")
    log = os.path.join(outdir, name + ".log")
    if os.path.exists(yuv):
        os.remove(yuv)
    cmd = [exe, "-b", f"{STREAMS}/{name}.bin", "-o", yuv, "-d", "0"] + list(extra)
    t0 = time.perf_counter()
    with open(log, "w") as lf:
        cpu = None
        if os.name == "nt":
            job = CpuJob()
            try:
                proc = job.start(cmd, stdout=lf, stderr=subprocess.STDOUT)
                proc.wait()
                cpu = job.cpu_seconds()
            finally:
                job.close()
        else:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            proc.wait()
    dt = time.perf_counter() - t0
    res = {"time": round(dt, 3), "cpu": round(cpu, 3) if cpu else None, "rc": proc.returncode}
    for line in open(log, errors="replace"):
        if line.startswith("Segment-parallel decoding:") and "decoding sequentially" in line:
            res["fallback"] = line.split(":", 1)[1].strip()   # the output then comes from a sequential decode
    if proc.returncode == 0 and os.path.exists(yuv):
        size = os.path.getsize(yuv)
        if frame_bytes is None:   # reference run: one "POC" line per decoded (and output) picture
            frames = sum(1 for line in open(log, errors="replace") if line.startswith("POC "))
            frame_bytes = size // frames if frames and size % frames == 0 else None
        res["frame_bytes"] = frame_bytes
        res["frames"] = frame_md5s(yuv, frame_bytes) if frame_bytes else None
        os.remove(yuv)
    return name, res


def check(ref, name, r):
    """'ok', or why the output differs from the reference decoder"""
    if r["rc"] != 0:
        return f"exit code {r['rc']}"
    want = ref.get(name, {}).get("frames")
    got = r.get("frames")
    if want is None:
        return "no reference"
    if got is None:
        return "no output"
    if len(got) != len(want):
        return f"{len(got)} frames instead of {len(want)}"
    bad = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
    return "ok" if not bad else f"{len(bad)} frames differ (first: frame {bad[0]})"


def exe_of(build):
    return f"{ROOT}/bin/{build}/DecoderApp.exe"


def cmd_run(args):
    exe = args.exe or exe_of(args.name)
    names = stream_names(args.streams)
    ref = {} if args.name == "dref" else load_ref()
    extra = shlex.split(args.extra or "")
    outdir = os.path.join(ROOT, "runs", "dec_" + args.name)
    res = {}
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(decode, exe, outdir, n, extra, ref.get(n, {}).get("frame_bytes")) for n in names]
        for fu in cf.as_completed(futs):
            n, r = fu.result()
            res[n] = r
            status = check(ref, n, r) if ref else f"{len(r.get('frames') or [])} frames"
            if "fallback" in r:
                status += f"  [sequential fallback: {r['fallback']}]"
            print(f"{n:14s} {r['time']:8.2f}s  cpu {r['cpu'] or 0:8.2f}s  {status}", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"dec_{args.name}.json")
    old = json.load(open(path)) if os.path.exists(path) and args.streams else {}
    old.update(res)
    json.dump(old, open(path, "w"), indent=1, sort_keys=True)
    if ref:
        bad = [n for n in names if check(ref, n, res[n]) != "ok"]
        fell_back = [n for n in names if "fallback" in res[n]]
        print("output vs reference decoder:", "ALL FRAMES MATCH" if not bad else f"*** MISMATCH: {bad} ***",
              f"| sequential fallbacks: {fell_back}" if fell_back else "")
        if args.require_parallel and fell_back:
            print("*** segment-parallel decoding was required but fell back ***")
            return 1
        return 1 if bad else 0
    return 0


def cmd_ab(args):
    names = stream_names(args.streams)
    ref = load_ref()
    sides = [(args.a, shlex.split(args.extra_a or "")), (args.b, shlex.split(args.extra_b or ""))]
    res = [{}, {}]
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {}
        for rep in range(args.reps):
            for n in names:
                for s, (build, extra) in enumerate(sides):
                    outdir = os.path.join(ROOT, "runs", f"decab{s}_{build}")
                    fb = ref.get(n, {}).get("frame_bytes")
                    futs[ex.submit(decode, exe_of(build), outdir + f"_r{rep}", n, extra, fb)] = s
        for fu in cf.as_completed(futs):
            s = futs[fu]
            n, r = fu.result()
            r["check"] = check(ref, n, r)
            prev = res[s].get(n)
            if prev:   # minimum over repetitions; every repetition must match
                r["check"] = r["check"] if r["check"] != "ok" else prev["check"]
                r["time"] = min(r["time"], prev["time"])
                r["cpu"] = min(r["cpu"], prev["cpu"]) if r["cpu"] and prev["cpu"] else None
            res[s][n] = r
    label = [f"{b}{' ' + e if e else ''}" for b, e in [(args.a, args.extra_a), (args.b, args.extra_b)]]
    print(f"A: {label[0]}\nB: {label[1]}\n")
    print(f"{'stream':14s} {'A wall':>9s} {'B wall':>9s} {'speedup':>8s} {'A cpu':>9s} {'B cpu':>9s} {'cpu x':>7s}  output")
    ta = tb = ca = cb = 0.0
    ok = True
    for n in names:
        a, b = res[0][n], res[1][n]
        good = a["check"] == "ok" and b["check"] == "ok"
        ok &= good
        ta += a["time"]
        tb += b["time"]
        ca += a["cpu"] or 0
        cb += b["cpu"] or 0
        cpux = f"{a['cpu'] / b['cpu']:6.2f}x" if a["cpu"] and b["cpu"] else ""
        status = "ok" if good else "A: " + a["check"] + ", B: " + b["check"]
        status += "".join(f" ({s} fell back to sequential)" for s, r in (("A", a), ("B", b)) if "fallback" in r)
        print(f"{n:14s} {a['time']:9.2f} {b['time']:9.2f} {a['time'] / b['time']:7.2f}x {a['cpu'] or 0:9.2f} "
              f"{b['cpu'] or 0:9.2f} {cpux:>7s}  {status}")
    cpux = f"{ca / cb:6.2f}x" if cb else ""
    print(f"{'TOTAL':14s} {ta:9.2f} {tb:9.2f} {ta / tb:7.2f}x {ca:9.2f} {cb:9.2f} {cpux:>7s}  "
          f"{'ALL FRAMES MATCH' if ok else '*** MISMATCH ***'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--exe")
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--streams")
    r.add_argument("--extra", help="extra decoder arguments, e.g. --extra=\"--ParallelSegments=8\"")
    r.add_argument("--require-parallel", action="store_true",
                   help="fail if a segment-parallel decode fell back to sequential decoding")
    p = sub.add_parser("ab")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--streams")
    p.add_argument("--extra-a")
    p.add_argument("--extra-b")
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_ab(args)


if __name__ == "__main__":
    sys.exit(main())
