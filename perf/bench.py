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
import queue
import subprocess
import sys
import time

ROOT = "C:/Code/ECM"   # the project folder, which is also the main clone
CFG = f"{ROOT}/cfg"
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


class CpuJob:
    """Windows job object around one encoder run: its accounting covers the process and every process it starts
    (segment-parallel workers), so the reported CPU time counts all of them. The process is created suspended and
    only resumed once it is in the job, so no child can escape."""

    def __init__(self):
        import ctypes
        self.k32 = ctypes.windll.kernel32
        self.k32.CreateJobObjectW.restype = ctypes.c_void_p
        self.job = self.k32.CreateJobObjectW(None, None)

    def start(self, cmd, **kw):
        import ctypes
        CREATE_SUSPENDED = 0x4
        proc = subprocess.Popen(cmd, creationflags=CREATE_SUSPENDED, **kw)
        if not self.job or not self.k32.AssignProcessToJobObject(ctypes.c_void_p(self.job),
                                                                 ctypes.c_void_p(int(proc._handle))):
            proc.kill()
            proc.wait()
            raise OSError("could not put the encoder into a job object")
        ctypes.windll.ntdll.NtResumeProcess(ctypes.c_void_p(int(proc._handle)))
        return proc

    def cpu_seconds(self):
        import ctypes

        class Accounting(ctypes.Structure):   # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
            _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                        ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                        ("TotalPageFaultCount", ctypes.c_uint32), ("TotalProcesses", ctypes.c_uint32),
                        ("ActiveProcesses", ctypes.c_uint32), ("TotalTerminatedProcesses", ctypes.c_uint32)]
        info = Accounting()
        ok = self.k32.QueryInformationJobObject(ctypes.c_void_p(self.job), 1, ctypes.byref(info),
                                                ctypes.sizeof(info), None)
        return (info.TotalUserTime + info.TotalKernelTime) / 1e7 if ok else None

    def close(self):
        import ctypes
        self.k32.CloseHandle(ctypes.c_void_p(self.job))


CPUS = None  # queue of logical CPU numbers (--cpus); each running encode is pinned to one of them


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


def pin(proc, cpu):
    """restrict a just-started child to one logical CPU (hybrid CPUs: keep every run on a P-core)"""
    import ctypes
    if not ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), ctypes.c_size_t(1 << cpu)):
        raise OSError("SetProcessAffinityMask failed")


def run_test(exe, outdir, name, more_args=(), fixed_core=None):
    cfg, yuv, w, h, fps, frames, qp, extra = TESTS[name]
    extra = list(extra) + list(more_args)
    bs = os.path.join(outdir, name + ".bin")
    log = os.path.join(outdir, name + ".log")
    cmd = [exe, "-c", f"{CFG}/{cfg}", "-i", f"{SEQ}/{yuv}", "-wdt", str(w), "-hgt", str(h), "-fr", str(fps),
           "--InputBitDepth=8", "-f", str(frames), "-q", str(qp), "-b", bs, "-o", "NUL"] + extra
    core = fixed_core if fixed_core is not None else (CPUS.get() if CPUS else None)
    try:
        t0 = time.perf_counter()
        with open(log, "w") as lf:
            if os.name == "nt":
                job = CpuJob()
                try:
                    proc = job.start(cmd, stdout=lf, stderr=subprocess.STDOUT)
                    if core is not None:
                        pin(proc, core)
                    proc.wait()
                    cpu = job.cpu_seconds()   # the encoder and all its child processes
                finally:
                    job.close()
            else:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
                if core is not None:
                    pin(proc, core)
                proc.wait()
                cpu = process_cpu_seconds(proc)
            rc = proc.returncode
        dt = time.perf_counter() - t0
    finally:
        if core is not None and fixed_core is None:
            CPUS.put(core)
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
    """Paired measurement: for every test, build A and build B run at the same time, so both see the same load.

    With --cpus the CPUs are used as pairs: A and B of a test run on the two CPUs of one pair, and repetition r swaps
    them (r odd), so a faster core does not favour one build. Times are then the mean over the repetitions."""
    names = args.tests.split(",") if args.tests else DEFAULT_TESTS
    res = {args.a: {}, args.b: {}}
    outdirs = {}
    for build in (args.a, args.b):
        for rep in range(args.reps):  # one folder per repetition: repetitions of a test may run at the same time
            outdirs[build, rep] = os.path.join(ROOT, "runs", "ab_" + build, f"rep{rep}")
            os.makedirs(outdirs[build, rep], exist_ok=True)

    def add(build, n, r):
        prev = res[build].get(n)
        if prev:  # md5 must agree over repetitions
            r["md5"] = r["md5"] if r["md5"] == prev["md5"] else None
            r["times"] = prev["times"] + [r["time"]]
            r["cpus"] = prev["cpus"] + [r["cpu"]]
        else:
            r["times"], r["cpus"] = [r["time"]], [r["cpu"]]
        res[build][n] = r
        print(f"{build:6s} {n:14s} {r['time']:9.2f}s  cpu {r['cpu']}  {r['md5']}", flush=True)

    ratios = {}
    if CPUS:
        cores = []
        while not CPUS.empty():
            cores.append(CPUS.get())
        pairs = queue.Queue()
        for k in range(0, len(cores) - 1, 2):
            pairs.put((cores[k], cores[k + 1]))

        def pair_job(n, rep):
            c = pairs.get()
            try:
                ca, cb = (c[1], c[0]) if rep % 2 else c
                with cf.ThreadPoolExecutor(2) as ex2:
                    fa = ex2.submit(run_test, f"{ROOT}/bin/{args.a}/EncoderApp.exe", outdirs[args.a, rep], n, (), ca)
                    fb = ex2.submit(run_test, f"{ROOT}/bin/{args.b}/EncoderApp.exe", outdirs[args.b, rep], n, (), cb)
                    return fa.result()[1], fb.result()[1], n, rep
            finally:
                pairs.put(c)

        with cf.ThreadPoolExecutor(len(cores) // 2) as ex:
            futs = [ex.submit(pair_job, n, rep) for rep in range(args.reps) for n in names]
            for fu in cf.as_completed(futs):
                ra, rb, n, rep = fu.result()
                ratios.setdefault(n, []).append((ra["cpu"] or 0) / (rb["cpu"] or 1))
                add(args.a, n, ra)
                add(args.b, n, rb)
        for build in res:
            for r in res[build].values():
                r["time"] = sum(r["times"]) / len(r["times"])
                r["cpu"] = sum(r["cpus"]) / len(r["cpus"]) if all(r["cpus"]) else None
    else:
        jobs = [(build, n, rep) for rep in range(args.reps) for n in names for build in (args.a, args.b)]
        with cf.ThreadPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(run_test, f"{ROOT}/bin/{build}/EncoderApp.exe", outdirs[build, rep], n): build
                    for build, n, rep in jobs}
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
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, f"ab_{args.a}_{args.b}.json"), "w"), indent=1, sort_keys=True)
    ref = json.load(open(os.path.join(RESULTS, "ref.json")))
    ok = True
    for build in (args.a, args.b):
        for n, r in res[build].items():
            ok &= r["md5"] is not None and r["md5"] == ref[n]["md5"]
    rc = compare(res[args.a], res[args.b], args.a, args.b)
    if ratios:
        # per repetition, A and B ran at the same time on the same core pair: the ratio cancels most of the load
        print(f"\nper-repetition CPU ratios {args.a}/{args.b} (median):")
        wa = wb = 0.0
        for n in sorted(ratios):
            r = sorted(ratios[n])
            med = r[len(r) // 2] if len(r) % 2 else (r[len(r) // 2 - 1] + r[len(r) // 2]) / 2
            print(f"  {n:14s} {med:6.3f}x   [{', '.join(f'{x:.3f}' for x in r)}]")
            wa += res[args.a][n]["cpu"] or 0
            wb += (res[args.a][n]["cpu"] or 0) / med
        print(f"  {'TOTAL':14s} {wa / wb:6.3f}x   (time-weighted)")
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
    for sp in (r, p):
        sp.add_argument("--cpus", help="comma-separated logical CPUs to pin runs to, one run per CPU (e.g. the P-cores)")
    args = ap.parse_args()
    if getattr(args, "cpus", None):
        global CPUS
        CPUS = queue.Queue()
        for c in args.cpus.split(","):
            CPUS.put(int(c))
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "ab":
        return cmd_ab(args)
    load = lambda n: json.load(open(os.path.join(RESULTS, n + ".json")))
    return compare(load(args.base), load(args.new), args.base, args.new)


if __name__ == "__main__":
    sys.exit(main())
