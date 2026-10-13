#!/usr/bin/env python3
"""Frame-thread output check: encodes multi-GOP clips with a test exe at several FrameThreads counts, writing the
reconstruction, and compares bitstream and reconstruction md5s with the single-thread reference: a reference exe's
encode, or the "ref" lines of a text file (p5_refs.txt: stock-identical single-thread encodes).
Prints the time and the peak commit of each run (a job object around the encoder).
usage: p5_test.py REFEXE|REFS.txt EXE TAG tests(comma) threads(comma) [extra encoder args...]
       (P5_MASK=hex: CPU mask of the encoders)"""
import ctypes
import hashlib
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import bench  # noqa: E402

T = {   # cfg, sequence, width, height, frames, QP[, extra args]
    "ai12": ("encoder_intra_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 12, 32),
    "ai33s8": ("encoder_intra_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 257, 32, ["--TemporalSubsampleRatio=8"]),
    "ra20": ("encoder_randomaccess_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 20, 32),
    "ra33": ("encoder_randomaccess_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 33, 32),
    "ra65": ("encoder_randomaccess_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 65, 32),
    "ra129": ("encoder_randomaccess_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 129, 32),
    "ra257": ("encoder_randomaccess_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 257, 32),
    "ld13": ("encoder_lowdelay_ecm.cfg", "foreman_128x128_257.yuv", 128, 128, 13, 32),
}
PEAK = [0.0]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(exe, test, args, out):
    cfg, yuv, w, h, frames, qp = T[test][:6]
    more = list(T[test][6]) if len(T[test]) > 6 else []
    os.makedirs(out, exist_ok=True)
    cmd = [exe, "-c", f"{bench.CFG}/{cfg}", "-i", f"{bench.SEQ}/{yuv}", "-wdt", str(w), "-hgt", str(h), "-fr", "30",
           "--InputBitDepth=8", "-f", str(frames), "-q", str(qp), "-b", out + "/s.bin", "-o", out + "/r.yuv"] + more + args
    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    t0 = time.perf_counter()
    with open(out + "/log.txt", "w") as log:
        # created suspended, put in a job object (peak commit) and on the CPU mask, then resumed
        job = k32.CreateJobObjectW(None, None)
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=0x4)
        k32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(int(p._handle)))
        mask = os.environ.get("P5_MASK")
        if mask:
            k32.SetProcessAffinityMask(int(p._handle), ctypes.c_size_t(int(mask, 16)))
        ctypes.windll.ntdll.NtResumeProcess(ctypes.c_void_p(int(p._handle)))
        rc = p.wait()
        info = (ctypes.c_size_t * 18)()
        k32.QueryInformationJobObject(ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info), None)
        PEAK[0] = info[17] / 2**20   # PeakJobMemoryUsed (JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
        k32.CloseHandle(ctypes.c_void_p(job))
    dt = time.perf_counter() - t0
    ok = rc == 0 and os.path.exists(out + "/s.bin")
    return (md5(out + "/s.bin") if ok else None, md5(out + "/r.yuv") if ok else None, dt, rc)


def main():
    refexe, exe, tag = sys.argv[1:4]
    tests = sys.argv[4].split(",")
    threads = sys.argv[5].split(",")
    extra = sys.argv[6:]
    out_root = os.path.join(bench.ROOT, "runs", "p5", tag)
    cached = {}
    if refexe.endswith(".txt"):
        for line in open(refexe):
            f = line.split()
            if len(f) >= 6 and f[1] == "ref":
                cached[f[0]] = (f[4], f[5], float(f[2]), 0)
    for test in tests:
        ref = cached[test] if cached else run(refexe, test, [], os.path.join(out_root, f"{test}_ref"))
        print(f"{test:8s} ref        {ref[2]:8.1f} s  {ref[0]}  {ref[1]}", flush=True)
        for n in threads:
            r = run(exe, test, [f"--FrameThreads={n}"] + extra, os.path.join(out_root, f"{test}_ft{n}"))
            same = "MATCH" if r[0] == ref[0] and r[1] == ref[1] else f"*** MISMATCH (rc {r[3]}) ***"
            print(f"{test:8s} FT={n:2s}      {r[2]:8.1f} s  {ref[2] / r[2]:5.2f}x  {r[0]}  {r[1]}  {same}  "
                  f"peak {PEAK[0]:6.0f} MB", flush=True)


if __name__ == "__main__":
    main()
