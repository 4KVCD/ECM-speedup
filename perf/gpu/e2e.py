# e2e.py <build> <stream...>: wall-clock decode time (best of 3) per configuration; output checked against stock
import hashlib, subprocess, sys, time, os
build, streams = sys.argv[1], sys.argv[2:]
exe = f"C:/Code/bin/{build}/DecoderApp.exe"
ref = "C:/Code/bin/ref/DecoderApp.exe"
configs = [("CPU", []), ("CPU 8 frame threads", ["--FrameThreads=8"]), ("GPU on", ["--GpuRecon=1"]),
           ("GPU on 8 frame threads", ["--GpuRecon=1", "--FrameThreads=8"])]
out = "C:/Code/runs/gpucheck"
for s in streams:
    bs = f"C:/Code/streams_gpu/{s}.bin"
    subprocess.run([ref, "-b", bs, "-o", f"{out}/{s}.ref.yuv"], capture_output=True)
    refmd5 = hashlib.md5(open(f"{out}/{s}.ref.yuv", "rb").read()).hexdigest()
    base = None
    for name, args in configs:
        best = 1e9
        ok = True
        for r in range(3):
            t = time.perf_counter()
            subprocess.run([exe, "-b", bs, "-o", f"{out}/{s}.e2e.yuv"] + args, capture_output=True)
            best = min(best, time.perf_counter() - t)
            ok &= hashlib.md5(open(f"{out}/{s}.e2e.yuv", "rb").read()).hexdigest() == refmd5
        base = base or best
        print(f"{s:12s} {name:24s} {best:7.3f} s  {base / best:5.2f}x  {'identical' if ok else 'MISMATCH'}")
