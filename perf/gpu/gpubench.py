# gpubench.py <build> [reps]: decodes every GPU test stream with --GpuRecon=2 `reps` times; prints per stream the
# GPU milliseconds per picture (best repetition), whether every picture was identical, and the totals
import os, re, subprocess, sys

ROOT = "C:/Code/ECM/streams_gpu"
build = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
exe = f"C:/Code/ECM/bin/{build}/DecoderApp.exe"
names = sorted(f[:-4] for f in os.listdir(ROOT) if f.endswith(".bin"))
tot = 0.0
totPics = 0
allok = True
for n in names:
    best = None
    for r in range(reps):
        out = subprocess.run([exe, "-b", f"{ROOT}/{n}.bin", "-o", "", "--GpuRecon=2"], capture_output=True, text=True).stdout
        m = re.search(r"\[gpu-intra\] (\d+) pictures: (\d+) verified \((\d+) with differences\), (\d+) skipped.*GPU ([\d.]+) ms", out)
        pics, ver, bad, skip, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4]), float(m[5])
        ok = bad == 0 and skip == 0 and ver == pics
        allok &= ok
        if not ok:
            print(f'  rep {r} of {n}: ' + m[0][:300])
        best = ms if best is None else min(best, ms)
    tot += best
    totPics += pics
    print(f"{n:14s} {pics} pictures  {best / pics:8.2f} ms/picture  {'identical' if ok else 'MISMATCH/SKIPPED'}")
print(f"TOTAL {totPics} pictures, {tot:.1f} ms GPU, {tot / totPics:.2f} ms/picture  {'ALL IDENTICAL' if allok else '*** NOT ALL IDENTICAL ***'}")
