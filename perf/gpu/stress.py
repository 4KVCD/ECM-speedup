# stress.py <build> <reps> [extra decoder args]: decodes every GPU test stream <reps> times with --GpuRecon=2 (or the
# given args); saves the full output of any decode whose summary is not "all verified, 0 differences, 0 skipped"
import os, re, subprocess, sys
ROOT = "C:/Code/streams_gpu"
OUT = "C:/Code/runs/stress"
build, reps = sys.argv[1], int(sys.argv[2])
extra = sys.argv[3:] or ["--GpuRecon=2"]
os.makedirs(OUT, exist_ok=True)
exe = f"C:/Code/bin/{build}/DecoderApp.exe"
names = sorted(f[:-4] for f in os.listdir(ROOT) if f.endswith(".bin"))
bad = 0
runs = 0
for r in range(reps):
    for n in names:
        p = subprocess.run([exe, "-b", f"{ROOT}/{n}.bin", "-o", "", *extra], capture_output=True, text=True)
        runs += 1
        m = re.search(r"\[gpu-intra\] (\d+) pictures: (\d+) verified \((\d+) with differences\), (\d+) skipped", p.stdout)
        ok = p.returncode == 0 and m and int(m[3]) == 0 and int(m[4]) == 0 and int(m[2]) == int(m[1])
        if not ok:
            bad += 1
            path = f"{OUT}/{n}_rep{r}.txt"
            open(path, "w").write(f"rc {p.returncode}\n" + p.stdout + "\n--- stderr\n" + p.stderr)
            print(f"FAIL rep {r} {n}: rc {p.returncode} " + (m[0] if m else "no summary"), flush=True)
print(f"{runs} decodes, {bad} failures", flush=True)
