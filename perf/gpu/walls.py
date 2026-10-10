# walls.py <build> <stream> [reps] -- configs separated by '|': best wall time per configuration (output not checked)
import subprocess, sys, time
build, stream = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "--" else 3
rest = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
configs, cur = [], []
for a in rest:
    if a == "|":
        configs.append(cur); cur = []
    else:
        cur.append(a)
configs.append(cur)
exe = f"C:/Code/bin/{build}/DecoderApp.exe"
for c in configs:
    best = 1e9
    for r in range(reps):
        t = time.perf_counter()
        subprocess.run([exe, "-b", f"C:/Code/streams_gpu/{stream}.bin", "-o", ""] + c, capture_output=True)
        best = min(best, time.perf_counter() - t)
    print(f"{stream:12s} {' '.join(c) or 'CPU':40s} {best:7.3f} s", flush=True)
