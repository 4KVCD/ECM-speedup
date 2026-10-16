# verify_all.py [-j N] <build> [streams...] [-- decoder args]: the full GPU regression - one --GpuRecon=2 decode
# (verification: the GPU's reconstruction compared block by block with the CPU's) of each test stream in
# C:/Code/ECM/streams_gpu (default: all 444), N at a time; reports every stream with GPU differences (and its first
# difference line), crashes, and the totals of verified / skipped pictures. Baseline at M66 (grW1): 444 streams,
# 1573 pictures, 1565 verified, 0 with differences, 8 skipped. The decoder is C:/Code/ECM/bin/<build>/DecoderApp.exe;
# the shader caches are partial.py's (run precompile.py after a shader change first).
import concurrent.futures, os, re, subprocess, sys

args = sys.argv[1:]
extra = ["--GpuRecon=2"]
if "--" in args:
    i = args.index("--")
    args, extra = args[:i], extra + args[i + 1:]
jobs = 4
if args[:1] == ["-j"]:
    jobs, args = int(args[1]), args[2:]
build, names = args[0], args[1:]
root = "C:/Code/ECM/streams_gpu"
if not names:
    names = sorted(f[:-4] for f in os.listdir(root) if f.endswith(".bin"))
exe = f"C:/Code/ECM/bin/{build}/DecoderApp.exe"
env = dict(os.environ, __GL_SHADER_DISK_CACHE_PATH="C:/Code/ECM/runs/nvcache", __GL_SHADER_DISK_CACHE_SIZE="8000000000",
           __GL_SHADER_DISK_CACHE_SKIP_CLEANUP="1", ECM_GPU_PIPELINE_CACHE="C:/Code/ECM/runs/pipecache")


def run(n):
    p = subprocess.run([exe, "-b", f"{root}/{n}.bin", "-o", "", *extra], capture_output=True, text=True, env=env)
    return n, p


tot = [0, 0, 0, 0]
bad = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
    for n, p in pool.map(run, names):
        m = re.search(r"\[gpu-intra\] (\d+) pictures: (\d+) verified \((\d+) with differences\), (\d+) skipped",
                      p.stdout)
        if p.returncode != 0 or not m:
            bad += 1
            print(f"FAIL {n}: rc {p.returncode} " + (m[0] if m else "no summary") + " " + p.stderr.strip()[-200:],
                  flush=True)
            continue
        for k in range(4):
            tot[k] += int(m[k + 1])
        if int(m[3]) != 0:
            bad += 1
            first = next((l for l in p.stdout.splitlines() if "[gpu-intra] POC" in l and "identical" not in l), "")
            print(f"DIFF {n}: {m[0]} | {first[:300]}", flush=True)
print(f"{len(names)} streams, {bad} with problems; pictures {tot[0]}, verified {tot[1]}, with differences {tot[2]}, "
      f"skipped {tot[3]}", flush=True)
