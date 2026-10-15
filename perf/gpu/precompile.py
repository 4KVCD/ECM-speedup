# precompile.py <build> [-j N] [variant:level ...]: compiles intra_rec's pipelines of a build (every IntraRecVariant x
# LmLevel by default) into the NVIDIA driver's shader cache that partial.py and the regressions use (runs/nvcache),
# one DecoderApp process per pipeline, N at a time (default: the CPU's threads). A cold pipeline takes 30 s to several
# minutes and a decoding process creates the ones it needs one after the other, so after a shader change the test runs
# spent most of their time compiling; with this they start warm. Each process decodes a one-picture stream (partial
# verification: the GPU path starts on any stream) with ECM_GPU_PRECOMPILE=variant:level: GpuIntraRec's createShared
# compiles that pipeline and ends the process.
# The pipelines also go into files of their own (VkCompute's ECM_GPU_PIPELINE_CACHE, runs/pipecache), which partial.py
# reads: the driver's cache does not keep all of intra_rec's (an entry is ~100-350 MB). The oldest files go first where
# the directory holds more than PIPECACHE_GB.
import glob, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

PIPECACHE_GB = 30

NUM_VARIANTS, NUM_LEVELS = 9, 7   # GpuIntraRec.cpp's IntraRecVariant, LmLevel

args = sys.argv[1:]
build = args.pop(0)
jobs = os.cpu_count() or 8
if args and args[0] == '-j':
    jobs = int(args[1])
    args = args[2:]
pipelines = args or [f'{v}:{l}' for v in range(NUM_VARIANTS) for l in range(NUM_LEVELS)]
cache = 'C:/Code/ECM/runs/nvcache'
os.makedirs(cache, exist_ok=True)   # (the driver does not create it: without it nothing is cached)
pipes = 'C:/Code/ECM/runs/pipecache'
os.makedirs(pipes, exist_ok=True)
files = sorted(glob.glob(pipes + '/*.bin'), key=os.path.getmtime)
size = sum(os.path.getsize(f) for f in files)
while files and size > PIPECACHE_GB << 30:
    size -= os.path.getsize(files[0])
    os.remove(files.pop(0))
base = dict(os.environ, __GL_SHADER_DISK_CACHE_PATH=cache, __GL_SHADER_DISK_CACHE_SIZE='8000000000',
            __GL_SHADER_DISK_CACHE_SKIP_CLEANUP='1', ECM_GPU_PIPELINE_TIMES='1', ECM_GPU_PARTIAL='1',
            ECM_GPU_PIPELINE_CACHE=pipes)


def run(p):
    env = dict(base, ECM_GPU_PRECOMPILE=p)
    start = time.time()
    r = subprocess.run([f'C:/Code/ECM/bin/{build}/DecoderApp.exe', '-b', 'C:/Code/ECM/streams/ai_foreman.bin', '-o', '',
                        '--GpuRecon=2'], capture_output=True, text=True, env=env)
    times = [l for l in r.stderr.splitlines() if l.startswith('[gpu] pipeline intra_rec')]
    return p, r.returncode, time.time() - start, times


start = time.time()
with ThreadPoolExecutor(jobs) as ex:
    for p, rc, secs, times in ex.map(run, pipelines):
        print(f'{p:5s} rc {rc} {secs:6.1f} s  {times[0] if times else "(cached)"}', flush=True)
print(f'{len(pipelines)} pipelines in {time.time() - start:.0f} s')
