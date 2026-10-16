# partial.py <build> [-j N] [stream...]: partial verification (ECM_GPU_PARTIAL=1 with --GpuRecon=2) of the regular
# streams in C:/Code/ECM/streams (default: all), in parallel: the CUs of tools the GPU path lacks come from the CPU's
# reconstruction, everything else is verified. Per stream the pictures verified / with differences, the first
# difference (the first block in decoding order whose prediction or residual differs is in the log), and the CUs taken
# from the CPU by reason (what to port next); then the reasons summed over all streams. Logs in runs/partial.
# The NVIDIA driver's shader cache: runs/nvcache, a cache of its own (the shared one stops storing at 1 GB); the
# pipelines' own files (VkCompute's ECM_GPU_PIPELINE_CACHE, which precompile.py fills): runs/pipecache.
import os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

args = sys.argv[1:]
build = args.pop(0)
jobs = 4
if args and args[0] == '-j':
    jobs = int(args[1])
    args = args[2:]
streams = args or sorted(f[:-4] for f in os.listdir('C:/Code/ECM/streams') if f.endswith('.bin'))
cache = 'C:/Code/ECM/runs/nvcache'
os.makedirs(cache, exist_ok=True)   # (the driver does not create it: without it nothing is cached)
pipes = 'C:/Code/ECM/runs/pipecache'
os.makedirs(pipes, exist_ok=True)
env = dict(os.environ, ECM_GPU_PARTIAL='1', __GL_SHADER_DISK_CACHE_PATH=cache,
           __GL_SHADER_DISK_CACHE_SIZE='8000000000', __GL_SHADER_DISK_CACHE_SKIP_CLEANUP='1',
           ECM_GPU_PIPELINE_CACHE=pipes)
total = Counter()


def run(name):
    p = subprocess.run([f'C:/Code/ECM/bin/{build}/DecoderApp.exe', '-b', f'C:/Code/ECM/streams/{name}.bin', '-o', '',
                        '--GpuRecon=2'], capture_output=True, text=True, env=env)
    out = p.stdout
    if p.returncode != 0:   # (a decoder that failed: its exit code and error output, in the log and the report)
        out += f'\n[exit code {p.returncode}] ' + ' | '.join(p.stderr.strip()[-500:].splitlines()) + '\n'
    os.makedirs('C:/Code/ECM/runs/partial', exist_ok=True)
    open(f'C:/Code/ECM/runs/partial/{name}.{build}.log', 'w').write(out)
    s = re.search(r'\[gpu-intra\] (\d+) pictures: (\d+) verified \((\d+) with differences\), (\d+) skipped([^;]*)', out)
    first = next((m.group(1) for m in re.finditer(r' ms: (MISMATCH[^\n]*?)POC', out)), '')
    reasons = Counter()
    m = re.search(r'CUs taken from the CPU:(.*)', out)
    if m:
        for part in m.group(1).split(';'):
            part = part.strip()
            if part:
                k, n = part.rsplit(' ', 1)
                reasons[k] += int(n)
    failure = re.search(r'\[exit code [^\n]*', out)
    return name, s.groups() if s else None, first, reasons, failure.group(0) if failure else ''


def report(name, s, first, reasons, failure):
    total.update(reasons)
    print(f'{name:16s} {"pictures %s: verified %s, differ %s, skipped %s%s" % s if s else "no summary"}', flush=True)
    if failure:
        print(f'{"":16s} {failure[:400]}', flush=True)
    if first:
        print(f'{"":16s} first: {first[:400]}', flush=True)


# a stream whose GPU device was lost (VK_ERROR_DEVICE_LOST, -4): Windows resets a GPU whose work runs past its
# timeout (2 s), which pictures of ~1.8 s alone (pj720_ai) reach when 8 decoders share the GPU; such a stream runs again
# alone after the others
lost = []
with ThreadPoolExecutor(jobs) as ex:
    for result in ex.map(run, streams):
        if 'VkResult -4' in result[4]:
            lost.append(result[0])
            print(f'{result[0]:16s} GPU device lost under the parallel load: run again alone below', flush=True)
            continue
        report(*result)
for name in lost:
    report(*run(name))
print('CUs from the CPU, all streams:', '; '.join(f'{k} {n}' for k, n in total.most_common()))
