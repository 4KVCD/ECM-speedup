# dbf_verify.py <build> [-j N] [stream...]: the GPU loop filters' verification (--GpuLoopFilter=2) of the streams in
# C:/Code/ECM/streams (default: all), each decoder under gpuslot.run: per stream the deblocked pictures the GPU
# deblocked and compared with the CPU's (GpuAlf::verifyDeblocking), those with differences and those it did not take
# (the first differing samples are in the log), the ALF / SAO verification's summary line, and the decoder's exit code.
# Logs in runs/dbf/<stream>.<build>.log. With --on: ON mode (--GpuLoopFilter=3, where the GPU deblocks inside the
# ALF stage) instead, the output compared byte for byte with the same build's CPU decoding (its md5 cached in
# runs/dbf; this PC has no stock decoder).
import hashlib, os, re, subprocess, sys
import gpuslot
from concurrent.futures import ThreadPoolExecutor

args = sys.argv[1:]
on = '--on' in args
args = [a for a in args if a != '--on']
build = args.pop(0)
jobs = 1
if args and args[0] == '-j':
    jobs = int(args[1])
    args = args[2:]
streams = args or sorted(f[:-4] for f in os.listdir('C:/Code/ECM/streams') if f.endswith('.bin'))
pipes = 'C:/Code/ECM/runs/pipecache'
os.makedirs(pipes, exist_ok=True)
env = dict(os.environ, ECM_GPU_PIPELINE_CACHE=pipes)
logs = 'C:/Code/ECM/runs/dbf'
os.makedirs(logs, exist_ok=True)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def dbfLine(name):
    log = open(f'{logs}/{name}.{build}.on.log').read() if os.path.exists(f'{logs}/{name}.{build}.on.log') else ''
    m = re.search(r'GPU deblocking: [^\n]*', log)
    return m.group(0) if m else 'no GPU deblocking'


def runOn(name):
    ref = f'{logs}/{name}.{build}.cpu.md5'
    if not os.path.exists(ref):   # the CPU's decoding
        subprocess.run([f'C:/Code/ECM/bin/{build}/DecoderApp.exe', '-b', f'C:/Code/ECM/streams/{name}.bin', '-o',
                        f'{logs}/{name}.ref.yuv', '-d', '0'], capture_output=True)
        open(ref, 'w').write(md5(f'{logs}/{name}.ref.yuv'))
        os.remove(f'{logs}/{name}.ref.yuv')   # (named ref: the build's CPU output)
    yuv = f'{logs}/{name}.{build}.on.yuv'
    rc, out, err = gpuslot.run([f'C:/Code/ECM/bin/{build}/DecoderApp.exe', '-b', f'C:/Code/ECM/streams/{name}.bin', '-o',
                                yuv, '-d', '0', '--GpuLoopFilter=3'], env)
    open(f'{logs}/{name}.{build}.on.log', 'w').write(out + '\n--- stderr ---\n' + err)
    same = rc == 0 and os.path.exists(yuv) and md5(yuv) == open(ref).read()
    if os.path.exists(yuv):
        os.remove(yuv)
    return name, rc, same


def run(name):
    rc, out, err = gpuslot.run([f'C:/Code/ECM/bin/{build}/DecoderApp.exe', '-b', f'C:/Code/ECM/streams/{name}.bin', '-o',
                                '', '--GpuLoopFilter=2'], env)
    open(f'{logs}/{name}.{build}.log', 'w').write(out + '\n--- stderr ---\n' + err)
    d = re.search(r'GPU deblocking verification: (\d+) pictures, (\d+) with differences, (\d+) not on the GPU; GPU ([\d.]+) ms', err)
    a = re.search(r'GPU loop filter verification: ([^\n]*)', err)
    first = re.search(r'GPU deblocking POC [^\n]*', err)
    return name, rc, d.groups() if d else None, a.group(1) if a else '', first.group(0) if first else ''


bad = 0
if on:
    with ThreadPoolExecutor(jobs) as pool:
        for name, rc, same in pool.map(runOn, streams):
            bad += not same
            print(f'{name:18s} {"identical" if same else "MISMATCH"}{"" if rc == 0 else f"; exit code {rc}"}; {dbfLine(name)}', flush=True)
    print(f'{len(streams) - bad} of {len(streams)} streams identical with the CPU decoding', flush=True)
    sys.exit(0)
with ThreadPoolExecutor(jobs) as pool:
    for name, rc, d, a, first in pool.map(run, streams):
        ok = rc == 0 and d is not None and d[1] == '0'
        bad += not ok
        dbf = f'deblocked {d[0]}, differ {d[1]}, not on the GPU {d[2]}, GPU {d[3]} ms per picture' if d else 'no deblocking summary'
        print(f'{name:18s} {dbf}; ALF: {a or "-"}{"" if rc == 0 else f"; exit code {rc}"}', flush=True)
        if first:
            print(f'{"":18s} first: {first[:300]}', flush=True)
print(f'{len(streams) - bad} of {len(streams)} streams without differences', flush=True)
