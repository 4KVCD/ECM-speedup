import os, subprocess, sys, threading, time

# nvvmprobe.py <build> [stream] [seconds]: whether the driver's compile of the build's intra_rec gets past the point
# where an NVVM failure ends the decoder (~160 s): runs one partial decode under gpuslot's watchdog, and if the decoder
# still runs after `seconds` the compile got further (the decoder - this script's child - is then stopped by its PID)
import gpuslot   # (the memory budget: one decoder slot)

build = sys.argv[1]
stream = sys.argv[2] if len(sys.argv) > 2 else 'ai_foreman'
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 240
env = dict(os.environ, ECM_GPU_PARTIAL='1', ECM_GPU_FULL_PIPELINE='1',
           ECM_GPU_PIPELINE_CACHE='C:/Code/ECM/runs/pipecache', __GL_SHADER_DISK_CACHE_PATH='C:/Code/ECM/runs/nvcache')
exe = f'C:/Code/ECM/bin/{build}/DecoderApp.exe'
t0 = time.time()
with gpuslot.slot():
    p = subprocess.Popen([exe, '-b', f'C:/Code/ECM/streams/{stream}.bin', '-o', '', '--GpuRecon=2'], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    threading.Thread(target=lambda: out.extend(p.stdout), daemon=True).start()
    while p.poll() is None and time.time() - t0 < limit:
        time.sleep(2)
    if p.poll() is None:
        print(f'still compiling after {time.time() - t0:.0f} s: NVVM got past the failure point; stopping PID {p.pid}')
        subprocess.run(['taskkill', '/PID', str(p.pid), '/F'], capture_output=True)
    else:
        text = ''.join(out)
        print(f'exited {p.returncode} after {time.time() - t0:.0f} s:',
              'NVVM failed' if 'NVVM compilation failed' in text else text[-300:])
