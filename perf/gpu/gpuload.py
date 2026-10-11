# gpuload.py <build> <seconds>: keeps the GPU busy for stress tests - decodes m3_park60 with the GPU reconstruction on
# and 8 frame threads (8 decoder contexts with 32 persistent workgroups each) in a loop. Run it beside stress.py:
# bugs of GPU synchronization showed up only while another process shared the GPU.
import subprocess, sys, time

build, seconds = sys.argv[1], float(sys.argv[2])
exe = f"C:/Code/ECM/bin/{build}/DecoderApp.exe"
end = time.time() + seconds
n = 0
while time.time() < end:
    subprocess.run([exe, "-b", "C:/Code/ECM/streams_gpu/m3_park60.bin", "--GpuRecon=3", "--FrameThreads=8"],
                   capture_output=True)
    n += 1
print(f"{n} load decodes")
