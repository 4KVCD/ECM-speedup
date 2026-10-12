# enc_crops.py <prefix> <count> <seed> [encoder args...]: <count> GPU test streams <prefix>_<i> of 64x64 crops (8
# frames) at random positions of the test sequences and random QPs (22-37), 4 encoders at a time (enc_m1.ps1; the crop
# sources go to C:\Code\ECM\seq\crops). Small pictures put many blocks at the picture's edges, where the reference and
# bilateral-filter padding paths differ. Deterministic for a seed: e.g. the M13 crops are
#   enc_crops.py m13c 48 13 <m13's arguments>
import os, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [("C:/Code/ECM/seq/foreman_352x288.yuv", 352, 288), ("C:/Code/ECM/seq/ParkJoy_1280x720.yuv", 1280, 720),
       ("C:/Code/ECM/seq/FourPeople_1280x720.yuv", 1280, 720), ("C:/Code/ECM/seq/CrowdRun_1920x1080.yuv", 1920, 1080),
       ("C:/Code/ECM/seq/Ducks_1920x1080.yuv", 1920, 1080)]

prefix, count, seed, extra = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
os.makedirs("C:/Code/ECM/seq/crops", exist_ok=True)
rnd = random.Random(seed)
jobs = []
for i in range(count):
    src, w, h = rnd.choice(SRC)
    x, y = rnd.randrange(0, w - 64, 2), rnd.randrange(0, h - 64, 2)
    qp = rnd.choice([22, 27, 32, 37])
    yuv = f"C:/Code/ECM/seq/crops/{prefix}_{i}_64x64.yuv"
    subprocess.check_call([sys.executable, f"{HERE}/crop.py", src, str(w), str(h), str(x), str(y), "64", "64", "8", yuv])
    jobs.append((f"{prefix}_{i}", yuv, qp))
running = []
for name, yuv, qp in jobs:
    running.append(subprocess.Popen(["powershell", "-NoProfile", "-File", f"{HERE}/enc_m1.ps1", name, yuv, "64", "64", "8",
                                     "--TemporalSubsampleRatio=1"] + extra + ["-q", str(qp)], stdout=subprocess.DEVNULL))
    if len(running) == 4:
        running.pop(0).wait()
for p in running:
    p.wait()
print(" ".join(n for n, _, _ in jobs))
