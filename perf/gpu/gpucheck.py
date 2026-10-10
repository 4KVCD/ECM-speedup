# gpucheck.py <build> [stream names...] [-- extra decoder args]
# decodes C:/Code/streams_gpu/<name>.bin with the stock decoder (cached) and with C:/Code/bin/<build>/DecoderApp.exe,
# compares every frame; prints the decoder's own timing lines
import hashlib, os, subprocess, sys, time

ROOT = "C:/Code/streams_gpu"
REF = "C:/Code/bin/ref/DecoderApp.exe"
OUT = "C:/Code/runs/gpucheck"

def frames(path, w, h):
    size = w * h * 3 // 2 * 2   # 4:2:0, 16-bit output
    with open(path, "rb") as f:
        data = f.read()
    return [hashlib.md5(data[i:i + size]).hexdigest() for i in range(0, len(data), size)]

def dims(log):
    # "POC ..." lines don't carry the size; the encoder log does
    for line in open(log, errors="replace"):
        if "Real     Format" in line or "Input          File" in line:
            pass
    return None

def decode(exe, stream, yuv, log, extra):
    t = time.perf_counter()
    with open(log, "w") as f:
        rc = subprocess.call([exe, "-b", stream, "-o", yuv, "-d", "0"] + extra, stdout=f, stderr=subprocess.STDOUT)
    return rc, time.perf_counter() - t

def main():
    args = sys.argv[1:]
    extra = []
    if "--" in args:
        i = args.index("--")
        args, extra = args[:i], args[i + 1:]
    build, names = args[0], args[1:]
    if not names:
        names = sorted(f[:-4] for f in os.listdir(ROOT) if f.endswith(".bin"))
    os.makedirs(OUT, exist_ok=True)
    exe = f"C:/Code/bin/{build}/DecoderApp.exe"
    allok = True
    for n in names:
        stream = f"{ROOT}/{n}.bin"
        ref = f"{OUT}/{n}.ref.yuv"
        if not os.path.exists(ref) or os.path.getmtime(ref) < os.path.getmtime(stream):
            rc, _ = decode(REF, stream, ref, f"{OUT}/{n}.ref.log", [])
            if rc:
                print(f"{n}: stock decoder failed ({rc})"); allok = False; continue
        test = f"{OUT}/{n}.{build}.yuv"
        rc, sec = decode(exe, stream, test, f"{OUT}/{n}.{build}.log", extra)
        a, b = open(ref, "rb").read(), open(test, "rb").read() if os.path.exists(test) else b""
        ok = rc == 0 and a == b
        allok &= ok
        extra_lines = [l.strip() for l in open(f"{OUT}/{n}.{build}.log", errors="replace") if l.startswith("[gpu") or "Total Time" in l]
        print(f"{n:16s} {sec:7.2f}s  {'ok' if ok else f'MISMATCH (rc {rc}, {len(a)} vs {len(b)} bytes)'}  " + " | ".join(extra_lines[-3:]))
        if not ok and len(a) == len(b) and a:
            # first differing byte -> frame
            i = next(k for k in range(len(a)) if a[k] != b[k])
            print(f"   first difference at byte {i}")
    print("ALL MATCH" if allok else "*** MISMATCH ***")
    sys.exit(0 if allok else 1)

main()
