# gpucheck.py [-j N] <build> [stream names...] [-- extra decoder args]
# Decodes C:/Code/ECM/streams_gpu/<name>.bin with the stock decoder (its output cached) and with
# C:/Code/ECM/bin/<build>/DecoderApp.exe, compares the two outputs byte for byte and, with --GpuRecon=2, requires the
# GPU's verification to report no differences. Prints one line per stream, in the given order, with the decoder's own
# summary lines, then ALL MATCH or *** MISMATCH ***. -j N decodes N streams at a time (default 1, which timings need).
import concurrent.futures, os, re, subprocess, sys, time

STREAMS = "C:/Code/ECM/streams_gpu"
STOCK_DECODER = "C:/Code/ECM/bin/ref/DecoderApp.exe"
OUT = "C:/Code/ECM/runs/gpucheck"


def decode(exe, stream, yuv, log, extra):
    """Decodes stream into yuv with exe, its console output into log; returns the exit code and the seconds taken."""
    start = time.perf_counter()
    with open(log, "w") as f:
        rc = subprocess.call([exe, "-b", stream, "-o", yuv, "-d", "0"] + extra, stdout=f, stderr=subprocess.STDOUT)
    return rc, time.perf_counter() - start


def check_stream(name, build, exe, extra):
    """Checks one stream; returns whether it passed and the lines reporting it."""
    stream = f"{STREAMS}/{name}.bin"
    ref = f"{OUT}/{name}.ref.yuv"
    # the stock decoder's output, decoded again only where the stream is newer
    if not os.path.exists(ref) or os.path.getmtime(ref) < os.path.getmtime(stream):
        rc, _ = decode(STOCK_DECODER, stream, ref, f"{OUT}/{name}.ref.log", [])
        if rc:
            return False, [f"{name}: stock decoder failed ({rc})"]
    test = f"{OUT}/{name}.{build}.yuv"
    log_path = f"{OUT}/{name}.{build}.log"
    rc, seconds = decode(exe, stream, test, log_path, extra)
    expected = open(ref, "rb").read()
    actual = open(test, "rb").read() if os.path.exists(test) else b""
    log = open(log_path, errors="replace").read()
    # verification (--GpuRecon=2) outputs the CPU's pictures: the GPU's differences are only in the summary lines
    # (one per decoder context with frame threads)
    gpu_diffs = sum(int(d) for d in re.findall(r"verified \((\d+) with differences\)", log))
    ok = rc == 0 and expected == actual and gpu_diffs == 0
    if ok:
        status = "ok"
    elif rc == 0 and expected == actual:
        status = f"{gpu_diffs} GPU PICTURES DIFFER"
    else:
        status = f"MISMATCH (rc {rc}, {len(expected)} vs {len(actual)} bytes)"
    summary = [l.strip() for l in log.splitlines() if l.startswith("[gpu") or "Total Time" in l]
    lines = [f"{name:16s} {seconds:7.2f}s  {status}  " + " | ".join(summary[-3:])]
    if not ok and expected and len(expected) == len(actual) and expected != actual:
        first = next(k for k in range(len(expected)) if expected[k] != actual[k])
        lines.append(f"   first difference at byte {first}")
    return ok, lines


def main():
    args = sys.argv[1:]
    extra = []
    if "--" in args:
        split = args.index("--")
        args, extra = args[:split], args[split + 1:]
    jobs = 1
    if args[:1] == ["-j"]:
        jobs, args = int(args[1]), args[2:]
    build, names = args[0], args[1:]
    if not names:
        names = sorted(f[:-4] for f in os.listdir(STREAMS) if f.endswith(".bin"))
    os.makedirs(OUT, exist_ok=True)
    exe = f"C:/Code/ECM/bin/{build}/DecoderApp.exe"
    all_ok = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        # map keeps the given order, whatever order the decodes finish in
        for ok, lines in pool.map(lambda name: check_stream(name, build, exe, extra), names):
            all_ok &= ok
            print("\n".join(lines), flush=True)
    print("ALL MATCH" if all_ok else "*** MISMATCH ***")
    sys.exit(0 if all_ok else 1)


main()
