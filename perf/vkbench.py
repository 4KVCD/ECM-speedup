#!/usr/bin/env python3
"""Paired A/B timing of GPU decoder builds: total time and GPU time per stage.

usage: vkbench.py EXE [EXE ...] [--stream S] [--reps N] [--args "..."] [--cpu C]

Runs the decoders alternately on one stream (default: C:/Code/runs/decbig/ra_cif.bin), pinned to one logical CPU,
and prints the minimum and median of the total time and of each stage's GPU time. The iGPU's clock varies from run
to run, so the minimum over several runs is the more stable figure.
"""
import argparse
import ctypes
import re
import statistics
import subprocess


def run(exe, stream, extra, cpu):
    cmd = [exe, "-b", stream, "-o", "NUL", *extra]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    ctypes.windll.kernel32.SetProcessAffinityMask(int(p._handle), ctypes.c_size_t(1 << cpu))
    out, _ = p.communicate()
    if p.returncode != 0:
        raise SystemExit(f"{exe} failed:\n{out[-2000:]}")
    res = {}
    m = re.search(r"Total Time:\s+([\d.]+)", out)
    res["total"] = float(m.group(1))
    m = re.search(r"GPU time:\s+([\d.]+)", out)
    if m:
        res["GPU"] = float(m.group(1))
    m = re.search(r"GPU time by stage \(sec\.\): (.*)", out)
    if m:
        for name, val in re.findall(r"([^,]+?) ([\d.]+)(?:,|$)", m.group(1)):
            res[name.strip()] = float(val)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exes", nargs="+")
    ap.add_argument("--stream", default="C:/Code/runs/decbig/ra_cif.bin")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--args", default="--Vulkan=1")
    ap.add_argument("--cpu", type=int, default=0)
    a = ap.parse_args()
    extra = a.args.split()
    results = {exe: [] for exe in a.exes}
    for _ in range(a.reps):
        for exe in a.exes:
            results[exe].append(run(exe, a.stream, extra, a.cpu))
    keys = list(results[a.exes[0]][0].keys())
    print(f"{'':32s}" + "".join(f"{exe[-40:]:>44s}" for exe in a.exes))
    for k in keys:
        cells = []
        for exe in a.exes:
            vals = [r.get(k, 0.0) for r in results[exe]]
            cells.append(f"min {min(vals):8.3f}  median {statistics.median(vals):8.3f}")
        print(f"{k:32s}" + "".join(f"{c:>44s}" for c in cells))


if __name__ == "__main__":
    main()
