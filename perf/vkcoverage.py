#!/usr/bin/env python3
"""Sums the 'GPU stages checked' counts of DecoderApp --Vulkan=1 --VulkanCheck=1 over the corpus.

usage: vkcoverage.py EXE [--only a,b] [--jobs N]
Shows which GPU stage paths the corpus exercises; a zero means that path was never compared with the CPU.
"""
import argparse
import concurrent.futures as cf
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import vkcorpus  # noqa: E402


def run(exe, name):
    out = subprocess.run([exe, "-b", f"{vkcorpus.OUT}/{name}.bin", "-o", "NUL", "--Vulkan=1", "--VulkanCheck=1"],
                         capture_output=True, text=True, errors="replace")
    line = next((ln for ln in out.stdout.splitlines() if "GPU stages checked" in ln), "")
    return name, out.returncode, [int(v) for v in re.findall(r"\d+", line)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    m = vkcorpus.load_manifest()
    names = [n for n in vkcorpus.selected(args.only) if n in m]
    total = None
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for name, rc, nums in ex.map(lambda n: run(args.exe, n), names):
            if rc != 0 or not nums:
                print(f"{name}: rc={rc}, no counts")
                continue
            total = nums if total is None else [a + b for a, b in zip(total, nums)]
    if total:
        labels = ["pictures", "deblocking segments"] + [f"SAO {t}" for t in ("EO0", "EO90", "EO135", "EO45", "BO")] + \
                 ["bilateral filter blocks"] + [f"CCSAO band {i}" for i in range(9)] + [f"CCSAO edge {i}" for i in range(4)]
        for label, v in zip(labels, total):
            print(f"{label:24s} {v:10d}{'   <-- never exercised' if v == 0 else ''}")


if __name__ == "__main__":
    main()
