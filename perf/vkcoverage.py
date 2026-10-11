#!/usr/bin/env python3
"""Sums the 'GPU stages checked' counts of DecoderApp --Vulkan=1 --VulkanCheck=1 over the corpus.

usage: vkcoverage.py EXE [--only a,b] [--jobs N]
Shows which GPU stage paths the corpus exercises; a zero means that path was never compared with the CPU. Pictures
whose ALF stayed on the CPU are listed with the reason.
"""
import argparse
import concurrent.futures as cf
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import vkcorpus  # noqa: E402

LABELS = (["pictures", "deblocking segments"] + [f"SAO {t}" for t in ("EO0", "EO90", "EO135", "EO45", "BO")]
          + ["bilateral filter blocks"] + [f"CCSAO band {i}" for i in range(9)] + [f"CCSAO edge {i}" for i in range(4)]
          + ["ALF pictures"]
          + [f"ALF luma {t}" for t in ("fixed set 0", "fixed set 1", "13_EXT_DB_RESI_DIRECT", "13_EXT_DB_RESI",
                                       "13_DB_RESI_DIRECT", "13_DB_RESI")]
          + [f"ALF classifier {i}" for i in range(3)]
          + ["ALF luma scaling CTUs", "ALF chroma 9", "ALF chroma 9_NO_FIX", "CCALF CTUs", "ALF chroma scaling pictures",
             "ALF mirrored regions", "refinement after deblocking", "refinement after ALF"])


def run(exe, name):
    out = subprocess.run([exe, "-b", f"{vkcorpus.OUT}/{name}.bin", "-o", "NUL", "--Vulkan=1", "--VulkanCheck=1"],
                         capture_output=True, text=True, errors="replace")
    lines = out.stdout.splitlines()
    line = next((ln for ln in lines if "GPU stages checked" in ln), "")
    cpu = [ln.strip() for ln in lines if "on the CPU:" in ln]
    return name, out.returncode, [int(v) for v in re.findall(r"\d+", line)], cpu


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
        for name, rc, nums, cpu in ex.map(lambda n: run(args.exe, n), names):
            if rc != 0 or not nums:
                print(f"{name}: rc={rc}, no counts")
                continue
            total = nums if total is None else [a + b for a, b in zip(total, nums)]
            for c in cpu:
                print(f"{name}: {c}")
    if total:
        for label, v in zip(LABELS, total):
            print(f"{label:28s} {v:10d}{'   <-- never exercised' if v == 0 else ''}")


if __name__ == "__main__":
    main()
