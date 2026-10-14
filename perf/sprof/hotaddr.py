"""Sampled leaf addresses that map to a source line, and the disassembly around them.

usage: hotaddr.py EncoderApp.exe FILE:LINE samples.txt [samples.txt ...]
"""
import collections
import re
import subprocess
import sys


def main():
    exe, where, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    out = subprocess.run(["objdump", "-p", exe], capture_output=True, text=True).stdout
    base = int(re.search(r"ImageBase\s+([0-9a-fA-F]+)", out).group(1), 16)
    cnt = collections.Counter()
    for f in files:
        for line in open(f):
            p = line.split()
            if len(p) < 2 or not p[0].isdigit() or p[1].startswith("x:"):
                continue
            cnt[int(p[1], 16)] += int(p[0])
    addrs = [a for a, _ in cnt.most_common(20000)]
    res = subprocess.run(["addr2line", "-e", exe], input="\n".join(hex(a + base) for a in addrs), capture_output=True,
                         text=True).stdout.splitlines()
    hot = sorted((a + base, cnt[a]) for a, l in zip(addrs, res) if where in l)
    print(len(hot), "addresses,", sum(n for _, n in hot), "samples")
    if not hot:
        return
    lo, hi = hot[0][0], hot[-1][0]
    dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", f"--start-address={hex(lo - 64)}",
                          f"--stop-address={hex(hi + 64)}", exe], capture_output=True, text=True).stdout
    hits = dict(hot)
    for line in dis.splitlines():
        m = re.match(r"\s*([0-9a-f]+):", line)
        n = hits.get(int(m.group(1), 16), 0) if m else 0
        print(f"{n:5d} {line}" if n else f"      {line}")


if __name__ == "__main__":
    main()
