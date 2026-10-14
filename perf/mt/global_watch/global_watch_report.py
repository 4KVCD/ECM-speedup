#!/usr/bin/env python3
"""Map the output of the global-memory watch (GlobalWatch.h) to symbols and list, per phase, the globals written.

usage: global_watch_report.py global_watch.txt EncoderApp.exe   (the exe that wrote the file; mingw bin on PATH)
Prints for each phase (between, setup, search, finish) every written symbol with the number of pictures in which the
phase wrote it and the bytes written, and at the end the symbols written by the search and also by another phase.
"""
import bisect
import collections
import subprocess
import sys

IMAGE_BASE = 0x140000000

watch, exe = sys.argv[1], sys.argv[2]
syms = []
for line in subprocess.run(["nm", "-C", "-S", exe], capture_output=True, text=True).stdout.splitlines():
    parts = line.split(None, 3)
    if len(parts) == 4 and parts[2] in "bBdD":
        syms.append((int(parts[0], 16) - IMAGE_BASE, int(parts[1], 16), parts[3]))
    elif len(parts) == 3 and parts[1] in "bBdD":
        syms.append((int(parts[0], 16) - IMAGE_BASE, 0, parts[2]))
syms.sort()
starts = [s[0] for s in syms]


def lookup(rva):
    i = bisect.bisect_right(starts, rva) - 1
    if i < 0:
        return "?"
    start, size, name = syms[i]
    if size and rva >= start + size:
        return f"? after {name}"
    return name


pics = collections.defaultdict(lambda: collections.defaultdict(set))
nbytes = collections.defaultdict(lambda: collections.Counter())
for line in open(watch):
    phase, poc, rva, length = line.split()
    rva, length = int(rva, 16), int(length)
    # a run can cover several symbols
    pos, end = rva, rva + length
    while pos < end:
        name = lookup(pos)
        pics[phase][name].add(int(poc))
        i = bisect.bisect_right(starts, pos) - 1
        nxt = syms[i + 1][0] if i + 1 < len(syms) else end
        step = min(end, nxt) - pos
        nbytes[phase][name] += step
        pos += max(step, 1)

for phase in ("between", "setup", "search", "finish"):
    print(f"== {phase}")
    for name in sorted(pics[phase], key=lambda n: -len(pics[phase][n])):
        print(f"  {len(pics[phase][name]):4d} pics {nbytes[phase][name]:10d} B  {name[:160]}")
print("== written by the search and by another phase")
for name in sorted(pics["search"]):
    other = [p for p in ("between", "setup", "finish") if name in pics[p]]
    if other:
        print(f"  {name[:140]}  ({', '.join(other)})")
