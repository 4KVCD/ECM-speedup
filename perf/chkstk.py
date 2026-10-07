#!/usr/bin/env python3
"""List the functions of an executable that probe their stack with ___chkstk_ms (frames > 4 KB), largest first.

usage: chkstk.py EncoderApp.exe [--top N]
"""
import argparse
import re
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-C", args.exe], capture_output=True, text=True,
                         errors="replace").stdout
    fn, size, res = None, None, {}
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if m:
            fn, size = m.group(1), None
            continue
        m = re.search(r"mov\s+\$0x([0-9a-f]+),%eax", line)
        if m:
            size = int(m.group(1), 16)
            continue
        if "___chkstk_ms" in line and "call" in line and fn:
            res[fn] = max(res.get(fn, 0), size or 0)
    for name, sz in sorted(res.items(), key=lambda x: -x[1])[: args.top]:
        print(f"{sz:8d}  {name[:150]}")
    print(f"{len(res)} functions")


if __name__ == "__main__":
    main()
