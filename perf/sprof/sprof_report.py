#!/usr/bin/env python3
"""Symbolize and summarize sprof output.

usage: sprof_report.py samples.txt EncoderApp.exe [--top N] [--lines FUNC_SUBSTR] [--callers FUNC_SUBSTR]
       [--leafcaller]   attribute samples in DLLs (memcpy, malloc, ...) to the exe function that called them
"""
import argparse
import bisect
import collections
import re
import subprocess


def load_symbols(exe):
    out = subprocess.run(["objdump", "-p", exe], capture_output=True, text=True).stdout
    image_base = int(re.search(r"ImageBase\s+([0-9a-fA-F]+)", out).group(1), 16)
    out = subprocess.run(["nm", "-n", "-C", "--defined-only", exe], capture_output=True, text=True,
                         errors="replace").stdout
    addrs, names = [], []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 3 or parts[1] not in "tTwW":
            continue
        addrs.append(int(parts[0], 16) - image_base)
        names.append(parts[2])
    return image_base, addrs, names


def short(name, width=120):
    n = re.sub(r"\(.*\)( const)?$", "", name)  # drop argument lists
    return n if len(n) <= width else n[: width - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("exe")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--lines", help="line-level breakdown of self time in functions matching this substring")
    ap.add_argument("--callers", help="who calls functions matching this substring (inclusive)")
    ap.add_argument("--callees", help="what functions matching this substring call (inclusive)")
    ap.add_argument("--callsites", help="source lines of the calls into functions matching this substring")
    args = ap.parse_args()

    image_base, addrs, names = load_symbols(args.exe)
    cache = {}

    def sym(frame):
        if frame.startswith("x:"):
            return frame[2:]
        if frame not in cache:
            a = int(frame, 16)
            i = bisect.bisect_right(addrs, a) - 1
            cache[frame] = short(names[i]) if i >= 0 else "?"
        return cache[frame]

    total = 0
    self_t = collections.Counter()
    incl_t = collections.Counter()
    self_addr = collections.Counter()
    callers = collections.Counter()
    callees = collections.Counter()
    callsites = collections.Counter()
    for line in open(args.samples):
        if line.startswith("#"):
            continue
        p = line.split()
        n = int(p[0])
        frames = p[1:]
        total += n
        funcs = [sym(f) for f in frames]
        self_t[funcs[0]] += n
        if not frames[0].startswith("x:"):
            self_addr[int(frames[0], 16)] += n
        seen = set()
        for i, f in enumerate(funcs):
            if f not in seen:
                seen.add(f)
                incl_t[f] += n
        if args.callers:
            for i, f in enumerate(funcs[:-1]):
                if args.callers in f:
                    callers[funcs[i + 1]] += n
                    break
        if args.callsites:
            for i, f in enumerate(funcs[:-1]):
                if args.callsites in f and not frames[i + 1].startswith("x:"):
                    callsites[int(frames[i + 1], 16)] += n
                    break
        if args.callees:
            for i in range(len(funcs) - 1, 0, -1):  # outermost match
                if args.callees in funcs[i]:
                    callees[funcs[i - 1]] += n
                    break

    def pr(title, counter):
        print(f"\n== {title} ==")
        for name, n in counter.most_common(args.top):
            print(f"{100.0 * n / total:6.2f}%  {name}")

    print(f"total samples: {total}")
    pr(f"self time (top {args.top})", self_t)
    pr(f"inclusive time (top {args.top})", incl_t)
    if args.callers:
        pr(f"callers of *{args.callers}*", callers)
    if args.callees:
        pr(f"direct callees of *{args.callees}*", callees)
    def by_source_line(addr_counts, inlines):
        hot = list(addr_counts.items())
        inp = "\n".join(hex(a + image_base) for a, _ in hot)
        cmd = ["addr2line", "-e", args.exe] + (["-i", "-a"] if inlines else [])
        out = subprocess.run(cmd, input=inp, capture_output=True, text=True).stdout.splitlines()
        res = collections.Counter()
        if inlines:  # with -a each address line starts a group; keep the innermost + outermost location
            groups, cur = [], None
            for l in out:
                if l.startswith("0x"):
                    cur = []
                    groups.append(cur)
                else:
                    cur.append(re.sub(r".*[/\\]", "", l.split(" (")[0]))
            for (a, n), g in zip(hot, groups):
                res[g[0] if len(g) == 1 else f"{g[0]}  [in {g[-1]}]"] += n
        else:
            for (a, n), loc in zip(hot, out):
                res[re.sub(r".*[/\\]", "", loc.split(" (")[0])] += n
        return res

    if args.callsites:
        pr(f"call sites into *{args.callsites}*", by_source_line(callsites, True))
    if args.lines:
        hot = collections.Counter({a: n for a, n in self_addr.items() if args.lines in sym("%x" % a)})
        if hot:
            pr(f"self-time lines in *{args.lines}*", by_source_line(hot, True))


if __name__ == "__main__":
    main()
