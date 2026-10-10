"""Self time by innermost source line (with inlining), over the whole program.

usage: inl.py samples.txt EncoderApp.exe [--grep REGEX] [--top N] [--outer]
  --grep   only count samples whose innermost location matches REGEX (e.g. 'IntraSearch.h:2[2-9][0-9]')
  --outer  group by "innermost [in outermost]" instead of innermost only
"""
import argparse
import collections
import re
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("exe")
    ap.add_argument("--grep")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--outer", action="store_true")
    args = ap.parse_args()
    out = subprocess.run(["objdump", "-p", args.exe], capture_output=True, text=True).stdout
    image_base = int(re.search(r"ImageBase\s+([0-9a-fA-F]+)", out).group(1), 16)
    total = 0
    leaf = collections.Counter()
    for line in open(args.samples):
        if line.startswith("#"):
            continue
        p = line.split()
        n = int(p[0])
        total += n
        if not p[1].startswith("x:"):
            leaf[int(p[1], 16)] += n
    hot = list(leaf.items())
    inp = "\n".join(hex(a + image_base) for a, _ in hot)
    out = subprocess.run(["addr2line", "-e", args.exe, "-i", "-a"], input=inp, capture_output=True,
                         text=True).stdout.splitlines()
    groups, cur = [], None
    for l in out:
        if l.startswith("0x"):
            cur = []
            groups.append(cur)
        else:
            cur.append(re.sub(r".*[/\\]", "", l.split(" (")[0]))
    res = collections.Counter()
    matched = 0
    for (a, n), g in zip(hot, groups):
        inner = g[0] if g else "?"
        if args.grep and not re.search(args.grep, inner):
            continue
        matched += n
        key = inner if not args.outer or len(g) < 2 else f"{inner}  [in {g[-1]}]"
        res[key] += n
    print(f"total {total} samples, matched {100.0 * matched / total:.2f}%")
    for k, n in res.most_common(args.top):
        print(f"{100.0 * n / total:6.2f}%  {k}")


if __name__ == "__main__":
    main()
