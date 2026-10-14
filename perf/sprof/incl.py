# inclusive share of selected functions per sprof sample file
# usage: incl.py EncoderApp.exe a.txt,b.txt "EncCu::compressCtu,ALFProcess"  (substrings of function names)
import sys, subprocess, re, bisect, collections
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import sprof_report as sr
exe = sys.argv[1]
files = sys.argv[2].split(",")
pats = sys.argv[3].split(",")
image_base, addrs, names = sr.load_symbols(exe)
cache = {}
def sym(f):
    if f.startswith("x:"):
        return f[2:]
    if f not in cache:
        a = int(f, 16)
        i = bisect.bisect_right(addrs, a) - 1
        cache[f] = sr.short(names[i]) if i >= 0 else "?"
    return cache[f]
print("%-44s" % "function" + "".join("%12s" % f.split("/")[-1][:11] for f in files))
res = {p: [] for p in pats}
for fn in files:
    tot = 0
    c = collections.Counter()
    for line in open(fn):
        if line.startswith("#"):
            continue
        p = line.split()
        n = int(p[0]); tot += n
        fs = [sym(x) for x in p[1:]]
        for pat in pats:
            if any(pat in f for f in fs):
                c[pat] += n
    for pat in pats:
        res[pat].append(100.0 * c[pat] / tot)
for pat in pats:
    print("%-44s" % pat[:44] + "".join("%11.1f%%" % v for v in res[pat]))
