#!/usr/bin/env python3
"""List the mutable static storage of object files: globals, class statics, function-local statics (.data/.bss
symbols), and thread_local variables (emutls control symbols), demangled, one line per symbol with its object file.
Use it on a non-LTO build tree to find state that concurrent pictures or workers would share.

usage: static_audit.py <mingw bin dir> <out.txt> <obj dir> [<obj dir> ...]
       e.g. static_audit.py C:/Code/ECM/tools/mingw64/bin statics.txt C:/Code/ECM/build/<name>/source/Lib
"""
import os
import re
import subprocess
import sys

bindir, out_path, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
nm = os.path.join(bindir, "nm.exe")
cxxfilt = os.path.join(bindir, "c++filt.exe")
rows = set()
for d in dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if not f.endswith(".obj"):
                continue
            res = subprocess.run([nm, "--defined-only", os.path.join(root, f)], capture_output=True, text=True)
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                kind, name = parts[1], parts[2]
                tls = re.search(r"__emutls_v\.(\S+)", name)
                if tls:
                    if not tls[1].startswith("_ZGV"):        # skip guard variables
                        rows.add((f, "tls", tls[1]))
                elif kind in "bBdDVvu" and not name.startswith(("_ZTV", "_ZTI", "_ZTS", "_ZGV", ".", "__")):
                    rows.add((f, kind, name))
names = sorted(rows)
demangled = subprocess.run([cxxfilt], input="\n".join(r[2] for r in names), capture_output=True,
                           text=True).stdout.splitlines()
with open(out_path, "w", encoding="utf-8") as out:
    for (f, kind, _), d in zip(names, demangled):
        out.write(f"{f}\t{kind}\t{d}\n")
print(len(names), "symbols written to", out_path)
