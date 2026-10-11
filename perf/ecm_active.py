#!/usr/bin/env python3
"""Writes copies of ECM source files with the inactive #if branches removed.

usage: ecm_active.py MACROS.txt OUTDIR file [file ...]

MACROS.txt is the output of `g++ -dM -E` for a translation unit of the default build. Conditionals whose value
follows from those macros (and from #define / #undef lines met earlier in the same file) are resolved, and their
directive lines are dropped. Conditionals that depend on anything else are kept unchanged, with all branches.
The copies are only for reading: line numbers differ from the originals.
"""
import os
import re
import sys

IDENT = re.compile(r"[A-Za-z_]\w*")
TOKEN = re.compile(r"\s*(0[xX][0-9a-fA-F]+[uUlL]*|\d+[uUlL]*|[A-Za-z_]\w*|&&|\|\||==|!=|<=|>=|<<|>>|[()!<>+\-*/%?:~&|^])")


class Unknown(Exception):
    pass


def load_macros(path):
    macros = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"#define (\w+)(\(.*?\))?\s*(.*)", line.rstrip("\n"))
        if m and not m.group(2):
            macros[m.group(1)] = m.group(3).strip()
    return macros


def tokenize(expr):
    toks, pos = [], 0
    expr = re.sub(r"//.*", "", re.sub(r"/\*.*?\*/", "", expr)).strip()
    while pos < len(expr):
        m = TOKEN.match(expr, pos)
        if not m:
            raise Unknown(expr)
        toks.append(m.group(1))
        pos = m.end()
    return toks


def expand(toks, macros, depth=0):
    if depth > 20:
        raise Unknown("recursion")
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t == "defined":
            if i + 1 < len(toks) and toks[i + 1] == "(":
                name, i = toks[i + 2], i + 4
            else:
                name, i = toks[i + 1], i + 2
            out.append("1" if name in macros else "0")
            continue
        if IDENT.fullmatch(t):
            if t not in macros:
                raise Unknown(t)
            val = macros[t]
            out.extend(["("] + expand(tokenize(val), macros, depth + 1) + [")"] if val else ["1"])
        else:
            out.append(t)
        i += 1
    return out


def evaluate(expr, macros):
    toks = expand(tokenize(expr), macros)
    py = []
    for t in toks:
        if t == "&&":
            py.append(" and ")
        elif t == "||":
            py.append(" or ")
        elif t == "!":
            py.append(" not ")
        elif t == "/":
            py.append("//")
        elif re.fullmatch(r"0[xX][0-9a-fA-F]+[uUlL]*|\d+[uUlL]*", t):
            py.append(str(int(t.rstrip("uUlL"), 0)))
        elif t in "?:":
            raise Unknown("ternary")
        else:
            py.append(t)
    try:
        return bool(eval("".join(py), {"__builtins__": {}}))
    except Exception as e:
        raise Unknown(str(e))


def process(lines, macros):
    out = []
    # Each frame: [mode, active_now, taken_already, parent_active]; mode is "resolved" or "kept".
    stack = []

    def active():
        return all(f[1] for f in stack if f[0] == "resolved") and all(f[3] for f in stack)

    i = 0
    while i < len(lines):
        line = lines[i]
        full = line
        while full.rstrip("\n").endswith("\\") and i + 1 < len(lines):
            i += 1
            full = full.rstrip("\n")[:-1] + lines[i]
        i += 1
        m = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)", full)
        if not m:
            if active():
                out.append(full)
            continue
        kind, rest = m.group(1), m.group(2).strip()
        if kind in ("if", "ifdef", "ifndef"):
            parent = active()
            nxt = next((ln for ln in lines[i:i + 3] if ln.strip()), "")
            if kind == "ifndef" and re.match(r"\s*#\s*define\s+" + re.escape(rest.split()[0]) + r"\b", nxt):
                # An include guard: the macro dump defines it, but the file's contents are wanted.
                stack.append(["kept", True, False, parent])
                if parent:
                    out.append(full)
                continue
            try:
                if kind == "if":
                    val = evaluate(rest, macros)
                else:
                    name = IDENT.match(rest).group(0)
                    val = (name in macros) == (kind == "ifdef")
                stack.append(["resolved", val, val, parent])
            except Unknown:
                stack.append(["kept", True, False, parent])
                if parent:
                    out.append(full)
        elif kind == "elif":
            f = stack[-1]
            if f[0] == "kept":
                if f[3]:
                    out.append(full)
                continue
            try:
                val = (not f[2]) and evaluate(rest, macros)
            except Unknown:
                # An unresolvable #elif after resolved branches: keep the rest of the chain as written.
                f[0], f[1] = "kept", True
                if f[3] and not f[2]:
                    out.append(full.replace("#elif", "#if", 1))
                elif f[3]:
                    out.append("#if 0 /* earlier branch taken */\n")
                continue
            f[1] = val
            f[2] = f[2] or val
        elif kind == "else":
            f = stack[-1]
            if f[0] == "kept":
                if f[3]:
                    out.append(full)
                continue
            f[1] = not f[2]
            f[2] = True
        elif kind == "endif":
            f = stack.pop()
            if f[0] == "kept" and f[3]:
                out.append(full)
        else:
            if active():
                out.append(full)
                dm = re.match(r"(\w+)(\(.*?\))?\s*(.*)", rest)
                if dm and kind == "define" and not dm.group(2):
                    macros[dm.group(1)] = dm.group(3).strip()
                elif dm and kind == "undef":
                    macros.pop(dm.group(1), None)
    return out


def main():
    base = load_macros(sys.argv[1])
    outdir = sys.argv[2]
    for src in sys.argv[3:]:
        lines = open(src, encoding="utf-8", errors="replace").readlines()
        out = process(lines, dict(base))
        dst = os.path.join(outdir, os.path.basename(os.path.dirname(src)), os.path.basename(src))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.writelines(out)
        print(f"{len(lines):7d} -> {len(out):7d}  {dst}")


if __name__ == "__main__":
    main()
