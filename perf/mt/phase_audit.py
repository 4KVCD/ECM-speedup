"""Frame-thread audit of the EncGOP members compressGOP uses: per member, the phases (setup / search / finish) that
write it and that read it, including the EncGOP helpers each phase calls (transitively). Writes are recognised
heuristically (assignment, ++/--, compound assignment, non-const member calls such as push_back/clear/insert/erase,
memset/memcpy targets); everything else counts as a read. Flags:
  A: written in a setup, read in a search or finish  (the finish of picture N may see the setup of N+1)
  B: written in a search or finish, read in a setup  (the setup of N+1 may run before the finish of N)
  C: written in a search                             (searches run at the same time)
Helpers are found by name, so a call of another object's init()/create() also pulls in EncGOP::init/create, and the
constructor is counted with the function above it: check every flag by hand.
usage: phase_audit.py <compressGOP start> <setup end> <finish start> <picture end>   (line numbers in EncGOP.cpp)"""
import os
import re
import sys

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "Lib", "EncoderLib")
src = open(os.path.join(LIB, "EncGOP.cpp"), encoding="utf-8").read().split("\n")
hdr = open(os.path.join(LIB, "EncGOP.h"), encoding="utf-8").read()
members = set(re.findall(r"\b(m_[A-Za-z0-9_]+)\b", hdr))

START, SEARCH, FINISH, END = (int(a) for a in sys.argv[1:5])


def phase_of(ln):
    return "setup" if ln < SEARCH else ("search" if ln < FINISH else "finish")


funcs = {}
cur = None
for i, l in enumerate(src, 1):
    m = re.match(r"^[A-Za-z].*\bEncGOP::(~?\w+)\s*\(", l)
    if m:
        if cur:
            funcs.setdefault(cur[0], []).append((cur[1], i - 1))
        cur = (m.group(1), i)
if cur:
    funcs.setdefault(cur[0], []).append((cur[1], len(src)))

WRITE_CALLS = r"(push_back|emplace_back|clear|insert|erase|resize|assign|swap|pop_front|pop_back|push_front|reset|set\w*|create|destroy|init\w*|update\w*|add\w*|store\w*|load\w*)"


def is_write(line, mm):
    pat = re.escape(mm)
    if re.search(pat + r"(\s*\[[^\]]*\])*\s*(=[^=]|\+=|-=|\*=|/=|\|=|&=|\^=|<<=|>>=|\+\+|--)", line):
        return True
    if re.search(r"(\+\+|--)\s*" + pat, line):
        return True
    if re.search(pat + r"(\s*\[[^\]]*\])*\s*(\.|->)\s*" + WRITE_CALLS + r"\s*\(", line):
        return True
    if re.search(r"(memset|memcpy|std::fill|::memset|std::swap)\s*\(\s*&?\s*" + pat, line):
        return True
    return False


def scan(lines, ph, tag, out, seen_funcs, depth=0):
    for ln in lines:
        l = src[ln - 1]
        s = l.strip()
        if s.startswith("//"):
            continue
        code = l.split("//")[0]
        for mm in set(re.findall(r"\b(m_[A-Za-z0-9_]+)\b", code)):
            if mm in members:
                kind = "W" if is_write(code, mm) else "R"
                out.setdefault(mm, {}).setdefault((ph, kind), set()).add(f"{tag}:{ln}")
        for f in funcs:
            if f in ("compressGOP",) or (ph, f) in seen_funcs:
                continue
            if re.search(r"\b" + re.escape(f) + r"\s*\(", code):
                seen_funcs.add((ph, f))
                for a, b in funcs[f]:
                    scan(range(a, b + 1), ph, f, out, seen_funcs, depth + 1)


use = {}
seen = set()
for ln in range(START, END + 1):
    scan([ln], phase_of(ln), "cG", use, seen)

for mm in sorted(use):
    d = use[mm]
    w = {p for (p, k) in d if k == "W"}
    r = {p for (p, k) in d if k == "R"}
    flags = []
    if "setup" in w and (r & {"search", "finish"}):
        flags.append("A")
    if (w & {"search", "finish"}) and "setup" in (r | w):
        flags.append("B")
    if "search" in w:
        flags.append("C")
    if not flags:
        continue
    print(f"{mm:34s} {''.join(flags):3s} W:{','.join(sorted(w))}  R:{','.join(sorted(r))}")
    for (p, k), locs in sorted(d.items()):
        print(f"      {p:6s} {k} " + " ".join(sorted(locs, key=lambda x: (x.split(':')[0], int(x.split(':')[1])))[:10]))
