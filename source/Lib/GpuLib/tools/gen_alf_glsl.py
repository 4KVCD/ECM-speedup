#!/usr/bin/env python3
"""Generates the tap lists of the ALF kernels (source/Lib/GpuLib/shaders/alf_gen_*.glsl) from the C++ filters in
source/Lib/CommonLib/AdaptiveLoopFilter.cpp, so that the GPU kernels use exactly the CPU's tap geometry and
coefficient orders:

  alf_gen_perm.glsl      coefficient order per transposition (FIX9_PERM, FIX13_PERM, RESI_PERM, MAIN13_PERM)
  alf_gen_fix9.glsl      fixed filter 9x9 + 9x9 before deblocking (AdaptiveLoopFilter::fixedFilterBlk)
  alf_gen_fix13.glsl     fixed filter 13x13 on the first fixed-filter output + 9x9 before deblocking
  alf_gen_resi.glsl      fixed filter of the residual (AdaptiveLoopFilter::fixedFilteringResi)
  alf_gen_main13.glsl    the 13x13 luma filter taps on the reconstruction (AdaptiveLoopFilter::filterBlk)
  alf_gen_fixbased.glsl  its taps on the second fixed-filter output, per transposition

The generated statements add to `sum` with the coefficient / clipping arrays `c` / `cl` (already in tap order) or
`coef` / `clip` (fixed-based taps, class order), the current sample `curr` and the sample macros the including
kernel defines: MAIN(dx, dy), DBF(dx, dy), FIX(dx, dy).

usage: gen_alf_glsl.py [AdaptiveLoopFilter.cpp] [output directory]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../../CommonLib/AdaptiveLoopFilter.cpp")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "../shaders")

lines = open(SRC, encoding="utf-8-sig").read().split("\n")


def find(anchor, start=0, end=None):
    end = len(lines) if end is None else end
    for i in range(start, end):
        if anchor in lines[i]:
            return i
    raise SystemExit(f"anchor not found: {anchor!r}")


def perm_lists(start, count, name):
    """the next `count` initializer lists 'filterCoeff = { coeff[a], ... }' (or coef[]) from line `start`"""
    out = []
    i = start
    while len(out) < count:
        m = re.search(r"filterCoeff\s*=\s*\{(.*)\};", lines[i])
        if m:
            out.append([int(v) for v in re.findall(r"coef+\[(\d+)\]", m.group(1))])
        i += 1
    for p in out:
        if sorted(p) != list(range(len(p))):
            raise SystemExit(f"{name}: not a permutation: {p}")
    return out, i


# pImgN of the functions: row offset of N (convention of each function)
def row_minus_odd(n):   # pImg1 = pImg0 - stride (fixedFilterBlk, fixedFilteringResi)
    return 0 if n == 0 else (-(n + 1) // 2 if n % 2 else n // 2)


def row_plus_odd(n):    # pImg1 = pImg0 + stride (filterBlk)
    return 0 if n == 0 else ((n + 1) // 2 if n % 2 else -n // 2)


def num(text):
    return int(text.replace(" ", "")) if text else 0


PIMG = re.compile(r"pImg(\d+)(BeforeDb|FixedBased)?\[\s*([+-]?\s*\d+)\s*\]")
PIMG_EXTRA = re.compile(r"pImg(\d+)(Gauss|Laplacian)\[0\]\[\s*([+-]?\s*\d+)\s*\]")      # filterBlk: pImg1Gauss = +1 row
RECDB = re.compile(r"pRecDbTmp(\d)\[\s*([+-]?\s*\d+)\s*\]")                               # Tmp1 = +1 row, Tmp2 = -1 row
RESI = re.compile(r"pResiTmp0\[\s*([+-]?\s*\d+)\s*\]")
FIXRES = re.compile(r"fixedFilterResults\[(\d) \+ fixedFilterSetIdx\]\[blkDst\.y \+ i \+ ii \+ padSize(\s*[+-]\s*\d+)?\]"
                    r"\[blkDst\.x \+ j \+ jj \+ padSize(\s*[+-]\s*\d+)?\]")
FIXRESI = re.compile(r"fixedFilterResiResults\[0 \+ 1 - fixedFilterSetIdx\]\[blkDst\.y \+ i \+ ii\]\[blkDst\.x \+ j \+ jj\]")


def operand(text, rows, planes):
    text = text.strip()
    if text == "0":
        return "0"
    m = PIMG.fullmatch(text)
    if m:
        n, kind, dx = int(m.group(1)), m.group(2) or "", num(m.group(3))
        return f"{planes[kind]}( {dx}, {rows(n)} )"
    m = PIMG_EXTRA.fullmatch(text)
    if m:
        return f"{'GAUSS' if m.group(2) == 'Gauss' else 'LAPL'}( {num(m.group(3))}, {row_plus_odd(int(m.group(1)))} )"
    m = RECDB.fullmatch(text)
    if m:
        return f"DBF( {num(m.group(2))}, {[0, 1, -1][int(m.group(1))]} )"
    m = RESI.fullmatch(text)
    if m:
        return f"RESI( {num(m.group(1))}, 0 )"
    m = FIXRES.fullmatch(text)
    if m:
        return f"{ {'0': 'FIXA', '2': 'FIXB'}[m.group(1)] }( {num(m.group(3))}, {num(m.group(2))} )"
    if FIXRESI.fullmatch(text):
        return "FIXRESI( 0, 0 )"
    raise SystemExit(f"operand not understood: {text!r}")


SUM = re.compile(r"sum\s*\+=\s*(filterCoeff|coef)\[(\d+)\]\s*\*\s*\(*\s*clipALF\(\s*(filterClipp|clip)\[(\d+)\]\s*,\s*(curr|0)\s*,(.*?)\)\s*\)*\s*;")


def taps(start, stop_pattern, rows, planes, coeff_names=("c", "cl")):
    """'sum += ...' statements from line `start` up to the first line matching stop_pattern"""
    out = []
    i = start
    while not re.search(stop_pattern, lines[i]):
        m = SUM.search(lines[i])
        if m:
            k, kc, ref = int(m.group(2)), int(m.group(4)), m.group(5)
            if k != kc:
                raise SystemExit(f"line {i + 1}: coefficient {k} with clipping {kc}")
            ops = [operand(t, rows, planes) for t in m.group(6).split(",")]
            c, cl = coeff_names
            if len(ops) == 2:
                out.append(f"sum += {c}[{k}] * clipAlf2( {cl}[{k}], {ref}, {ops[0]}, {ops[1]} );")
            else:
                out.append(f"sum += {c}[{k}] * clipAlf1( {cl}[{k}], {ref}, {ops[0]} );")
        elif "sum +=" in lines[i] or "sum+=" in lines[i]:
            raise SystemExit(f"line {i + 1}: statement not understood: {lines[i].strip()}")
        i += 1
    return out, i


def glsl_array(name, rows):
    inner = ",\n  ".join(f"int[{len(r)}]( {', '.join(map(str, r))} )" for r in rows)
    return f"const int {name}[{len(rows)}][{len(rows[0])}] = int[{len(rows)}][{len(rows[0])}](\n  {inner} );\n"


HEADER = "// generated by source/Lib/GpuLib/tools/gen_alf_glsl.py from source/Lib/CommonLib/AdaptiveLoopFilter.cpp; do not edit\n"


def write(name, text):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="\n") as f:
        f.write(HEADER + text)
    print("wrote", path)


# ---- fixedFilterBlk: 13x13 on the first fixed-filter output and 9x9, each with 9x9 taps before deblocking
f0 = find("void AdaptiveLoopFilter::fixedFilterBlk( AlfClassifier **classifier, const CPelBuf &src, const Area &curBlk")
f1 = find("void AdaptiveLoopFilter::calcClassNew(", f0)
i = find("if( filtType == ALF_FIXED_FILTER_13_DB_9 )", find("const short* clipp;", f0, f1), f1)
fix13_perm, i = perm_lists(i, 4, "fix13")
i = find("else if( filtType == ALF_FIXED_FILTER_9_DB_9 )", i, f1)
fix9_perm, i = perm_lists(i, 4, "fix9")
i = find("if( filtType == ALF_FIXED_FILTER_13_DB_9 )", find("int sum = 0;", i, f1), f1)
fix13_taps, i = taps(i, r"else if\( filtType == ALF_FIXED_FILTER_9_DB_9 \)", row_minus_odd, {"": "MAIN", "BeforeDb": "DBF"})
fix9_taps, i = taps(i, r"CHECK\( 1", row_minus_odd, {"": "MAIN", "BeforeDb": "DBF"})
assert len(fix13_taps) == 64 and len(fix9_taps) == 41, (len(fix13_taps), len(fix9_taps))

# ---- fixedFilteringResi (JVET_AE0139 branch: 9x9 shape, the first lists of the function)
r0 = find("void AdaptiveLoopFilter::fixedFilteringResi(")
r1 = find("void AdaptiveLoopFilter::deriveClassificationLaplacianBig(", r0)
resi_perm, i = perm_lists(r0, 4, "resi")
i = find("#if JVET_AE0139_ALF_IMPROVED_FIXFILTER", find("int sum = 0;", i, r1), r1)
resi_taps, i = taps(i, r"^#else", row_minus_odd, {"": "MAIN"})
assert len(resi_taps) == 21, len(resi_taps)

# ---- filterBlk: 13x13 shape of the long fixed-filter variant, and the fixed-based taps per transposition
b0 = find("void AdaptiveLoopFilter::filterBlk(AlfClassifier **classifier, const PelUnitBuf &recDst,")
b1 = find("void AdaptiveLoopFilter::filterBlkCcAlf(", b0)
i = find("#if JVET_AD0222_ALF_LONG_FIXFILTER", b0, b1)
m13, i = perm_lists(i, 4, "main13")          # transposeIdx 1, 2, 3, then 0
main13_perm = [m13[3], m13[0], m13[1], m13[2]]
i = find("#if JVET_AD0222_ALF_LONG_FIXFILTER", find("const Pel curr = pImg0[+0];", i, b1), b1)
main13_taps, i = taps(i, r"^#else", row_plus_odd, {"": "MAIN"})
assert len(main13_taps) == 10, len(main13_taps)
i = find("int filterSetIdx = 2 + fixedFilterSetIdx;", i, b1)
fixbased = {}
for t, anchor in ((1, "if( transposeIdx == 1 )"), (2, "else if( transposeIdx == 2 )"), (3, "else if( transposeIdx == 3 )"),
                  (0, "else")):
    i = find(anchor, i, b1)
    fixbased[t], i = taps(i + 1, r"^\s*\}", row_plus_odd, {"FixedBased": "FIX"}, ("coef", "clip"))
    assert len(fixbased[t]) == 18, (t, len(fixbased[t]))

# ---- filterBlk: the 9x9 shape (chroma) with its taps on the fixed-filter output, the extra luma taps of the two
# residual filter types (long fixed filter + additional fixed filter variant)
s0 = find("const Pel curr = pImg0[+0];", b0, b1)
i = find("if( filtType == ALF_FILTER_9 || filtType == ALF_FILTER_9_EXT || filtType == ALF_FILTER_9_EXT_DB", s0, b1)
f9_taps, i = taps(i, r"if\( filtType == ALF_FILTER_9 \)", row_plus_odd, {"": "MAIN"})
f9_fix, i = taps(i + 1, r"^\s*\}", row_plus_odd, {"": "MAIN"})
assert len(f9_taps) == 20 and len(f9_fix) == 5, (len(f9_taps), len(f9_fix))
i = find("#if JVET_AD0222_ALF_LONG_FIXFILTER && JVET_AD0222_ADDITONAL_ALF_FIXFILTER",
         find("else if (filtType == ALF_FILTER_13_EXT_DB_RESI_DIRECT)", s0, b1), b1)
type9_taps, i = taps(i + 1, r"^#elif", row_plus_odd, {"": "MAIN"}, ("coef", "clip"))
i = find("#if JVET_AK0091_LAPLACIAN_INFO_IN_ALF",
         find("#if JVET_AD0222_ALF_LONG_FIXFILTER && JVET_AD0222_ADDITONAL_ALF_FIXFILTER",
              find("else if (filtType == ALF_FILTER_13_EXT_DB_RESI)", s0, b1), b1), b1)
type10_taps, i = taps(i + 1, r"^#else", row_plus_odd, {"": "MAIN"}, ("coef", "clip"))
assert len(type9_taps) == 11 and len(type10_taps) == 13, (len(type9_taps), len(type10_taps))

# ---- filterBlkCcAlf: the taps of the luma-residual / chroma-SAO / cross-chroma variant, translated to GLSL
c0 = find("void AdaptiveLoopFilter::filterBlkCcAlf(const PelBuf &dstBuf, const CPelUnitBuf &recSrc, const Area &blkDst,"
          " const Area &blkSrc, const ComponentID compId, const int16_t *filterCoeff, const ClpRngs &clpRngs,"
          " CodingStructure &cs, const CPelUnitBuf &resiSrc, const Pel clippingValues[4], const int coeffPrec ")
i = find("sum += filterCoeff[0] * (srcCross[offset8 + jj2] - currSrcCross + srcCross[offset7 + jj2] - currSrcCross);", c0)
LUMA_ROW = {0: 0, 1: 1, 2: -1, 3: 2, 4: -2, 5: 3, 6: -3, 7: 4, 8: -4}
ccalf = []
while not lines[i].startswith("#else"):
    t = lines[i].strip()
    if t.startswith("sum +="):
        t = re.sub(r"filterCoeff\[(\d+)\]", r"c[\1]", t)
        t = re.sub(r"srcCross\[offset(\d) \+ jj2\s*(?:([+-])\s*(\d+))?\s*\]",
                   lambda m: f"LUM( {num((m.group(2) or '') + (m.group(3) or ''))}, {LUMA_ROW[int(m.group(1))]} )", t)
        t = t.replace("currSrcCross", "curL")
        t = re.sub(r"srcResiCross\[([+-]\d) \* resiStride \+ jj2 ([+-]) (\d+)\]",
                   lambda m: f"RES( {num(m.group(2) + m.group(3))}, {num(m.group(1))} )", t)
        t = re.sub(r"clipALF\(clipValue, 0, (RES\([^)]*\))\)", r"clipAlf1( clipValue, 0, \1 )", t)
        t = re.sub(r"srcSAOCross\[([+-]\d) \* saoStride \+ jj ([+-]) (\d+)\]",
                   lambda m: f"SAO( {num(m.group(2) + m.group(3))}, {num(m.group(1))} )", t)
        t = t.replace("currSrcSAOCross", "curS")
        t = re.sub(r"crossChromaPtr\[([+-]\d) \* crossChromaStride \+ jj ([+-]) (\d+)\]",
                   lambda m: f"XC( {num(m.group(2) + m.group(3))}, {num(m.group(1))} )", t)
        t = t.replace("crossChromaCurr", "curX")
        if re.search(r"src|offset|jj|clipALF|Stride", t):
            raise SystemExit(f"line {i + 1}: CCALF tap not translated: {t}")
        ccalf.append(t)
    i += 1
assert len(ccalf) == 33, len(ccalf)

write("alf_gen_perm.glsl", glsl_array("FIX9_PERM", fix9_perm) + glsl_array("FIX13_PERM", fix13_perm)
      + glsl_array("RESI_PERM", resi_perm) + glsl_array("MAIN13_PERM", main13_perm))
write("alf_gen_filter9.glsl", "\n".join(f9_taps) + "\n")
write("alf_gen_filter9fix.glsl", "\n".join(f9_fix) + "\n")
write("alf_gen_type9.glsl", "\n".join(type9_taps) + "\n")
write("alf_gen_type10.glsl", "\n".join(type10_taps) + "\n")
write("alf_gen_ccalf.glsl", "\n".join(ccalf) + "\n")
write("alf_gen_fix9.glsl", "\n".join(fix9_taps) + "\n")
write("alf_gen_fix13.glsl", "\n".join(fix13_taps) + "\n")
write("alf_gen_resi.glsl", "\n".join(resi_taps) + "\n")
write("alf_gen_main13.glsl", "\n".join(main13_taps) + "\n")
fb = []
for t in (0, 1, 2, 3):
    fb.append(f"{'if' if t == 0 else 'else if'}( transposeIdx == {t} )\n{{\n  " + "\n  ".join(fixbased[t]) + "\n}")
write("alf_gen_fixbased.glsl", "\n".join(fb) + "\n")
