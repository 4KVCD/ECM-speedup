"""Upper bound of frame-level parallelism for a random-access encode, from an encoder log.

A picture can start when its reference pictures and the previous picture of its temporal layer (temporal CABAC
context store keyed by layer and QP, AMVP-SbTMVP area statistics keyed by layer) have finished. With unlimited
threads the encode then takes the longest path of the dependency graph, weighted with the per-picture times [ET].
Intra periods are independent (segment-parallel already runs them in parallel), so the bound is reported per
intra period and for the whole log.

usage: ra_critical_path.py log [intra_period]
"""
import re
import sys

log = sys.argv[1]
period = int(sys.argv[2]) if len(sys.argv) > 2 else 32
pics = []  # coding order: (poc, tid, qp, et, refs)
for line in open(log):
    m = re.match(r"POC\s+(\d+)\s+LId:\s*\d+\s+TId:\s*(\d+)\s+\(\s*\S+,\s*(\S)-SLICE,\s*QP\s+(\d+)\s*\).*\[ET\s+([\d.]+)\s*\](.*)", line)
    if not m:
        continue
    poc, tid, stype, qp, et, rest = int(m[1]), int(m[2]), m[3], int(m[4]), float(m[5]), m[6]
    refs = set()
    for lst in re.findall(r"\[L[01]([^\]]*)\]", rest):
        refs.update(int(t.rstrip("c")) for t in lst.split())
    pics.append((poc, tid, stype, qp, et, refs))

finish = {}
last_same_layer = {}
total = 0.0
for poc, tid, stype, qp, et, refs in pics:
    if poc % period == 0:   # intra picture: starts a new chain, and the period's pictures may start from scratch
        last_same_layer = {}
    chain = last_same_layer.get(tid, 0.0) if stype != "I" else 0.0   # the context store keeps inter slices only
    start = max([finish[r] for r in refs if r in finish] + [chain, 0.0])
    finish[poc] = start + et
    if stype != "I":
        last_same_layer[tid] = finish[poc]
    total += et

# per intra period (POCs (k-1)*P+1 .. k*P plus the intra picture k*P)
periods = {}
for poc, tid, stype, qp, et, refs in pics:
    k = (poc + period - 1) // period
    periods.setdefault(k, []).append((poc, et))
print(f"pictures: {len(pics)}, sum of picture times: {total:.1f} s")
et_of = {poc: et for poc, tid, stype, qp, et, refs in pics}
for k in sorted(periods):
    pocs = [p for p, _ in periods[k]]
    if k >= 1 and (k - 1) * period in et_of:
        pocs.append((k - 1) * period)   # a chunk encodes the intra picture before it again (segment-parallel)
    work = sum(et_of[p] for p in pocs)
    # critical path inside the period: recompute with the period's pictures only
    fin, same = {}, {}
    for poc, tid, stype, qp, et, refs in pics:
        if poc not in pocs:
            continue
        chain = same.get(tid, 0.0) if stype != "I" else 0.0
        start = max([fin[r] for r in refs if r in fin] + [chain, 0.0])
        fin[poc] = start + et
        if stype != "I":
            same[tid] = fin[poc]
    cp = max(fin.values())
    print(f"period {k:2d}: {len(pocs):2d} pictures, work {work:7.1f} s, critical path {cp:7.1f} s, bound {work / cp:4.2f}x")
