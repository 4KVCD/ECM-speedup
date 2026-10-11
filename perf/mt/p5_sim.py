#!/usr/bin/env python3
"""Overlapping-GOP (P5) schedule simulation from the log of a serial encode: event-driven list scheduling with K
threads and S search stacks. A picture's search needs its references and the last earlier picture of its temporal
layer since the last IRAP (temporal CABAC store) finished, a free stack (held from search start to finish end) and a
free thread; ready pictures start in coding order. Finishes take the given fraction of the work and run in coding
order, globally or only within an intra period (an IRAP's finish does not wait for the previous period's).
usage: p5_sim.py <serial encoder log> [finish fraction, default 0.05]"""
import heapq
import re
import sys

log = sys.argv[1]
fin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

pics = []
for line in open(log, errors="replace"):
    m = re.match(r"POC\s+(\d+) LId:\s*\d+ TId: (\d+) \( (\w+), .*\[ET\s+([\d.]+) \] \[L0([^\]]*)\] \[L1([^\]]*)\]", line)
    if m:
        poc, tid, nal, et = int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4))
        refs = {int(x.rstrip("c")) for x in (m.group(5) + " " + m.group(6)).split() if x}
        pics.append((poc, tid, et, refs, nal in ("IDR_N_LP", "IDR_W_RADL", "CRA")))
total = sum(p[2] for p in pics)
n = len(pics)
idx_of = {p[0]: i for i, p in enumerate(pics)}

# static dependencies (coding-order indices) and the finish predecessor of each picture
deps, fin_pred_global, fin_pred_period = [], [], []
last_of_tid, period_start = {}, 0
for i, (poc, tid, et, refs, irap) in enumerate(pics):
    if irap:
        last_of_tid, period_start = {}, i
    d = {idx_of[r] for r in refs if r in idx_of}
    if tid in last_of_tid:
        d.add(last_of_tid[tid])
    deps.append(d)
    last_of_tid[tid] = i
    fin_pred_global.append(i - 1 if i > 0 else None)
    fin_pred_period.append(i - 1 if i > period_start else None)


def simulate(K, S, period_finishes):
    fin_pred = fin_pred_period if period_finishes else fin_pred_global
    done = [None] * n                 # finish end
    search_end = [None] * n
    started = [False] * n
    threads, stacks = K, S
    t = 0.0
    events = []                       # (time, kind, i)
    finishing = set()

    def try_start(now):
        nonlocal threads, stacks
        # finishes first (they free stacks): a searched picture whose finish predecessor is done
        for i in range(n):
            if search_end[i] is not None and search_end[i] <= now and i not in finishing and done[i] is None:
                p = fin_pred[i]
                if (p is None or (done[p] is not None and done[p] <= now)) and threads > 0:
                    threads -= 1
                    finishing.add(i)
                    heapq.heappush(events, (now + pics[i][2] * fin, 1, i))
        for i in range(n):
            if started[i] or threads == 0 or stacks == 0:
                continue
            if all(done[j] is not None and done[j] <= now for j in deps[i]):
                started[i] = True
                threads -= 1
                stacks -= 1
                heapq.heappush(events, (now + pics[i][2] * (1 - fin), 0, i))

    try_start(0.0)
    while events:
        t, kind, i = heapq.heappop(events)
        if kind == 0:
            search_end[i] = t
            threads += 1
        else:
            done[i] = t
            threads += 1
            stacks += 1
        try_start(t)
    return max(done)


print(f"{n} pictures, serial work {total:.0f} s, finish fraction {fin}")
for K in (2, 4, 8, 16):
    row = []
    for S in (K, 2 * K, 4 * K):
        a = simulate(K, S, False)
        b = simulate(K, S, True)
        row.append(f"S={S:2d}: {total / a:4.2f}x / {total / b:4.2f}x")
    print(f"K={K:2d} threads  (global finishes / per-period finishes)  " + "   ".join(row))
