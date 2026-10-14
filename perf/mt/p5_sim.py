#!/usr/bin/env python3
"""Frame-thread schedule simulation (P5) from the log of a serial encode: event-driven list scheduling with K threads,
S search stacks, C finish contexts and at most W pictures in flight (set up and not done).
A picture's search needs its references and the last earlier picture of its temporal layer since the last IRAP
(temporal CABAC store) finished, a stack (held from search start to finish end) and a thread. Finishes take the given
fraction of a picture's time and run in coding order within an intra period (from an IRAP on); a period's first
finish waits for the period C before it (its finish context), so C=1 is the global coding order.
Stack policies: 'fifo' = stacks taken in coding order before the search inputs are done (stage A); 'ready' = taken
by pictures whose inputs are done, earliest first, while the earliest picture not done has one or one stays free for
it (stage B), held to the end of the finish (B-2) or released at the end of the search (B-3, the finish uses its
period's context). The last intra period can be repeated to model a longer sequence.
usage: p5_sim.py <serial encoder log> [finish fraction, default 0.03] [intra periods to extend to]"""
import heapq
import re
import sys

log = sys.argv[1]
fin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
extend_to = int(sys.argv[3]) if len(sys.argv) > 3 else 0

pics = []
for line in open(log, errors="replace"):
    m = re.match(r"POC\s+(\d+) LId:\s*\d+ TId: (\d+) \( (\w+), .*\[ET\s+([\d.]+) \] \[L0([^\]]*)\] \[L1([^\]]*)\]", line)
    if m:
        poc, tid, nal, et = int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4))
        refs = {int(x.rstrip("c")) for x in (m.group(5) + " " + m.group(6)).split() if x}
        pics.append((poc, tid, et, refs, nal in ("IDR_N_LP", "IDR_W_RADL", "CRA")))

if extend_to:
    starts = [i for i, p in enumerate(pics) if p[4]]
    last = pics[starts[-1]:]
    span = last[0][0] - pics[starts[-2]][0]
    k = 1
    while sum(1 for p in pics if p[4]) < extend_to:
        pics += [(p[0] + k * span, p[1], p[2], {r + k * span for r in p[3]}, p[4]) for p in last]
        k += 1

total = sum(p[2] for p in pics)
n = len(pics)
idx_of = {p[0]: i for i, p in enumerate(pics)}
deps, fin_pred, period_of = [], [], []
last_of_tid, period_start, num_periods = {}, 0, 0
for i, (poc, tid, et, refs, irap) in enumerate(pics):
    if irap:
        last_of_tid, period_start = {}, i
        num_periods += 1
    period_of.append(num_periods - 1)
    d = {idx_of[r] for r in refs if r in idx_of}
    if tid in last_of_tid:
        d.add(last_of_tid[tid])
    deps.append(d)
    last_of_tid[tid] = i
    fin_pred.append(i - 1 if i > period_start else None)
period_last = {}
for i in range(n):
    period_last[period_of[i]] = i


def simulate(K, S, C, W, policy, release=False):
    done = [None] * n
    search_end = [None] * n
    has_stack = [False] * n
    searching = [False] * n
    threads, stacks = K, S
    events = []
    finishing = set()
    oldest = 0

    def is_done(j, now):
        return done[j] is not None and done[j] <= now

    def try_start(now):
        nonlocal threads, stacks, oldest
        while oldest < n and is_done(oldest, now):
            oldest += 1
        for i in range(n):
            if search_end[i] is not None and search_end[i] <= now and i not in finishing and done[i] is None:
                p = fin_pred[i]
                q = period_of[i] - C
                ctx_free = q < 0 or is_done(period_last[q], now)
                if (p is None or is_done(p, now)) and ctx_free and threads > 0:
                    threads -= 1
                    finishing.add(i)
                    heapq.heappush(events, (now + pics[i][2] * fin, 1, i))
        window = range(oldest, min(n, oldest + W))
        for i in window:
            if has_stack[i] or stacks == 0:
                continue
            if policy == "fifo":
                if i > oldest and not has_stack[i - 1]:
                    break
            else:
                if not all(is_done(j, now) for j in deps[i]):
                    continue
                if i != oldest and not has_stack[oldest] and stacks == 1:
                    continue
            has_stack[i] = True
            stacks -= 1
        for i in window:
            if has_stack[i] and not searching[i] and threads > 0 and all(is_done(j, now) for j in deps[i]):
                searching[i] = True
                threads -= 1
                heapq.heappush(events, (now + pics[i][2] * (1 - fin), 0, i))

    try_start(0.0)
    while events:
        t, kind, i = heapq.heappop(events)
        if kind == 0:
            search_end[i] = t
            if release:
                stacks += 1
        else:
            done[i] = t
            if not release:
                stacks += 1
        threads += 1
        try_start(t)
    assert all(d is not None for d in done), "deadlock"
    return max(done)


print(f"{n} pictures, {num_periods} intra periods, serial work {total:.0f} s, finish fraction {fin}")
for K in (4, 8):
    print(f"K=S={K}  stage A (fifo, C=1): " + "  ".join(f"W={W}: {total / simulate(K, K, 1, W, 'fifo'):4.2f}x"
                                                     for W in (32, 64, 128)))
    for C in (1, 2, 4, 8):
        print(f"K=S={K}  ready, C={C}:          " + "  ".join(f"W={W}: {total / simulate(K, K, C, W, 'ready'):4.2f}x"
                                                          for W in (32, 64, 128)))
        print(f"        released at search end: " + "  ".join(
            f"W={W}: {total / simulate(K, K, C, W, 'ready', True):4.2f}x" for W in (32, 64, 128)))
