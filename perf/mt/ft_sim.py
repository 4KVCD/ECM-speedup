"""Frame-thread (--FrameThreads) schedule simulation from the log of a serial encode. Per picture, the work (ET) splits
into a search, which waits for its references, for the last earlier picture of its temporal layer since the last IRAP
(temporal CABAC context store and AMaxBT statistics) and for a free stack (at most K pictures in flight, stacks taken
in coding order), and a finish of the given fraction of the work, in coding order. Prints the makespan for
K = 1..32 with a barrier after every GOP (as implemented) and without it.
usage: ft_sim.py <serial encoder log> [finish fraction, default 0.1] [GOP size, default 32]"""
import re
import sys

log = sys.argv[1]
fin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
gop = int(sys.argv[3]) if len(sys.argv) > 3 else 32

pics = []
for l in open(log, errors="replace"):
    m = re.match(r"POC\s+(\d+) LId:\s*\d+ TId: (\d+) \( (\w+), .*\[ET\s+([\d.]+) \] \[L0([^\]]*)\] \[L1([^\]]*)\]", l)
    if m:
        poc, tid, nal, et = int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4))
        refs = {int(x.rstrip("c")) for x in (m.group(5) + " " + m.group(6)).split() if x}
        pics.append((poc, tid, et, refs, nal in ("IDR_N_LP", "IDR_W_RADL", "CRA")))
total = sum(p[2] for p in pics)


def gop_of(i):
    return 0 if i == 0 else (i - 1) // gop + 1   # the first picture is a GOP of its own


def simulate(K, barrier):
    done = {}          # poc -> finish end time
    stack_free = [0.0] * K
    last_finish = 0.0
    last_of_tid = {}
    t_gop_start = 0.0
    for i, (poc, tid, et, refs, irap) in enumerate(pics):
        if irap:
            last_of_tid = {}   # RAS init: the context store and the AMaxBT statistics start again
        g = gop_of(i)
        if barrier and g > 0 and i > 0 and gop_of(i - 1) != g:
            t_gop_start = max(done.values())
        start = max([t_gop_start, stack_free[i % K]] + [done[r] for r in refs if r in done])
        if tid in last_of_tid:
            start = max(start, done[last_of_tid[tid]])
        search_end = start + et * (1 - fin)
        fin_start = max(search_end, last_finish)
        fin_end = fin_start + et * fin
        last_finish = fin_end
        done[poc] = fin_end
        stack_free[i % K] = fin_end
        last_of_tid[tid] = poc
    return max(done.values())


print(f"{len(pics)} pictures, serial work {total:.0f} s, finish fraction {fin}")
for K in (1, 2, 4, 8, 16, 32):
    a = simulate(K, True)
    b = simulate(K, False)
    print(f"K={K:2d}  GOP barrier {a:7.0f} s ({total / a:4.2f}x)   overlapping GOPs {b:7.0f} s ({total / b:4.2f}x)")
