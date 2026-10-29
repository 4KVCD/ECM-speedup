# clocksum.py <build> [stream ...]: a block-clock build's (CMake's ECM_GPU_BLOCK_CLOCK) per-picture lines "[gpu-intra]
# POC n block clock: <kind> <ms> ms (<records>); ..." in its partial-verification logs (runs/partial), summed per kind
# over the pictures of each stream and over all streams, by falling time; with streams named, those alone
import glob, os, re, sys
from collections import defaultdict

LINE = re.compile(r'\[gpu-intra\] POC \d+ block clock:(.*)')
ITEM = re.compile(r'\s*(.+?) ([0-9.]+) ms \((\d+)\)')


def show(name, sums, top):
    total = sum(v[0] for v in sums.values())
    print(f'{name}: {total:.1f} ms')
    for kind, (ms, n) in sorted(sums.items(), key=lambda kv: -kv[1][0])[:top]:
        share = 100 * ms / total if total else 0
        print(f'  {kind:<28} {ms:10.1f} ms {share:5.1f} % {n:8d}  {1000 * ms / max(n, 1):8.1f} us each')


build, only = sys.argv[1], set(sys.argv[2:])
total = defaultdict(lambda: [0.0, 0])
for f in sorted(glob.glob(f'C:/Code/ECM/runs/partial/*.{build}.log')):
    name = os.path.basename(f)[:-len(f'.{build}.log')]
    if only and name not in only:
        continue
    per = defaultdict(lambda: [0.0, 0])
    for m in LINE.finditer(open(f, encoding='utf-8', errors='replace').read()):
        for item in m.group(1).split(';'):
            i = ITEM.match(item)
            if i:
                for d in (per, total):
                    d[i.group(1)][0] += float(i.group(2))
                    d[i.group(1)][1] += int(i.group(3))
    show(name, per, 12)
show('all', total, 100)
