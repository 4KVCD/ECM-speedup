# cpu_cus.py <build> [<build> ...]: per build, the CUs partial verification took from the CPU over all regular streams'
# logs (runs/partial/<stream>.<build>.log, written by partial.py: the "CUs taken from the CPU" line), in total and per
# reason, the largest first - what to port next. (The inter / IBC / palette CUs, ~32800, are inter pictures' and
# stay until inter prediction is ported.)
import collections, glob, re, sys

for build in sys.argv[1:]:
    total = collections.Counter()
    for path in glob.glob(f'C:/Code/ECM/runs/partial/*.{build}.log'):
        for line in open(path, encoding='utf-8', errors='replace'):
            if 'CUs taken from the CPU:' in line:
                for reason, n in re.findall(r' ([^;:]+?) (\d+);', line.split('CUs taken from the CPU:')[1]):
                    total[reason.strip()] += int(n)
    print(f'{build}: {sum(total.values())} CUs from the CPU')
    for reason, n in total.most_common(30):
        print(f'  {n:6d}  {reason}')
