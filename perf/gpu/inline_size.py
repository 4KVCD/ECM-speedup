# inline_size.py <module.spv> [N]: the size of a shader's entry point with every call inlined (SPIR-V instructions),
# and the N functions (default 15) with the largest share of it: own instructions times the times they are inlined.
# The drivers inline every call, so this is what they compile: NVIDIA's compile of a module grows much faster than
# this size (on an RTX 3070: inter_mc at 3.9M took over 40 GB and 15 minutes - more than the PC's memory budget -, at
# 0.84M ~4-7 minutes and 11 GB; intra_rec at 1.0M 1-15 minutes by its features). A function of many instructions
# should have few call sites: gather its calls into one (a loop with [[dont_unroll]], arguments chosen before the
# call) - and verify the result, as the driver's code can change with it. Needs a module that keeps its functions:
# glslang without -O (spv_build.cmake's .raw.spv) or with ECM_SPIRV_INLINE=OFF (.spv), and their names (OpName).
import struct, sys
from collections import defaultdict
from functools import lru_cache

data = open(sys.argv[1], 'rb').read()
top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
w = struct.unpack(f'<{len(data) // 4}I', data)
OP_NAME, OP_ENTRY_POINT, OP_FUNCTION, OP_FUNCTION_END, OP_FUNCTION_CALL = 5, 15, 54, 56, 57
names, own, calls, entry, cur = {}, {}, defaultdict(list), None, None
i = 5
while i < len(w):
    op, n = w[i] & 0xffff, w[i] >> 16
    if op == OP_NAME:
        names[w[i + 1]] = struct.pack(f'<{n - 2}I', *w[i + 2:i + n]).split(b'\0')[0].decode(errors='replace')
    elif op == OP_ENTRY_POINT:
        entry = w[i + 2]
    elif op == OP_FUNCTION:
        cur = w[i + 2]
        own[cur] = 0
    elif op == OP_FUNCTION_END:
        cur = None
    elif cur is not None:
        own[cur] += 1
        if op == OP_FUNCTION_CALL:
            calls[cur].append(w[i + 3])
    i += max(n, 1)


@lru_cache(None)
def inlined(f):
    return own[f] + sum(inlined(g) for g in calls[f])


times = defaultdict(int)   # how many copies of each function the inlined entry point holds


def count(f, k):
    times[f] += k
    sites = defaultdict(int)
    for g in calls[f]:
        sites[g] += 1
    for g, c in sites.items():
        count(g, k * c)


sys.setrecursionlimit(10000)
count(entry, 1)
total = inlined(entry)
print(f'{sys.argv[1]}: {len(own)} functions, {sum(own.values())} instructions, inlined {total}')
for share, f in sorted(((own[f] * times[f], f) for f in own), reverse=True)[:top]:
    print(f'{share:9d} {100 * share / total:5.1f} %  x{times[f]:<6d} own {own[f]:6d}  inlined {inlined(f):8d}  '
          f'{names.get(f, str(f)).split("(")[0]}')
