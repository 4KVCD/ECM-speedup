# spirv_noinline.py <in.spv> <out.spv> <min_inlined> [name ...]: marks functions DontInline (OpFunction's function
# control) so that a driver that honors it keeps them as calls instead of copying them into every call site: the
# named ones, else every function whose fully inlined size (inline_size.py's measure) is at least min_inlined SPIR-V
# instructions and which is inlined more than once. An experiment on the drivers' compile time and memory of large
# kernels (intra_rec), which grow much faster than the inlined size. Needs a module that keeps its functions
# (ECM_SPIRV_INLINE=OFF) and their names. Prints what it marked. On the RTX 3070 (driver 616.92, 2026-10-25) intra_rec
# with its 29 largest such functions marked compiled in half the memory, but decoded wrongly and 20 % slower.
import struct, sys
from collections import defaultdict
from functools import lru_cache

src, dst, minInlined = sys.argv[1], sys.argv[2], int(sys.argv[3])
wanted = set(sys.argv[4:])
data = bytearray(open(src, 'rb').read())
w = list(struct.unpack(f'<{len(data) // 4}I', data))
OP_NAME, OP_ENTRY_POINT, OP_FUNCTION, OP_FUNCTION_END, OP_FUNCTION_CALL = 5, 15, 54, 56, 57
DONT_INLINE = 2
names, own, calls, funcAt, entry, cur = {}, {}, defaultdict(list), {}, None, None
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
        funcAt[cur] = i
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


callers = defaultdict(int)
for f, cs in calls.items():
    for g in cs:
        callers[g] += 1
sys.setrecursionlimit(10000)
marked = []
for f in own:
    name = names.get(f, str(f)).split('(')[0]
    pick = name in wanted if wanted else ( f != entry and inlined(f) >= minInlined and callers[f] > 1 )
    if pick:
        w[funcAt[f] + 3] |= DONT_INLINE   # (OpFunction: result type, result id, function control, function type)
        marked.append((inlined(f), callers[f], name))
open(dst, 'wb').write(struct.pack(f'<{len(w)}I', *w))
for size, c, name in sorted(marked, reverse=True):
    print(f'{size:8d} inlined, {c:3d} call sites  {name}')
print(f'{len(marked)} functions marked DontInline')
