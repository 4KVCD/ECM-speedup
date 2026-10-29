# shared_size.py <build> [limit]: the shared (Workgroup) memory of each of a build's SPIR-V modules, largest variables
# first, against the device's limit (default 49152 bytes: maxComputeSharedMemorySize of the RTX 3070 and most GPUs).
# The driver creates a pipeline above the limit without complaint and the device is lost when it runs (grn196 on
# 2026-10-29: intra_rec 180 bytes over), so run it after every shader change that adds shared memory. Exits 1 where
# a module is above the limit. Sizes without padding (the scalars, vectors and arrays of them a module declares).
import glob, struct, sys

build = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 49152
over = False
for path in sorted(glob.glob(f'C:/Code/ECM/build/{build}/source/Lib/GpuLib/spv/*.spv.h.spv')):
    w = open(path, 'rb').read()
    words = struct.unpack(f'<{len(w) // 4}I', w)
    names, size, consts, pointee, shared = {}, {}, {}, {}, []
    i = 5
    while i < len(words):
        op, n = words[i] & 0xFFFF, words[i] >> 16
        a = words[i + 1:i + n]
        if op == 5:   # OpName
            names[a[0]] = b''.join(struct.pack('<I', x) for x in a[1:]).split(b'\0')[0].decode()
        elif op in (21, 22):   # OpTypeInt, OpTypeFloat
            size[a[0]] = a[1] // 8
        elif op == 20:   # OpTypeBool
            size[a[0]] = 4
        elif op == 23:   # OpTypeVector
            size[a[0]] = size[a[1]] * a[2]
        elif op == 30:   # OpTypeStruct
            size[a[0]] = sum(size.get(m, 0) for m in a[1:])
        elif op == 43:   # OpConstant
            consts[a[1]] = a[2]
        elif op == 28:   # OpTypeArray (of a length that is no plain constant: 0)
            size[a[0]] = size.get(a[1], 0) * consts.get(a[2], 0)
        elif op == 32:   # OpTypePointer
            pointee[a[0]] = a[2]
        elif op == 59 and a[2] == 4:   # OpVariable in the Workgroup storage class
            shared.append((size.get(pointee[a[0]], 0), names.get(a[1], str(a[1]))))
        elif op == 54:   # OpFunction: the module's declarations end here
            break
        i += n
    if not shared:
        continue
    total = sum(s for s, _ in shared)
    module = path.replace('\\', '/').split('/')[-1].split('.')[0]
    print(f'{module}: {total} bytes ({limit - total:+d} to the limit)')
    if total > limit:
        over = True
        for s, name in sorted(shared, reverse=True)[:8]:
            print(f'  {s:6d} {name}')
sys.exit(1 if over else 0)
