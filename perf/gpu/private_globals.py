# private_globals.py <module.spv>...: the Private (per-invocation global) variables of a SPIR-V module that one
# function other than the entry point uses alone. SPIRV-Tools' private-to-local pass (in the optimizer's performance
# passes, slangopt.py's too) moves such a variable into that function, which is only right where the function runs
# once; without inlining (ECM_SPIRV_INLINE=OFF) a function called several times then loses the value between calls.
# Run it on the module before optimization (spv_build's .raw.spv); each one listed must be used by a second function
# (e.g. reset where its state begins) or become a local.
import struct, sys

for path in sys.argv[1:]:
    w = open(path, 'rb').read()
    words = struct.unpack(f'<{len(w) // 4}I', w)
    names, private, entries, users = {}, set(), set(), {}
    func, i = None, 5
    while i < len(words):
        op, n = words[i] & 0xFFFF, words[i] >> 16
        a = words[i + 1:i + n]
        if op == 5:   # OpName
            names[a[0]] = b''.join(struct.pack('<I', x) for x in a[1:]).split(b'\0')[0].decode()
        elif op == 15:   # OpEntryPoint
            entries.add(a[1])
        elif op == 59 and a[2] == 6 and func is None:   # OpVariable, Private, at module scope
            private.add(a[1])
        elif op == 54:   # OpFunction
            func = a[1]
        elif op == 56:   # OpFunctionEnd
            func = None
        elif func is not None:
            for x in a:
                if x in private:
                    users.setdefault(x, set()).add(func)
        i += n
    for v in sorted(private, key=lambda v: names.get(v, '')):
        fs = users.get(v, set())
        if len(fs) == 1 and not fs & entries:
            f = next(iter(fs))
            print(f'{path}: {names.get(v, v)} only in {names.get(f, f)}')
