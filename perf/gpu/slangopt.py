# slangopt.py [spirv-opt options...] <in.spv> -o <out.spv>: a stand-in for spirv-opt where the Vulkan SDK is not
# installed but NVIDIA Nsight Graphics is. Its slang-glslang.dll holds a SPIRV-Tools optimizer behind
# glslang_compile_1_2 (Slang's request, action 2: optimize SPIR-V) that raises the ID limit to 0xFFFFFC, so intra_rec
# optimizes there although glslang -O stops at SPIR-V's 4.19M IDs ("ID overflow"). The pass list is Slang's (level 1,
# its default: SPIRV-Tools' performance passes with exhaustive inlining), not the options given, except for one:
# without --inline-entry-points-exhaustive (spv_build.cmake's ECM_SPIRV_INLINE=OFF) the DLL's registration of the
# inlining pass is skipped, which needs the known DLL build (Nsight Graphics 2026.3.1; checked byte for byte). The
# driver's cold compile of an inlined intra_rec pipeline takes minutes and many GB more. Used through slangopt.cmd as
# -DECM_SPIRV_OPT=.../perf/gpu/slangopt.cmd; ECM_SLANG_GLSLANG names the DLL if it is not under Program Files' Nsight
# Graphics. Prints nothing on success, as spv_build.cmake expects of spirv-opt.
import ctypes, glob, os, sys

OutputFunc = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)


class Request(ctypes.Structure):   # Slang's glslang_CompileRequest_1_2 (0x70 bytes)
    _fields_ = [('sizeInBytes', ctypes.c_size_t), ('sourcePath', ctypes.c_char_p),
                ('inputBegin', ctypes.c_void_p), ('inputEnd', ctypes.c_void_p),
                ('diagnosticFunc', OutputFunc), ('diagnosticUserData', ctypes.c_void_p),
                ('outputFunc', OutputFunc), ('outputUserData', ctypes.c_void_p),
                ('slangStage', ctypes.c_int), ('action', ctypes.c_uint),
                ('optimizationLevel', ctypes.c_uint), ('debugInfoType', ctypes.c_uint),
                ('spirvTargetName', ctypes.c_char_p), ('spirvVersion', ctypes.c_uint64),
                ('entryPointName', ctypes.c_char_p), ('reserved', ctypes.c_uint64)]


ACTION_OPTIMIZE_SPIRV = 2
LEVEL_DEFAULT = 1
# the default level's registration of inline-entry-points-exhaustive (create the pass, RegisterPass, destroy the
# token), at this offset of the 2026.3.1 DLL; NOPs there leave the other passes as they are
INLINE_REGISTRATION_RVA = 0x7667
INLINE_REGISTRATION = bytes.fromhex('48 8d 4d d7 e8 10 37 19 00 90 48 8b d0 48 8d 4d 6f e8 c3 6e 19 00 90'
                                    '48 8d 4d d7 e8 49 f6 18 00')


def skipInlining(lib):
    k32 = ctypes.windll.kernel32
    k32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    k32.FlushInstructionCache.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    at, n = lib._handle + INLINE_REGISTRATION_RVA, len(INLINE_REGISTRATION)
    if ctypes.string_at(at, n) != INLINE_REGISTRATION:
        sys.exit('slangopt.py: not the known slang-glslang.dll build: cannot leave out inlining')
    old = ctypes.c_uint32()
    if not k32.VirtualProtect(at, n, 0x40, ctypes.byref(old)):   # PAGE_EXECUTE_READWRITE
        sys.exit('slangopt.py: VirtualProtect failed')
    ctypes.memset(at, 0x90, n)
    k32.VirtualProtect(at, n, old.value, ctypes.byref(old))
    k32.FlushInstructionCache(k32.GetCurrentProcess(), at, n)


args = sys.argv[1:]
if '-o' not in args:
    sys.exit('slangopt.py: no -o <out.spv>')
dst = args[args.index('-o') + 1]
inputs = [a for i, a in enumerate(args) if not a.startswith('-') and (i == 0 or args[i - 1] != '-o')]
if len(inputs) != 1:
    sys.exit(f'slangopt.py: one input module expected, got {inputs}')
dll = os.environ.get('ECM_SLANG_GLSLANG') or next(iter(sorted(glob.glob(
    os.path.expandvars(r'%ProgramFiles%\NVIDIA Corporation\Nsight Graphics *\host\windows-desktop-nomad-x64'
                       r'\slang-glslang.dll')), reverse=True)), '')
if not os.path.isfile(dll):
    sys.exit('slangopt.py: no slang-glslang.dll (set ECM_SLANG_GLSLANG)')

data = open(inputs[0], 'rb').read()
buf = ctypes.create_string_buffer(data, len(data))
out, diag = [], []
onOut = OutputFunc(lambda p, n, u: out.append(ctypes.string_at(p, n)))
onDiag = OutputFunc(lambda p, n, u: diag.append(ctypes.string_at(p, n)))
req = Request(sizeInBytes=ctypes.sizeof(Request), inputBegin=ctypes.addressof(buf),
              inputEnd=ctypes.addressof(buf) + len(data), diagnosticFunc=onDiag, outputFunc=onOut,
              action=ACTION_OPTIMIZE_SPIRV, optimizationLevel=LEVEL_DEFAULT)
lib = ctypes.CDLL(dll)
if '--inline-entry-points-exhaustive' not in args:
    skipInlining(lib)
rc = lib.glslang_compile_1_2(ctypes.byref(req))
result = b''.join(out)
if rc or diag or len(result) < 20:
    sys.exit(f'slangopt.py: {inputs[0]}: rc {rc}, {len(result)} bytes\n' + b''.join(diag).decode(errors='replace'))
bound = int.from_bytes(result[12:16], 'little')
if bound > 4194303:   # (SPIR-V's universal limit: a driver may refuse the module)
    sys.exit(f'slangopt.py: {inputs[0]}: ID bound {bound} past 4194303')
open(dst, 'wb').write(result)
