# spirv_ids.py [worktree]: compiles intra_rec.comp's two modules (the second with CCP_LFNST_MODULE) as the build does
# (glslang -V --target-env vulkan1.3 -O) and prints each one's SPIR-V ID bound. The NVIDIA driver rejects a module
# above ~4.19M IDs (4194303); the second module is the larger (4.01M at M66). Large functions inline at every call
# site: keep one call site for each (see perf/HANDOFF.md).
import os, struct, subprocess, sys, tempfile

GLSLANG = r'C:\Code\ECM\tools\glslang\bin\glslang.exe'
tree = sys.argv[1] if len(sys.argv) > 1 else r'C:\Code\ECM\wt-gr'
shaders = os.path.join(tree, 'source', 'Lib', 'GpuLib', 'shaders')
with tempfile.TemporaryDirectory() as tmp:
    for name, defs in (('intra_rec', []), ('intra_rec (CCP_LFNST_MODULE)', ['-DCCP_LFNST_MODULE'])):
        out = os.path.join(tmp, 'm.spv')
        p = subprocess.run([GLSLANG, '-V', '--target-env', 'vulkan1.3', '-O', '-I' + shaders, *defs, '-o', out,
                            os.path.join(shaders, 'intra_rec.comp')], capture_output=True, text=True)
        if p.returncode != 0:
            print(f'{name}: glslang failed\n{p.stdout}{p.stderr}')
            continue
        with open(out, 'rb') as f:
            bound = struct.unpack('<5I', f.read(20))[3]
        print(f'{name}: {bound:,} IDs ({4194303 - bound:,} left)')
