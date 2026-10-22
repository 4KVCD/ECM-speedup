# spirv_ids.py [worktree]: compiles intra_rec.comp's two modules (the second with CCP_LFNST_MODULE) as the build does
# with ECM_SPIRV_OPT (GpuLib's spv_build.cmake: glslang without its optimizer, then spirv-opt with glslang -O's passes
# and --compact-ids) and prints each one's SPIR-V ID bound and compile time. The ID limit of a module (4194303, which
# the NVIDIA driver also enforces) no longer binds after compacting (intra_rec ~0.3M); glslang's own optimizer, the
# build without ECM_SPIRV_OPT, fails past it while optimizing (~3.3M IDs allocated for the main module at M70).
import os, struct, subprocess, sys, tempfile, time

GLSLANG = r'C:\Code\tools\glslang\bin\glslang.exe'
SPIRV_OPT = r'C:\Code\tools\vksdk\Bin\spirv-opt.exe'
tree = sys.argv[1] if len(sys.argv) > 1 else r'C:\Code\ECM\wt-gr'
lib = os.path.join(tree, 'source', 'Lib', 'GpuLib').replace('\\', '/')
with tempfile.TemporaryDirectory() as tmp:
    for name, defs in (('intra_rec', ''), ('intra_rec (CCP_LFNST_MODULE)', '-DCCP_LFNST_MODULE')):
        out = os.path.join(tmp, 'm.spv.h').replace('\\', '/')
        start = time.time()
        p = subprocess.run(['cmake', f'-DGLSLANG={GLSLANG}', f'-DSPIRV_OPT={SPIRV_OPT}',
                            f'-DSHADER={lib}/shaders/intra_rec.comp', f'-DOUT={out}', '-DVAR=spv_m',
                            f'-DINCLUDE={lib}/shaders', f'-DDEFINES={defs}', '-P', f'{lib}/spv_build.cmake'],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(f'{name}: failed\n{(p.stdout + p.stderr)[-3000:]}')
            continue
        with open(out + '.spv', 'rb') as f:
            bound = struct.unpack('<5I', f.read(20))[3]
        print(f'{name}: {bound:,} IDs ({time.time() - start:.0f} s)')
