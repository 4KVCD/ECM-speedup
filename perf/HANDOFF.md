# Handoff: ECM-20.0 decoder speed-up and the Vulkan GPU decoder

Written 2026-10-10 on the i7-10700 PC so that another computer (and its Claude session) can continue. Read this
first, then `perf/DECODER.md` (the decoder speed-up log, incl. the section "GPU reconstruction") and
`perf/OPTIMIZATIONS.md` (the encoder log).

## 1. Goal and status in one paragraph

The user wants the JVET ECM-20.0 codec much faster with **bit-exact output** (decoder: every decoded frame identical
to the stock decoder `C:\Code\ECM\bin\ref\DecoderApp.exe`; encoder: identical bitstream). Target 10x single-thread; the
wall-clock gains with threads are "the easy baseline". Decoder status: single-thread ~1.2-1.3x (details in DECODER.md), frame threads
(`--FrameThreads=N`, all-intra 720p ~3x), segment-parallel decoding (`--ParallelSegments=N`), GPU loop filters
(`--GpuLoopFilter=1`: ALF, SAO, BIF, CCSAO, LF-CCCM, offset refinement on Vulkan). The user then asked for **a full
GPU decoder in Vulkan, optimized until nothing is left** ("The GPU decoder needs to fully decode the frame on the
GPU"; "Your first Vulkan version is going to be slow. That's OK"). That work is branch `dec-gpurecon`: intra
reconstruction on the GPU (`--GpuRecon`), tool levels M1-M9 done and verified bit-exact; it is **not yet faster than
the CPU decoder** (see section 8) and supports only a restricted tool set (all real CTC streams fall back to the CPU,
first reason: sign prediction).

## 2. The user's rules (follow them exactly)

Correctness / testing
- A speed-up must never change the output. Prove it on real files (identical decoded frames / md5, identical
  bitstreams). If output moves, report it and let the user decide; never accept a difference yourself.
- Find the real cause before fixing; no guessed fixes. A reproducible failure is a real bug: fix the code, not the test.
- Test on real, long, full-resolution content (1080p and 4K), not only small crops. (So far the GPU work only used
  CIF/720p; see section 9.)
- Tests must not depend on elapsed time or machine load. Canary builds when shrinking/reusing buffers. Fair CPU
  comparisons (same input/build type, paired runs, all processes counted). A QA round after every task (re-read the
  diff for bugs, run the real path, check performance).
- A canary must be able to change the output, and rebuild after removing it.

Commits / git
- One commit per optimization, each buildable and bit-exact; bug-check before committing. Fold fixes into the
  unpushed commit they belong to (`git commit --fixup <sha>` then
  `git -c sequence.editor=: rebase -i --autosquash <base>`, or set `GIT_SEQUENCE_EDITOR=true`); never a buggy commit
  followed by a fix commit. For already-pushed commits a separate fix commit is fine.
- Detailed commit messages: subject names the exact function / subsystem / metric; body = problem, cause, change,
  measurements (paired, bit-exact checks), hardware; end with `Co-Authored-By: Claude <noreply@anthropic.com>`.
  Commit with `git -c user.email=thebrianzheng@gmail.com commit ...`.
- Never use `git stash`. Never commit in the main folder (`C:\Code\ECM`); work in worktrees `C:\Code\ECM\wt-*`.
  Don't merge other sessions' branches (unless the user asks, as with the GitHub pull of 2026-10-09).
- Don't push / post / edit anything on any remote without the user's explicit permission, each time. Don't keep
  asking to push: raise it only for a release or when another PC (e.g. an AMD one) needs to test.

Process
- At most ONE agent at a time and no multi-agent workflows, even if "ultracode" appears.
- Plans: review and refine every plan up to 3 rounds before executing (order, conflicts, shared build trees, CPU
  load during benchmarks). Re-read every message to the user for accuracy and brevity (up to 3 rounds).
- Report every optimization as % speedup of the step and cumulative, with before/after numbers (at 1080p and 4K
  when possible), failures plainly, commit hashes, machine (CPU, OS, compiler, GPU + driver) and what to re-check
  elsewhere. A benchmark plan for *reported final/release numbers* needs the user's approval first.
- Keep working through usage-limit warnings; stop only when the user says so (then let benchmarks finish).

Machine
- Kill processes only by PID and only your own. Don't benchmark while another session benchmarks (check
  `Get-Process`). Never touch power or display settings (no powercfg / sleep / monitor-off). Ask before downloading
  anything (file name, source, size).

## 3. Repository, branches, what is where

- GitHub: https://github.com/4KVCD/ECM-speedup (public; remote `github`). Upstream: vcgit.hhi.fraunhofer.de ECM
  (remote `origin`, read-only). gh CLI logged in as 4KVCD on the old PC.
- `perf` (default branch on GitHub): the encoder work (done mainly on the other computer).
- `dec`: the decoder work (single-thread steps, frame threads, segment-parallel, GPU loop filters). Pushed
  2026-10-10 at 447ecde1 = 3e43989f (frame-thread drain fix: all-intra streams resend SPS/PPS each picture, which
  drained the threads; fp720_ai FT8 3.00 s -> 0.88 s) + 447ecde1 (merge of github/perf b8e5f2ba: 25 encoder
  speed-ups). QA of 447ecde1 passed: all 19 decoder streams in 12 decoder modes identical to stock, 779
  GPU-verified pictures with 0 differences, encoder suite bit-exact (1.47x vs ref).
- `dec-gpurecon` (pushed 2026-10-10, on top of `dec`): the GPU reconstruction. Commits, oldest first:
  9435a745 M1a, d7e8556b M2, 07d08f15 / 640419bb / 4fb7ccaa / ae34848e / 1001a254 / 73aca6b3 / 1b405737
  (kernel optimizations, 52x cumulative GPU time), 99987a7a ON mode (`--GpuRecon=1`; includes the SPS::m_timdMrg
  init fix), 4d691d97 host costs, 78c90fe0 frame-thread contexts, 58c2a529 DECODER.md section, 32695197 M3
  bilateral filter, ddf77bcb background creation + shared kernels + `--GpuRecon=3`, 180ccabc scheduling data on the
  GPU, then this handoff commit.
- Local-only branches on the old PC, NOT pushed (ask the user before pushing them):
  - `dec-gpu-dbf-wip` 62298b1b: GPU deblocking (G6) work in progress, parked; based on an old `dec`
    (19 commits not in `dec`, most of them older versions of GPU loop-filter commits that `dec` has in final form).
  - `dec-parsefirst` 134a0a45: experiments (env ECM_PARSE_FIRST, ECM_WPP_LAG); findings are in DECODER.md.
  - old encoder branches alf, byref, dq, exp, mc, meas-s12, tmp, merge (superseded by the other computer's `perf`).

## 4. Setting up the new computer

Directory layout the scripts assume (change the paths in `perf/gpu/*` if yours differ). Everything of the project
lives inside the main clone C:\Code\ECM (C:\Code holds one folder per project); the clone's .gitignore covers
build/ and bin/, and .git/info/exclude (local, not in git) the other folders. Never run `git clean -x` in the main
clone: it would delete the test material and the toolchain.
```
C:\Code\ECM                   main clone (never commit here)  C:\Code\ECM\wt-<name>      worktrees (git worktree add)
C:\Code\ECM\tools             portable toolchain (env.ps1 puts it on PATH; perf\build.ps1 dot-sources ..\..\tools\env.ps1)
C:\Code\ECM\build\<name>      build dirs                      C:\Code\ECM\bin\<name>     the exes of each build
C:\Code\ECM\bin\ref           the STOCK ECM-20.0 decoder/encoder (reference for every comparison)
C:\Code\ECM\seq               raw YUV test sources            C:\Code\ECM\streams        the 19 decoder test streams
C:\Code\ECM\streams_gpu       the GPU test streams            C:\Code\ECM\runs           outputs/logs
```
Don't reuse a build dir configured before 2026-10-10 without deleting it first: its CMakeCache.txt records the
old C:\Code\build and C:\Code\wt-* paths, so CMake refuses to configure it. The exes in bin\ are not affected.
Toolchain used (Windows 11): WinLibs GCC 16.2 (mingw64), CMake 4.4.4, Ninja, MinGit, gh. Vulkan: no SDK needed;
`vulkan-1.dll` is loaded at run time. Build inputs (download only with the user's OK): Vulkan-Headers tag
`vulkan-sdk-1.4.363.0` (C:\Code\ECM\tools\vulkan-headers\Vulkan-Headers-vulkan-sdk-1.4.363.0) and glslang 16.6.0
(C:\Code\ECM\tools\glslang\bin\glslang.exe). The GPU must support Vulkan 1.3 compute with buffer device address and
the **vulkanMemoryModel** feature (GpuIntraRec::create returns null otherwise); tested only on an RTX 3070 (driver
616.92).

Builds (from a worktree, e.g. `C:\Code\ECM\wt-gr` on branch dec-gpurecon):
```
perf\build.ps1 gr -NoLto '-DECM_VULKAN=ON' '-DECM_VULKAN_HEADERS=C:/Code/ECM/tools/vulkan-headers/Vulkan-Headers-vulkan-sdk-1.4.363.0/include' '-DECM_GLSLANG=C:/Code/ECM/tools/glslang/bin/glslang.exe'
perf\build.ps1 grL '-DECM_VULKAN=ON' <same two paths>        # LTO, for timing
```
Quote the -D arguments in PowerShell: unquoted, `-DNAME=C:/...` is split at the colon and the configure fails.
Likewise PowerShell 5.1 drops an empty argument, so `-o ""` makes the next option the output file name.
Pitfalls: all build dirs of one worktree share its in-source `lib/` and `bin/`: before switching build dir run
`perf\gpu\cleanlibs_gr.ps1` (edit its path to your worktree) or the next build may link stale libraries. LTO needs
`-DCMAKE_POLICY_DEFAULT_CMP0069=NEW` (build.ps1 passes it). Shaders (`source/Lib/GpuLib/shaders/*.comp`) are
compiled by CMake into `spv/<name>.spv.h`; new .comp files are picked up automatically. A source restored with
Copy-Item keeps its old mtime and is NOT rebuilt (touch it). PowerShell 5.1 `Set-Content -Encoding utf8` writes a
BOM; write commit messages to a file without BOM and use `git commit -F file`. The tool environment blocks
commands that combine `Remove-Item` with `//` text. `perf\gpu\crlf.ps1 <worktree>` converts changed files to CRLF
(the repo uses CRLF).

## 5. Test material

- Sources in C:\Code\ECM\seq (not in git): foreman_352x288.yuv, ParkJoy_1280x720.yuv (62 frames),
  FourPeople_1280x720.yuv; y4m originals in seq\y4m. 1080p/4K: none of our own yet; the VVenc-Vulkan session on the
  old PC had crowd_run/ducks/park_joy 1080p50 y4m in C:\Code\VVenc-Vulkan\data (another session's folder: ask
  first). The user wants 1080p/4K testing.
- GPU test streams (C:\Code\ECM\streams_gpu, not in git, regenerate with the scripts; all-intra, ECM tools off except the
  level's tools, `perf\gpu\enc_m1.ps1` lists the disabled tools; it uses the bit-exact optimized encoder build qaCpu, which writes the same bitstream as stock):
  - `enc_set.ps1 m1`                                         M1: planar/DC/angular, DCT-II only
  - `enc_set.ps1 m2 --IntraToolControlMode=1 --IntraPredBf=0` M2: + reference filter, PDPC, intra fusion
  - `enc_set.ps1 m3 --IntraToolControlMode=1 --IntraPredBf=1` M3: + intra-prediction bilateral filter
  - `enc_set.ps1 m4 --IntraToolControlMode=1 --IntraPredBf=1 --DualITree=1` M4: + dual tree
  - `enc_set.ps1 m5 --IntraToolControlMode=1 --IntraPredBf=1 --DualITree=1 --LMCSEnable=1` M5: + LMCS; `m5s`: the
    same with `--DualITree=0`. The encoder turns LMCS off in the picture headers of the CIF foreman pictures, so
    only the 720p streams exercise the LMCS paths.
  - M6 (M5's arguments plus): `m6` `--DepQuant=2` (dependent quantization, 8 states, as in CTC), `m6q`
    `--DepQuant=2 --QCS=1` (as in CTC), `m6s` `--QCS=1` (scalar quantization with QCS), `m6d1` `--DepQuant=1`
    (4 states).
  - M7 (m6q's arguments plus `--JointCbCr=1`): `m7` dual tree, `m7s` `--DualITree=0`, `m7q` `--DualITree=0
    --CbQpOffset=3 --CrQpOffset=-3 --CbCrQpOffset=-2` (unequal chroma QPs). In m7 the foreman streams use the
    negative ICT modes, park720 the positive ones (counted with a temporary print).
  - M8 (m7's arguments plus `--MTS=1 --MTSExt=1`): `m8`, explicit intra MTS; its foreman streams use all seven intra
    transform types in 31 horizontal/vertical pairs. Implicit MTS (`--MTSImplicit=1`) cannot be encoded without DIMD
    (CHECK "Wrong candModeListForTransform size").
  - M9 (m8's arguments plus `--TransformSkip=1 --ChromaTS=1 --BDPCM=1`): `m9` (camera content: transform skip in
    ~15 % of the coded blocks of m9_fore32, a few BDPCM blocks) and `m9_screen22/32/37` (`enc_m1.ps1 m9_screen<q>
    C:\Code\ECM\seq\screen_352x288.yuv 352 288 4 --TemporalSubsampleRatio=1 -q <q> <args>`; mostly transform skip,
    hundreds of BDPCM blocks incl. chroma, transform skip with joint CbCr). screen_352x288.yuv is synthetic
    screen-like content from `perf\gpu\gen_screen.py` (deterministic).
  - each set: fore22/32/37 (CIF), park720, four720 (2-4 pictures). Longer ones: m2_fore33, m2_park16, m3_park16
    (`enc_m1.ps1 <name> <yuv> 1280 720 16 --TemporalSubsampleRatio=1 -q 32 <level args>`), m3_park60 (60 frames,
    `enc_long.ps1 m3_park60 ...\ParkJoy_1280x720.yuv 1280 720 4 15 -q 32 --IntraToolControlMode=1 --IntraPredBf=1`:
    4 parallel segments concatenated; each starts with an IDR). Move the `*_seg*` files out of streams_gpu before
    running the checkers (they pick up every .bin).
  - The AI cfg needs `--TemporalSubsampleRatio=1`; the 350x286 encode crashes the stock encoder.
  - Note: EnableTMnoninterTools=0 in these streams leaves stock `SPS::m_timdMrg` unparsed (fixed by a ctor init).
- The 19 regular decoder streams (C:\Code\ECM\streams, CTC-like AI/RA/LD incl. *_md5) with `perf\decbench.py`; stock
  references in C:\Code\ECM\runs\ref_big etc.

## 6. Scripts (perf/gpu, copied from the old session's scratchpad)

| script | use |
|---|---|
| `gpucheck.py <build> [streams] -- <decoder args>` | decodes every GPU stream with the build and with the stock decoder (cached), compares all frames, prints the `[gpu-intra]` lines. Use `--GpuRecon=2` (verify) and `--GpuRecon=3`, also with `--FrameThreads=8`. |
| `stress.py <build> <reps> [args]` | repeats verify decodes, saves the output of any run that is not "all verified, 0 differences, 0 skipped" (caught the m_timdMrg bug, 1 in ~650 runs) |
| `gpuload.py <build> <seconds>` | keeps the GPU busy (decodes with `--GpuRecon=3 --FrameThreads=8`); run it beside stress.py, GPU synchronization bugs needed the load to show (intra_resi scratch, 2026-10-11) |
| `gpubench.py <build> [reps]` | GPU kernel time per stream (verify mode, best of reps) |
| `e2e.py <build> <streams>` | wall time CPU / CPU FT8 / GPU on / GPU on FT8, output checked |
| `walls.py <build> <stream> <reps> -- args \| args ...` | wall time per configuration (no output check) |
| `enc_m1.ps1`, `enc_set.ps1`, `enc_long.ps1` | test stream encoding (section 5) |
| `gen_screen.py <out.yuv> [w h frames]` | synthetic screen-like 4:2:0 content (transform skip / BDPCM test streams) |
| `qa.ps1` | the full QA before a push of `dec` (3 builds, 12 decoder modes on the 19 streams, encoder suite); paths point at wt-dec |
| `buildgpu.ps1`, `cleanlibs.ps1`, `cleanlibs_gr.ps1`, `crlf.ps1` | build helpers (paths: wt-dec / wt-gr) |

Profiling: `perf\sprof\sprof.exe out.txt <command>` (sampling profiler, ~1 ms) and
`python perf\sprof\sprof_report.py out.txt <exe> [--top N] [--callers F] [--callees F]` (needs mingw64\bin on PATH
for nm/objdump; profile a non-LTO build).

## 7. The GPU reconstruction: design

Modes (`--GpuRecon`): 1 on (the CPU reconstructs until the background-created device is ready, then the GPU),
2 verify (the CPU reconstructs, the GPU result is compared per block and per sample, plus the GPU's scheduling data
against the host's), 3 on and waiting for the device at the first picture (use for tests/benchmarks on short
streams). Works in every frame-thread context.

Flow per supported intra picture (ON mode): DecSlice parses the whole slice first (parsing never needs the
reconstruction - proven on all 19 streams), then `decompressCtu` runs derive-only (`DecCu::setGpuDeriveOnly`,
`IntraPrediction::setParamsOnly`): per TU `GpuIntraRec::record` stores geometry, final mode, IntraPrediction
parameters (`IpaSnapshot`), dequant/transform params and the nonzero coefficient region; then
`GpuIntraRec::endPicture` uploads and runs on one Vulkan stream: `intra_resi` (all residuals; dequant folded into
the first DCT-II stage), `intra_deps` passes 0-3 + `scan` (owner maps, availability, read ranges, dependency graph,
ready queue), `intra_rec` (persistent workgroups, default 32, env ECM_GPU_GROUPS; a block is ready when the blocks
owning the samples it reads are done; the last finishing dependency hands its dependent to its own workgroup;
Vulkan memory model with coherent buffers, release/acquire atomics), then downloads the planes into the
reconstruction buffer. If a picture or block is unsupported (`GpuIntraRec::supported()` SPS gate, `record()` per
CU), the CPU reconstructs it.

Rule that keeps it bit-exact: the host may compute anything that depends only on syntax/geometry/decoding order;
everything that depends on samples runs on the GPU, implemented line by line after the C++ (IntraPrediction.cpp,
TrQuant.cpp, BilateralFilter.cpp). `supported()` must reject every tool whose mode derivation or prediction reads
reconstructed samples or that the kernels lack.

Files: `source/Lib/GpuLib/GpuIntraRec.{h,cpp}`, `shaders/intra_common.glsl` (Tu = 63 ints and Params layouts,
static_asserts on the host), `intra_resi.comp`, `intra_rec.comp`, `intra_deps.comp`, `scan.comp`, `VkCompute.*`
(device with vulkanMemoryModel + optional shader clock; up to 8 queues round-robin); `DecoderLib/DecCu.cpp`
(record calls), `DecSlice.cpp` (parse-first, derive pass, CPU fallback), `DecLib.cpp` (setGpuRecon,
xPollGpuRecon), `DecFrameThreads.cpp`; option in `DecApp/DecAppCfg.cpp`.

Tools implemented (M1-M3): planar (incl. directional planar), DC, angular with 16/10-tap cubic and 6-tap Gaussian
luma / 2-tap chroma interpolation, 8x reference lengths, reference filter, PDPC (planar/DC, pure H/V, gradient,
inverse angle), intra fusion, DisableRefFilter, intra-prediction bilateral filter (JVET-AK0118: angular reference
texture gate over the full 8W/8H arrays, prediction MAD gate, mirror + reconstruction padding with the CPU's
skip-the-last-unit quirk, 5x5 diamond), DCT-II 2..128, scalar dequant, adaptive luma clipping, 4:2:0/4:0:0, one
slice per picture. M4: the dual tree of intra slices (no kernel change). M5: LMCS (mapped luma clipping range, the
bilateral filter on inverse-mapped samples, chroma residual scaling from the luma next to the CU; the host uploads
the LUTs and the chroma scale per luma value, `Reshape::getChromaAdjNeighbours` finds the neighbouring luma CUs).
M6: dependent quantization (the host records quantization indices) and QCS (`m_coeffShift64` from Quant.h).
M7: joint CbCr (the Cb block computes both residuals in intra_resi; Tu::jccr, jccrOffset).
M8: explicit intra MTS (TrQuant::getInvTransformTypes per block; every transform matrix read back by impulses with
TrQuant::getInvTransformMatrix; Tu::mtxV, mtxH). M9: transform skip and BDPCM (Tu::transformSkip; invResDPCM on
the host; BDPCM blocks predict as mode 18 / 50 with the filters off).

## 8. Numbers (i7-10700 8C/16T, Win11, GCC 16.2, RTX 3070 driver 616.92)

GPU kernel time, 24 M1/M2 pictures: 2845.5 ms (first kernel) -> ~55 ms (~52x; step table in DECODER.md).
Per 720p picture: M2 ~5 ms, M3 8.3 ms (park720) / 12.8 ms (four720); M3's angular BF gate reads the full
reference arrays, which roughly doubles the longest dependency chain (park720 1127 -> 1761, four720 1198 -> 2325;
without those reads 1081 / 1192). A restructure of the BF with subgroup-only reductions was slower (dropped).

Wall time, m3_park16 (16 x 720p intra, LTO, best of 3) at ddf77bcb:
CPU 1.51 s, GPU on 1.52 s (3 of 16 pictures still on the CPU while the device starts), CPU FT8 0.73 s.
GPU on with FT8 was 1.37 s at the build before ddf77bcb (GPU engaged in every context from the start): with frame
threads the GPU path is currently a net loss. 180ccabc (scheduling on the GPU) is not timed yet.

Profile of a GPU-on decode (non-LTO, m2_park16, share of all samples): parsing ~29 %, derive-only pass ~19 %
(MPM derivation getIntraMPMs -> getSpatialIpm -> getPURestricted 6.6 %, xIntraRecBlk 8.7 %), endPicture 14 %
(host dependency analysis ~7 % - now moved to the GPU by 180ccabc - and the GPU wait ~5 %), loop filter CPU work
9 % (even with LoopFilterDisable), Vulkan start-up 17 % (now hidden by background creation).

## 9. Next steps (in the order I would take them)

1. Time 180ccabc (paired grL builds, `walls.py` with `--GpuRecon=3`, CPU vs GPU, FT8; use m3_park60).
2. Done (2026-10-11): M4 dual tree, M5 LMCS, M6 dependent quantization and QCS, M7 joint CbCr, M8 explicit MTS,
   M9 transform skip and BDPCM.
   All 19 regular streams now stop at "sign prediction" (the next sequence-level check; the block-level tools of
   step 4 follow).
3. Make the GPU path win on CPU time per picture: the derive-only pass (record during parsing or a leaner pass;
   speed up getSpatialIpm/getPURestricted, which also helps the CPU decoder), GPU behaviour with frame threads
   (8 contexts x 32 persistent workgroups oversubscribe the GPU; spinning workgroups; try fewer groups per context,
   more frame threads than cores to hide GPU waits, overlapping a picture's GPU work with the next picture's parse).
4. More tool levels, roughly in the order CTC AI needs them: MRL/TMRL, ISP, MIP, LFNST/NSPT, implicit MTS (with
   DIMD), sign prediction (reads reconstruction: GPU), CCLM/CCCM and the other
   cross-component tools (GPU), DIMD/TIMD/MPM sorting/chroma reordering/DBV (mode derivations that read the
   reconstruction: these have to move to the GPU too), IntraTMP/IBC, then inter pictures. Each level: new test set
   with exactly that tool on, verify mode identical, canaries, commit.
5. In-loop filters in ON mode on the GPU without a CPU round trip (deblocking: see dec-gpu-dbf-wip; ALF/SAO/... are
   on the GPU already via --GpuLoopFilter, but on the CPU's picture).
6. 1080p/4K verification and timing once sources are agreed with the user; check on other GPU makers (AMD, Intel)
   before any release.

## 10. Findings worth knowing

- RTX 3070: plain loads through `coherent`/`volatile` buffer references returned stale L1 data across workgroups;
  use `#pragma use_vulkan_memory_model` + GL_KHR_memory_scope_semantics. Persistent workgroups + an atomic ready
  queue are deadlock-free dependency scheduling.
- RTX 3070, also with the memory model: data passed between the invocations of one workgroup through a global buffer
  and a barrier was read stale by whole warps while another process loaded the GPU (intra_resi, fixed by shared
  memory). Pass data inside a workgroup through shared memory, and stress-test with `gpuload.py` running.
- `supported()` / `record()` must reject every tool the kernels lack, including tools that only change a CPU code
  path the GPU replaces (dequantization, transforms, clipping): dependent quantization was not rejected until M6,
  and the reconstruction on decoded such streams wrong. With a new level, look through the CPU path it replaces for
  flags the GPU ignores, and encode a test set with each such flag on.
- Class layouts must not depend on `ECM_VULKAN`: the define reaches only the targets that link DecoderLib or GpuLib,
  not EncoderLib, and EncCu derives from DecCu. GPU members under `#if ECM_VULKAN` in DecCu made the encoder of every
  Vulkan build of dec-gpurecon crash at start-up (fixed 2026-10-11). Run the encoder suite on a Vulkan build too.
- The NVIDIA GPU drops to a low clock between pictures (kernels ~3x slower); power settings are off limits.
- `BilateralFilter` derives its LUTs in `create()`, not in the constructor.
- ECM intra reconstruction is raster-serial within a picture once intra/IBC CUs exist (JVET_AJ0172 IBC/IntraTMP
  reference area, BV-guided tools); parallelism comes from independent blocks, components and pictures.
- A dependent per-CU decision chain runs ~7x slower on one GPU workgroup than on one CPU core.
- Edge capacity of the GPU dependency graph: 8 per block at first, grows to 1.25x the largest seen; an overflow
  makes all kernels stand down and the host resubmits the picture (tested with capacity 16).
