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
reconstruction on the GPU (`--GpuRecon`), tool levels M1-M25 done and verified bit-exact (M19: DIMD, M20: OBIC, M21:
intra merge, M22: the transforms of DIMD CUs, M23: LFNST of MIP, M24: TIMD, M25: OBIC and intra merge with TIMD
neighbours); it is **not yet
faster than the CPU decoder** (see section 8) and supports only a restricted tool set (all real CTC streams fall back
to the CPU, first reason: TMRL).

## 2. The user's rules (follow them exactly)

Correctness / testing
- A speed-up must never change the output. Prove it on real files (identical decoded frames / md5, identical
  bitstreams). If output moves, report it and let the user decide; never accept a difference yourself.
- Find the real cause before fixing; no guessed fixes. A reproducible failure is a real bug: fix the code, not the test.
- Test on real, long, full-resolution content (1080p and 4K), not only small crops. (The GPU test streams have three
  short 1080p ones since 2026-10-11; long 1080p/4K runs are still to do, see section 9.)
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
  FourPeople_1280x720.yuv, CrowdRun / Ducks / ParkJoy_1920x1080.yuv (60 frames each, from crowd_run / ducks /
  park_joy_1080p.y4m); y4m originals in seq\y4m. 4K: none yet. The user wants 1080p/4K testing.
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
  - M10 (m9's arguments plus `--MIP=1`): `m10`; its foreman streams have 1235 MIP blocks of all three size classes,
    transposed and not, 4x4 to 32x32.
  - M11 (m9's arguments plus `--IntraLFNSTISlice=1 --LFNSTExt=1 --NSPT=1`, without MIP, as LFNST of MIP blocks is not
    supported yet): `m11`; its foreman streams: lfnstIdx 1-3 on all components, 382 blocks with separable LFNST and
    3348 NSPT blocks (4x4 to 32x8) among 7896 coded blocks. `m11m_fore32` (`enc_m1.ps1`, also `--MIP=1`): has MIP
    blocks with LFNST, so its pictures fall back to the CPU ("LFNST of MIP").
  - 1080p (m11's arguments, 2 pictures, QP 32): `h11_crowd`, `h11_ducks`, `h11_park` (`enc_m1.ps1 h11_crowd
    C:\Code\ECM\seq\CrowdRun_1920x1080.yuv 1920 1080 2 --TemporalSubsampleRatio=1 -q 32 <m11 args>`; ~20 min each).
  - M12 (m11's arguments plus `--MRL=1`; TMRL and MIP stay off): `m12`; its fore32 and fore22 streams have 1292 MRL
    blocks on all lines (1: 690, 3: 320, 5: 128, 7: 92, 12: 62), 817 with the fusion line, 45 DC (40 with the
    bilateral filter), all angle signs, and MRL unit groups unavailable at the picture's edges. `enc_m12_edges.ps1`:
    `m12e_*` (64x64 / 128x64 crops via `crop.py`: many blocks at the edges) and `m12s_*` (synthetic content via
    `gen_mrl_edge.py` on which the encoder picks lines 5-12 near the top).
  - M13 (m12's arguments plus `--ISP=1`): `m13`; its foreman streams have 1312 ISP sub-partitions (both splits,
    1x16 to 32x4, 114 in shared 4-wide regions, 93 with a coded 1-D transform, 39 with the side along the split
    unavailable, planar / DC / all angle classes, PDPC). `m13c_0..47` (`enc_crops.py m13c 48 13 <m13 args>`, 64x64
    crops): 122 bilateral filters of ISP sub-partitions with partial padding, 32 of them with the CPU's writes past the
    row (counted with a temporary print).
  - M14 (sign prediction as in CTC; LFNST / NSPT off, MIP on): `m14` with `--IntraToolControlMode=1 --IntraPredBf=1
    --DualITree=1 --LMCSEnable=1 --DepQuant=2 --QCS=1 --JointCbCr=1 --MTS=1 --MTSExt=1 --TransformSkip=1 --ChromaTS=1
    --BDPCM=1 --MIP=1 --MRL=1 --ISP=1 --NumSignPred=8 "--NumSignPredOverrideByQP=22 4"`; `m14s` the same with
    `--DualITree=0`; `m14a` with `"--NumSignPredOverrideByQP=0 8" --Log2SignPredArea=5` (the 32x32 sign area at QP 22
    and 37, where the AI configuration does not force 8x8); `m14l_fore32` (`enc_m1.ps1`, 2 pictures, m13's arguments
    plus sign prediction): LFNST / NSPT blocks with predicted signs (fell back until M16). Counted with a temporary
    print: m14_fore32 has 2258 blocks with predicted signs (luma and chroma, joint
    CbCr with Cb and with Cr coded, ISP, MIP, MRL, the bilateral filter, at the picture's edges, 1 to 8 signs), 1822
    with a negative decided sign (the residual computed again), 1324 with a wrong prediction; m14a_fore22 predicts
    coefficients up to (19, 19); m14_park720 / m14s_park720 have 613 / 1215 chroma blocks with LMCS chroma scaling.
    1080p: `h14_crowd`, `h14_park` (`enc_m1.ps1`, 2 pictures, QP 32, m14's arguments).
  - M15 (LFNST and NSPT on the GPU; the m11-m13 sets have both): `m15n` (m13's arguments with `--NSPT=0`: LFNST also
    on the small blocks NSPT otherwise takes, i.e. its 4x4 and 8x8 regions; m15n_fore32 has 362 / 125 LFNST blocks
    with a 4x4 region (plain / transposed), 485 / 157 with 8x8, 162 / 55 with 16x16), `m15s` (m13's arguments with
    `--DualITree=0`: in a single tree only luma takes LFNST / NSPT). NSPT in m11_fore32: all 12 block shapes, 870
    plain / 352 transposed; chroma and joint CbCr blocks with LFNST / NSPT in the dual-tree sets.
  - M16 (sign prediction of LFNST / NSPT blocks): `m16` (m13's arguments plus `--NumSignPred=8
    "--NumSignPredOverrideByQP=22 4"`), `m16n` (the same with `--NSPT=0`). Counted with a temporary print:
    m16_fore32 has 191 LFNST blocks (16x16 region) and 1052 NSPT blocks (345 transposed) with predicted signs, 176 /
    919 of them with a negative decided sign; m16n_fore32 456 / 604 / 203 LFNST blocks with 4x4 / 8x8 / 16x16 regions
    (373 / 520 / 179 with a negative sign).
  - `fx_wpp_park720`, `fx_tiles_park720` (m11's arguments plus `--WaveFrontSynchro=1`, resp. `--EnablePicPartitioning=1
    --TileColumnWidthArray=3 --TileRowHeightArray=2 --RasterScanSlices=1 --RasterSliceSizes=100`: one slice of 4x3
    tiles): must fall back to the CPU ("wavefronts", "tiles").
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
| `crop.py`, `gen_mrl_edge.py`, `enc_m12_edges.ps1` | small crops of the test sequences and synthetic MRL content; the M12 edge-case streams |
| `enc_crops.py <prefix> <count> <seed> [args]` | test streams of random 64x64 crops (many blocks at the picture's edges), e.g. the M13 crops |
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

Files: `source/Lib/GpuLib/GpuIntraRec.{h,cpp}`, `shaders/intra_common.glsl` (Tu = 84 ints and Params layouts,
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
the host; BDPCM blocks predict as mode 18 / 50 with the filters off). M10: MIP (Tu::mipMatrix etc.; recorded with the
DC mode for the read range and the bilateral filter). M11: LFNST / NSPT on the host side (replaced by M15). M12: MRL
(Tu::mrl; the MRL unit groups left of and above the corner in avail[line][5..6]; Tu::bfAvail, line 0's availability
for the bilateral filter of MRL DC blocks). M13: ISP (Tu::isp, cuX..cuH, regX / regW the prediction region, ispSide, ispPrev the
scheduling chain; refSource in GpuIntraRec.cpp and intra_common.glsl gives every reference entry's source sample for
the fill and the dependencies; 1-D transforms in intra_resi). M14: sign prediction (GpuIntraRec::prepareSignPred takes
a block's predicted levels and coded signs with DepQuant::getPredictedSigns and gives the TU its levels back;
intra_resi computes the templates into the signPred buffer and the residual with the predicted signs positive;
intra_rec's signPrediction decides the signs after the prediction and computes the residual again where one is
negative; Tu::spCoded, spOffset, resiFrom: with joint CbCr the coded component's block decides and the other block
waits for it). M15: LFNST and NSPT on the GPU (GpuIntraRec's invSecondary gives a block's matrix, scan, sizes and
transposition with the CPU's choices; the CPU's int8 kernel tables, the scans' positions and the LFNST output
arrangement are uploaded once (Shared::secKernels, matrices); intra_resi runs them; Tu::secKind etc.). M16: sign
prediction of LFNST / NSPT blocks (intra_resi's templates through NSPT / LFNST, intra_rec's recomputed residual too).
M17: intra modes derived on the GPU, the base for DIMD / TIMD / MPM sorting, whose modes depend on reconstructed
samples. Per CU and channel type a record (`Cu`: the parsed MPM syntax, the CUs at getSpatialIpm's 23 positions as
getPURestricted finds them, a chroma CU's co-located luma CU, the modes the syntax leaves) and, for a mode the GPU
derives, a node record before the CU's first block (`Tu::derive & DERIVE_NODE`, no samples): intra_rec runs the MPM
lists there (`intra_modes.glsl`: PU::getIntraMPMs without MPM sorting, DIMD, MDIP, PDP, NN) or chroma DM / the
candidates' substitution and publishes the mode (`Params::cuMode`); the CU's blocks depend on the node and set their
parameters from the mode (`applyModeParams`: initPredIntraParams, fusion, directional planar, the bilateral filter's
mode rule; GpuIntraRec.cpp has a copy for the host's read ranges); intra_deps reads the union of the reference ranges
of the modes the syntax leaves (`Cu::modes`: planar for mpm_idx 0, a chroma candidate or VDIA, else all 67). The host
still derives every mode itself (no supported tool needs the GPU's yet) and makes no CU records then (they cost 4 % of
the wall time); `ECM_GPU_MODES=1` (`gpuDerivesModes`, `Impl::deriveModes`) makes the GPU derive all those the MPM
lists give, on every stream, and verification compares each derived parameter with the CPU's.
M18: the transforms of a derived mode (`Tu::derive & DERIVE_TRANSFORM`: ECM's explicit MTS with MTSExt, LFNST, NSPT)
on the GPU: intra_rec's `selectTransform` picks the kernel pair (getTrTypes' set of the size class and the clamped wide
angle, IDTR near horizontal / vertical) and the LFNST / NSPT set, transposition and matrix
(getFinalIntraModeForTransform's mode: the wide angle of the TU, of the CU with ISP, in NSPT's numbering) from tables
the host builds once with the CPU's own lookups (the TR_TAB_* directory in matrices: matrix offsets per type and size,
g_aucIpmToTrSet / g_aucTrSet / g_aucTrIdxToTr / g_aiIdLut, g_lfnstLut / g_nsptLut, the LFNST / NSPT matrices' offsets per
size class, set, index and bucket from TrQuant::getNsptMatrix); the host gives what does not depend on the mode
(Tu::mtsSet, secIdx, secBucket, the kind, sizes and scan). intra_resi skips such blocks: intra_rec's inChainResidual
computes the residual before the prediction in chunks of rows whose first stage fits shared memory (without LFNST /
NSPT it dequantizes from the levels buffer: with TU_256 a 128 x 128 MTS block keeps all its coefficients) and stores it
device-coherent where intra_resi would, followed by a workgroup barrier with buffer semantics (with joint CbCr the Cr
block reads its residual like resiFrom); signPrediction computes the base border and templates itself (signPredData)
and always recomputes. Without ECM_GPU_MODES and DIMD no block has DERIVE_TRANSFORM.
M19: DIMD. With DIMD every MPM list starts with DIMD's modes, so the GPU derives all modes the syntax does not give
(`Impl::deriveModes` whenever the SPS has DIMD). A CU's node computes its DIMD first (`intra_dimd.glsl`): the
gradient histograms around the CU with the arguments of `IntraPrediction::dimdHistogramCalls` (the CPU's own
availability checks; `Cu::dimd`, `dimdNum` index the `DimdCall`s) - buildHistogram in parallel, buildHistogramAdaptive
in the CPU's sample order with a per-batch subgroup prefix sum that stops at the sample reaching maxAmp, and for CUs
of 64 or more its first template line by one invocation (a region state carried from sample to sample) over
gradients computed in parallel - then the modes, blending, weights and location dependence (`dimdSelect`, into
`Params::dimdOut`), which the MPM list (JVET-AL0125: DIMD's modes after planar) or a DIMD CU (`CU_DIMD`) takes. A DIMD
CU's block (`DERIVE_DIMD`) with blending predicts the main mode, planar and the blend modes one after the other, each
with its own parameters, filtered or unfiltered references (both kept) and JVET-AN0158 filter length, sums them per
location-dependence class into a global per-workgroup buffer (`BLEND_ACC`; another invocation may handle a sample in
the next prediction: coherent, with buffer barriers), then divides and blends location-dependently
(xLocationdepBlending; its double-precision clip ranges are reproduced exactly in integers, `dimdScaledWeight`) and
applies the bilateral filter to the blend. DIMD chroma (`CU_CHROMA_DIMD`) is a node too: one histogram of the luma
inside the co-located area and the Cb / Cr neighbours (`dimdChromaHistogramCalls`), then the co-located luma mode's
substitution. Nodes depend on the owners of their histograms' samples. Verification compares every node's DIMD with
the CPU's. A picture with DIMD runs the full intra_rec pipeline, one without DIMD a pipeline without its code (and
without M18's derived transforms where it has none; `IntraRecVariant`, section 10).
M20: OBIC (JVET-AH0076 with JVET-AK0056's weights). An OBIC CU (`CU_OBIC`, a DIMD CU with obicFlag) is a node that
first computes its own DIMD (whose main mode and first blend mode OBIC counts half), then the histogram of occurrence
of its neighbours' modes (`intra_obic.glsl`): the host picks the neighbours with deriveObicMode's steps 1-3
(`IntraPrediction::obicNeighbours`: adjacent and non-adjacent positions, intra, distinct, the nearest 20) and records
each one's CU and sample count (`ObicNbr`, `Cu::obic`; MIP CUs count nothing in intra slices and are left out); the
node reads a DIMD neighbour's DimdOut (planar and blend modes by their weights relative to the main mode's), an OBIC
neighbour's ObicOut, any other CU's mode, from those CUs' nodes (dependencies), then picks the six strongest modes
but planar with their dominant class, weights them (DIMD's division by table, 64-bit products, the excess / shortfall
loops) and adds planar (`Params::obicOut`). Its histogram reuses `s_dimdHist` (shared memory is full). The blocks
(`DERIVE_OBIC`) blend through DIMD's path, generalised to a list of (mode, weight, class) per block (`s_blendMode`,
`s_blendW`, `s_blendClass`; DIMD's filter length by weight only for DIMD); an OBIC CU always blends (no fusion, the
bilateral filter on the blend). With the reconstruction on, the CPU's OBIC can lack a mode (its neighbours' DIMD
comes from samples it has not reconstructed) and its later MPM lists would index with -1: DecCu then keeps a
placeholder (ECM_VULKAN, ON mode only).
M21: intra merge (JVET-AM0074). An intra merge CU (`CU_MERGE`, a DIMD CU with intraMergeMode; no DIMD of its own) takes
the parameters of a neighbour: its node (`intra_merge.glsl`) walks deriveIntraMergeCand's neighbours in the CPU's order
(the host lists them, `IntraPrediction::intraMergeNeighbours`, with how each may be a candidate: an intra merge or OBIC
CU always, a DIMD or MPM CU where its DIMD blends - which the GPU derives, so the list is whole and the GPU stops at
five) and keeps a valid entry at a new position with new parameters (an OBIC neighbour's converted as
setIntraMergeCandParam does). Two or more are ranked as reorderIntraMergeCand does, by template costs
(`intra_timd.glsl`, the base for TIMD): the template of 2 or 4 lines left of and above the CU, its reference line (8 x
the template's block, filled from the host's unit availability, `IntraPrediction::timdReferenceUnits`, padded in the
CPU's order), every mode of the candidates predicted on the template at 1/64 precision (initPredTimdIntraParams'
wide angles and tables, the cubic filter g_aiExtIntraCubicFilter, the PDPC kinds of predTimdIntraAng), the SATD of the
reconstruction against it with xGetHADs_SIMD's kernels (2x2, 4x4, 8x4, 4x8; the double `sad / sqrt(32) * 2` of 8x4 /
4x8 is exactly isqrt(sad^2 / 8), checked for every sad below 2^24), then each candidate's weighted cost (unsigned 64-bit
products, planar blend modes free, the first of equal costs); all modes at once where `s_tmp` holds their templates.
The choice goes into the CU's DimdOut slot in DIMD's layout, so its blocks (`DERIVE_MERGE`) blend like DIMD's (always,
regular filter lengths, the bilateral filter on the blend). The CPU derives DIMD for every MPM CU, also where the list
starts with planar (only intra merge reads it): the host keeps those CUs' DIMD histograms aside (`pendingDimd`) and
puts them to use when an intra merge CU lists the CU. OBIC counts an intra merge neighbour's modes without a class
(`s_obicPlain`); a mode only those gave has obicLocDep -1, whose weight the CPU adds to the first class's total but to
no class's blend (and its class-blend shortcuts let the last prediction take it): `blendTotals`. An OBIC CU counting
such a neighbour indexes `histoLocDep[mode][-1]`, which in the CPU's row-major array is `[mode - 1][2]`, and the GPU
counts it there (`obicCount`; no test stream has it: checked by a temporary test that forced class -1 on CPU and GPU
alike, 216 such counts identical, and failed with the GPU counting elsewhere). Verification also compares the template
costs (`IntraPrediction::m_mergeTmCost`, a copy reorderIntraMergeCand keeps). A picture with intra merge CUs runs a
fourth intra_rec pipeline (`WITH_MERGE`); in the one with DIMD, intra merge's code made pictures without it ~11 %
slower. Rejected: intra merge above 10 bits (the CPU's SATD adds in 16-bit lanes), with a block vector, CUs above 128
(the node's shared arrays).
M22: the transforms of DIMD, OBIC and intra merge CUs (explicit MTS, LFNST / NSPT, MTSS) through M18's derived
transforms. Explicit MTS takes the CU's mode (getTrTypes: pu.intraDir[0] - an OBIC CU's first OBIC mode); LFNST / NSPT
take a DIMD or OBIC CU's own DIMD main mode (getFinalIntraModeForTransform's DIMD branch; intra merge: its mode), so
`selectTransform` reads the CU's DimdOut for DERIVE_DIMD / DERIVE_OBIC blocks. MTSS (JVET-AK0217: mdirIdx, parsed for
a DIMD CU with LFNST of 128 samples or more; no SPS switch) chooses another LFNST / NSPT mode: DIMD's second strongest
mode (candModeListForTransform[1], `DimdOut::trSecond`) or an OBIC CU's second OBIC mode, where its LFNST set or
transposition differs, else from a mode list of the CU's own (deriveDimdModeList: deriveDimdMode's three plain
histograms, `IntraPrediction::dimdHistogramCalls` with modeList; the top three, ties to the lower mode) - the CPU's
loop over that list does not stop, so its last entry wins where any differs, else its first -, and NSPT's second
bucket where the chosen mode is within 2 of the first. The CU's node computes all of it once (`Cu::mtss`, the list's
histograms in `s_dimdHist` after DIMD's / OBIC's; `mtssList`, `mtssChoice`), its blocks (DERIVE_MTSS) read the mode
and bucket from DimdOut (`DIMD_OUT_MTSS_*`), which verification compares with the CPU's own
getFinalIntraModeForTransform, as the list and DIMD's second mode. Test sets: m22 (the stock encoder with the
transforms, MIP off: MIP CUs with LFNST fall back, "LFNST of MIP") and m22o (encTest2, OBIC preferred, no intra merge).
Rejected: DIMD CUs with several transform blocks (the CPU blends planar predicted from the CU's references at the
block's size; the intra configuration's CTU 128 and transform blocks up to 128 leave none), implicit MTS (supported()).
M23: LFNST / NSPT of MIP blocks. Their mode is a DIMD of the block's own MIP prediction (deriveIpmForTransform, run
for every MIP block since OBIC: buildHistogram's 3x3 gradients inside the block, every second sample both ways from
128 samples on, the strongest mode `mipDimdMode` and the second `dimdDerivedIntraDir2nd`), so a MIP block with LFNST
(DERIVE_MIP_DIMD) derives it from its reduced prediction (`mipTransformMode`, the samples by `mipSample` as the
upsampling gives them). Every block of derived transforms chooses them (`selectTransform`) and computes its residual
(`inChainResidual`) in one place, after the references and MIP's reduced prediction, before predicting: a second copy
of that code in the MIP branch made the pictures with DIMD ~7 % slower (code that is merely present; 1 to 3 % on most
sets remain against M22, a candidate for the speed work). With MTSS (in 4 to
15 % of them) the MIP CU gets a node although its mode is known (CU_KNOWN, the host's mode kept), which derives only
MTSS's mode list; the block chooses with its second mode and the list (mtssChoice) and depends on that node.
Verification compares mipDimdMode, the second mode, the list and choice (the CU's DimdOut) and the block's
transforms. Test set m23: the m22 arguments with MIP. MTS of MIP takes planar's kernels (getTrTypes), unchanged.
Rejected: LFNST of MIP without DIMD in the SPS (the CPU's mode is planar then; no test), of a MIP CU of several blocks
and of chroma MIP (4:4:4).
M24: TIMD (plain: TIMD-SAD and TIMD merge are rejected per CU). A TIMD CU (`CU_TIMD`, with the DIMD its MPM list starts
with) gets a node (`intra_timd.glsl`'s `timdNode`) that does deriveTimdMode: M21's template costs, generalised to
templates of one arm (`TemplateUnits`, also in intra merge's record: at the picture's left / top edge the CPU takes the
above / left arm alone, with its own DC) and to costs per arm (the location dependence compares them), for planar, DC,
the angular modes of the MPM list built with isForcedValid (all 22 entries, then HOR / VER where missing:
`mpmSpatial` / `mpmFill`, shared with deriveLumaMode) and the wide angles beyond the regular ones that the host's four
PU checks let the reference reach (getTimdRegularAngleExt / getTimdWideAngleExt: `timdRegularAngle`, `timdWaipAngle`;
the CPU's loop skips the regular range by jumping to maxModeOrig - 1, so the next one is maxModeOrig + 4). All of them
are costed at once, then invocation 0 replays the CPU's sequential scan with its early ends (computing more costs
than the CPU changes no choice); a second batch costs the neighbours of the best and the secondary mode, replayed
with the CPU's quirks (the secondary's refinement on the best cost's condition, a refined best that does not demote the
old one). The final choice maps wide angles back, swaps, lets an angular best blend with itself where the secondary
costs twice as much (JVET-AG0092), and weighs three modes with planar / DC as the CPU's doubles do (exactly in
integers: floor((64 (sum - c) + sum) / (2 sum)) for costs far below 2^40) or two by X0149's table. Its TimdOut holds
every CodingUnit field deriveTimdMode sets and the initial values of those it leaves (location dependences of a
non-blended CU are the scan's leftovers), compared with the CPU's in verification; the published mode is MAP131TO67's
(what neighbours' MPM lists and chroma DM read). The blocks (`DERIVE_TIMD`) predict with 1/64-sample angles (TIMD's
tables, its wide angles where checkWA is off), the 1/64 N-tap filters from the filter buffer (`EXT_NTAP`), planar's
reference filter only, no reference filter or Gaussian interpolation for angular modes (JVET-AM0104 / AM0163); a blend
of the mode, the secondary and planar / DC goes through DIMD's per-class path with xGetNumTapsTIMD's filter lengths,
its location-dependent range 8 from 128 samples on, the secondary predicted from the next reference line alone
(no PDPC but of horizontal / vertical) where JVET-AG0128's rule fuses the two, which also overrides the CU's
fetchRef2nd (the next line is read in a CTU's first row too), and the bilateral filter on the blend only. Such a block
keeps its unfiltered line in `s_ref[0]` and the next line in `s_ref[1]`; its planar reads the filtered line from
`s_tmp`. The read ranges of TIMD blocks are both lines' whole arms. Transforms: explicit MTS and LFNST / NSPT by the
published mode, MTSS with the secondary (MAP131TO67) where the CU blends. The pictures with TIMD run a fifth pipeline
(`WITH_TIMD`). Rejected: TIMD-SAD, TIMD merge, TIMD with ISP or MRL (no test stream: the stock encoder's TIMD-MRL
candidates hardly ever survive its SATD checks, none in the m24 set), TIMD CUs of several blocks, TIMD above 10 bits,
IntraTMP / IBC (the only sources of TIMD's block vector candidates). Test sets: m24, encTest4 (ECM_TEST_NO_OBIC,
NO_INTRA_MERGE, NO_TIMD_SAD) with the m23 arguments and `--EnableTMnoninterTools=1 --TIMD=1 --TIMDMerge=0` (without
EnableTMnoninterTools the encoder turns TIMD off); m24s, the same on screen_352x288, whose flat templates give angular
TIMD CUs of template cost 0, which do not blend (an angular best mode otherwise always does, with itself where the
secondary costs twice as much). GPU time of the m24 set: 47 to 68 ms per CIF picture, 133 (park720) and 195 (four720)
per 720p picture; without the TIMD nodes' template costs (an experiment) ~15 ms less per CIF picture, the first
candidate for the speed work. The other sets are unchanged within 1 %.
M25: OBIC and intra merge CUs with TIMD neighbours, both reading the neighbour's node (its TimdOut). deriveObicMode
counts a TIMD neighbour's mode (MAP131TO67) with all its samples in the mode's location dependence class and, where it
blends with a secondary other than itself, the secondary by its weight relative to the mode's, and with it the
non-angular mode by its own (`obicHistogram`). An intra merge candidate from a TIMD CU (`MERGE_NBR_TIMD`; valid where
its modes blend or its mode is angular, CU::isAValidNeighborForIntraMerge) converts its TIMD as
setIntraMergeCandParamFromTimd does (`mergeTimdParams`): the mode, the secondary as blend mode (where both map to one
mode of the 67, the mode two of their distances beyond), the weights reordered to [mode, non-angular as planar,
secondary], or without a secondary planar at 16; where the conversion gives up (a mode or blend mode planar / DC) the
candidate takes the CU's own DIMD (the host rejects a TIMD neighbour without one). Test sets: m25 (encTest4 with
ECM_TEST_NO_TIMD_SAD alone, the m24 arguments: intra merge CUs with TIMD neighbours) and m25o (encTest7, see section 9,
with ECM_TEST_PREFER_OBIC, NO_INTRA_MERGE, NO_TIMD_SAD, FORCE_OBIC, TIMD_SPARSE and TIMD_CHECKER=3). An OBIC CU never
has a TIMD CU among the neighbours that intra merge checks (left column, top row, corners and two CUs on from the
below-left and above-right ones: such a neighbour allows intra merge, which excludes OBIC), only among its farther
ones, and the stock encoder also drops OBIC wherever TIMD's SATD is 1.3 times lower; so the m25o encoder allows TIMD
only in sparse 8x8 islands and DIMD only there, and favours OBIC wherever it is available. m25s and m25os are the
two on screen_352x288, for TIMD neighbours that do not blend (as m24s). GPU time: m25 48 to 68 ms per CIF picture,
168 (park720) and 212 (four720) per 720p picture; m25o 30 to 55, 93 and 165; the m24 set ~1 % slower than with M24
(the new branches are code in the TIMD pipeline also where no CU takes them), the other sets unchanged.

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
   M9 transform skip and BDPCM, M10 MIP, M11 LFNST / NSPT (computed by the host until M15), M12 MRL, M13 ISP, M14 sign
   prediction, M15 LFNST / NSPT on the GPU (the VVC kernels, LFNSTExt=0, rejected), M16 sign prediction of LFNST /
   NSPT blocks. 2026-10-12: M17 intra modes derived on the GPU (tested with `ECM_GPU_MODES=1`), M18 the transforms of
   derived modes on the GPU, M19 DIMD (the m19d test set), M20 OBIC (the m20 and m20o test sets), M21 intra merge
   (the m19 set, encoded by the stock encoder with DIMD: OBIC and intra merge everywhere), M22 the transforms of DIMD
   CUs and MTSS (the m22 and m22o sets), M23 LFNST / NSPT of MIP (the m23 set), M24 TIMD (the m24 and m24s sets),
   M25 OBIC and intra merge with TIMD neighbours (the m25, m25o, m25s and m25os sets).
   All 19 regular streams now stop at "TMRL" (the next sequence-level check; the block-level tools of step 4 follow).
3. Make the GPU path win on CPU time per picture: the derive-only pass (record during parsing or a leaner pass;
   speed up getSpatialIpm/getPURestricted, which also helps the CPU decoder), GPU behaviour with frame threads
   (8 contexts x 32 persistent workgroups oversubscribe the GPU; spinning workgroups; try fewer groups per context,
   more frame threads than cores to hide GPU waits, overlapping a picture's GPU work with the next picture's parse).
4. More tool levels. CTC turns on DIMD, TIMD, TMRL, MPM sorting, MDIP, PDP and chroma reordering together, and every
   one of them makes the intra modes depend on reconstructed samples (DIMD's modes are in every MPM list), so with
   CTC the GPU derives every mode (M17's nodes) and everything that depends on a mode moves to the GPU too (M18 did
   the transforms, M19 DIMD, M20 OBIC, M21 intra merge). ECM's encoder uses OBIC and intra merge whenever DIMD is on
   and has no switch for them; for tests of fewer tools, test encoders built with a local patch to IntraSearch.cpp
   that is not committed (`perf/gpu/enc_m1.ps1` with ECM_ENC_EXE):
   `bin\encNoOM` (m19d: ECM_TEST_NO_OBIC_MERGE skips both), `bin\encTest` (m20: ECM_TEST_NO_INTRA_MERGE;
   ECM_TEST_NO_OBIC skips OBIC), `bin\encTest2` (the same, and for m20o ECM_TEST_PREFER_OBIC, which skips plain DIMD
   where OBIC is available, for many more OBIC CUs) - the patch adds `&& !getenv(...)` to `testObic` /
   `testIntraMerge` and sets `m_skipDimdMode` / `m_skipDimdLfnstMtsPass` where OBIC is tested; `bin\encTest4` (m24)
   adds ECM_TEST_NO_TIMD_SAD (to `testTimdSad`, in estIntraPredLumaQT and reduceHadCandList); `bin\encTest7` (m25o)
   adds ECM_TEST_TIMD_CHECKER=<log2 cell size> (`testTimd`, `m_skipDimdMode` and `m_skipDimdLfnstMtsPass` off in the
   cells of the CU's top left where the cell indices' sum is odd; with ECM_TEST_TIMD_SPARSE in all cells but those
   whose indices are both multiples of 4) and ECM_TEST_FORCE_OBIC (an OBIC mode's RD cost in estIntraPredLumaQT divided
   by 4; xCheckRDCostIntra recomputes the CU's cost from its bits and distortion, so the partitioning sees the true
   cost). Next TIMD-SAD, TIMD merge, MPM
   sorting / MDIP / PDP, TMRL,
   CCLM/CCCM and the other cross-component tools, chroma reordering / DBV, IntraTMP / IBC, implicit MTS (not in CTC),
   then inter pictures.
   Intra merge's GPU time (m19: park720 152, four720 183 ms per picture, from 347 / 392 at first) has room: the
   candidate walk and the reference padding are single-invocation steps on the dependency chain. Each level: new
   test set with exactly that tool on, verify mode identical, ON mode identical (and `ECM_GPU_DUMP` +
   `perf/gpu/dumpdiff.py`: the records of both runs agree), canaries, commit. Speed of derived modes: with
   `ECM_GPU_MODES=1` the GPU time of four 720p pictures sets is 25.3 instead of 14.0 ms per picture (the union of the
   read ranges lengthens the dependency chains, the residuals of derived transforms are computed in the chain); exact
   ranges would need dependencies a node registers once it knows the mode (a dependents list per block, closed when
   the block finishes).
5. In-loop filters in ON mode on the GPU without a CPU round trip (deblocking: see dec-gpu-dbf-wip; ALF/SAO/... are
   on the GPU already via --GpuLoopFilter, but on the CPU's picture).
6. 1080p/4K verification and timing on long streams once sources are agreed with the user (three short 1080p
   streams verify identical since M12); check on other GPU makers (AMD, Intel) before any release.

## 10. Findings worth knowing

- RTX 3070: plain loads through `coherent`/`volatile` buffer references returned stale L1 data across workgroups;
  use `#pragma use_vulkan_memory_model` + GL_KHR_memory_scope_semantics. Persistent workgroups + an atomic ready
  queue are deadlock-free dependency scheduling.
- RTX 3070, also with the memory model: data passed between the invocations of one workgroup through a global buffer
  and a barrier was read stale by whole warps while another process loaded the GPU (intra_resi, fixed by shared
  memory). Pass data inside a workgroup through shared memory, and stress-test with `gpuload.py` running. Where it
  cannot fit (M18's in-chain residuals of up to 128 x 128 samples), store and load through a `coherent` buffer
  reference with `controlBarrier( gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsBuffer,
  gl_SemanticsAcquireRelease )` between: without `coherent` the memory model's accesses are private and no barrier
  makes them visible (likely intra_resi's old failure); 120 stress decodes of such blocks under load were identical.
  Code a kernel never runs still costs: M18's in-chain paths made intra_rec ~4 % slower on pictures that never take
  them (a per-sample test of a global flag in storeRecon cost as much again; folded into Tu::resiFrom), M19's DIMD
  blending ~15 % (spread over the loop around the prediction and the code inside it; no single part), M21's intra
  merge ~11 % on pictures with DIMD, a second call site of selectTransform and inChainResidual ~7 % (M23: one call
  site each instead). Since M19 intra_rec is built as several pipelines with specialization constants
  (`WITH_DIMD`, `WITH_DERIVED_TRANSFORMS`, `WITH_MERGE`; `gpu::Kernel::create`'s specialization, GpuIntraRec.cpp's
  `IntraRecVariant`) and a picture runs the one without the features it does not use. The compiler is not monotonic
  here: the pipeline with DIMD alone was slower than the one with DIMD and the derived transforms in every case
  measured, so that one is not built. Measure each new variant.
- `supported()` / `record()` must reject every tool the kernels lack, including tools that only change a CPU code
  path the GPU replaces (dequantization, transforms, clipping): dependent quantization was not rejected until M6,
  and the reconstruction on decoded such streams wrong. With a new level, look through the CPU path it replaces for
  flags the GPU ignores, and encode a test set with each such flag on.
- The kernels' reference availability is "decoded earlier" (owner maps); the CPU's (`CodingStructure::
  getCURestricted`) also excludes other slices and tiles, with wavefronts the CTUs right of the current one in the
  row above, and with GDR the dirty area. Tiles, wavefronts and GDR were not rejected until 2026-10-11: a stream with
  tiles or WPP decoded wrong with the reconstruction on (test streams `fx_tiles_park720`, `fx_wpp_park720`, now
  skipped). Any new availability rule of the CPU must be mirrored or rejected.
- Vulkan objects are never created or destroyed while a stream records or has a submission in flight
  (`gpu::Context::inFlight()`), and the device is destroyed after decoding (`DecLib::releaseGpu` /
  `releaseGpuDevice`, called by DecApp): with the frame threads' contexts created in the background, NVIDIA's driver
  (616.92) now and then never signalled the fence of a completed submission, and a device left to the end of the
  process now and then hung its termination (2026-10-11, ~1 in 200 decodes of m13c_25 with 8 frame threads;
  DECODER.md). Once a fence signalled before its submission's commands had run (an all-zero picture): every
  submission now ends by writing its number into a host-visible word that `Stream::wait` checks (a late end is waited
  for, a missing one runs the commands of a repeatable stream again, else throws). Keep new GPU code to these rules,
  and test frame-thread changes with many repeated short decodes under load whose outputs are compared, not only once.
- DepQuant::getPredictedSigns (sign prediction) changes the TU's levels in place: the parsed signs of the sign area go
  to the positions sorted by level. Host code that calls it (GpuIntraRec::prepareSignPred) must give the TU its levels
  back, as the CPU calls it again when it verifies or reconstructs a picture the GPU path rejected. With the
  reconstruction on, the TUs keep the undecided signs (in the decoder only parsing, dequantization and the inverse
  transforms read coefficients; the loop filters read none).
- The scheduling allows a dependency on a later block: with joint CbCr whose Cr is coded, the Cr block decides the
  signs and writes the Cb block's residual (Tu::resiFrom). Statistics over the graph must not assume earlier blocks.
  Any new edge between the blocks of one CU can close a cycle with it: a first M17 version let a CU's later blocks wait
  for its first one (which derived the mode), and the Cb block (waiting for its Cr block) and the Cr block (waiting
  for the Cb block) hung the GPU (VK_ERROR_DEVICE_LOST). Edges to the earlier nodes only are cycle-free.
- `gpucheck.py --GpuRecon=2` compares the output, which in verification is the CPU's: it also requires the summary's
  "0 with differences" since 2026-10-12 (before, only the ON modes caught a wrong GPU reconstruction).
- The owner maps have 4x4 luma units; ISP sub-partitions 1 or 2 wide or high share them. The last block owning a
  unit must win everywhere (intra_deps pass 0 writes with atomicMax: plain parallel writes raced), and a block
  reading a shared unit only waits for that last owner, so the sub-partitions of a CU are chained (Tu::ispPrev;
  without the chain every m13 picture differed).
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
- With the reconstruction on, the CPU's derive-only pass derives DIMD (and every later sample-based mode) from samples
  it has not reconstructed, so its modes are garbage there and right in verification. A field the host takes from the
  CPU's mode of a block whose mode the GPU derives is therefore wrong only with the reconstruction on, which
  verification cannot see: M19's first ON run differed almost everywhere because the bilateral filter's LMCS flag came
  from the CPU's mode. Check every new level with `ECM_GPU_DUMP=<prefix>` on a verification and an ON decode and
  `perf/gpu/dumpdiff.py <prefix> <POCs>` (the records agree apart from the fields intra_rec derives). The CPU's
  parameter snapshot after DIMD / OBIC blending is the last blended mode's, not the main mode's (record() restates
  them). Garbage modes can also be out of range for the CPU's own later code: OBIC without a mode left -1, which
  getSpatialIpm writes an array with (DecCu keeps a placeholder in ON mode).
- Shared memory of intra_rec is nearly full (by the declarations, about 48 500 of the 49 152 bytes Vulkan gives a
  workgroup on the RTX 3070): OBIC's histogram reuses DIMD's, intra merge's candidates and costs live in s_spCost
  and s_tmp (free in a node).
- Verification reports the first difference by kind of check (scheduling, intra merge costs, DIMD / merge
  parameters, OBIC, block parameters, samples), not in decoding order: a node's output can differ only because an
  earlier block's samples did (M21's first mismatch was an MPM CU's DIMD, caused by an earlier OBIC block). Find the
  earliest differing block before chasing the reported one.
- A rare case no encoder setting reaches (OBIC's class -1 counted by another OBIC CU: ECM's encoder picks OBIC a few
  times per picture where intra merge is allowed) can be tested by forcing the rare value on the CPU and the GPU alike
  with temporary edits on streams rich in the surrounding tool: a counter on the CPU shows the case runs, verification
  must stay identical, and a canary on the GPU side must be reported.
- A double-precision expression of the CPU (xLocationdepBlending's clip ranges) can be reproduced exactly in integers
  when its inputs are small: `dimdScaledWeight` (checked exhaustively against IEEE doubles for inputs below 200).
