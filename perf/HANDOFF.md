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
reconstruction on the GPU (`--GpuRecon`), tool levels M1-M45 done and verified bit-exact (M19: DIMD, M20: OBIC, M21:
intra merge, M22: the transforms of DIMD CUs, M23: LFNST of MIP, M24: TIMD, M25: OBIC and intra merge with TIMD
neighbours, M26: TIMD-SAD, M27: OBIC and intra merge with TIMD-SAD neighbours, M28: TIMD merge, M29: TMRL of luma,
M30: chroma TMRL, M31: CCLM, M32: LFNST / NSPT of CCLM, M33: GLM's gradient patterns, M34: CCCM, M35: GL-CCCM, M36:
CCCM with several downsampling filters, M37: CCCM without subsampling, M38: CCCM with LFNST / NSPT, M39: GLM with
luma, M40: LM chroma fusion of CCCM, M41: CFLM, M42: GLM with luma with LFNST / NSPT, M43: the SPS switches of the
CCP tools, M44 / M45: the CCP models); it is **not yet faster than the CPU decoder** (see section 8) and supports
only a restricted tool set (all real CTC streams fall back to the CPU, first reason: chroma reordering / DBV; CCP
merge, decoder-derived CCP fusion and BVG-CCCM CUs are rejected per block).

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
| `gpucheck.py [-j N] <build> [streams] -- <decoder args>` | decodes every GPU stream with the build and with the stock decoder (cached), compares all frames, prints the `[gpu-intra]` lines. Use `--GpuRecon=2` (verify) and `--GpuRecon=3`, also with `--FrameThreads=8`. `-j N` decodes N streams at a time (6 for verification, 3 with 8 frame threads; 1, the default, for timings): a full regression then takes minutes instead of most of an hour. |
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
(`WITH_TIMD`). Rejected: TIMD merge (until M28), TIMD with ISP or MRL (no test stream: the stock encoder's TIMD-MRL
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
M26: TIMD-SAD (JVET-AJ0146). A TIMD-SAD CU's node does deriveTimdMode as before, then deriveTimdModeSad, whose result
(the *Sad fields: the second half of the CU's TimdOut, from TIMD_OUT_SAD) its blocks predict. Its first batch is
planar, DC and a list of up to 25 modes (`timdSadCandidates`): the first 15 angular modes deriveTimdMode costed in its
MPM list's scan, up to that scan's early end, by cost then mode (m_timdModeCostList, stable-sorted), then the CU's DIMD
modes (all 8 entries), HOR, VER and the modes around TIMD's two and DIMD's, each once and none of the modes TIMD
found; then the wide angles beyond the regular ones as TIMD's. The scan, the second batch, the refinement and the
final choice are TIMD's (`timdScanFirst`, `timdSecondBatch`, `timdRefine`, `timdFinal`, shared), with as costs the
mean-removed SAD per arm (RdCost::xGetMRSAD: the arm's mean difference truncated toward 0, then the sum of the
differences' distances from it; its SIMD forms, in saturating 16-bit lanes, agree up to 11 bits, the GPU's TIMD stops
at 10), the scan passing over a list mode whose above arm alone costs the secondary cost or more (JVET-AN0093), and
no location dependence. Its template (TimdHead::ts) has TIMD's arms doubled, at most the CU's distance from the
picture's edge (2 at least, which also scales the costs where the arm is missing), 8 only where the samples 8 out
beside both ends of the CU's side are decoded CUs of its slice and tile (else 4); its reference units follow TIMD's
in the record (intra slices keep TIMD-SAD CUs at 1024 samples, which keeps them within intra_timd.glsl's
`s_timdUnits`; the host checks: TIMD_UNITS_MAX). The blocks predict as TIMD's from the *Sad fields
with the default filter lengths (xPredIntraAng takes numTapsTIMD only without TIMD-SAD). Rejected: TIMD-SAD without
DIMD in the SPS (its list takes the CU's DIMD modes; no test stream). Test sets: m26 (encTest4 with ECM_TEST_NO_OBIC
and NO_INTRA_MERGE, TIMD-SAD allowed, the m24 arguments: a few to 30 TIMD-SAD CUs per CIF picture), m26o (encTest8 with
ECM_TEST_PREFER_TIMD_SAD as well, plain TIMD untested where TIMD-SAD is: 30 to 70 per CIF picture). On
screen_352x288 the encoder chose no TIMD-SAD CU, even with the switch. TIMD-SAD's code runs in a pipeline of its own
(VARIANT_TIMD_SAD, WITH_TIMD_SAD): in the TIMD pipeline it slowed the TIMD pictures without TIMD-SAD by ~5 %, and
its MRSAD sums take s_dimdHist (shared memory of their own slowed every pipeline by ~3 %). GPU time: m26 96, m26o
73 ms per picture; m24 and m25 1 to 1.5 % slower than with M25, the other sets unchanged.
M27: OBIC and intra merge CUs with TIMD-SAD neighbours. OBIC's histogram (`obicHistogram`) and intra merge's
conversion (`mergeTimdParams`) read a TIMD neighbour's modes from the fields its blocks predict (`timdPredBase`: a
TIMD-SAD CU's *Sad fields); intra merge's validity of a TIMD-SAD neighbour (`mergeTimdValid`): its TIMD-SAD blends or
its TIMD-SAD mode is angular, or its plain TIMD's mode is (CU::isAValidNeighborForIntraMerge tests timdMode before
timdModeSad). Test sets: m27 (encTest8 with ECM_TEST_NO_OBIC and PREFER_TIMD_SAD, the m24 arguments: intra merge CUs
with TIMD-SAD neighbours), m27o (encTest8 with m25o's switches and PREFER_TIMD_SAD: OBIC CUs with TIMD-SAD
neighbours). GPU time: m27 92, m27o 75 ms per picture.
M28: TIMD merge (JVET-AJ0061, one merge candidate: NUM_TIMD_MERGE_MODES). A TIMD merge CU (`CU_TIMD` with
TimdHead::merge; no DIMD, no MPM list) gets a node (`timdMergeNode`) that does deriveTimdMergeModes. The host finds
searchTimdMrgCus' neighbours (up to five TIMD CUs, the nearest, in its order; IntraPrediction::searchTimdMrgCus is now
static) and puts their records in the TimdHead (their nodes are the merge node's dependencies). The node takes each
neighbour's fields as its blocks predict them (a TIMD merge neighbour's candidate, a TIMD-SAD one's TIMD-SAD) unless
a candidate has its mode and secondary already (genTimdMrgList), costs two or more by the SATD of their mode on the
CU's template of arms one sample thick (calcTimdMrgCandCosts calls CU::deriveTimdRefType with 1, not TIMD's 2 / 4),
with the SIMD kernels of such arms (JVET-AJ0096: 1x16, which leaves out its last butterfly stage and takes the even
samples' DC term; 1x8 scaled by 2 / sqrt(8); 1xN for 4), and takes the cheapest, or the second cheapest where it costs
less than 1.1 times as much (the CPU's comparison in doubles exactly in integers: `timdMergeSwaps`). Its TimdOut holds
the candidate's fields with every checkWA set (timdMrgModeCheckWA); its blocks predict as a TIMD CU's but for the
fusion of a blend (initPredIntraParams' TIMD merge branch, DERIVE_TIMD_MERGE): the next line always fetched, the
secondary predicted from it alone where its slope is fractional. Transforms: no MTS or transform skip; without LFNST in
dual tree (TrQuant::xIT's TIMD merge branch) a side of 4 to 16 takes DST-VII, one above IntraMTSMaxSize DCT-II, any
other the candidate's type (timdmTrType: DERIVE_INHERIT_HOR / VER, with DERIVE_TRANSFORM), for which every TIMD CU's
node stores its own getTrTypes types in its TimdOut (TIMD_OUT_TR_*: explicit MTS's by its mode, else the host's), in
the pipeline with TIMD merge only. Intra merge drops a TIMD merge neighbour whose TIMD does not convert (it has no
DIMD to fall back on: MERGE_NBR_TIMD_MERGE); OBIC reads it as a TIMD neighbour. Verification compares the candidate's
fields and every TIMD CU's transform types. The pictures with TIMD merge run a pipeline of their own
(VARIANT_TIMD_MERGE, WITH_TIMD_MERGE; with TIMD-SAD). Test sets: m28 (encTest4 with ECM_TEST_NO_OBIC and
NO_INTRA_MERGE, TIMD-SAD allowed, the m24 arguments with `--TIMDMerge=1`: 36 to 164 TIMD merge CUs per CIF picture,
573 per four720 picture, of whose blocks 62 inherit transform types), m28m (encTest4 without switches: ~930 intra
merge CUs per four720 picture next to 287 TIMD merge CUs). GPU time: m28 96, m28m 110 ms per picture; the other sets
unchanged within 0.4 % (the kernels of the one-sample template exist in the TIMD merge pipeline only).
M29: TMRL of luma (JVET-AB0157 with JVET-AD0085's 131-mode set, the DIMD modes first in its list: JVET-AL0125). A
TMRL CU (`CU_TMRL`, with its DIMD and MPM neighbours; tmrlListIdx in Cu::ipredIdx) gets a node (`tmrlNode`,
intra_tmrl.glsl) that does getTmrlList: the ten modes of getTmrlSearchRange (`tmrlModeList`: the CU's DIMD modes, the
neighbours' modes in the 131-mode set - a TIMD neighbour's from its TimdOut, a TMRL one's from its TmrlOut, any other's
MAP67TO131 - planar never, then fillMPMList's angles around them and its defaults), on each reference line 1, 3, 5, 7,
12 inside the picture (L < y), each line filled as xFillReferenceSamples fills it for the CU (its availability scanned
by the node: `refLineAvailability`, shared with intra_deps), predicted on the template one sample thick (the above arm,
the left one where x > 0) as xPredTmrlIntraAng / xPredTmrlIntraDc do, costed by the SATD of TIMD merge's one-sample
kernels; it takes the entry of rank tmrlListIdx in the costs' stable order (updateCandList keeps the earlier of equal
costs first; the decoder's early skip of a candidate whose above cost alone exceeds the list's last changes nothing) and
writes mode and line to its TmrlOut (Params::tmrlOut). Its block (DERIVE_TMRL) predicts that mode (1/64-sample angles,
no PDPC, no reference filter) from that line, fused with the next line where the CU's fetchRef2nd (size and the SPS's
switch only: a TMRL line is never 0, and 12 only below row 12) and the slope allow it, the bilateral filter for DC
only; as intra_deps cannot know the line, the block depends on the whole arms of every line its node may choose and of
the next one, and finds its line's availability itself. Rejected: chroma TMRL (JVET-AJ0081, next), TMRL CUs of several
blocks, TMRL above 10 bits (the SATD's 16-bit lanes, as TIMD). The pictures with TMRL run a pipeline of their own
(VARIANT_TMRL, WITH_TMRL; with TIMD merge). Test sets: m29 (encTest9, see section 9, with ECM_TEST_NO_CHROMA_TMRL; the
m24 arguments with `--TIMDMerge=1 --TMRL=1`: 74 to 358 TMRL CUs per CIF picture, 462 per park720 and 389 per four720
picture) and m29n_fore32 (the same with `--DIMD=0 --TIMD=0 --TIMDMerge=0 --MIP=0`: a mode list without DIMD modes;
without DIMD the GPU rejects LFNST of MIP) and m29s_screen32 / 22 (screen_352x288: 135 to 181 TMRL CUs per picture).
GPU time: m29 130, m29n_fore32 34 ms per picture (the TMRL nodes' up to
50 template predictions and their line fills are the next speed candidate); the other sets unchanged within 1 %.
M30: chroma TMRL (JVET-AJ0081, dual tree; the only chroma MRL). A chroma TMRL CU (`CU_CHROMA_TMRL`; chromaTmrlIdx in
Cu::ipredIdx, its TmrlOut shared with luma TMRL's) gets its DIMD chroma like a DIMD chroma CU (deriveDimdChromaMode and
DecCu's exchange with the co-located luma mode) and a node (`chromaTmrlNode`) that does getChromaTmrlList: eight modes of
the 67-mode set (`chromaTmrlModeList`: the co-located luma mode, the DIMD chroma modes, DC, the luma CUs' at the centre
and corners of the co-located area, the final modes of the chroma neighbours at five positions where they are intra and
neither LM nor DBV - the host records these ten CUs in Cu::nbr - then fillMPMList's, `mpmFill` with a list size), on
each chroma line 1, 3, 5 inside the picture (PU::hasChromaTmrl allows it from chroma row 4 on), costed as 9 x the SATD
of the co-located luma block predicted from the luma line twice as far plus the SAD of the Cb and Cr templates, one
sample thick, the above arms weighed by 4 x the CU's height, the left ones (where x > 0) by 4 x its width, in 64 bits.
The luma prediction is predCoLuma's: a plain luma PU (no PDPC, reference filter, fusion or bilateral filter; the cubic
filter, 10 taps along rows shorter than 8) from xFillReferenceSamplesForCoLuma's line (`coLumaReference`, a literal port
on one invocation: its availability is by position - inside the picture, above-right and below-left only up to the
CTU's edges - with quirks in its unit flags, count and padding), the reference arrays of `angularArrays` (main()'s
line-0 setup, factored out for it). Its SATD (`coLumaSatd`) takes xGetHADs_SIMD's AVX2 tiles (16x8, 8x16, 8x4, 4x8, 8x8
- the 16x16 kernel sums four - and 4x4), band by band of 16 rows through s_tmp, the 16x8 / 8x16 scale floor(sad /
sqrt(32)) exactly in integers (`hadScaleSqrt32`, checked for every sum below 2^25). The templates' chroma lines have
arms of 2 x (the side + 1), whose last unit may be partial (fillSource rounds the unit counts up now, as the CPU), and
xPredTmrlIntraAng steps through the regular 1/32-sample angles as 1/64 samples (`chromaTmrlParams`). The Cb and Cr
blocks predict the chosen mode as regular chroma blocks with that line (DERIVE_TMRL with applyModeParams; no fusion),
their availability found at run time as luma TMRL's; intra_deps gives them the whole arms of every candidate line, the
node the co-located luma block, its lines up to the CTU's edges, the chroma lines, templates and the ten CUs' nodes.
Rejected: chroma TMRL without DIMD in the SPS (no test stream), above 10 bits or with unequal luma and chroma bit
depths, in a picture smaller than the SPS's (the positional availability takes the SPS's size), CUs of several blocks.
Test set: m30 (encTest9 without ECM_TEST_NO_CHROMA_TMRL, m29's arguments: up to 84 chroma TMRL CUs per CIF picture,
134 to 218 per 720p picture, among them 16x2 ones whose co-located luma block is 4 high). GPU time: m30 135 ms per
picture; m29 131 -> 135 (the chroma node's code is in the TMRL pipeline also where no CU takes it: a pipeline of its own
would win that back), the other sets unchanged within 0.5 %.
M31: CCLM (LMChroma: LM_CHROMA, MDLM_L / MDLM_T and, with MMLM, their two-model variants; ECM's least-squares model,
LMS_LINEAR_MODEL). A chroma block in an LM mode carries Tu::lm's LM_ACTIVE: its mode number does not tell, as a luma
TIMD / TMRL block derives modes of the 131-mode set, whose values from 67 on are angular (a first version that took
them for LM read luma at twice their position, past the end of the plane, and lost the device). Its model is fitted on
one invocation (`cclmModel`, intra_cclm.glsl): xGetLMParametersLMS's template samples (availability in chroma units of
2; MDLM's 2W / 2H arms where the above-right / below-left units are available) of the luma downsampled as
xGetLumaRecPixels does (JVET-AN0090's filter by the SPS's collocated flags: the mean of 2 x 2, the 5-tap cross, or
1 2 1 over two rows; in a CTU's first row from the one luma row above), in 64-bit sums (JVET-AJ0237), the division
table m_auShiftLM computed in the shader; MDLM pads its template to a power of two, MMLM splits at the mean
(divSigTable) and pads each group (xLMSampleClassifiedTraining), and each model's slope takes the CU's cclmOffsets
(JVET-Z0050, xUpdateCclmModel around the middle luma of that model's samples). All invocations then apply it to the
downsampled co-located block, with LBCCP's 3x3 filter (JVET-AD0120, ccInsideFilter: the reconstruction above / left
where available, else the prediction's border) where the CU takes it. intra_deps gives an LM block the read range
2W / 2H and the owners of the co-located luma block and of the 3 luma rows above it and 3 columns left of it, each
twice the co-located block's side long (MDLM's template). The host gives LM blocks no angular, PDPC, reference filter or fusion parameters
(initPredIntraParams leaves the angles as the previous block's; ECM_GPU_DUMP's records differed there, and dumpdiff.py
now also leaves a TMRL block's line, which its node chooses, out). Rejected: GLM, CCCM, LM chroma fusion (JVET-AC0119),
CCLM with LFNST / NSPT (its mode is a DIMD of the CU's Cb and Cr predictions: next), CCLM CUs of several blocks
(DecCu predicts Cr over the CU's area with the Cb block, JVET-AK0064), above 10 bits or with unequal bit depths; CCP
merge, decoder-derived CCP fusion and BVG-CCCM at sequence level, as CCCM. Test sets: m31n (encTest10, see section 9,
with ECM_TEST_NO_GLM and ECM_TEST_NO_CHROMA_FUSION; m29's arguments with `--LMChroma=1 --IntraLFNSTISlice=0 --NSPT=0
--LFNSTExt=0`: 170 and 184 CCLM blocks per fore22 picture, 1526 and 1680 per park720 picture, all variants: MMLM,
MDLM, LBCCP; the 1 2 1 filter), m31a_fore32 / 22 (`--HorCollocatedChroma=0`: the 2 x 2 mean), m31c_fore32 / 22
(`--VerCollocatedChroma=1`: the cross) and m31 (with LFNST: M32). The CCLM code is in the
pipelines with WITH_CCLM only (a specialization constant beside each IntraRecVariant; `intraRecKernel` builds the
pair on first use): in all pipelines it slowed the plain set by 30 % and the TIMD ones by 5 %. GPU time: m31n 134.1,
m31a / m31c 84.6 ms per picture; the other sets unchanged within 0.8 % (plain 12.28 -> 12.38).
M32: LFNST / NSPT of CCLM blocks (JVET-AK0064). Their mode is deriveChromaIpmForTransform's, which DecCu calls with the
Cb block: the strongest mode of buildHistogram's 3x3 gradients (no subsampling) inside the CU's Cb prediction, its Cr
prediction or both, by the TU's joint CbCr mode (1: Cb, 2: Cr, 0 and 3: both; Tu::lm's LM_HIST_*). A block that needs
it (DERIVE_CCLM_DIMD with DERIVE_TRANSFORM; with joint CbCr whose Cr is coded also the Cr block, which decides the
signs with the Cb block's transforms) finds it before its transforms (`cclmTransformMode`): the other component's
references into s_ref[1] (its availability is the block's own: both chroma components' owner maps, decoded before
it, and none of them reads the CU), its model into a second slot (its slope adjustments carried in Tu::lm), both
predictions into s_tmp up to 32 x 32 (larger blocks, rare, compute each gradient's samples again), then selectTransform
with that mode (s_predDimdMode, MIP's too). intra_deps (and the host) add the owners of the other component's line.
With the reconstruction on, the CPU does not predict, so DecCu gives the CU's ccpChromaDimdMode a placeholder in range
for the host's transform choice (the GPU chooses again). Rejected: without DIMD in the SPS (the CPU's mode is then
unset). Test set: m31 (M31's arguments with LFNST: 5 to 31 such blocks per CIF picture, 77 to 122 per 720p picture,
one above 32 x 32 in park720). GPU time: m31 137.7 ms per picture; plain 12.38 -> 12.59 (code it never runs; a
fillReferencePar with its old signature measured the same), the other sets within 0.6 %.
M33: GLM's gradient patterns (JVET-AA0126; glmIdc 1 .. NUM_GLM_PATTERN = 4 with CCCM). A CCLM block with one models
the pattern's gradient instead of the downsampled luma, everywhere xGetLumaRecPixels takes it: its template (the row
above from the rows around ly - 2 also in a CTU's first row) and the co-located block (`glmGradient`: xGlmGetLumaVal,
which with JVET-AB0092 returns the gradient alone, as a Pel; the taps left of, at and right of the luma sample in its
row, the row below and, for the cross location type of JVET-AP0105, the row above, padded as the downsampling pads;
the patterns of the three location types in intra_cclm.glsl's GLM_PATTERN, Rom.cpp's g_glmPattern). Tu::lm carries the
pattern (LM_GLM_SHIFT) and with LFNST / NSPT the other component's (cclmTransformMode's other prediction). Rejected:
GLM with luma (glmIdc 5 .. 8: a CCCM model, xGlmCalcModel; next after CCCM). Test sets (encTest11, see section 9, with
ECM_TEST_GLM_GRADIENT_ONLY and ECM_TEST_NO_CHROMA_FUSION): m33 (M31n's arguments without its LFNST switches, which
default off: 14 to 72 GLM blocks per CIF picture, 92 per four720 picture, none in park720), m33l_fore32 / 22 / 37
(with LFNST and NSPT, as m31), m33a_fore22 / m33c_fore22 (the averaging and the cross location types, with LFNST).
GPU time: m33 130.3 ms per picture; the other sets within 0.7 %.
M34: CCCM (JVET-AA0057), the plain kind. A chroma block of a CCCM CU (an LM mode with cccmFlag) fits each chroma
component a 7-tap model - the downsampled luma sample (minus a luma offset, JVET-AB0174), its four neighbours, a
nonlinear term, a bias - to the template of its reference area (up to 6 lines above / left of the block with the
above-right / below-left units available; the CU's template selection, JVET-AB0143: both sides, left, above) by
CccmCovariance::solve2's division-free least squares (JVET-AC0053's Gaussian elimination): one model, or for the MMLM
modes two split at the template's mean luma; the prediction is clipped to the template chroma's range (JVET-AP0168).
In intra slices with the TM non-inter tools LBCCP lets a second set compete by the SATD of the template's chroma, Cb's
and Cr's summed: a single model fitted to the template's two lines next to the block, or two models split at the
block's mean luma (still fitted to the whole template: the first version fitted them to the two lines too); then
LBCCP's inside filter (intra_cclm.glsl's, reading the CCCM prediction). intra_cccm.glsl builds the luma buffer in
xCccmCreateLumaRef's order (the padding right of / below the area's parts, then in a dual tree the luma right of /
below the block within its CTU over it) in the four angular arrays, or for areas above 7168 samples in the workgroup's
scratch (blendAcc); per model 42 sums (ATA's upper triangle, ATCb, ATCr) of 6 invocations each, 32-bit (the host
rejects W + H > 164: chroma blocks above 64 x 64 / 128 x 32, which need a CTU of 256); the 7 x 9 elimination one row
per invocation; the models as 64-bit words in s_spCost; the costs with timdHadLine's one-sample-thick kernels. Each
block (Cb, Cr) fits both components' models: the choice needs both costs. Host: Tu::lm's cccmFlag (LM_CCCM_SHIFT),
Params::dualTree / cccmLbccp, the WITH_CCCM pipelines (LmLevel beside each IntraRecVariant), a CCCM block's parameter
snapshot reset (DecCu calls no initIntraPatternChType for it, so it held the previous block's, a chroma TMRL CU's
reference line too); intra_deps and the host add both components' templates and the luma area. Rejected: GL-CCCM,
CCCM without subsampling, with several downsampling filters, BVG-CCCM, with LFNST / NSPT (block level; next), CCP
merge and decoder-derived CCP fusion (sequence level). Test sets (encTest13, see section 9): m34 (m33's arguments with
`--CCCM=1`: 22 to 244 CCCM blocks per CIF picture, 666 to 822 per 720p picture, all template selections, one and two
models), m34a_fore22 / m34c_fore22 (the averaging and the cross location types). The scratch luma buffer and the
prediction recomputed for blocks above 2048 samples were also verified with both thresholds lowered (896 / 64
samples, so most blocks take them). GPU time: m34 136.3 ms per picture (m33, its arguments without CCCM, 130.6); the
other sets within 0.3 % (the CCCM code is in the WITH_CCCM pipelines only).
M35: GL-CCCM (JVET-AC0054). A GL-CCCM CU's model takes, instead of the luma sample's four neighbours, a vertical and a
horizontal gradient of its 3x3 neighbourhood (above minus below, left minus right, the middle tap doubled) and the
sample's row and column in the reference area ((y + 8) << 3, (x + 8) << 3); LBCCP's second set does not apply to it
(predIntraCCCM's condition), its inside filter does. Tu::lm's LM_CCCM_GL (bit 12: a CCCM block has no other
component's slope adjustments). The gradient products reach 2^26, so a model's sums are now 64-bit: each lane
accumulates in 64 bits and stores its partial sum in s_tmp (four 16-bit words per value), 42 invocations add each
sum's six; M34's 32-bit bound (W + H <= 164) is gone, the host rejects only a block whose largest luma buffer and
prediction do not fit the workgroup's scratch (128 x 128 chroma, CTU 256 only). A block above 2048 samples keeps its
prediction in the scratch after the luma buffer instead of computing it again at each read: M34's recomputation,
inlined at every read of LBCCP's inside filter, made intra_rec's SPIR-V 3.3 MB (M33 2.4, M35 2.7) and its glslang -Os
twice as slow (107 against 61 s, both under load). Test set (encTest14, see section 9): m35 (m34's arguments: 22 to 90
GL-CCCM blocks per CIF picture, 280 to 516 per 720p picture, plain CCCM beside them). Also verified with the shared
capacities lowered (896 / 64 samples). GPU time: m35 137.2 ms per picture, m34 136.9 -> 135.5 (the smaller code), the
other sets within 0.9 %.
M36: CCCM with several downsampling filters (JVET-AD0202, cccmMultiFilterIdx 1 .. 3). Its model has 10 (mode 1) or 11
(modes 2, 3) parameters over the luma of two or four of xCccmGetLumaVal's filters (filter 0 the mean, 1 to 3
gradients with negative values 0, per location type): mode 1 the samples of filters 0 to 3, three nonlinear terms,
the row and the column; mode 2 the sample and its left / right neighbours of filters 0 and 1, filter 0's nonlinear
terms, the column; mode 3 the neighbours above-right / below-left, filter 3, the row; no LBCCP second set. The CCCM
code now takes M parameters (the sums' numbering, as many lanes per sum as fit the workgroup: 6, 3 or 2; the
elimination's rows after it in s_tmp, the pivot row in s_spCost) and a luma buffer per filter, all of them in the
workgroup's CCCM scratch, so the rejection of CCCM blocks above BLEND_ACC is gone: Params::blendAcc became
Params::scratch, blending's sums at their constant stride, then from Params::cccmScratchOffset per workgroup
Params::cccmScratchWords (the picture's largest CCCM block's buffers and prediction). Two versions measured worse:
one stride read from Params for both (blending's too) made the m19 set 4 % slower (81.2 -> 84.5 ms per picture), a
CCCM buffer of its own (its address a further Params field) the CCCM sets 0.7 % slower than this layout. Tu::lm's
LM_CCCM_MDF_SHIFT (bits 13, 14). Test set (encTest14 without its CCCM switches: GL-CCCM and every filter):
m36 (m35's arguments: 26 to 142 CCCM blocks of several filters per CIF picture, 250 to 320 per 720p picture, beside
plain and GL-CCCM ones). Also verified with the shared capacities lowered (896 / 64 samples). GPU time: m36 143.4 ms
per picture; m34 135.3 -> 136.2, m35 136.9 -> 137.9, the other sets within 0.6 %.
M37: CCCM without subsampling (JVET-AC0147, cccmNoSubFlag; CTC's CCCM=2). Its luma buffer (xCccmCreateLumaNoSubRef)
holds the luma itself, minus the luma offset, over the reference area at twice its size per direction, padded by two
samples; the model's taps lie around the co-located luma sample (2x, 2y), by the chroma location type
(xCccmCalcModels' chromaloctype): the averaging type's 2 x 2 samples and their nonlinear terms (9 parameters), the
cross's sample and its four neighbours (11), the 6-tap filter's two rows of three and four nonlinear terms (11). The
template and the prediction are at the chroma positions (every second luma sample), the two-model threshold is the
mean of all the template's luma samples (four per chroma sample); no LBCCP second set; no minimum clipping span (above
7 parameters). The taps reach at most (2x + 1, 2y + 1), never the buffer's right / bottom padding, which is built
anyway as the CPU builds it. intra_cccm.glsl's luma buffers are now in the buffer's units (g_cccmShift: cccmBufferW,
cccmLumaIndex, cccmColocated; cccmBuildLuma builds both kinds, the padding a chroma sample wide, in a dual tree the
CTU checks of luma positions against the chroma CTU size as on the CPU); a buffer up to 7168 samples stays in shared
memory. Tu::lm's LM_CCCM_NOSUB (bit 15). Test set (encTest14 without its CCCM switches, `--CCCM=2`): m37 (m36's
arguments: 10 to 42 CCCM blocks without subsampling per CIF picture, 54 to 442 per 720p picture, beside the other
kinds), m37a_fore22 / m37c_fore22 (the averaging and the cross location types). Also verified with the shared
capacities lowered (896 / 64 samples). GPU time: m37 138.1 ms per picture; m34 135.9 -> 136.6, m35 137.6 -> 138.3,
m36 142.8 -> 143.6, the other sets within 0.7 %.
M38: CCCM with LFNST / NSPT (JVET-AK0064 for CCCM CUs). Their mode is the DIMD of the CU's predictions
(deriveChromaIpmForTransform: Cb's, Cr's or both by the TU's joint CbCr mode), which predIntraCCCM makes for both
components at the Cb block, LBCCP's inside filter applied. On the GPU each such block (cclmTransformMode's CCCM
branch) predicts both components with cccmPrediction's `both` (Cb's prediction, then Cr's: in s_tmp up to 2 x 1024
samples, else in the CCCM scratch, which the host sizes for two), the gradients read them through the inside filter
where the CU has it, and the prediction section predicts the block again (as for CCLM). The host leaves a CCCM
block's other-component field to its kind (no slope adjustments). The second prediction showed a bug: wgSum2's s_red
slots must be zero before each use and the block's start clears them once, so the second two-model threshold
included the first one's sum; cccmPrediction now clears its slots (the threshold's, LBCCP's block mean). Found with
temporary debug builds: the per-block comparison of predictions and residuals past the first parameter mismatch (the
first difference a two-model block's prediction, off by one), then a build without the first prediction (right).
Test set (encTest14 without its CCCM switches, `--CCCM=2 --IntraLFNSTISlice=1 --NSPT=1 --LFNSTExt=1`): m38 (6 to
51 CCCM blocks with LFNST / NSPT per CIF picture, 75 to 118 per 720p picture, of every kind, with and without the
inside filter, every joint CbCr mode). Also verified with the shared capacities lowered (896 / 64 samples). GPU time:
m38 143.3 ms per picture; the other sets within 0.7 % (m31, CCLM with LFNST / NSPT, the same).
M39: GLM with luma (JVET-AB0092, glmIdc 5 .. 8 of an LM_CHROMA / MDLM_L / MDLM_T block: xGlmCalcModel /
xGlmApplyModel). A single model of 3 parameters - the value of the block's GLM gradient pattern (xGlmGetGradVal: no
luma offset, the samples left of / above the picture its first column / row), the downsampled luma minus the luma
offset, a bias - over CCCM's reference area, its template by the LM mode (MDLM_L the left part, MDLM_T the part above,
LM_CHROMA both without the units above-right / below-left), solved as CccmCovariance::solve1, which is solve2's
elimination of one right-hand side with the matrix scaled to 28 bits instead of 22 (each right-hand side is
eliminated alone, so solve2's model of the component is solve1's), clipped to the sample range alone; no inside filter,
no LBCCP. intra_cccm.glsl has it as a model kind (g_cccmGlmLuma: two luma buffers, the second the gradient
pattern's, CCCM_FILTER_GLM; cccmTemplate's restriction; GLM_MATRIX); the host records it as a CCCM block (Tu::lm's
LM_CCCM_GLM_LUMA, bit 16, the template selection its LM mode's, its glmIdc in the GLM field), so the CCCM pipelines,
dependencies and scratch serve it. Still rejected with LFNST / NSPT (the DIMD of both components' predictions, whose
kinds may differ). Test set (encTest14 without ECM_TEST_GLM_GRADIENT_ONLY, m33's arguments: CCCM off): m39 (8 to 70
blocks of GLM with luma per CIF picture, 58 to 146 per 720p picture), m39a_fore22 / m39c_fore22 (the averaging and the
cross location types). Also verified with the shared capacities lowered (896 / 64 samples). GPU time: m39 138.5 ms per
picture; the other sets within 0.8 %.
M40: LM chroma fusion (JVET-AC0119), its first type (isChromaFusion 1; CFLM, types 2 and 3, still rejected): a chroma
block of a regular mode blended with the plain CCCM prediction of two models (geneChromaFusionPred with JVET-AL0191 in
an intra slice: MMLM_CHROMA_IDX's models, the whole template, xCccmCalcRefAver's threshold, xSetMinMaxClpRng's
clipping, no LBCCP, no inside filter), (pred * w0 + cccm * (4 - w0) + 2) >> 2 with w0 1 where the CUs below-left
and above-right of the CU (getPURestricted) are both LM CUs, 3 where neither is, else 2 (the host's). The block
computes the CCCM prediction first, its luma buffer and prediction in the CCCM scratch (the regular prediction then
takes s_tmp and the angular arrays; cccmPrediction's g_cccmFusion), and storeRegion, which every final prediction
passes, blends each sample (cccmFusionBlend: after PDPC, MRL, intra fusion). Tu::lm's LM_FUSION (bit 30) and the
weight (bits 0, 1: the first slope adjustment's field, which a fusion block does not have), the template selection 1
of the models (not LM_ACTIVE: the block is not an LM block), so the CCCM pipelines, dependencies and scratch serve
it. A first version had LM_FUSION at bit 17, inside a CCLM block's field of the other component's slope adjustments
(bits 12 .. 19, with LFNST / NSPT): CCLM blocks of LFNST whose other component's second adjustment set that bit took
the fusion path (the full regression's m38_park720 / four720 differed; the scratch they wrote beyond was unsized).
Rejected where PU::cccmMultiModeAvail(MMLM, fusion) does not allow the CCCM models (the CPU then blends in the LM
buffer of a previous block), for a CU of several transform blocks, and CFLM. Test sets (encTest15: encTest14 with
ECM_TEST_CHROMA_FUSION_LM_ONLY, the fusion search without CFLM; encTest16 adds ECM_TEST_PREFER_CHROMA_FUSION, a
fusion candidate's RD cost divided by 4): m40 (m37's arguments, chroma fusion on: 0 to 8 fusion blocks per CIF
picture, all of weight 2, 1 in park720, 28 in four720, of every weight), m40f_fore32 / 22 / 37 (fusion preferred:
192 to 292 fusion blocks per stream, every weight, sizes from 2 x 8 to 64 x 64). Also verified with the shared
capacities lowered (896 / 64 samples). GPU time: m40 147.1 ms per picture, m40f 85.2; the CCCM sets 0.8 to 0.9 %
slower (m34 136.1 -> 137.3), the others within 0.9 %.
M41: CFLM, LM chroma fusion's types 2 and 3 (JVET-AC0119's isChromaFusion 2: one model, 3: two split at the
template luma's mean, xCflmCalcRefAver's PU::getMeanValue): the block's regular prediction (any mode, after PDPC, MRL,
intra fusion) is replaced by a model of 3 parameters - the downsampled luma minus the luma offset, the regular
prediction minus the chroma offset, a bias - fitted by solve1 (GLM_MATRIX bits, xSetMinMaxClpRng's clipping) to the
template of 2 lines above / 2 columns left of the block (those the picture has; no corner, no units above-right /
below-left), where the regular prediction is the block's mode predicted for the template (xCflmCreateChromaPred:
xFillTimdReferenceSamples for the template block with arms of twice its sides, predTimdIntraAng's chroma path with
initPredIntraParams' parameters of the block: the 1/32-sample angles interpolated by two taps, planar, DC and the PDPC
kinds as TIMD's). The GPU: intra_timd.glsl's template code with TimdParams::chroma and timdFillReference of any
component and arm length; intra_cccm.glsl's cflmArea before the block's references, storeRegion keeping the regular
prediction in the area's chroma buffer in the CCCM scratch (cflmStore), cflmPrediction once it is complete (the luma
buffer like CCCM's, cccmAccumulate / cccmSolve / cccmPredict with g_cccmCflm: the template without its corner, the
features, solve1). The host records the template's reference units (templateUnits of any block and arm length:
IntraPrediction::timdReferenceUnits takes the component and the arms' lengths) in Params::timdData at Tu::cflm (a new
field: Tu is 101 int32) and the fusion type in Tu::lm's bits 4, 5 (LM_FUSION_TYPE_SHIFT; the first type's records
too). A CPU quirk the GPU reproduces: xFillTimdReferenceSamples' units of the row number predSize / unitW rounded up
but begin after the corner unit, so with CFLM's arms of 2 and chroma units of 2 the last one writes 2 samples past the
row, into the column's first entries (the CPU's buffer holds the column right after the row): in the fill where the
unit is available (a later padding of the corner unit or from the first available unit overwrites them), in its
padding where it is not (TIMD's luma units of 4 end with the row); the first verification found it (template samples
next to the corner off by a few, the model's prediction by 1). The CFLM code is in pipelines of its own (WITH_CFLM,
LmLevel's LEVEL_CFLM): in the CCCM pipelines it slowed the CCCM sets by ~2 %. Rejected: CFLM with cccmNoSubFlag, a
CU of several transform blocks; LM chroma fusion of either type above 10 bits or with unequal bit depths (the first
type's check is new). Test sets (encTest16 without ECM_TEST_CHROMA_FUSION_LM_ONLY: the fusion search with every type):
m41 (m40's arguments: 2 to 92 CFLM blocks per stream, both types), m41f_fore32 / 22 / 37 (ECM_TEST_PREFER_CHROMA_FUSION
too: 84 to 186 CFLM blocks per stream, every kind of mode, templates of both arms and of one, sizes 2 x 8 to 64 x 64).
Also verified with the shared capacities lowered (896 / 64 samples). GPU time: m41 148.1 ms per picture, m41f 87.0;
the other sets within 0.6 %.
M42: GLM with luma with LFNST / NSPT (their mode the DIMD of both chroma components' predictions). M39 rejected it,
taking the components' kinds for independent; they are not: CABACReader::glmIdc gives Cr Cb's glmIdc (JVET-AB0092)
and no slope adjustment comes with GLM (PU::hasCclmDeltaFlag), so the other component's model is of the block's kind
and gradient pattern, which cclmTransformMode's CCCM path (cccmPrediction of both components) already predicts. The
host writes the other component's slope adjustments and glmIdc into a plain CCLM block's Tu::lm only (a GLM-with-luma
block's bits 12 .. 19 hold its CCCM kind: they would have set LM_CCCM_NOSUB) and rejects components of different
glmIdc (no stream can have them). Test set (m39's encoder and arguments with LFNST, NSPT and LFNSTExt on): m42 (1 to
38 blocks of GLM with luma with LFNST / NSPT per stream, LM_CHROMA and MDLM_L, every pattern). GPU time: m42 144.4 ms
per picture; no kernel change.
M43: the SPS switches of CCP merge (JVET-AD0188), decoder-derived CCP fusion (JVET-AG0154) and BVG-CCCM (JVET-AE0100)
accepted; their CUs are rejected per block (idxNonLocalCCP, decoderDerivedCcpMode; BVG-CCCM by otherCccm as before),
their pictures reconstructed by the CPU. The switches only add the tools' syntax and their candidates' reordering:
the other LM blocks are predicted as without them, and in ON mode the derive-only pass predicts nothing, so it stores
no CCP model (CU::saveModelsInHCCP takes typed models only) a merge CU could inherit. What the tools need on the GPU
(from the decoder: DecCu::xIntraRecBlk 1416 .. 1617): every LM block's model stored as PredictionUnit::curCand
(CCPModelCandidate, Common.h; set in predIntraChromaLM / predIntraCCCM) and pushed after its CU into the CTU row's
history (cs.ccpLut, CU::saveModelsInHCCP; reset per slice and CTU row) and the picture's model map; a merge CU's
candidates (PU::getCCPModelCandidateList: 5 adjacent positions, non-adjacent ones, the history, defaults; pruned by
equal models) ranked by template costs (reorderCCPCandidates), with adjustment (adjustCCPCandidates), fusion with
CCCM or DIMD chroma; decoder-derived CCP fusion's candidate models costed on the template and the best pair blended
(decoderDerivedCcp); BVG-CCCM needs IBC / IntraTMP luma CUs (bvgCccmModeAvail). Test sets (encTest17: encTest16 with
ECM_TEST_NO_CCP, the chroma searches' conditions of the three false; the m41 set's arguments with --CCPMerge=1
--DDCCPFusion=1 --BvgCccm=1): m43 (no CU of the tools), m43c_fore32 / 22 / 37 (without ECM_TEST_NO_CCP: every picture
has CCP merge and decoder-derived CCP fusion CUs, so the CPU reconstructs all 8).
M44: the LM CUs' CCP models, CCP merge's first step. Every LM CU's blocks write the model the CPU keeps as
PredictionUnit::curCand (predIntraChromaLM's cclmModelToCcpParams, predIntraCCCM's cccmModelToCcpParams,
xGlmCalcModel's glmModelToCcpParams) into a record of CCP_OUT int32 in Params::ccpOut (at Tu::ccp, a new field: Tu is
102 int32; one record per PU): the type (CCLM, GLM's gradient patterns, GLM with luma, CCCM, GL-CCCM, several filters,
without subsampling; MMLM with two models), glmIdc, the threshold of two models, the luma offset, the reference
area's columns / rows left of / above the block (0 without subsampling), the filters' index, the parameters' number,
per component the shifts, per model and component the template chroma's range and the 64-bit parameters. A CCLM
block writes its component's part after its model (cclmStoreCcp: a and b with the slope adjustments, the shift), the
Cb block of a CCCM CU both components' after LBCCP's choice (cccmStoreCcp: the chosen set's models and threshold; each
block fits both components' models), each GLM-with-luma block its component's. Verification compares each record's
fields of its type (those CCPModelCandidate::operator== and predCCPCandidate read) with the CPU's curCand after the
block's prediction. Nothing reads the records yet. Not covered: LM chroma fusion CUs (their mode is not an LM mode,
but geneChromaFusionPred stores their CCCM models as curCand: spatial merge candidates) and LBCCP's inside filter,
which CU::saveCcInsideFilterFlagInCCP adds to curCand after the CU (not to the history's copy). The CPU predicts both
components of a CCLM CU at the Cb block (JVET-AK0064), so its curCand is complete after either block. Canaries: 14 (a
field of each type off, the threshold, the ranges, LBCCP's set, NS-CCCM's type / corner offsets / parameters' number,
a record per block instead of per CU). GPU time against M43: the plain, CCLM, GLM, CCCM, CCCM-kinds, GLM-with-luma and
CFLM sets within -0.3 % .. +0.7 % (m31 136.3 ms per picture, m41 148.8).
M45: the rest of the CCP models a merge CU can inherit. LM chroma fusion CUs (the first type: a regular chroma mode)
have a record too: geneChromaFusionPred stores their CCCM models of MMLM_CHROMA_IDX as curCand (CCCM | MMLM, never
GL-CCCM: the CU has no glCccmFlag), so they are spatial candidates of their neighbours, though not in the CTU row's
history (CU::saveModelsInHCCP takes LM CUs only); their Cb block writes it (cccmStoreCcp). The record holds LBCCP's
inside filter twice (CCP_OUT_INSIDE: the CU's curCand after CU::saveCcInsideFilterFlagInCCP, the CU's flag;
CCP_OUT_INSIDE_HIST: the history's copy, which saveModelsInHCCP took before that: 0 for an LM CU); a merge CU will
write the inherited flag into both. Verification now reads the CPU's curCand after the picture (the PU of each record;
with the flag) and checks that every chroma PU without a record has an untyped curCand (else a merge CU next to it
would inherit a model the GPU lacks). Canaries: 4 (the fusion record not written, typed with one model, the inside
flag dropped, a fusion CU without a record). GPU time against M44: the same sets and m40 within -0.2 % .. +0.4 %.

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
   M25 OBIC and intra merge with TIMD neighbours (the m25, m25o, m25s and m25os sets), M26 TIMD-SAD (the m26 and
   m26o sets), M27 OBIC and intra merge with TIMD-SAD neighbours (the m27 and m27o sets), M28 TIMD merge (the m28 and
   m28m sets). 2026-10-13: M29 TMRL of luma (the m29 set, m29n_fore32, m29s_screen32 / 22), M30 chroma TMRL (the m30
   set), M31 CCLM (the m31n set, m31a_fore32 / 22, m31c_fore32 / 22), M32 LFNST / NSPT of CCLM (the m31 set), M33
   GLM's gradient patterns (the m33 set, m33l_fore32 / 22 / 37, m33a_fore22, m33c_fore22), M34 CCCM (the m34 set,
   m34a_fore22, m34c_fore22), M35 GL-CCCM (the m35 set), M36 CCCM with several downsampling filters (the m36 set),
   M37 CCCM without subsampling (the m37 set, m37a_fore22, m37c_fore22), M38 CCCM with LFNST / NSPT (the m38 set),
   M39 GLM with luma (the m39 set, m39a_fore22, m39c_fore22), M40 LM chroma fusion of CCCM (the m40 and m40f sets),
   M41 CFLM (the m41 and m41f sets), M42 GLM with luma with LFNST / NSPT (the m42 set), M43 the SPS switches of the
   CCP tools (the m43 and m43c sets). 2026-10-14: M44 the LM CUs' CCP models, M45 those of LM chroma fusion CUs and
   the inside filter's flags (verified on the LM sets). All 19 regular streams now stop at "chroma reordering / DBV"
   (the next sequence-level check; the block-level tools of step 4 follow, CCP merge's first: M46 the merge list, its
   costs and prediction, then merge fusion, merge adjustment, decoder-derived CCP fusion).
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
   cost); `bin\encTest8` (m26o) adds ECM_TEST_PREFER_TIMD_SAD (`testTimd` off where `testTimdSad`); `bin\encTest9`
   (m29) adds ECM_TEST_NO_CHROMA_TMRL (the `PU::hasChromaTmrl(pu) && ...getUseTmrl()` conditions of the chroma search
   false); `bin\encTest10` (m31) adds ECM_TEST_NO_GLM (the `PU::hasGlmFlag` conditions of the chroma search false) and
   ECM_TEST_NO_CHROMA_FUSION (its `PU::hasChromaFusionFlag` conditions false); `bin\encTest11` (m33) adds
   ECM_TEST_GLM_GRADIENT_ONLY (xFindBestGlmIdcSATD searches the gradient patterns only, not GLM with luma);
   `bin\encTest12` adds ECM_TEST_CCCM_PLAIN (the chroma searches' CCCM candidates without GL-CCCM's, the first
   downsampling filter only) and `bin\encTest13` (m34) extends it to LBCCP's search (its GL-CCCM candidate and its other
   downsampling filters; encTest12's streams still had them); `bin\encTest14` (m35) adds ECM_TEST_CCCM_NO_MDF (the
   downsampling-filter switches of ECM_TEST_CCCM_PLAIN alone: GL-CCCM stays; m36: none of them; m37: none, with
   `--CCCM=2`; m38: m37's with LFNST / NSPT; m39: m33's arguments without ECM_TEST_GLM_GRADIENT_ONLY; m42: m39's with
   LFNST / NSPT); `bin\encTest15`
   (m40) adds ECM_TEST_CHROMA_FUSION_LM_ONLY (the chroma fusion search's `uiFusionModeNum` 1: no CFLM) and
   `bin\encTest16` (m40f) ECM_TEST_PREFER_CHROMA_FUSION (a fusion candidate's RD cost divided by 4; m41 / m41f: its
   search with every type, without ECM_TEST_CHROMA_FUSION_LM_ONLY); `bin\encTest17` (m43) ECM_TEST_NO_CCP (the
   chroma searches' PU::hasNonLocalCCP, hasBvgCccmFlag and hasDecoderDerivedCCP conditions false). The user's order
   (2026-10-13): all intra first, then inter. Next the CCP tools' CUs (each LM block's model on the GPU, compared
   with the CPU's curCand; CCP merge's list, ranking, adjustment and fusion; decoder-derived CCP fusion; BVG-CCCM
   with IntraTMP / IBC), chroma reordering / DBV, BVG-DIMD, MDIP / MPM sorting, IntraTMP / IBC, EIP / SGPM, PDP, NN
   intra, implicit MTS (not in CTC; its test set, the m24 arguments with `--MTS=0 --MTSImplicit=1`, is encoded:
   `streams_gpu\pending\imts_*`), deblocking on the GPU, then inter pictures.
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
  site each instead), M31's CCLM 30 % on plain pictures. Since M19 intra_rec is built as several pipelines with
  specialization constants (`WITH_DIMD`, `WITH_DERIVED_TRANSFORMS`, `WITH_MERGE`, ..., `WITH_CCLM`, `WITH_CCCM`;
  `gpu::Kernel::create`'s specialization, GpuIntraRec.cpp's `IntraRecVariant` and `LmLevel`) and a picture runs the one without the
  features it does not use. The compiler is not monotonic
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
- PowerShell 5.1 drops an empty-string argument to a native program: `DecoderApp -b x.bin -o '' --GpuRecon=2` ran as
  `-o --GpuRecon=2`, a CPU decode into a file named `--GpuRecon=2` without a summary line. Decode through the Python
  scripts (`verify_all.py`, `gpucheck.py`), which pass `-o ""` intact.
- NVIDIA's shared shader cache (`%LOCALAPPDATA%\NVIDIA\GLCache`) holds 1 GB. At that limit (2026-10-13, after many
  canary builds) it stored no new pipelines for a while (later the driver pruned it to ~320 MB): every decode of a
  new build compiled all of intra_rec's pipelines, ~96 s per decoder start (six decodes of one canary build took ~90 s
  each). Since then intra_rec's variants are compiled on first use (`intraRecKernel`: ~19 s for a cold start that
  needs one), and the test scripts give the driver a cache of their own (`__GL_SHADER_DISK_CACHE_PATH`,
  `__GL_SHADER_DISK_CACHE_SIZE`, `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP`, which the Windows driver honors).
- Build time of a shader change (2026-10-13): ~15 s, of which glslang `-Os` on intra_rec.comp takes 14.1 s (0.3 s
  without the optimizer, `-O` 12.2 s); without `-Os` the driver's cold compile of a pipeline takes 30.7 instead of
  ~19 s, so `-Os` stays. Each shader depends on the includes glslang's depfile names (CMake 3.21+), not on all of them;
  moving the SPIR-V arrays out of GpuIntraRec.cpp gained nothing (its compile runs beside glslang's).
- Buffer references (device addresses) have no robustness: a read past a plane's end loses the device
  (VK_ERROR_DEVICE_LOST, nvlddmkm event 153; all three decodes running at the time failed), and one past a row's end reads
  the next row unnoticed. To find such a read without more device losses (M31's first version: luma TIMD / TMRL blocks
  of 131-mode values from 67 on took the CCLM branch), a temporary check in `recoAt` that returns 0 out of the
  picture and records the block (`atomicMax` into the unused `counter[3]`, downloaded with the dependency count;
  `atomicCompSwap` of the coordinates into a Tu field that verification downloads) named the blocks and the positions
  in one decode (stubbing parts of the new code out had only narrowed it to the prediction step).
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
