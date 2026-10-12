# ECM-20.0 encoder speed-up log

Ground rules
- **Bit-exact**: every step must reproduce the stock ECM-20.0 bitstreams byte for byte (md5 of the `.bin`).
- *Step* = speed increase vs. the previous step, *cumulative* = vs. the stock baseline
  (`t_prev / t_new - 1`, `t_base / t_new - 1`).
- Machine: i7-10700 (8C/16T), Windows 11, GCC 16.2 (WinLibs), static builds.

Measurement method
- Build-level steps 1-3 were measured with `perf/bench.py run <name> --jobs 5` on an otherwise idle machine.
- Unpaired suite runs turned out to drift by ±2 % with background load, so code steps are measured **paired**:
  `perf/bench.py ab <prev> <new> --jobs 6` runs the same test with both builds at the same time (both see the same
  load); the table gives the total wall-clock ratio over the 5 tests. Code steps are compared on LTO builds
  (no PGO); PGO is applied again on the final build.

Benchmark suite (`perf/bench.py`)

| test | config | sequence | frames | QP |
|---|---|---|---|---|
| ai_foreman | encoder_intra_ecm.cfg | foreman 352x288 | 1 | 32 |
| ai_parkjoy | encoder_intra_ecm.cfg | ParkJoy 256x256 crop | 2 | 27 |
| ra_foreman | encoder_randomaccess_ecm.cfg | foreman 176x144 crop | 9 | 32 |
| ra_parkjoy | encoder_randomaccess_ecm.cfg | ParkJoy 256x256 crop | 5 | 37 |
| ld_foreman | encoder_lowdelay_ecm.cfg | foreman 176x144 crop | 5 | 32 |

## Single-thread results

| # | optimization | bit-exact | step | cumulative |
|---|---|---|---|---|
| 0 | stock ECM-20.0 (GCC 16.2 -O3, static): 986.0 s | - | - | - |
| 1 | `-march=x86-64-v3 -ffp-contract=off` (kept as `ECM_MARCH` option, default off) | yes | +0.3 % (noise) | +0.3 % |
| 2 | link-time optimization | yes | +5.7 % | +5.7 % |
| 3 | profile-guided optimization on top of LTO (`perf/pgo.ps1`) - re-applied at the end | yes | +6.3 % | +12.3 % |
| 4 | SIMD for 1-2 sample wide and 16n+1 wide interpolation (template rows/columns; 56 % of filter calls were scalar) | yes | +1.8 % | |
| 5 | large structs (merge contexts, CU/PU, MotionInfo) passed by const reference | yes | +1.5 % | |
| 6 | sparse CABAC context restore (dirty tracking) - **reverted**: only 20-39 % of the 6.5 M context copies per frame qualified, no measurable gain | yes | ~0 % | |
| 7 | stable sorts of short lists without heap temporaries; reused sign-prediction buffer | yes | +1.2 % | |
| 8 | ALF covariance statistics with SSE over independent float accumulators | yes | +1.8 % | |
| 9-11 | MC / template interpolation result cache (54-56 % of calls repeat identical inputs) - three variants, **reverted**: the repeated calls are the cheap ones, lookup + arena traffic cost as much as recomputing | yes | ~0 % | |
| 12 | copy only the used entries of the HMVP/CCP/EIP history tables (15+5.6+3.2 KB per coding-structure init) | yes | +0.5 % | |
| - | clang 23 + ThinLTO instead of GCC: AI +1.5 %, but inter coding crashes (UB-sensitive) - dropped | - | - | |
| 13 | AVX2 paths for 12-tap luma interpolation (horizontal 8 columns per iteration, vertical 16) - same arithmetic as the 4-wide SSE kernels | yes | +0.8 % (RA +1.0-1.3 %) | |
| 14 | 8-wide 12-tap vertical interpolation via the generic AVX2 M8 kernel | yes | +0.5 % (CPU 1.00x: noise level) | |
| 15 | CABAC context store copies skip unchanged 32-context blocks (per-block version stamps, dirty marking on write) | yes | +2 % CPU (+4 % wall), RA +3 % | |
| - | CU/PU position lookups through a cached parent chain - **reverted**: walks are short, the bookkeeping cost more (0.98x) | yes | -2 % | |
| 16 | forward-transform (256 KB) and angular-prediction (128 KB) work buffers moved off the stack: every call probed all those stack pages | yes | +1 % (CPU 1.01x on every test) | |
| 17 | SSE2 sign-prediction hypothesis costs; inline copies of small context sets | yes | AI +1 %, RA/LD ~0 | |
| 18 | lazy clearing of the per-CU affine AMVP buffer; inline copies of 8-64 byte rows | yes | +0.5 % | |

| - | `-fno-tree-loop-distribute-patterns` (keep small copy loops inline) | yes | 1.00x - dropped | |
| 19 | angular-prediction reference buffers kept zero between calls (each call clears the range it wrote; verified by a build that checks the whole buffers after every call); SSE2 dot products for the CCCM normal equations | yes | +0.6 % CPU (LTO) | |
| 20 | long-tap angular interpolation rows specialized on the tap count (no out-of-line call + switch per sample) | yes | AI +1 %, RA/LD 0 | |
| 21 | context store restores skip the stamp comparison when the source is unchanged since the last sync | yes | +1 % CPU (AI +1-2 %) | |
| 22 | AVX2 NSPT / LFNST kernels without gathers (axpy over contiguous matrix rows) | yes | +0.4 % (inverse NSPT; forward NSPT / inverse LFNST 0) | |
| - | getCURestricted: wavefront check only when wavefronts are on | yes | 1.00x - dropped | |
| 23 | small-object allocator behind operator new / delete (size-class free lists, 16-byte headers) | yes | +2 % CPU (every test 1.02-1.03x) | |
| - | GPM split-mode reordering sort: carry-swap loop instead of the backward copy that GCC turns into two memmove calls per insertion | yes | 1.00x - dropped (AI 1.03-1.04x although all-intra never runs GPM, i.e. layout noise; LD 0.95x) | |

Direct paired measurement stock vs current (LTO, no PGO, steps 4-22): **1.17x** (every test 1.15-1.19x, wall and
CPU, bit-exact); with step 23: ~1.19x. PGO (+~6 %) is re-applied at the end.

Steps 15-18 were measured on builds without LTO (perf/build.ps1 did not enable it; it does now), both sides of each
comparison built the same way; steps 19+ on LTO builds.

Paired totals (both builds run each test at the same time):
- code steps 4-12 on top of PGO: `pgo` -> `best`: **+5.8 %** (every test 1.05-1.07x)
- stock -> `best` (LTO + PGO + code steps): **1.15x** (every test 1.13-1.17x, wall and CPU)

## Segment-parallel encoding (`--ParallelSegments=N`)

- Random access: chunks of whole intra periods are encoded by concurrent child processes and joined (leading IDR
  access unit of each later chunk dropped, POC LSBs rebased at RBSP level).
- All intra (every picture a CRA, parameter sets repeated): a worker starts directly at its chunk's POC
  (`--SegmentPOCOffset`, with the RAS-init state the full encode has at that CRA), so chunks are disjoint and the
  bitstreams are simply concatenated. By default every picture is its own chunk and each worker takes the next
  picture when it is done (process start-up ~0.5 s vs ~16 s per 128x128 picture).
- Verified byte-identical to the sequential encode (bitstream and reconstruction): RA 65 frames (2 chunks),
  AI 10 frames (1-3 picture chunks; overlapping mode with `ReWriteParamSets=0`), ra_long, ai_long.

| test | coded frames | stock serial | parallel | wall-clock speedup | bit-exact |
|---|---|---|---|---|---|
| ra_long (foreman 128x128 crop, RA, QP 32, intra period 32) | 257 | 6011 s (5929 s CPU) | 1090.6 s (best, 8 chunks) | **5.51x** | yes |
| ai_long (foreman 128x128 crop, AI, QP 32, 257 input frames, TemporalSubsampleRatio 8), first version: overlapping chunks, 8 workers | 33 | 579.2 s | 152.2 s (best2) | 3.80x | yes |
| ai_long, disjoint chunks, 8 workers | 33 | 579.2 s | 133.2 s (dev = steps 4-14 without LTO/PGO) | 4.35x | yes |
| ai_long, 1-picture chunks, 16 workers (hyperthreads) | 33 | 579.2 s | 87.0 s (dev) | **6.66x** | yes |

RA parallel efficiency is ~60 % (1.15x single-thread x 8 workers would be 9.2x): shared L3/DRAM bandwidth,
lower all-core turbo, one duplicated intra frame per chunk, unequal chunk complexity. With intra period 32 a
257-frame sequence has only 8 chunks, so hyperthreads cannot be used.

## Frame threads (`--FrameThreads=N`, `--FramePeriods=M`)

- The pictures run in threads of their own, at most N searching at a time. The setups take turns in coding order on
  a setup stack of their own; a picture takes a search stack once the pictures it reads are done, the earliest such
  picture first, and gives it back when its search ends: its finish uses its intra period's finish stack
  (`perf/HANDOFF_MULTITHREADING.md`, P1 and P5). The GOPs overlap: the next GOPs are read and started while a GOP's
  pictures are in flight, and the reconstructions are written in output order as they are done.
- The finishes (loop filters, APSs, slice writing) of an intra period run in coding order. With M > 1 the finishes of
  up to M intra periods run at the same time, each period with its own copy of the filter encoders and APSs, while the
  log lines, access units and PSNR sums still go out in coding order. A period starts at a RAS-init intra picture
  (the CRA of each intra period in RA, every picture in AI), whose finish resets what the filter encoders carry.
- Output identical at every N and M: bitstream, reconstruction and log (except the [ET] times).
- M = 0 (default) uses min(N, 4) periods, and 1 for one long period (IntraPeriod -1, LD) and for the options whose
  finishes share state between periods (buffering period, picture timing, decoding unit, frame-field, frame packing,
  scalable nesting and DRAP SEIs, DeblockingFilterMetric, a scaling-list file, HDR metrics), which M > 1 rejects.
- AI pictures are independent, so AI scales with N, also onto the efficiency cores: 33 pictures unpinned on all 20
  cores of the 265K took 27.1 s with N = 12, 25.8 s with 16 and 23.7 s with 20 (7.96x, 8.37x, 9.12x; 24.7 GB at 20).
  In RA the temporal CABAC context store chains each picture to the previous one of its temporal layer, so a single
  intra period gains about 1.4x; the periods then overlap. The dependency chains inside the periods bound RA at about
  4.4x for 129 frames (5 periods) whatever N; longer RA sequences gain from more periods (`perf/mt/p5_sim.py` models
  10 periods at 4.4x with M = 4 and 5.9x with M = 8, at a cost in memory). LD gains nothing. `--ParallelSegments`
  combines with it.
- Memory: a search stack commits about 0.85 GB whatever the resolution; the setup stack and the finish stacks are
  small. Each period beyond the first adds a finish context (about 33 MB at 128x128, 75 MB at CIF and 0.8 GB at 1080p,
  mostly ALF and SAO statistics), and up to max(N, GOPSize) x M pictures are in flight. A picture that waits for its
  search holds its buffers, coding structure and temporary buffers (about 68 MB at 1080p); its search data (about
  80 MB more) is created when its search starts. RA 1080p with N = 8 and M = 4 committed about 27 GB in its first
  5 minutes.
- Core Ultra 7 265K, one run at a time on the 8 P-cores, foreman 128x128 QP 32 (ai_long's and ra_long's
  configurations), the previous commit's single-thread time as base, automatic M:

| FrameThreads | AI, 33 pictures (TemporalSubsampleRatio 8) | RA, 65 frames (3 intra periods) | RA, 129 frames (5) | peak commit (AI / RA 65 / RA 129) |
|---|---|---|---|---|
| 1 | 215.7 s | 556.2 s | 1064.5 s | |
| 2 | 104.0 s, 2.07x | | | 2.8 GB |
| 4 | 56.0 s, 3.85x | 203.2 s, 2.74x | | 5.4 / 5.7 GB |
| 8 | 30.9 s, 6.98x | 199.1 s, 2.79x | 259.8 s, 4.10x | 10.2 / 10.2 / 10.8 GB |

  Earlier steps, same tests: finishes in coding order and stacks held to the end of the finish (P5 stage A, B-1): RA
  65 387.6 s with 4 threads and 383.9 s with 8 (1.44x, 1.45x); finishes of several periods at the same time (B-2):
  RA 65 327.1 s and 199.0 s (1.70x, 2.79x), RA 129 403.4-429.2 s with 8 (2.48-2.64x). LD 13 frames: 118.5 s with one
  thread, 120.6 s with 8.

## CU threads (`--CuThreads=N`)

- In the intra luma RD passes without a running cost threshold (the DCT2 pass and the LFNST passes, not the MTS
  pass), N helpers (the picture's own thread and N-1 workers) code the RD candidates of a CU at the same time. The
  candidate loop runs twice: the first run sets every candidate up and records it (CU and PU bytes, LFNST mode
  indices), the helpers code the recorded candidates, and the second run takes each result where it would code the
  candidate, after checking that it set the candidate up the same way. Output identical at every N;
  `--CuThreadsCheck=1` codes every candidate in the picture's own search too and stops at a difference
  (`perf/HANDOFF_MULTITHREADING.md`, P2).
- Coded serially as before: ISP, PNN, TMP with LIC, BDPCM, ACT, CUs larger than the maximum transform size, passes
  with fewer than two candidates. A helper is a light stack of ~130 MB.
- The small-object allocator keeps per-thread free lists; with one shared spinlock the helpers' candidate coding took
  1.7x the serial time.
- The gain is small: the candidates take ~5 us each, and the rest of the loop (setups, ISP, PNN) stays serial.
  Workers spin 2 ms after each job (a sleeping thread wakes too late for a loop), which costs the picture's thread
  ~3 % of turbo frequency: foreman in LD and RA, where intra RD is a small part, gains 0-1 %; ParkJoy in RA, with
  many intra CUs, gains about as much as intra pictures.
- Core Ultra 7 265K, one run at a time on 4 P-cores (8 for N=8):

| test | CuThreads=1 | 2 | 4 | 8 | peak commit (1 / 4 / 8) | bit-exact |
|---|---|---|---|---|---|---|
| ai_foreman (CIF, 1 intra picture) | 33.2 s | 30.9 s, 1.07x | 29.8 s, 1.11x | 31.6 s, 1.05x | 1.5 / 2.0 / 2.5 GB | yes |
| ai_parkjoy (256x256, 2 intra pictures) | 46.7 s | 44.2 s, 1.06x | 43.5 s, 1.07x | 44.5 s, 1.05x | 1.4 / 2.0 / 2.5 GB | yes |
| ld_foreman (QCIF, 5 frames) | 41.8 s | 41.7 s, 1.00x | 41.7 s, 1.00x | 42.3 s, 0.99x | 1.5 / 2.0 / 2.5 GB | yes |
| ra_parkjoy (256x256, 5 frames) | 55.9 s | 52.4 s, 1.07x | 51.1 s, 1.09x | 52.8 s, 1.06x | 1.5 / 2.0 / 2.5 GB | yes |
| ra_foreman (QCIF, 9 frames) | 89.5 s | 88.4 s, 1.01x | 88.4 s, 1.01x | 91.1 s, 0.98x | 1.5 / 2.1 / 2.6 GB | yes |

## Power throttling (`--PowerThrottling`, default 0)

- Windows 11 throttles processes running in the background (EcoQoS: lower clocks, efficiency cores). An encoder started
  without a foreground window (a script, a batch job) ran up to a third slower on the 265K, and its time varied with
  where the scheduler put it. By default the encoder now asks Windows not to throttle it (SetProcessInformation,
  ProcessPowerThrottling, for this process only and only while it runs); `--PowerThrottling=1` leaves it to Windows.
  Output identical. Unpinned, one run at a time:

| test | before | now | `--PowerThrottling=1` |
|---|---|---|---|
| ld_foreman, 1 thread | 52.4 s (49-56) | 41.5 s, 1.26x | 52.1 s |
| ai_foreman, 1 thread | 39.7 s (36-44) | 32.0 s, 1.24x | 37.8 s |
| AI 33 pictures, FrameThreads=8 | 37.6 s | 32.8 s, 1.15x | |

  The times now equal those pinned to P-cores; benchmarks in this file are pinned, so they did not change.

## Notes
- GCC 16 needed two build fixes (missing `<cstdint>`; `-fno-aggressive-loop-optimizations`, because
  `EncAdaptiveLoopFilter.cpp` reads `m_chromaScaleIdxFinal[4][1]` with alternative index up to 7).
- MinGW/Win64: AVX code needs `-Wa,-muse-unaligned-vector-move` (GCC bug 54412, no 32-byte stack realignment).
- The SADL intra-NN translation units pick implementations through `__AVX2__`/`__SSE4_2__` and are kept on the
  stock ISA whatever `ECM_MARCH` says.
- Profile concentration (stock): 50 % of RA self time is spread over 37 functions, 90 % over 286 functions.
- Per-test differences of up to ~4 % can come from code layout alone (the dropped GPM step sped up all-intra
  tests, which never run GPM), so small steps are judged on the total and on the tests that run the code.
- `bench.py` reports the CPU time of the whole process tree (a Windows job object); earlier CPU numbers of
  segment-parallel runs counted only the parent process.
- The allocator has a canary build for testing (`-DFASTALLOC_CANARY=1`: guard bytes, magic words, poisoning);
  it ran the suite bit-exact without an abort.
