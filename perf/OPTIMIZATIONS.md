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

## Notes
- GCC 16 needed two build fixes (missing `<cstdint>`; `-fno-aggressive-loop-optimizations`, because
  `EncAdaptiveLoopFilter.cpp` reads `m_chromaScaleIdxFinal[4][1]` with alternative index up to 7).
- MinGW/Win64: AVX code needs `-Wa,-muse-unaligned-vector-move` (GCC bug 54412, no 32-byte stack realignment).
- The SADL intra-NN translation units pick implementations through `__AVX2__`/`__SSE4_2__` and are kept on the
  stock ISA whatever `ECM_MARCH` says.
- Profile concentration (stock): 50 % of RA self time is spread over 37 functions, 90 % over 286 functions.
