# ECM-20.0 encoder speed-up log

Ground rules
- **Bit-exact**: every step must reproduce the stock ECM-20.0 bitstreams byte for byte (md5 of the `.bin`).
- *Step* = speed increase vs. the previous step, *cumulative* = vs. the stock baseline
  (`t_prev / t_new - 1`, `t_base / t_new - 1`).
- Machines: steps 1-43 on an i7-10700 (8C/16T), steps 44 and later on a Core Ultra 7 265K (8 performance and 12
  efficiency cores, measured on the performance cores); Windows 11, GCC 16.2 (WinLibs), static builds.
- Setup, workflow and what to try next: `perf/HANDOFF.md`.

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

### Steps 24-43 (i7-10700): dependent quantization and the loop-filter encoders

Paired against the previous step on LTO builds; "CPU" is the 5-test CPU-time ratio. The DepQuant steps also give the
cycles spent in `DepQuant::quant` (paired short encodes); their product is about 1.7x.

| # | optimization | commit | step |
|---|---|---|---|
| 24 | DepQuant: trellis states point to the sig/gtx rate tables instead of copying them | ea22f995 | 1.00x |
| 25 | DepQuant: one decision per state and trellis stage, skipped sub-blocks resolved in the backtracking | 7aa76060 | 1.00x |
| 26 | DepQuant: per-TU rate setup reads the context models through the concrete CtxStore | b37ce617 | 1.01x (quant, steps 24-26: 1.06x) |
| 27 | DepQuant: sub-block levels kept as one 16-byte vector, outside-template init shared per sub-block | 5a8aaadd | 1.02x (quant 1.19x) |
| 28 | DepQuant: branch-free decision per trellis state | 2dfd1992 | 1.00x (quant 1.015x) |
| 29 | DepQuant: trellis loop templated on the number of states, decision and update inlined | 2babb37f | 1.00x (quant 1.03x) |
| 30 | DepQuant: trellis states as a structure of arrays, the 8 state updates of a sub-block together | 9121b57a | 1.01x (quant 1.14x) |
| 31 | DepQuant: decision templated on the scan position type, branch-free context-coded level bits | 838e5d3a | 1.01x (quant 1.07x) |
| 32 | DepQuant: sig/gtx/sub-block rate tables reused while their context models are unchanged | e374a08a | noise (quant 1.055x) |
| 33 | ALF clipping-search results reused within a picture for bit-identical inputs | 405e9e95 | +0.5 % wall, CPU 1.00x |
| 34 | ALF clipping search solves all trials of a pass at once in AVX2 lanes | 106f722e | +2.2 % wall, CPU 1.02x |
| 35 | ALF class merging with compact bin-0 statistics and batched pair solves | 7df0b34e | +0.7 % wall, CPU 1.01x |
| 36 | CC-ALF training reuses filter derivations and CTU distortions for identical inputs | eb87d76b | +0.6 % wall, CPU 1.01x |
| 37 | ALF block statistics class by class with a register-tiled AVX2 kernel | 08d3d948 | +1.7 % wall, CPU 1.01x |
| 38 | CCSAO edge statistics in compact per-pass histograms | 968fa098 | +0.2 % wall, CPU 1.00x |
| 39 | ALF class merging solves singular pair systems with the regularized lanes solve | 030a9e81 | +0.8 % wall, CPU 1.01x |
| 40 | CCSAO band statistics of unchanged sets not recomputed during the parameter search | 20106d8d | +0.7 % wall, CPU 1.01x |
| 41 | ALF coefficient quantization searches use the gathered covariance of their clipping | 8a33ab0f | 1.00x |
| 42 | ALF clipping search batches reuse the shared factorization prefix of kE | 6acabdee | CPU 1.01x |
| 43 | covariance tile kernel keeps its 8 accumulators in registers | 53fee4e2 | +0.7 % wall, CPU 1.00x |

### Steps 44-64 (Core Ultra 7 265K)

Paired on LTO builds with `-DECM_MARCH=arrowlake`: `perf/bench.py ab A B --reps 4 --cpus 0,1,6,7,8,9`. Each pair of
performance cores runs one test with both builds at the same time, and the cores are swapped every repetition. The
step is the time-weighted median of the per-repetition CPU ratios. A/A noise: ±0.8 % per test, ±0.3 % total. Steps
named with a `*_CHECK` macro were first run as a check build that compares the old and the new result on every call.

| # | optimization | commit | step |
|---|---|---|---|
| 44 | build: `-march=arrowlake` (`ECM_MARCH`) | - | +3 % |
| 45 | intra luma mode bits: the mode flags coded once per CU state and replayed (MODEBITS_CHECK) | 15b98db3, 611cf47b | +0.3 % (AI +1.2 %) |
| 46 | GPM split-mode tables (1.6 MB per CU) cleared row by row on first use | b5a45457 | +1.0 % |
| 47 | pixel-level affine MC: 1-2 wide sub-blocks filtered in one fused 12-tap 2-D call (AFFINE_NARROW_CHECK) | 6fcd5f57 | +2.4 % |
| 48 | InterPrediction clips MVs with an inlined clipMvInPic instead of the clipMv function pointer | 63dc9b0e | +0.5 % |
| 49 | SIMD inverse transforms skip zero coefficients | 9f898094 | +0.7 % |
| 50 | intra mode lists copy ModeInfo as plain memory (defaulted assignment) | ca587554 | +0.5 % |
| 51 | multi-hypothesis search orders its results through an index array | 6562da2e | +0.8 % |
| 52 | GPM split-mode ranking counts ranks without branches (GEOSORT_CHECK) | 23e178b2 | +1.0 % |
| 53 | sign prediction orders its coefficient lists without std::stable_sort (SIGNSORT_CHECK) | 8b280616 | +0.3 % |
| 54 | DIMD gradient histograms find the angle index without a search loop (DIMDANG_CHECK) | e886603f | +0.7 % |
| 55 | long-tap angular interpolation, 8 samples per step in SSE4.1 | 619216e4 | +0.3 % |
| 56 | DIMD gradient histogram, 8 pixels per step in AVX2 | e56f1fb5 | +0.2 % |
| 57 | PGO trained on FourPeople and CrowdRun (`perf/pgo.ps1`) | 8a81c706 | PGO +3.8 % |
| 58 | DepQuant rate-table reuse checks the context block stamps first | b18ffb02 | +0.2 % |
| 59 | CCCM: 7-tap model applied and statistics summed in AVX2 (CCCMAPPLY_CHECK) | 2b37bc4e | +0.4 % |
| 60 | GPM and multi-hypothesis searches build only the buffer views they use | c17d52d7 | +0.5 % |
| 61 | PDP intra prediction loads each output pair's coefficients with one AVX2 load (PDPOPT_CHECK) | a9bc3137 | +0.4 % |
| 62 | CCCM model parameters in a fixed-capacity vector instead of std::vector | 404c0296 | +0.5 % |
| 63 | intra TMP linear-filter prediction through the CCCM AVX2 helper | 7a9d0401 | +0.2 % (AI +0.8 %) |
| 64 | GPM rank counts in 64-bit AVX2 lanes | 49fa654f | +0.5 % (LD +1.1 %) |

bd1be75c gates the new SIMD paths like the stock SIMD code and fixes builds other than the default one; the default
build does not change. The other commits on the 265K are tooling (`bench.py` pinning and pairing bcfcdc96) and
documentation.

End to end, stock -> 49fa654f + PGO, paired, 4 repetitions (the encoder code of b8e5f2ba is the same):

| test | stock | ECM-speedup | wall | CPU (median ratio) |
|---|---|---|---|---|
| ai_foreman | 49.5 s | 34.7 s | 1.42x | 1.436x |
| ai_parkjoy | 74.6 s | 52.0 s | 1.43x | 1.440x |
| ld_foreman | 72.5 s | 45.2 s | 1.60x | 1.604x |
| ra_foreman | 144.9 s | 95.4 s | 1.52x | 1.523x |
| ra_parkjoy | 92.3 s | 60.0 s | 1.54x | 1.533x |
| total | 433.8 s | 287.4 s | 1.51x | **1.512x** |

Dropped on the 265K (paired, bit-exact unless noted):
- mode-bits memo extended to TMP/MIP/TIMD/EIP/SGPM/TMRL modes (0.997x)
- bypass-sign counting in codePredictedSigns (1.001x)
- CCCM training-sample gather in AVX2 with compaction (1.000x)
- sign-prediction cost loop in AVX2 (1.001x)
- BDOF parameter sums via lane permutes (0.999x / 0.997x)
- TMP joint SAD two rows per step (0.993x)
- bilateral filter small blocks packed per AVX2 step (1.001x)
- MC filter-section result cache, exact with 70 M hits checked (0.972x, RA foreman 0.937x)
- xGetPredBlkTpl result cache (-3 %)
- DepQuant result cache (10 % repeats)
- TM-refinement cache (its key misses inputs: not exact)
- AVX2 DepQuant decision, AVX2 inverse transform on top of the zero skip, GEO std::sort of packed keys, inlined
  buffer accessors, a neighbour-availability shortcut (no gain)
- `-fipa-pta` (0.994x), `-flto-partition=one` (1.002x), `-funroll-loops` (-2 %), `-O2` (-5 %)

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

On the Core Ultra 7 265K (PGO builds; every md5 equals the sequential stock encode):

| test | stock, 1 core | ECM-speedup, 1 core | 8 workers | 20 workers |
|---|---|---|---|---|
| ai_long | 311.2 s | 210.7 s | 34.9 s (8.9x) | 18.4 s (17.0x) |
| ra_long | 2647.9 s | 1842.8 s | 451.6 s (5.9x) | - (8 intra periods) |

- Chunk order (f19e21c3): with more chunks than workers, chunks start in order of decreasing luma gradient activity
  of their input pictures. For ai_long, that activity correlates 0.95 with the per-picture encoding time.
  - ai_long, 20 workers: 24.6 -> 18.4 s (1.34x).
  - ai_long, 8 workers: 37.1 -> 34.9 s (1.06x).
- Dropped: workers pinned by core type, with performance cores taking the expensive end of the order and efficiency
  cores the cheap end. It was slower (22.9 / 20.3 s): a chunk pinned to the efficiency cores cannot move to a
  performance core that has gone idle.
- An efficiency core runs this encoder at 0.76x of a performance core (ai_foreman pinned: 45.7 s vs 34.7 s).
- Parallel work inside a picture:
  - A thread pool for the intra TMP sparse search was bit-exact (149 K searches compared with the sequential scan)
    but 0 % faster (ai_foreman 42.0 s with 1, 4 and 8 threads). The scan is about 2 % of the time, and busy-waiting
    workers lower the main core's turbo.
  - CTUs are serial: each mode search starts from the exact CABAC state and reconstruction of the previous CTU.
- Frame-level parallelism inside an intra period is limited by the temporal CABAC store, which chains the pictures of
  each temporal layer and QP, and by the per-layer AMVP-SbTMVP statistics. `perf/ra_critical_path.py` on the serial
  ra_long log gives a bound of 1.27-1.80x per intra period. Low delay is fully serial.

## Decoder

Stock DecoderApp vs the PGO build of f19e21c3, measured with `perf/decbench.py`: both decoders run at the same time
on two performance cores, swapped every repetition. Times are the decoder's own timer, which includes writing the
YUV. Every decoded YUV md5 is identical; 13 streams were checked in total.

| stream | content | runs | stock | ECM-speedup | speedup |
|---|---|---|---|---|---|
| ra_cif | RA 352x288, 289 frames, QP 32 | 6 | 7.420 s | 6.061 s | 1.22x |
| ld_cif | LD 352x288, 33 frames, QP 32 | 6 | 0.700 s | 0.589 s | 1.19x |
| ra_long | RA 128x128, 257 frames, QP 32 | 20 | 1.166 s | 0.899 s | 1.30x |
| ra_97 | RA 128x128, 97 frames, QP 32 | 6 | 0.608 s | 0.487 s | 1.25x |

The decoder gains come from the shared CommonLib code and the build. PGO is trained on encoder runs only, and the
decoder has not been profiled on its own.

## Notes
- GCC 16 needed two build fixes (missing `<cstdint>`; `-fno-aggressive-loop-optimizations`, because
  `EncAdaptiveLoopFilter.cpp` reads `m_chromaScaleIdxFinal[4][1]` with alternative index up to 7).
- MinGW/Win64: AVX code needs `-Wa,-muse-unaligned-vector-move` (GCC bug 54412, no 32-byte stack realignment).
- The SADL intra-NN translation units pick implementations through `__AVX2__`/`__SSE4_2__` and are kept on the
  stock ISA whatever `ECM_MARCH` says.
- Profile concentration (stock): 50 % of RA self time is spread over 37 functions, 90 % over 286 functions.
  At 49fa654f (all 5 tests): 25 % of self time in 12 functions, 50 % in 47, 90 % in 331.
- Stock ECM-20.0 does not compile with `ENABLE_HIGH_BITDEPTH=ON` or with `JVET_AG0061_INTER_LFNST_NSPT` set to 0;
  bd1be75c keeps the new code ready for both.
