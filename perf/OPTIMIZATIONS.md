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
| 9 | cache of motion-compensation interpolation results within a picture (54 % of MC calls repeated) | yes | pending | |
| 10 | template interpolations share the MC cache (56 % of 22.6 M template calls repeated) | yes | pending | |

Code steps 4-8 together: +6.5 % on top of LTO (product of the paired ratios).

## Segment-parallel encoding (`--ParallelSegments=N`)

Chunks of whole intra periods are encoded by concurrent child processes and joined (leading IDR access unit of
each later chunk dropped, POC LSBs rebased at RBSP level). Verified byte-identical to the sequential encode:
RA 65 frames (2 chunks), AI 9 frames (4 chunks) incl. reconstruction.

Stock serial baselines for the long tests (measured under background load, CPU time is the reliable number):

| test | frames | stock wall | stock CPU |
|---|---|---|---|
| ra_long (foreman 128x128 crop, RA, QP 32) | 257 | 6011 s | 5929 s |
| ai_long (foreman 128x128 crop, AI, QP 32) | 65 | 242 s | 226 s |

## Notes
- GCC 16 needed two build fixes (missing `<cstdint>`; `-fno-aggressive-loop-optimizations`, because
  `EncAdaptiveLoopFilter.cpp` reads `m_chromaScaleIdxFinal[4][1]` with alternative index up to 7).
- MinGW/Win64: AVX code needs `-Wa,-muse-unaligned-vector-move` (GCC bug 54412, no 32-byte stack realignment).
- The SADL intra-NN translation units pick implementations through `__AVX2__`/`__SSE4_2__` and are kept on the
  stock ISA whatever `ECM_MARCH` says.
- Profile concentration (stock): 50 % of RA self time is spread over 37 functions, 90 % over 286 functions.
