# ECM-20.0 encoder speed-up log

Ground rules
- **Bit-exact**: every step must reproduce the stock ECM-20.0 bitstreams byte for byte (md5 of the `.bin`).
- Timings: `perf/bench.py run <name> --jobs 5` on an i7-10700 (8C/16T), 5 tests in parallel, total of the
  per-test wall-clock times. Run-to-run noise is about ±1.5 %.
- *Step* = speed increase vs. the previous step, *cumulative* = vs. the stock baseline
  (`t_prev / t_new - 1`, `t_base / t_new - 1`).

Benchmark suite (`perf/bench.py`)

| test | config | sequence | frames | QP |
|---|---|---|---|---|
| ai_foreman | encoder_intra_ecm.cfg | foreman 352x288 | 1 | 32 |
| ai_parkjoy | encoder_intra_ecm.cfg | ParkJoy 256x256 crop | 2 | 27 |
| ra_foreman | encoder_randomaccess_ecm.cfg | foreman 176x144 crop | 9 | 32 |
| ra_parkjoy | encoder_randomaccess_ecm.cfg | ParkJoy 256x256 crop | 5 | 37 |
| ld_foreman | encoder_lowdelay_ecm.cfg | foreman 176x144 crop | 5 | 32 |

## Results

| # | optimization | bit-exact | suite time | step | cumulative |
|---|---|---|---|---|---|
| 0 | stock ECM-20.0 (GCC 16.2 -O3, static) | - | 986.0 s | - | - |
| 1 | `-march=x86-64-v3 -ffp-contract=off` (kept as `ECM_MARCH` option, default off) | yes | 982.7 s | +0.3 % (noise) | +0.3 % |
| 2 | link-time optimization | yes | 932.8 s | +5.3 % | +5.7 % |
| 3 | profile-guided optimization on top of LTO (`perf/pgo.ps1`, trained on FourPeople crops) | yes | 877.7 s | +6.3 % | +12.3 % |

In progress (not yet measured): built-in segment-parallel encoding (`--ParallelSegments=N`, `--SegmentLength`),
`source/App/EncoderApp/SegmentParallel.cpp`. The manual chunk+parcat procedure is verified byte-identical; the
built-in mode still has to pass `runs/segpar` checks (RA 65 frames vs md5 C683AD08..., AI 9 frames vs serial).

## Notes
- GCC 16 needed two build fixes (missing `<cstdint>`; `-fno-aggressive-loop-optimizations`, because
  `EncAdaptiveLoopFilter.cpp` reads `m_chromaScaleIdxFinal[4][1]` with alternative index up to 7).
- MinGW/Win64: AVX code needs `-Wa,-muse-unaligned-vector-move` (GCC bug 54412, no 32-byte stack realignment).
- The SADL intra-NN translation units pick implementations through `__AVX2__`/`__SSE4_2__` and are kept on the
  stock ISA whatever `ECM_MARCH` says.
- Segment-parallel encoding (intra-period chunks + parcat) was verified byte-identical to a full
  65-frame RA encode (`perf/segtest.ps1`).
