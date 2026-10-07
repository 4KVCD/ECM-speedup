# ECM-20.0 decoder speed-up log

Goal: 10x faster single-threaded decoding with bit-exact output (every decoded frame identical to the stock
ECM-20.0 decoder); multithreading first, then the largest single-thread costs.

Machine: i7-10700 8C/16T, Windows 11, GCC 16.2 (WinLibs), static builds with link-time optimization
(`perf/build.ps1`). Baseline: the stock ECM-20.0 decoder (`C:/Code/bin/ref`, GCC 16.2 -O3, static, no LTO).

Tests: `perf/decbench.py` decodes the streams in `C:/Code/streams` (ECM streams made with the bit-exact encoder,
several with MD5 picture-hash SEI) and compares the md5 of every output frame with the stock decoder; timings
are wall time and the CPU time of all processes, paired (both builds at the same time) unless noted. The streams
so far are short (128x128 to 720p, 1-4 s); the 1080p/4K benchmark plan is pending.

Stock decoder speed (single thread, without its ~0.29 s process start-up): 720p AI 2.4 fps, CIF RA (one GOP of
33 frames) 21.8 fps, CIF LD 29 fps, 128x128 RA 120-150 fps.

Timing set below: 720p AI (FourPeople, ParkJoy; 8 frames each), CIF RA and LD (33 frames), 128x128 RA (257 frames,
intra period 32 and 64).

## Single-thread

| # | change | output | step (CPU) | cumulative vs stock |
|---|---|---|---|---|
| 0 | code shared with the encoder already carries the encoder work (interpolation, transforms, intra prediction, ...), LTO | identical | 1.09x | 1.09x |
| 1 | `Mv` constructors constexpr: start-up no longer zero-fills the encoder's 2 x 175 MB motion caches (start-up 0.39 -> 0.19 s, peak memory 635 -> 300 MB) | identical | 1.07x (fixed ~0.2 s per process: 1.02-1.03x on the 3-4 s 720p streams, 1.10-1.18x on the 1-2 s streams) | |

## Multithreading

D1: segment-parallel decoding (`--ParallelSegments=N`): the stream is split at CRA pictures, child processes decode
the segments (a segment whose first CRA has RASL pictures starts at the previous CRA, those pictures being references
only), the outputs are concatenated. Output identical on all streams with 2, 3, 4, 8 and 16 workers. Measured one
decode at a time on the idle machine, 3 repetitions:

| stream | random-access points | sequential | 8 workers | 16 workers |
|---|---|---|---|---|
| 720p AI FourPeople, 8 frames | 8 | 2.93 s | 0.81 s (3.6x) | 0.79 s (3.7x) |
| 720p AI ParkJoy, 8 frames | 8 | 3.69 s | 1.04 s (3.55x) | 1.02 s (3.6x) |
| 128x128 RA, 257 frames, intra period 32 | 9 | 1.83 s | 0.60 s (3.0x) | 0.64 s (2.9x) |
| 128x128 RA, 257 frames, intra period 64 | 5 | 1.46 s | 0.96 s (1.5x) | 0.90 s (1.6x) |
| CIF RA, 33 frames | 2 | 1.37 s | 1.45 s (0.95x, nothing to split) | |

The CPU time of a parallel decode is about twice the sequential one: every child starts the process (~0.2 s),
parses the slice headers of the pictures before its segment, and, for RA, decodes the previous intra period's
pictures it needs as references; clocks are lower with all cores busy. Longer and larger streams amortize this.
LD streams (one intra picture) and streams with a single random-access point cannot be split; they are decoded
sequentially.

## Fixes found on the way

- Decoding that starts at a CRA in the middle of an RA stream (`-s`, random access) crashed the stock decoder:
  `Slice::xGetRefPic` returned `*end()` for reference pictures that are not in the DPB. Ported VTM commit
  76b70fb45f (nullptr for missing pictures, the RPL checks skip them).
- `IntraPrediction::ANG_REF_BUF_SIZE` (encoder step 19) was odr-used without a definition: unoptimized builds did
  not link. Now `constexpr`.

## Profile notes

- All-intra 720p: NN intra prediction (SADL MatMul) ~11 %, IntraTMP template search ~11.5 %, ALF/CCALF ~10.5 %,
  TIMD ~4.7 %, intra prediction ~4.4 %, CU/PU lookups ~3.3 %, DIMD histograms ~3 %.
- Random access (CIF): interpolation filters ~20 %, SATD of decoder-side template matching ~5 %, BDOF, OBMC and
  affine merge/template work 1-2 % each, frees of large blocks (NtFreeVirtualMemory) ~2 %.
- Low delay (CIF): interpolation ~15 %, ALF ~4 %, CU/PU lookups ~3 %, CCCM loop filter ~2 %, sign prediction 1.5 %.
- Start-up after step 1: ~0.19 s (initROM builds all transform matrices from ~700k cos/sin calls,
  createPdpFilters, AdaptiveLoopFilter::create).
