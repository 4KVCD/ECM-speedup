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
| 2 | the encoder's small-object allocator (EncoderApp/FastAlloc.cpp) also replaces operator new/delete in the decoder; canary build frame-identical on all streams | identical | 1.03x (every stream 1.01-1.04x) | 1.17x (measured directly, every stream 1.10-1.31x) |
| 3 | SIMD inverse transforms skip the zero-out region (as the scalar path does) and run the DCT-II butterflies vectorized with the rounding folded in | identical | 1.00x (128-point DCT-II share in RA profiles 4.05 % -> ~2 %; within the per-test noise); encoder 1.02x | ~1.17x |
| 4 | per-picture set-up keeps the ALF picture buffers and two intra-prediction buffers instead of freeing and allocating them for every picture; everything else a fresh creation sets (TALF CTB flags, ALF flag arrays, filter shapes) is set as before | identical (also with 8 segment workers; encoder bit-exact) | 1.01x (3 paired repetitions; RA and LD streams 1.00-1.02x, AI streams 0.99-1.02x) | ~1.18x |
| 5 | sign-prediction templates (border response of every predictable coefficient, built once per process and transform type/size) computed from one first-stage transform per row and the second-stage kernel instead of one full inverse transform per coefficient position; a verification build (SIGN_PRED_TEMPLATE_VERIFY=1) compared every template with the original method and every transform function with the formula | identical (also with 8 segment workers; encoder bit-exact) | 1.03x (3 paired repetitions; RA/LD streams 1.01-1.12x, AI 1.00-1.03x; a fixed ~0.03-0.1 s per process, so it matters most for short streams and segment workers) | ~1.21x |
| 6 | OBMC sub-block predictions (mostly 4x4 luma / 2x2 chroma with neighbouring motion; ~1.3 M per 33-frame CIF RA decode) made with the same filter calls and averaging as the general xPredInterBi/Uni/Blk chain, without the chain; uncommon cases still use the chain; a verification build (OBMC_FAST_MC_VERIFY=1) ran both for every block | identical (also with 8 segment workers; encoder bit-exact) | 1.01x (3 paired repetitions; RA/LD streams 1.02-1.05x, AI 1.00x) | ~1.23x |
| 7 | profile-guided build (`perf/pgo_dec.ps1`: instrumented build, training decodes of 9 of the 18 test streams, -fprofile-use rebuild), same source as step 6 | identical (also with 8 segment workers) | 1.06x (3 paired repetitions; RA/LD 1.08-1.11x, AI 1.00-1.13x; streams not in the training set gained as much as the trained ones, e.g. ra_long 1.10x, ld_cif33 1.10x, ai720 1.13x) | ~1.31x |

| 8 | the most frequent tiny interpolations in one call each instead of a horizontal and a vertical filter call through the general dispatch: 2x2 chroma OBMC sub-blocks (6 taps) and the 4x1 / 1x4 luma template rows and columns of the sub-block template matching (4-tap TM filter); ~1.35 M such blocks per 33-frame CIF RA decode; a verification build (SMALL_MC_VERIFY=1) ran the original calls after every one and compared | identical (also with 8 segment workers; encoder bit-exact) | 1.01x CPU, 1.02x wall (3 paired repetitions; RA streams 1.04-1.06x wall, LD 1.02-1.03x, AI 1.00x) | ~1.33x |

Steps after 7 are measured against non-PGO builds (each PGO build takes ~20 min); the cumulative figure assumes the
PGO gain carries over, and is re-measured with a PGO build from time to time.

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
- After step 4 (dprof3, 3 runs per stream): RA inclusive: inter prediction (xPredInterBi/Uni) ~35 %, merge-list
  derivation with template matching (xDeriveCUMV) 31 %, OBMC 13 %, sign prediction 7.6 % (6.6 % of it the one-time
  template generation, step 5), loop filters 7.5 %. The profile is flat: half of the self time is spread over 47
  functions; the largest single one is the MC wrapper xPredInterBlk (3 %).
- Tried and dropped (no measurable gain): border-only inverse transform for the per-TU sign-prediction residual and
  the LFNST sign-prediction templates (exact, verified against xIT, but only 28-38 % of the calls qualify - the
  rest use NSPT - and those TUs are small); passing the TU to the template lambdas by reference instead of copying;
  a vectorized kernel for the hottest IntraTMP template-cost case (L-shape, requiredTemplate 0: ~95 % of ~7 M calls
  per 8-frame 720p AI decode), verified equal to the general code (full sums, and the early-stop contract), first
  without and then with the same early stops: 1.00x - the search stops most positions after the first template row,
  where the general code is already cheap. (Found on the way: calcDiffExtraction(Delta)4 sum only lanes 0-2, so the
  column next to the block weighs 8, not 9.)
- Tried and dropped: intra-NN matrix products (SADL, ~15 % of 720p AI decoding) with the horizontal sums, shifts and
  saturation of 8 outputs done at once and the dense output layer loading each input vector once for 8 outputs;
  verified equal on every call, but 1.00x: each model has 0.4-0.6 MB of int16 weights (full 16-bit range, so no exact
  narrower storage), more than the 256 KB L2, so an inference is bound by streaming the weights from L3.
- Interpolation work repeated within a CU (same reference samples, filter and size; CIF RA): 20 % of the horizontal
  first-pass and 11 % of the vertical filter work, so a cache could save at most ~3-4 % of the decoding time. The
  filters are called ~6 M times per 33-frame CIF RA decode, mostly for 1-4 sample wide templates, 4x4 luma and 2x2
  chroma OBMC blocks; the call chain (xPredInterBlk / xGetPredBlkTpl -> filterHor/filterVer -> simdFilter set-up)
  costs ~100 cycles per call.
- MC call counts (CIF RA, 33 frames, 2981 inter CUs): ~1040 motion-compensated predictions per inter CU, mostly
  tiny: 953k 4x1/1x4 sub-block templates, 826k 2x2 chroma and 478k 4x4 luma OBMC predictions (OBMC runs per 4x4
  sub-block wherever the motion inside a CU changes), 294k ARMC block templates; per-call overhead rather than
  filtering dominates these.
