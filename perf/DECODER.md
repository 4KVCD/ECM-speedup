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

| 9 | the two 13x13 ALF luma kernels (256-bit paths) keep the class-dependent coefficient and clipping vectors of each 16-sample segment, which depend only on its 16 class values and the filter set, and reuse them when the next row has the same class values (the classification is per 2x2 block; 77 % of the segments of 720p RA hit, 65 % of 720p AI); a verification build (ALF_PARAM_CACHE_VERIFY=1) rebuilt them for every segment and compared | identical (also with 8 segment workers; encoder bit-exact) | 1.02x (3 paired repetitions; 720p streams 1.02-1.03x, CIF and smaller 1.00-1.02x) | ~1.36x |

Timing set from step 9 on also includes ra720_pj17 (720p ParkJoy RA, 17 frames; stock decoder 6.00 s).

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

D2: frame-level threading (`--FrameThreads=N`): the main thread parses the high-level syntax, manages the picture
buffers and the output, and hands each picture (once its slices are parsed) to one of N worker contexts (each a
DecLib with its own prediction, transform, CABAC and loop-filter objects). A worker waits for the reference pictures
it uses and for the decode-order state it reads (temporal CABAC contexts, AMVP-SbTMVP area, CCSAO / chroma ALF /
CCALF CTU control, ALF residual scales: explicit slots, commit e44a65e1), then decodes the CTUs, runs the loop
filters and pads the picture. Pictures with more than one slice and tools that change other pictures while a picture
is decoded (long-term references, subpictures, GDR, NN loop filter, multi-layer) are decoded by the main thread as
before. The POC lines and the output keep the sequential order. Output identical on all 19 streams sequentially,
with one picture at a time on 8 contexts (`--FrameThreadsSerial=1`) and with 2, 4, 8 and 16 threads (3 repetitions
each, the 4 random-access streams with the most overlap 20 more times).

The first parallel runs differed on a few RA streams in about half of the runs: RdCost::xGetMRHADs (mean-removed
Hadamard, used by the DMVR bilateral search) works on a copy of the block in the global `orgCopy`, which two threads
overwrote (ECM marks it threadprivate for OpenMP builds only); now thread_local, with the other statics and
lazily created tables (commit 14f272bb). Found by locating the first diverging prediction unit (per-PU motion and
per-CU reconstruction dumps of parallel vs sequential decodes) and then the first diverging bilateral-search cost.

Single thread, LTO builds, 5 paired repetitions: the thread-safety changes 0.99x (per stream 0.95-1.07x, noise),
the threaded decoder run sequentially 0.99x wall, 1.00x CPU.

Wall time, one decode at a time on the idle machine, best of 3 (LTO, no PGO):

| stream | sequential | 2 threads | 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|---|
| 720p AI FourPeople, 8 frames | 2.78 s | 1.57 s (1.77x) | 1.06 s (2.62x) | 0.93 s (2.99x) | 1.07 s (2.60x) |
| 720p AI ParkJoy, 8 frames | 3.59 s | 2.08 s (1.73x) | 1.37 s (2.62x) | 1.19 s (3.02x) | 1.32 s (2.72x) |
| 720p RA ParkJoy, 17 frames | 4.70 s | 4.05 s (1.16x) | 3.61 s (1.30x) | 3.49 s (1.35x) | 3.75 s (1.25x) |
| CIF RA, 33 frames | 1.25 s | 1.06 s (1.18x) | 0.98 s (1.28x) | 0.96 s (1.30x) | 1.15 s (1.09x) |
| 128x128 RA, 257 frames, intra period 32 | 1.59 s | 1.36 s (1.17x) | 1.13 s (1.41x) | 1.00 s (1.59x) | 1.06 s (1.50x) |
| 128x128 RA, 257 frames, intra period 64 | 1.22 s | 1.03 s (1.18x) | 0.89 s (1.37x) | 0.85 s (1.44x) | 0.97 s (1.26x) |
| CIF LD, 33 frames | 0.98 s | 1.00 s (0.98x) | 1.07 s (0.92x) | 1.19 s (0.82x) | 1.43 s (0.69x) |
| 1-frame streams (AI 128x128 - 352x288) | 0.12-0.13 s | +0.04 s | +0.08 s | +0.15 s | +0.30 s |
| all 19 streams | 19.30 s | 15.69 s (1.23x) | 14.11 s (1.37x) | 14.60 s (1.32x) | 17.82 s (1.08x) |

Limits of this first version: every worker context is created at start-up (~20 ms each, which dominates short
streams), a worker takes the oldest queued picture and waits for its references, and a picture waits for its
reference pictures to be completely decoded and filtered, so RA gains are bounded by the dependency chains and LD
(each picture references the previous one) only pays the overhead. CPU time with 8 threads is 1.40x the sequential
one (start-up, lower clocks with all cores busy).

## GPU (Vulkan)

Bit-exact GPU work is limited to what does not depend on CTU-by-CTU parsing and reconstruction: the in-loop
filters and picture-level steps. `source/Lib/GpuLib` (built with `-DECM_VULKAN=ON -DECM_VULKAN_HEADERS=<Vulkan-Headers
include dir> -DECM_GLSLANG=<glslang.exe>`; default off) is a small Vulkan compute runtime (`VkCompute`: the loader
`vulkan-1.dll` / `libvulkan.so.1` is opened at run time, buffers are addressed through buffer device addresses passed in
push constants, one command stream per decoder context) and the kernels, GLSL compiled to embedded SPIR-V at build
time. `--GpuLoopFilter=1` filters on the GPU, `=2` filters on the CPU and compares every GPU intermediate result and
the output with it; without a usable device the CPU filters.

G1: ALF (luma with the fixed filters, classifiers, Gaussian / Laplacian / residual inputs and residual scaling;
chroma ALF with its fixed filter; CCALF; chroma residual scaling) on the GPU. The CPU filters CTU by CTU and derives
the intermediate results of each CTU and its surroundings (classes, fixed-filter outputs, Gaussian and Laplacian
images); these depend only on the sample position (the CPU recomputes them per CTU with the same arithmetic; the
picture boundary is mirror-padded), so the kernels derive each one for the whole picture and then filter every CTU.
The tap geometries and coefficient orders are extracted from the C++ by `GpuLib/tools/gen_alf_glsl.py`. The CPU keeps
ALFProcess's state work (APS reconstruction, residual-scale state, CTU-control reuse, the copies later stages read).
Checked first that the scalar C filters give the same output as the SIMD ones (all 19 streams), so the kernels follow
the C code. Pictures that use what the kernels do not implement (virtual boundaries, several slices / tiles /
sub-pictures, CU QP classification, other filter types or chroma formats) take the CPU path.

Output identical on all 19 streams with `--GpuLoopFilter=1`, with 8 frame threads, and the verification mode found
no difference in any intermediate result (779 pictures); encoder suite bit-exact (CommonLib: the hook only).
ALF of a 720p picture: ~46 ms on the CPU, ~4.5 ms with the GPU (RTX 3070: ~2.6 ms GPU time, the rest the
copies to the upload buffer, the state work and the read-back). Creating the Vulkan device costs ~0.3 s (instance
0.17 s, device 0.11 s; implicit layers are not the cause), so it runs in a background thread started at the first
SPS with pictures of at least 640x360 (smaller pictures stay on the CPU: the fixed costs exceed what the GPU saves),
and pictures filtered before it is ready take the CPU path.

LTO builds, paired, 3 repetitions, single thread: 720p AI FourPeople 1.10x, ParkJoy 1.09x, 720p RA ParkJoy 1.06x,
the 1- and 3-picture 720p streams 0.95x / 0.97x (device set-up and tear-down), smaller streams unchanged; all 19
streams 1.04x. With 8 frame threads 0.94-1.03x: every worker context allocates its buffers at its first picture.

G2: the decoder contexts share a pool of at most two GPU filter instances (stream and buffers; a picture holds one
for a few ms), the first one allocated for the stream's picture size by the background thread; the device, kernels
and pool live until the process ends (destroying the device at exit took ~45 ms). Device memory at 720p: ~26 MB per
instance, so at most 52 MB for all decoder contexts instead of 26 MB per context (210 MB with 8 frame threads).
Single thread: 720p AI 1.11x / 1.09x, RA ParkJoy 1.06x, ra720 (3 pictures) 1.01x, ai720 (1 picture) 0.98x, all 19
streams 1.05x; 8 frame threads 1.00x / 1.01x / 1.04x, 0.97x / 0.96x on the 3- and 1-picture streams: on these short
streams creating the device (~0.3 s of a core) costs about what the GPU saves (~40 ms of CPU per 720p picture).

G3: LF-CCCM (JVET-AL0153: per window of a CTU a least-squares cross-component model of the chroma, derived from the
picture before ALF and added to the chroma after ALF) on the GPU, from the picture the ALF kernels upload. One kernel
derives the downsampled luma of the whole picture (the CPU's per-CTU buffer depends only on the position; its padding
is the boundary value); one workgroup per window computes the statistics (exact integer sums, so in any order), runs
the Gaussian elimination with the CPU's division-free fixed-point arithmetic (one row per invocation), applies the
model and takes the CPU's fallbacks for failed models; the ALF chroma kernel adds the correction. The CPU writes the
outputs of a failed model up to its failing sample and then overwrites them with the reconstruction, so only the final
outcome of each class matters. Reproduced: the 16-bit wrap of the CPU's Pel accumulators (chroma-offset sums,
sub-class threshold sums), the regression window that extends one sample further right / down at the CTU's left / top
boundary, the regularization with the CTU's nonlinear-term flag when a smaller model replaces the CTU's. Pictures
whose ALF takes the CPU path derive the correction on the CPU (the hook calls the CPU code before ALF).

Verification: 779 pictures, 171 with LF-CCCM (CIF to 720p, odd sizes), no difference; output identical on all 19
streams, also with 8 frame threads; the CPU-only build too. 720p RA ParkJoy (LF-CCCM on 16 of 17 pictures, 14144
windows of 4x4 each): ~17.4 ms per picture on the CPU, 0.7 ms on the GPU at full clock (up to ~3.8 ms when the GPU
has lowered its clock between pictures); +1.4 MB device memory per instance. Single thread vs the CPU-only build: RA
ParkJoy 1.13x (G2: 1.06x), AI streams unchanged (no LF-CCCM), all 19 streams 1.06x; 8 frame threads: RA ParkJoy
1.09x (1.04x), the six larger streams 1.05x.

G4: SAO, the bilateral filter and CCSAO (with their joint clipping) on the GPU. The decoder defers them to the ALF
hook, which runs them from the deblocked picture (uploaded instead of the picture after SAO), so they share ALF's
upload and submission; when the picture's ALF takes the CPU path, the hook runs them on the CPU first. Kernels: the
bilateral filter per transform block (one workgroup: the strength from the block's mean absolute deviation, then the
5x5 diamond with the CPU's padding of the block: rows / columns beyond the picture replaced, and the row one sample
beyond a CTU's top / bottom read instead of the second; the C filter equals the SIMD one, checked on all 19
streams); SAO + CCSAO + the bilateral offsets + clipping per sample (the samples the CPU's loops reach: the CTU's
availability flags with each class position's own rows and columns; two ECM quirks reproduced: the band classifier's
below-left position reads chroma row (y + 1) >> 1 for luma samples, its above position reads luma and chroma one row
higher for chroma samples in the top CTU row); then the mirror padding ALF reads. The CPU keeps the parameter work
(SAO merges, the transform blocks the bilateral filter applies to) and gets the luma after SAO back when TALF or the
offset refinement after ALF reads it. The streams use luma SAO of every type, the bilateral filter, CCSAO edge
classes on luma and some band positions; a stress test (TEMP: random SAO and CCSAO parameters of every kind per CTU
on intra pictures) found no difference either, and reports both quirks when they are left out.

Verification: 779 pictures, all with SAO on the GPU, no difference; output identical on all 19 streams, with 8 frame
threads, and from the CPU-only build; encoder suite bit-exact (CommonLib: a friend declaration). +5.5 MB device
memory per instance at 720p (deblocked planes, bilateral offsets). Single thread vs the CPU-only build: AI
FourPeople 1.16x (G3: 1.11x), AI ParkJoy 1.12x (1.09x), RA ParkJoy 1.15x (1.13x), all 19 streams 1.08x (1.06x);
8 frame threads: AI ParkJoy 1.07x, RA ParkJoy 1.09x, AI FourPeople 0.98x (its 8 pictures finish together and wait
for the two pool instances), the six larger streams 1.05x; CPU time 1.06-1.16x less on the 720p streams.

G5: the offset refinement after deblocking (AdaptiveLoopFilter::calcOffsetRefinement stage 0, every CTU; C only) is
deferred with SAO (unless the NN loop filter reads the picture in between) and is the first GPU stage: per luma
sample a log2-table offset from the deblocking's change, with the thresholds at coding-block boundaries. Its input
planes were already uploaded for ALF. The streams use refinement index 1; a stress test (TEMP: the refinement on every
picture, index 0 / 1 alternating) found no difference and reports a wrong index-0 threshold. pj720_ai and ra720_pj17
use it on most pictures (~2.2-2.6 ms on the CPU): single thread 1.14x (G4: 1.12x) and 1.17x (1.15x); all 19 streams
1.08x.

G6: the LF-CCCM kernel runs with 32-invocation workgroups instead of 64 (one subgroup on NVIDIA: cheaper barriers
between its many steps, fewer idle invocations on the 4x4 windows): GPU time of the phase with the fixed filters and
LF-CCCM 2.34 -> 1.88 ms per 720p RA ParkJoy picture; wall time unchanged (below the noise). The bilateral filter
kernel stays at 64: with 32 it took 0.70 -> 1.28 ms per 720p AI FourPeople picture (Ampere runs at most 16 workgroups
per SM, so 32-invocation groups halve its occupancy). Only raodd and ra_long use the 10- and 11-parameter models;
a stress test (TEMP: random LF-CCCM parameters per CTU) covers them on all 9 streams with LF-CCCM. Verification:
779 pictures, no difference; all 19 streams identical, also with 8 frame threads. LTO builds vs the CPU-only build,
single thread: AI FourPeople 1.16x, AI ParkJoy 1.14x, RA ParkJoy 1.16x, all 19 streams 1.09x; vs the stock decoder
(paired, 3 repetitions) AI FourPeople 1.32x, AI ParkJoy 1.32x, RA ParkJoy 1.47x, all 19 streams 1.47x wall time and
1.42x CPU time (the short streams gain most from the faster start-up).

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
- Tried and dropped: a 12-tap horizontal interpolation kernel without shuffles (even and odd output columns from plain
  loads and pmaddwd, ~2 shuffles instead of ~15 per 8 outputs), verified equal on every call: 1.00-1.01x (6 paired
  repetitions on the RA/LD streams), although the replaced kernel had 2.4 % of the 720p RA profile; the filters seem
  to wait on the reference samples rather than on the shuffle port.
- Tried and dropped: a per-CU cache of the merge-candidate template predictions (getBlkAMLRefTemplate, keyed by
  reference picture, MV, size, bi and filter-selecting flags; LIC/GPM/IBC/OBMC cases excluded), verified equal on
  every hit: 19-25 % hits but 0.97-0.99x, because one template prediction costs only ~600 cycles and the lookup and
  copies cost about as much as the hits save.
- Tried and dropped: IntraTMP search with the first template row's SAD checked before the cost call (exact: that row is
  a lower bound of the cost): 1.00x - the search loop's own time was only ~1.9 % of 720p AI decoding (6 % was its
  inclusive time).
- Tried and dropped (measured with time-stamp counters around the sections, ra720_pj17): the CCCM loop filter's sample
  collection without branches (lfCccmSumStatisticsDecoder, 230 -> 152 M ticks) and its regression sums four at a time
  (171 -> 193 M ticks, slower); net 0.4 % of the decoding time, below what paired wall-clock runs resolve. The whole
  lfCccmFiltersConvolution is ~740 M ticks (~5 % of the decode).
- Interpolation work repeated within a CU (same reference samples, filter and size; CIF RA): 20 % of the horizontal
  first-pass and 11 % of the vertical filter work, so a cache could save at most ~3-4 % of the decoding time. The
  filters are called ~6 M times per 33-frame CIF RA decode, mostly for 1-4 sample wide templates, 4x4 luma and 2x2
  chroma OBMC blocks; the call chain (xPredInterBlk / xGetPredBlkTpl -> filterHor/filterVer -> simdFilter set-up)
  costs ~100 cycles per call.
- MC call counts (CIF RA, 33 frames, 2981 inter CUs): ~1040 motion-compensated predictions per inter CU, mostly
  tiny: 953k 4x1/1x4 sub-block templates, 826k 2x2 chroma and 478k 4x4 luma OBMC predictions (OBMC runs per 4x4
  sub-block wherever the motion inside a CU changes), 294k ARMC block templates; per-call overhead rather than
  filtering dominates these.
