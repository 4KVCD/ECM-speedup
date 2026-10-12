# ECM-20.0 decoder speed-up log

Goal: 10x faster single-threaded decoding with bit-exact output (every decoded frame identical to the stock
ECM-20.0 decoder); multithreading first, then the largest single-thread costs.

Machine: i7-10700 8C/16T, Windows 11, GCC 16.2 (WinLibs), static builds with link-time optimization
(`perf/build.ps1`). Baseline: the stock ECM-20.0 decoder (`C:/Code/ECM/bin/ref`, GCC 16.2 -O3, static, no LTO).

Tests: `perf/decbench.py` decodes the streams in `C:/Code/ECM/streams` (ECM streams made with the bit-exact encoder,
several with MD5 picture-hash SEI) and compares the md5 of every output frame with the stock decoder; timings
are wall time and the CPU time of all processes, paired (both builds at the same time) unless noted. The streams
so far are short (128x128 to 720p, 1-4 s; the GPU test streams also three 1080p ones of 2 pictures); the 1080p/4K
benchmark plan is pending.

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

Checks before the first push of this work (2026-10-08): a review of every commit; all 19 streams on the CPU-only build,
the GPU build (on, verification, 8 and 16 frame threads, one picture at a time on 8 contexts, 8 segment workers,
verification with 8 frame threads) and a build with libstdc++ assertions; 8- and 12-bit streams and CTU sizes 64 and
256 in verification mode (identical to the stock decoder; ECM-20.0 itself crashes on the CTU-64 AI stream, the stock
decoder as well); the CPU fallback without a usable device; the encoder suite, also in its decode-bitstream mode.
The review of the threading and segment code found nine defects that only rare inputs reach; each fix is part of the
commit that caused it:
- frame threads: a second PPS of an SPS that changed kept PreCalcValues of the old SPS (now per SPS generation);
  after an error the workers decoded the queued pictures while main() freed the PDP / MIP tables, and a failed picture
  could be written before the error was reported; pictures waiting to be written could read a replaced SPS / PPS
  (they are written before a parameter set NAL unit is decoded);
- segments: a prefix SEI / APS between two slices of a picture became the start of the next picture unit (a segment
  could end in the middle of a picture); temporary file names were the same for concurrent decodes of a stream and
  were left behind when writing one failed; without -d children could write different bit depths (streams whose
  SPSs differ are no longer split then); children did not reset the POC MSB of the stream's first CRA; a CRA whose
  leading pictures are RADL was a direct split point although ECM resets its decode-order state only at the first
  trailing picture;
- the encoder's decode-bitstream mode read CCSAO CTU control from the shared entries, which the decoder no longer
  filled (it does again when it does not decode in threads).

### GPU reconstruction (first stage of the full GPU decoder, branch dec-gpurecon)

Groundwork (experiments on branch dec-parsefirst):
- parsing needs nothing from the reconstruction: parsing a whole slice first and reconstructing afterwards is
  bit-exact on all 19 streams (a canary that skips one CTU breaks every stream). The CPU can parse, the GPU reconstruct.
- reconstruction is serial within a picture in CTU raster order as soon as it has intra or IBC CUs: with
  JVET_AJ0172 the IBC / IntraTMP reference area is everything above the CTU row, and the IntraTMP merge list and the
  BV-guided intra tools follow such vectors; non-adjacent merge candidates reach 8x the CU width. Parallelism comes
  from independent blocks (components, distant parts of the picture), from pictures decoded at once and from the
  samples of a block.
- a dependent per-CU decision chain (merge-list-like code, microbenchmark) runs 7.2x slower on one GPU workgroup than
  on one CPU core; the GPU only beats 16 CPU threads with ~50 independent chains.

Implemented (`--GpuRecon=2` verifies: the CPU reconstructs, the GPU result is compared per block and per sample;
`--GpuRecon=1` reconstructs on the GPU: parse-first, a derive-only pass of decompressCtu, GPU picture written into
the reconstruction buffer; works in every frame-thread context):
- tool levels so far (test streams C:\Code\ECM\streams_gpu, encoded with the ECM tools off): M1 planar / directional
  planar / DC / angular with the 16-, 10-tap cubic and 6-tap Gaussian luma and 2-tap chroma interpolation, DCT-II,
  scalar dequantization, adaptive luma clipping; M2 adds reference filtering, PDPC (planar / DC, pure H/V,
  gradient, inverse angle) and intra fusion; M3 adds the intra prediction bilateral filter (JVET-AK0118: the angular
  reference texture gate, the prediction texture gate and strength, the 5x5 diamond over the prediction padded with
  the neighbouring reconstruction as the CPU copies it); M4 adds the dual tree of intra slices (separate luma and
  chroma coding trees): nothing in the kernels changed, because the host records the transform blocks in decoding
  order and the owners and availability are per component; M5 adds LMCS (luma mapping with chroma scaling) on intra
  pictures: the reconstruction stays in the mapped domain, the luma clipping range is mapped, the intra bilateral
  filter works on inverse-mapped prediction and padding samples and maps its output back (also where its texture
  gate skips the filter), and the chroma residual is scaled by the rounded mean of the luma column left of and the
  row above the CU (Reshape::calculateChromaAdjVpduNei with LMCS_CHROMA_CALC_CU; the host records the CU's luma area
  and which neighbours exist, the GPU averages, and the chroma block depends on the luma blocks it reads); the host
  uploads the picture's forward and inverse LUTs and the chroma scale of every luma value; M6 adds dependent
  quantization (4 and 8 states) and QCS (quantization center shift): the host walks the quantizer states down each
  block's scan (they depend on the levels' parities only) and records quantization indices instead of levels;
  intra_resi dequantizes them like DepQuant's dequantBlock (64-bit arithmetic) and the levels of the other blocks like
  Quant::dequant (32-bit), with QCS blending in the dequantization of the next larger index. Until M6 dependent
  quantization was not rejected although the kernel dequantized every level as a scalar one: a stream with dependent
  quantization and otherwise supported tools decoded wrong with the reconstruction on (no test set and no regular
  stream had that combination); M7 adds joint coding of the chroma residuals (JCCR): intra_resi computes the coded
  residual of a pair (Cb, or Cr with its own QP) at the Cb block and writes the other component's residual derived
  from it (TrQuant::invTransformICT, all six ICT modes); the Cr block, which the decoder records right after its Cb
  block, takes it from there; M8 adds explicit intra MTS with ECM's extension (the transform pairs of all seven intra
  types DCT-II, DCT-VIII, DST-VII, DCT-V, DST-IV, DST-I and identity): the host takes each block's pair from the
  decoder's own TrQuant (getInvTransformTypes, i.e. xGetInvTrParams) and the GPU multiplies with matrices read back
  from xIT's transform functions by impulses (getInvTransformMatrix; the functions form exact integer sums with one
  rounding, butterflies included), so intra_resi has one code path for every type and size. Implicit MTS stays on
  the CPU: it selects by modes only DIMD derives, and without DIMD the encoder stops at a CHECK. The static kernel
  tables of TrQuant's x86 transforms are now set up under std::call_once, as the GPU path initializes a TrQuant on
  its device-creation thread while the decoder initializes its own; M9 adds transform skip (the dequantized levels,
  with the QP of transform skip and no transform shift, are the residual; also for joint CbCr) and BDPCM (the host
  accumulates the levels with the CPU's own invResDPCM; the prediction copies the unfiltered left column or above row,
  which is the GPU's mode 18 / 50 prediction with reference filter, PDPC, fusion and bilateral filter off). Transform
  skip with the regular residual coding (dequantized by DepQuant) stays on the CPU; M10 adds MIP (matrix intra
  prediction): intra_rec Haar-downsamples the unfiltered references, multiplies with the mode's matrix (the host
  uploads MipData.h's matrices) and computes each upsampled sample in closed form (the CPU's running sums are
  before * (factor - pos) + behind * pos, rounded once); the host records MIP blocks with the DC mode, whose read
  range covers MIP's and whose bilateral filter has no angular gate, as MIP's; M11 adds LFNST and NSPT, both syntax
  only: for blocks with lfnstIdx the host runs the CPU's own dequantization and inverse LFNST
  (TrQuant::getInvLfnstOutput) and the GPU the primary transform, and where NSPT replaces the primary transform the
  host computes the residual with TrQuant::invTransformNxN. That keeps them exact by construction, but NSPT blocks are
  many (in m11 ~40 % of the coded blocks), so the host does much of the transform work there: NSPT (and LFNST) on the
  GPU is a speed-up still to do. LFNST of MIP blocks stays on the CPU (its kernel follows a DIMD mode derived from the
  MIP prediction); M12 adds MRL (multiple reference lines 1, 3, 5, 7 and 12 with JVET-Y0116's extended list; the
  fusion line is the next one, up to 13): the reference fill follows xFillReferenceSamples' unit model, where a line
  beyond the first unit has extra units left of and above the corner (7 availability groups per line instead of 5),
  DC averages from entry multiRefIdx + 1, the angular prediction reads the arrays from the line's own corner on
  (refMain + multiRefIdx) with deltaPos starting at angle * (1 + multiRefIdx) and the main array's end replicated over
  (multiRefIdx << s) + 6 entries, and the bilateral filter, which xPredIntraAng turns off for MRL but the DC
  prediction keeps, pads from line 0's availability (Tu::bfAvail). MRL with non-square availability units (the CPU
  counts the corner samples of both arms by the unit height) and TMRL stay on the CPU; M13 adds ISP (intra
  sub-partitions): each sub-partition is a block of its own whose references follow initIntraPatternChTypeISP - the
  first fills the CU's references with arms of twice the CU's size, a later one shifts the CU's arm along the split by
  its offset (where that side is available) and takes the other arm from the previous sub-partition's last row /
  column with its last sample repeated - and whose angular arrays use the CPU's arm lengths (the CU's size plus the
  sub-partition's); vertical sub-partitions 1 or 2 wide share the prediction of a 4-wide region, of which each keeps
  its columns; their 1-D transforms are one stage with shift_2nd + 1 in intra_resi; the bilateral filter of planar /
  DC sub-partitions pads at the CU's origin with the CU's units, as the CPU does, including its writes that run past
  its block's row of iWidthExtSIMD samples into the next rows when the sub-partition is narrower than the CU; the
  sub-partitions of a CU are chained in the scheduling, as the 4x4 owner units cannot tell sub-partitions 1 or 2
  wide or high apart (which also made intra_deps' parallel owner writes race: the last block now wins with
  atomicMax, as on the host); M14 adds sign prediction (TrQuant::predCoeffSigns, decoder side): the host takes each
  block's predicted coefficients and their coded signs with the CPU's own DepQuant::getPredictedSigns (which also
  hands the other levels of the sign area their signs in the decoder's order; the TU gets its levels back, as the CPU
  calls it again when it reconstructs the picture after all) and uploads the levels with the predicted ones positive;
  intra_resi computes that residual and per predicted coefficient its template (the left column and top row of the
  residual of the coefficient alone with amplitude 256 - xIT of one coefficient is one product per stage - times its
  dequantized magnitude, >> 8); intra_rec keeps the block's prediction in shared memory, compares the border the
  neighbours predict (TU::predBorderResi: twice the sample next to the block minus the one beyond it, minus the
  prediction) with the border of every sign hypothesis (up to 256, one invocation each, in 16-bit sums like the CPU's
  Pel; scaled with LMCS chroma scaling), decides the signs in the CPU's order (a subgroup's minimum per search) and,
  where a decided sign is negative, dequantizes and transforms the block again before it reconstructs it. With joint
  CbCr the coded component's block decides (the Cr block when Cr is coded) and writes the other block's residual,
  which waits for it (a scheduling edge that can point forward); the deciding block also waits for the owners of the
  two columns / rows the border reads. Sign prediction of LFNST / NSPT blocks stays on the CPU (the host's LFNST /
  NSPT results depend on the signs). GpuIntraRec::supported()
  rejects every tool whose mode derivation or prediction reads reconstructed samples or that the kernels lack, before
  anything is skipped. All 19 regular streams now fall back to the CPU at TMRL, the next sequence-level check (before
  M14: sign prediction, before M6: QCS, before M5: LMCS, before M4: dual tree); many block-level tools follow. The
  kernels' reference availability is "decoded earlier", while the CPU's (CodingStructure::getCURestricted) also
  excludes other tiles,
  with wavefronts the CTUs right of the current one in the row above, and with GDR the dirty area: until that fix
  pictures with tiles or wavefronts were not rejected and decoded wrong with the reconstruction on (their verification
  showed prediction differences at the tile and CTU boundaries); now tiles, wavefronts and GDR fall back to the CPU.
- kernels: intra_resi (all residuals of the picture at once, dequantization folded into the first DCT-II stage,
  only the nonzero coefficient region) and intra_rec (persistent workgroups take ready blocks from a queue; the
  host gives each block its dependency count, its dependents and the availability the CPU sees as seven numbers per
  reference line, all from syntax and decoding order; the dependencies are the owners of exactly the reference
  samples the block's mode reads; the last dependency of a block to finish continues with it; Vulkan memory model)

GPU time over the 24 pictures of the M1 and M2 test sets (RTX 3070), every block and sample identical to the CPU:

| step | commit | GPU time | step | cumulative |
|---|---|---|---|---|
| first working kernel (one workgroup, sequential reference fill) | 9435a745 / d7e8556b | 2845.5 ms | | 1.00x |
| reference samples filled by every invocation | 07d08f15 | 1191.8 ms | 2.39x | 2.39x |
| independent blocks on 32 concurrent workgroups (memory model) | 640419bb | 376.0 ms | 3.17x | 7.6x |
| residuals in a parallel pre-pass | 4fb7ccaa | 245.7 ms | 1.53x | 11.6x |
| ready queue, exact dependencies and read ranges, host availability | ae34848e | 80.6 ms | 3.0x | 35.3x |
| direct hand-off of released blocks, dependents prefetched | 1001a254 | 68.4 ms | 1.18x | 41.6x |
| residual pass over the nonzero coefficient region only | 73aca6b3 | 56.8 ms | 1.20x | 50.1x |
| filter tables in shared memory | 1b405737 | ~55 ms | 1.03x | ~52x |

A 720p intra picture now takes 4-5 ms of GPU time (its longest chain of blocks reading each other: 1127 of 29448).
With the reconstruction on, the decode time of a 720p picture is parsing ~47 ms + derive-only pass ~24 ms + GPU path
~17 ms (non-LTO; CPU decoding ~100 ms), so the CPU parts dominate; frame threads parse several pictures at once.
Found on the way: plain loads through coherent / volatile buffer references returned stale L1 data on the RTX 3070
(the kernel uses the Vulkan memory model since); a block that hands its released dependent to its own workgroup
saves the queue round trip; prefetching the dependents' block records into shared memory was slower (dropped).

Fix (2026-10-11): intra_resi passed the first transform stage to the second through a global scratch buffer with a
workgroup barrier in between. While other work kept the GPU busy, whole warps of the second stage read the scratch
before the other warps' writes had reached it: in a failing verification decode of m3_park60 the residuals of hundreds
to thousands of blocks per picture were wrong for a few consecutive pictures, in whole warps (sample indices of a
block such as 128-255, 64-127 or 32-63); the prediction and the scheduling data were never wrong. A device-scope
barrier and coherence did not help. It depends on the load: with `perf/gpu/gpuload.py` (decodes with the GPU
reconstruction on and 8 frame threads, i.e. 256 spinning workgroups) running beside the verification loop, 5 of 40
decodes differed; without it, 9 of 88 in an hour with other work on the PC and 0 of 60 in a quiet hour. The first
stage now stays in shared memory (int16, exact because it is clipped to the 16-bit transform range; extended precision
is rejected by the host): 0 of 40 differed under the same load, and the global 16 MB scratch buffer is gone.

Fix (2026-10-11): decodes with the reconstruction on and frame threads hung now and then for good (a gpuload.py child
had run for 11 hours; m13c_25, 64x64, with a CPU load beside it: about one decode in 200 hung at a picture, one in 5
to 100 at the end). Picture hangs were always the process's first submission: its work had run completely (every
block reconstructed, every workgroup past its last step, the submission's tag seen by the kernels), but its fence
never signalled, or vkGetQueryPoolResults with WAIT_BIT spun forever after it; meanwhile the other frame threads'
contexts were being created in detached threads. Vulkan allows that; it looks like a race in the driver (616.92):
with every context created before decoding, or with one frame thread, no hang. End-of-process hangs: the output was
complete and the process hung in its termination, with the device left to the end of the process on purpose. Now no
Vulkan object is created or destroyed while a submission is in flight (gpu::Context::inFlight(), shared from submit
to wait, exclusive for creation), the timestamps are read without waiting, and the device is destroyed after
decoding (DecLib::releaseGpu, releaseGpuDevice): 0 hangs in 1500 decodes under the same load. Later the same day one
decode of m14_fore37 with 8 frame threads (gpuload.py running beside it) gave an all-zero second picture: a context's
first submission whose fence had signalled while its timestamps were unwritten and its results were the buffers'
initial contents (zero planes, a zero dependency count) - without WAIT_BIT that case, once a hang, had become a silent
wrong picture. It happened once in ~1100 decodes with 8 frame threads that day and not again in 880 more of the same
code and a diagnostic version (with and without GPU and CPU load). Since then the shared lock covers a stream from
the recording of its commands on, and every
submission ends with a full barrier and its number written into a host-visible word, which Stream::wait checks: a
late end is waited for, a missing one makes a repeatable stream (GpuIntraRec's: every picture's commands upload and
fill what they read) run the commands again after draining its queue, else wait() throws; the decoder prints the
count of such submissions at the end when there were any.

Sign prediction's decision in intra_rec (M14) is part of the chain. Over the 16 pictures of seven m14 streams
(fore32/22/37, park720, four720, m14a_fore22, m14s_park720; GPU time, best of 3, builds alternated, every picture
identical) M14 took 178.1 ms, 111 ms with the decision step left out (a timing experiment). Each hypothesis' border
is now the sum of two halves computed once per border entry (the changes of its low and of its high four bits;
16-bit sums suffice, the CPU's are Pel) instead of up to 8 template additions per entry: ~150 ms; every subgroup
making the decision itself (no barrier, no shared copy) and the negative positions in registers: 149.0 ms (1.20x in
all). The decision's phases at 150 ms, by stopping after each: border, templates and halves ~8 ms, costs ~8 ms,
decision ~7 ms, residual recomputed ~12 ms. Computing the halves in intra_resi instead (no work in the chain) was no
faster (148.6 ms) and moves more memory; not kept.

## Fixes found on the way

- Decoding that starts at a CRA in the middle of an RA stream (`-s`, random access) crashed the stock decoder:
  `Slice::xGetRefPic` returned `*end()` for reference pictures that are not in the DPB. Ported VTM commit
  76b70fb45f (nullptr for missing pictures, the RPL checks skip them).
- `IntraPrediction::ANG_REF_BUF_SIZE` (encoder step 19) was odr-used without a definition: unoptimized builds did
  not link. Now `constexpr`.
- Tooling: `perf/build.ps1 -NoLto` never configured (the one-element cmake argument list became a plain string, which
  was splatted character by character); `pgo.ps1` / `pgo_dec.ps1` deleted `C:\Code\ECM\build\<name>` without checking
  the name (`..` would have deleted `C:\Code`). Both scripts and `build.ps1` now accept plain names only.

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
