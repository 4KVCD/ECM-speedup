# GPU decoder (Vulkan)

Work in progress: a port of the ECM-20.0 decoder to Vulkan compute shaders. Every decoded picture must stay
byte-identical to the stock decoder.

## Build and run

The GPU decoder needs the Vulkan headers and a GLSL compiler at build time; the Vulkan loader (`vulkan-1.dll`) is
opened at run time.

- With the LunarG Vulkan SDK installed, CMake finds both through `VULKAN_SDK` and turns `ECM_VULKAN` on.
- Without it, pass `-DECM_VULKAN=ON -DVULKAN_HEADERS_DIR=<Vulkan-Headers>/include -DGLSLANG_EXE=<glslang.exe>`.
  `perf\vkbuild.ps1 <name>` does this for the tools under `C:\Code\tools`.

DecoderApp options:

| option | meaning |
|---|---|
| `--Vulkan=1` | run the stages ported to the GPU (see Status) |
| `--VulkanCheck=1` | with `--Vulkan`: also run those stages on the CPU, start each GPU stage from the CPU's input and stop at the first sample that differs; at the end, print how much work each GPU stage did |
| `--VulkanValidate=1` | enable the Khronos validation layer (from the SDK) |
| `--VulkanDevice=N` | choose the GPU; -1 prefers a discrete GPU |
| `--ParseSliceFirst=1` | parse all CTUs of a slice before reconstructing them; the output must not change |

`perf/vkcoverage.py EXE` sums those counts over the corpus, so that paths the corpus never exercises show up.

With `--Vulkan=1` the decoder prints the GPU time in total and per stage. With the environment variable
`VK_PROFILE=N` it also times every dispatch on its own, with barriers around it, and prints the time per shader.
Dispatches that do not filter in place run N times (up to 8) and the fastest run counts, because the iGPU's clock
varies; the others run once. Profiling slows the decoder down, so its total time means nothing.

`perf/vkbench.py EXE [EXE ...]` runs decoder builds alternately on one stream, pinned to one logical CPU, and prints
the minimum and median of the total time, the GPU time and each stage's GPU time.

`VulkanTestApp` runs unit tests of shader code against the CPU: `selftest` and `sf64`, the software double arithmetic
that replaces FP64 on GPUs without it.

## Test corpus

`perf/vkcorpus.py` builds the test streams (`prepare`, `encode`), records the stock decoder's YUV md5 (`ref`) and
checks any decoder build against it (`check EXE "--args=--Vulkan=1"`). Streams and the manifest are in
`C:\Code\corpus`. The ECM-20.0 encoder fails on several configurations that VVC allows; those streams are absent:

- 4:4:4, 4:2:2 and 4:0:0 (asserts in intra prediction and buffer copies), RGB, lossless and the VTM tool configs
  (unknown options), sign hiding with sign prediction, CTU size 64 with the default BT sizes, periodic IDR with
  IntraPeriod 32, GDR, rectangular tiles in all-intra.
- The stock decoder crashes on 200x136 pictures (access violation).

## Design

- Parsing stays on the CPU. It does not depend on reconstructed samples: decoding with `--ParseSliceFirst=1` gives
  identical output on the whole corpus.
- Pictures live in buffers that both CPU and GPU can access. On a unified-memory GPU every buffer is device-local
  and host-visible, so uploads and downloads are plain copies.
- Shaders get buffer device addresses through push constants; there are no descriptor sets.
- The loop filter stages of a picture record into one command buffer, which `finishPicture` submits without
  waiting: the GPU filters picture N while the CPU goes on with picture N+1. The result comes back, and the
  picture's border padding (`mcFramePad`, which reads the filtered border blocks) runs, when the picture is first
  needed: by a slice that references it, for output, before the next picture's loop filters, before a picture
  buffer that its padding reads is reused, or before a new SPS or PPS. A stage that falls back to the CPU syncs
  first. Check mode submits and waits stage by stage.
- The GPU has no FP64 (`shaderFloat64` is false on the Intel iGPU), so the decoder's double arithmetic will use
  `shaders/sf64.glsl`, which rounds like SSE2 bit for bit.

### Deblocking

The CPU walks the coding units as before and records, per 4-sample edge segment, the boundary strength, filter
lengths, QP and offsets (`DeblockEdgeMaps`). The GPU filters the samples. Edges are not independent: two edges 4
samples apart, one with filter length 3, read each other's output. The GPU therefore filters each strip of 4 lines
(vertical edges) or 4 columns (horizontal edges) in the CPU's order, left to right or top to bottom, one strip per
invocation.

### SAO, bilateral filter and CCSAO

As on the CPU, SAO, the bilateral filter (BIF) and CCSAO read the deblocked picture and add their offsets without
clipping; one joint clip follows. SAO runs per sample: the CPU's sign-line recurrences reduce to two sign
comparisons. The CPU lists the transform blocks that BIF filters, split at virtual boundaries, with the border
availability that decides their padding; the GPU filters one block per workgroup. CCSAO walks each CTU line by line
with pointers that it advances differently per case, and band case 6 starts its line loop at y = 1 on line 0, so a
4:2:0 luma line y reads chroma line (y + 1) / 2. The GPU replays the CPU's pointer advances per line to get every
case right.

### ALF

ALFProcess stays the master of each picture: it walks the CTUs as before, derives the filters, and keeps all the
state that later pictures depend on. With a GPU plan set (`AlfGpuPlan`) it records each CTU's filter choice and
every region it computes of its intermediate result planes (first and second stage fixed filters, the residual
fixed filter, gauss and laplacian results, chroma fixed filters) and their mirrored padding, and skips the sample
work. The GPU then computes:

- the classes of every 2x2 luma block: the CPU's sliding window sums are sums over fixed windows around the block
  (4x4 and 12x12 laplacians, a 10x10 variance), computed with the laplacian cells in shared memory;
- the recorded regions of the result planes, then their mirrored padding, in the CPU's stage order;
- the luma filter per CTU (the four APS filter types the decoder can receive, or a fixed filter result alone), with
  the residual scaling correction, and chroma ALF, CCALF and the chroma scaling correction in one pass.

The transpose index of a class turns the filter taps geometrically: transposed, mirrored horizontally or rotated by
90 degrees. The shaders therefore keep one tap table per filter instead of the CPU's 4 hand-written coefficient
orders.

In check mode the GPU's result planes must also equal the CPU's in every region the picture wrote, and a
difference is traced to the classes that feed it.

The decoder creates ALF anew for every picture (`DecLib` calls `AdaptiveLoopFilter::create`), so the result planes
start uninitialized in every picture. In some layouts of CTU filter types ALFProcess would read samples of them that
the picture did not write. The plan tracks every write and read; such a picture is filtered on the CPU, as stock
does. No corpus stream does this: with the result planes overwritten by a picture-dependent pattern at the start of
every ALFProcess, all 48 streams still decode identically.

The plan covers one-slice pictures in 4:2:0 and 4:0:0. ALF of other pictures runs on the CPU: 4:2:2 and 4:4:4,
several slices, virtual boundaries, loop filtering disabled across slice, tile or subpicture boundaries, CU-level
QP for the fixed filters (`cuQpDeltaSubdiv`) and the NN loop filter. The decoder prints how many pictures took the
CPU path and why.

### Offset refinement

The luma offset refinement after deblocking and after ALF (`calcOffsetRefinement`) runs per sample: a small
correction against the stage's change, from the log2 of its magnitude, with a threshold at block boundaries (the
deblocking's coding information, which the GPU receives once per picture after deblocking). After ALF it skips the
CTUs with luma ALF off.

### LF-CCCM

LF-CCCM (`LoopFilterCccm`) fits linear models of chroma from the downsampled luma per window of each CTU and adds
the change after ALF. The GPU computes the downsampled luma for the whole picture, then runs the CPU's sequence of
up to 6 model fits per window, with one of three kernels chosen by the size of the regression window:

- `lfcccm_window_tiny`, up to 100 samples (windows up to 8x8, most of them): one window per invocation, sequential
  as on the CPU. A workgroup per window would leave most of its invocations idle.
- `lfcccm_window_small`, up to 1024 samples: one workgroup of 64 per window; each invocation sums whole statistics
  terms, the rows of the integer Gaussian elimination of `CccmCovariance::solve3` are updated in parallel one pivot
  at a time, and the model outputs stay in shared memory.
- `lfcccm_window`, larger: one workgroup of 256 per window; each invocation sums all terms over its share of the
  samples.

The model is applied in parallel up to the first sample (in raster order) that deviates too much, which is where the
CPU's loop stops.
Quirks kept: the chroma offsets and the sub-model threshold are summed in 16-bit variables on the CPU and wrap; the
rounding term of `xGetDivScaleRoundShift` shifts by the bit count modulo 32, as x86 does; the regression window at a
CTU's left or top edge keeps its start and still grows by 2 samples; `clipArea` clips only at the right and bottom.

### Temporal ALF

The temporal ALF (`TAlfProcess`) adds to the luma of each 4x4 block of the CTUs with TALF on a correction from 13
pairs of samples around the block's position in one or two reference pictures, moved by the motion of a nearby
block. `AdaptiveLoopFilter::talfGpuPlan` runs the CPU's block selection (`getRefPics`, `getMotionOffset`) and
records the accepted blocks with their reference pictures, integer motion and filter; `talf.comp` filters them, one
invocation per sample. The reference samples can lie up to 29 samples outside the picture, in the border that
`finishPicture` filled with motion-compensated padding, so the GPU keeps copies of the last 8 reference pictures'
luma with a 32-sample border, taken from the CPU when a picture first serves as a TALF reference.

The GPU follows the AVX2 code that the stock decoder runs: the correction is added without clipping (adaptive
clipping clips the picture afterwards), and the rounding offset `1 << (shift - 1)` takes the shift count modulo 32.
`simdSetBiInput` steps through the second reference block's upper samples with the first block's stride; both
strides are equal unless the reference pictures differ in size, and such pictures run on the CPU.

## Status

| stage | GPU | checked on the corpus |
|---|---|---|
| deblocking, luma and chroma | yes | 48 streams identical |
| deblocking with LADF or the asymmetric GDR filter | yes | not exercised: no corpus stream uses them |
| SAO (all 4 edge classes, band offset), bilateral filter luma and chroma, CCSAO edge directions 0-3 and band candidates 4-8, joint clipping | yes | 48 streams identical |
| CCSAO band candidates 0-3, virtual boundaries | yes | not exercised |
| ALF luma: fixed filter sets alone, APS filter types 13_EXT_DB_RESI_DIRECT, 13_EXT_DB_RESI, 13_DB_RESI_DIRECT, 13_DB_RESI, classifiers 0-2, residual scaling; chroma ALF_FILTER_9, CCALF, chroma scaling | yes | 48 streams identical, 320 pictures on the GPU |
| ALF chroma ALF_FILTER_9_NO_FIX | yes | not exercised |
| ALF in pictures the plan does not cover (see ALF) | CPU | - |
| offset refinement after deblocking and after ALF | yes | 48 streams identical, 34 and 84 pictures |
| LF-CCCM: CTU model types 0, 1, 3-6, window sizes 0-2, 4, 5, all update factors, every branch of the window process | yes | 48 streams identical, 73 pictures, 31612 windows |
| LF-CCCM CTU model types 2 and 7, window sizes 3, 6, 7 | yes | not exercised |
| temporal ALF modes FORWARD_TALF_MV, BACKWARD_TALF_MV, BIDIR_TALF_MV, template shapes 0 and 1 | yes | 48 streams identical, 320 pictures, 5 with filtered blocks, 10420 blocks |
| temporal ALF modes FORWARD_TALF, BACKWARD_TALF, BIDIR_TALF (the two nearest reference pictures, no motion) | yes | not exercised |
| temporal ALF after an ALF on the CPU, with reference pictures of another size, with GDR | CPU | 48 streams identical with every picture forced to the CPU path |
| everything else | CPU | - |

Speed: ra_cif (289 frames CIF, decoder pinned to one core of the Core Ultra 7 265K, Intel iGPU), `perf/vkbench.py`,
minimum and median of paired runs. The machine's speed drifts by about 10 % between sessions (other load), so
compare rows measured together:

| build | total | GPU time |
|---|---|---|
| stock ECM-20.0 | 6.53 s, median 6.53 | - |
| this branch, CPU path | 5.72 s, median 5.91 | - |
| GPU: deblocking, SAO, BIF, CCSAO | 7.3-8.0 s (3 runs) | 0.35-0.44 s |
| GPU: + ALF | 8.5-9.3 s (3 runs) | 1.2-2.1 s |
| GPU: + offset refinement, LF-CCCM with one workgroup per window | 10.10 s, median 10.44 | 3.78 s, median 3.83 |
| GPU: + LF-CCCM kernels by window size, ALF parameters read in place | 7.94 s, median 8.00 | 1.88 s, median 1.91 |
| later session: stock ECM-20.0 | 7.23 s, median 7.29 | - |
| later session: this branch, CPU path | 6.39 s, median 6.42 | - |
| later session: GPU, + temporal ALF | 8.16 s, median 8.40 | 0.93 s, median 1.25 |
| later session: GPU, + one submission per picture without waiting | 6.48 s, median 7.31 | 0.84 s, median 0.86 |

The GPU time of a stage depends on how busy the GPU is: the iGPU clocks down between submissions, so with less
work per picture the unchanged stages (deblocking, SAO) take longer. Waking the idle iGPU took about 2.4 ms per
picture when every stage waited for its result. With one submission per picture the CPU still waits about 2 ms
per picture, when the next picture references the one the GPU is filtering.
