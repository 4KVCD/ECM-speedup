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
| everything else | CPU | - |

Speed: on the hybrid path the per-picture copies and the synchronous waits cost more than the GPU saves. ra_cif
(289 frames CIF, decoder pinned to one core of the Core Ultra 7 265K, Intel iGPU), 3 runs each:

| stages on the GPU | total | GPU time |
|---|---|---|
| none | 6.1-6.3 s | - |
| deblocking, SAO, BIF, CCSAO | 7.3-8.0 s | 0.35-0.44 s |
| deblocking, SAO, BIF, CCSAO, ALF | 8.5-9.3 s | 1.2-2.1 s |
