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
| `--Vulkan=1` | deblock on the GPU |
| `--VulkanCheck=1` | with `--Vulkan`: also deblock on the CPU and stop at the first sample that differs, with the edge parameters and samples around it |
| `--VulkanValidate=1` | enable the Khronos validation layer (from the SDK) |
| `--VulkanDevice=N` | choose the GPU; -1 prefers a discrete GPU |
| `--ParseSliceFirst=1` | parse all CTUs of a slice before reconstructing them; the output must not change |

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

## Status

| stage | GPU | checked on the corpus |
|---|---|---|
| deblocking, luma and chroma | yes | 48 streams identical |
| deblocking with LADF or the asymmetric GDR filter | yes | not exercised: no corpus stream uses them |
| everything else | CPU | - |

Speed: on the hybrid path the per-picture copies and the synchronous wait cost more than the GPU saves. ra_cif
(289 frames CIF, Core Ultra 7 265K iGPU): 6.01 s CPU deblocking, 6.43 s GPU deblocking, of which 0.105 s GPU time.
