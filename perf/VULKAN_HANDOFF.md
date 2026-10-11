# Handoff: the Vulkan decoder work of branch `vulkan` (Core Ultra 7 265K PC)

Written 2026-10-11 on the Core Ultra 7 265K PC, which switches to multithreading the encoder on the CPU. The
other computer (i7-10700) takes over all Vulkan work. Read this together with that computer's own handoff
(`perf/HANDOFF.md` on branch `dec-gpurecon`): both computers built a Vulkan decoder independently, and this file
says what branch `vulkan` has that `dec` / `dec-gpurecon` may not, what was measured, and what was learned.

## 1. Status in one paragraph

Branch `vulkan` (9 commits on top of `opt` c853a4cb, the encoder speed-up state of 2026-10-10) runs every in-loop
filter of ECM-20.0 on the GPU, bit-exact: deblocking, SAO, bilateral filter, CCSAO, offset refinement (both stages),
LF-CCCM, ALF with CCALF and the scaling corrections, the temporal ALF and the adaptive luma clipping. The CPU still
parses and reconstructs. The loop filters of a picture go to the GPU in one submission that runs while the CPU
decodes the next picture. On ra_cif (289 frames CIF, decoder pinned to one core) the GPU path takes 6.48 s against
7.23 s for stock ECM-20.0 measured in the same session (1.12x), and against 6.39 s for this branch's CPU path:
the loop filters are only about 8 % of the decode time, so moving them can never win much. The user then asked
for every part of the decoder on Vulkan; branch `vulkan-recon-wip` holds the first, unfinished step of that (the
inverse transform, section 6). `dec-gpurecon` on the other computer is much further along with GPU reconstruction.

## 2. Branches (pushed 2026-10-11)

`vulkan`, oldest first. Every commit: all 48 corpus streams decode identically to stock on the CPU path, with
`--Vulkan=1` and with `--Vulkan=1 --VulkanCheck=1`; VulkanTestApp passes; encoder md5 suite (perf/bench.py, 5 tests)
identical whenever CommonLib changed.

| commit | step | ra_cif, one core, paired min |
|---|---|---|
| 88f4a280 | Vulkan compute context (`vkd::Core`) and GPU deblocking | - |
| 5d103bdc | SAO, bilateral filter (luma and chroma), CCSAO, joint clipping | 7.3-8.0 s |
| 039f7189 | ALF with CCALF and scaling corrections (AlfGpuPlan recorder in CommonLib) | 8.5-9.3 s |
| b23970be | offset refinement after deblocking and after ALF | |
| 77f42427 | LF-CCCM | 10.8-11.2 s |
| f72d104f | GPU time per stage, `VK_PROFILE=N` per-shader profiling, perf/vkbench.py | no change |
| 55c35a20 | LF-CCCM windows in three kernels by size, ALF parameters read in place | 10.10 -> 7.94 s (1.27x) |
| c14b2be5 | temporal ALF | no change |
| 54a34c7c | one submission per picture, waits deferred to first use, adaptive clipping on the GPU | 8.16 -> 6.48 s (1.26x) |
| (this file) | handoff | |

`vulkan-recon-wip` (one commit on top of `vulkan`, unfinished, see section 6): hooks through which CommonLib hands
reconstruction functions to the GPU, the recorder of the inverse transform's decisions, the shader
`inv_transform.comp`, and two command contexts in `vkd::Core`. It builds and decodes the corpus identically (the
hooks are off), but the host side `VkRecon.cpp` is not written and the shader has never run.

Not pushed, on this PC only: nothing else of value. Working notes in `C:\Code\work` (alf_port_notes.md, patch
scripts, commit messages) are scratch.

## 3. How branch `vulkan` works

- Options of DecoderApp: `--Vulkan=1` (GPU loop filters), `--VulkanCheck=1` (the CPU runs every GPU stage too;
  each GPU stage starts from the CPU's input and must give the same samples; at the end a coverage line counts the
  paths each stage took), `--VulkanValidate=1` (Khronos validation layer; the SDK was never installed here, so it
  never ran), `--VulkanDevice=N`, `--ParseSliceFirst=1` (parse a whole slice before reconstructing it; output is
  identical on the corpus). Environment `VK_PROFILE=N` times every dispatch alone (fastest of N runs for the
  dispatches that do not filter in place).
- Code: `source/Lib/VulkanLib` (VkCore: device, buffers with device addresses, pipelines from embedded SPIR-V,
  command contexts; VkLoopFilter.cpp deblocking and SAO; VkAlf.cpp; VkLfCccm.cpp; VkTalf.cpp; shaders/*.comp,
  compiled by glslang at build time). Shaders take buffer device addresses in push constants, no descriptor sets.
  The design notes, quirks and coverage table are in `perf/VULKAN.md`.
- The CPU records what each stage would do instead of doing it: `DeblockEdgeMaps` (edge parameters per 4-sample
  segment), `AlfGpuPlan` (per CTU filter choices, filters, regions of the intermediate planes written and mirrored),
  `TalfGpuPlan` (accepted 4x4 blocks with motion and filter). These recorders call the CPU's own decision code, so
  they stay exact by construction.
- Asynchronous chain (54a34c7c): the stages record into one command buffer; `finishPicture` submits it and leaves
  the picture pending (`DecLib::m_vkPending`). `DecLib::vulkanResolve` waits, downloads the picture and runs the
  deferred border padding (`xPadPicture`: `mcFramePad` reads the filtered border blocks, so it must wait too). It is
  called when the picture is first needed: by a slice whose reference lists contain it, before the next picture's
  loop filters, before a picture buffer that the padding reads is reused (the pending picture or its references,
  `xVulkanPendingUses`), before a new SPS or PPS (the padding reads them), before a lost picture is copied, and from
  DecApp before a picture is written or the output flushed. GDR pictures and pictures with a hash SEI complete in
  finishPicture. A stage that falls back to the CPU syncs first (ALF through `AlfGpuPlan::beforeCpuSamples`).
- Tools: `perf/vkbuild.ps1 <name> [-Configure]` (Ninja + MinGW build with the Vulkan headers and glslang under
  `C:\Code\tools`; re-configure after adding .cpp or .comp files, the lists are globbed), `perf/vkcorpus.py`
  (prepare / encode / ref / check; the corpus streams are in `C:\Code\corpus` with the stock md5 in
  manifest.json, not in git: rebuild them from the clips in `C:\Code\seq` with `prepare`, `encode` and `ref`), `perf/vkcoverage.py` (sums the
  check-mode coverage over the corpus and lists the paths never exercised), `perf/vkbench.py` (paired runs pinned to
  one logical CPU, min and median of total and per-stage GPU time).

## 4. Bit-exactness findings (all reproduced in the GPU code; details in perf/VULKAN.md)

- Deblocking: edges 4 samples apart can depend on each other, so the GPU filters strip by strip in the CPU's order
  instead of all edges at once.
- ALF: `DecLib` calls `AdaptiveLoopFilter::create` for every picture, so the intermediate result planes (fixed
  filter, gauss, laplacian) are reallocated each picture and some CTU layouts read samples the picture never wrote,
  i.e. uninitialized memory in stock. The plan tracks every write and read per 2x2 cell and sends such pictures to
  the CPU. A poison test (the planes overwritten with a pattern) changed no corpus stream, so the corpus never does
  it. `transposeIdx` is a geometric transform of the taps (T0 identity, T1 (dy,dx)->(dx,dy), T2 (dy,-dx), T3
  (-dx,dy)) because the filters are point-symmetric.
- LF-CCCM: the CPU sums the chroma offsets and the sub-model threshold in 16-bit variables that wrap; the rounding
  term of `xGetDivScaleRoundShift` shifts by the bit count modulo 32 as x86 does; the regression window at a CTU's
  left or top edge keeps its start and still grows by 2 samples; `clipArea` clips only at the right and bottom.
- Temporal ALF: the decoder runs the AVX2 code. `simdSetBiInput` steps through the second reference block's upper
  samples with the first block's stride (equal unless the references differ in size, which the GPU path leaves to
  the CPU); the correction is added without clipping (adaptive clipping clips later); the rounding offset
  `1 << (shift - 1)` takes the shift modulo 32. The reference samples lie up to 29 samples outside the picture, in
  the border that `mcFramePad` filled with motion-compensated padding, not edge replication.
- Inverse transform (for the reconstruction port): every inverse transform type and size runs through
  `fastInverseTransform_SIMD` (a matrix product; DCT2 sizes of 16 and more use an even/odd split that should give the
  same int32 sums as the full product, not yet verified on the GPU); only size 2 takes the generic loop that honours the zero-out lines. The VVC-style LFNST on 8x8 with a
  transposed mode reads LFNST outputs beyond the 48 it computed (stale values of an earlier call); ECM streams use
  the extended LFNST, where this does not happen.

## 5. Performance findings

- iGPU (Intel Graphics 0x7d67, driver 101.8991): unified memory (every memory type device-local and host-visible),
  no FP64, int64/int16/int8 and 8/16-bit storage, subgroups 8-32, 48 KB shared memory, timeline semaphores,
  `VK_EXT_external_memory_host` with 4 KB alignment (zero-copy picture buffers are possible, not done).
- Waking the idle iGPU costs about 2.4 ms. Before 54a34c7c the first submission of each picture waited 3.3 ms for
  0.9 ms of deblocking, and the decoder was blocked in `Core::submit` for 25 % of its time; after it, about 4 %
  (the CPU still waits ~2 ms per picture when the next picture references the one on the GPU).
- The iGPU's clock varies up to 2x between runs; take the minimum of paired runs. On this PC the System process
  used about 1.4 cores on logical CPUs 7 and 8 (Intel Graphics Software overlay running), and the GPU path's times
  were bimodal (about 6.3 or 7.2 s per run on ra_cif) with constant GPU time; a spinning fence wait did not change
  that. Pin to quiet cores (6, 9, 0).
- Sampling profile of the CPU decoder on ra_cif: `DecCu::xDeriveCUMV` 43 %, `DecCu::xReconInter` 33 % (OBMC 12 %,
  motion compensation 11 %, GPM 6 %), intra 3 %, loop filters 8 %, inverse transforms 0.7 %, sign prediction 1.3 %,
  CABAC parsing 1.7 %. Motion derivation is spread over many tools (affine merge, BDMVR, ARMC, template matching,
  each a few percent) and reads the reconstructed neighbours, so reconstruction is serial CU by CU.
- `vkQueueSubmit` takes about 0.025 ms when timed directly; the sampling profiler (sprof) shows it much larger,
  which is an artifact of suspending the thread. sprof must not share a pinned core with the decoder.

## 6. The reconstruction port that had just started (`vulkan-recon-wip`)

Approach: replace leaf functions of CommonLib one at a time and move up the call tree. `GpuReconHooks` (CommonLib,
`GpuRecon.h`) is a global hook object that only the decoder installs; e.g. `TrQuant::invTransformNxN` asks it first
and `invTransformNxNCpu` is the original body. `TrQuant::recordInvTransform` (TrQuantGpu.cpp) records every
decision of the active TypeDef.h configuration (dequantization parameters of `Quant::dequant` or the dependent
quantizer, LFNST kernel and scans, transform types, NSPT kernel, skip widths) into `GpuInvTransformJob`;
`shaders/inv_transform.comp` does the arithmetic. Missing: `VkRecon.cpp` (pack the job, map the CPU table pointers to
offsets in GPU tables, dispatch on `Core::CTX_RECON`, compare with `invTransformNxNCpu` in check mode), the DecLib
option, tests. Given `dec-gpurecon`, it is probably better to take ideas from here than to continue this branch.

## 7. Multithreading findings (relevant to frame threads and to a GPU decoder that runs pictures in parallel)

- Random access with GOP 32 (the CTC config used for ra_cif): 16 of 32 pictures per GOP are in the top temporal
  layer and reference only lower layers, so they can decode in parallel; the critical path is about 6 pictures per
  GOP. Low delay has no such parallelism (each picture references the previous one); only CTU-row wavefronts help
  there, and at CIF there are 3 CTU rows of 128.
- Parsing is independent of reconstruction within a slice (`--ParseSliceFirst=1` identical on all 48 streams).
  Cross-picture state that orders the work: the temporal CABAC contexts and CABAC retraining (parse order), ALF
  APS state, `AlfScalePrev` and `g_pictureControlInfo` (loop filter order), the temporal CCP and EIP model stores
  (`CS::saveTemporalCcpModel`, `saveTemporalEipModel` in executeLoopFilters).
- Thread-safety hazards in CommonLib: function-local `static` scratch, e.g. `static std::vector<Mv> cMvdFiltered` in
  `InterPrediction::initOffsets` (MVD sign prediction); DecCu, InterPrediction, IntraPrediction and TrQuant keep
  per-call scratch as members, so each thread needs its own set.
- Deferring work on a decoded picture (as frame threads or an asynchronous GPU do) must respect what reads it:
  `mcFramePad` reads the picture's own filtered border blocks, its references' samples and its parameter sets, so it
  must run before a reference of the picture is reused or a new SPS/PPS replaces the old ones; `finishPicture` frees
  the CU index maps (`destroyTemporaryCsData`), so any plan built from `cs.getPU()` must be made before it.

## 8. Rules

The user's rules are listed in `perf/HANDOFF.md` section 2 on `dec-gpurecon` and in the global CLAUDE.md: output
bit-exact always, one commit per step with a detailed message, no `git stash`, work in worktrees, nothing pushed
without the user's yes each time, at most one agent, report per-step and cumulative gains with hardware.
