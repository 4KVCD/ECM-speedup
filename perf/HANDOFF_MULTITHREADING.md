# Handoff: multithreading the ECM encoder on the CPU

Written 2026-10-11 by the Claude session on the i7-10700 PC, which now switches to the Vulkan decoder port. The
multithreading work moves to the other computer. This file has the user's request, the measurements, the
dependency analysis and a plan ranked by expected gain. The Progress section records the work done since.

## The request

The user asked to "optimize the CPU encoder significantly", multithreading first ("find ways to multithread as many
things as possible. The things that look like they have to be done serially? Find a way to multithread them"), and
then to "spend more time on the plan. Make sure the plan optimizes for the biggest areas for multithreading gains".
SIMD work is deferred. As always, output must stay bit-exact with stock ECM-20.0 at every thread count (bitstream
md5 and reconstruction).

User rules that shaped this work, besides the usual commit/push/QA rules in `perf/HANDOFF.md` of `dec-gpurecon`:
- At most one agent at a time and no multi-agent workflows. Tell every agent in its prompt not to start agents of
  its own. Here one analysis agent started three helpers by itself, which the user did not want.
- A benchmark plan for reported final numbers needs the user's approval. Plans are reviewed up to 3 rounds.

## Git state

- Branch `enc-mt` = `dec` 2912a239 + this handoff and the tools in `perf/mt/`. Start the work here. A paired A/B
  (suite, 1 rep, 8 jobs) showed `dec` 1.1 % faster than `github/perf` c853a4cb, both bit-exact; that decided the base.
- Only on the i7-10700 PC and not pushed (ask the user if they are needed):
  - `enc-gpu` fa3cb65e: IntraTMP template search on the GPU (bit-exact, off by default) and the study "Encoder on the
    GPU" in its `perf/OPTIMIZATIONS.md`. Conclusion: the GPU does not pay for the encoder, because the search is
    a long chain of small, dependent steps and every round trip to the GPU costs more than the step.
  - `wt-simd`: uncommitted edits that make `perf/bench.py`, `build.ps1`, `build_pinned.ps1` and `pgo.ps1` take the
    project folder from `ECM_ROOT`.
- Line numbers below refer to `enc-mt` (= `dec` 2912a239). Those marked "(agent)" come from a read-only code
  analysis by agents and should be re-checked before they are relied on.

## Progress (Core Ultra 7 265K PC, from 2026-10-11)

Commits on `enc-mt`, all bit-exact with stock ECM-20.0 (bitstream md5 of the suite), not pushed:
- P0: `c10f520f` one `EncSearchStack`; `03aa9a96` the globals and statics the mode search writes become stack
  members; `f6ef8708` RdCost's LMCS weights per object; `fba9e8f2` `--SearchStacks=K` (pictures use K stacks in
  turn on one thread, the check of the stacks).
- P1: `--FrameThreads=N` (the commit after `fba9e8f2`). What the P1 table below missed, found by bisecting threaded
  mismatches (a setup made to wait at checkpoints for the previous picture's search or finish):
  - **The picture header**: `EncLib::m_picHeader` is one object every picture points to, so the next setup
    overwrote fields the current search and slice writing read. Each stack holds a copy taken at the end of the setup.
    The values `HLSWriter::codePictureHeader` infers reach the next pictures only through the shared header; of
    those, only `MaxNumAffineMergeCand` is read before a setup sets it (field-level diffs of the header between
    setups, offsets mapped with an `offsetof` program built with `#define private public`).
  - **`Slice::generateIntersectingMv`** (MVP extension) in the setup reads the reference pictures' motion: moved to
    the search start.
  - `Picture::referenced` (only the log's b/B) and EncLib's APS map (setup LMCS/scaling-list lookups vs the
    finish's ALF APS insertions).
  - `perf/mt/phase_audit.py` lists the `EncGOP` members each phase writes and reads; `perf/mt/ft_sim.py` simulates
    the frame-thread schedule from a serial log.
- P1 result: as expected from the same-layer CABAC chain, plus the finishes in coding order (the next intra period's
  CRA finishes after the whole previous period, and all its pictures reference it), frame threads alone give about
  1.3x on RA (`ft_sim.py` on the serial ra_long log: at most ~1.35x with any number of threads). The larger gains
  are P2 (all configs) and, for long RA encodes, `--ParallelSegments`, which overlaps intra periods with processes
  and combines with frame threads.
- P2, first step: `--CuThreads=N` (the commit after the FrameThreads one): the intra luma RD candidates of the DCT2
  and LFNST passes are coded by N helpers at the same time (light stacks of ~130 MB). Bit-exact at 1/2/4/8 and with
  FrameThreads. It gains little: 1.06-1.11x on the intra test pictures and ra_parkjoy at 2-4 threads, 1.00-1.01x
  on foreman LD/RA; 8 threads are slower than 4 (OPTIMIZATIONS.md has the table).
- Grain probe (now run, `perf/mt/grain_probe`, small clips, one P-core each). Loop time and average task:

  | loop | AI foreman CIF 1 f | LD foreman QCIF 5 f | RA foreman QCIF 9 f | probe speed-up at 4 workers |
  |---|---|---|---|---|
  | encode total | 34.8 s | 46.0 s | 96.6 s | |
  | intra RMD (rough mode decision) | 8.9 s, 0.4 us | 5.1 s, 0.4 us | 5.3 s, 0.4 us | 0.6x (too fine) |
  | intra RD DCT2 pass | 5.2 s, 9 us | 1.4 s, 11 us | 2.0 s, 9 us | 2.6x |
  | intra RD LFNST passes | 4.2 s, 5 us | 1.5 s, 5 us | 3.7 s, 6 us | 2.7x |
  | intra RD MTS pass | 2.7 s, 8 us | 0.7 s, 7 us | 1.8 s, 10 us | 2.8x (needs speculation) |
  | intra RD ISP | 2.0 s, 0.8 us | 0.6 s, 0.9 us | 1.2 s, 1.1 us | 1.2x |
  | merge SATD pass | - | 3.9 s, 4 us | 13.7 s, 7 us | 2.6-3.1x |
  | merge full RD | - | 3.8 s, 29 us | 6.3 s, 27 us | 2.9x (needs speculation) |
  | uni/bi ME per reference | - | 2.4 s, 6 us | 3.3 s, 9 us | 1.2-2.5x |
  | GPM MC, SAD, blend, RD | - | 1.8 s | 4.1 s | 1.4-3.4x |

  The probed loops cover about half of each encode; the rest (merge list building and its TM/BDMVR refinements,
  affine ME, AMVP, GPM candidate stages, chroma, TMP search) is spread over dozens of functions of 0.5-3 % each
  (LD profile: merge test 29 %, inter test 21 %, GPM 17 %, each split over 15-20 callees). At CIF the task sizes
  are the same as at QCIF.
- What the first step measured, against the probe's model of 2 us per task:
  - The real cost per loop and task was far higher at first: the main thread's record, compare and install of a
    result, the workers' state sync and wake-up (12 us per loop while they slept), and the helpers' candidate coding
    at 1.7x the serial time (shared allocator lock). After the fixes in the commit: ~1.7 s of main-thread overhead
    for ~1.1 M candidates of 5 us each (ai_foreman), helpers at 1.2x the serial time.
  - Workers that spin cost the main thread ~3 % of turbo frequency (a serial encode next to three spinning P-cores:
    33.0 s -> 34.1 s). Sleeping workers wake too late for loops of 50 us, so they spin 2 ms after each job; in LD and
    RA, where the intra loops are a small part, that cancels the gain.
  - Revised estimate for P2 on LD, all probed loops at the probe's efficiency: about 1.25x; realistically 1.1-1.2x,
    after a separate engineering step for each of many loops.
- P5 stage A (the commit after the PDP fix): GOPs overlap. `EncLib::encode` returns while the GOP's pictures are in
  flight, so the next GOPs are read and started meanwhile (at most a GOP, or a picture per search stack, ahead);
  setups and finishes keep their turns in coding order over all GOPs; EncLib hands the reconstructions out in output
  order as they are done, and a picture's buffer is not reused before. The setups and EncLib's reading of a picture
  share a lock on the picture list (the reading also rotates the recon ring and writes the shared picture header's
  SPS/PPS IDs). Pictures read ahead are marked unused for reference by earlier pictures' setups (as within a GOP
  before), so buffer reuse also checks that the picture has been output.
  - AI gets most of what `ft_sim.py`/`p5_sim.py` predict: 33 pictures 3.47x with 4 threads, 6.17x with 8 (P-cores). RA gains only at GOP boundaries (65 frames, 3 intra periods: 1.32x with 4 threads, 1.50x with 8, against 1.25x and 1.35x
    before),
    LD nothing (in coding-order finishes, one stack per picture in turn).
  - `perf/mt/p5_sim.py` (list scheduling with K threads and S stacks): with the finishes ordered only within an intra
    period, RA would reach 5.2x with 8 threads and 8 stacks (7.2x with 16 stacks), AI 7.2x. That is stage B: per-period
    copies of the finish state (ALF/SAO/CCSAO encoders, APS map, the HLS writer), the sequential finish state decided
    in the setup (`m_bSeqFirst`, AUD) or moved to an output step in coding order (AU output, PSNR sums, log line), and
    stacks taken by pictures that are ready.
- P5 stage B-1: setups run on a setup stack of their own (only what a setup uses is initialised: slice encoder,
  lambdas, adaptive search ranges, IBC hash map, reshaper); `EncSlice::saveSetup`/`loadSetup` move the result to the
  search stack the picture takes when its search inputs are done. Stacks go to ready pictures, earliest first, while
  the earliest picture not finished has one or one stays free for it (deadlock-free with any finish order where that
  picture waits for nothing). Taking stacks strictly in coding order once ready blocks RA behind pictures waiting for
  their references (modelled 1.11x instead of 1.45x with 8 threads; measured ra20 1.04x at 4 threads).
- P5 stage B-2 (`--FramePeriods`): the finishes of up to M intra periods run at the same time. EncLib holds a finish
  context per period in flight (`EncFinishContext`: ALF and SAO encoders, loop filter, ALF-CCCM, MC padding, HLS
  writer, APS map, with the ALF CTU control and CCSAO history that were globals); EncGOP's filter pointers are
  thread-local and point to the picture's context during its finish. A period's finishes run in coding order through
  `EncPicOrder::finishPrev` (the last picture that used the context). An output step in coding order prints each
  finish's captured log (`g_msgCapture`), writes its access unit and adds its PSNR sums and RVM bits. The first picture
  (`m_bFirst`, `m_bSeqFirst`) is decided in the setup; parameter-set and SEI writing take locks; the embedded-APS list,
  AUD and BP flags are per thread; the ALF scale statics became members, the ALF coefficient caches fill under a lock,
  the slice writer's stored contexts are per thread. EncLib reuses a picture buffer only when no picture in flight
  references it (the next period's CRA marks the previous period's pictures unused while they are still read).
  - A period starts only at a RAS-init IRAP: the RAS-init reset keeps the SAO disabled rate of layer 0, which a
    non-intra RAS-init picture reads (ra20's POC 16: SAO slice flags differed, reconstruction identical).
  - Results (8 P-cores, previous commit's single thread as base): AI 33 pictures 6.83x with 8 threads, RA 65 frames
    1.70x with 4 and 2.79x with 8 (stage A: 1.32x / 1.50x), as `p5_sim.py` predicts (1.65x / 2.79x); RA 129
    frames 2.48-2.64x with 8 (two runs).
  - Not covered by the tests: ALF APSs (ECM embeds APSs in the picture header, and no test sequence signalled an ALF
    APS; LMCS APSs appear at every intra picture and are covered).
- P5 stage B-3: a picture gives its search stack back when its search ends. A timeline of RA 129 frames with B-2
  (temporary `ECM_FT_TRACE` build, `C:\Code\work\mt\trace_*.py`) showed 3.35 of the 8 stacks held on average by
  pictures that had searched and waited for their period's earlier finishes, only 2.46 searching. Each finish context
  now has a finish stack (`EncFinishStack`: slice encoder, CABAC encoder, context cache, IntraPrediction for ALF-CCCM,
  RdCost and reshaper copies); the finish reaches it through EncGOP's `m_pcFinish*` pointers. What the finish needs
  of the search stack moves with the picture: RdCost, reshaper, the slice encoder's AMVP-SbTMVP area count (the
  writer counts into it; only the search start resets it), the separate-tree partitioner of the CABAC encoder.
  - Results: RA 129 frames 4.10x with 8 threads (B-2 2.48-2.64x), RA 65 2.74x with 4 (1.70x) and 2.79x with 8.
  - A full `EncSearchStack` object was 343 MB even when nothing in it is initialised (InterSearch 279 MB, of which
    `m_gpmacsSplitModeTmSel` was 250 MB; EncCu 51 MB). The finish stacks are therefore small structs.
- P5 stage B-4: the setup stack is a small struct too (`EncSetupStack`: slice encoder, CABAC encoder, TrQuant, RdCost,
  reshaper). Its slice encoder (`EncSlice::initSetup`) keeps the adaptive search ranges itself and leaves the IBC hash
  map to the search stack, which resets it when it takes the picture (`loadSetup`). Peak commit 0.33 GB lower with
  frame threads (RA 129 frames with 8 threads 11.1 -> 10.8 GB).
- `InterSearch::m_gpmacsSplitModeTmSel`, the GPM split modes of every pair of MMVD indices and merge candidates
  (250 MB per search stack), is a cache of one CU: a CU's GPM test starts with all rows unavailable, and no row of an
  earlier CU is read (probe: 84 million reads). A CU now takes a 43 kB block per index pair it uses from a pool
  (at most 525 pairs, 22 MB, in the probe), which the next CU reuses.
- P5 stage C: with frame threads, a picture's temporary coding-structure data and refined TMVP data are created when
  its search starts (`xStartPicSearch`), not in the setup. A commit probe at 1080p (temporary counters around each
  allocation) put a picture at about 150 MB: buffers 16 MB and coding structure 30-33 MB when read, temporary buffers
  19.5 MB, temporary coding-structure data 35 MB and refined TMVP data 35-51 MB in the setup. RA 1080p with 8 threads
  and 4 periods, first 5 minutes: 37.4 -> 26.7 GB (M = 8 before: 58.8 GB).
  - Still held by a waiting picture: its reconstruction and coding structure (motion fields and per-4x4 maps, about
    40 MB at 1080p; EncGOP's setup code touches neither directly) and the temporary buffers (true and filtered
    original, which the temporal filter writes in the setup, and the residual). A finish context is about 0.8 GB at
    1080p, measured with the same probe: SAO statistics (`createEncData`) 625 MB, ALF 97 MB, loop filter 7 MB, the
    rest 28 MB. A finish frees and creates the SAO statistics around its SAO (as stock), so contexts beyond the first no
    longer create them at startup: an idle context is about 0.13 GB.
- Automatic FramePeriods (M = 0): a period per frame thread, capped by the periods of the sequence and by 70 % of the
  physical memory (`SystemMemory.cpp`; figures in `EncAppCfg::xCheckParameter`), all intra min(N, 4). RA 257 frames
  with 8 threads: 4.16x with M = 4, 6.49x with M = 8 (automatic); AI 33 pictures with 8 threads 32.0 s with M = 4,
  32.3 s with 8. RA 1080p with 8 threads on this 64 GB machine takes 8 periods (34 GB on average in the first
  5 minutes).
- P5 stage C3: a waiting picture holds neither its reconstruction nor its coding structure's per-4x4 arrays (motion,
  intra modes, CCP and EIP indices, QT depth): EncLib releases them after reading the picture
  (`Picture::releaseSearchBuffers`), `xStartPicSearch` creates them (`createSearchBuffers`); `getPicWidthInLumaSamples`
  reads the picture's area instead of the reconstruction. RA 1080p with 8 threads and 8 periods, minutes 2.5-5:
  34.0 -> 23.6 GB on average (peaks 38.7 -> 27.7 GB). The automatic FramePeriods counts a waiting picture at 17 bytes
  per luma sample (measured about 13).
- P5 RAM step T: a waiting picture holds of its temporary buffers only the true original, and only if the temporal
  filter changed its original (`EncTemporalFilter::filtersFrame`, `Picture::createTrueOrigBuf`); `xStartPicSearch`
  makes the residual, prediction, true and filtered originals (copies of the original, which nothing changes in
  between: a check build compared them on 168 pictures). 1080p: about 25 -> 7 MB a waiting picture; RA 1080p with 8
  threads and 8 periods on six E-cores, minutes 2.5-5: 22.3 -> 17.4 GB on average. The automatic FramePeriods counts a
  waiting picture at 6 bytes per luma sample.
- RAM step U: `CodingStructure::freeCachedUnits` at the end of each picture's finish frees the CUs, PUs and TUs that
  `destroyTemporaryCsData` returned to the picture's own unit cache; before, every picture buffer kept the units of the
  largest picture it had coded (a cache per buffer since 03aa9a96; foreman CIF: 6.4 MB for an intra picture, about
  20 times that at 1080p).
- RAM step S: with several periods, `m_ftSaoStatsMutex` lets one finish at a time hold its SAO statistics
  (createEncData to destroyEncData: pre-deblocking statistics, deblocking, SAO, CCSAO); finishes of several periods
  at once had raised the peak by 0.64 GB each at 1080p. RA 1080p with 8 threads and 8 periods, 150 s: peak
  18.3 -> 14.7 GB (working set 12.6 -> 9.1 GB); RA 257 frames with 8 threads 282.1 s, as before.
- RAM step R1: the reused uni-prediction MV tables of a search stack (`InterSearch::ReusedUniMvs`, two per stack)
  are sized for the CTU size in use instead of 256x256 CTUs: 2 x 167 -> 2 x 35 MB of commit per stack (calloc'ed,
  hardly touched). Peak commit with 8 threads: RA 65 frames 8.29 -> 6.14 GB, AI 33 pictures 8.31 -> 6.14 GB. An
  allocation probe (`C:\Code\work\mt\ram\allocprobe`: every live allocation with its call site and resident bytes,
  QueryWorkingSetEx) showed these tables as 2.7 GB of the 9.1 GB committed at 1080p before the first picture.
- RAM step R2: a waiting picture keeps its original and true original in 8 bits per sample where that is lossless
  (`Picture::packWaitingBufs` at the setup end, `unpackWaitingBufs` at the search start). RA 1080p with 8 threads and
  8 periods, 75-150 s: 11.8 -> 10.9 GB on average (working set 8.1 -> 7.3 GB).
- RAM step R3: the sub-PU, sub-spatial and sub-block motion buffers of EncCu, InterSearch and InterPrediction
  (`CtuMotionBufs<N>`) are sized for the CTU in use instead of MAX_CU_SIZE: about 34 MB less per search stack, in the
  working set. Peak commit with 8 threads: RA 65 frames 6.13 -> 5.85 GB, AI 33 pictures 6.14 -> 5.87 GB.
- RAM step R4: `BestEncInfoCache` makes a slot's TU buffers at its first store (`xCreateTuBuffers`) instead of all
  slots' at the first CTU (about 128 MB of commit per stack, under 10 % touched). Peak commit RA 65 frames with 8
  threads 5.85 -> 5.14 GB; 1.2 % faster on the suite (compact buffers).
- RAM step R5: `BestEncInfoCache` also makes a slot itself (CU, PU, TUs) at its first store (`xCreateSlot`), not
  every slot at create (about 85 MB per stack, all touched). Peak commit RA 65 frames with 8 threads 5.14 -> 4.37 GB;
  2.1 % faster on the suite.
- RAM step R6: the CCSAO edge statistics per CTU (`CcSaoCtuStatData`) sum in 32 bits: a running finish's SAO
  statistics 0.64 -> 0.43 GB at 1080p; 1.3 % faster on the suite. RA 1080p with 8 threads and 8 periods, 75-150 s,
  after R1-R6: about 8.9 GB on average (working set 6.0 GB), peak 9.6 GB (working set 6.9 GB).
- RAM step R7: a waiting picture holds no original when the input is a file and the temporal filter left the frame
  as read: the setup drops it (`Picture::dropWaitingOrig`) and the search start reads the frame again
  (`EncTemporalFilter::readFrame`, the filter's own reader, initialized for every encode now; the frame index comes
  from `EncLib::encodePrep`). Filtered pictures keep their packed originals as in R2. 1080p RA with no options (20
  threads, 10 periods), 75-150 s: commit 14.07 -> 13.31 GB, working set 8.04 -> 7.27 GB. A check build
  (`-DORIG_RELOAD_CHECK=1`) compares every frame read again with the dropped one. Found on the way: the application
  sets the temporal filter's input file after `EncLib::init`, so the decision is made at the first setup
  (`EncLib::isOrigReloadable`).
- RAM step R8: `MotionInfo` (one per 4x4 block of every picture and temporary coding structure) no longer carries the
  CU's additional hypotheses (`MultiHypVec`, 136 of its 224 bytes): the motion field never provided them (the span
  functions build entries without them; merge candidates inherit hypotheses from the HMVP table only), so the table
  entries keep them in `HmvpMotionInfo`. 224 -> 88 bytes; 1080p RA with no options, 75-150 s: commit 13.31 -> 12.47 GB,
  working set 7.27 -> 6.45 GB. The decoder shares the code (decoded three bitstreams to the encoder's reconstruction).
  Allocation probe at 20 threads (`C:\Code\work\mtam\probe2`, exe `bin/allocprobe2`, `dumps.py` for sites over
  time): at 1080p a finish's ALF covariance statistics are 1.04 GB (147 per CTU x 52 KB; 4.2 GB at 4K), created
  before `ALFProcess` and freed after it, so concurrent finishes add up; `InterSearch::ReusedUniMvs` commits 66 MB a
  stack with 30 KB used (R10); at CIF after 900 s nearly every picture of the window (286 of 288) held its motion
  field, original and reconstruction: finished pictures keep them until their buffer is reused, and completed periods
  wait for earlier ones' output (R11).
- Speed step S1: in all intra, the pictures waiting for a search stack take them in order of decreasing luma activity
  (`EncGOP::xFtSearchPriority`, the cost estimate of `--ParallelSegments`) instead of coding order, so the expensive
  pictures do not start last. AI 33 pictures, 2 runs each alternating with the previous build: 8 P-cores 34.5 -> 33.0 s
  (4.5 %), all 20 cores 24.8 -> 23.6 s (4.9 %). Less than the schedule simulation's 23.3 -> 16.3 s: the setups run in
  coding order and each picture takes a free stack as soon as its setup ends, so the first N pictures still start in
  coding order; only the pictures that wait are reordered (more of them in a long sequence). 20 processes of
  `--ParallelSegments` take 16.8 s: they keep 17-19 cores busy to 80 % of the run, the frame threads 17-18 cores to
  50 % and fewer after (`C:\Code\work\mt\cpu_trace.py`). RA and LD keep coding order.
- Speed step S2: `--FrameThreads=0`, now the default, picks the number of frame threads (`EncAppCfg::
  xAutomaticFrameThreads`): the logical processors of the encoder's affinity (`ProcessorCount.cpp`), at most one per
  coded picture, fewer if 70 % of the physical memory does not hold them with one period
  (`xFrameThreadsMemoryBytes`, the figures of the automatic FramePeriods); 1 for low delay in one period and for the
  coding tools frame threads do not support (`xFrameThreadsUnsupported`; FrameThreads > 1 is an error there as
  before). `--ParallelSegments` workers get `--FrameThreads=1` unless FrameThreads is given. No options, unpinned on
  the 265K (20 threads chosen): AI 33 pictures 201.2 -> 23.1 s (8.7x), RA 65 frames 540.0 -> 202.7 s (2.66x; two
  periods, chain-bound: 2.8 busy cores), RA 513 frames 327.8 s (8 threads on the 8 P-cores took 612.3 s). `bench.py`
  pins before the encoder starts, so a pinned run picks one thread.
- Speed step S4: in all intra, while pictures are still being passed in, a picture takes a search stack only when the
  input side waits in `waitFrameThreads` (at the in-flight limit or at the end) and every picture passed is set up
  (`m_ftInputWaits`), so S1's order covers the first wave too (the setups take microseconds, so before, the first N
  pictures took the stacks in coding order). Fix of S1 (42c9fbd3) in the same step: the earliest picture not
  finished may take the stack kept for it even when it is not first in priority order; before, that stack stayed
  idle while a cheap early picture waited behind expensive ones (8 threads: 29.1 -> 32.4 s with the first-wave change
  alone). AI 33 pictures, 2 runs each alternating with the previous commit: 20 threads unpinned 21.85 -> 19.0 s,
  8 threads on the P-cores 29.1 -> 29.2 s (coding order was already balanced there: about 7 s of CPU a picture on 8
  cores). `--ParallelSegments=20` takes 16.8 s. RA and LD: unchanged (coding order; the earliest is always first).
- Speed step S5: `EncLib::create` and `init` make the search stacks and the finish objects in threads of their own
  (`runInParallel`): 20 stacks took about a second serially before the first picture could start (memory-bound:
  allocation and first touch of about 0.6 GB each). AI 33 pictures with 20 threads 21.6 -> 20.65 s (2 runs each).
  Bounds found with `sim_prio.py` (the trace's costs, 17 periods): with unlimited cores the sequence reaches 14.8x;
  the finish order costs 8 % of that and the same-layer CABAC chain 0.2 % (the chain pictures of layers 4 and 5 are no
  references, so a search-end context store would cut it, for nothing). A period's critical path is its reference
  chain plus the layer-4 chain (I, T1, T2, T3, then 8 layer-4 pictures each waiting for the previous one's contexts):
  about half the period's work, so a period alone gains at most about 2x, and the heaviest period's chain (333 s)
  exceeds the ideal 20-core time (246 s): long RA is bound by its heaviest period, bit-exactly. Only shorter pictures
  (wavefront or tile parallelism inside a picture, which changes the bitstream) move that.
- RA 513 frames with 20 threads, timeline (`ECM_FT_TRACE` probe, `C:\Code\work\mt\trace_patch2.py`): all 16
  periods start at once and all 20 stacks search from about 60 to 160 s; the second half is a tail (16 down to 1
  searching) of the heaviest periods' chains (period cost varies 6x on this sequence). The schedule simulation
  (`C:\Code\work\mt\sim_prio.py`) gives 13 % for the critical-path order with the true costs, but no order without
  them was better on both cost sets tried (per-layer critical path -6.8 % / +3.3 %, lowest layer first +0.5 % /
  -4.9 %), and the input measures do not predict a period's cost here (spatial activity r = -0.77, frame differences
  r = -0.15).
- Tried and dropped (speed): `--FinishSaoStats=K`, letting K finishes of different periods hold their SAO statistics
  at once instead of one (RAM step S; `C:\Code\work\mt\s3_patch.py`): RA 513 frames with 20 threads 360.5 s with
  one, 367.4 s with 17 (the finishes are 3 % of the work; within a period they run in coding order anyway).
- Open: one segmentation fault in RA 513 frames with 20 threads (a probe build of 42c9fbd3, about 46 s in, right
  after POC 16's output); not reproduced since in 13 full runs (release and probe builds, with and without gdb) and
  25 runs of the first 70 s (`C:\Code\work\mt\crash_loop3.sh`). The encoder now prints the faulting thread's call
  stack on a crash (`CrashReport.cpp`); resolve the addresses with addr2line on a -g build of the same source.
- Tried and dropped (RAM): freeing the original and the per-4x4 arrays of a finished picture that waits only for its
  output and is no reference (`C:\Code\work\mt\ram\w_dropped.diff`). In the first minutes at 1080p with 8 threads
  and 8 periods the list holds about 280 waiting pictures, 9 being searched and at most 13 finished ones, all still
  references (census probe); 5-minute runs: 20.1 vs 19.1 GB without it (noise). The commit peaks come from finishes
  of several periods at once, each with its SAO/CCSAO statistics (0.64 GB at 1080p; the CCSAO edge statistics keep
  every candidate of every CTU for the per-CTU decisions).
- P2 for LD, motion search per reference picture (not committed; `C:\Code\work\mt\me_patch.py`, `me_job.cpp`):
  the AMVP and search of each reference picture of predInterSearch's uni-prediction loop on the CuThreads helpers
  (an InterSearch per helper stack), the loop replaying the results with the PU fields, RdCost predictor and cost
  scale, TM-AMVP buffer entry and the CU's fill-once caches they leave; a check mode compares with the serial search.
  Bit-exact on LD foreman, but the loop is only 6.5 % of LD 128x128 (7.3 of 113 s, 57 % of its calls LIC CUs that the
  helpers did not take): the jobs saved about 1.4 s, and workers spinning on 3 more P-cores cost the main thread as
  much (LD 13 frames 113.7 s with one thread, 114.3 s with 4, 112.9 s with 2; with the 2 ms spin of EncCuWorkers,
  sleeping workers made it slower: 115.4 s with 4, 125.8 s with 8). UMWAIT instead of spinning and helpers on the
  efficiency cores did not help either.
- Found by the matrix of stage A: `--FrameThreads` >= 3 stopped with "PDP predictor unavailable" on ra_cif (bug of
  cfe3c39a, fixed in its own commit): a search stack's first picture used IntraPrediction's PDP buffer pointers before
  any CU of that stack had set them.
- Unpinned runs on this hybrid CPU were slower than pinned ones even with one thread (LD 17 frames: 201 s vs 174 s on
  P-cores): Windows throttled the background encoder (EcoQoS) and put it on E-cores. The encoder now opts out
  (`--PowerThrottling`, next commit): unpinned single-thread runs 1.24-1.26x faster, AI with 8 frame threads 1.15x.

## Tools in perf/mt

- `classify_profile.py samples.txt EncoderApp.exe`: splits an sprof profile into the classes used below.
- `static_audit.py <mingw bin> out.txt <obj dirs>`: lists every mutable global, class static, function static and
  `thread_local` in the object files of a non-LTO build.
- `phase_audit.py <compressGOP start> <setup end> <finish start> <picture end>`: per `EncGOP` member, which phases of
  `compressGOP` (setup, search, finish, with the helpers they call) write and read it, flagged where frame threads
  could see another picture's value. Heuristic: check each flag by hand.
- `ft_sim.py <serial log> [finish fraction] [GOP size]`: frame-thread schedule simulation (references, same-layer
  CABAC chain reset at IRAPs, stacks in coding order, finishes in coding order) for 1-32 threads.
- `grain_probe/insert_grain_probes.py <tree>`: adds timers to every candidate loop in a throwaway checkout of 2912a239
  and prints, at exit, the speed-up a 2/4/8/16-worker pool would reach with 2 us overhead per task (results in
  Progress). Its task times include each candidate's setup.
- `p5_sim.py <serial log> [finish fraction] [periods]`: event-driven schedule with K threads, S stacks, C finish
  contexts and W pictures in flight, stacks in coding order (stage A) or to ready pictures (stage B); can extend the
  log's last intra period to model longer sequences.
- `cu_matrix.py EXE TAG REFEXE`: the bit-exactness matrix of the thread options (default, CuThreads 2/4/8, the
  check, FrameThreads 2/4/8 and combinations), jobs on their own P-cores; `cu_scale.py EXE tests counts`: CuThreads timings, one run at
  a time on P-cores, with each run's peak commit.
- `p5_test.py REFS|REFEXE EXE TAG tests threads [args]`: frame-thread output check (bitstream and reconstruction md5
  against the single-thread references in `p5_refs.txt`, time and peak commit per run; `P5_MASK` = CPU mask).
- `p5_par.py OUT EXE TAG mask:tests:threads[:args] ...`: several p5_test.py jobs at once, each on its own CPU mask.
- `qmatrix.py EXE TAG REFEXE [groups]`: the matrix with every job on its own core pair, all at once.
- Per step (correctness, about 5-10 minutes, the machine shared): `qmatrix.py EXE tag REFEXE default,ft8` and
  `p5_par.py out EXE tag C3:ra65:8 3C00:ra33:4:--FramePeriods=2 3C000:ai33s8,ld13:4` (E-core and P-core masks of the
  265K). Times only on an idle machine and once per series of steps: `bench.py ab` for the default path, p5_test.py
  ra257 8 alone (or two builds at once on the two halves of the P-cores). No HD encodes for checks: a 1080p frame
  takes 20-40 minutes on one core.

## Measurements (i7-10700 8C/16T, 64 GB, Windows 11, GCC 16.2)

### What is already parallel

Segment-parallel encoding (`--ParallelSegments`, child processes per intra period or per picture) gives 5.51x on
ra_long (8 chunks) and 6.66x on ai_long (16 workers). It does nothing for low delay (IntraPeriod -1): **LD encodes
use one core today**, so parallelism inside a picture is the only way to speed them up. RA chunks also encode one
intra picture twice.

### Where the time goes

`compressSlice` (the CTU mode search) is 98.7 % of an RA encode; loop-filter encoders 1.3 %, temporal pre-filter
0.1 %. Profiles split with `classify_profile.py` (encProf = non-LTO build with symbols; RA/AI ParkJoy and AI
FourPeople were large inputs, 720p judging from bits and times, the command lines were not logged):

| class | RA ParkJoy (I + 2 B) | RA foreman CIF 17 f | LD ParkJoy 256x256 17 f | AI ParkJoy 2 f | AI FourPeople |
|---|---|---|---|---|---|
| cand_merge (list, refinements, SATD pass) | 14.3 | 30.7 | 16.3 | 0 | 0 |
| cand_geo (GPM MC, SAD tables, blends) | 10.0 | 14.0 | 13.5 | 0 | 0 |
| cand_intra_luma (rough mode decision) | 9.3 | 7.0 | 8.3 | 21.3 | 27.8 |
| cand_intra_derive (TMP search, TIMD/DIMD, CU bits) | 4.8 | 3.8 | 3.4 | 12.1 | 16.3 |
| cand_intra_chroma | 6.5 | 3.5 | 2.3 | 4.3 | 3.5 |
| **candidate passes total** | **44.9** | **59.0** | **43.8** | **37.8** | **47.7** |
| rd_intra_luma | 18.1 | 9.0 | 9.8 | 48.6 | 40.2 |
| rd_intra_chroma | 3.5 | 1.4 | 1.4 | 3.9 | 2.8 |
| rd_inter (merge, GPM, ME candidates) | 19.0 | 9.9 | 22.0 | 1.7 | 0.7 |
| OBMC inside full RD | 3.7 | 5.3 | 2.4 | 0.1 | 0.1 |
| **full-RD loops total** | **44.3** | **25.6** | **35.6** | **54.3** | **43.8** |
| me (motion and BV search) | 7.6 | 9.8 | 15.5 | 4.6 | 4.1 |
| filter + cu_other + other | 3.1 | 5.6 | 5.1 | 3.3 | 4.4 |

Per-picture encode times (decFix2 = LTO build of `dec`, one core each, seconds):
- RA foreman CIF QP 32, first 18 coded pictures: I0 118, I32 138, POC16 (T1) 227, POC8 (T2) 237, T3 139/171,
  T4 66-105, T5 21-30.
- LD ParkJoy 256x256 QP 32: I 117, then B pictures from 4 to 340 s depending on the GOP position (the last
  picture of each 8-picture GOP is the most expensive).
- ra_long (foreman 128x128, 257 frames): the frame-level dependency bound (sum of picture times divided by the longest
  chain, `perf/ra_critical_path.py` on branch `perf`) is 1.57, 1.72, 1.85, 1.71, 1.61, 1.60, 1.28, 1.64x per intra
  period. The model counts reference pictures and the same-layer CABAC chain.
- Memory: a 1080p RA encoder holds 2.4 GB private (this includes the 33 input pictures of a GOP), so extra
  per-picture search objects fit easily in 64 GB.

## What cannot run in parallel bit-exactly (and why)

- **CTUs of a picture.** Every CTU's rate estimates start from the CABAC contexts the previous CTU left
  (`EncSlice::encodeCtus`), and the uni-MV and affine MV histories and `g_reusedUniMVs` are reset per slice only, so
  they carry from CTU to CTU (agent). Speculating on the start contexts does not work: decisions depend on every
  context, and contexts used rarely never converge within a picture.
- **Split branches of a CU node.** The caches are first-visitor-wins: `BestEncInfoCache` (`ETM_RECO_CACHED`),
  `CacheBlkInfoCtrl`, `SaveLoadEncInfoSbt`, `g_reusedUniMVs`. TT/QT skip rules read the BT results, and every
  child gets `newMaxCostAllowed` from the parent's best cost and its earlier siblings (`EncCu.cpp` ~3003, agent).
  Most sub-blocks occur in several split paths, so conflicts are the rule, not the exception.
- **The tests of one node** (order in a B slice: merge, GPM, ME imv0, LIC, imv1/3/4, MHP, IBC, separate-tree
  intra, splits). GPM needs merge's candidate list and best SATD; intra reads the inter Hadamard cost and best
  inter cost; MHP consumes every earlier ME result; IMV, LIC and BCW skip rules chain them (agent).
- **The same-layer CABAC chain in RA**: a picture loads the contexts that the previous picture with the same
  temporal layer and QP stored at the end of its `encodeSlice`. All 16 layer-5 pictures of a GOP32 form one chain, which is what limits
  frame-level parallelism to 1.3-1.85x.

The way around all four is the same: run more work concurrently inside each link of the chain, and overlap
independent chains.

## Plan, ranked by expected gain

Estimates are Amdahl from the profile shares above, with an assumed effective parallelism of 4 for deterministic
loops and 2.5 for speculative ones on 8 cores; the grain measurement (next step 1) should replace them.

### P0 Infrastructure (needed by everything)

1. **Search stacks.** One `EncSearchStack` per picture or worker holding `InterSearch`, `IntraSearch`,
   `BilateralFilter`, `TrQuant`, `CABACEncoder`, `RdCost`, `CtxCache`, `EncCu`, `EncSlice`, a snapshot of the
   reshaper (`EncReshape` holds only values, so copy-assignment works), a `LoopFilter` for the deblocking-aware RDO
   (`EncCu` calls `xDeblockCU`, `EncCu.cpp` 30050-30311) and the MC padding `InterPrediction`. Today these are
   single members of `EncLib` (`EncLib.h` 88-182), created in `EncLib::create` (105) and wired in `EncLib::init`
   (731-785). `EncCu::init` (`EncCu.cpp` 666) and `EncSlice::init` (`EncSlice.cpp` 93) fetch their partners through
   `EncLib` getters, so give them the stack explicitly. In `EncGOP` the per-picture objects are used in
   `compressGOP`, `xPicInitLMCS`, `xPicInitRateControl`, `preLoopFilterPicAndCalcDist` and
   `xFindDistortionPlaneWPSNR` only. The dead `ENABLE_SPLIT_PARALLELISM` code (`TypeDef.h` 865 = 0) shows the old
   VTM attempt at the same thing.
2. **Global state the mode search writes** (from `static_audit.py` on CommonLib and EncoderLib):
   - `g_picAmvpSbTmvpEnabledArea` (incremented in `CABACWriter.cpp` 5082): make it per stack.
   - `g_intraModeCost` and `g_timdMrgCost`: reset per CU at `EncCu.cpp` 3933. `g_timdMrgCost`'s only writer is
     compiled out (agent).
   - `g_reusedUniMVs` and `g_isReusedUniMVsFilled` (+ LIC): reset per slice at `EncSlice.cpp` 1808.
   - `g_compressPartitionerSST`: allocated with `new` for every CTU at `EncCu.cpp` 767 and never freed.
   - `g_encPartitionerSST` (`CABACWriter.cpp` 52).
   - `clipMv`/`clipMv2`: set per picture, normally to the same value. `DecSlice::selectClipMv` already handles this
     for the decoder.
   - `TComHash` CRC calculators: static members, used through `getBlockHashValue`.
   - RdCost statics: `m_reshapeLumaLevelToWeightPLUT`, `m_lumaLevelToWeightPLUT`, `m_signalType`,
     `m_chromaWeight`, `m_lumaBD` (`RdCost.h` 159-163). The LMCS weight table is rewritten in `xPicInitLMCS` at
     intra pictures (RA) or periodically (LD, `updateCtrl` 2), so it must become a per-stack copy.
   - EncoderLib function statics without `thread_local` (agent): `InterSearch.cpp` 4182,
     `IntraSearch.cpp` 7817-7828, 9734, 9802, 12345, 12361, `CABACWriter.cpp` 11895, 11969-11970.
   - `DQIntern::g_Rom` is filled lazily in the `DepQuant` constructor, so construct stacks one after the other.
   - Encoder pictures share `g_globalUnitCache`. The decoder already gives each picture its own `XUCache`
     (`Picture::unitCache`, `Picture.h` 381); do the same.
   - Existing `thread_local`s are per-call scratch, except `InterPrediction.cpp` 26744 (`scale_s`, `shift_s`,
     `offset_s`), which carries LIC parameters between two consecutive calls of one invocation; that is fine on one
     thread.
   - MinGW GCC implements `thread_local` with emutls, a function call per access, so keep it out of hot loops and
     prefer stack members.
3. **Worker pool** with spinning workers (tasks of a few microseconds), each bound to its own search stack.
4. **Shadow check mode**: run a parallel region also serially on the main stack and compare its outputs. This finds
   missing per-CU state long before a bitstream md5 differs.

### P1 Frame-level threads for RA (expected 1.3-1.85x per intra period; nothing for LD/AI)

`EncLib::encode` calls `EncGOP::compressGOP` once per picture (`picIdInGOP`), and the app writes output after the
GOP, so a scheduler can run a whole GOP in the first call. Split each picture into:

- **setup**: `compressGOP` from its start (`EncGOP.cpp` 2057) to `precompressSlice`/`compressSlice` (3796/3800). Runs
  in coding order on the picture's stack. It touches shared state: `m_iLastIDR`, RPL marking,
  `xUpdateRasInit`, picture headers and the master reshaper.
- **mode search**: starts as soon as its inputs from earlier pictures are final (table below).
- **finish**: loop filters, APS, AMaxBT statistics (`EncGOP.cpp` 4704), `encodeSlice`, NAL output, PSNR. Runs in
  coding order after the previous picture's finish. The filter encoders' cross-picture state then needs no change:
  APS map, `g_pictureControlInfo`, CCSAO history `g_ccSaoPrvParam`, ALF scale statics, SAO disabled rate per layer.

| input of the mode search from earlier pictures | written | how to keep the serial value |
|---|---|---|
| reference pictures | finish of each reference | wait for it |
| temporal CABAC store: key (B/P, layer, QP), 5 entries per type (`CommonDef.h` 2135), cleared at RAS init; a stored state replaces the probability states, weights and window sizes of all contexts (`Contexts.h` 1348-1540) | `EncSlice::encodeSlice` (2721) | per-picture load and store slots replayed in coding order at setup, as `DecLib::xSetupSliceState` (`DecLib.cpp` 1509) does for the decoder threads; the mode search waits for its load slot. The store lives in each `CABACEncoder` today, so it must become shared |
| CABAC init table `m_encCABACTableIdx` (`CABAC_INIT_PRESENT_FLAG` = 1) | `encodeSlice` of the previous picture in coding order (`EncSlice.cpp` 2709) | read in setup (`EncGOP.cpp` 2665-2675) and used by `initCtxModels` (`CABACWriter.cpp` 59). It only matters for the mode search when no stored state is loaded, i.e. the first picture of a key in a period, whose predecessor is a reference in the CTC GOPs. The slice header bit must still get the serial value at write time |
| AMVP-SbTMVP area per layer | `encodeSlice` (2729) | slots, like the decoder (`DecLib.cpp` 1549-1571) |
| AMaxBT block statistics per depth, cleared at I slices | finish (`EncGOP.cpp` 4704-4714) | read and reset in setup (2566-2597); move that read to the mode-search start, after the finish of the previous picture of the same depth |
| LMCS model and RdCost LMCS weight table | setup (`xPicInitLMCS`, `EncGOP.cpp` 1891) | master copies updated in setup, snapshot into the picture's stack at the end of setup; finish uses the snapshot (its `getUseSliceReshaper()` is per picture) |

Validation order: K stacks with one thread, picture i on stack i mod K, must be bit-exact before any thread is
started; this finds state carried inside the stack objects. Then 1/2/4/8/16 threads, repeated runs, suite +
ra_long + 720p/1080p RA. The finish stage stays serial, so its filter statistics (P4) end up on the critical path.

### P2 Deterministic parallel loops inside a picture (all configs; LD's main gain)

These loops compute per-candidate results from inputs fixed before the loop. Run the iterations on the workers,
then do the ranking (`updateCandList` / `orderCandList`, strict `<`, earlier candidate wins ties) serially in the
original order. Estimated 1.5-1.9x on its own.

- **Intra luma, first-pass RD candidates.** rd_intra_luma (all passes, ISP included) is the largest single item:
  18 % of RA, 40-49 % of AI; the first pass is its main part (the grain probe splits it by pass). In the DCT2/no-LFNST
  pass, every non-ISP candidate starts from `ctxStart` with no cost threshold (`IntraSearch.cpp` 6709-6767), and the
  winner is chosen by strict `<`.
  - Each worker needs a private view of the CU's picture area, because candidates write reco and pred into
    `cs.picture`.
  - The side lists must be re-serialised: `m_regIntraRDListWithCosts`, `m_modesForMTS`, `m_uiRdModeCostList`.
  - `m_coeffAbsSumDCT2` is per-candidate scratch.
  - ISP candidates stay serial: `xGetNextISPMode` adapts the next mode to the previous results (agent).
- **Intra rough mode decision** (7-28 %): angular SATD, MIP, TIMD, SGPM, EIP, TMP integer search.
  - The distortion part is parallel.
  - The mode bits are not independent: with PNN off, the `DimdFlag` context is not restored between candidates
    and drifts in serial order (`IntraSearch.cpp` 1004-1015 vs 19960; agent). So compute the bits serially (about
    1 % of the time) or replay the context.
  - TMP's search (`candidateSearchIntra`, 2 %) can split its search region with a deterministic min-reduce.
- **Merge SATD pass** (`EncCu.cpp` 9349-9965).
  - Every candidate restores `ctxStart` before its rate (`xCalcPuMeBits`), and there is no early exit inside.
  - Re-serialise `m_baseResultsForMH`.
  - Prediction buffers are swapped by rank, so each worker needs private buffers.
  - The list-building refinements are per candidate too: TM `deriveTMMv`, BDMVR, affine TM, CIIP-TM (`EncCu.cpp`
    5932-9150). `dmvr4TMImpreciseMv` is an OR-reduction; the affine-TM append needs a prefix count (agent).
- **GPM**:
  - per-candidate MC (12209-12422);
  - 64 split modes x candidates SAD tables (12772-13005);
  - pair blend + SATD (13147-13532);
  - MMVD and TM stages (16191-16700).

  `m_mvBufEncBDOF4GPM` aliases the merge test's BDOF buffers, and `setGeoSplitModeToSyntaxTable` memoises into
  `InterSearch` tables (agent).
- **Motion search per reference picture** (uni loop `InterSearch.cpp` 6155-6250; bi iteration 6441). The
  searches are independent; reduce with strict `<` in index order.
  - Seeds (`m_uniMvList`, the `CacheBlkInfoCtrl` MV) are only written after the loop, or per (list, ref) key.
  - With `getFastMEForGenBLowDelayEnabled`, L1 copies L0's result for equal references (6187-6200; agent).
- **Chroma intra**: CCCM/CCLM/GLM SATD candidates restore `ctxStart` each; the RD loops are independent per
  candidate apart from the winner copy.

Each worker needs its own InterPrediction, InterSearch, IntraPrediction, TrQuant/DepQuant, CABAC estimator,
CtxCache, RdCost and CS buffers, plus the per-CU state of the main stack. The agents listed many of those buffers
(templates, LIC parameters, OBMC neighbour buffers, BDMVR pointers). Syncing this per-CU state is the main risk,
which is what the shadow check is for.

### P3 Speculative full-RD loops (another ~1.3-1.5x on top of P2)

- **Merge and GPM full-RD candidates** (`EncCu.cpp` 9995-11525, 15397-15944): ~22 % of RA, 24 % of LD.
- **Intra MTS/LFNST passes.** In the MTS pass, `m_globalBestCostStore` truncates candidate k's transform loop by the
  minimum over earlier candidates (`IntraSearch.cpp` ~17841; agent).
- **Inter residual options**: DCT2 → SBT → MTS → LFNST passes in `xEncodeInterResidual`.

They couple through `bestCS->cost` thresholds, `bestIsSkip` (FDM), `m_sbtCostSave`/`m_mtsCostSave`/`m_LNCostSave`
and the SBT save-load cache. Run candidates speculatively from a frozen state and record each threshold comparison
and cache read. Then commit in serial order: accept a result when the comparisons and reads come out the same
against the true state, otherwise recompute it serially. A recomputation is usually cheap, because it is the early
exit the serial code would have taken.

### P4 Data-parallel finish stage

Deblocking, SAO/CCSAO statistics and ALF classification and statistics per CTU. Only ~1-3 % of the time, but on
the RA critical path once P1 runs.

### P5 Overlapping intra periods in one process

A RAS-init picture resets the ALF APS map, APS IDs and `g_pictureControlInfo` (`EncAdaptiveLoopFilter.cpp`
2240-2274), the CCSAO history (`EncSampleAdaptiveOffset.cpp` 3021-3029) and the ALF scale statics. So it reads
nothing from the previous period. Its finish does overwrite state the previous period's last pictures still read in
coding order, so overlap needs one copy of the filter-encoder state per period in flight, plus an ordered output
buffer. It saves the intra picture segment-parallel encodes twice, and balances better. Lower priority, because
segment-parallel already covers long RA encodes.

### Order

P0 → P1 → P2 (intra first-pass RD and rough decision first, then merge, GPM, ME) → P3 → P4 → P5.

P1 comes before P2 because it is the bigger certain gain for RA and builds the stacks and globals P2 needs. If the
user cares most about LD, P2 should come first. Expected combined result on 8 cores (estimate): RA ~3x, LD and
AI-latency ~2-2.5x.

## Next steps

1. The grain probe and the first P2 step changed the ranking above; the user decided (2026-10-11): P5 first, then
   more P2 for LD, then any further threading gain. The two options were:
   - P5 first: let later GOPs start while earlier ones finish, in one process. AI pictures and RA intra periods read
     nothing from earlier periods after the RAS-init reset (`--ParallelSegments` relies on that), so AI throughput
     should approach the thread count (ParallelSegments: 6.66x with processes) and RA gain several times with
     FrameThreads, without the duplicated intra picture of RA chunks. This is what ParallelSegments already gives with
     processes; the new part is one process. Work: EncLib::encode returns while GOPs are in flight, picture buffers are
     reused only when done, setups and finishes keep coding order across GOPs, output in order.
   - More P2 for LD, the only configuration nothing else speeds up: merge SATD pass, ME per reference, the GPM stages,
     intra RMD in chunks, then P3 speculation for the merge/GPM full RD and the intra MTS pass. Expected 1.1-1.2x on
     LD in all, each loop a step like the first (state sync, record and compare, shadow check).
2. P0, P1, the first P2 step and P5 (stages A to C and the automatic periods) are done (see Progress). LD: the
   motion search per reference picture gained nothing on this CPU (see Progress); further P2 loops for LD are smaller
   still, so LD stays at the single-thread speed unless spinning workers get cheaper. RA with more threads than
   cores of one kind: RA 257 frames unpinned on all 20 cores took 283.5 s with 8 threads, 271.2 s with 12 and 272.7 s
   with 16 (6.39x, 6.68x, 6.64x; 9.9, 13.9 and 17.8 GB): the periods' chains bound it. Frame threads with CU threads
   for short RA, where the chains leave cores idle (8 P-cores): RA 65 frames 198.2 s with 8 frame threads, 203.9 s
   with 4 x 2 CU threads and 198.7 s with 8 x 2; RA 33 frames 190.3 s with 8, 190.0 s with 4 x 2 and 197.2 s with
   2 x 4: the CU helpers code intra candidates, little of an inter picture's work.
3. Report each step as step % and cumulative % with before/after numbers and commit hashes, one commit per step.
