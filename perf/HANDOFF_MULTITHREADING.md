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
- `p5_sim.py <serial log> [finish fraction]`: event-driven schedule of overlapping GOPs with K threads and S stacks,
  finishes in coding order globally or per intra period.
- `cu_matrix.py EXE TAG REFEXE`: the bit-exactness matrix of the thread options (default, CuThreads 2/4/8, the
  check, FrameThreads 2/4/8 and combinations), jobs on their own P-cores; `cu_scale.py EXE tests counts`: CuThreads timings, one run at
  a time on P-cores, with each run's peak commit.

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
- **Inter residual options**: DCT2 鈫?SBT 鈫?MTS 鈫?LFNST passes in `xEncodeInterResidual`.

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

P0 鈫?P1 鈫?P2 (intra first-pass RD and rough decision first, then merge, GPM, ME) 鈫?P3 鈫?P4 鈫?P5.

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
2. P0, P1, the first P2 step and P5 stage A are done (see Progress). Next: P5 stage B (Progress has the plan).
3. Report each step as step % and cumulative % with before/after numbers and commit hashes, one commit per step.
