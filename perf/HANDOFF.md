# Handoffs: the ECM-20.0 speed-up

This file holds two handoffs, one per line of work, which were separate files on their branches until the merge of
2026-10-14: part 1 the encoder speed-up (branch `perf`; the other PC's clone `C:\Code\opt`), part 2 the decoder
speed-up and the Vulkan GPU decoder (branch `dec-gpurecon`; worktree `C:\Code\ECM\wt-gr` on this PC).

---

# Handoff to the next Claude session on this PC

State on 2026-10-10. The user is switching Claude accounts. A new Claude session continues this work on the same PC
(Intel Core Ultra 7 265K, Windows 11).

Everything under `C:\Code` and the user's global `C:\Users\Brian\.claude\CLAUDE.md` stay. The previous session's
memory, scratchpad and claude.ai artifacts do not carry over.

## 0. Start here

1. Read this file, then `perf/OPTIMIZATIONS.md` (every step with its measured gain). Skim
   `git -C C:/Code/opt log --oneline 6d442b2a..`: each commit message has problem, cause, change, verification and
   limits.
2. The previous session's memory notes are copied to `C:\Code\tools\claude-notes\*.md`. They hold the same facts as
   this file, in short form.
3. Check the state:
   - `git -C C:/Code/opt status`: clean, branch `opt` at 7d41000f or a later handoff commit.
   - `git -C C:/Code/opt pull`: `opt` tracks `origin/perf` (the `merge` branch, which mirrored `perf`, was deleted on
     2026-10-14).
   - `gh auth status`: logged in as 4KVCD.
4. Before changing code, confirm the setup: run the suite with the current build. Every md5 must match
   `C:\Code\runs\results\ref.json`.

   ```bash
   python C:/Code/opt/perf/bench.py run check0 --exe C:/Code/bin/pgo5/EncoderApp.exe --jobs 5 --cpus 0,1,6,7,8
   ```
5. The user's global rules (`C:\Users\Brian\.claude\CLAUDE.md`) apply. For this project they mean section 2.

## 1. Where things stand

- **Branches** on github.com/4KVCD/ECM-speedup: `perf` (default) holds all kept work; `merge`, which only mirrored it,
  was deleted on 2026-10-14. Locally the work branch is `opt` in `C:\Code\opt`, which tracks `origin/perf`. The
  frame-thread work was done on `enc-mt` in `C:\Code\vk` (perf/HANDOFF_MULTITHREADING.md) and merged into `perf`.
  - `dev` has 3 commits from an earlier session on the user's other PC (an i7-10700) that are not merged; see
    section 9.
  - The stock baseline is 6d442b2a: tag ECM-20.0 (3391a5f0) plus two GCC 16 build fixes.
- **Rule: every bitstream is byte-identical to stock ECM-20.0.** The user confirmed this explicitly. Nothing may change
  a decision of the encoder; the work is only about computing the same thing faster.
- **Results on this PC:**
  - Encoder, one core: **1.51x** (CPU time, 5-test suite, paired; per test 1.44-1.60x).
  - Encoder with `--ParallelSegments`: all-intra 33 pictures **17.0x** with 20 processes; random access 257 frames
    **5.9x** with 8 processes (wall-clock vs stock on one core).
  - Decoder: **1.19-1.30x**. It was never profiled or tuned on its own; the gains come from the shared CommonLib code.
  - The README (the GitHub landing page) presents these numbers.
- **The original goal** was 10x single-thread with identical output. That is not reachable with the current approach:
  - The profile is flat. About 96 % of encoding time is `EncCu::compressCtu` (the mode search); 25 % of self time is
    in 12 functions, 50 % in 47, 90 % in 331.
  - Single changes now give 0.2-1 %.
  - The user then asked for multithreading "getting as much performance as possible" (section 6 and section 8).

## 2. Ground rules (the user's rules, as applied to this project)

- **Bit-exact**: compare the bitstream md5 of all 5 tests with `runs/results/ref.json`. Any difference means the change
  is wrong: find out why. Never tune it away.
- **Check builds** for anything that is not obviously equivalent:
  - Put the old code behind `#ifdef XXX_CHECK` next to the new code and compare the results on every call.
  - `abort()` with a message on the first difference.
  - Print the number of compared calls at exit (a static object's destructor).
  - Build with `-DCMAKE_CXX_FLAGS=-DXXX_CHECK` and run the suite. A counter of 0 means the path was never exercised.
  - Examples in the code: `MODEBITS_CHECK`, `GEOSORT_CHECK`, `CCCMAPPLY_CHECK`, `PDPOPT_CHECK`, `DIMDANG_CHECK`,
    `SIGNSORT_CHECK`, `AFFINE_NARROW_CHECK`.
- **Measure paired**: both builds of a test run at the same time on two performance cores, swapped every
  repetition. Use the median per-repetition CPU-time ratio (`bench.py ab`, section 4.3). Unpaired runs drift ±2 %.
- **Report every result** as "step" (vs the previous step) and "cumulative" (vs stock); the user asked for both.
- **Commits** (the messages so far are in `C:\Code\tools\scripts\c*.txt`):
  - One commit per kept step, made in `C:\Code\opt`, never in the main clone `C:\Code\ECM`.
  - Subject: what changed, plus "(bit-exact)".
  - Body: problem, cause, change, effect, verification (counters, md5, hardware), limits, and the benchmark table.
  - End with the `Co-Authored-By:` line.
  - No `git stash`, no rebase, no force-push. Amend only commits that are not on GitHub yet; otherwise add a commit
    that names the one it fixes.
- **Pushing**: only with the user's explicit yes, every time (section 4.7).
- **Process control**: kill only your own processes, by PID. Ask before downloading anything; the user allowed
  downloads for this project on 2026-10-07, but that was said to the previous session.

## 3. This PC: what exists and where

**Cores:**
- Performance cores (benchmarks): logical CPUs 0,1,6,7,8,9,18,19.
- Efficiency cores (builds): 2-5 and 10-17, affinity mask 0x3FC3C.
- `python perf/cores.py` prints them.
- An efficiency core runs this encoder at 0.76x of a performance core.

**Toolchain:**

| tool | version | where |
|---|---|---|
| GCC | WinLibs `gcc-16.2.0-mingw-w64ucrt-14.0.0-r2` (UCRT, posix-seh) | `C:\Code\tools\mingw64` |
| CMake | 4.4.4 | `C:\Program Files\CMake` |
| Ninja | 1.13 | winget package `Ninja-build.Ninja` |
| Python | 3.14 (no numpy) | `%LOCALAPPDATA%\Programs\Python\Python314` |
| GitHub CLI | logged in as 4KVCD | Windows keyring |

- `C:\Code\tools\env.ps1` puts these on PATH for the build scripts.
- The Bash tool's PATH lacks the MinGW tools. Prefix `export PATH=/c/Code/tools/mingw64/bin:$PATH` when a command
  needs g++, objdump, nm or addr2line.
- Git identity (global `.gitconfig`): Brian <thebrianzheng@gmail.com>.

**Directories:**
```
C:\Code\ECM      main clone, branch perf at 2ffd9ac3 (behind origin/perf; pull before using it). Never commit here.
                 bench.py reads its cfg files from C:\Code\ECM\cfg (unchanged from stock).
C:\Code\opt      work worktree, branch opt -> origin/perf (pull before using it)
C:\Code\vk       worktree, branch enc-mt -> origin/enc-mt (the frame-thread work; merged into perf)
C:\Code\stock    worktree at 6d442b2a (stock + GCC 16 fixes), built as C:\Code\bin\stock
C:\Code\probe, probe2, probe3, probe4
                 detached scratch worktrees for experiments and probes; reset before use:
                 git -C C:/Code/probe4 checkout -- . ; git -C C:/Code/probe4 checkout --detach <commit>
C:\Code\tools    env.ps1, mingw64\, bin\sprof.exe, scripts\, claude-notes\, landing\
C:\Code\seq      clips (section A.3), src\ (foreman_cif.y4m, park_joy_1080p50 and FourPeople 720p, first 12 frames),
                 foreman_352x288_300.yuv and ParkJoy_416x240.yuv for extra tests
C:\Code\bin\<n>  one folder per build (EncoderApp.exe, DecoderApp.exe)
C:\Code\build\<n> build trees
C:\Code\runs     results\*.json, ab_*.txt, logs, profiles
```

**Useful builds in `C:\Code\bin`:**

| build | what |
|---|---|
| `stock` | reference: stock 6d442b2a, `-O3`, static, no LTO, no `-march`. Every speedup is measured against it |
| `pgo5` | PGO build of f19e21c3, `-march=arrowlake`: the current best encoder and decoder. The encoder code of 5bcd8f6d is the same |
| `gr2` | LTO build of 49fa654f (`-march=arrowlake`): the last A/B baseline for code steps |
| `hotest` | LTO build of 5bcd8f6d (`-march=arrowlake`) |
| `cmg3` | LTO + `-g` build of c17d52d7; matches the profiles in `runs\prof\cmg3` |

**Useful files in `C:\Code\runs`:**
- `results\ref.json`: stock md5 and times; the md5 reference for every check.
- `ab_stock_pgo4.txt`: the 1.512x end-to-end run.
- `prof\cmg3\*.txt` and `all.txt`: stack samples of the 5 tests.
- `decbig\ra_cif.bin`, `decbig\ld_cif.bin`: larger streams for decoder benchmarks.
- `extra\`: AI QP 22, LD-P and RA 416x240 streams.
- `pgo5_serial\ra_long.log`: per-picture times; input for `ra_critical_path.py`.

**Scripts in `C:\Code\tools\scripts`** (outside the repo; the repo has copies of the general ones in `perf\`):
- `b.ps1 <name> <tree> [-Lto] [cmake args]`: the build used so far, pinned to the efficiency cores. The repo
  equivalent is `perf\build_pinned.ps1 <name> <tree> -Lto -Affinity 0x3FC3C`.
- `prof.ps1`, `cores.py`, `decbench_pair.py`, `ra_critical_path.py`, `incl.py`, `inl.py`, `hotaddr.py`.
- `*_patch.py`: the patch script of almost every step, a template for new ones.
- `*_dropped.diff`: the diffs of dropped experiments (section 7).
- `taskpool\`: the thread pool of the dropped intra-picture threading experiment.
- `c*.txt`: commit messages.

**Landing page:** the README on GitHub is the landing page. The earlier claude.ai artifact belongs to the old account.
Its source is `C:\Code\tools\landing\ecm-speedup.html`, if a new artifact is wanted.

## 4. Workflow

### 4.1 Profile

Build with symbols (`-DCMAKE_CXX_FLAGS=-g`; LTO keeps them), e.g.
`C:\Code\tools\scripts\b.ps1 prof1 C:\Code\opt -Lto -DECM_MARCH=arrowlake "-DCMAKE_CXX_FLAGS=-g"`.

Then `perf\prof.ps1 prof1` samples the 5 tests with `C:\Code\tools\bin\sprof.exe`, each pinned to a performance core,
into `C:\Code\runs\prof\prof1`. Rebuild the sampler with
`g++ -O2 -std=c++17 perf/sprof/sprof.cpp -o C:/Code/tools/bin/sprof.exe -ldbghelp -lpsapi -lwinmm`.

Reports (they need the MinGW `bin` on PATH):
- `sprof_report.py samples.txt EncoderApp.exe --top 60`: self and inclusive time.
  - `--callers F` / `--callees F` / `--callsites F`: who calls F, what F calls, from which lines.
  - `--lines F`: hot source lines of F.
- `sprof/incl.py exe file1,file2 func1,func2`: inclusive share of functions per test.
- `sprof/inl.py samples exe --grep FILE:LINE`: self time by innermost inlined line.
- `sprof/hotaddr.py exe FILE:LINE samples...`: disassembly around the hot addresses of a line.

### 4.2 Change, check, measure

1. Find structural waste in the profile (section 6), not just a hot function.
2. Implement it in `C:\Code\opt`, or in a probe worktree for experiments. For edits of more than a few lines, write
   a patch script (Python that reads and writes the files; see the `*_patch.py` scripts) so a step can be re-applied
   or dropped cleanly.
   - Never write C or Python with backslash escapes through bash heredocs or `python -c`: the shell eats them. Use the
     file tools.
3. If the change is not trivially equivalent, run a check build over the suite: no abort, md5 equal, counter > 0.
4. Run LTO builds A/B (4.3). Keep the step if the total gain is clear of noise and no test loses.
5. Commit (format in section 2). Save the diff of a dropped step as `C:\Code\tools\scripts\<name>_dropped.diff`.

### 4.3 Benchmarks (`perf/bench.py`)

```bash
python perf/bench.py run NAME --exe C:/Code/bin/NAME/EncoderApp.exe --jobs 5 --cpus 0,1,6,7,8   # md5 + times vs ref
python perf/bench.py ab OLD NEW --reps 4 --cpus 0,1,6,7,8,9    # paired A/B: per-test median CPU ratios, ~6 min
python perf/bench.py run NAME --exe ... --tests ai_long "--extra=--ParallelSegments=20" --jobs 1   # no --cpus here
```

- A/A noise: ±0.8 % per test, ±0.3 % total.
- Code layout alone moves single tests by several percent. On the i7-10700, a change that only touched GPM made
  all-intra 1.03-1.04x "faster" (all-intra never runs GPM) and ld_foreman 0.95x. Judge a change by the total and by
  the tests that run the changed code.
- Interactive use by the user (Chrome) pushed ra_foreman to ±5 %. Never build or run anything else while measuring;
  builds on the efficiency cores are fine during pinned CPU-time runs, but not during wall-clock runs that use all
  cores.
- Gains measured on LTO builds shrink to about 0.6x of their size under PGO.
- Re-run PGO only at milestones: `perf\pgo.ps1 pgoN -DECM_MARCH=arrowlake` (about 20 min, uses the machine). Then
  `bench.py ab stock pgoN --reps 4` for the end-to-end number.
- `--cpus` pins each encode to one core, and a segment-parallel run's child processes inherit it. Never pass
  `--cpus` with `--ParallelSegments`.
- The CPU column counts only the top process, so segment-parallel runs show about 0 s. The `dev` branch fixes this
  with a job object.

### 4.4 Decoder

`python perf/decbench_pair.py C:/Code/bin/stock/DecoderApp.exe C:/Code/bin/NEW/DecoderApp.exe C:/Code/runs/decX C:/Code/runs/decbig/ra_cif.bin,C:/Code/runs/decbig/ld_cif.bin --reps 6 --cpus 0,1`

- Both decoders run at the same time; the YUV md5 must be identical.
- It reports CPU time and the decoder's own timer. The timer excludes process start-up (about 0.2 s) but includes
  writing the YUV.

### 4.5 Segment-parallel

- `bench.py run ... --tests ai_long,ra_long "--extra=--ParallelSegments=N"`: the md5 must equal the stock serial ref.
- Reordering only happens with more chunks than workers. To test it on random access, use `--SegmentLength=32` with
  fewer workers than intra periods; for example, 97 frames on 2 workers gave the serial md5.

### 4.6 Frame-level bound

`python perf/ra_critical_path.py C:/Code/runs/pgo5_serial/ra_long.log 32` prints, per intra period, the work and the
critical path through references and the same-layer temporal CABAC chains. That is the most that encoding
random-access pictures concurrently could gain.

### 4.7 Pushing

Push only after the user says yes for that push. The Git Credential Manager dialog cannot open in Claude's shell,
so use the gh login:

```bash
git -C C:/Code/opt -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin opt:perf
```

It is a fast-forward when the branch contains `origin/perf`; otherwise merge `origin/perf` into it first (no rebase,
no force-push). `perf` is the default branch and shows the README.

## 5. Reference numbers (this PC)

| run | result |
|---|---|
| 5-test suite, stock vs 49fa654f+PGO (4 paired reps) | 1.512x CPU; ai_foreman 1.436, ai_parkjoy 1.440, ld_foreman 1.604, ra_foreman 1.523, ra_parkjoy 1.533 |
| ai_long serial: stock / pgo5 | 311.2 s / 210.7 s |
| ai_long, `--ParallelSegments` 8 / 20 (pgo5) | 34.9 s / 18.4 s |
| ra_long serial: stock / pgo5 | 2647.9 s / 1842.8 s |
| ra_long, 8 workers (pgo5) | 451.6 s |
| decoder, stock vs pgo5 | ra_cif 289 frames 1.22x, ld_cif 33 frames 1.19x, ra_long 1.30x, 97-frame RA 1.25x |
| extra encodes identical to stock | AI QP 22 foreman CIF 1 frame `6d142bad…`; LD-P foreman 176x144 3 frames `445f299e…`; RA ParkJoy 416x240 crop (752,420) 4 frames QP 37 `0d1880f7…` |

The `decbig` CIF streams were encoded by pgo5 only: foreman CIF RA, 289 frames, QP 32, 19 workers; LD, 33 frames,
QP 32. Their decodes were checked stock vs pgo5.

## 6. Lessons: what pays and what does not

**Pays:** structural waste in hot paths.
- Per-pixel calls or calls through function pointers (fused affine filter +2.4 %).
- Large clears of buffers that are mostly unused (GPM tables, 1.6 MB per CU, +1 %).
- Heap allocation in hot paths (allocator +2 %, static_vector for CCCM parameters +0.5 %).
- Member-by-member copies of large structs (+0.5 %).
- Branchy searches and sorts with mispredicted exits (GPM rank counting +1 %, DIMD angle lookup +0.7 %).
- Recomputation of results whose inputs did not change, with a complete, cheap validity check (context-store
  stamps +3 %, mode-bits memo, ALF reuse).
- Vectorizing kernels that ran scalar.

**Does not pay:**
- Result caches for cheap SIMD kernels with long reuse distance, even when exact. Examples: MC result cache (exact,
  34-80 % repeats, 0.972x), template-prediction cache (-3 %), DepQuant result cache (10 % repeats). The cold cache data
  costs more than recomputing from hot reference data.
- Caches whose key misses an input (TM refinement): not exact; never ship them.
- Wider SIMD on loops with an early exit per row (TMP joint SAD: 0.993x); packing small blocks to fill lanes (BIF:
  1.001x).
- Compiler flags: `-funroll-loops` (-2 %), `-O2` (-5 %), `-fipa-pta` (0.994x), `-flto-partition=one` (1.002x),
  `-fno-tree-loop-distribute-patterns` (1.00x). Clang 23 ThinLTO: +1.5 % on all-intra but crashes in inter coding
  (undefined behaviour in ECM).
- Memoizing short lookups: getCU walks average 3.1 parent steps; a cached parent chain lost 2 %.

**Multithreading** (details in `perf/OPTIMIZATIONS.md`, segment-parallel section):
- Segment-parallel encoding (`--ParallelSegments`) is the only large exact gain. It scales with the number of intra
  periods (random access) or pictures (all-intra).
- Inside a picture, CTUs are serial: the mode search of CTU n+1 needs the exact CABAC states and the reconstruction
  that CTU n leaves.
- Split subtrees share caches (`BestEncInfoCache`, `CacheBlkInfoCtrl`, IBC BV caches) and the picture's recon
  buffer, so a parallel split search cannot be exact.
- A thread pool for the IntraTMP scan was exact but 0 % faster (the scan is 2 % of the time). Busy-waiting workers
  also lower the main core's turbo.
- Between pictures, two things chain the pictures of each temporal layer: the temporal CABAC store (`Contexts.h`,
  `CtxStateStore`, key = temporal layer and slice QP, 5 entries) and the AMVP-SbTMVP area statistics
  (`g_picAmvpSbTmvpEnabledArea`, keyed by layer). Low delay needs each previous picture after ALF, so it is fully
  serial.

## 7. Dropped experiments (do not retry as they were)

Diffs are in `C:\Code\tools\scripts\*_dropped.diff`. Dropped on this PC, all bit-exact unless noted:
- Luma mode-bits memo extended to TMP/MIP/TIMD/EIP/SGPM/TMRL modes: 0.997x (`modebits2_dropped.diff`).
- Bypass-sign counting in codePredictedSigns: 1.001x.
- CCCM training-sample gather in AVX2 with compaction: 1.000x (`cccmgather_dropped.diff`).
- Sign-prediction cost loop in AVX2: 1.001x (`signpred_avx2_dropped.diff`).
- BDOF parameter sums via lane permutes: 0.999x / 0.997x (`bdofsum_dropped.diff`).
- TMP joint SAD two rows per step: 0.993x (`tmprows2_dropped.diff`).
- Bilateral filter with small blocks packed per AVX2 step: 1.001x (`bifrows_dropped.diff`).
- MC filter-section result cache: exact, 70 M hits checked, but 0.972x (`mccache_dropped.diff`).
- AVX2 DepQuant decision, AVX2 inverse transform on top of the zero skip, GEO std::sort of packed keys, inlined buffer
  accessors, a neighbour-availability shortcut: no gain.
- Thread pool for the IntraTMP sparse scan: exact, 0 % (`tmppar_dropped.diff`, pool in `taskpool\`).
- ParallelSegments workers pinned to performance/efficiency cores: slower (`segcores_dropped.diff`).

Dropped on the i7-10700: listed in `perf/OPTIMIZATIONS.md` (sparse context restore, MC caches, cached CU lookups,
wavefront check). The GPM carry-swap sort (1.00x) is logged on `dev`.

Probes that found little: intra CU re-evaluation duplicates (2-4 %); content-identical angular predictions (AI 1 %,
RA 18 % of samples, but hashing costs as much); chroma CCCM work repeated in later LFNST passes (about 0.2 %);
getCURestricted (38 % repeat one of the last 2 queries, but a memo costs what the walk saves).

## 8. What to try next

1. **Encode independent random-access pictures concurrently.** This is the largest remaining multithreading gain,
   and exact by construction if every shared state is handled. The user has not decided on it yet; it is several
   days of work.
   - Bound from `ra_critical_path.py` on ra_long: 1.27-1.80x per intra period. ra_long with 8 workers could go from
     451.6 s to about 250 s.
   - Needs per-picture encoder object stacks. The leftover `ENABLE_SPLIT_PARALLELISM` code in EncLib shows how
     EncCu/IntraSearch/InterSearch/TrQuant/RdCost/CABAC stacks were duplicated, but it is out of date.
   - Needs a split of the 3,300-line `EncGOP::compressGOP` loop into setup / mode search / loop filters + write.
   - State to audit:
     - the temporal CABAC store
     - `g_picAmvpSbTmvpEnabledArea` and the per-layer area statistics
     - the ALF APS state and `g_pictureControlInfo`
     - `g_globalUnitCache`
     - about 50 function-local statics (`grep -n "^\s*static [^c(]"`); some, like `cclmSAD`/`cccmSAD` in
       IntraPrediction.cpp, carry state between calls
     - `g_timdMrgCost`, `g_intraModeCost`
2. **Decoder**: profile it (never done), and train PGO with decoder runs too (`pgo.ps1` trains only the encoder).
3. **Merge `dev`** if the user wants it (section 9).
4. **Single-thread leftovers**, each probably 0.2-0.5 %: CU re-initialization in `initSubStructure`, OBMC work reuse
   (`subBlockOBMC` is 6.9 % inclusive in ra_foreman), and items in section 7's probes.
5. A bit-exact WPP/tile mode is not possible. Any mode that changes the bitstream for more threads, for low delay or
   for single pictures, is a separate product decision for the user.

## 9. Known issues

- `dev` holds three commits not in `perf`, from the other PC:
  - 05b53351: the allocator with an extra canary build mode `-DFASTALLOC_CANARY=1` (guard bytes, poisoning).
  - df395834: `bench.py` counts the CPU time of all processes of a run through a job object.
  - f07bd2f1: log notes on the dropped GPM carry-swap sort, layout noise and CPU accounting.

  The previous session did not merge them, following the rule "don't merge other sessions' branches"; ask the user.
  A merge conflicts in `FastAlloc.cpp`: take dev's file, which is perf's plus the canary mode, and confirm the default
  build gives the same md5.
- Stock ECM-20.0 does not compile with `ENABLE_HIGH_BITDEPTH=ON` or with `JVET_AG0061_INTER_LFNST_NSPT` set to 0.
  bd1be75c keeps the new code ready for both.
- Windows Python cannot open paths over 260 characters, and the Claude scratchpad path is longer. Use the Bash or
  file tools for files there, and keep work files under `C:\Code`.

## Appendix A: setting up a different computer

### A.1 Tools and layout

Install the toolchain of section 3, and copy `perf/env.example.ps1` to `C:\Code\tools\env.ps1`. Then:

```bash
git clone https://github.com/4KVCD/ECM-speedup.git C:/Code/ECM
cd C:/Code/ECM
git worktree add -b work C:/Code/opt origin/perf
git worktree add --detach C:/Code/stock 6d442b2a
```

### A.2 Builds

```powershell
perf\build_pinned.ps1 stock C:\Code\stock                          # reference: -O3, no LTO, no -march
perf\build_pinned.ps1 base C:\Code\opt -Lto -DECM_MARCH=native      # or the CPU's -march (znver4, arrowlake, ...)
perf\pgo.ps1 pgo1 -DECM_MARCH=native                                # instrumented build, 6 training encodes, rebuild
```

CMakeLists already adds what bit-exactness and MinGW need:
- `-ffp-contract=off`: RD costs are doubles, and FMA contraction changes them.
- `-Wa,-muse-unaligned-vector-move`: GCC cannot realign the stack for AVX spills on Win64.
- `-fno-aggressive-loop-optimizations`: stock ALF code reads past an array bound that GCC 16 would exploit.

`ECM_MARCH` empty means the stock SSE4.1 build. The SADL (intra NN) units always stay on the stock ISA.

### A.3 Clips

Sources: https://media.xiph.org/video/derf/y4m/ (`foreman_cif.y4m`, `park_joy_1080p50.y4m`,
`FourPeople_1280x720_60.y4m`, `crowd_run_1080p50.y4m`; only the first frames are used).

```bash
python perf/y4m2yuv.py foreman_cif.y4m            C:/Code/seq/foreman_352x288.yuv     10
python perf/y4m2yuv.py foreman_cif.y4m            C:/Code/seq/foreman_176x144.yuv     10  176 144  88  72
python perf/y4m2yuv.py foreman_cif.y4m            C:/Code/seq/foreman_128x128_257.yuv 257 128 128 112  80
python perf/y4m2yuv.py park_joy_1080p50.y4m       C:/Code/seq/ParkJoy_256x256.yuv     10  256 256 832 412
python perf/y4m2yuv.py FourPeople_1280x720_60.y4m C:/Code/seq/FourPeople_256x256.yuv  10  256 256 512 232
python perf/y4m2yuv.py crowd_run_1080p50.y4m      C:/Code/seq/CrowdRun_256x256.yuv    10  256 256 832 412
```

md5 on this PC (all but CrowdRun re-made from the sources on 2026-10-10 and matched):

| clip | md5 |
|---|---|
| foreman_352x288.yuv | e9196f8bb1a25503d000c5d8e92fdb4e |
| foreman_176x144.yuv | 0b4abe3ae59ab26f137d370d42e1c71d |
| foreman_128x128_257.yuv | dbf9b50e3491b269ae8512d68278fad6 |
| ParkJoy_256x256.yuv | 9da12de43694a2998ed2c746aa798a0a |
| FourPeople_256x256.yuv | 30ae8f86a7c50ef855a3f09db243f827 |
| CrowdRun_256x256.yuv | 868e66359ca196f59fd1a9737cdb0428 |

FourPeople and CrowdRun are used only to train PGO.

### A.4 Reference md5 (stock encoder, these clips)

`python perf/bench.py run ref --exe C:/Code/bin/stock/EncoderApp.exe --jobs 5`, then again with
`--tests ai_long,ra_long`. With the same clips and compiler these md5 come out:

| test | md5 |
|---|---|
| ai_foreman | 49b14890e727fa82c2cf7ba8ab854f3f |
| ai_parkjoy | 4efed6bf8631e34c8d6d24446d04308e |
| ld_foreman | 12becc9afbf268afc998cd6aaeeccaca |
| ra_foreman | b93c9b3d7b66fb128bf09b101dd36193 |
| ra_parkjoy | 913b6da3b7e03f8b6271c6cb6615d6d0 |
| ai_long | 1f53b8f377235cfc5a977e12e11bd5de |
| ra_long | 3072c03937f193a5ef4e467d5345773b |

If they differ, compare clips and toolchain before anything else. The i7-10700's references came from different
crops, so its md5 differ.

### A.5 Cores

- Hybrid Intel: measure on the performance cores and build on the efficiency cores.
- AMD or other SMT CPUs: use one logical processor per physical core and leave the siblings idle while measuring.

---

# Handoff: ECM-20.0 decoder speed-up and the Vulkan GPU decoder

Written 2026-10-10 on the i7-10700 PC so that another computer (and its Claude session) can continue; last updated
2026-10-15 at M66, when the work moved to another computer. Read sections 1, 1a, 2, 9 first, then the M-level notes in
section 7 that touch your next step, then `perf/DECODER.md` (the decoder speed-up log, incl. "GPU reconstruction").

## 1. Goal and status in one paragraph

The user wants the JVET ECM-20.0 codec much faster with **bit-exact output** (decoder: every decoded frame identical
to the stock decoder `C:\Code\ECM\bin\ref\DecoderApp.exe`; encoder: identical bitstream). Target 10x single-thread; the
wall-clock gains with threads are "the easy baseline". Decoder status: single-thread ~1.2-1.3x (details in DECODER.md), frame threads
(`--FrameThreads=N`, all-intra 720p ~3x), segment-parallel decoding (`--ParallelSegments=N`), GPU loop filters
(`--GpuLoopFilter=1`: ALF, SAO, BIF, CCSAO, LF-CCCM, offset refinement on Vulkan). The user then asked for **a full
GPU decoder in Vulkan, optimized until nothing is left** ("The GPU decoder needs to fully decode the frame on the
GPU"; "Your first Vulkan version is going to be slow. That's OK"). That work is branch `dec-gpurecon`: intra
reconstruction on the GPU (`--GpuRecon`), tool levels M1-M117 done and verified bit-exact (M19: DIMD, M20: OBIC, M21:
intra merge, M22: the transforms of DIMD CUs, M23: LFNST of MIP, M24: TIMD, M25: OBIC and intra merge with TIMD
neighbours, M26: TIMD-SAD, M27: OBIC and intra merge with TIMD-SAD neighbours, M28: TIMD merge, M29: TMRL of luma,
M30: chroma TMRL, M31: CCLM, M32: LFNST / NSPT of CCLM, M33: GLM's gradient patterns, M34: CCCM, M35: GL-CCCM, M36:
CCCM with several downsampling filters, M37: CCCM without subsampling, M38: CCCM with LFNST / NSPT, M39: GLM with
luma, M40: LM chroma fusion of CCCM, M41: CFLM, M42: GLM with luma with LFNST / NSPT, M43: the SPS switches of the
CCP tools, M44 / M45: the CCP models, M46: CCP merge, M47 / M48: CCP merge fusion with CCCM / DIMD chroma, M49: CCP
merge adjustment, M50 / M51: decoder-derived CCP fusion, its merge fusion, M52: CCP merge fusion with CCCM by a blend
model, M53 / M54: CCP merge and decoder-derived fusion with LFNST / NSPT, M55: PDP, M56: NN intra prediction, M57: the
MPM lists of MPM sorting, MDIP and PDP, M58: EIP, M59-M61: block vectors - TIMD, DIMD and OBIC with vectors, M62:
IntraTMP's search, without LIC / fusion / FLM / sub-sample vectors / OBMC, M63: EIP with a vector, M64: intra
merge with vectors, M65: BVG-DIMD, M66: SGPM without vectors or regression, M67: SGPM's integer vectors with
their LIC models, M68: IBC's interpolation, SGPM's fractional vectors, M69: IntraTMP's sub-sample vectors, M70:
IntraTMP's OBMC, M71-M73: IntraTMP's LIC, fusion and FLM, M74: intra merge's IntraTMP neighbours, M75: 12 bits, M76: SGPM's
regression, M77 / M78: single tree, M79 / M80: VVC's LFNST, LFNST of MIP); it is
**not yet faster than the CPU
decoder** (see section 8) and supports only a restricted tool set (in ON mode the real CTC streams still fall back to
the CPU, first reason: chroma reordering / DBV). Partial verification (`perf/gpu/partial.py`, section 7's "Partial
verification") runs the 19 regular CTC streams with the CUs of missing tools taken from the CPU: at M66 every intra
picture is identical but the known CCP merge blocks (fp720_ai's 3 pictures, ra720's first), and 17265 intra CUs
still come from the CPU (IntraTMP LIC 4159, intra merge with a neighbour of a missing kind 2586, IntraTMP
sub-sample vectors 1900, SGPM with vectors 3152 and regression 795, IntraTMP OBMC / fusion / FLM ~2200, ...; plus
32777 inter / IBC / palette CUs of inter pictures). M67-M70 (2026-10-16 / 21) took SGPM's vectors, IntraTMP's
sub-sample ones and its OBMC, M71-M74 (2026-10-22) the rest of IntraTMP and intra merge's IntraTMP neighbours, M75-M80 (the
agents') 12-bit, the rest of SGPM, single tree, VVC's LFNST and LFNST of MIP; then section 9's d and e.

## 1a. Working fast (development time)

The user found the first week far too slow (M1-M54 in ~a week: a milestone was ~30-60 min of coding and 4-5 h of
waiting for encodes, canaries, regressions and timing). After the changes below (2026-10-15) a milestone took ~70
min instead of ~2 h, each covering a whole tool family on real streams. Keep it that way:
- Test set = the 19 regular CTC streams in partial verification (`partial.py <build> -j 8`, ~3 min with warm
  pipelines). A step = a tool family; its gate = every stream identical (GPU vs CPU, block by block) plus ON mode
  where it applies. No dedicated encodes per tool unless no stream exercises the code; canaries only for code no
  stream reaches; the full regression (`verify_all.py <build> -j 4`, 444 streams, ~10 min warm) once a day and
  before every push; timing only when a phase ends.
- Measure before designing: a host-only build that splits a rejection reason finer (e.g. by what a CU's list
  holds) costs ~3 min (no shader change, so no pipeline compile) and shows where the CUs are (`cpu_cus.py <build>`).
  The SGPM split showed fractional vectors dominate, which set the order M67 (integer vectors) -> M68 (IBC
  interpolation).
- Shader changes cost pipeline compiles: an intra_rec pipeline takes 225-480 s cold. After a shader change run
  `precompile.py <build> -j 7 8:0 8:1 8:2 8:3 8:4 8:5 8:6` (~7 min: the MPM variant, which every regular stream's
  pictures take since EIP / IntraTMP / SGPM CUs force it), not all 63 (~26 min with 14 at a time; only needed before
  the 444-stream regression). Pipelines then load from runs/pipecache in 0.0 s. Check a shader edit with glslang
  first (`spirv_ids.py`: compile errors and each module's size in about a minute; since 2026-10-22 the build's
  spirv-opt step removed the SPIR-V ID budget that bound M45-M70). Host-only edits need no compile at all.
- Never wait idle: builds, precompiles and test runs go to the background while you read the CPU code for the next
  step or write the next part. Use at most one agent (the user's rule); parallelize with background processes.
- Debug with the verification's first differing field / block (the report names the CU, the field, CPU vs GPU
  values) and fix the cause; most bugs were a missed CPU branch (read the whole CPU function and its SIMD version).
- pj720_ai's pictures take ~1.8 s of GPU time in partial verification; under 8 parallel decoders some pass Windows'
  2 s GPU timeout (VK_ERROR_DEVICE_LOST); partial.py runs such a stream again alone.

## 2. The user's rules (follow them exactly)

Correctness / testing
- A speed-up must never change the output. Prove it on real files (identical decoded frames / md5, identical
  bitstreams). If output moves, report it and let the user decide; never accept a difference yourself.
- Find the real cause before fixing; no guessed fixes. A reproducible failure is a real bug: fix the code, not the test.
- Test on real, long, full-resolution content (1080p and 4K), not only small crops. (The GPU test streams have three
  short 1080p ones since 2026-10-11; long 1080p/4K runs are still to do, see section 9.)
- Tests must not depend on elapsed time or machine load. Canary builds when shrinking/reusing buffers. Fair CPU
  comparisons (same input/build type, paired runs, all processes counted). A QA round after every task (re-read the
  diff for bugs, run the real path, check performance).
- A canary must be able to change the output, and rebuild after removing it.

Commits / git
- One commit per optimization, each buildable and bit-exact; bug-check before committing. Fold fixes into the
  unpushed commit they belong to (`git commit --fixup <sha>` then
  `git -c sequence.editor=: rebase -i --autosquash <base>`, or set `GIT_SEQUENCE_EDITOR=true`); never a buggy commit
  followed by a fix commit. For already-pushed commits a separate fix commit is fine.
- Detailed commit messages: subject names the exact function / subsystem / metric; body = problem, cause, change,
  measurements (paired, bit-exact checks), hardware; end with `Co-Authored-By: Claude <noreply@anthropic.com>`.
  Commit with `git -c user.email=thebrianzheng@gmail.com commit ...`.
- Never use `git stash`. Never commit in the main folder (`C:\Code\ECM`); work in worktrees `C:\Code\ECM\wt-*`.
  Don't merge other sessions' branches (unless the user asks, as with the GitHub pull of 2026-10-09).
- Don't push / post / edit anything on any remote without the user's explicit permission, each time. Don't keep
  asking to push: raise it only for a release or when another PC (e.g. an AMD one) needs to test.

Process
- At most ONE agent at a time and no multi-agent workflows, even if "ultracode" appears.
- Plans: review and refine every plan up to 3 rounds before executing (order, conflicts, shared build trees, CPU
  load during benchmarks). Re-read every message to the user for accuracy and brevity (up to 3 rounds).
- Report every optimization as % speedup of the step and cumulative, with before/after numbers (at 1080p and 4K
  when possible), failures plainly, commit hashes, machine (CPU, OS, compiler, GPU + driver) and what to re-check
  elsewhere. A benchmark plan for *reported final/release numbers* needs the user's approval first.
- Keep working through usage-limit warnings; stop only when the user says so (then let benchmarks finish).

Machine
- Kill processes only by PID and only your own. Don't benchmark while another session benchmarks (check
  `Get-Process`). Never touch power or display settings (no powercfg / sleep / monitor-off). Ask before downloading
  anything (file name, source, size).

## 3. Repository, branches, what is where

- GitHub: https://github.com/4KVCD/ECM-speedup (public; remote `github`). Upstream: vcgit.hhi.fraunhofer.de ECM
  (remote `origin`, read-only). gh CLI logged in as 4KVCD on the old PC.
- `perf` (default branch on GitHub, the main line; there is no master / main): the encoder work, with `dec` and
  `dec-gpurecon` merged in (2026-10-14). 2026-10-15 it was fast-forwarded to `dec-gpurecon` at M66 (+ the tooling
  and this handoff).
- `dec-gpurecon-m67-wip` (pushed 2026-10-15): one WIP commit on top of `dec-gpurecon` with M67 half done; M67 was
  committed on `dec-gpurecon` from its diff on 2026-10-16, so the branch can go once that is pushed.
- `dec`: the decoder work (single-thread steps, frame threads, segment-parallel, GPU loop filters). Pushed
  2026-10-10 at 447ecde1 = 3e43989f (frame-thread drain fix: all-intra streams resend SPS/PPS each picture, which
  drained the threads; fp720_ai FT8 3.00 s -> 0.88 s) + 447ecde1 (merge of github/perf b8e5f2ba: 25 encoder
  speed-ups). QA of 447ecde1 passed: all 19 decoder streams in 12 decoder modes identical to stock, 779
  GPU-verified pictures with 0 differences, encoder suite bit-exact (1.47x vs ref).
- `dec-gpurecon` (pushed 2026-10-10, on top of `dec`): the GPU reconstruction. Commits, oldest first:
  9435a745 M1a, d7e8556b M2, 07d08f15 / 640419bb / 4fb7ccaa / ae34848e / 1001a254 / 73aca6b3 / 1b405737
  (kernel optimizations, 52x cumulative GPU time), 99987a7a ON mode (`--GpuRecon=1`; includes the SPS::m_timdMrg
  init fix), 4d691d97 host costs, 78c90fe0 frame-thread contexts, 58c2a529 DECODER.md section, 32695197 M3
  bilateral filter, ddf77bcb background creation + shared kernels + `--GpuRecon=3`, 180ccabc scheduling data on the
  GPU, then this handoff commit.
- Local-only branches on the old PC, NOT pushed (ask the user before pushing them):
  - `dec-gpu-dbf-wip` 62298b1b: GPU deblocking (G6) work in progress, parked; based on an old `dec`
    (19 commits not in `dec`, most of them older versions of GPU loop-filter commits that `dec` has in final form).
  - `dec-parsefirst` 134a0a45: experiments (env ECM_PARSE_FIRST, ECM_WPP_LAG); findings are in DECODER.md.
  - old encoder branches alf, byref, dq, exp, mc, meas-s12, tmp, merge (superseded by the other computer's `perf`).

## 4. Setting up the new computer

Directory layout the scripts assume (change the paths in `perf/gpu/*` if yours differ). Everything of the project
lives inside the main clone C:\Code\ECM (C:\Code holds one folder per project); the clone's .gitignore covers
build/ and bin/, and .git/info/exclude (local, not in git) the other folders. Never run `git clean -x` in the main
clone: it would delete the test material and the toolchain.
```
C:\Code\ECM                   main clone (never commit here)  C:\Code\ECM\wt-<name>      worktrees (git worktree add)
C:\Code\ECM\tools             portable toolchain (env.ps1 puts it on PATH; perf\build.ps1 dot-sources ..\..\tools\env.ps1)
C:\Code\ECM\build\<name>      build dirs                      C:\Code\ECM\bin\<name>     the exes of each build
C:\Code\ECM\bin\ref           the STOCK ECM-20.0 decoder/encoder (reference for every comparison)
C:\Code\ECM\seq               raw YUV test sources            C:\Code\ECM\streams        the 19 decoder test streams
C:\Code\ECM\streams_gpu       the GPU test streams            C:\Code\ECM\runs           outputs/logs
```
Don't reuse a build dir configured before 2026-10-10 without deleting it first: its CMakeCache.txt records the
old C:\Code\build and C:\Code\wt-* paths, so CMake refuses to configure it. The exes in bin\ are not affected.
Toolchain used (Windows 11): WinLibs GCC 16.2 (mingw64), CMake 4.4.4, Ninja, MinGit, gh. Vulkan: no SDK needed;
`vulkan-1.dll` is loaded at run time. Build inputs (download only with the user's OK): Vulkan-Headers tag
`vulkan-sdk-1.4.363.0` (C:\Code\ECM\tools\vulkan-headers\Vulkan-Headers-vulkan-sdk-1.4.363.0) and glslang 16.6.0
(C:\Code\ECM\tools\glslang\bin\glslang.exe); since 2026-10-22 also spirv-opt from the Vulkan SDK (1.4.357, installed
copy-only with the user's OK to C:\Code\tools\vksdk: `vulkansdk-windows-X64-1.4.357.0.exe --root C:\Code\tools\vksdk
--accept-licenses --default-answer --confirm-command install copy_only=1`; only Bin\spirv-opt.exe and its DLL are used),
passed as `-DECM_SPIRV_OPT=C:/Code/tools/vksdk/Bin/spirv-opt.exe`. The GPU must support Vulkan 1.3 compute with buffer device address and
the **vulkanMemoryModel** feature (GpuIntraRec::create returns null otherwise); tested only on an RTX 3070 (driver
616.92).
Without the Vulkan SDK (the RTX 3070 PC): glslang -O cannot build intra_rec any more ("ID overflow", no module), and
glslang without -O gives SPIR-V the driver compiles far slower. `-DECM_SPIRV_OPT=C:/Code/ECM/wt-gr/perf/gpu/slangopt.cmd
-DECM_SPIRV_INLINE=OFF` stands in for spirv-opt with the SPIRV-Tools inside NVIDIA Nsight Graphics 2026.3.1
(slang-glslang.dll, Slang's pass list with a raised ID limit, its inlining pass left out; slangopt.py says how). On
this PC a cold intra_rec pipeline takes 1-10 minutes and 7-20+ GB, and a decoding process keeps 3-5 GB per pipeline
it compiled: run `precompile.py` with `ECM_GPU_SLOTS=1` (two at a time pass the 60 GB budget) before partial
verification, and partial.py with -j 1 or 2 once the pipelines are in runs/pipecache.

Builds (from a worktree, e.g. `C:\Code\ECM\wt-gr` on branch dec-gpurecon):
```
perf\build.ps1 gr -NoLto '-DECM_VULKAN=ON' '-DECM_VULKAN_HEADERS=C:/Code/ECM/tools/vulkan-headers/Vulkan-Headers-vulkan-sdk-1.4.363.0/include' '-DECM_GLSLANG=C:/Code/ECM/tools/glslang/bin/glslang.exe' '-DECM_SPIRV_OPT=C:/Code/tools/vksdk/Bin/spirv-opt.exe'
perf\build.ps1 grL '-DECM_VULKAN=ON' <same three paths>      # LTO, for timing
```
Quote the -D arguments in PowerShell: unquoted, `-DNAME=C:/...` is split at the colon and the configure fails.
Likewise PowerShell 5.1 drops an empty argument, so `-o ""` makes the next option the output file name.
Pitfalls: all build dirs of one worktree share its in-source `lib/` and `bin/`: before switching build dir run
`perf\gpu\cleanlibs_gr.ps1` (edit its path to your worktree) or the next build may link stale libraries. LTO needs
`-DCMAKE_POLICY_DEFAULT_CMP0069=NEW` (build.ps1 passes it). Shaders (`source/Lib/GpuLib/shaders/*.comp`) are
compiled by CMake into `spv/<name>.spv.h`; new .comp files are picked up automatically. A source restored with
Copy-Item keeps its old mtime and is NOT rebuilt (touch it). PowerShell 5.1 `Set-Content -Encoding utf8` writes a
BOM; write commit messages to a file without BOM and use `git commit -F file`. The tool environment blocks
commands that combine `Remove-Item` with `//` text. `perf\gpu\crlf.ps1 <worktree>` converts changed files to CRLF
(the repo uses CRLF).

## 5. Test material

- Sources in C:\Code\ECM\seq (not in git): foreman_352x288.yuv, ParkJoy_1280x720.yuv (62 frames),
  FourPeople_1280x720.yuv, CrowdRun / Ducks / ParkJoy_1920x1080.yuv (60 frames each, from crowd_run / ducks /
  park_joy_1080p.y4m); y4m originals in seq\y4m. 4K: none yet. The user wants 1080p/4K testing.
- GPU test streams (C:\Code\ECM\streams_gpu, not in git, regenerate with the scripts; all-intra, ECM tools off except the
  level's tools, `perf\gpu\enc_m1.ps1` lists the disabled tools; it uses the bit-exact optimized encoder build qaCpu, which writes the same bitstream as stock):
  - `enc_set.ps1 m1`                                         M1: planar/DC/angular, DCT-II only
  - `enc_set.ps1 m2 --IntraToolControlMode=1 --IntraPredBf=0` M2: + reference filter, PDPC, intra fusion
  - `enc_set.ps1 m3 --IntraToolControlMode=1 --IntraPredBf=1` M3: + intra-prediction bilateral filter
  - `enc_set.ps1 m4 --IntraToolControlMode=1 --IntraPredBf=1 --DualITree=1` M4: + dual tree
  - `enc_set.ps1 m5 --IntraToolControlMode=1 --IntraPredBf=1 --DualITree=1 --LMCSEnable=1` M5: + LMCS; `m5s`: the
    same with `--DualITree=0`. The encoder turns LMCS off in the picture headers of the CIF foreman pictures, so
    only the 720p streams exercise the LMCS paths.
  - M6 (M5's arguments plus): `m6` `--DepQuant=2` (dependent quantization, 8 states, as in CTC), `m6q`
    `--DepQuant=2 --QCS=1` (as in CTC), `m6s` `--QCS=1` (scalar quantization with QCS), `m6d1` `--DepQuant=1`
    (4 states).
  - M7 (m6q's arguments plus `--JointCbCr=1`): `m7` dual tree, `m7s` `--DualITree=0`, `m7q` `--DualITree=0
    --CbQpOffset=3 --CrQpOffset=-3 --CbCrQpOffset=-2` (unequal chroma QPs). In m7 the foreman streams use the
    negative ICT modes, park720 the positive ones (counted with a temporary print).
  - M8 (m7's arguments plus `--MTS=1 --MTSExt=1`): `m8`, explicit intra MTS; its foreman streams use all seven intra
    transform types in 31 horizontal/vertical pairs. Implicit MTS (`--MTSImplicit=1`) cannot be encoded without DIMD
    (CHECK "Wrong candModeListForTransform size").
  - M9 (m8's arguments plus `--TransformSkip=1 --ChromaTS=1 --BDPCM=1`): `m9` (camera content: transform skip in
    ~15 % of the coded blocks of m9_fore32, a few BDPCM blocks) and `m9_screen22/32/37` (`enc_m1.ps1 m9_screen<q>
    C:\Code\ECM\seq\screen_352x288.yuv 352 288 4 --TemporalSubsampleRatio=1 -q <q> <args>`; mostly transform skip,
    hundreds of BDPCM blocks incl. chroma, transform skip with joint CbCr). screen_352x288.yuv is synthetic
    screen-like content from `perf\gpu\gen_screen.py` (deterministic).
  - M10 (m9's arguments plus `--MIP=1`): `m10`; its foreman streams have 1235 MIP blocks of all three size classes,
    transposed and not, 4x4 to 32x32.
  - M11 (m9's arguments plus `--IntraLFNSTISlice=1 --LFNSTExt=1 --NSPT=1`, without MIP, as LFNST of MIP blocks is not
    supported yet): `m11`; its foreman streams: lfnstIdx 1-3 on all components, 382 blocks with separable LFNST and
    3348 NSPT blocks (4x4 to 32x8) among 7896 coded blocks. `m11m_fore32` (`enc_m1.ps1`, also `--MIP=1`): has MIP
    blocks with LFNST, so its pictures fall back to the CPU ("LFNST of MIP").
  - 1080p (m11's arguments, 2 pictures, QP 32): `h11_crowd`, `h11_ducks`, `h11_park` (`enc_m1.ps1 h11_crowd
    C:\Code\ECM\seq\CrowdRun_1920x1080.yuv 1920 1080 2 --TemporalSubsampleRatio=1 -q 32 <m11 args>`; ~20 min each).
  - M12 (m11's arguments plus `--MRL=1`; TMRL and MIP stay off): `m12`; its fore32 and fore22 streams have 1292 MRL
    blocks on all lines (1: 690, 3: 320, 5: 128, 7: 92, 12: 62), 817 with the fusion line, 45 DC (40 with the
    bilateral filter), all angle signs, and MRL unit groups unavailable at the picture's edges. `enc_m12_edges.ps1`:
    `m12e_*` (64x64 / 128x64 crops via `crop.py`: many blocks at the edges) and `m12s_*` (synthetic content via
    `gen_mrl_edge.py` on which the encoder picks lines 5-12 near the top).
  - M13 (m12's arguments plus `--ISP=1`): `m13`; its foreman streams have 1312 ISP sub-partitions (both splits,
    1x16 to 32x4, 114 in shared 4-wide regions, 93 with a coded 1-D transform, 39 with the side along the split
    unavailable, planar / DC / all angle classes, PDPC). `m13c_0..47` (`enc_crops.py m13c 48 13 <m13 args>`, 64x64
    crops): 122 bilateral filters of ISP sub-partitions with partial padding, 32 of them with the CPU's writes past the
    row (counted with a temporary print).
  - M14 (sign prediction as in CTC; LFNST / NSPT off, MIP on): `m14` with `--IntraToolControlMode=1 --IntraPredBf=1
    --DualITree=1 --LMCSEnable=1 --DepQuant=2 --QCS=1 --JointCbCr=1 --MTS=1 --MTSExt=1 --TransformSkip=1 --ChromaTS=1
    --BDPCM=1 --MIP=1 --MRL=1 --ISP=1 --NumSignPred=8 "--NumSignPredOverrideByQP=22 4"`; `m14s` the same with
    `--DualITree=0`; `m14a` with `"--NumSignPredOverrideByQP=0 8" --Log2SignPredArea=5` (the 32x32 sign area at QP 22
    and 37, where the AI configuration does not force 8x8); `m14l_fore32` (`enc_m1.ps1`, 2 pictures, m13's arguments
    plus sign prediction): LFNST / NSPT blocks with predicted signs (fell back until M16). Counted with a temporary
    print: m14_fore32 has 2258 blocks with predicted signs (luma and chroma, joint
    CbCr with Cb and with Cr coded, ISP, MIP, MRL, the bilateral filter, at the picture's edges, 1 to 8 signs), 1822
    with a negative decided sign (the residual computed again), 1324 with a wrong prediction; m14a_fore22 predicts
    coefficients up to (19, 19); m14_park720 / m14s_park720 have 613 / 1215 chroma blocks with LMCS chroma scaling.
    1080p: `h14_crowd`, `h14_park` (`enc_m1.ps1`, 2 pictures, QP 32, m14's arguments).
  - M15 (LFNST and NSPT on the GPU; the m11-m13 sets have both): `m15n` (m13's arguments with `--NSPT=0`: LFNST also
    on the small blocks NSPT otherwise takes, i.e. its 4x4 and 8x8 regions; m15n_fore32 has 362 / 125 LFNST blocks
    with a 4x4 region (plain / transposed), 485 / 157 with 8x8, 162 / 55 with 16x16), `m15s` (m13's arguments with
    `--DualITree=0`: in a single tree only luma takes LFNST / NSPT). NSPT in m11_fore32: all 12 block shapes, 870
    plain / 352 transposed; chroma and joint CbCr blocks with LFNST / NSPT in the dual-tree sets.
  - M16 (sign prediction of LFNST / NSPT blocks): `m16` (m13's arguments plus `--NumSignPred=8
    "--NumSignPredOverrideByQP=22 4"`), `m16n` (the same with `--NSPT=0`). Counted with a temporary print:
    m16_fore32 has 191 LFNST blocks (16x16 region) and 1052 NSPT blocks (345 transposed) with predicted signs, 176 /
    919 of them with a negative decided sign; m16n_fore32 456 / 604 / 203 LFNST blocks with 4x4 / 8x8 / 16x16 regions
    (373 / 520 / 179 with a negative sign).
  - `fx_wpp_park720`, `fx_tiles_park720` (m11's arguments plus `--WaveFrontSynchro=1`, resp. `--EnablePicPartitioning=1
    --TileColumnWidthArray=3 --TileRowHeightArray=2 --RasterScanSlices=1 --RasterSliceSizes=100`: one slice of 4x3
    tiles): must fall back to the CPU ("wavefronts", "tiles").
  - each set: fore22/32/37 (CIF), park720, four720 (2-4 pictures). Longer ones: m2_fore33, m2_park16, m3_park16
    (`enc_m1.ps1 <name> <yuv> 1280 720 16 --TemporalSubsampleRatio=1 -q 32 <level args>`), m3_park60 (60 frames,
    `enc_long.ps1 m3_park60 ...\ParkJoy_1280x720.yuv 1280 720 4 15 -q 32 --IntraToolControlMode=1 --IntraPredBf=1`:
    4 parallel segments concatenated; each starts with an IDR). Move the `*_seg*` files out of streams_gpu before
    running the checkers (they pick up every .bin).
  - The AI cfg needs `--TemporalSubsampleRatio=1`; the 350x286 encode crashes the stock encoder.
  - Note: EnableTMnoninterTools=0 in these streams leaves stock `SPS::m_timdMrg` unparsed (fixed by a ctor init).
- The 19 regular decoder streams (C:\Code\ECM\streams, CTC-like AI/RA/LD incl. *_md5) with `perf\decbench.py`; stock
  references in C:\Code\ECM\runs\ref_big etc.

## 6. Scripts (perf/gpu, copied from the old session's scratchpad)

| script | use |
|---|---|
| `partial.py <build> [-j N] [streams]` | partial verification of the 19 regular streams (C:\Code\ECM\streams): per stream the pictures verified / differing and the first difference, then the CUs taken from the CPU by reason; logs in runs/partial. The test gate of every step since M57. A stream given as a path (`C:/Code/ECM/streams_ld/ld_foreman_q32.bin`) runs from there: the RTX 3070 PC's inter streams (LD-B, LD-P, RA of foreman / ParkJoy / FourPeople, made with grn's EncoderApp and the ECM cfgs) are in C:\Code\ECM\streams_ld |
| `private_globals.py <module.spv>...` | the per-invocation globals one function other than the entry point uses alone, which the optimizer's private-to-local pass breaks without inlining (run it on the build's `spv/*.raw.spv` after adding such a global; each must be used by a second function) |
| `inline_size.py <module.spv> [N]` | a shader's size with every call inlined (what the drivers compile) and the functions that make it; check inter_mc / intra_rec after adding a call to a large function |
| `slangopt.cmd` / `slangopt.py` | spirv-opt's stand-in from Nsight Graphics' SPIRV-Tools (section 4's build without the SDK); `ECM_GPU_INTRA_REC_SPV=<file.spv>` / `ECM_GPU_INTER_MC_SPV=<file.spv>` make a decoder take that module for intra_rec's first module / inter_mc (an experiment on a changed shader, e.g. the driver's compile, without a rebuild) |
| development switches | `ECM_GPU_NO_CCP_LFNST=1`: partial verification leaves the pictures that need LEVEL_CCP_LFNST's pipeline (intra_rec's second module) unverified - on the 3070 PC its cold compile needs over 35 GB, past the budget, and most intra pictures need it. `ECM_GPU_CPU_KINDS=GPM,affine,...`: those inter CU kinds (interKind's names) keep the CPU's prediction (which kind a difference is in). `ECM_GPU_PIPELINE_TIMES=1` also names a pipeline before it compiles. `ECM_GPU_FAULT=1` enables VK_EXT_device_fault: a lost device prints the faulting addresses, and every buffer prints its address range at its creation. `ECM_GPU_NO_DERIVED_CCP=1`: as ECM_GPU_NO_CCP_LFNST for LEVEL_CCP_DERIVED's pipelines (the MPM one did not compile within the budget on 2026-10-25). `ECM_GPU_DIFFS=1`: a picture's first differing block also lists its samples, CPU / GPU, prediction and residual. `ECM_GPU_MIN_LEVEL=<n>` runs every intra picture in a pipeline of at least that level (a speed comparison of the pipelines on the same pictures; the output is the same). Every pipeline the driver compiles prints its statistics on stdout (VK_KHR_pipeline_executable_properties: `[gpu] pipeline <name> CS: Register Count ...; Local Memory Size ...` - NVIDIA's local memory reads 2^36 + the bytes per invocation -, Shared Memory Size, Binary Size), so a new build's partial.py logs have intra_rec's; one from a cache prints nothing, and `ECM_GPU_PIPELINE_STATS=1` (with `__GL_SHADER_DISK_CACHE=0`) compiles them all anew for it. NVIDIA's driver gives intra_rec either 255 registers (one workgroup per SM) or 128 (two: 2 x 256 x 128 fills the register file), with ~11 KB of local memory per invocation either way (mostly local arrays): M171's shaders 255 / 10,672 bytes, the same with the sub-sample search split (grn215) 128 / 11,408 bytes and 4 % slower. Which one a build gets moves its whole time by several %, so compare builds of the same register count, or a change's phase on the block clock. |
| `spirv_noinline.py <in.spv> <out.spv> <min> [names]` | marks functions DontInline (the named ones, or those of at least `min` inlined SPIR-V instructions called more than once): an experiment on the driver's compile memory via ECM_GPU_INTRA_REC_SPV. On the 3070 (2026-10-25) it halved intra_rec's but the output was wrong and 20 % slower: not for NVIDIA |
| `precompile.py <build> [-j N] [variant:level ...]` | compiles intra_rec's pipelines into runs/nvcache and runs/pipecache (see 1a: after a shader change `8:0` .. `8:6`) |
| `verify_all.py [-j N] <build> [streams]` | the full regression: `--GpuRecon=2` over the 444 GPU test streams (C:\Code\ECM\streams_gpu); M66 baseline 1565 verified, 8 skipped, 0 differences |
| `cpu_cus.py <build> ...` | sums partial.py's "CUs taken from the CPU" per reason over the logs: what to port next |
| `long_lines.py <worktree> <files>` | added lines over 120 columns (the style rule) |
| `shared_size.py <build> [limit]` | each SPIR-V module's shared memory against the limit (49152 bytes; exit 1 above it, with the largest variables). Run it after a shader change that adds shared memory: NVIDIA's driver creates a pipeline above the limit and the device is lost when it runs (intra_rec is within ~200 bytes of it) |
| `spirv_ids.py [worktree]` | compiles intra_rec's two modules as the build does (spv_build.cmake), prints their ID bounds and times |
| `gpucheck.py [-j N] <build> [streams] -- <decoder args>` | decodes every GPU stream with the build and with the stock decoder (cached), compares all frames, prints the `[gpu-intra]` lines. Use `--GpuRecon=2` (verify) and `--GpuRecon=3`, also with `--FrameThreads=8`. `-j N` decodes N streams at a time (6 for verification, 3 with 8 frame threads; 1, the default, for timings): a full regression then takes minutes instead of most of an hour. |
| `stress.py <build> <reps> [args]` | repeats verify decodes, saves the output of any run that is not "all verified, 0 differences, 0 skipped" (caught the m_timdMrg bug, 1 in ~650 runs) |
| `gpuload.py <build> <seconds>` | keeps the GPU busy (decodes with `--GpuRecon=3 --FrameThreads=8`); run it beside stress.py, GPU synchronization bugs needed the load to show (intra_resi scratch, 2026-10-11) |
| `gpubench.py <build> [reps]` | GPU kernel time per stream (verify mode, best of reps) |
| `e2e.py <build> <streams>` | wall time CPU / CPU FT8 / GPU on / GPU on FT8, output checked |
| `walls.py <build> <stream> <reps> -- args \| args ...` | wall time per configuration (no output check) |
| `enc_m1.ps1`, `enc_set.ps1`, `enc_long.ps1` | test stream encoding (section 5) |
| `gen_screen.py <out.yuv> [w h frames]` | synthetic screen-like 4:2:0 content (transform skip / BDPCM test streams) |
| `crop.py`, `gen_mrl_edge.py`, `enc_m12_edges.ps1` | small crops of the test sequences and synthetic MRL content; the M12 edge-case streams |
| `enc_crops.py <prefix> <count> <seed> [args]` | test streams of random 64x64 crops (many blocks at the picture's edges), e.g. the M13 crops |
| `qa.ps1` | the full QA before a push of `dec` (3 builds, 12 decoder modes on the 19 streams, encoder suite); paths point at wt-dec |
| `buildgpu.ps1`, `cleanlibs.ps1`, `cleanlibs_gr.ps1`, `crlf.ps1` | build helpers (paths: wt-dec / wt-gr) |

Profiling: `perf\sprof\sprof.exe out.txt <command>` (sampling profiler, ~1 ms) and
`python perf\sprof\sprof_report.py out.txt <exe> [--top N] [--callers F] [--callees F]` (needs mingw64\bin on PATH
for nm/objdump; profile a non-LTO build).

## 7. The GPU reconstruction: design

Modes (`--GpuRecon`): 1 on (the CPU reconstructs until the background-created device is ready, then the GPU),
2 verify (the CPU reconstructs, the GPU result is compared per block and per sample, plus the GPU's scheduling data
against the host's), 3 on and waiting for the device at the first picture (use for tests/benchmarks on short
streams). Works in every frame-thread context.

Flow per supported intra picture (ON mode): DecSlice parses the whole slice first (parsing never needs the
reconstruction - proven on all 19 streams), then `decompressCtu` runs derive-only (`DecCu::setGpuDeriveOnly`,
`IntraPrediction::setParamsOnly`): per TU `GpuIntraRec::record` stores geometry, final mode, IntraPrediction
parameters (`IpaSnapshot`), dequant/transform params and the nonzero coefficient region; then
`GpuIntraRec::endPicture` uploads and runs on one Vulkan stream: `intra_resi` (all residuals; dequant folded into
the first DCT-II stage), `intra_deps` passes 0-3 + `scan` (owner maps, availability, read ranges, dependency graph,
ready queue), `intra_rec` (persistent workgroups, default 32, env ECM_GPU_GROUPS; a block is ready when the blocks
owning the samples it reads are done; the last finishing dependency hands its dependent to its own workgroup;
Vulkan memory model with coherent buffers, release/acquire atomics), then downloads the planes into the
reconstruction buffer. If a picture or block is unsupported (`GpuIntraRec::supported()` SPS gate, `record()` per
CU), the CPU reconstructs it.

Rule that keeps it bit-exact: the host may compute anything that depends only on syntax/geometry/decoding order;
everything that depends on samples runs on the GPU, implemented line by line after the C++ (IntraPrediction.cpp,
TrQuant.cpp, BilateralFilter.cpp). `supported()` must reject every tool whose mode derivation or prediction reads
reconstructed samples or that the kernels lack.

Files: `source/Lib/GpuLib/GpuIntraRec.{h,cpp}`, `shaders/intra_common.glsl` (Tu = 84 ints and Params layouts,
static_asserts on the host), `intra_resi.comp`, `intra_rec.comp`, `intra_deps.comp`, `scan.comp`, `VkCompute.*`
(device with vulkanMemoryModel + optional shader clock; up to 8 queues round-robin); `DecoderLib/DecCu.cpp`
(record calls), `DecSlice.cpp` (parse-first, derive pass, CPU fallback), `DecLib.cpp` (setGpuRecon,
xPollGpuRecon), `DecFrameThreads.cpp`; option in `DecApp/DecAppCfg.cpp`.

Tools implemented (M1-M3): planar (incl. directional planar), DC, angular with 16/10-tap cubic and 6-tap Gaussian
luma / 2-tap chroma interpolation, 8x reference lengths, reference filter, PDPC (planar/DC, pure H/V, gradient,
inverse angle), intra fusion, DisableRefFilter, intra-prediction bilateral filter (JVET-AK0118: angular reference
texture gate over the full 8W/8H arrays, prediction MAD gate, mirror + reconstruction padding with the CPU's
skip-the-last-unit quirk, 5x5 diamond), DCT-II 2..128, scalar dequant, adaptive luma clipping, 4:2:0/4:0:0, one
slice per picture. M4: the dual tree of intra slices (no kernel change). M5: LMCS (mapped luma clipping range, the
bilateral filter on inverse-mapped samples, chroma residual scaling from the luma next to the CU; the host uploads
the LUTs and the chroma scale per luma value, `Reshape::getChromaAdjNeighbours` finds the neighbouring luma CUs).
M6: dependent quantization (the host records quantization indices) and QCS (`m_coeffShift64` from Quant.h).
M7: joint CbCr (the Cb block computes both residuals in intra_resi; Tu::jccr, jccrOffset).
M8: explicit intra MTS (TrQuant::getInvTransformTypes per block; every transform matrix read back by impulses with
TrQuant::getInvTransformMatrix; Tu::mtxV, mtxH). M9: transform skip and BDPCM (Tu::transformSkip; invResDPCM on
the host; BDPCM blocks predict as mode 18 / 50 with the filters off). M10: MIP (Tu::mipMatrix etc.; recorded with the
DC mode for the read range and the bilateral filter). M11: LFNST / NSPT on the host side (replaced by M15). M12: MRL
(Tu::mrl; the MRL unit groups left of and above the corner in avail[line][5..6]; Tu::bfAvail, line 0's availability
for the bilateral filter of MRL DC blocks). M13: ISP (Tu::isp, cuX..cuH, regX / regW the prediction region, ispSide, ispPrev the
scheduling chain; refSource in GpuIntraRec.cpp and intra_common.glsl gives every reference entry's source sample for
the fill and the dependencies; 1-D transforms in intra_resi). M14: sign prediction (GpuIntraRec::prepareSignPred takes
a block's predicted levels and coded signs with DepQuant::getPredictedSigns and gives the TU its levels back;
intra_resi computes the templates into the signPred buffer and the residual with the predicted signs positive;
intra_rec's signPrediction decides the signs after the prediction and computes the residual again where one is
negative; Tu::spCoded, spOffset, resiFrom: with joint CbCr the coded component's block decides and the other block
waits for it). M15: LFNST and NSPT on the GPU (GpuIntraRec's invSecondary gives a block's matrix, scan, sizes and
transposition with the CPU's choices; the CPU's int8 kernel tables, the scans' positions and the LFNST output
arrangement are uploaded once (Shared::secKernels, matrices); intra_resi runs them; Tu::secKind etc.). M16: sign
prediction of LFNST / NSPT blocks (intra_resi's templates through NSPT / LFNST, intra_rec's recomputed residual too).
M17: intra modes derived on the GPU, the base for DIMD / TIMD / MPM sorting, whose modes depend on reconstructed
samples. Per CU and channel type a record (`Cu`: the parsed MPM syntax, the CUs at getSpatialIpm's 23 positions as
getPURestricted finds them, a chroma CU's co-located luma CU, the modes the syntax leaves) and, for a mode the GPU
derives, a node record before the CU's first block (`Tu::derive & DERIVE_NODE`, no samples): intra_rec runs the MPM
lists there (`intra_modes.glsl`: PU::getIntraMPMs without MPM sorting, DIMD, MDIP, PDP, NN) or chroma DM / the
candidates' substitution and publishes the mode (`Params::cuMode`); the CU's blocks depend on the node and set their
parameters from the mode (`applyModeParams`: initPredIntraParams, fusion, directional planar, the bilateral filter's
mode rule; GpuIntraRec.cpp has a copy for the host's read ranges); intra_deps reads the union of the reference ranges
of the modes the syntax leaves (`Cu::modes`: planar for mpm_idx 0, a chroma candidate or VDIA, else all 67). The host
still derives every mode itself (no supported tool needs the GPU's yet) and makes no CU records then (they cost 4 % of
the wall time); `ECM_GPU_MODES=1` (`gpuDerivesModes`, `Impl::deriveModes`) makes the GPU derive all those the MPM
lists give, on every stream, and verification compares each derived parameter with the CPU's.
M18: the transforms of a derived mode (`Tu::derive & DERIVE_TRANSFORM`: ECM's explicit MTS with MTSExt, LFNST, NSPT)
on the GPU: intra_rec's `selectTransform` picks the kernel pair (getTrTypes' set of the size class and the clamped wide
angle, IDTR near horizontal / vertical) and the LFNST / NSPT set, transposition and matrix
(getFinalIntraModeForTransform's mode: the wide angle of the TU, of the CU with ISP, in NSPT's numbering) from tables
the host builds once with the CPU's own lookups (the TR_TAB_* directory in matrices: matrix offsets per type and size,
g_aucIpmToTrSet / g_aucTrSet / g_aucTrIdxToTr / g_aiIdLut, g_lfnstLut / g_nsptLut, the LFNST / NSPT matrices' offsets per
size class, set, index and bucket from TrQuant::getNsptMatrix); the host gives what does not depend on the mode
(Tu::mtsSet, secIdx, secBucket, the kind, sizes and scan). intra_resi skips such blocks: intra_rec's inChainResidual
computes the residual before the prediction in chunks of rows whose first stage fits shared memory (without LFNST /
NSPT it dequantizes from the levels buffer: with TU_256 a 128 x 128 MTS block keeps all its coefficients) and stores it
device-coherent where intra_resi would, followed by a workgroup barrier with buffer semantics (with joint CbCr the Cr
block reads its residual like resiFrom); signPrediction computes the base border and templates itself (signPredData)
and always recomputes. Without ECM_GPU_MODES and DIMD no block has DERIVE_TRANSFORM.
M19: DIMD. With DIMD every MPM list starts with DIMD's modes, so the GPU derives all modes the syntax does not give
(`Impl::deriveModes` whenever the SPS has DIMD). A CU's node computes its DIMD first (`intra_dimd.glsl`): the
gradient histograms around the CU with the arguments of `IntraPrediction::dimdHistogramCalls` (the CPU's own
availability checks; `Cu::dimd`, `dimdNum` index the `DimdCall`s) - buildHistogram in parallel, buildHistogramAdaptive
in the CPU's sample order with a per-batch subgroup prefix sum that stops at the sample reaching maxAmp, and for CUs
of 64 or more its first template line by one invocation (a region state carried from sample to sample) over
gradients computed in parallel - then the modes, blending, weights and location dependence (`dimdSelect`, into
`Params::dimdOut`), which the MPM list (JVET-AL0125: DIMD's modes after planar) or a DIMD CU (`CU_DIMD`) takes. A DIMD
CU's block (`DERIVE_DIMD`) with blending predicts the main mode, planar and the blend modes one after the other, each
with its own parameters, filtered or unfiltered references (both kept) and JVET-AN0158 filter length, sums them per
location-dependence class into a global per-workgroup buffer (`BLEND_ACC`; another invocation may handle a sample in
the next prediction: coherent, with buffer barriers), then divides and blends location-dependently
(xLocationdepBlending; its double-precision clip ranges are reproduced exactly in integers, `dimdScaledWeight`) and
applies the bilateral filter to the blend. DIMD chroma (`CU_CHROMA_DIMD`) is a node too: one histogram of the luma
inside the co-located area and the Cb / Cr neighbours (`dimdChromaHistogramCalls`), then the co-located luma mode's
substitution. Nodes depend on the owners of their histograms' samples. Verification compares every node's DIMD with
the CPU's. A picture with DIMD runs the full intra_rec pipeline, one without DIMD a pipeline without its code (and
without M18's derived transforms where it has none; `IntraRecVariant`, section 10).
M20: OBIC (JVET-AH0076 with JVET-AK0056's weights). An OBIC CU (`CU_OBIC`, a DIMD CU with obicFlag) is a node that
first computes its own DIMD (whose main mode and first blend mode OBIC counts half), then the histogram of occurrence
of its neighbours' modes (`intra_obic.glsl`): the host picks the neighbours with deriveObicMode's steps 1-3
(`IntraPrediction::obicNeighbours`: adjacent and non-adjacent positions, intra, distinct, the nearest 20) and records
each one's CU and sample count (`ObicNbr`, `Cu::obic`; MIP CUs count nothing in intra slices and are left out); the
node reads a DIMD neighbour's DimdOut (planar and blend modes by their weights relative to the main mode's), an OBIC
neighbour's ObicOut, any other CU's mode, from those CUs' nodes (dependencies), then picks the six strongest modes
but planar with their dominant class, weights them (DIMD's division by table, 64-bit products, the excess / shortfall
loops) and adds planar (`Params::obicOut`). Its histogram reuses `s_dimdHist` (shared memory is full). The blocks
(`DERIVE_OBIC`) blend through DIMD's path, generalised to a list of (mode, weight, class) per block (`s_blendMode`,
`s_blendW`, `s_blendClass`; DIMD's filter length by weight only for DIMD); an OBIC CU always blends (no fusion, the
bilateral filter on the blend). With the reconstruction on, the CPU's OBIC can lack a mode (its neighbours' DIMD
comes from samples it has not reconstructed) and its later MPM lists would index with -1: DecCu then keeps a
placeholder (ECM_VULKAN, ON mode only).
M21: intra merge (JVET-AM0074). An intra merge CU (`CU_MERGE`, a DIMD CU with intraMergeMode; no DIMD of its own) takes
the parameters of a neighbour: its node (`intra_merge.glsl`) walks deriveIntraMergeCand's neighbours in the CPU's order
(the host lists them, `IntraPrediction::intraMergeNeighbours`, with how each may be a candidate: an intra merge or OBIC
CU always, a DIMD or MPM CU where its DIMD blends - which the GPU derives, so the list is whole and the GPU stops at
five) and keeps a valid entry at a new position with new parameters (an OBIC neighbour's converted as
setIntraMergeCandParam does). Two or more are ranked as reorderIntraMergeCand does, by template costs
(`intra_timd.glsl`, the base for TIMD): the template of 2 or 4 lines left of and above the CU, its reference line (8 x
the template's block, filled from the host's unit availability, `IntraPrediction::timdReferenceUnits`, padded in the
CPU's order), every mode of the candidates predicted on the template at 1/64 precision (initPredTimdIntraParams'
wide angles and tables, the cubic filter g_aiExtIntraCubicFilter, the PDPC kinds of predTimdIntraAng), the SATD of the
reconstruction against it with xGetHADs_SIMD's kernels (2x2, 4x4, 8x4, 4x8; the double `sad / sqrt(32) * 2` of 8x4 /
4x8 is exactly isqrt(sad^2 / 8), checked for every sad below 2^24), then each candidate's weighted cost (unsigned 64-bit
products, planar blend modes free, the first of equal costs); all modes at once where `s_tmp` holds their templates.
The choice goes into the CU's DimdOut slot in DIMD's layout, so its blocks (`DERIVE_MERGE`) blend like DIMD's (always,
regular filter lengths, the bilateral filter on the blend). The CPU derives DIMD for every MPM CU, also where the list
starts with planar (only intra merge reads it): the host keeps those CUs' DIMD histograms aside (`pendingDimd`) and
puts them to use when an intra merge CU lists the CU. OBIC counts an intra merge neighbour's modes without a class
(`s_obicPlain`); a mode only those gave has obicLocDep -1, whose weight the CPU adds to the first class's total but to
no class's blend (and its class-blend shortcuts let the last prediction take it): `blendTotals`. An OBIC CU counting
such a neighbour indexes `histoLocDep[mode][-1]`, which in the CPU's row-major array is `[mode - 1][2]`, and the GPU
counts it there (`obicCount`; no test stream has it: checked by a temporary test that forced class -1 on CPU and GPU
alike, 216 such counts identical, and failed with the GPU counting elsewhere). Verification also compares the template
costs (`IntraPrediction::m_mergeTmCost`, a copy reorderIntraMergeCand keeps). A picture with intra merge CUs runs a
fourth intra_rec pipeline (`WITH_MERGE`); in the one with DIMD, intra merge's code made pictures without it ~11 %
slower. Rejected: intra merge above 10 bits (the CPU's SATD adds in 16-bit lanes), with a block vector, CUs above 128
(the node's shared arrays).
M22: the transforms of DIMD, OBIC and intra merge CUs (explicit MTS, LFNST / NSPT, MTSS) through M18's derived
transforms. Explicit MTS takes the CU's mode (getTrTypes: pu.intraDir[0] - an OBIC CU's first OBIC mode); LFNST / NSPT
take a DIMD or OBIC CU's own DIMD main mode (getFinalIntraModeForTransform's DIMD branch; intra merge: its mode), so
`selectTransform` reads the CU's DimdOut for DERIVE_DIMD / DERIVE_OBIC blocks. MTSS (JVET-AK0217: mdirIdx, parsed for
a DIMD CU with LFNST of 128 samples or more; no SPS switch) chooses another LFNST / NSPT mode: DIMD's second strongest
mode (candModeListForTransform[1], `DimdOut::trSecond`) or an OBIC CU's second OBIC mode, where its LFNST set or
transposition differs, else from a mode list of the CU's own (deriveDimdModeList: deriveDimdMode's three plain
histograms, `IntraPrediction::dimdHistogramCalls` with modeList; the top three, ties to the lower mode) - the CPU's
loop over that list does not stop, so its last entry wins where any differs, else its first -, and NSPT's second
bucket where the chosen mode is within 2 of the first. The CU's node computes all of it once (`Cu::mtss`, the list's
histograms in `s_dimdHist` after DIMD's / OBIC's; `mtssList`, `mtssChoice`), its blocks (DERIVE_MTSS) read the mode
and bucket from DimdOut (`DIMD_OUT_MTSS_*`), which verification compares with the CPU's own
getFinalIntraModeForTransform, as the list and DIMD's second mode. Test sets: m22 (the stock encoder with the
transforms, MIP off: MIP CUs with LFNST fall back, "LFNST of MIP") and m22o (encTest2, OBIC preferred, no intra merge).
Rejected: DIMD CUs with several transform blocks (the CPU blends planar predicted from the CU's references at the
block's size; the intra configuration's CTU 128 and transform blocks up to 128 leave none), implicit MTS (supported()).
M23: LFNST / NSPT of MIP blocks. Their mode is a DIMD of the block's own MIP prediction (deriveIpmForTransform, run
for every MIP block since OBIC: buildHistogram's 3x3 gradients inside the block, every second sample both ways from
128 samples on, the strongest mode `mipDimdMode` and the second `dimdDerivedIntraDir2nd`), so a MIP block with LFNST
(DERIVE_MIP_DIMD) derives it from its reduced prediction (`mipTransformMode`, the samples by `mipSample` as the
upsampling gives them). Every block of derived transforms chooses them (`selectTransform`) and computes its residual
(`inChainResidual`) in one place, after the references and MIP's reduced prediction, before predicting: a second copy
of that code in the MIP branch made the pictures with DIMD ~7 % slower (code that is merely present; 1 to 3 % on most
sets remain against M22, a candidate for the speed work). With MTSS (in 4 to
15 % of them) the MIP CU gets a node although its mode is known (CU_KNOWN, the host's mode kept), which derives only
MTSS's mode list; the block chooses with its second mode and the list (mtssChoice) and depends on that node.
Verification compares mipDimdMode, the second mode, the list and choice (the CU's DimdOut) and the block's
transforms. Test set m23: the m22 arguments with MIP. MTS of MIP takes planar's kernels (getTrTypes), unchanged.
Rejected: LFNST of MIP without DIMD in the SPS (the CPU's mode is planar then; no test), of a MIP CU of several blocks
and of chroma MIP (4:4:4).
M24: TIMD (plain: TIMD-SAD and TIMD merge are rejected per CU). A TIMD CU (`CU_TIMD`, with the DIMD its MPM list starts
with) gets a node (`intra_timd.glsl`'s `timdNode`) that does deriveTimdMode: M21's template costs, generalised to
templates of one arm (`TemplateUnits`, also in intra merge's record: at the picture's left / top edge the CPU takes the
above / left arm alone, with its own DC) and to costs per arm (the location dependence compares them), for planar, DC,
the angular modes of the MPM list built with isForcedValid (all 22 entries, then HOR / VER where missing:
`mpmSpatial` / `mpmFill`, shared with deriveLumaMode) and the wide angles beyond the regular ones that the host's four
PU checks let the reference reach (getTimdRegularAngleExt / getTimdWideAngleExt: `timdRegularAngle`, `timdWaipAngle`;
the CPU's loop skips the regular range by jumping to maxModeOrig - 1, so the next one is maxModeOrig + 4). All of them
are costed at once, then invocation 0 replays the CPU's sequential scan with its early ends (computing more costs
than the CPU changes no choice); a second batch costs the neighbours of the best and the secondary mode, replayed
with the CPU's quirks (the secondary's refinement on the best cost's condition, a refined best that does not demote the
old one). The final choice maps wide angles back, swaps, lets an angular best blend with itself where the secondary
costs twice as much (JVET-AG0092), and weighs three modes with planar / DC as the CPU's doubles do (exactly in
integers: floor((64 (sum - c) + sum) / (2 sum)) for costs far below 2^40) or two by X0149's table. Its TimdOut holds
every CodingUnit field deriveTimdMode sets and the initial values of those it leaves (location dependences of a
non-blended CU are the scan's leftovers), compared with the CPU's in verification; the published mode is MAP131TO67's
(what neighbours' MPM lists and chroma DM read). The blocks (`DERIVE_TIMD`) predict with 1/64-sample angles (TIMD's
tables, its wide angles where checkWA is off), the 1/64 N-tap filters from the filter buffer (`EXT_NTAP`), planar's
reference filter only, no reference filter or Gaussian interpolation for angular modes (JVET-AM0104 / AM0163); a blend
of the mode, the secondary and planar / DC goes through DIMD's per-class path with xGetNumTapsTIMD's filter lengths,
its location-dependent range 8 from 128 samples on, the secondary predicted from the next reference line alone
(no PDPC but of horizontal / vertical) where JVET-AG0128's rule fuses the two, which also overrides the CU's
fetchRef2nd (the next line is read in a CTU's first row too), and the bilateral filter on the blend only. Such a block
keeps its unfiltered line in `s_ref[0]` and the next line in `s_ref[1]`; its planar reads the filtered line from
`s_tmp`. The read ranges of TIMD blocks are both lines' whole arms. Transforms: explicit MTS and LFNST / NSPT by the
published mode, MTSS with the secondary (MAP131TO67) where the CU blends. The pictures with TIMD run a fifth pipeline
(`WITH_TIMD`). Rejected: TIMD merge (until M28), TIMD with ISP or MRL (no test stream: the stock encoder's TIMD-MRL
candidates hardly ever survive its SATD checks, none in the m24 set), TIMD CUs of several blocks, TIMD above 10 bits,
IntraTMP / IBC (the only sources of TIMD's block vector candidates). Test sets: m24, encTest4 (ECM_TEST_NO_OBIC,
NO_INTRA_MERGE, NO_TIMD_SAD) with the m23 arguments and `--EnableTMnoninterTools=1 --TIMD=1 --TIMDMerge=0` (without
EnableTMnoninterTools the encoder turns TIMD off); m24s, the same on screen_352x288, whose flat templates give angular
TIMD CUs of template cost 0, which do not blend (an angular best mode otherwise always does, with itself where the
secondary costs twice as much). GPU time of the m24 set: 47 to 68 ms per CIF picture, 133 (park720) and 195 (four720)
per 720p picture; without the TIMD nodes' template costs (an experiment) ~15 ms less per CIF picture, the first
candidate for the speed work. The other sets are unchanged within 1 %.
M25: OBIC and intra merge CUs with TIMD neighbours, both reading the neighbour's node (its TimdOut). deriveObicMode
counts a TIMD neighbour's mode (MAP131TO67) with all its samples in the mode's location dependence class and, where it
blends with a secondary other than itself, the secondary by its weight relative to the mode's, and with it the
non-angular mode by its own (`obicHistogram`). An intra merge candidate from a TIMD CU (`MERGE_NBR_TIMD`; valid where
its modes blend or its mode is angular, CU::isAValidNeighborForIntraMerge) converts its TIMD as
setIntraMergeCandParamFromTimd does (`mergeTimdParams`): the mode, the secondary as blend mode (where both map to one
mode of the 67, the mode two of their distances beyond), the weights reordered to [mode, non-angular as planar,
secondary], or without a secondary planar at 16; where the conversion gives up (a mode or blend mode planar / DC) the
candidate takes the CU's own DIMD (the host rejects a TIMD neighbour without one). Test sets: m25 (encTest4 with
ECM_TEST_NO_TIMD_SAD alone, the m24 arguments: intra merge CUs with TIMD neighbours) and m25o (encTest7, see section 9,
with ECM_TEST_PREFER_OBIC, NO_INTRA_MERGE, NO_TIMD_SAD, FORCE_OBIC, TIMD_SPARSE and TIMD_CHECKER=3). An OBIC CU never
has a TIMD CU among the neighbours that intra merge checks (left column, top row, corners and two CUs on from the
below-left and above-right ones: such a neighbour allows intra merge, which excludes OBIC), only among its farther
ones, and the stock encoder also drops OBIC wherever TIMD's SATD is 1.3 times lower; so the m25o encoder allows TIMD
only in sparse 8x8 islands and DIMD only there, and favours OBIC wherever it is available. m25s and m25os are the
two on screen_352x288, for TIMD neighbours that do not blend (as m24s). GPU time: m25 48 to 68 ms per CIF picture,
168 (park720) and 212 (four720) per 720p picture; m25o 30 to 55, 93 and 165; the m24 set ~1 % slower than with M24
(the new branches are code in the TIMD pipeline also where no CU takes them), the other sets unchanged.
M26: TIMD-SAD (JVET-AJ0146). A TIMD-SAD CU's node does deriveTimdMode as before, then deriveTimdModeSad, whose result
(the *Sad fields: the second half of the CU's TimdOut, from TIMD_OUT_SAD) its blocks predict. Its first batch is
planar, DC and a list of up to 25 modes (`timdSadCandidates`): the first 15 angular modes deriveTimdMode costed in its
MPM list's scan, up to that scan's early end, by cost then mode (m_timdModeCostList, stable-sorted), then the CU's DIMD
modes (all 8 entries), HOR, VER and the modes around TIMD's two and DIMD's, each once and none of the modes TIMD
found; then the wide angles beyond the regular ones as TIMD's. The scan, the second batch, the refinement and the
final choice are TIMD's (`timdScanFirst`, `timdSecondBatch`, `timdRefine`, `timdFinal`, shared), with as costs the
mean-removed SAD per arm (RdCost::xGetMRSAD: the arm's mean difference truncated toward 0, then the sum of the
differences' distances from it; its SIMD forms, in saturating 16-bit lanes, agree up to 11 bits, the GPU's TIMD stops
at 10), the scan passing over a list mode whose above arm alone costs the secondary cost or more (JVET-AN0093), and
no location dependence. Its template (TimdHead::ts) has TIMD's arms doubled, at most the CU's distance from the
picture's edge (2 at least, which also scales the costs where the arm is missing), 8 only where the samples 8 out
beside both ends of the CU's side are decoded CUs of its slice and tile (else 4); its reference units follow TIMD's
in the record (intra slices keep TIMD-SAD CUs at 1024 samples, which keeps them within intra_timd.glsl's
`s_timdUnits`; the host checks: TIMD_UNITS_MAX). The blocks predict as TIMD's from the *Sad fields
with the default filter lengths (xPredIntraAng takes numTapsTIMD only without TIMD-SAD). Rejected: TIMD-SAD without
DIMD in the SPS (its list takes the CU's DIMD modes; no test stream). Test sets: m26 (encTest4 with ECM_TEST_NO_OBIC
and NO_INTRA_MERGE, TIMD-SAD allowed, the m24 arguments: a few to 30 TIMD-SAD CUs per CIF picture), m26o (encTest8 with
ECM_TEST_PREFER_TIMD_SAD as well, plain TIMD untested where TIMD-SAD is: 30 to 70 per CIF picture). On
screen_352x288 the encoder chose no TIMD-SAD CU, even with the switch. TIMD-SAD's code runs in a pipeline of its own
(VARIANT_TIMD_SAD, WITH_TIMD_SAD): in the TIMD pipeline it slowed the TIMD pictures without TIMD-SAD by ~5 %, and
its MRSAD sums take s_dimdHist (shared memory of their own slowed every pipeline by ~3 %). GPU time: m26 96, m26o
73 ms per picture; m24 and m25 1 to 1.5 % slower than with M25, the other sets unchanged.
M27: OBIC and intra merge CUs with TIMD-SAD neighbours. OBIC's histogram (`obicHistogram`) and intra merge's
conversion (`mergeTimdParams`) read a TIMD neighbour's modes from the fields its blocks predict (`timdPredBase`: a
TIMD-SAD CU's *Sad fields); intra merge's validity of a TIMD-SAD neighbour (`mergeTimdValid`): its TIMD-SAD blends or
its TIMD-SAD mode is angular, or its plain TIMD's mode is (CU::isAValidNeighborForIntraMerge tests timdMode before
timdModeSad). Test sets: m27 (encTest8 with ECM_TEST_NO_OBIC and PREFER_TIMD_SAD, the m24 arguments: intra merge CUs
with TIMD-SAD neighbours), m27o (encTest8 with m25o's switches and PREFER_TIMD_SAD: OBIC CUs with TIMD-SAD
neighbours). GPU time: m27 92, m27o 75 ms per picture.
M28: TIMD merge (JVET-AJ0061, one merge candidate: NUM_TIMD_MERGE_MODES). A TIMD merge CU (`CU_TIMD` with
TimdHead::merge; no DIMD, no MPM list) gets a node (`timdMergeNode`) that does deriveTimdMergeModes. The host finds
searchTimdMrgCus' neighbours (up to five TIMD CUs, the nearest, in its order; IntraPrediction::searchTimdMrgCus is now
static) and puts their records in the TimdHead (their nodes are the merge node's dependencies). The node takes each
neighbour's fields as its blocks predict them (a TIMD merge neighbour's candidate, a TIMD-SAD one's TIMD-SAD) unless
a candidate has its mode and secondary already (genTimdMrgList), costs two or more by the SATD of their mode on the
CU's template of arms one sample thick (calcTimdMrgCandCosts calls CU::deriveTimdRefType with 1, not TIMD's 2 / 4),
with the SIMD kernels of such arms (JVET-AJ0096: 1x16, which leaves out its last butterfly stage and takes the even
samples' DC term; 1x8 scaled by 2 / sqrt(8); 1xN for 4), and takes the cheapest, or the second cheapest where it costs
less than 1.1 times as much (the CPU's comparison in doubles exactly in integers: `timdMergeSwaps`). Its TimdOut holds
the candidate's fields with every checkWA set (timdMrgModeCheckWA); its blocks predict as a TIMD CU's but for the
fusion of a blend (initPredIntraParams' TIMD merge branch, DERIVE_TIMD_MERGE): the next line always fetched, the
secondary predicted from it alone where its slope is fractional. Transforms: no MTS or transform skip; without LFNST in
dual tree (TrQuant::xIT's TIMD merge branch) a side of 4 to 16 takes DST-VII, one above IntraMTSMaxSize DCT-II, any
other the candidate's type (timdmTrType: DERIVE_INHERIT_HOR / VER, with DERIVE_TRANSFORM), for which every TIMD CU's
node stores its own getTrTypes types in its TimdOut (TIMD_OUT_TR_*: explicit MTS's by its mode, else the host's), in
the pipeline with TIMD merge only. Intra merge drops a TIMD merge neighbour whose TIMD does not convert (it has no
DIMD to fall back on: MERGE_NBR_TIMD_MERGE); OBIC reads it as a TIMD neighbour. Verification compares the candidate's
fields and every TIMD CU's transform types. The pictures with TIMD merge run a pipeline of their own
(VARIANT_TIMD_MERGE, WITH_TIMD_MERGE; with TIMD-SAD). Test sets: m28 (encTest4 with ECM_TEST_NO_OBIC and
NO_INTRA_MERGE, TIMD-SAD allowed, the m24 arguments with `--TIMDMerge=1`: 36 to 164 TIMD merge CUs per CIF picture,
573 per four720 picture, of whose blocks 62 inherit transform types), m28m (encTest4 without switches: ~930 intra
merge CUs per four720 picture next to 287 TIMD merge CUs). GPU time: m28 96, m28m 110 ms per picture; the other sets
unchanged within 0.4 % (the kernels of the one-sample template exist in the TIMD merge pipeline only).
M29: TMRL of luma (JVET-AB0157 with JVET-AD0085's 131-mode set, the DIMD modes first in its list: JVET-AL0125). A
TMRL CU (`CU_TMRL`, with its DIMD and MPM neighbours; tmrlListIdx in Cu::ipredIdx) gets a node (`tmrlNode`,
intra_tmrl.glsl) that does getTmrlList: the ten modes of getTmrlSearchRange (`tmrlModeList`: the CU's DIMD modes, the
neighbours' modes in the 131-mode set - a TIMD neighbour's from its TimdOut, a TMRL one's from its TmrlOut, any other's
MAP67TO131 - planar never, then fillMPMList's angles around them and its defaults), on each reference line 1, 3, 5, 7,
12 inside the picture (L < y), each line filled as xFillReferenceSamples fills it for the CU (its availability scanned
by the node: `refLineAvailability`, shared with intra_deps), predicted on the template one sample thick (the above arm,
the left one where x > 0) as xPredTmrlIntraAng / xPredTmrlIntraDc do, costed by the SATD of TIMD merge's one-sample
kernels; it takes the entry of rank tmrlListIdx in the costs' stable order (updateCandList keeps the earlier of equal
costs first; the decoder's early skip of a candidate whose above cost alone exceeds the list's last changes nothing) and
writes mode and line to its TmrlOut (Params::tmrlOut). Its block (DERIVE_TMRL) predicts that mode (1/64-sample angles,
no PDPC, no reference filter) from that line, fused with the next line where the CU's fetchRef2nd (size and the SPS's
switch only: a TMRL line is never 0, and 12 only below row 12) and the slope allow it, the bilateral filter for DC
only; as intra_deps cannot know the line, the block depends on the whole arms of every line its node may choose and of
the next one, and finds its line's availability itself. Rejected: chroma TMRL (JVET-AJ0081, next), TMRL CUs of several
blocks, TMRL above 10 bits (the SATD's 16-bit lanes, as TIMD). The pictures with TMRL run a pipeline of their own
(VARIANT_TMRL, WITH_TMRL; with TIMD merge). Test sets: m29 (encTest9, see section 9, with ECM_TEST_NO_CHROMA_TMRL; the
m24 arguments with `--TIMDMerge=1 --TMRL=1`: 74 to 358 TMRL CUs per CIF picture, 462 per park720 and 389 per four720
picture) and m29n_fore32 (the same with `--DIMD=0 --TIMD=0 --TIMDMerge=0 --MIP=0`: a mode list without DIMD modes;
without DIMD the GPU rejects LFNST of MIP) and m29s_screen32 / 22 (screen_352x288: 135 to 181 TMRL CUs per picture).
GPU time: m29 130, m29n_fore32 34 ms per picture (the TMRL nodes' up to
50 template predictions and their line fills are the next speed candidate); the other sets unchanged within 1 %.
M30: chroma TMRL (JVET-AJ0081, dual tree; the only chroma MRL). A chroma TMRL CU (`CU_CHROMA_TMRL`; chromaTmrlIdx in
Cu::ipredIdx, its TmrlOut shared with luma TMRL's) gets its DIMD chroma like a DIMD chroma CU (deriveDimdChromaMode and
DecCu's exchange with the co-located luma mode) and a node (`chromaTmrlNode`) that does getChromaTmrlList: eight modes of
the 67-mode set (`chromaTmrlModeList`: the co-located luma mode, the DIMD chroma modes, DC, the luma CUs' at the centre
and corners of the co-located area, the final modes of the chroma neighbours at five positions where they are intra and
neither LM nor DBV - the host records these ten CUs in Cu::nbr - then fillMPMList's, `mpmFill` with a list size), on
each chroma line 1, 3, 5 inside the picture (PU::hasChromaTmrl allows it from chroma row 4 on), costed as 9 x the SATD
of the co-located luma block predicted from the luma line twice as far plus the SAD of the Cb and Cr templates, one
sample thick, the above arms weighed by 4 x the CU's height, the left ones (where x > 0) by 4 x its width, in 64 bits.
The luma prediction is predCoLuma's: a plain luma PU (no PDPC, reference filter, fusion or bilateral filter; the cubic
filter, 10 taps along rows shorter than 8) from xFillReferenceSamplesForCoLuma's line (`coLumaReference`, a literal port
on one invocation: its availability is by position - inside the picture, above-right and below-left only up to the
CTU's edges - with quirks in its unit flags, count and padding), the reference arrays of `angularArrays` (main()'s
line-0 setup, factored out for it). Its SATD (`coLumaSatd`) takes xGetHADs_SIMD's AVX2 tiles (16x8, 8x16, 8x4, 4x8, 8x8
- the 16x16 kernel sums four - and 4x4), band by band of 16 rows through s_tmp, the 16x8 / 8x16 scale floor(sad /
sqrt(32)) exactly in integers (`hadScaleSqrt32`, checked for every sum below 2^25). The templates' chroma lines have
arms of 2 x (the side + 1), whose last unit may be partial (fillSource rounds the unit counts up now, as the CPU), and
xPredTmrlIntraAng steps through the regular 1/32-sample angles as 1/64 samples (`chromaTmrlParams`). The Cb and Cr
blocks predict the chosen mode as regular chroma blocks with that line (DERIVE_TMRL with applyModeParams; no fusion),
their availability found at run time as luma TMRL's; intra_deps gives them the whole arms of every candidate line, the
node the co-located luma block, its lines up to the CTU's edges, the chroma lines, templates and the ten CUs' nodes.
Rejected: chroma TMRL without DIMD in the SPS (no test stream), above 10 bits or with unequal luma and chroma bit
depths, in a picture smaller than the SPS's (the positional availability takes the SPS's size), CUs of several blocks.
Test set: m30 (encTest9 without ECM_TEST_NO_CHROMA_TMRL, m29's arguments: up to 84 chroma TMRL CUs per CIF picture,
134 to 218 per 720p picture, among them 16x2 ones whose co-located luma block is 4 high). GPU time: m30 135 ms per
picture; m29 131 -> 135 (the chroma node's code is in the TMRL pipeline also where no CU takes it: a pipeline of its own
would win that back), the other sets unchanged within 0.5 %.
M31: CCLM (LMChroma: LM_CHROMA, MDLM_L / MDLM_T and, with MMLM, their two-model variants; ECM's least-squares model,
LMS_LINEAR_MODEL). A chroma block in an LM mode carries Tu::lm's LM_ACTIVE: its mode number does not tell, as a luma
TIMD / TMRL block derives modes of the 131-mode set, whose values from 67 on are angular (a first version that took
them for LM read luma at twice their position, past the end of the plane, and lost the device). Its model is fitted on
one invocation (`cclmModel`, intra_cclm.glsl): xGetLMParametersLMS's template samples (availability in chroma units of
2; MDLM's 2W / 2H arms where the above-right / below-left units are available) of the luma downsampled as
xGetLumaRecPixels does (JVET-AN0090's filter by the SPS's collocated flags: the mean of 2 x 2, the 5-tap cross, or
1 2 1 over two rows; in a CTU's first row from the one luma row above), in 64-bit sums (JVET-AJ0237), the division
table m_auShiftLM computed in the shader; MDLM pads its template to a power of two, MMLM splits at the mean
(divSigTable) and pads each group (xLMSampleClassifiedTraining), and each model's slope takes the CU's cclmOffsets
(JVET-Z0050, xUpdateCclmModel around the middle luma of that model's samples). All invocations then apply it to the
downsampled co-located block, with LBCCP's 3x3 filter (JVET-AD0120, ccInsideFilter: the reconstruction above / left
where available, else the prediction's border) where the CU takes it. intra_deps gives an LM block the read range
2W / 2H and the owners of the co-located luma block and of the 3 luma rows above it and 3 columns left of it, each
twice the co-located block's side long (MDLM's template). The host gives LM blocks no angular, PDPC, reference filter or fusion parameters
(initPredIntraParams leaves the angles as the previous block's; ECM_GPU_DUMP's records differed there, and dumpdiff.py
now also leaves a TMRL block's line, which its node chooses, out). Rejected: GLM, CCCM, LM chroma fusion (JVET-AC0119),
CCLM with LFNST / NSPT (its mode is a DIMD of the CU's Cb and Cr predictions: next), CCLM CUs of several blocks
(DecCu predicts Cr over the CU's area with the Cb block, JVET-AK0064), above 10 bits or with unequal bit depths; CCP
merge, decoder-derived CCP fusion and BVG-CCCM at sequence level, as CCCM. Test sets: m31n (encTest10, see section 9,
with ECM_TEST_NO_GLM and ECM_TEST_NO_CHROMA_FUSION; m29's arguments with `--LMChroma=1 --IntraLFNSTISlice=0 --NSPT=0
--LFNSTExt=0`: 170 and 184 CCLM blocks per fore22 picture, 1526 and 1680 per park720 picture, all variants: MMLM,
MDLM, LBCCP; the 1 2 1 filter), m31a_fore32 / 22 (`--HorCollocatedChroma=0`: the 2 x 2 mean), m31c_fore32 / 22
(`--VerCollocatedChroma=1`: the cross) and m31 (with LFNST: M32). The CCLM code is in the
pipelines with WITH_CCLM only (a specialization constant beside each IntraRecVariant; `intraRecKernel` builds the
pair on first use): in all pipelines it slowed the plain set by 30 % and the TIMD ones by 5 %. GPU time: m31n 134.1,
m31a / m31c 84.6 ms per picture; the other sets unchanged within 0.8 % (plain 12.28 -> 12.38).
M32: LFNST / NSPT of CCLM blocks (JVET-AK0064). Their mode is deriveChromaIpmForTransform's, which DecCu calls with the
Cb block: the strongest mode of buildHistogram's 3x3 gradients (no subsampling) inside the CU's Cb prediction, its Cr
prediction or both, by the TU's joint CbCr mode (1: Cb, 2: Cr, 0 and 3: both; Tu::lm's LM_HIST_*). A block that needs
it (DERIVE_CCLM_DIMD with DERIVE_TRANSFORM; with joint CbCr whose Cr is coded also the Cr block, which decides the
signs with the Cb block's transforms) finds it before its transforms (`cclmTransformMode`): the other component's
references into s_ref[1] (its availability is the block's own: both chroma components' owner maps, decoded before
it, and none of them reads the CU), its model into a second slot (its slope adjustments carried in Tu::lm), both
predictions into s_tmp up to 32 x 32 (larger blocks, rare, compute each gradient's samples again), then selectTransform
with that mode (s_predDimdMode, MIP's too). intra_deps (and the host) add the owners of the other component's line.
With the reconstruction on, the CPU does not predict, so DecCu gives the CU's ccpChromaDimdMode a placeholder in range
for the host's transform choice (the GPU chooses again). Rejected: without DIMD in the SPS (the CPU's mode is then
unset). Test set: m31 (M31's arguments with LFNST: 5 to 31 such blocks per CIF picture, 77 to 122 per 720p picture,
one above 32 x 32 in park720). GPU time: m31 137.7 ms per picture; plain 12.38 -> 12.59 (code it never runs; a
fillReferencePar with its old signature measured the same), the other sets within 0.6 %.
M33: GLM's gradient patterns (JVET-AA0126; glmIdc 1 .. NUM_GLM_PATTERN = 4 with CCCM). A CCLM block with one models
the pattern's gradient instead of the downsampled luma, everywhere xGetLumaRecPixels takes it: its template (the row
above from the rows around ly - 2 also in a CTU's first row) and the co-located block (`glmGradient`: xGlmGetLumaVal,
which with JVET-AB0092 returns the gradient alone, as a Pel; the taps left of, at and right of the luma sample in its
row, the row below and, for the cross location type of JVET-AP0105, the row above, padded as the downsampling pads;
the patterns of the three location types in intra_cclm.glsl's GLM_PATTERN, Rom.cpp's g_glmPattern). Tu::lm carries the
pattern (LM_GLM_SHIFT) and with LFNST / NSPT the other component's (cclmTransformMode's other prediction). Rejected:
GLM with luma (glmIdc 5 .. 8: a CCCM model, xGlmCalcModel; next after CCCM). Test sets (encTest11, see section 9, with
ECM_TEST_GLM_GRADIENT_ONLY and ECM_TEST_NO_CHROMA_FUSION): m33 (M31n's arguments without its LFNST switches, which
default off: 14 to 72 GLM blocks per CIF picture, 92 per four720 picture, none in park720), m33l_fore32 / 22 / 37
(with LFNST and NSPT, as m31), m33a_fore22 / m33c_fore22 (the averaging and the cross location types, with LFNST).
GPU time: m33 130.3 ms per picture; the other sets within 0.7 %.
M34: CCCM (JVET-AA0057), the plain kind. A chroma block of a CCCM CU (an LM mode with cccmFlag) fits each chroma
component a 7-tap model - the downsampled luma sample (minus a luma offset, JVET-AB0174), its four neighbours, a
nonlinear term, a bias - to the template of its reference area (up to 6 lines above / left of the block with the
above-right / below-left units available; the CU's template selection, JVET-AB0143: both sides, left, above) by
CccmCovariance::solve2's division-free least squares (JVET-AC0053's Gaussian elimination): one model, or for the MMLM
modes two split at the template's mean luma; the prediction is clipped to the template chroma's range (JVET-AP0168).
In intra slices with the TM non-inter tools LBCCP lets a second set compete by the SATD of the template's chroma, Cb's
and Cr's summed: a single model fitted to the template's two lines next to the block, or two models split at the
block's mean luma (still fitted to the whole template: the first version fitted them to the two lines too); then
LBCCP's inside filter (intra_cclm.glsl's, reading the CCCM prediction). intra_cccm.glsl builds the luma buffer in
xCccmCreateLumaRef's order (the padding right of / below the area's parts, then in a dual tree the luma right of /
below the block within its CTU over it) in the four angular arrays, or for areas above 7168 samples in the workgroup's
scratch (blendAcc); per model 42 sums (ATA's upper triangle, ATCb, ATCr) of 6 invocations each, 32-bit (the host
rejects W + H > 164: chroma blocks above 64 x 64 / 128 x 32, which need a CTU of 256); the 7 x 9 elimination one row
per invocation; the models as 64-bit words in s_spCost; the costs with timdHadLine's one-sample-thick kernels. Each
block (Cb, Cr) fits both components' models: the choice needs both costs. Host: Tu::lm's cccmFlag (LM_CCCM_SHIFT),
Params::dualTree / cccmLbccp, the WITH_CCCM pipelines (LmLevel beside each IntraRecVariant), a CCCM block's parameter
snapshot reset (DecCu calls no initIntraPatternChType for it, so it held the previous block's, a chroma TMRL CU's
reference line too); intra_deps and the host add both components' templates and the luma area. Rejected: GL-CCCM,
CCCM without subsampling, with several downsampling filters, BVG-CCCM, with LFNST / NSPT (block level; next), CCP
merge and decoder-derived CCP fusion (sequence level). Test sets (encTest13, see section 9): m34 (m33's arguments with
`--CCCM=1`: 22 to 244 CCCM blocks per CIF picture, 666 to 822 per 720p picture, all template selections, one and two
models), m34a_fore22 / m34c_fore22 (the averaging and the cross location types). The scratch luma buffer and the
prediction recomputed for blocks above 2048 samples were also verified with both thresholds lowered (896 / 64
samples, so most blocks take them). GPU time: m34 136.3 ms per picture (m33, its arguments without CCCM, 130.6); the
other sets within 0.3 % (the CCCM code is in the WITH_CCCM pipelines only).
M35: GL-CCCM (JVET-AC0054). A GL-CCCM CU's model takes, instead of the luma sample's four neighbours, a vertical and a
horizontal gradient of its 3x3 neighbourhood (above minus below, left minus right, the middle tap doubled) and the
sample's row and column in the reference area ((y + 8) << 3, (x + 8) << 3); LBCCP's second set does not apply to it
(predIntraCCCM's condition), its inside filter does. Tu::lm's LM_CCCM_GL (bit 12: a CCCM block has no other
component's slope adjustments). The gradient products reach 2^26, so a model's sums are now 64-bit: each lane
accumulates in 64 bits and stores its partial sum in s_tmp (four 16-bit words per value), 42 invocations add each
sum's six; M34's 32-bit bound (W + H <= 164) is gone, the host rejects only a block whose largest luma buffer and
prediction do not fit the workgroup's scratch (128 x 128 chroma, CTU 256 only). A block above 2048 samples keeps its
prediction in the scratch after the luma buffer instead of computing it again at each read: M34's recomputation,
inlined at every read of LBCCP's inside filter, made intra_rec's SPIR-V 3.3 MB (M33 2.4, M35 2.7) and its glslang -Os
twice as slow (107 against 61 s, both under load). Test set (encTest14, see section 9): m35 (m34's arguments: 22 to 90
GL-CCCM blocks per CIF picture, 280 to 516 per 720p picture, plain CCCM beside them). Also verified with the shared
capacities lowered (896 / 64 samples). GPU time: m35 137.2 ms per picture, m34 136.9 -> 135.5 (the smaller code), the
other sets within 0.9 %.
M36: CCCM with several downsampling filters (JVET-AD0202, cccmMultiFilterIdx 1 .. 3). Its model has 10 (mode 1) or 11
(modes 2, 3) parameters over the luma of two or four of xCccmGetLumaVal's filters (filter 0 the mean, 1 to 3
gradients with negative values 0, per location type): mode 1 the samples of filters 0 to 3, three nonlinear terms,
the row and the column; mode 2 the sample and its left / right neighbours of filters 0 and 1, filter 0's nonlinear
terms, the column; mode 3 the neighbours above-right / below-left, filter 3, the row; no LBCCP second set. The CCCM
code now takes M parameters (the sums' numbering, as many lanes per sum as fit the workgroup: 6, 3 or 2; the
elimination's rows after it in s_tmp, the pivot row in s_spCost) and a luma buffer per filter, all of them in the
workgroup's CCCM scratch, so the rejection of CCCM blocks above BLEND_ACC is gone: Params::blendAcc became
Params::scratch, blending's sums at their constant stride, then from Params::cccmScratchOffset per workgroup
Params::cccmScratchWords (the picture's largest CCCM block's buffers and prediction). Two versions measured worse:
one stride read from Params for both (blending's too) made the m19 set 4 % slower (81.2 -> 84.5 ms per picture), a
CCCM buffer of its own (its address a further Params field) the CCCM sets 0.7 % slower than this layout. Tu::lm's
LM_CCCM_MDF_SHIFT (bits 13, 14). Test set (encTest14 without its CCCM switches: GL-CCCM and every filter):
m36 (m35's arguments: 26 to 142 CCCM blocks of several filters per CIF picture, 250 to 320 per 720p picture, beside
plain and GL-CCCM ones). Also verified with the shared capacities lowered (896 / 64 samples). GPU time: m36 143.4 ms
per picture; m34 135.3 -> 136.2, m35 136.9 -> 137.9, the other sets within 0.6 %.
M37: CCCM without subsampling (JVET-AC0147, cccmNoSubFlag; CTC's CCCM=2). Its luma buffer (xCccmCreateLumaNoSubRef)
holds the luma itself, minus the luma offset, over the reference area at twice its size per direction, padded by two
samples; the model's taps lie around the co-located luma sample (2x, 2y), by the chroma location type
(xCccmCalcModels' chromaloctype): the averaging type's 2 x 2 samples and their nonlinear terms (9 parameters), the
cross's sample and its four neighbours (11), the 6-tap filter's two rows of three and four nonlinear terms (11). The
template and the prediction are at the chroma positions (every second luma sample), the two-model threshold is the
mean of all the template's luma samples (four per chroma sample); no LBCCP second set; no minimum clipping span (above
7 parameters). The taps reach at most (2x + 1, 2y + 1), never the buffer's right / bottom padding, which is built
anyway as the CPU builds it. intra_cccm.glsl's luma buffers are now in the buffer's units (g_cccmShift: cccmBufferW,
cccmLumaIndex, cccmColocated; cccmBuildLuma builds both kinds, the padding a chroma sample wide, in a dual tree the
CTU checks of luma positions against the chroma CTU size as on the CPU); a buffer up to 7168 samples stays in shared
memory. Tu::lm's LM_CCCM_NOSUB (bit 15). Test set (encTest14 without its CCCM switches, `--CCCM=2`): m37 (m36's
arguments: 10 to 42 CCCM blocks without subsampling per CIF picture, 54 to 442 per 720p picture, beside the other
kinds), m37a_fore22 / m37c_fore22 (the averaging and the cross location types). Also verified with the shared
capacities lowered (896 / 64 samples). GPU time: m37 138.1 ms per picture; m34 135.9 -> 136.6, m35 137.6 -> 138.3,
m36 142.8 -> 143.6, the other sets within 0.7 %.
M38: CCCM with LFNST / NSPT (JVET-AK0064 for CCCM CUs). Their mode is the DIMD of the CU's predictions
(deriveChromaIpmForTransform: Cb's, Cr's or both by the TU's joint CbCr mode), which predIntraCCCM makes for both
components at the Cb block, LBCCP's inside filter applied. On the GPU each such block (cclmTransformMode's CCCM
branch) predicts both components with cccmPrediction's `both` (Cb's prediction, then Cr's: in s_tmp up to 2 x 1024
samples, else in the CCCM scratch, which the host sizes for two), the gradients read them through the inside filter
where the CU has it, and the prediction section predicts the block again (as for CCLM). The host leaves a CCCM
block's other-component field to its kind (no slope adjustments). The second prediction showed a bug: wgSum2's s_red
slots must be zero before each use and the block's start clears them once, so the second two-model threshold
included the first one's sum; cccmPrediction now clears its slots (the threshold's, LBCCP's block mean). Found with
temporary debug builds: the per-block comparison of predictions and residuals past the first parameter mismatch (the
first difference a two-model block's prediction, off by one), then a build without the first prediction (right).
Test set (encTest14 without its CCCM switches, `--CCCM=2 --IntraLFNSTISlice=1 --NSPT=1 --LFNSTExt=1`): m38 (6 to
51 CCCM blocks with LFNST / NSPT per CIF picture, 75 to 118 per 720p picture, of every kind, with and without the
inside filter, every joint CbCr mode). Also verified with the shared capacities lowered (896 / 64 samples). GPU time:
m38 143.3 ms per picture; the other sets within 0.7 % (m31, CCLM with LFNST / NSPT, the same).
M39: GLM with luma (JVET-AB0092, glmIdc 5 .. 8 of an LM_CHROMA / MDLM_L / MDLM_T block: xGlmCalcModel /
xGlmApplyModel). A single model of 3 parameters - the value of the block's GLM gradient pattern (xGlmGetGradVal: no
luma offset, the samples left of / above the picture its first column / row), the downsampled luma minus the luma
offset, a bias - over CCCM's reference area, its template by the LM mode (MDLM_L the left part, MDLM_T the part above,
LM_CHROMA both without the units above-right / below-left), solved as CccmCovariance::solve1, which is solve2's
elimination of one right-hand side with the matrix scaled to 28 bits instead of 22 (each right-hand side is
eliminated alone, so solve2's model of the component is solve1's), clipped to the sample range alone; no inside filter,
no LBCCP. intra_cccm.glsl has it as a model kind (g_cccmGlmLuma: two luma buffers, the second the gradient
pattern's, CCCM_FILTER_GLM; cccmTemplate's restriction; GLM_MATRIX); the host records it as a CCCM block (Tu::lm's
LM_CCCM_GLM_LUMA, bit 16, the template selection its LM mode's, its glmIdc in the GLM field), so the CCCM pipelines,
dependencies and scratch serve it. Still rejected with LFNST / NSPT (the DIMD of both components' predictions, whose
kinds may differ). Test set (encTest14 without ECM_TEST_GLM_GRADIENT_ONLY, m33's arguments: CCCM off): m39 (8 to 70
blocks of GLM with luma per CIF picture, 58 to 146 per 720p picture), m39a_fore22 / m39c_fore22 (the averaging and the
cross location types). Also verified with the shared capacities lowered (896 / 64 samples). GPU time: m39 138.5 ms per
picture; the other sets within 0.8 %.
M40: LM chroma fusion (JVET-AC0119), its first type (isChromaFusion 1; CFLM, types 2 and 3, still rejected): a chroma
block of a regular mode blended with the plain CCCM prediction of two models (geneChromaFusionPred with JVET-AL0191 in
an intra slice: MMLM_CHROMA_IDX's models, the whole template, xCccmCalcRefAver's threshold, xSetMinMaxClpRng's
clipping, no LBCCP, no inside filter), (pred * w0 + cccm * (4 - w0) + 2) >> 2 with w0 1 where the CUs below-left
and above-right of the CU (getPURestricted) are both LM CUs, 3 where neither is, else 2 (the host's). The block
computes the CCCM prediction first, its luma buffer and prediction in the CCCM scratch (the regular prediction then
takes s_tmp and the angular arrays; cccmPrediction's g_cccmFusion), and storeRegion, which every final prediction
passes, blends each sample (cccmFusionBlend: after PDPC, MRL, intra fusion). Tu::lm's LM_FUSION (bit 30) and the
weight (bits 0, 1: the first slope adjustment's field, which a fusion block does not have), the template selection 1
of the models (not LM_ACTIVE: the block is not an LM block), so the CCCM pipelines, dependencies and scratch serve
it. A first version had LM_FUSION at bit 17, inside a CCLM block's field of the other component's slope adjustments
(bits 12 .. 19, with LFNST / NSPT): CCLM blocks of LFNST whose other component's second adjustment set that bit took
the fusion path (the full regression's m38_park720 / four720 differed; the scratch they wrote beyond was unsized).
Rejected where PU::cccmMultiModeAvail(MMLM, fusion) does not allow the CCCM models (the CPU then blends in the LM
buffer of a previous block), for a CU of several transform blocks, and CFLM. Test sets (encTest15: encTest14 with
ECM_TEST_CHROMA_FUSION_LM_ONLY, the fusion search without CFLM; encTest16 adds ECM_TEST_PREFER_CHROMA_FUSION, a
fusion candidate's RD cost divided by 4): m40 (m37's arguments, chroma fusion on: 0 to 8 fusion blocks per CIF
picture, all of weight 2, 1 in park720, 28 in four720, of every weight), m40f_fore32 / 22 / 37 (fusion preferred:
192 to 292 fusion blocks per stream, every weight, sizes from 2 x 8 to 64 x 64). Also verified with the shared
capacities lowered (896 / 64 samples). GPU time: m40 147.1 ms per picture, m40f 85.2; the CCCM sets 0.8 to 0.9 %
slower (m34 136.1 -> 137.3), the others within 0.9 %.
M41: CFLM, LM chroma fusion's types 2 and 3 (JVET-AC0119's isChromaFusion 2: one model, 3: two split at the
template luma's mean, xCflmCalcRefAver's PU::getMeanValue): the block's regular prediction (any mode, after PDPC, MRL,
intra fusion) is replaced by a model of 3 parameters - the downsampled luma minus the luma offset, the regular
prediction minus the chroma offset, a bias - fitted by solve1 (GLM_MATRIX bits, xSetMinMaxClpRng's clipping) to the
template of 2 lines above / 2 columns left of the block (those the picture has; no corner, no units above-right /
below-left), where the regular prediction is the block's mode predicted for the template (xCflmCreateChromaPred:
xFillTimdReferenceSamples for the template block with arms of twice its sides, predTimdIntraAng's chroma path with
initPredIntraParams' parameters of the block: the 1/32-sample angles interpolated by two taps, planar, DC and the PDPC
kinds as TIMD's). The GPU: intra_timd.glsl's template code with TimdParams::chroma and timdFillReference of any
component and arm length; intra_cccm.glsl's cflmArea before the block's references, storeRegion keeping the regular
prediction in the area's chroma buffer in the CCCM scratch (cflmStore), cflmPrediction once it is complete (the luma
buffer like CCCM's, cccmAccumulate / cccmSolve / cccmPredict with g_cccmCflm: the template without its corner, the
features, solve1). The host records the template's reference units (templateUnits of any block and arm length:
IntraPrediction::timdReferenceUnits takes the component and the arms' lengths) in Params::timdData at Tu::cflm (a new
field: Tu is 101 int32) and the fusion type in Tu::lm's bits 4, 5 (LM_FUSION_TYPE_SHIFT; the first type's records
too). A CPU quirk the GPU reproduces: xFillTimdReferenceSamples' units of the row number predSize / unitW rounded up
but begin after the corner unit, so with CFLM's arms of 2 and chroma units of 2 the last one writes 2 samples past the
row, into the column's first entries (the CPU's buffer holds the column right after the row): in the fill where the
unit is available (a later padding of the corner unit or from the first available unit overwrites them), in its
padding where it is not (TIMD's luma units of 4 end with the row); the first verification found it (template samples
next to the corner off by a few, the model's prediction by 1). The CFLM code is in pipelines of its own (WITH_CFLM,
LmLevel's LEVEL_CFLM): in the CCCM pipelines it slowed the CCCM sets by ~2 %. Rejected: CFLM with cccmNoSubFlag, a
CU of several transform blocks; LM chroma fusion of either type above 10 bits or with unequal bit depths (the first
type's check is new). Test sets (encTest16 without ECM_TEST_CHROMA_FUSION_LM_ONLY: the fusion search with every type):
m41 (m40's arguments: 2 to 92 CFLM blocks per stream, both types), m41f_fore32 / 22 / 37 (ECM_TEST_PREFER_CHROMA_FUSION
too: 84 to 186 CFLM blocks per stream, every kind of mode, templates of both arms and of one, sizes 2 x 8 to 64 x 64).
Also verified with the shared capacities lowered (896 / 64 samples). GPU time: m41 148.1 ms per picture, m41f 87.0;
the other sets within 0.6 %.
M42: GLM with luma with LFNST / NSPT (their mode the DIMD of both chroma components' predictions). M39 rejected it,
taking the components' kinds for independent; they are not: CABACReader::glmIdc gives Cr Cb's glmIdc (JVET-AB0092)
and no slope adjustment comes with GLM (PU::hasCclmDeltaFlag), so the other component's model is of the block's kind
and gradient pattern, which cclmTransformMode's CCCM path (cccmPrediction of both components) already predicts. The
host writes the other component's slope adjustments and glmIdc into a plain CCLM block's Tu::lm only (a GLM-with-luma
block's bits 12 .. 19 hold its CCCM kind: they would have set LM_CCCM_NOSUB) and rejects components of different
glmIdc (no stream can have them). Test set (m39's encoder and arguments with LFNST, NSPT and LFNSTExt on): m42 (1 to
38 blocks of GLM with luma with LFNST / NSPT per stream, LM_CHROMA and MDLM_L, every pattern). GPU time: m42 144.4 ms
per picture; no kernel change.
M43: the SPS switches of CCP merge (JVET-AD0188), decoder-derived CCP fusion (JVET-AG0154) and BVG-CCCM (JVET-AE0100)
accepted; their CUs are rejected per block (idxNonLocalCCP, decoderDerivedCcpMode; BVG-CCCM by otherCccm as before),
their pictures reconstructed by the CPU. The switches only add the tools' syntax and their candidates' reordering:
the other LM blocks are predicted as without them, and in ON mode the derive-only pass predicts nothing, so it stores
no CCP model (CU::saveModelsInHCCP takes typed models only) a merge CU could inherit. What the tools need on the GPU
(from the decoder: DecCu::xIntraRecBlk 1416 .. 1617): every LM block's model stored as PredictionUnit::curCand
(CCPModelCandidate, Common.h; set in predIntraChromaLM / predIntraCCCM) and pushed after its CU into the CTU row's
history (cs.ccpLut, CU::saveModelsInHCCP; reset per slice and CTU row) and the picture's model map; a merge CU's
candidates (PU::getCCPModelCandidateList: 5 adjacent positions, non-adjacent ones, the history, defaults; pruned by
equal models) ranked by template costs (reorderCCPCandidates), with adjustment (adjustCCPCandidates), fusion with
CCCM or DIMD chroma; decoder-derived CCP fusion's candidate models costed on the template and the best pair blended
(decoderDerivedCcp); BVG-CCCM needs IBC / IntraTMP luma CUs (bvgCccmModeAvail). Test sets (encTest17: encTest16 with
ECM_TEST_NO_CCP, the chroma searches' conditions of the three false; the m41 set's arguments with --CCPMerge=1
--DDCCPFusion=1 --BvgCccm=1): m43 (no CU of the tools), m43c_fore32 / 22 / 37 (without ECM_TEST_NO_CCP: every picture
has CCP merge and decoder-derived CCP fusion CUs, so the CPU reconstructs all 8).
M44: the LM CUs' CCP models, CCP merge's first step. Every LM CU's blocks write the model the CPU keeps as
PredictionUnit::curCand (predIntraChromaLM's cclmModelToCcpParams, predIntraCCCM's cccmModelToCcpParams,
xGlmCalcModel's glmModelToCcpParams) into a record of CCP_OUT int32 in Params::ccpOut (at Tu::ccp, a new field: Tu is
102 int32; one record per PU): the type (CCLM, GLM's gradient patterns, GLM with luma, CCCM, GL-CCCM, several filters,
without subsampling; MMLM with two models), glmIdc, the threshold of two models, the luma offset, the reference
area's columns / rows left of / above the block (0 without subsampling), the filters' index, the parameters' number,
per component the shifts, per model and component the template chroma's range and the 64-bit parameters. A CCLM
block writes its component's part after its model (cclmStoreCcp: a and b with the slope adjustments, the shift), the
Cb block of a CCCM CU both components' after LBCCP's choice (cccmStoreCcp: the chosen set's models and threshold; each
block fits both components' models), each GLM-with-luma block its component's. Verification compares each record's
fields of its type (those CCPModelCandidate::operator== and predCCPCandidate read) with the CPU's curCand after the
block's prediction. Nothing reads the records yet. Not covered: LM chroma fusion CUs (their mode is not an LM mode,
but geneChromaFusionPred stores their CCCM models as curCand: spatial merge candidates) and LBCCP's inside filter,
which CU::saveCcInsideFilterFlagInCCP adds to curCand after the CU (not to the history's copy). The CPU predicts both
components of a CCLM CU at the Cb block (JVET-AK0064), so its curCand is complete after either block. Canaries: 14 (a
field of each type off, the threshold, the ranges, LBCCP's set, NS-CCCM's type / corner offsets / parameters' number,
a record per block instead of per CU). GPU time against M43: the plain, CCLM, GLM, CCCM, CCCM-kinds, GLM-with-luma and
CFLM sets within -0.3 % .. +0.7 % (m31 136.3 ms per picture, m41 148.8).
M45: the rest of the CCP models a merge CU can inherit. LM chroma fusion CUs (the first type: a regular chroma mode)
have a record too: geneChromaFusionPred stores their CCCM models of MMLM_CHROMA_IDX as curCand (CCCM | MMLM, never
GL-CCCM: the CU has no glCccmFlag), so they are spatial candidates of their neighbours, though not in the CTU row's
history (CU::saveModelsInHCCP takes LM CUs only); their Cb block writes it (cccmStoreCcp). The record holds LBCCP's
inside filter twice (CCP_OUT_INSIDE: the CU's curCand after CU::saveCcInsideFilterFlagInCCP, the CU's flag;
CCP_OUT_INSIDE_HIST: the history's copy, which saveModelsInHCCP took before that: 0 for an LM CU); a merge CU will
write the inherited flag into both. Verification now reads the CPU's curCand after the picture (the PU of each record;
with the flag) and checks that every chroma PU without a record has an untyped curCand (else a merge CU next to it
would inherit a model the GPU lacks). Canaries: 4 (the fusion record not written, typed with one model, the inside
flag dropped, a fusion CU without a record). GPU time against M44: the same sets and m40 within -0.2 % .. +0.4 %.
M46: CCP merge (JVET-AD0188, with JVET-AE0097: no offset update), intra_ccp.glsl. A merge CU's blocks (idxNonLocalCCP,
dirMode LM_CHROMA_IDX, Tu::lm 0 until the GPU sets the chosen model's kind) build PU::getCCPModelCandidateList on the
GPU from the host's entry in Params::ccpData (at Tu::ccpMerge, a new field: Tu is 103 int32): the records of the PUs
at the 5 adjacent and 87 non-adjacent positions (getPUFromPos: decoded before and typed, i.e. recorded with a record;
a single-tree PU takes the luma sizes for the intervals and offsets, as the CPU does), each record once; the CTU row's
history (the records CU::saveModelsInHCCP pushed into cs.ccpLut, reset at each CTU row: LM-mode PUs, merge CUs
included, not a single tree's luma ISP CUs); idxNonLocalCCP - 1; 12 spare records per block for the defaults. A block
walks the history back from the latest model keeping each one unequal to those taken (addCCPToLut's last 6 distinct,
most recent first), takes the first CCLM model's slopes for the defaults, writes its 12 default records, builds the
list with CCPModelCandidate::operator=='s pruning (by type: CCLM / GLM the slopes and shifts, the others the luma
offset and parameters, with two models the threshold and the second model, several filters the index), costs each
candidate (xGetOneCCPCandCost: Cb's and Cr's SADs of the row above / column left where the block has CUs; CCLM / GLM
models by their first model, unclipped, as a Pel, from xGetLumaRecPixels' template luma; the CCCM kinds and GLM with
luma from intra_cccm.glsl's luma buffers of the CU's area built with the candidate's luma offset - the CPU's buffer
plus m_cccmLumaOffset minus the candidate's -, the candidate's corner offsets in GL-CCCM's and several filters'
position features, g_cccmLocDX / DY, its clipping ranges as an inherited model's, cccmClipRange), sorts stably, takes
the entry at idxNonLocalCCP - 1 and predicts like the LM block of its type (a CCLM model in s_cclm's slot, else
cccmPredict into s_tmp / the scratch; the inside filter flag of the list entry). The Cb block writes the CU's record
(the candidate's, its flag as the CU's and as the history's). Each block ranks the list itself: the Cr block first
waited for the Cb block's record, which with joint CbCr and sign prediction (the Cb block waits for the Cr block's
sign decision, resiFrom: a forward edge) closed a cycle - the first m46 decodes deadlocked and the driver reset the
GPU (TDR, three times) until an ECM_GPU_DRY dump showed the host's data sound and probes narrowed it to intra_deps. A
merge block waits for the blocks writing its sources' and its history's records (Params::ccpData's CCP_WRITERS per
record; intra_deps' ccpMergeDeps, the host's alike) and for what a CCCM block of any kind reads (cccmDeps).
Pipelines: LEVEL_CCP_MERGE (WITH_CCP_MERGE, constant_id 10; with CFLM). Rejected: merge fusion (ccpMergeFusionFlag),
merge adjustment (ccpMergeAdjustFlag), decoder-derived merge fusion (ddNonLocalCCPFusion), merge with LFNST / NSPT.
Code size: glslang -Os inlines every call site, and the merge code's first version overflowed SPIR-V's ID bound
(4 194 303; glslang then writes the unoptimized module and still exits 0): with one call site per large function and
the merge branch inside the LM branch (one emission loop, the block's own Tu changed in place - a copy through the
inside filter's inlined calls cost 90k IDs) intra_rec's bound is 3.72M (M45 3.52M), so the next features need the
same care (or spirv-opt with --compact-ids, which is not installed). Test set (encTest18: encTest17 with
ECM_TEST_NO_CCP_FUSION, the merge searches' fusion modes off; m41's arguments with --CCPMerge=1 --DDCCPFusion=0
--maxNumCcpMergeAdjustCand=0): m46 (20 to 800 merge blocks per picture, chosen candidates of every type),
m46u_four720 with 256 x 256 CTUs (--CTUSize=256), h46_crowd (1080p, 256 x 256 CTUs; 4160 and 4362 merge blocks).
ECM_GPU_DRY (verification): the records built and dumped (ECM_GPU_DUMP, with Params::ccpData) but not reconstructed on
the GPU. Canaries: 14; on the m46 CIF streams 8 caught (no spatial sources, the history oldest first, CCLM's offset
one more in the costs, the block's own luma offset, an unstable sort, the record without the inside flag, GL-CCCM by
the block's positions, no non-adjacent round 2), on the 720p streams 3 more (the history with the PU's inside flag,
CCLM's offsets compared in the pruning, the defaults without the CCLM candidate's slopes), on m46u_four720 (256 x 256
CTUs, where inherited models' clipping differs) the clipping ranges not inherited, on h46_crowd the history never
reset (elsewhere a CTU row's first merge CUs fill their lists from the row above); never caught: the defaults' slope
not raised to shift 3 (no CCLM candidate with a smaller shift). GPU time per picture: the plain and LM sets as
M45's (within noise), m46 145.03 ms (M45 sent its pictures to the CPU).
M47: CCP merge fusion with CCCM (JVET-AG0059's ccpMergeFusionFlag with ccpMergeFusionType 0; the flag only at
idxNonLocalCCP 1, JVET-AL0191). predCCPCandidate blends the merge prediction, after its inside filter, with the plain
CCCM models of MMLM_CHROMA_IDX (pu2: cccmFlag 1, no GL-CCCM, no filters; xCccmCalcRefAver's threshold, two models,
an empty template at the picture's corner gives clearModel's) - geneChromaFusionPred's CCCM exactly, without its
cccmMultiModeAvail gate. applyBlendFusion averages, (P0 + P1 + 1) >> 1, where the SPS's ccpFusionMode is 0 (the
encoder's with IntraPeriod 1: no blend model); with 1 (random access / low delay) the weights come from JVET-AN0168's
regression over the template predictions (rejected: "CCP merge fusion with a blend model"). On the GPU a merge fusion
block carries LM chroma fusion's first type's fields with the weight 2 ((2 P0 + 2 P1 + 2) >> 2 is the average):
intra_rec computes the CCCM prediction before its own, ccpMergePrediction keeps the fields (LM_FUSION_FIELDS) and
storeRegion blends. That CCCM prediction now sits at the end of the workgroup's scratch (cccmFusionPredBase, for LM
chroma fusion too): the merge prediction after it builds luma buffers and predictions from the scratch's start and
changes the globals cccmScratchPred reads (a canary with the old place differs on every picture). Its pass writes no
record for a merge block (the CU keeps the inherited model). Rejected: fusion with DIMD chroma (type 1: M48),
fusion with a blend model, merge adjustment. Test set (encTest19: encTest18 with ECM_TEST_CCP_FUSION_CCCM, the merge
searches' fusion modes none and CCCM; m46's arguments): m47_fore32 / 22 / 37, m47_park720, m47u_four720 (256 x 256
CTUs): 2 to 62 merge fusion blocks per picture. Canaries: 4; caught 3 (the weight 3 of 4, the fusion fields dropped
by the merge prediction, the CCCM prediction after the luma buffers: each 8 of 8); not caught: no scratch room for
the fusion prediction (a picture's scratch is its largest block's need, and these fusion blocks' candidates never fill
the room a merge block reserves). GPU time per picture: the plain, LM chroma fusion and m46 sets as M46's (within
noise), m47 158.52 ms (M46 sent its pictures to the CPU).
M48: CCP merge fusion with DIMD chroma (ccpMergeFusionType 1). DecCu derives the CU's DIMD chroma mode at the Cb block
as for a DIMD_CHROMA_IDX CU (deriveDimdChromaMode, then the co-located luma mode's check); predCCPCandidate predicts
both components by it (intraDir DIMD_CHROMA_IDX: initIntraPatternChType's initPredIntraParams takes the mode, no
stale parameters) and averages, (2 P0 + 2 P1 + 2) >> 2, after the merge prediction's inside filter - always, without
a blend model in any configuration (JVET-AN0168's regression is fusion with CCCM's only). On the GPU the CU gets a
CU_CHROMA_DIMD node (cuRecord), its blocks DERIVE_MODE (the node's mode and its parameters) and DERIVE_CCP_DIMD,
which intra_rec blends as DIMD / OBIC blocks: two predictions of weight 1 - the merge prediction (intra_ccp.glsl,
its inside filter), then the DIMD chroma mode's (applyModeParams; the LM branch skipped for it, the merge's LM fields
stay) - summed in the workgroup's blending sums, the angular arrays cleared between them, the sum's average rounded
(the finish's divTrunc of sum + 1 by 2). A first version with its own second pass of the prediction loop (a loop
bound of 2 for these blocks and the block's fields reset per pass) slowed every pipeline by about 1.2 % (the plain
and the DIMD sets too, whose specialization removes that code: the driver's code for the loop changed); reusing the
blend loop costs nothing measurable. No new call site (intra_rec's ID bound 3.72M). The comparison of derived
parameters skips these blocks (the CPU sets them only inside predCCPCandidate, after the record; the node's DIMD chroma
mode is compared). The host's bilateral filter base is luma's only, so the chroma prediction has none, as the CPU's.
Test set (encTest19 without its switch: the merge searches' fusion modes none, CCCM and DIMD chroma; m46's
arguments): m48_fore32 / 22 / 37, m48_park720, m48u_four720 (256 x 256 CTUs): 4 to 136 blocks of fusion with DIMD
chroma and 0 to 310 of fusion with CCCM per picture. Canaries: 4, all caught (the average without its rounding, the
second prediction by the LM branch, the CU without its DIMD chroma node, planar instead of the node's mode). GPU
time per picture: the plain, blending (m19, m20o, m24) and m47 sets as M47's (within noise), m48 162.09 ms (M47 sent
its pictures to the CPU).
M49: CCP merge adjustment (JVET-AL0126; ccpMergeAdjustFlag 1 or 2, then idxNonLocalCCP up to the SPS's
maxNumCcpMergeAdjustCand; never at the picture's corner, never with merge fusion). DecCu's list for it
(getCCPModelCandidateList with the flag) holds the CCCM models alone (types above CCP_TYPE_GLM4567), at most 15
(MAX_CCP_CAND_LIST_INIT_SIZE), no defaults; deriveCcpMergeAdjustCands appends, unpruned while there is room, up to 6
models fitted on the fly - plain CCCM of the block's own luma and luma offset over the template of the block's
position (above and left; in the picture's top row the left part, in its left column the part above): two models
split at xCccmCalcRefAver's threshold, one, one of the two template lines next to the block (trainingRange 2), one of
the reference samples alone (useRefSampOnly: the template's row above the block and column left of it take their own
sample for the neighbour that lies in the block), two of those, two of the two lines - each where a coefficient but
the bias is not 0 (its second model's fields of a single model stay the previous fit's: never compared). The list is
ranked as before, and adjustCCPCandidates moves the chosen model (flag 2: a pass of flag 1 first): per model and
component a delta of the co-located sample's coefficient (flag 1: {1, -1, 3, -3, 5, -5, 7, -7}) or of its neighbours'
(flag 2: {1, -1, 2, -2, 3, -3, 4, -4}; plain CCCM's 4, GL-CCCM's 2, without subsampling 3 to 5, several filters' 3 or
2), each coefficient moved away from 0 by delta / 32, chosen by xGetCostCCCM (the component's template SAD, with two
models over one model's samples): the first positive and negative deltas decide the sign (ties: the next two), the
rest of it only while not costlier than its first, ties take the later delta, a model of two without template
samples takes the set's last delta. On the GPU (intra_ccp.glsl) each block builds the list alike (s_ccp now 15 long),
fits the on-the-fly models into its spare records (ccpFlyModels: intra_cccm.glsl's accumulation and solver with the
merge pipelines' overrides g_ccpFlyTwoLines / g_ccpFlyRefOnly; the CPU's template selection by position is the whole
template there, which the block's own selection gives: a canary of it cannot differ), ranks, and searches its own
component's deltas in the CCCM parameter words (ccpMergeAdjust: every
invocation decides alike from wgSum2's totals); the CU's record follows: the Cb block writes the candidate's words and
its coefficients, the Cr block its own coefficients. The merge entry carries the flag (CCP_MERGE_ADJUST). Test set
(encTest19 without its switch, m48's arguments with the encoder's default of 6 adjustment candidates): m49_fore32 /
22 / 37, m49_park720, m49u_four720 (256 x 256 CTUs): 6 to 188 adjusted merge blocks per picture, besides merge fusion
of both types. intra_rec's ID bound 2.08M. Canaries: 10, all caught (on the CIF streams the list keeping CCLM / GLM
models, no two-line fits, no reference-sample fits, the delta not against the sign, ties keeping the earlier delta,
an empty model unmoved, flag 2 without its first pass, the Cr coefficients not written; on the 720p ones every
on-the-fly model valid and no early stop). GPU time per picture: the plain, CCCM and LM chroma fusion sets as M48's,
the merge sets about 0.3 % slower (the merge pipeline's larger code; checked in both orders: m46 144.6 -> 145.1, m48
160.5 -> 160.9 ms), m49 164.36 ms (M48 sent its pictures to the CPU). A follow-up could move the fits and the search
out of the merge pipeline that blocks without adjustment take (a pipeline level of their own).
M50: decoder-derived CCP fusion (JVET-AG0154; pu.decoderDerivedCcpMode, which CABACReader parses right after the LM
flag for chroma blocks of more than 16 samples, not at the picture's top-left: intraDir[1] LM_CHROMA_IDX, no other
LM syntax, so LM chroma is always on with it). DecCu derives at the Cb block (decoderDerivedCcp): the modes CCLM
(LM_CHROMA_IDX, xGetLMParametersLMS of both components), plain CCCM of one model fitted to the template's two lines
next to the block (trainingRange 2), the same with LBCCP's inside filter, two models over the whole template split at
the block's mean luma (xCccmCalcRefAver's trainingRange 2), the same with the filter, and GL-CCCM of one model where
the block has more than 32 samples (in I slices; 8 otherwise). A mode's cost is the SAD of both components'
predictions of the probe lines, the row above the block and the column left of it where a CU is there (CCLM's
clipped); the modes with the filter filter that line with the line two away from the block (where a CU is three
away, else the next line again) and the block's first line, padded with the reconstruction before the lines' start
where a CU is there (filterPredInsideProbeLine). The lowest cost (the first of equal ones) predicts alone where it is
at most (W + H) / 2; else each pair of modes (in order) blends its probe samples, (3 P + P' + 2) >> 2 with P the lower
cost's (the first of equal ones), and the pair of the lowest blend cost predicts where that cost is below the best
mode's: (3 P0 + P1 + 2) >> 2 of the two modes' predictions, each through its inside filter
(predDecoderDerivedIntraCCCMFusions). The CU's curCand is list[0]'s model with its LBCCP flags 0 (DecCu clears the
PU's flag before CU::saveCcInsideFilterFlagInCCP would copy it), its intraDir[1] list[0]'s mode, LM or MMLM. In
verification the host records the block after that, so record() gives the block the parsed LM_CHROMA_IDX (the first
version fitted its CCLM models as MMLM's where the CPU had chosen two models; found by printing both sides' costs). On
the GPU (intra_ccp.glsl) such a block is a CCP merge block of kind CCP_KIND_DERIVED (an entry without sources and
history, its spare records for its models; Tu::lm LM_ACTIVE without slope adjustments for the references and the
dependencies): it fits the CCLM models of both components into its first spare record (ccpDerivedLinear, before
cccmArea, whose words overlap CCLM's slots; the other component's references into s_ref[1] as cclmTransformMode's),
its CCCM models into the next three (ccpFitModels, adjustment's fits generalized), costs its modes as merge
candidates (the probe samples with the filter's taps: ccpProbeTap; kept in the scratch), chooses (ccpDerivedChoice:
the pairs' costs by shared atomics), predicts list[1] into the fusion slot (through cclmInsideFiltered of the stored
prediction where it has the filter), then list[0] as a merge block's candidate; storeRegion blends with LM chroma
fusion's weight 3. Pipelines: LEVEL_CCP_DERIVED (WITH_CCP_DERIVED, constant_id 11) above CCP merge's, and the probe
samples' branch apart from the merge cost loop: with the derivation in the merge pipeline's shared loop the merge
pictures were about 1.2 % slower than M49's, now about 0.9 % (m46 146.8 -> 148.2, m48 163.0 -> 164.4, m49 164.1 ->
165.4 ms; a reuse of the luma buffers between candidates made them slower still and was dropped); the plain, CCCM and
LM chroma fusion sets as M49's. m50 169.7 ms (M49 sent its pictures to the CPU). intra_rec's ID bound 2.14M. Test set
(encTest20 = encTest19 with ECM_TEST_NO_DD_MERGE_FUSION; m49's arguments with --DDCCPFusion=1): m50_fore32 / 22 / 37,
m50_park720, m50u_four720 (256 x 256 CTUs): 28 to 564 blocks of decoder-derived fusion per picture, besides merge
blocks of every kind. Canaries: 17; on the CIF streams 15 caught (the probes without the inside filter, the far line
never predicted, no reconstruction before the lines, half the block mean as threshold, the single model of all
lines, no GL-CCCM mode, the other CCLM model of the block's references, the pair blended in its order, fusion only
above W + H, fusion at equal costs, list[1] without its filter, weight 2, the record's flags list[0]'s, the CPU's
choice as the recorded mode, the inside flags in another order), on the 720p ones ties taking the later mode; never
caught: the CCLM probe samples unclipped (no CCLM model's template prediction leaves the sample range there).
M51: decoder-derived CCP merge fusion (JVET-AG0154's ddNonLocalCCPFusion: after the merge flag its own flag and a
unary index below getMaxNumCcpFusionModel - 8, or 12 with the SPS's ccpFusionMode -, no candidate index, no
adjustment, no merge fusion). reorderCCPCandidates keeps each candidate's template predictions of the merge costs (by
its place before the sort), then pairs the first 9 sorted candidates (in order) and ranks the pairs by cost into that
many slots (each before the first slot of a higher cost; the CPU's cost array starts as { MAX_INT } and zeros, which
ranks alike): useBlendModelDdccp is always true, so a pair's cost is the SAD of the average of the two template
predictions (xGetCostCCPFusion) or, where lower, of their blend by JVET-AN0168's model (xComputeBlendFusionModelJoint:
list[1]'s weight of 32 linear in the position, fitted by solve1 to both components' template samples - the features
(P1 - P0) x where a CU is above, (P1 - P0) y where one is left, with x and y one less along the line than the
sample's own, and P1 - P0, the target the reconstruction minus P0, all Pels; matrix scale 28 bits; its parameters
taken as ints unscaled, because the CPU measures them before setting them; shift 11; weights clipped to 1 .. 31;
equal weights where those at the block's corners differ by 4 at most). predDecoderDerivedCcpMergeFusion predicts the
slot's second candidate, then its first (each through its inside filter: predCCPCandidate), blends them
((32 - w) P0 + w P1 + 16) >> 5, or averages them; the CU's curCand is the first candidate's. On the GPU (intra_ccp.glsl)
such a block is a merge block of kind CCP_KIND_DD_FUSION (its entry's index the slot), in the derived pipelines: its
cost steps keep the candidates' template samples in the scratch (where decoder-derived fusion keeps its probe
samples: before list[1]'s prediction at the scratch's end, ccpProbeBase, now for both kinds), the sort carries the
candidates' places (CCP_LIST_ORIG), ccpDdFusionChoice fits each pair in an invocation of its own (ccpBlendModel: the
sums, the 3 x 3 elimination, the model and both costs) and ranks them, the fusion and final steps predict the
pair's candidates as decoder-derived fusion's list[1] and list[0] (a CCLM / GLM list[1] by cclmPredSample with its
pattern and models, through the inside filter where it has it), and storeRegion blends by the pair's model in
CCP_BLEND (cccmFusionBlend's weight 0, ccpFusionBlend). Params::ccpFusionPairs gives the slots. Test set (encTest20
without its switch, m50's arguments): m51_fore32 / 22 / 37, m51_park720, m51u_four720 (256 x 256 CTUs): 4 to 580
blocks of decoder-derived merge fusion per picture, besides decoder-derived fusion and merge blocks of every kind; all
identical at the first build. intra_rec's ID bound 2.15M. GPU time per picture: every earlier set as M50's (plain
12.51 -> 12.61, m34 137.14 -> 137.11, m40 147.05 -> 147.00, m46 145.95 -> 145.95, m48 162.06 -> 161.96, m49 162.96
-> 162.97, m50 167.16 -> 165.54 ms), m51 159.13 ms (M50 sent its pictures to the CPU). Canaries: 14, 13 caught on
the CIF streams (the model never or always taken, the fit's positions without their offset, no regularization, CCCM's
matrix scale, the corner test at 3, weights 0 .. 32, the blend's and the average's rounding, pairs of 8 candidates,
the samples by the sorted places, the pair's candidates swapped, ranking ties); never caught: the fit's features not
as Pels ((P1 - P0) x stays within 16 bits on these streams).
M52: CCP merge fusion with CCCM by a blend model (JVET-AN0168's useBlendModelCcpm, the SPS's ccpFusionMode, which
ECM's encoder sets unless the intra period is 1). predCCPCandidate predicts the two-model CCCM of MMLM_CHROMA_IDX as
M47's, then also its predictions of both components' template lines, the row above and the column left of the block
where a CU is there (ccpmComputeTemplateFusionModel), and fits M51's blend model to the chosen candidate's template
predictions (P0, kept by its place before the sort) and CCCM's (P1) (ccpmComputeBlendFusionModelJoint), always taken
(no comparison with the average; equal weights at a corner spread of 4 at most). On the GPU such a block is a merge
block of kind CCP_KIND_FUSION_BLEND in the derived pipelines: cccmPrediction's final loop also predicts the template
lines into the scratch slot after the candidates' (ccpFusionTemplateBase), the merge cost steps keep the candidates'
template samples as M51's, after the sort the first invocation fits the model (ccpBlendModel) into CCP_BLEND, and
storeRegion blends by it (the host's weight 0, as M51's). Test set (encTest21 with ECM_TEST_CCP_FUSION_MODE and
ECM_TEST_CCP_FUSION_CCCM, m51's arguments): m52_fore32 / 22 / 37, m52_park720, m52u_four720 (256 x 256 CTUs): 0 to 54
blocks of the blend per picture, besides decoder-derived merge fusion (12 slots with the mode) and merge blocks of
every kind. intra_rec's ID bound 2.16M. GPU time per picture: the plain, CCCM, LM chroma fusion and merge sets as
M51's (plain 12.48 -> 12.60 with M51 timed first, 12.61 -> 12.50 the other way round; m34 137.16 -> 137.29, m40 147.06
-> 147.18, m46 146.00 -> 145.78, m48 161.94 -> 161.90, m49 162.99 -> 162.86 ms), the sets of the derived pipelines
about 0.8 % slower in both orders (m50 165.65 -> 166.82, m51 159.23 -> 160.39 ms: cccmPrediction's loop also predicts
the template now; with the template in a loop of its own, a second cccmPredict, they were 1.8 % slower), m52 162.6 ms
(M51 sent its pictures to the CPU). Canaries: 7, all caught on the CIF streams (the CCCM template not kept, its column
at the block's, its components swapped, the model compared with the average, the candidate by its sorted place, the
CCCM prediction as P0, the host's weight 2).
M53: CCP merge and decoder-derived CCP fusion with LFNST / NSPT (JVET-AK0064: DecCu's deriveChromaIpmForTransform at
the Cb block of an LM mode with lfnstIdx - a merge or decoder-derived CU's uiChFinalMode is LM_CHROMA_IDX, parsed before
DecCu changes intraDir[1] - on the CU's final predictions, fusion and inside filter included: the strongest mode of the
3 x 3 gradients inside Cb's, Cr's or both into one histogram by the TU's joint CbCr mode). On the GPU
(intra_ccp.glsl's ccpMergeTransformMode, before the transforms) the block predicts each component the histogram reads
as that component's block does - the other one first (g_ccpOther: in 24 more spare records per merge CU, so the other
block's are left alone, and without the CU's record), then its own, which also leaves the references (a derived
block's CCLM fit) and merge fusion's CCCM prediction the block's again - and adds the gradients of the final samples
(ccpMergeFinalSample: cclmPrediction through the inside filter, blended with the fusion slot as storeRegion blends);
the histogram after the merge list's words in s_dimdHist (CCP_DIMD_HIST); the prediction section predicts the block
again. The host gives a merge block LM_HIST_* by the joint mode and DERIVE_CCLM_DIMD as an LM block's. The second
merge prediction adds ~1M SPIR-V IDs (intra_rec 2.16M -> 3.19M of the 4.19M a module may have; behind a constant
false the optimizer removes the call, but the bound keeps the IDs of the inlined function), so these pictures'
pipelines (LEVEL_CCP_LFNST) come from intra_rec's second module: GpuLib's CMakeLists.txt compiles intra_rec.comp
again with CCP_LFNST_MODULE defined (spv_header(), spv_intra_rec_lfnst). The first module is M52's code with its IDs
renumbered (the same instructions, the differing words a one-to-one renaming: a word-by-word comparison script), so
every other set runs as with M52 (m50 166.88 -> 166.93 ms). Rejected: merge fusion with DIMD chroma with LFNST /
NSPT (the other component's angular prediction). Test set (encTest21 with ECM_TEST_CCP_FUSION_MODE alone - the merge
searches' fusion modes none, CCCM and DIMD chroma -, m52's arguments with `--IntraLFNSTISlice=1 --NSPT=1
--LFNSTExt=1`): m53_fore32 / 22 / 37, m53_park720, m53u_four720 (256 x 256 CTUs): 7 of 12 pictures on the GPU (the
others have merge fusion with DIMD chroma with LFNST), 0 to 68 merge blocks with LFNST / NSPT per picture, besides LM
and decoder-derived blocks with them; all identical at the first build. GPU time per picture: m53_fore32 78.5,
m53_fore37 72.8 ms, m53_park720's picture on the GPU 287.5 ms (M52 sent them to the CPU). A cold pipeline (a new
build: the driver's cache misses) is slow to create: m53_fore37's first decode took 242 s, m52_fore37's (a derived
pipeline of the first module) 154 s, both under a second once cached. Canaries: 10; 9 caught in
verification on m53_fore32 / 37 / park720 (the other component's histogram skipped, the fusion not blended in,
without the inside filter, a derived block without each component's references, the other component's fusion with
the block's own CCCM prediction, the block's CCCM fusion not restored, the gradients at the block's edge, ties taking
the later mode, both histograms for every joint CbCr mode), the tenth (merge blocks without DERIVE_CCLM_DIMD) in ON
mode only: verification records a block after the CPU predicted it, so the host's transform choice reads the CPU's
mode.
M54: CCP merge fusion with DIMD chroma with LFNST / NSPT: ccpMergeTransformMode also predicts the CU's DIMD chroma
mode for each component it reads (ccpDimdChromaPrediction: that component's reference line, intra_rec's chroma
planar / DC / angular code with PDPC, into the fusion slot the host now sizes for these blocks) and averages it into
the final samples ((P0 + P1 + 1) >> 1). Second module only. m53: all 12 pictures on the GPU, identical (ON mode too),
2 to 4 such blocks in m53_fore22 / park720 / four720 pictures. Canaries: 7, 5 caught (the DIMD prediction not
averaged in, the average unrounded, the other component from the block's line, horizontal modes untransposed, two
taps rounded down); PDPC off changed no picture (a probe zeroing the angular predictions with PDPC did: the blocks
have them, but PDPC never moved the histogram's maximum); no m53 block has a planar / DC DIMD chroma mode (a probe
zeroing them changed nothing), so that branch is untested.
Partial verification (2026-10-15; `ECM_GPU_PARTIAL=1` with `--GpuRecon=2`, `perf/gpu/partial.py <build>` over the
regular streams in C:\Code\ECM\streams): the regular streams verify with the tools ported so far instead of falling back
whole. supported() leaves out the tool families it checks per CU; a CU with a block the GPU cannot reconstruct is taken
from the CPU (record()'s rejections, DecCu's inter / IBC / palette CUs: cpuCu): its blocks become nodes that own their
areas (DERIVE_CPU_FILL with DERIVE_NODE: intra_deps' owner maps take them, so the blocks after them read them as
decoded), the planes start as the CPU's reconstruction there and 0 in the GPU's blocks, it keeps a CU_KNOWN record of
its mode as chroma CUs read a co-located one (an EIP CU's DIMD mode, TIMD / TMRL mapped to 67 modes), and its CCP models
(an intra PU's curCand, a non-intra CU's TUs', JVET-AF0073) get records and their place in the CTU row's history, written
from the CPU's models (cpuCcpWords). A CU whose mode derivation the GPU lacks takes the CPU's mode: chroma DM /
candidates with chroma reordering (until M57 also CU_MPM where MPM sorting, MDIP or PDP change the MPM list, and TIMD /
TMRL CUs were taken from the CPU then); TIMD CUs with block-vector candidates (JVET-AM0138, M57) and, until M55, PDP
candidate blocks (pdpCandidate: predIntraAng's conditions but the references') are taken from the CPU. Reports: the
first block in decoding order whose prediction or residual differs and the first CCP record that differs, beside the
first check's, and per run the CUs taken from the CPU by reason. First results (2026-10-15): every regular stream's
intra pictures identical but 3 of
fp720_ai's 8 and ra720's first: a CCP merge block (ra720's at 208,144, chroma 16x16) whose GPU choice is GL-CCCM where the
CPU takes two-model CCCM (a candidate the GPU lacks or ranks otherwise, not found yet). CUs from the CPU over all
streams: PDP ~119000, inter / IBC / palette ~33000 (most in inter pictures), TIMD / TMRL behind MPM sorting / MDIP /
PDP ~22000, IntraTMP ~9300, SGPM ~5600, EIP ~1700, MTSS ~280, DBV ~200, BVG-CCCM ~70: the order to port them in.
Development speed (2026-10-15, at the user's request): the regular streams in partial verification are the test set of
each tool family (no more test sets encoded per tool); canaries only for code no stream reaches, a full regression once
a day or before a push, timing at the end of a phase. A cold intra_rec pipeline takes 30 s (plain) to 220 s (CCP LFNST)
to create; VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT does not shorten it, SPIR-V without glslang's optimizer doubles it.
A decoding process creates the pipelines it needs one after the other, and parallel test processes each compile the same
ones: after a shader change `perf/gpu/precompile.py <build>` first compiles all 63 (IntraRecVariant x LmLevel), one
process per pipeline (ECM_GPU_PRECOMPILE=variant:level) on every CPU thread, into runs/nvcache - the cache partial.py
uses; point the regressions there too (__GL_SHADER_DISK_CACHE_PATH, _SIZE 8000000000, _SKIP_CLEANUP 1). The M57
regression without it spent ~25 min compiling. (Since M65 also into pipeline cache files, runs/pipecache, which keep
them all: see the note after M65.) partial.py reports a decoder that fails (its exit code and error output, also in its
log); a stream whose GPU device is lost (VkResult -4) runs again alone after the others: pj720_ai's pictures take up
to ~1.8 s of GPU time in partial verification, and with 8 decoders sharing the GPU some pass Windows' 2 s GPU timeout
(TDR), which resets the device (seen from M66 on; the same build verifies it alone).
M55: PDP (JVET-AH0209, intra_pdp.glsl): a luma block of a size with PDP filters whose references are available is
predicted by its mode's filter per sample over two reference lines (one for 16x32 .. 32x32, whose filters cover every
second sample and whose samples between are averaged in place, in the CPU's order); also a MIP block's by MIP's filter
sets (JVET-AI0208; its LFNST / NSPT mode then a DIMD of that prediction), DIMD's, OBIC's and intra merge's main mode and
DIMD's / intra merge's blend modes (not the blend's planar, nor OBIC's other modes). The CPU's references' availability
and padding (xFillReferenceSamples2) now come in parameters-only mode too (initIntraPatternChType; the snapshot's
pdpRefAvailable / pdpPadW / pdpPadH); the filters (createPdpFilters' and createMipFilters' layouts, ~6M int16) are in
the matrices buffer, a table per mode / MIP set and size gives their offsets; intra_rec decides per mode (a derived
mode's too). The prediction goes through s_spPred (a blend keeps its filtered references in s_tmp). Partial
verification found an OBIC CU counting a CPU-filled neighbour as a plain CU (deriveObicMode counts TIMD, SGPM, NN ...
neighbours otherwise): such an OBIC CU is taken from the CPU. Results (grP7): every regular stream identical but the
known CCP merge blocks (fp720_ai's 3 pictures, ra720's first); PDP left the CPU-fill reasons entirely (ai720: 4340 ->
~3990 CUs from the CPU; NN intra prediction 1626 is now the largest).
M56: NN intra prediction (JVET-AJ0249, intra_nn.glsl, GpuPnn.cpp): the six SADL models (MatMul layers of int16 weights,
int32 sums shifted and saturated to +-32767, leaky ReLUs clipping to int16) parsed from the model files into packs of 8
weights per output column (the sparse layers' packs, the dense layers cut into packs; int32 wraps, so neither order
nor zeros matter); per block the context (extractContextPortions: lines above with the corner and left, arms of twice
the sides as far as available, downsampled for 32x32 and up, transposed for tall blocks, less its rounded mean, masks
0 before the zero-mean shift), the layers, the output plus the mean, upsample2d from the reference column / row
(closed form per sample), the bilateral filter as for DC. IntraPredictionNN::gpuBlock gives the geometry (the CPU's
own context availability, now a static function). Transforms that depend on the mode (LFNST / NSPT, explicit MTS):
the prediction (filtered) is kept in s_spPred as sign prediction keeps it, mipTransformMode's DIMD derives
indicesRepresentationPnn from it, then the transforms (LFNST / NSPT by getFinalIntraModeForTransform's lfnstSecFlag
rule), the residual and the reconstruction (DERIVE_PNN_DIMD). An NN CU's record holds planar as its mode (what the MPM
lists and chroma TMRL read; chroma DM reads indicesRepresentationPnn, still the CPU's in partial verification). Not yet:
NN CUs above 32 x 32 or with MTSS. Bug found on the way: a workgroup sum in a slot the bilateral filter takes zero.
partial.py now creates the driver cache's folder (runs/nvcache did not exist: nothing was cached, every run compiled
every pipeline). The NN's layers are slow so far (ai720: GPU 122 -> 495 ms; each column's packs read serially).
M57: CTC's MPM lists (intra_mpm.glsl): MPM sorting (JVET-AD0085), MDIP (JVET-AK0059) and PDP's list (JVET-AK0061) with
JVET-AL0125's improvements. A regular CU whose lists they shape (an SPS with any of them; not where the syntax selects
planar, the list's first entry) and an MDIP CU get a record (Cu::mpm, MpmHead in timdData: flags, the template of the
sortings - TIMD's kind with arms of one sample -, PDP's size and the padding of the CU shifted up / left
(IntraPrediction::pdpTemplatePads), MDIP's histograms (mdipHistogramCalls)); their node derives the mode as DecCu
does: deriveDimdMode's excluded modes where MDIP is not allowed, deriveMdipMode (histograms of lines further out, its
list, deriveMdipSorted's template SATD, buildExcludingMode), getSpatialIpm with the DIMD modes first and PDP's modes
(getPDPPredMode), deriveMPMSorted (the template costs of the first entries - TIMD's predictions, or for a mode with PDP
filters PDP's of the CU shifted up / left, from xFillPDPTempReferenceSamples2's vectors; the cheapest iBestN, the
earlier of equal costs first), fillMPMList / fillNonMPMList with MDIP's mode and the excluded modes left out and PDP's
modes first. Pictures with records take the WITH_MPM pipelines (VARIANT_MPM, above TMRL). Verification compares each
record's mdipMode and excludingMode with the CPU's; the mode shows in the blocks. TIMD and TMRL CUs leave the CPU on
the regular streams too (their lists are plain: TIMD's getIntraMPMs runs with isModeExcluded off, unsorted). Found on
the way: a TIMD record took its index before its last rejections ("a TIMD merge neighbour that is not a recorded TIMD
CU"), so every later TIMD CU's CPU fields were compared with another CU's (only partial verification records past a
rejection); deriveTimdMode tries the block vectors of DecCu's list (getItmpMergeCandidate, getArBvMergeCandidate:
IntraTMP / IBC CUs and those that took a vector around it) beside planar and DC, and one costing least there for a
while zeroes the location dependences even where none is chosen: a TIMD CU with any such candidate is the CPU's.
Results (grP19, grP20): every regular stream identical but the known CCP merge blocks; per 720p intra picture ~490
records (~180 sorted, ~480 with MDIP). Not reached (partial verification takes intra slices only): the sorting of
inter slices (from two entries, CUs up to 1024 samples) and with it PDP's template predictions of 16x32 .. 32x32
(filters for every second sample, their averages from the long vector).
M58: EIP (intra_eip.glsl; JVET-AG0058 with JVET-AH0086's bias and clipping, JVET-AI0066's regularization by
JVET-AL0106's tables, JVET-AJ0082's two models, JVET-AL0205's filter shapes). An EIP CU's record (Cu::eip, EIP_HEAD_*:
its EIP index, fitted or merge, its mode's shape / regions / models, its reference area - IntraPrediction::
eipReferenceArea, initEipParams' geometry - and its luma block; a merge CU's sources, each as CU record and EIP index:
the EIP CUs at getNeiEipCands' positions - eipMergeNeighbours - and its CTU row's EIP CUs before it, the history).
Its node (CU_EIP) fits the model as getCurEipCands / solveEip do: the area's clipping range and, with two models, the
threshold (the mean of the area's inner half); the 135 sums of each model over its regions' samples (setInputsVec,
each region's taps clamped to its own end; two models split by the chooser), one invocation per sum; the L2
regularization; CCCM's elimination with 15 parameters (rows by invocations). An EIP merge CU's node takes the
neighbours' models, then the history's 6 most recent distinct ones (addEipToLut's LUT, rebuilt backwards), each
unless the list has the same model, up to 12, ranked by the SATD of the template row / column each predicts. Its luma
block (DERIVE_EIP) predicts the block's diagonals one after the other (each sample's 14 inputs lie on earlier
diagonals), then the DIMD of its prediction (mipTransformMode: eipDimdMode, eipDimdMode2nd) - written as the CU's mode,
the one chroma DM reads - and the transforms where they depend on it (DERIVE_PNN_DIMD). The MPM lists read an EIP CU
as planar (getIntraDirLuma: a CU_KNOWN record of planar, mpmPlanarCu), as do chroma TMRL's five luma positions; OBIC
counts nothing of it in an intra slice. Verified per EIP CU: its model (parameters, shape, two models, threshold) and
DIMD modes against the CPU's; EIP CUs from the CPU upload theirs (for merge and the history). Found on the way: OBIC
counts an NN neighbour's indicesRepresentationPnn, which the GPU's NN blocks do not publish (no stream had met it: an
EIP neighbour sent those OBIC CUs to the CPU before) - such OBIC CUs stay on the CPU now. Not yet: EIP with a block
vector (IntraTMP's search), EIP in inter slices (temporal candidates). Results (grE4): every regular stream identical
but the known CCP merge blocks; per 720p intra picture ~196 EIP CUs on the GPU (~63 merge, ~88 with two models).
M59-M61: block vectors (JVET-AM0138 and the tools before it; intra_bv.glsl). Every luma CU has a record of its block
vectors as later CUs' lists take them (Params::bvOut, BV_OUT per CU by its CodingStructure index, getNeighBv's order:
IBC, IntraTMP / BVG-DIMD / EIP with a vector, a DIMD CU's bvDimd, a TIMD CU's timdBv, SGPM's), written by the GPU's
nodes or uploaded for the CPU's CUs (cpuBlockVectors), and a map of the luma units to CUs (Params::bvMap). A node
whose CU takes vectors from the CUs around it (Cu::bvList: its list record, BV_LIST_*) builds getItmpMergeCandidate's
list (the CUs at its 13 neighbour and 18 non-adjacent positions, bvNeighbourCus) and, for TIMD, getArBvMergeCandidate's
cascade (per vector the CUs at the CU's centre and corners moved by it) - only where those CUs may have vectors
(bvPotentials: from the syntax, cascaded through DIMD / TIMD / SGPM CUs), else the list is empty and there is no
record. Such a node reads at vectors that may point anywhere decoded before, so it waits for the scheduler's prefix of
finished blocks (DERIVE_PREFIX: intra_rec.comp keeps a done flag per block in depTmp, which intra_deps no longer needs,
and a counter of the finished prefix moved on by compare-and-swap; whoever moves it to a waiting node releases it;
intra_deps counts the wait as one more dependency). TIMD (intra_timd.glsl): planar, DC and the list's vectors (modes
from TIMD_BV_IDX, their templates copied from the reconstruction) costed and scanned in batches as the CPU's first loop
(a vector best zeroes the location dependences; deriveTimdModeSad passes over one whose template leaves the picture),
into TimdOut (isBvTimd, timdBv; the fields' modes planar) and the CU's record; its luma block predicts those parts as
copies (g_bvSubs, bvOfSub). DIMD, OBIC and intra merge CUs (dimdVectorChoice: getBestNonAnglularMode, both arms of
TIMD's template, the list kept to 30 / 15, no cascade): planar's SATD against each vector's, the first of the least cost
replaces planar's predictions (a DIMD or merge blend's planar, an OBIC blend's planar blend modes). With a vector the
transforms of a TIMD, DIMD or OBIC block take the DIMD of its prediction (getTimdVimp / getDimdVimp: MTS and LFNST /
NSPT; g_bvTransform defers the reconstruction as EIP's does; a TIMD CU's block writes its timdmTrType). M59: the MPM
lists read an SGPM CU (all on the CPU) as getSpatialIpm does, its partition's angle or with regression SGPM the DIMD
of its prediction (mpmModeCu records of one mode each), an IBC / inter neighbour as its motion field's mode, a TIMD /
TMRL neighbour from the CPU of a TMRL CU's list in the 131-mode set (CU_KNOWN_EXT). Every DIMD / OBIC / merge node
writes its choice, planar where it has no list: its blocks read the CU's DimdOut, which otherwise holds an earlier
picture's (the first run mispredicted from the second picture on). Stay on the CPU: TIMD merge CUs with a neighbour
predicting from a vector, TIMD / DIMD with a vector and MTSS, lists beyond the GPU's room (80), intra merge whose
candidates take vectors (reorderIntraMergeCand's dimdBv), intra merge with an IntraTMP neighbour
(getItmpBasedIntraMergeCand: the parameters at its vector - found on the way: the GPU's record had left it out; such
CUs had all been on the CPU for other reasons). Results (grB6): every regular stream identical but the known CCP merge
blocks; the intra CUs from the CPU over the 21 streams 40621 (M58) -> 29378 (M59 / M60) -> 24816. Partial
verification now reports the CU first in decoding order among the per-CU checks' mismatches.
M62: IntraTMP (intra_tmp.glsl; searchCandidateFromOnePicIntra with JVET-AM0138's lists), stage 1: no LIC, fusion,
FLM or sub-sample vector (tmpFracIdx 0), a template (else DC), up to 1024 samples (s_spPred), no OBMC. The CU's node
(Cu::tmp: TmpHead, TMP_HEAD_*: template type, index, sampling, LIC mode, needTopLeft, the refined list's length, the
CTU above 128, the six regions' bounds from clipMvIntraConstraint, which sortInOut repeats) waits for every block
before it (DERIVE_PREFIX; its list record is made even without vectors around, as it records its own), builds its
neighbours' list (bvNeighbourList, tmpSortInOut, bvCascade), runs the sparse search (regions 3, 4, 5, 0, 1, 2; rows by
TMP_SAMPLING from the bottom, each row's start staggered; regions 4 / 5 from the first decoded block of a row; a small
block skips a region inside a neighbour vector's window), cascades from the sparse candidates (bvCascadeStep with the
sparse records), clusters the list's vectors into the sparse lists (adaptive windows by cost for the neighbours'
vectors; the >128 CTU and the other window rules), refines every candidate's window, merges the top / left lists into
the L-shape's from INIT_TL_POS. Costs (calcTemplateDiff's exact sums: weighted cost, top part, left part) are computed
by all invocations, chunk by chunk of TMP_CHUNK positions in the CPU's order (the workgroup's scratch, TS_*), and
invocation 0 replays updateCandList and the thresholds from them: every test is "cost < threshold", which the CPU's
early ends cannot change. The list goes into the CU's DimdOut (DIMD_OUT_TMP_NUM, DIMD_OUT_TMP_BV: tmpXYdisp, padded
cyclically), the vector of tmpIdx and the list's first into its record of vectors (getNeighBv). The luma block
(DERIVE_TMP, waiting for the node) copies the reconstruction at its vector (DIMD_OUT_BV) into s_spPred and, as an EIP
block, takes the DIMD of the prediction (mipTransformMode: intraTmpDimdMode, dimdDerivedIntraDir2nd, MTSS's choice)
for its transforms.
The MPM lists and chroma TMRL read an IntraTMP neighbour as planar from a record of their own (mpmModeCu), so they do
not wait for its search. SPIR-V: ccpMergeTransformMode's gradients read the final chroma samples computed once into
the workgroup's blending sums (the CCP-LFNST module was at 4.09M of the 4.19M IDs; now ~350k fewer); IntraTMP adds
~141k: the modules are at 3.06M / 3.88M. Results (grT9): every regular stream identical but the known CCP merge
blocks; 208 IntraTMP CUs on the GPU, the intra CUs from the CPU 24816 -> 24608; the rest of IntraTMP: LIC 4159 (with
fusion 813), sub-sample vectors 1900, OBMC 919, fusion 731, FLM 589. The MPM variant's pipelines compile in 244 / 271 /
~290 / 351 s. Cu::tmp must start at -1 wherever a Cu is made (cuRecord's initializer missed it at first: every CU then
looked like it had a search record, and every picture took the MPM pipelines).
M63: EIP with a block vector (getBvEipCands, JVET-AL0106). Its node runs IntraTMP's sparse search alone (tmpNode with
TMP_HEAD_ONLY_BEST, the record made by tmpSearchRecord: mtmpNumSparse 1, after the neighbours' list as an IntraTMP
CU's; its list record and prefix wait as one's), puts the best vector (m_tmpSparseSearchXdisp) into its DimdOut's
DIMD_OUT_BV and its record of vectors (pu.bv), then eipNode (EIP_BV) fits the square shape's single model to the block
at the vector with the BV_EIP_TEMP_SIZE lines above / left of it where the picture has them (its samples from (3, 3)
on; EIP_REGION_BV), with the clipping range and luma offset of the CU's own reference area. eipEntrySum reads through
an origin, so the sums and the solver keep one call site each. Results (grU3): every regular stream identical but the
known CCP merge blocks; all 1263 EIP CUs with a vector on the GPU, the intra CUs from the CPU 24608 -> 23345.
M64: intra merge with block vectors (isBvDimdExtAvail, JVET-AM0138). An intra merge CU whose neighbours may have
vectors (its list record) makes its choice of planar or a vector (dimdVectorChoice) before it ranks its candidates, as
DecCu orders them; that pass also leaves the vectors' template costs and location dependences in the workgroup's
scratch, cheapest first (up to MERGE_BVS). mergeNode then lets each vector replace the last part of the chosen
candidate - its main mode, planar, its blend modes - of nonzero weight not replaced yet whose mode's template cost is
above the vector's; the main mode's or a blend mode's location dependence becomes the vector's. The replaced parts'
bits and vectors go into the CU's DimdOut (DIMD_OUT_EXT_FLAGS, DIMD_OUT_EXT_BV); the merge block copies the
reconstruction at those vectors for those parts (planar's not with ISP) and then never takes planar's from bvDimd.
A TIMD merge CU in dual tree takes its candidates' transform types; a candidate that predicts from a vector has them
set by its luma block (by the DIMD of its prediction), so the merge node waits for each candidate's luma block too
(TimdHead::block, TIMD_HEAD_BLOCK, in the host's dependencies and intra_deps'). That holds for every candidate: whether
one predicts from a vector is known only once its node has run, and a condition on its list record (Cu::bvList, set
at the picture's end) made the host's dependencies differ from intra_deps' ("scheduling data differ", grV2). SPIR-V:
the modules are at 3.23M / 4.07M IDs (the CCP-LFNST module ~125k under the limit). Results (grV6, verified with M65 on
top): every regular stream identical but the known CCP merge blocks; the intra merge CUs with vectors (876) and the
TIMD merge CUs with a vector-predicting neighbour (304) are no longer taken from the CPU for those reasons.
M65: BVG-DIMD (block-vector-guided DIMD, JVET-AL0108), its CUs and their use as OBIC and intra merge neighbours. A
BVG-DIMD CU (CU_BVG) records an IntraTMP search (tmpSearchRecord with TMP_SEARCH_BVG: mtmpNumSparse 1, no top-left
lists, a refined list of BVG_DIMD_BV_LIST_SIZE) and a list record of its neighbours' vectors, as an IntraTMP CU does.
Its node runs tmpNode - now at one call site for IntraTMP, EIP with a vector and BVG-DIMD, which took ~90k / ~110k IDs
off the two modules (3.14M / 3.96M) - whose BVG path keeps the sparse search's best, lets each neighbour vector (no
cascade from the sparse candidates) cheaper than it replace it (bvgCandList of MAX_BVG_ROUGH_NUM 1, a vector's window
5 around it), and refines that one window with the threshold 1.2 times its cost plus 1 (the CPU's double products are
exactly 6 / 5 in integers; validItmpBv with isAvailablePairFound for a vector's window that leaves the decoded area)
and the list's threshold the lesser of 1.2 times its cheapest plus 1 and its last. intra_bvg.glsl's bvgNode then cuts
the list at 1.2 times the cheapest (m_bvListSorted), weighs the vectors (convertDiff2Weight, its 32-bit products as
the CPU's ints), merges the blocks at overlapping vectors into areas while the bounding box stays within 2 blocks plus
one per vector taken and its last sample is decoded, takes the DIMD histogram of their insides (MTSS's subsampling
from 128 samples on), and the two strongest modes with their weights (deriveBvgDimdMode). The node writes the blend in
DIMD's fields of its DimdOut (deriveIntraMergeCand's conversion: the first mode, planar's weight, the second mode as
the first blend mode; bvgDimdMode[1] as the second for MTSS), the vectors and weights in DIMD_OUT_BVG_* and its first
vector into its record (pu.bv, also (0, 0) where it found none, as the CPU's). Its luma block (DERIVE_DIMD with
DERIVE_BVG) blends as a DIMD block whose planar prediction is the vectors' fusion (bvgFusionSample:
tmpFusionBlending), the blend modes without PDP and with the longest taps, no fusion and no bilateral filter
(predIntraAng returns before it; no OBMC either: of these only IntraTMP's prediction adds it, and every intra CU of 32
samples or more has obmcFlag set by parsing). OBIC counts a BVG-DIMD neighbour's modes with all its samples and no
class (as an intra merge neighbour's), intra merge takes one as an intra merge neighbour's parameters
(MERGE_NBR_MERGE). Results (grV6): every regular stream identical but the known CCP merge blocks; all 3486 BVG-DIMD
CUs and the 644 intra merge CUs held back by a BVG-DIMD neighbour leave that reason; with M64 the intra CUs from the
CPU go 23345 -> 19093. Remaining: SGPM 5598, IntraTMP LIC 4159 (with fusion 813), intra merge with a neighbour of a
kind the GPU lacks 2835 (an SGPM neighbour is one: the CPU takes SGPM CUs, which the merge record reads as MPM CUs),
sub-sample vectors 1900, IntraTMP's OBMC 919, an IntraTMP neighbour of intra merge 855, fusion 731, FLM 589.
Pipeline cache files (2026-10-15): the NVIDIA disk cache (runs/nvcache) did not keep all 63 intra_rec pipelines (its
files hold ~100-350 MB each and were rewritten as new pipelines came in): the M64 / M65 partial run after
precompile.py still spent ~10 minutes per decoder compiling evicted ones. VkCompute's Kernel::create now takes
ECM_GPU_PIPELINE_CACHE=<dir>: each pipeline's VkPipelineCache data in a file of its own (named by its kernel and a hash
of its SPIR-V, specialization and the device's pipelineCacheUUID), written where it was compiled, loaded later (the
MPM pipeline: 122 s cold, 0.0 s from its file of ~4 MB). precompile.py and partial.py set it to runs/pipecache;
precompile.py trims that to 30 GB, oldest first. Point the regressions there too.
M66: SGPM (spatial GPM, JVET-AB0155 with JVET-AJ0107's 26 partitions), its CUs without vector candidates or regression
SGPM, up to 1024 samples. An SGPM CU (CU_SGPM) records its DIMD, its MPM neighbours and a TIMD record (deriveTimdMode
with bHorVer alone reads the MPM list), the five CUs getSgpmIntraMPMs reads with their kinds (an intra CU's mode as
getIntraDirLuma gives it, an inter or IBC CU's motion-field mode), its list index (less the 4 regression entries where
the CU allows them), its size's partition weights and its one-sample template (addSgpmRecord; sgpmTables in the
transform tables, TR_TAB_SGPM). Its node runs timdNode's first batch only (the MPM list's angular modes, HOR, VER; no
wide angles beyond) and takes timdHor / timdVer from the arm costs (sgpmTimdModes), then intra_sgpm.glsl's sgpmNode:
the 52 parts' lists, every named mode's template differences (predTimdIntraAng with SGPM's wide angles: getWideAngleExt
with bSgpm, one step further), each mode's SAD and each list entry's SAD weighted by its partition (sgpmSadTM), the 234
pairs' costs in integers (the CPU's doubles are exact) and the pair of the CU's rank (ascending, the CPU's order among
equal costs). It writes the partition and modes into DimdOut (DIMD_OUT_SGPM_*), the first mode as the CU's and the
partition's angle into a record of its own for the MPM lists after it (CU_SGPM_MPM: getSpatialIpm reads the angle).
Its luma block (DERIVE_SGPM) blends both modes' predictions in intra_rec's blending, weighted per sample by the
partition (emitPred, g_sgpmWeights; (sum + 16) >> 5), keeps the blend for a DIMD of it (DIMD_OUT_SGPM_PRED) and takes
explicit MTS by the partition's angle, LFNST / NSPT by getFinalIntraModeForTransform's SGPM rule (JVET-AK0222). OBIC
counts an SGPM neighbour's two modes, intra merge reads one's DIMD. The dependencies add the CUs its lists read and
its template, whose reference line TIMD's thicker template does not cover. SPIR-V: 3.19M / 4.01M IDs (the CCP-LFNST
module ~180k under the limit). Results (grW1): every regular stream identical but the known CCP merge blocks; 1643 SGPM
CUs on the GPU, the intra CUs from the CPU 19093 -> 17265; the full regression (444 streams, 1565 pictures verified,
8 skipped) clean with M64-M66. Remaining SGPM: with vector candidates 2056 (the list's
m_sgpmMvBasedMergeCandidates), with a block vector 1096, regression 795. A full precompile of the 63 pipelines takes
~26 min after a shader change (14 at a time, 225-480 s each).
M67 (2026-10-16): SGPM's block vectors (JVET-AG0152 with AH0200's reordering) and their LIC models (JVET-AM0157:
BV_OUT_LIC models of the CPU's IBC / IntraTMP CUs, sgpmLic on the template copies and the block's copy,
DIMD_OUT_SGPM_LIC). The integer-vector part was verified on the RTX 3070 (build grW4: every regular stream identical
but the known CCP merge blocks; SGPM CUs on the GPU 1643 -> 2494, intra CUs from the CPU 17265 -> 16349); the version
with LIC on an Intel iGPU with a corpus of 48 streams of the CTC tools (C:\Code\corpus there, copied to
C:\Code\ECM\streams), where every stream gave M66's results (the differences it showed were older bugs, fixed after
M67). Remaining: "SGPM with a fractional vector neighbour" (~830) and "SGPM with an EIP neighbour with a vector"
(~450) need IBC's interpolation (getPredIBCBlk: bilinear for the template, the regular luma filter for a block at a
fractional vector), which IntraTMP's sub-sample CUs (1900) need too (M68, M69). (The EIP rule was not needed: M68.)
Of the 3152 SGPM CUs with vectors in their lists, 1693 had a fractional one (EIP with a vector gives its integer
vector as 1/16 samples to SGPM's list, IntraTMP's sub-sample CUs their fractional mv), 178 one with LIC (after
DecCu's cap; ~1000 lists hold a LIC neighbour's valid vector before it), none a flipped one.
The GPU takes the others: an SGPM CU has a list record (Cu::bvList, keeping DecCu's 30 / 15, a prefix wait) and its
node builds getItmpMergeCandidate's list for SGPM from the records of the CUs around it (Params::bvOut: getNeighBv's
SGPM vectors are those in 1/16 samples for every CU kind but the ones the host rejects - an EIP neighbour with a
vector, an IBC / IntraTMP neighbour with LIC, flips or a fractional vector, an inter IBC neighbour; verification
checks the CPU's list against that), valid for its one-sample template (validIBCItmpMv), costs each vector's template
copy (sadTM), takes the cheapest 6 (stableSortSmall) as modes 67..72 behind each part's three, and ranks the pairs over
up to 9 x 9 per partition (the costs and pairs in the workgroup's scratch). It writes the chosen modes' vectors
(DIMD_OUT_SGPM_BV) and into its vector record (for the lists after it); its CU's mode is planar where its first mode
is a vector's. Its block copies the reconstruction at a vector for that part (g_bvSubs). TIMD's pass of an SGPM CU
takes no vectors (deriveTimdMode's useBv), OBIC counts no vector's mode. SPIR-V: 3.21M / 4.04M IDs (~155k left in
the CCP-LFNST module).
M68 (2026-10-16): IBC's interpolation (intra_ibc.glsl: getPredIBCBlk -> InterPrediction::xPredInterBlk with isIBC,
luma, rndRes, no LIC) and SGPM's fractional vectors. ibcPredict gives a sample of a region (w x h at a CU's position,
which checkValidBv takes even for a template) at a vector in 1/16 samples: a copy at an integer vector or without
the SPS's IBCFracFlag (Params::ibcFrac; the integer part then), else m_lumaFilter's 8 taps (IF_12TAP's x 4 table,
filterIdx -1) or the bilinear filter (filterIdx 1: SGPM's templates), one pass rounded to the sample or two through
the 14-bit intermediate (headroom max(2, 14 - bitDepth)). ibcAreaValid is PU::searchBv with the filter's reach
(JVET-AJ0172's area); where it fails (IBC_INT_BV_VALID) ibcRef pads as xPredIBCBlkPadding does, per sample: rows of
the region left / right of it from the first unit not decoded on take the sample next to that unit, then rows above /
below take the one nearer the region where their unit is not decoded. One ibcRef call site per module (it inlines).
SGPM: a record's vectors for SGPM's lists are its vectors x 16 unless BV_OUT_SGPM gives its own (count, 2 x / y in
1/16: a CPU IBC CU's mv, an IntraTMP CU's mv[0] - sub-sample ones point between samples -, an SGPM CU's own, written
by the host for CPU CUs (cpuSgpmVectors) and by a GPU SGPM node; a flipped IBC CU gives none, as it keeps the SGPM CU
on the CPU where its flipped vector is valid). The SGPM list holds integer parts in s_bvList and fractions in
s_spCost (SG_FRAC); its template copies are computed once (bilinear at a fraction) into the scratch (SC_COPY) for the
vectors' costs and the modes' template costs; DimdOut's SGPM_BV holds x / y in 1/16 (DIMD_OUT 45); the block's copy
interpolates over the CU with the 8-tap filter (mvOfSub). An SGPM node's record: the vectors rounded as
Mv::changePrecision rounds (for the other lists) and in 1/16 (BV_OUT_SGPM). Finding: an EIP CU with a vector gives
SGPM's list its vector x 16 (setEipBvInfo; checked with a debug print in CPU and verify mode), so the M67 rejection
"SGPM with an EIP neighbour with a vector" was never needed. Results on the Intel PC's 48-stream corpus (build gpu2):
every stream as before (the 2 single-tree streams differ as at M66), SGPM CUs on the GPU 700 -> 884, intra CUs from
the CPU 2966 -> 2767 (both rejections gone; "an intra merge neighbour of a kind the GPU lacks" 339 -> 308).
The padding ran: a debug print in xPredIBCBlkPadding found 4 padded SGPM predictions in the corpus (ai_foreman352_q22:
two blocks with the 8-tap filter, CUs at 88,128 and 136,24; ld_fourpeople_q37: the two bilinear templates of the CU
at 232,240); all three CUs were the GPU's in verification (debug print in addSgpmRecord). SPIR-V 3.22M / 4.04M IDs (152k
left in the CCP-LFNST module). Run verify_all.py on the RTX PC (its 444 streams are not on the Intel PC).
M69 (2026-10-16): IntraTMP's sub-sample vectors (tmpFracIdx 1, JVET-AM0229's searchFracCandidate). The host's record
flag TMP_HEAD_FRAC; the node, after its list, runs tmpFracSearch: xPadForFracSearchInterpolation's region around the
template at the block's vector copied into the workgroup's scratch (TS_FBUF; reads clamped into the picture - the GPU's
planes have no margin, and an unclamped read 2 rows above the picture lost the device on ai_foreman352_q22 - every
such sample is replaced by the padding anyway), its five padding steps ported literally (invocation 0), the cost of
each of the 128 (fracPrecX, fracPrecY, direction) combinations searchFracCandidate's loop takes (one invocation each:
xTmpFracSearchIF's 4-tap cubic filters of defineFilterTmpSubpel0 / 1, each pass rounded and clipped; calcTemplateDiff's
cost of the entry's required template - m_mtmpRequiredTemplate, which the merged list now carries in TE_REGION's bits
8+), the first cheapest (updateCandList keeps the first of equal costs). The block's vector in 1/16 samples goes to
DimdOut (DIMD_OUT_TMP_MV, compared with pu.mv[0]) and to the record for SGPM's lists (bvStoreSgpm); tmpPredict
interpolates with ibcPredict over the CU (8 taps) with xPredIBCBlkPadding's deltaX / deltaY: for such a CU the
padding's region moves by 1 where the vector's integer part is left of / above pu.bv, while the filter still reads
from the integer part (the function's last line points refBuf back; found with debug prints of the padding's geometry
- a poison test of the local buffer had changed nothing). Results (Intel PC, 48 streams, build gpu4): every stream
as before (45 identical, the 2 single-tree ones as at M66, ra_wpp_q32 without an intra picture to verify); over the
46 streams both runs report, intra CUs from the CPU 2546 -> 2520: "IntraTMP (sub-sample)" 127 -> 0, but most of those
CUs also have OBMC ("IntraTMP with OBMC" 323 -> 423), so IntraTMP's OBMC is next. ai_crowdrun832_q27 (~3.0 s of GPU
time per picture on this iGPU, M68 2.96 s) lost the device once in partial.py's lone rerun and verified identical in
3 of 3 lone runs: Windows' GPU timeout on the heaviest picture, not a fault of the code.
M70 (2026-10-21): IntraTMP's OBMC (InterPrediction::subBlockOBMC for an IntraTMP CU, JVET-AK0076 with JVET-AJ0161's
intra predictions; intra_obmc.glsl). The CPU's flow, read from the preprocessed sources (g++ -E -P with the build's
flags: the #if forest of subBlockOBMC resolves to ~500 lines): where the SPS's OBMC, the CU's obmcFlag, no flip, 32
samples and no SCC CU around it (isSCC: palette / BDPCM CUs, MotionInfo::isSCC) hold, per side (top, then left) runs
of 4-sample units (getSameNeigMotion): a run of one valid IBC-like vector (IBC, IntraTMP, EIP with a vector; not
flipped, FLM, fusion, filtered, two-model LIC, BVG-DIMD; checkIsIBCCandidateValidBi for the strip so far) that is not
the CU's own gets the copy at that vector (getPredIBCBlk: ibcPredict) blended by weightObmcBoundary mode 2 (rows 0-3:
26/6, 28/4, 30/2, 31/1 of 32) unless skipObmcConditionByPixel (the copy at the vector clipped and rounded to samples,
where valid, is 96 << (bitDepth - 8) or more off the prediction before OBMC anywhere); runs of other units, where
getGradForOBMC gave the side modes (DIMD on; calcGradForOBMC: per run of intra units of one mode, a histogram of
dimdAngle over the 2x2 gradients of the two rows / columns next to the CU, with the 2 samples either side carried
over), get per run of one mode within the side's angles (above 34..66, left 3..34) an angular prediction of the strip
(subBlockIntraForOBMC: predIntraAng with isobmcMC - no reference filter, cubic taps, no fusion or PDP, PDPC only for a
side's first unit; the references the regular fill for the first unit, else xFillReferenceSamplesOBMC with that side
alone, and with prepareRef the samples beyond the strip's corner along the side replacing the projected main-array
entries) blended by mode -1 (the same weights). Host: addTmpObmcRecord (TMP_HEAD_OBMC: flags and per unit the kind,
the record whose Params::cuMode is its intra mode - what the MPM lists read, planar for EIP / NN / MIP -, the vector or
its GPU source); rejections "IntraTMP OBMC with a neighbour with LIC", "... of two vectors or inter", "... with an SGPM
neighbour" (an SGPM CU's mode per position) - 35 CUs on the corpus. GPU: tmpPredict (moved to intra_obmc.glsl) plans
the runs (obmcPlan, invocation 0) and predicts the block and the motion runs in one loop (one ibcPredict call site);
tmpObmc after the DIMD of the prediction (the CPU derives intraTmpDimdMode before OBMC) skips / predicts the intra
strips (obmcIntraStrip: applyModeParams on a strip Tu, obmcFillReferences - the CPU's regular line-0 fill or its
OBMC variant -, angularArrays, the overrides, the long-tap cubic loop) and blends. The CPU's reference fill for a
strip after the first unit leaves top entry 1 stale (only left), which a filtered corner would read: it cannot happen
as isobmcMC turns the reference filter off (no strip of the corpus had it). Results (Intel PC, 48 streams, build
gpu5): every stream as before (the 2 single-tree ones differ as at M66); over the 46 streams both runs report, intra
CUs from the CPU 2520 -> 2267 ("IntraTMP with OBMC" 423 -> 0; 35 CUs rejected by the narrower rules), IntraTMP CUs on
the GPU 65 -> 318; the first three streams verified alone had 128 intra strips (15 after a side's first unit, 5 with
prepareRef) and 41 motion runs on the CPU (its debug counts). SPIR-V: 3.34M / 4.17M IDs - only 26k left in
the CCP-LFNST module: the next milestone must make room first (the strip's angular code is a second copy of the main
path's; a shared call site would save ~60k).
M76 (2026-10-22, agent sg): regression SGPM and the rest of SGPM: JVET-AJ0112's
regression SGPM, SGPM without blending and without TIMD / DIMD. A regression SGPM CU (PU::isRegressionSgpm: sgpmIdx
below RSGPM_CAND_NUM in a CU above 4 x 4) skips deriveSgpmModeOrdered's partitions: its list is the MPM list's first
six (planar, getSpatialIpm forced valid without MDIP's exclusions - DecCu clears isModeExcluded around it -,
fillMPMList: mpmSpatial + mpmFill on the CU's MPM neighbours and DIMD) and then the cheapest SGPM_NUM_BVS vectors; per
pair of entries (the first ascending, then the second after it) deriveSgpmBlending fits the second mode's weight as
p0 x + p1 y + p2 to the one-sample template (row above at y = -1, column left at x = -1) by CccmCovariance::solve1
with 3 parameters - features (P1 - P0) x, (P1 - P0) y, P1 - P0, target reco - P0, each truncated to a Pel; the dot
products as the CPU's SSE cccmDotProduct sums them (madd: in each full group of 8 samples two products added in 32
bits); matrix bits 28 (32 above 10 bits), decimation 16 (22), regularization 2 << (bitDepth - 8) - and the model's
params as ints with shift decim - 5 (the CPU's shiftA is always 0: it sizes the freshly constructed model it was
given, {16, 16, 0}); equal weights {0, 0, 16}, shift 0 where the clipped weights at the CU's corners differ by 4 at
most; cost the SAD of ((32 - w) P0 + w P1 + 16) >> 5 against the template; the CU's sgpmIdx picks among the four
cheapest (updateCandList: the first of equal costs first). intra_sgpm.glsl: the node builds that list (invocation
0), keeps each slot's template prediction in the scratch (SC_RPRED) instead of its differences, one invocation per
pair runs sgpmRegression (int64 elimination as ccpBlendModel's, with the decimation bits as a parameter), the rank
loop is the partitions'; DimdOut gets DIMD_OUT_SGPM_MODEL (p0, p1, p2, shift; DIMD_OUT 45 -> 49; verification
compares it with cu.blendModel), sgpmSplitDir 0, and the record the MPM lists read planar (sgpmDimdMode, which the
decoder never sets: initData's 0; sgpmMpmMode did so already for CPU CUs). Its block blends with the model's weights
(weightAffineBlk / weightedBlendBlk: 32 - w for the first mode, w for the second): sgpmBlockWeights now writes every
SGPM block's first-mode weights into the workgroup's scratch after the blending sums (SGPM_BLOCK_W) and emitPred
reads them there (computing the model's weight in emitPred's 8 call sites had cost ~19k SPIR-V IDs); its transforms
take sgpmDimdMode (planar) for MTS and, where TrQuant / getFinalIntraModeForTransform take the angle, the DIMD of the
blend. DecCu derives timdHor / timdVer only for non-regression CUs of 1024 samples at most with the SPS's TIMD, else
they stay initData's -1 (getSgpmIntraMPMs skips them): such a CU has no TIMD record and its node skips the TIMD pass
(SGPM_NO_TIMD, DimdOut's TIMD fields -1). Without the SPS's DIMD the CU has no DIMD (c.dimd -1: getSgpmIntraMPMs and
getSpatialIpm take none) and deriveIpmForTransform returns planar with a second mode of 0: mipTransformMode adds no
gradients for such an SGPM block (SGPM_NO_DIMD); verification now compares an SGPM CU's fields without a DIMD too.
The record's SGPM_HEAD_NUM (unread) became SGPM_HEAD_FLAGS. Without blending (the PPS's sgpmNoBlend) the block takes
g_geoWeights' first blending width (a third weight table per size in sgpmTables; the template weights are the same).
Results (Intel Core Ultra 7 265K iGPU, build agsg3 with ECM_SPIRV_OPT and ECM_SPIRV_INLINE=OFF): the 43 corpus
streams with these CUs (incl. ai_foreman352_q37, ra_foreman_q32) as before - 41 identical, the 2 single-tree ones
with the same first difference as at M66 (more samples differ after it); over them "SGPM (regression)" 137 -> 0,
"SGPM without blending" 14 -> 0, "SGPM without TIMD / DIMD" 9 -> 0, SGPM CUs on the GPU 860 -> 1019, intra CUs from
the CPU 2206 -> 2001 ("IntraTMP OBMC with an SGPM neighbour" 11 -> 16: more SGPM neighbours are the GPU's).
Remaining: "SGPM above 1024 samples" (1 CU, ra_parkjoy_q37): the block's deferred prediction lives in s_spPred (32 x
32), as IntraTMP's; a larger SGPM block needs it elsewhere (and its blend weights beyond SGPM_BLOCK_W's 1024).
SPIR-V 3.26M / 4.08M IDs (113k left in the CCP-LFNST module).
M71 (2026-10-22): IntraTMP's LIC (tmpLicFlag, JVET-AE0078), with or without the sub-sample search. The
search (useMR) costs mean-removed SADs - each template part's sample less the difference of the part's means, each
mean the part's sum floored by its size's log2 (calcTemplateDiff / calcTargetMean, the SIMD versions; the top part's
mean also for the L-shape's corner) - keeps the sparse lists at MTMP_NUM_SPARSE_FOR_LIC 15 / TL_NUM 3 without the
SPS's LIC mode, and moves each sparse position half a step towards its region's middle (iXOffsetMetric; "transferred"
back where that leaves the region, or in regions 4 / 5 where its block is not decoded before the row's first such),
whose flags its refinement window keeps (clipMvIntraConstraintRefine: from the position to S - 1 beyond). The node then
fits the LIC model of the block's vector (LicItmp -> xLocalIlluComp with ibcLicFlag: the row above / column left of
the CU at the vector where checkIsIBCCandidateValid allows, sampled as xGetLICParamGeneral steps by getLicDimBit; one
model of m_licShift 5 with m_licMultApprox's division, or with IBC_LIC_IDX_M MMLM's two by cclmFit; required
template from the list unless ItmpLicExtension) into its DimdOut (DIMD_OUT_TMP_LIC) and the slot SGPM's lists read
(BV_OUT_LIC, which the host reserves for a GPU CU); with the sub-sample search that model goes through the padded
template region before the costs, and the model is fitted again at the sub-sample vector (bilinear template
predictions, the padding region moved as the block's). The block applies it after the DIMD of its prediction (the CPU
derives intraTmpDimdMode in generateTMPrediction, before LIC). OBMC of a LIC CU: its own vector's runs are not skipped
(getSameNeigMotion: the neighbour's LIC flag differs).
M72 (2026-10-22): IntraTMP fusion (tmpFusionFlag): the search for the group's last entry
(sizeUsed), then by tmpIdx the group of 5 (getTmpFusionIdx / Number): below TMP_GROUP_IDX a blend - where the group
has entries of the top / left template alone the location-dependent blend (xLocationdepBlending, range 16) of the
first of each over the blend of up to 3 others (32 : 32, 22 : 21 : 21), else convertDiff2Weight's weights of the
costs -, from TMP_GROUP_IDX on a filter model of the 5 candidates and a bias fitted by solve1 on the template (each
candidate less its reference point; with LIC through its own model, fitted per candidate at the node). The node writes
the group's vectors as cpuBlockVectors' fusion branch takes them (BV_OUT, BV_OUT_SPARSE, BV_OUT_FUSION) and the fusion
data into DimdOut (DIMD_OUT_TMP_FUS*, DIMD_OUT 45 -> 82); the block computes each sample (tmpFusionSample). Found on
verification: getSameNeigMotion compares no motion for a fusion / FLM / two-model LIC CU, so OBMC runs at its own
vector are not skipped (the first build's OBMC differed at those runs; the CPU's subBlockOBMC / xSubblockTMOBMC printed
the runs to find it).
M73 (2026-10-22): IntraTMP FLM (tmpFlmFlag): the area at the vector (block and template) less its corner sample, padded
by xCalcTmpFlmRefArea's rule (a vector at its region's bound, or for regions 1 / 3 / 4 / 5 samples right / below not
decoded), a 5-tap model (the sample and its 4 neighbours, a bias) fitted by solve1 on the template parts its required
template takes; the node fits it (tmpSolve1, shared with fusion's filter), the block convolves.
Results M71-M73 (Intel PC, 48 streams, build gpu11: ECM_SPIRV_OPT with ECM_SPIRV_INLINE=OFF): every stream identical
but the 2 single-tree ones (as at M66; ra_wpp_q32 unverifiable); intra CUs from the CPU 2453 -> 1697 ("IntraTMP
(LIC)" 378, "(LIC, fusion)" 223, "(fusion)" 217, "(FLM)" 83 -> 0; "IntraTMP OBMC with a neighbour with LIC" 25 ->
102 and "... with an SGPM neighbour" 11 -> 42 as more CUs reach those checks). These were the first milestones on the
spirv-opt build (868f31d26) and the non-inlined SPIR-V (f84f0f418): its pipelines compile in 144 s / 4.1 GB instead of
760 s / 16.9 GB (MPM + CCLM), which the PC's 60 GB budget needs.
M74 (2026-10-22): intra merge's IntraTMP neighbours (CU::setIntraMergeCandParam with itmpIbcMode ->
getItmpBasedIntraMergeCand): such a neighbour is the CU its vector points at - of the CUs at the 4 x 4 positions of
its area moved by its pu.bv (getCURestricted), the first of the largest overlap - as that CU's class makes it a
candidate, with two position checks (isDifferentIntraBlk of the IntraTMP CU, then of that CU) and the IntraTMP CU's
position for the candidate (the outer call overwrites the inner one's). The host lists the neighbour as MERGE_NBR_TMP
by its CodingStructure index and, after the map of block vectors' CUs, a table per CodingStructure index (MERGE_INFO:
position, size, record, class - mergeClass, the neighbour rules of addMergeRecord, -1 for a kind the GPU lacks); the
node finds the CU from the vector the neighbour's block-vector record holds first (BV_OUT: pu.bv, a GPU CU's from its
node) and the map (mergeTmpRef). The host still checks the CU the CPU found ("... whose reference CU the GPU lacks";
it puts an MPM CU's DIMD kept aside to use) and makes such merge nodes prefix nodes. Results (48 streams, build
gpu12): every stream identical but the 2 single-tree ones; intra CUs from the CPU 1697 -> 1477 ("an IntraTMP
neighbour of intra merge" 200 -> 0, 9 for the new reason; "an intra merge neighbour of a kind the GPU lacks" 308 ->
279).
M81 (2026-10-22): IntraTMP OBMC with a neighbour with IBC LIC: the run's copy goes through the neighbour's model (its
licScale / licOffset, shift 5: xPredInterBlk's linearTransform with isobmcMC), also for skipObmcConditionByPixel's
copy; the record's unit carries it (OBMC_U_LIC: the CU's fields, or a GPU IntraTMP CU's node model); the CU's own
vector counts as a neighbour's only with equal LIC flags and, with LIC, scale and offset (getSameNeigMotion).
M82 (2026-10-22): IntraTMP OBMC with an SGPM neighbour: the unit's mode is the SGPM CU's mode of its part there
(spanIpmInfoSgpm's tpmMask per 4 x 4 unit, tabled per CU size and partition after sgpmTables' weights; a vector's
mode planar), from the SGPM node's partition and modes. Results (9 streams with these CUs, build gpu16): all
identical; "IntraTMP OBMC with a neighbour with LIC" 102 and "... with an SGPM neighbour" 49 -> 0.
M83 (2026-10-22): a palette CU as an MPM neighbour: its mode at the position as the CPU keeps it (getIpmInfo), as
for IBC and inter neighbours. Results (ai_scc2_q32, ai_scc_q27, ld_scc_q32, build gpu16): identical; "an MPM neighbour
that is not a recorded intra CU" 58 -> 0 (most of those CUs now stop at "DBV").
M84 (2026-10-22): partial verification of inter slices: supported() takes them with ECM_GPU_PARTIAL (ON mode still
stops at "not an intra slice"), their inter / IBC / palette CUs come from the CPU (cpuCu) and their intra CUs are
verified like an intra slice's. What inter slices change for intra CUs: CCP merge (getCCPModelCandidateList) takes
temporal candidates (JVET-AE0043) after the 5 adjacent positions - the collocated picture's models at C0 / C1 per
distance (m_ccpModelLUT, the CPU's: host-written records, ccpTemporal; ON mode will need the collocated picture's
records kept on the GPU) - and again after the history, moved by the first inter neighbour's vector to the
collocated picture (Params::ccpData's CCP_MERGE_NUM_AFTER sources, after the LUT in intra_ccp.glsl's ccpMergeSource;
the first CCLM candidate is looked for there too); no history with separate trees in an inter slice
(ccpMergeHistory); an inter CU's TU models (JVET-AF0073) enter the history only as CU::saveModelsInHCCP finds them,
before CU::saveProCcpInfo gives the TUs without one the collocated picture's model at their motion (candidates only:
cpuCuModels' second call); LBCCP (CCCM's second set of models) is intra slices' only (Params::cccmLbccp); decoder-
derived CCP fusion's GL-CCCM mode is there above 8 chroma samples instead of 32 (Params::ccpDerivedGl). Rejected for
now: a CCP merge CU with an inter CCCM candidate (CCP_TYPE_INTER_CCCM: its own model kind), LM chroma fusion's first
type in an inter slice (it blends an MMLM CCLM prediction there). Results (50 streams, build gpu22): every verified
picture identical (327 pictures, 273 of them inter pictures (54 before)); the intra CUs from the CPU in inter
pictures by reason: EIP in an inter slice 67, IntraTMP OBMC with a neighbour of two vectors or inter 25 (9 before), CCP merge
with an inter CCCM candidate 14, an inter OBIC neighbour 5, LM chroma fusion in an inter slice 3.
M75 (2026-10-22, agent hb): 12-bit internal bit depth for the tools rejected above 10 bits (JVET-AJ0237 is compiled
in; the SPS has one bit depth for luma and chroma, sps_bitdepth_minus8, so "unequal bit depths" cannot occur in a
decoded stream and those checks stay). The template costs: xGetHADs_SIMD's kernels keep 16-bit lanes up to 12 bits -
the 4x4 kernel wholly (the column and row transforms wrap, each lane - one vertical frequency - adds its four absolute
values in 16 bits, unsigned, then sad - absDc + absDc / 4 in uint32_t), the 8x4 / 4x8 kernels for the column transform
only (>= 10 bits: widened to 32 for the rows), the thin ones (1x16, 1x8) only for the difference; timdHad now does the
column transform first and wraps where the kernels do (lane16): identical to the kernels at 8, 10 and 12 bits in a
harness of 24M random / extreme / overshooting blocks (the old code differed at 12 bits on 4x4 and 4x8; nothing wraps
up to 10 bits). TIMD-SAD's xGetMRSAD_SIMD adds an AVX2 lane's 8 distances of a 128-wide arm in 16 bits, saturating at
32767 (reachable from 12 bits): timdTemplateCosts sums such an arm per row and lane, min 32767. Each call site calls
timdHad once (arguments by arm; it is inlined: the CCP-LFNST module 26k -> 54k IDs left with all of this). The
CCP / EIP models: Params::cccmDecim (DECIM_BITS
of the luma bit depth, 22 above 10 bits) replaces CCCM_DECIM in the divisions (xGetDivScaleRoundShift's scale,
xCccmDivideLowPrec), the eliminations, back substitutions, the bias offsets, clearModel's bias, the convolutions, the
blend model's shift (decimBits - 5) and CCP merge adjustment's deltas (<< decimBits - 5); cccmMatrixBits gives the
solvers' matrix bits (32 above 10 bits for solve1 / solve2 / solveEip); cccmNonlinear returns a Pel as
CccmModel::nonlinear does (the nonlinear term of a several-filters gradient passes 2^15 at 12 bits). CCLM's fit already
was JVET-AJ0237's 64-bit code. The rejections now read "above 12 bits" (the scalar distortion forms and 13+ bits are not
ported). Test set: t12 (C:\Code\ECM\streams_hb, not in git: the stock encoder's AI configuration with every tool,
--InternalBitDepth=12 from 8-bit sources, 2-3 pictures: foreman CIF QP 22 / 27 / 37, ParkJoy 416x240 QP 27 / 37,
CrowdRun and FourPeople 256x256 QP 32, gen_screen.py's screen content QP 22 / 32). Results (Intel Core Ultra 7 265K
iGPU, build aghb2, partial verification): every t12 picture as with gpu5 (all identical but t12_four32's POC 2, whose
first difference - indicesRepresentationPnn of an NN CU - gpu5 shows alike); intra CUs from the CPU on the 9 streams by
the seven reasons 11138 -> 0 (MPM sorting / MDIP 3545, TIMD 3052, CCLM 2309, TMRL 1165, EIP 579, intra merge 424, LM
chroma fusion 64), 370 of them now rejected for a neighbour the GPU lacks (an IntraTMP / intra merge neighbour of intra
merge 363). ra_internal12_q32: identical, intra CUs from the CPU 147 -> 4 (the seven reasons' 142 -> 0);
ai_foreman352_q37, ra_foreman_q32, ra_10bit_q32 identical with the same CUs from the CPU as gpu5.
Limits: the 4x4 kernel's uint32_t result can wrap (a lane's sum past 2^16 below absDc); an arm's GPU cost is a 32-bit
sum, the CPU's 64-bit, so two such blocks in one arm would differ (needs differences averaging above ~2^10 in a 4x4
block's DC lane). Chroma TMRL above 10 bits stays rejected (its co-located luma SATD takes the 16x8 / 8x16 AVX2
kernels: not ported; 99 CUs on t12).
M77 (2026-10-22, agent st): single tree (SPS DualITree=0, where a CU's PU holds luma and chroma): the two single-tree streams
differed since M66 (section 9's a2), from two host bugs. (1) A CCP merge PU of fusion with DIMD chroma
(ccpFusionDimd) set DERIVE_CCP_DIMD on its luma block too: intra_rec blended the luma prediction with "the DIMD chroma
mode" (ra_nodualtree_q32's luma MPM block at 32,24, then DIMD chroma's second mode from its luma; in
ai_nodualtree_q27 the 3 such CUs' luma, then NN's indicesRepresentationPnn from that reconstruction). Now chroma
blocks only. (2) Partial verification: a single-tree CU taken from the CPU at its luma block (cpuFillCu) made its
chroma record at once, before the CPU decoded its chroma - curCand still untyped, so no CCP record, no history entry
(and DIMD chroma's mode unset): the merge CUs after it lacked that candidate (ai_nodualtree_q27's CCP merge block at
0,56: the CPU's list 128, 1, 16, ..., the GPU's without the CCLM model of a CPU CU). Now cpuFillCu defers the chroma
record (cpuChromaPending, cpuKnownRecord) to the CU's end: DecCu's intra case calls cpuCuModels after
CU::saveModelsInHCCP, which makes it then. Results (Intel PC, build agst5, non-inlined SPIR-V): ai_nodualtree_q27,
ra_nodualtree_q32 identical (were MISMATCH); ai_scc2_q32, ai_foreman352_q37, ra_foreman_q32 as before (same CPU CU
reasons). Host only.
M78 (2026-10-22, agent st): IntraTMP's chroma in a single tree: the rejection "IntraTMP's chroma (single tree)" (134 CUs
on ai_nodualtree_q27 1, ra_nodualtree_q32 2, ai_scc2_q32 131) is gone: the CPU predicts such a CU's chroma as any
chroma CU's (xIntraRecBlk: initIntraPatternChType, no IntraTMP step; its OBMC, LIC and sub-sample vectors are luma's;
DM and the candidates' substitution read planar, getIntraDirLuma of an IntraTMP PU, which the CU's node publishes as
its mode). What remains is the chroma mode: an IntraTMP luma CU makes DBV available (PU::dbvModeAvail) and 133 of the
134 CUs take a DBV mode, now counted as "DBV" (4 -> 137 on those streams; BVG-CCCM, also offered by it, stays
rejected as before); one CU (ra_nodualtree_q32) moved to the GPU. DBV (refineChromaBv's template refinement, IBC's
fractional interpolation, RR-IBC flips) is the next step for these CUs. Results (build agst6): the 5 streams
identical, the CPU CU reasons otherwise unchanged. Host only.
M79 (2026-10-22, agent lf): LFNST without the SPS's LFNSTExt: VVC's kernels (TrQuant::xInvLfnst / invLfnstNxN with lfnstExtFlag
off: g_vvcLfnstLut's 4 sets, lfnstIdx 1-2, g_vvcLfnst4x4 16 x 16 or g_vvcLfnst8x8 16 x 48; no 16 x 16 region, so a
block of 8 x 8 and above takes the 8x8 region and g_coefTopLeftDiagScan8x8; 16 inputs, getLFNSTMatrixDim's 16, for
every size). Host only: invSecondary chooses them like the CPU, the kernel tables join secKernels, the output
arrangements (lfnstOutputLayout with ext off: 48 outputs, untransposed rows 0-3 of 8 and 4-7 of 4) and a second
TR_TAB directory (Params::trTables by the SPS) whose LFNST matrix table holds VVC's matrices under g_lfnstLut's set
(g_vvcLfnstLut is a function of it, checked at creation; MTSS compares g_lfnstLut's sets in any case), so intra_rec's
selectTransform and mtssChoice are unchanged. CPU quirk: transposed, xInvLfnst copies the 8x8 region as for the
extension's 64 outputs, columns 6-7 from entries 48-63 of m_tempOutMatrix, which no VVC kernel writes (zero); those
levels are zero too (lfnstIdx is parsed only with the last level in the first 16 scan positions), so the layout leaves
them out. Results (Intel PC, build aglf1): ai_off15_q27, ra_off15_q32, ra_off5_q32 identical, "LFNST without its
extension (the VVC kernels)" 77 / 106 / 106 -> 0 CUs from the CPU (a debug count saw every kind: 4x4 / 8x8 regions,
both indices, transposed or not); ai_foreman352_q37 and ra_foreman_q32 identical with the same CPU CUs. No shader
change.
M80 (2026-10-22, agent lf): LFNST of MIP, the rest of M23's rejections: without DIMD in the SPS deriveIpmForTransform returns planar
at once (mipDimdMode planar, dimdDerivedIntraDir2nd 0), so the host's choice is the CPU's (CodingUnit::initData's
values in ON mode) and the block derives nothing (no DERIVE_MIP_DIMD); a single tree's chroma blocks of a MIP CU with
lfnstIdx take no LFNST / NSPT (luma only) and were rejected for nothing. Still rejected: a MIP CU of several blocks with
DIMD (none on the streams). Results (build aglf3, with the ECM_SPIRV_OPT / ECM_SPIRV_INLINE=OFF shaders): "LFNST of
MIP" 33 -> 0 CUs from the CPU (ai_off15_q27 8, ra_off15_q32 15, ra_off4_q32 2, ai_nodualtree_q27 4,
ra_nodualtree_q32 4); ai_off15_q27, ra_off15_q32, ra_off4_q32, ra_off5_q32, ai_foreman352_q37, ra_foreman_q32
identical; the two single-tree streams differ first where they did before (gpu5: the same field and block). Their 8
MIP blocks with LFNST / NSPT (DERIVE_MIP_DIMD) lie after that difference, so a temporary build that took every other
CU from the CPU checked them: all 8 identical (prediction, residual, derived transforms); ra_nodualtree_q32 then
identical, ai_nodualtree_q27 differed at the chroma block of one of them, a CCP merge block (its model's type), the
kind of block the known single-tree difference starts at - open.
M85 (2026-10-22, agent st): DBV of a single tree (JVET-AC0071's DBV_CHROMA_IDX in a CU whose own luma has a block vector):
IntraPrediction::predIntraDbv's first branch. refineChromaBv takes the luma PU's mv[0] where its integer chroma part
is a valid vector for the CU's Cb block (checkIsChromaBvCandidateValid: checkValidBv without the fraction, PU::searchBv
on the chroma decoded map), else bv x 16; the chroma block is then the copy at it or, with the SPS's IBCFrac and a
fraction of 1/32, getPredIBCBlk's chroma interpolation (m_chromaFilter, 6 taps, one pass or two through the 14-bit
intermediate, xPredIBCBlkPadding where checkValidBv with the taps' reach - 6 x 2 luma samples - says
IBC_INT_BV_VALID); getFinalIntraMode makes it planar where bv is 0. GPU: a chroma CU kind CU_CHROMA_DBV (17) whose
node (DERIVE_PREFIX: the copy may read anything decoded before) publishes DBV_MODE (74, DBV_CHROMA_IDX) or planar from
the luma CU's vector as its node wrote it (dbvLumaVector: an IntraTMP CU's DIMD_OUT_TMP_MV / DIMD_OUT_BV, an EIP CU's
with a vector DIMD_OUT_BV, a BVG-DIMD CU's first DIMD_OUT_BVG_BV; mv[0] their x 16 but IntraTMP's); its blocks with
DBV_MODE take no mode parameters (0, also in the host's record for the comparison) and predict with ibcChromaValid /
ibcChromaPredict (intra_ibc.glsl; IbcRef has a component now, ibcRef pads in that component's owner map). Host:
record()'s "DBV" became "DBV (dual tree)" plus the narrower single-tree rules (one chroma block per CU - predIntraDbv
copies at the PU's position -, 4:2:0, no LM chroma fusion, IBC LIC or RR-IBC flip, a luma CU of those kinds). Results
(Intel PC, build agst9, non-inlined SPIR-V, the 50 streams of C:/Code/ECM/streams, partial.py -j 1): every stream
identical; intra CUs from the CPU 2453 -> 2317 against gpu5 ("IntraTMP's chroma (single tree)" 134 -> 0, "DBV" 30 ->
"DBV (dual tree)" 28). The 270 DBV blocks of the three single-tree streams (debug count): 266 integer copies, 4 with a
fraction (interpolated, none padded), all at mv[0], luma IntraTMP (266) and BVG-DIMD (4); no stream has bv 0 (planar),
an invalid mv[0], padding or an EIP luma CU. Open: the dual tree's DBV (deriveChromaBv's luma CUs at 5 positions with
adjustChromaBv, chroma reordering's choice among them - which partial verification would take from the CPU as it takes
the reordered modes -, refineChromaBv's template matching over mv[0] +- one luma sample - SAD, SATD for blocks of at
least 4 x 4, on one-row templates - with RR-IBC flips, the flipped reconstruction).
M86 (2026-10-22, agent lf): NN intra prediction without DIMD and of 64 x 64 CUs: without DIMD in the SPS xPredIntraPnn's
deriveIpmForTransform returns planar at once and its second mode stays DC, so the CU's indicesRepresentationPnn are
{planar, DC} - what its transforms (getFinalIntraModeForTransform: LFNST / NSPT, also by the lfnstSecFlag rule,
explicit MTS; getNSPTBucket) and chroma DM read. The host sets them in ON mode (the CPU has not predicted; checked equal
to the CPU's in verification) and chooses the transforms itself (no DERIVE_PNN_DIMD). A 64 x 64 NN CU (the only shape
above 32 x 32: a 16 x 16 model, factor 4) has a prediction that does not fit s_spPred: pnnPredict stops after
upsample2d's first stage (s_spHigh, 16 rows of 64) and pnnSample computes each sample of the second stage where it is
read (intra_rec's g_pnnLarge: emitPred, the DIMD of the prediction for DERIVE_PNN_DIMD's transforms - mipPredAt -, the
deferred reconstruction; storeRecon keeps nothing then). No bilateral filter there (W x H < 1024) nor sign prediction
(up to 32 x 32). Still rejected: an NN CU of several blocks ("NN intra prediction of a split CU"; none seen). Results
(build aglf5): "NN intra prediction without DIMD" 47 -> 0 (ra_off4_q32; 25 of them with LFNST / NSPT or MTS),
"NN intra prediction of a split CU or above 32 x 32" 11 -> 0 (ai_parkjoy_q27 2, ra_parkjoy832_q32 9; all 64 x 64, 10
of them with DERIVE_PNN_DIMD: MTS, LFNST, lfnstSecFlag); those three streams, ai_foreman352_q37 and ra_foreman_q32
identical, the others' CPU CUs unchanged.
M87 (2026-10-22, agent lf): MTSS of SGPM CUs and of TIMD CUs without DIMD: "MTSS" were SGPM CUs with mdirIdx (isMdirAllowed's last
kind; the rule came before SGPM's own rejections, so 17 of its 28 CUs were regression SGPM, now reported as that). An
SGPM CU's node derives MTSS's mode list (CU_SGPM with the list-only kinds), its block after sgpmTransformModes chooses
(sgpmMtss: getFinalIntraModeForTransform's SGPM branch - the mode tried first is the prediction's DIMD second mode
where the LFNST / NSPT mode, in NSPT's numbering, equals the partition's angle, its main mode where it does not and is
not that mode, else the second; at the angle the main one where the second lies more than 2 from both of the CU's
modes - then mtssChoice) into its DimdOut before selectTransform; mipTransformMode leaves an SGPM block's MTSS alone.
Without DIMD in the SPS the mode list is empty (Cu::mtssNspt's MTSS_NO_LIST): mtssChoice then takes the mode tried first
(a blended TIMD CU's secondary) even where its set and transposition are the first one's, else planar (the CPU's
testMode as it starts), NSPT's second bucket by the same rule; the node skips the list's histograms and writes -1s,
which verification compares with the CPU's empty list. Still rejected: MTSS without DIMD of MIP, EIP and IntraTMP CUs
(no test stream). Results (build aglf6, the 27 streams with either reason): "MTSS" 28 -> 0 (11 SGPM CUs on the GPU, 17
"SGPM (regression)"), "MTSS without DIMD" 2 -> 0 (ra_off4_q32's TIMD CUs); every stream identical but the two
single-tree ones, which differ first where they did before.
M88 (2026-10-22, agent lf): EIP CUs of a single tree: "EIP CU with several transform blocks" were the chroma blocks of single-tree
EIP CUs (record() compared the chroma block with the CU's luma block); EIP is luma only, the chroma blocks predict by
their own mode (DM reads the luma block's eipDimdMode, which it publishes as the CU's mode) - the rule now applies to
luma alone. Results (build aglf7): 10 -> 0 CUs from the CPU (ai_nodualtree_q27 and ra_nodualtree_q32, 5 each: EIP CUs
on the GPU 0 -> 5 per stream); both differ first where they did before; a temporary build that took every CU but the
EIP ones from the CPU had both identical; ai_foreman352_q37 and ra_foreman_q32 identical, unchanged.
Integration of M85-M88 on M84 (2026-10-22, builds gpu23 / gpu24): all 50 streams identical. MTSS of regression SGPM
(rejected while merging M87, which predates M76's regression SGPM) verified on the 18 streams with such CUs (23) and
taken: identical.
M89 (2026-10-22): inter CUs on the GPU from the CPU's prediction (partial verification; the first step of the inter
port, the residual side): DecCu calls interCu before an inter / IBC / palette CU (a CU the GPU lacks something for is
taken from the CPU as before, with a finer reason: IBC CU, palette CU, inter LFNST / NSPT, inter CCCM, inter CCP
merge, ASBT, inter sign prediction) and recordInterCu after the CPU reconstructed it: per TU and component a
DERIVE_INTER block whose prediction is the CPU's (Params::interPred at the block's samples; intra_rec forward-maps luma
by LMCS where DecCu::xDecodeInterTU maps it, LMCS_INTER_FWD) and whose residual is the GPU's as an intra block's
(intra_resi; record()'s transform, dequantization and level parts became shared helpers, 4766bd917), reconstructed with
xDecodeInterTU's clipping (luma adaptive, mapped with LMCS), the chroma residual scaled by the CU's neighbouring luma
(crsSetup; also where the CTU has no LMCS), rootCbf 0 as no residual in any block. Such a block reads only what its
residual needs (intra_deps' and the host's crsDeps / residualDeps: joint CbCr, chroma residual scaling). The report
counts the inter CUs whose prediction the CPU made by kind (interKind: the first of multi-hypothesis, GPM, affine, CIIP,
sub-block motion, LIC, non-local LIC, OBMC flag, bi- / uni-prediction) - what the motion compensation port has to do.
Results (50 streams, build gpu26): every verified picture identical; inter CUs reconstructed from the CPU's prediction
10912 (OBMC flag 5836, affine 2359, GPM 1423, LIC 557, uni-prediction 481, bi-prediction 119, non-local LIC 113, CIIP
14, multi-hypothesis 10); still taken from the CPU: inter sign prediction 3157, IBC CU 2472, inter LFNST / NSPT 1255,
inter CCP merge 128, palette CU 123, ASBT 101, inter CCCM 16.
M90 (2026-10-22): sign prediction of inter blocks on the GPU: DecCu::xDecodeInterTU calls prepareInterSignPred before
TrQuant::predCoeffSigns (the CPU decodes the CU before recordInterCu, so the pending signs are kept per TU and
component, interSign); recordInterBlock takes them as record() does (the levels with the predicted signs positive, the
signPred data, joint CbCr's pairing) and intra_rec's DERIVE_INTER branch runs signPrediction after its prediction (in
s_spPred; luma forward-mapped by LMCS as xDecodeInterTU predicts the signs on). The border's reads (signDeps / the host's
residualDeps) are shared with intra blocks. Inter blocks may predict signs up to SIGN_PRED_MAX_BS_INTER (128); the
GPU's signPrediction keeps 32 x 32: "inter sign prediction above 32 x 32". Results (50 streams, build gpu28): every
picture identical; "inter sign prediction" 3157 -> 127 (above 32 x 32); inter CUs from the CPU's prediction 10912 ->
13873.
M92 (2026-10-22): LFNST / NSPT of inter blocks on the GPU (JVET-AG0061; a single tree's: luma only): their mode is
getFinalIntraModeForTransform's inter branch, dimdDerivedIntraDir (with lfnstIntra, JVET-AI0050's inter MTSS, the
second mode), DecCu::xDecodeInterTexture's deriveIpmForTransform of the unmapped prediction of the CU's luma (of SBT's
part with residual); the GPU derives it from the block's prediction (intra_rec's interTransformMode: buildHistogram's
3x3 gradients - the 2x2 ones up to DIMD_SMALL_BLOCK_THR samples, JVET-AJ0203's aligned design -, subsampled from
MIN_MTSS_SIZE on; DERIVE_INTER_SECOND for lfnstIntra), then selectTransform and inChainResidual as for a derived intra
mode; verification compares the transforms chosen. A CU whose transform block is not its luma (several TUs without
SBT) or without DIMD stays the CPU's. Results (50 streams, build gpu30): every picture identical; "inter LFNST / NSPT"
1255 -> 22 (without DIMD); inter CUs from the CPU's prediction 13873 -> 15000.
M93 (2026-10-22): motion compensation on the GPU (inter_mc.comp, the first step of the inter prediction port):
uni-prediction CUs without other tools (InterPrediction::xPredInterUni without bi, xWeightedAverage's copyClip): per
inter block a job (McJob: the block, the PU's size, its list's vector clipped by clipMv as xPredInterUni clips it, the
reference slot, AMVR's half-sample flag, the bit depth and the slice's clipping range) that writes the block's
prediction into Params::interPred before intra_rec (the CPU's staged prediction stays below it: ECM_GPU_NO_MC=1 keeps
it); InterpolationFilter's filters line by line (m_lumaFilter12, the alternative half-sample filter, m_chromaFilter;
the copy, one pass or two through 16-bit rows, rounded and clipped as filter<N, isVertical, isFirst, isLast> does), the
tables appended to Shared::matrices from the CPU's own (mcFilters). The reference pictures are the CPU's, uploaded with
their margins each picture: the margins hold JVET-AA0096 / JVET-AK0085's motion-compensated boundary padding, so the
GPU cannot replicate the picture's edge (ON mode will need that padding on the GPU). Found on the way (ra_scc_q32):
AMVP with SbTMVP (JVET-AG0098's amvpSbTmvpFlag) predicts per sub-block (xSubPuMC) with mergeType DEFAULT_N, keeping the
PU's own vector beside them - interKind counts it as sub-block motion now (also affine sub-block BDOF's refined
vectors); and a merge CU's motion is derived after interCu (xDeriveCUMV), so the choice is made in recordInterCu. A
scaled reference (RPR) is rejected per PU, not the SPS's RPR switch. Results (50 streams, build gpu32): every picture
identical; 585 uni-prediction CUs predicted by the GPU (all the corpus has).
M91 (2026-10-22, agent lf): the deblocking filter on the GPU (LoopFilter.cpp, GpuLib's dbf.comp, GpuAlf). The host derives
everything that does not read samples with the CPU's own code: LoopFilter::setGpuRecord makes loopFilterPic record,
per segment it filters (4 luma lines, or a chroma part of uiLoopLength lines, per component), a GpuDbfSeg - position,
flags (component, direction, lines, luma maxFilterLengthP / Q after the affine limit and sidePisLarge / sideQisLarge,
the palette's bPartP / QNoFilter, m_applyAsymmetricDB, chroma largeBoundary and isChromaHorCTBBoundary, the decision's
second line, LADF), luma tc / beta table indices before LADF's shift, chroma tc / beta -, with or without filtering (ON
mode: derive only). Edge flags, boundary strengths (motion from the CPU), filter lengths, availability across slices /
tiles / sub-pictures and virtual boundaries are thus the CPU's; a later GPU motion field would only have to fill the
same records. dbf.comp makes the decisions (xCalcDP / DQ, xUseStrongFiltering, LADF's luma level of lines 0 and 3) and
filters (xPelFilterLuma, xFilteringPandQ / xBilinearFilter with their asymmetric variants, xPelFilterChroma) line by
line after the C++. Order: the CPU filters all vertical edges, then all horizontal ones, CU by CU in coding order, each
edge reading what the ones before it wrote; along a row of segments (a column for horizontal edges) coding order is
left to right (top to bottom), and segments of different rows never share samples. So the host sorts the segments into
strips (pass, component, row / column of segments; GpuAlf::dbfPrepare, which refuses two segments at one place) and one
invocation filters a strip in order, in place, after a copy of the picture (mode 0). Not on the GPU: 4:0:0, several
slices (the slices' clipping ranges), LADF with a minimum CU size other than 4 (LADF's level comes from a part's first
segment). --GpuLoopFilter=2 (verify) deblocks a copy of the picture before deblocking right after the CPU's
loopFilterPic, also without ALF, and compares every sample (GpuAlf::verifyDeblocking; "GPU deblocking verification: N
pictures, M with differences" at the end, the first 3 differing samples per picture). ON mode (1, and 3: from the first
picture of any size, waiting for the device - tests) defers it to ALFProcess like SAO: when SAO runs on the GPU,
dbf.comp deblocks m_inDb (the picture before deblocking, which ALF uploads anyway) into m_inDeblocked (now with a margin
of 8: the filters read, and do not use, samples up to 8 beyond a segment) before the offset refinement and SAO, else
GpuDbfJob::cpu deblocks first. Creating a buffer waits for every submission (Context::inFlight), the recording
stream's own too: dbfPrepare allocates before the stream begins (the first version deadlocked there). New script:
perf/gpu/dbf_verify.py <build> [--on] (decoders under gpuslot.run; --on compares ON mode's output with the build's CPU
decoding). Results (Intel PC, build aglfd7): all 50 streams in C:\Code\ECM\streams, 331 pictures, deblocking identical
(ra_nodbk_q32, ra_dbkparams_q32, 10 / 12 bits, RPR, CTU 256, LMCS off, ALF off, SCC included); ALF / SAO
verification unchanged; ON mode (3) output identical with the CPU's on all 50, 318 pictures deblocked on the GPU, 8 on
the CPU (no SAO on the GPU). GPU time per picture (copy and both passes): 0.02 ms (deblocking off) to 1.4 ms on these
small pictures (ai_crowdrun832_q27 0.9 ms, ra_parkjoy832_q32 1.4 ms); there is no 1080p stream on this PC. Not known to be exercised by any stream:
asymmetric deblocking (GDR), LADF, virtual boundaries, 4:2:2 / 4:4:4 (palette only as far as the SCC streams use it).
M94 (2026-10-22, agent lf): IBC and palette CUs on the GPU (partial verification; DERIVE_IBC bit 30 when merged, 29 is M92's): interCu / recordInterCu take them as inter
CUs (DERIVE_INTER blocks: the residual intra_resi's, the reconstruction xDecodeInterTU's) whose prediction the GPU
makes. An IBC CU's block (DERIVE_IBC; Tu::ibc, its record in timdData: the CU's region of the component, its vector,
RR-IBC's flip, IBC GPM's second vector and weights, IBC LIC's index) waits for every block before it (DERIVE_PREFIX, as
IntraTMP's nodes) and is predicted in intra_rec (intra_ibccu.glsl's ibcBlock) at the CPU's final vector: the copy or
IBCFrac's interpolation with xPredIBCBlkPadding's padding (intra_ibc.glsl's ibcPredict / ibcChromaPredict; the CPU's IBC
buffer is a whole-picture copy of the reconstruction, so the GPU reads the picture), luma and 4:2:0 chroma; RR-IBC of a
one-TU CU reconstructed into the mirrored positions (xDecodeInterTU's flipSignal; prediction and residual unflipped);
IBC LIC of luma-only (dual-tree) CUs by intra_tmp.glsl's tmpLicModel (xLocalIlluComp's templates and fit, as IntraTMP
LIC's); IBC GPM of two IBC parts: DecCu hands over the parts' final vectors (ibcGpmVectors, after
motionCompensationIbcGpm), the host walks xWeightedGeoBlkRounded's weights, the GPU blends (w p0 + (32 - w) p1 + 16) >>
5. A merge CU's LIC, filter, flip, bi-prediction and OBMC flags come from its candidate while it is predicted, so ibcWhy
runs again in recordInterCu (ld_scc_q32 differed at an IBC LIC merge CU interCu had passed). The host's block-vector
table (Params::bvOut) now gives every non-intra CU's vectors, which no node derives (with IBC CUs on the GPU and none
there, their neighbours' IntraTMP / TIMD lists differed). Palette CUs: DecCu calls recordInterCu after xReconIntraQT;
the host computes xReconPLT's samples (palette entries, escape values dequantized; checked equal to xReconPLT's in
verification) into the block's prediction (cbf 0, no LMCS, clipped to the bit depth), the GPU writes them. Still from
the CPU, by reason: IBC with OBMC (subBlockOBMC: the neighbours' inter / intra predictions), IBC GPM with an intra part,
IBC bi-prediction (xPredInterBi's average), IBC filter (xCalIbcFilterParam's fit), IBC LIC with chroma (chroma templates
not ported), IBC CIIP, IBC CU in a GDR interval, RR-IBC flip of a CU of several TUs, IBC / palette CUs of other layouts
or with ACT. Results (50 streams, build aglfi9): every verified picture identical; "IBC CU" 2472 -> 0 (2389 IBC CUs on
the GPU, 41 of them IBC GPM; from the CPU: IBC with OBMC 38, IBC GPM with an intra part 17, IBC bi-prediction 16, IBC
filter 6, IBC LIC with chroma 1, and 5 now under inter sign prediction, 3157 -> 3162), "palette CU" 123 -> 0 (all on the
GPU); every other reason unchanged. Limit: each IBC block waits for every block before it; dependencies on the owners of
the area its vector reads would let more run at once (the four SCC streams' GPU time in partial verification, on a
shared GPU: ai_scc_q27 339.5 -> 341.0, ai_scc2_q32 1410.1 -> 1423.3, ld_scc_q32 147.4 -> 154.8, ra_scc_q32 113.6 ->
120.0 ms).
M95 (2026-10-22, agent st): chroma reordering and the dual tree's DBV (JVET-AH0136's deriveNonCcpChromaModes, JVET-AC0071's
DBV in a dual tree): until now partial verification took every reordered chroma CU's mode from the CPU and ON mode
rejected each sequence with chroma reordering or DBV in its SPS ("chroma reordering / DBV"). GPU: a chroma CU kind
CU_CHROMA_REORDER (18, intra_reorder.glsl) whose node (DERIVE_PREFIX: its costs read the co-located luma block and the
chroma around it) builds buildChromaModeList - DM (the co-located luma CU's mode; of an NN CU the DIMD of its
prediction, which its block now derives in every dual tree with reordering, also without coefficients or with transform
skip), DIMD chroma with its strongest five modes (dimdBlendModeChroma), planar, DC, HOR, VER, VDIA for a duplicate of
DM, the luma modes at the five co-located positions, the chroma neighbours' modes (not LM, not DBV) and with DBV
candidates (hasChromaBvFlag) the DBV entries 16, 17.. of deriveChromaBv's sources: a GPU luma CU's vector (IntraTMP,
BVG-DIMD, EIP with a vector: its node's) or the CPU's adjustChromaBv vector and flip (IBC: both vectors of a bi-vector
CU) -, then costs each entry as the CPU: 8 x the SATD of predCoLuma (coLumaReference's line, the mode's reference
filter, planar / DC / angular with PDPC, coLumaFullSample; a DBV entry the luma copy or IBC interpolation at its vector
by searchBv's luma rules, the edge fixes, the flip, coLumaDbvSample) plus predChromaTM's template SADs (templates of one
sample, CFLM's chroma TIMD path; the LEFT_NEIGHBOR / ABOVE_NEIGHBOR template types with their one-sample offsets,
crEdgePredict; DBV arms at their checkIsChromaBvCandidateValidChromaTm vectors) shifted by log2(H) + 2 / log2(W) + 2,
the early exit of the first two (entry 16 against DM with DBV), else updateCandList's stable order. DecCu passes the
parsed index (chromaReorderIndex); in ON mode it skips deriveNonCcpChromaModes and leaves planar placeholders. A DBV
entry's block: refineChromaBv (crDbvRefine: the vector and its one-luma-sample steps right / down valid for the Cb
block, the arms' costs against the block's reference line - the 1-row Hadamard of timdHadLine from 4 x 4 on, else SAD -,
the first minimum), predIntraDbv's copy or chroma interpolation (ibcChromaPredict), the reconstruction written flipped
(g_recFlip), the transforms' mode planar (getFinalIntraModeForTransform). Host: supported()'s gate is gone where
GPU_CHROMA_REORDER; what the GPU lacks is rejected per CU (ON) or keeps the CPU's mode (partial: "[gpu-intra] chroma
reordering CUs with the CPU's mode"): DBV candidates with LM chroma fusion (xCflmCreateChromaPred's DBV branch, its
2-line DBV templates and pre-flip), DBV candidates in a CU of several chroma blocks, no luma record, 12 bits, an
adaptive dual tree's separate tree, unrecorded chroma neighbours, a DBV template side unavailable. Found on the way
(ra_parkjoy_q37): an NN block derived the DIMD of its prediction only where its transforms read it, so the DM of a
chroma CU over an NN CU without coefficients read 0. Results (Intel PC, builds agst20 / agst21, non-inlined SPIR-V,
partial.py -j 1, the 50 streams): every stream identical; of 1029 reordered chroma CUs 994 now take the GPU's list
(before: all the CPU's), 35 the CPU's (DBV with LM chroma fusion 22 in 16 streams - one of them has a DBV mode, the
others only DBV candidates -, no luma record 7, 12 bits 6); intra CUs from the CPU 2317 -> 2292 ("DBV (dual tree)" 28 ->
2, both over a co-located luma CU without a record; "DBV with LM chroma fusion, LIC or a flip" 0 -> 1). ON mode
(--GpuRecon=3, all 50 streams identical to the CPU): no stream reaches the GPU yet; the first reason is now BVG-DIMD
(45), IntraTMP / IBC (3), MDIP / MPM sorting (1), wavefronts (1), each the gate after chroma reordering. Test streams
encoded here (runs/agents/st/enc: AI cfg without BVG-DIMD, MDIP, MPM sorting, IntraTMP, IBC, EIP, SGPM, PDP, NN; foreman
352x288 and ParkJoy 416x240, 2 pictures): partial verification identical (crowd32, fore22, fore32, park27, park37: 141
reordered CUs, all the GPU's); ON mode identical without in-loop filters (crf_fore32 32, crf_park27 15 reordered CUs),
but different with them - also without chroma reordering (nocr_fore32) -, and bisected to ALF (bis_alf: ALF, CCALF,
ALFCCCM on, other filters off; deblocking, SAO / CCSAO and BIF alone identical): an ON-mode ALF issue outside chroma
reordering, open.
Integration of M93-M95 (2026-10-22, build gpu33): all 50 streams identical after three fixes - an IBC block with sign
prediction (M90 lets inter blocks keep it) is reconstructed by signPrediction in intra_rec's DERIVE_IBC branch too
(else nothing wrote it: ai_parkjoy_q27, ai_crowdrun832_q27); IBC with LFNST / NSPT and RR-IBC flips with sign prediction
stay the CPU's; IBC / palette CUs take no inter_mc job (their refIdx is not a reference picture's); chroma reordering's
NN rule (an NN block derives the DIMD of its prediction for the DM mode) only with DIMD in the SPS, as M86 has it
(ra_off4_q32).
M96 (2026-10-22): bi-prediction on the GPU (inter_mc.comp's MC_BI): bi-prediction CUs without DMVR or BDOF and without
other tools (xPredInterBi, then xWeightedAverage's addWeightedAvg). A bi job carries list 1's vector (clipped as list
0's) and slot, BCW's weights (getBcwWeight), which lists are out of boundary somewhere (isMvOOB at the unclipped vectors)
with the unclipped vectors, the PU's luma position and the picture's size. Each list is predicted at the 14-bit
intermediate precision (rndRes off: the last stage's int16 sum >> IF_FILTER_PREC), with JVET-AI0094's sharp luma
filter (m_lumaFilter12bi) where the pass's block is above 4 x 4 (filterHor / filterVer's width > 4 && height > 4; the
horizontal pass of a 2-D block covers H + 11 rows, so there W > 4 alone), then averaged with the weights; JVET-Z0136:
a sample whose reference lies more than half a sample outside the picture in one list (per luma sample, chroma at
twice its position) takes the other list alone, rounded from the intermediate. Bi-prediction of one reference and one
vector (xCheckIdenticalMotion) is list 0's uni-prediction. The host rejects DMVR (bdmvrRefine, mvRefine with
checkDMVRCondition) and BDOF; found on the way (ra_crowdrun_q32): BDOF_RM_CONSTRAINTS removes BDOF's 8 x 8 / 128-sample
limit, so a 4 x 4 BDOF CU had passed. Results (50 streams, build gpu35): every picture identical; 79 bi-prediction CUs
predicted by the GPU (uni-prediction 585 as before); 56 bi-prediction CUs remain the CPU's (DMVR or BDOF).
M97 (2026-10-22): BDOF without DMVR on the GPU (xPredInterBi's bioApplied: BDOF's switches, equal distances in
different directions or JVET-AJ0097's same direction with isMergeIndexBDOFCondition, no SMVD, BCW's default weight).
inter_mc.comp's first dispatch (mode 1) refines each luma sub-block of the PU - min(16, W) x min(16, H), JVET-AG0067's
scaleBDOF 4 - into a PU-sized luma prediction (mcBdofBuf), which the luma jobs of its blocks copy (MC_BDOF); its chroma
blocks are bi-prediction jobs. Per sub-block one workgroup with a shared-memory tile: both lists' 14-bit predictions
over the sub-block grown by 3 (xPredInterBlk with bioApplied: the PU grown by 2, the sharp filter as that block is
above 4 x 4, the vectors clipped at the position grown by BIO_EXTEND_SIZE + 1 - chroma's too, xPredInterBiSubPuBDOF -,
the ring beyond the grown PU a replication, AF0159's padding), the SAD of dI against bioBlockDistTh ((bioDx bioDy) <<
(2 + (init QP >> 5))): below it addAvg, else subBlockBiOptFlow per sample - calcBIOParamSum5NOSIM's 5 x 5 weighted
sums of the high-precision parameters with JVET-AG0067's mean removal (not under JVET-AL0081's noMeanRm4Sample),
divTable's division into vx / vy clipped to +-31, addBIOAvgN at the sample moved by one where |v| > 16 (JVET-AI0046),
or without the move where the sub-block has a sample out of boundary in either list, whose samples take one list alone
(JVET-Z0136). The CPU's SIMD kernels (addAvg_SSE, addBIOAvgN_SSE, calcBIOParamSum5NOSIMCore_SSE, gradFilter_SSE) were
checked against the C reference the shader follows. Partial verification's totals now name bi-prediction with BDOF /
DMVR. Results (50 streams, build gpu37): every picture identical; 6 CUs of bi-prediction with BDOF predicted by the
GPU (all the corpus has); bi-prediction with DMVR 50 CUs from the CPU.
M98 (2026-10-22): multi-pass DMVR on the GPU (bdmvrRefine CUs; VVC's DMVR, mvRefine, stays rejected). The refined
vectors per 16 x 16 sub-PU are the CPU's (DecCu hands InterPrediction's m_bdmvrSubPuMvBuf over after the CU's
prediction, dmvrVectors - motion derivation stays the CPU's); the prediction is inter_mc.comp's mode 2, one workgroup per
PU replaying xPredInterBi's DMVR branch and motionCompensation's xPredInterBiSubPuBDOF in their order into a PU-sized
prediction (luma and chroma, mcBdofBuf) that all the PU's blocks copy. Without BDOF (xPredInterBiBDMVR's bioApplied,
which lacks mcBdof's merge-index condition): per run of sub-PUs of one vector pair - rows, or columns where H > W -
bi-prediction at the run's size with BCW's weights. With BDOF: unless JVET-AL0081's same direction or JVET-AG0067's
small PU (area < 512 or a side of 4) skips it, a first pass per run computes each BDOF block's vector offset
(applyBiOptFlow with isBdofMvRefine: min(16, W) x min(16, H) blocks, the SAD of dI against bioBlockDistTh, else
calcBIOParamSum4HighPrecision's weighted sums - g_bdofWeight's tables as calcBIOParamSumHighPrecision_SSE reads them,
JVET-AI0046's own 12 x 12 weights among them, appended to inter_mc's filters -, the mean removal by getMean unless the
same direction, the 2 x 2 solve in 64 bits with divTable and x86's 32-bit offset shift, rounded and halved); then
xPredInterBiBDMVR2's two iterations over runs of equal offsets (16-wide, then 8 / 4) predict again at the refined
vectors, write the luma of the blocks whose SAD is small (addAvg) or whose new offset is 0 (subBlockBiOptFlow), and
accumulate the others; xPredInterBiSubPuBDOF finally predicts the luma of the blocks still moving at their final
vectors with subBlockBiOptFlow everywhere and the chroma of all (bi-prediction clipped at the grown position). The
tiles live in the PU's scratch (coherent buffer references, barriers between the steps); the offsets are per 4 x 4
unit as m_bdofSubPuMvOffset / m_bdofSubPuMvOffse2. clipMvInl's bounds come with the job (subpicture clipping is
rejected). Results (50 streams, build gpu38): every picture identical; bi-prediction with DMVR 50 CUs
predicted by the GPU (all the corpus has: no bi-prediction CU remains the CPU's).
M99 (2026-10-22): OBMC on the GPU for CUs whose base prediction the GPU makes (kind "OBMC" without sub-block motion;
DecCu's subBlockOBMC). The host lays out each boundary's segments of one neighbouring motion with the CPU's own
PU::getSameNeigMotion (partial verification: the CPU's motion), the JVET-AD0193 screen-content exit (isSCC, mirrored),
the template availability (xAMLIsTopTempAvailable / Left), and writes a record per CU (Params::timdData, OBMC_REC_*;
Tu::obmc, new, marks the blocks on the CU's top / left band). inter_mc.comp's mode 3 (after the blocks' jobs) does per
segment what does not read the current picture: the band's prediction with the neighbour's motion (bi-prediction with
the neighbour's BCW weight and out-of-boundary samples, the CU's half-sample flag; mcSample, the blocks' per-sample
prediction refactored out of mode 0), skipObmcConditionByPixel per component (each list of the neighbour's motion at
its integer vector against the CU's prediction there, threshold 96 << (bd - 8)), and with JVET-Z0061's TM OBMC the
one-sample templates beside the segment predicted with the CU's and the neighbour's motion through the 4-tap TM filter
(xPredInterBlk's fastOBMC: filterIdx 1 with use4TapTM, g_aiExtIntraCubicFilter appended to the filters); its data go
after the blocks' samples in Params::interPred. intra_rec then, in the dependency chain (intra_deps adds the owners of
the CU's one-sample templates), chooses each segment's mode as selectOBMCmode does - the SADs of the reconstructed
template (inverse-mapped with LMCS) against the current motion's, the neighbour's and their 26/6 blend, without
DISTORTION_PRECISION_ADJUSTMENT as FULL_NBIT is set - and blends the band's samples in place in Params::interPred
(weightObmcBoundary's 26/6, 28/4, 30/2, 31/1 from the mode's first row; chroma one row; the top segments, then the
left ones over their result; the SIMD kernel's per-row weights are the same fractions), before the inter LFNST's DIMD,
sign prediction and LMCS read the prediction (coherent reads). The verifier's mirror of the scheduling data adds the
same template owners. Found on the way (ra_foreman_q22): RdCost's SAD keeps all bits here, a shift by bd - 8 chose
other modes. Still the CPU's, by reason (partial.py's "OBMC (...)"): OBMC of sub-block motion (DMVR CUs: inner
sub-block OBMC), LIC / non-local LIC neighbours, intra neighbours with DIMD (JVET-AJ0161's intra OBMC), IBC
neighbours. Results (50 streams, build gpu42): every picture identical; OBMC 3000 CUs predicted by the GPU; from the
CPU: OBMC (its base prediction - the DMVR CUs, whose inner sub-block OBMC M100 adds) 3083, OBMC with a LIC neighbour
481, with an intra neighbour 327, with an IBC neighbour 4.
M100 (2026-10-23): OBMC of DMVR CUs on the GPU (subBlockOBMC's bSubMotion: the boundaries at the CU's stored sub-PU
motion, then the inner sub-blocks). The host lays out per 4 x 4 inner sub-block the neighbours inside the CU whose
motion differs (getNeighborMotion with JVET-AL0081's threshold: half a sample for bi-prediction from one direction at
unequal distances, else 1) into inner jobs (OBMC_I_*: one MC job per neighbour); inter_mc.comp's mode 4 predicts them
after the blocks' jobs, and intra_rec blends them in place after the boundaries' segments (xSubblockOBMCBlending: luma
per position with currWeights for the set of differing neighbours and neigWeights per neighbour, in 1/128; chroma's
2 x 2 with the corner's two neighbours at 27/128 each). Results with M101.
M101 (2026-10-23): OBMC with LIC and non-local LIC neighbours on the GPU. A LIC neighbour's band and its templates take
the neighbour's stored models (JVET-AD0213: per list and component, a list it lacks 32 / 0; uni-prediction from list 1
reads list 1's - MC_L1); bi-prediction applies them to each list's 14-bit intermediate as xLicCompAdj's OBMC branch
does (mcLicBi); a non-local LIC neighbour's model (JVET-AG0276, secAltLMParaUnit) goes on the band's final prediction.
skipObmcConditionByPixel uses the model the CPU's band prediction left in m_iRefListIdx (list 1 after bi-prediction,
else its one list) for both lists. Results (50 streams, build gpu47, M100 and M101): every picture identical; OBMC
6353 CUs predicted by the GPU; from the CPU: OBMC with an intra neighbour 418, its base prediction 120, with an IBC
neighbour 4.
M102 (2026-10-23): affine prediction on the GPU (xPredAffineBlk per list, then xPredInterBi's average), with OBMC at
the boundaries (JVET-AL0079: affine without affine BDOF has no inner sub-block OBMC). The host writes a record per CU
(inter_mc.comp's AF_*, Params' mcAffineBuf) as xPredAffineBlk derives it per list and component class: the base vector
and deltas of the control points (4- / 6-parameter), the adaptive sub-block size (JVET-AC0158: from 1 up while the
scaled delta stays below the threshold, a quarter for luma; pixel blocks where a side stays below 4), PROF (luma
sub-blocks of 4 x 4 and up, unless the picture header disables it or the motion is translational),
xCheckIdenticalMotion, BCW, and the stored luma motion per 4 x 4 unit that chroma's sub-blocks average. A block's job
with MC_AFFINE predicts per sample: its sub-block's vector (roundAffineMv to 1/16, clipToStorageBitDepth, clipMvInl;
luma at 1/64 through m_lumaFilter64 - JVET-AJ0158, appended to the filters - where the integer parts agree), PROF's
refinement of the intermediate by the per-position offsets and 3-tap gradients (the ring around the sub-block the
integer samples), bi-prediction's average with BCW and per-sub-block out-of-boundary samples. Still the CPU's, by
reason (partial.py's "affine (...)"): affine BDOF and its inner OBMC, LIC / non-local LIC, sub-block merge, CIIP
(geneWeightedCIIPAffinePred: found on the way, ra_foreman_q22 - the CIIP blend runs after OBMC), DMVR, intra / IBC
neighbours of OBMC. Results (50 streams, build gpu52): every picture identical (ra_wpp_q32 skipped as before,
wavefronts); affine 1028 CUs predicted by the GPU; from the CPU: affine (sub-block merge) 810, (with LIC) 663, (OBMC
of sub-block motion: affine BDOF) 663, (CIIP) 197, (OBMC with an intra neighbour) 132, (other: affineGpu's remaining
conditions) 97, (OBMC with an IBC neighbour) 4.
M103 (2026-10-23): LIC CUs on the GPU (kind "LIC" without non-local LIC, BDOF, DMVR or reference picture resampling).
The CPU keeps the models it applied on the CU (DecCu's setLicParam after a derivation, JVET-AH0314's inherited ones
otherwise; the shift is always 5, xUpdateLicModel's slope adjustment keeps it), so the host appends them to the affine
records' buffer and a block job with MC_LIC applies them in mcSample: uni-prediction on the final samples (list 1's
model for list 1, MC_L1), bi-prediction per list on the 14-bit intermediate as xLicCompAdj does (mcLicBi), before
BCW's average and the out-of-boundary samples. With OBMC: selectOBMCmode predicts the template of the CU's own motion
while the CU still holds its own LIC flag and models (the neighbour's replace them for the neighbour's template and
band only), so an OBMC job of a LIC CU carries the CU's models too (OBMC_J_CLIC, OBMC_CLIC); found on the way
(ra_foreman_q22, ra_parkjoy832_q32): without them the TM mode differed. Partial verification names why a LIC CU stays
the CPU's. Results (50 streams, build gpu53): every picture identical (ra_wpp_q32 skipped as before); LIC 539 CUs
predicted by the GPU; from the CPU: LIC (OBMC with an intra neighbour) 149, (with DMVR) 87, (resampling) 42, (with
BDOF) 33, (other) 23, (OBMC with an IBC neighbour) 1.
M104 (2026-10-23): affine CUs with LIC on the GPU. xPredAffineBlk derives a uni-prediction's model from its
sub-blocks' templates and applies it to the final prediction, bi-prediction's go through xLicCompAdj per list on the
intermediate; the CU keeps them (setLicParam), so the affine record points at the CU's 12 models (AF_LIC) and afSample
applies them as M103's mcLic / mcLicBi do. An inherited model (licInheritPara) stays the CPU's: affine uni-prediction
derives anew while the CU keeps the inherited one. Results (50 streams, build gpu54): every picture identical
(ai_foreman352_q22 lost the device in the parallel run and its solo rerun - also with gpu53, M103 - and passed in two
later runs alone: a device loss on that all-intra stream that does not depend on the inter work, to look at with the
dispatch sizes; ra_wpp_q32 skipped as before); affine 1408 CUs predicted by the GPU (+380); from the CPU: affine
(sub-block merge) 810, (OBMC of sub-block motion) 663, (OBMC with an intra neighbour) 209, (CIIP) 197, (with non-local
LIC) 150, (other) 105, (with inherited LIC or resampling) 48, (OBMC with an IBC neighbour) 4.
M105 (2026-10-23): non-local LIC CUs on the GPU (JVET-AG0276, kind "non-local LIC" and affine ones; not with BDOF,
DMVR or resampling). xPredInterBi applies the CU's altLMParaUnit to the final prediction (all components, shift 5,
clipped); after the CU's reconstruction DecCu moves that model to secAltLMParaUnit and derives another into
altLMParaUnit for its merge neighbours (xDevSecLicPara), so the host, which records the CU afterwards, takes
secAltLMParaUnit: found on the way (ra_parkjoy832_q32: altLMParaUnit gave other pictures). A block job with MC_NLIC
points at the model (MC_NLIC_AT) and mcSample, now mcSampleBase plus the model, applies it - also to OBMC's view of
the CU's prediction; selectOBMCmode's template of the CU's own motion takes it too (OBMC_J_CNLIC). Results (50
streams, build gpu55): every picture identical (ra_wpp_q32 skipped as before); non-local LIC 113 CUs predicted by the
GPU, affine 1490 (+82); from the CPU: non-local LIC (with BDOF) 17, (OBMC with an intra neighbour) 12, (other) 3,
(resampling) 2.
Sign prediction of inter blocks above 32 x 32 (2026-10-22, agent lf; partial verification): an inter block predicts
signs up to SIGN_PRED_MAX_BS_INTER (128 x 128), intra_rec's signPrediction keeps 32 x 32 in shared memory. A block wider
or taller than 32 takes signPredictionLarge, whose arrays are in the workgroup's scratch (SPL_*: the prediction, which
storeRecon keeps there instead of s_spPred, the border, the base residual, the templates, the halves; their values cut
to the CPU's Pel as the shared arrays' int16 cut them) and which computes the first transform stage in chunks of rows
that fit s_spStage1 (as inChainResidual): the residual with the decided signs (without LFNST / NSPT the negated levels
dequantized on the fly) and, with LFNST / NSPT (DERIVE_TRANSFORM: their mode a DIMD of the prediction), the base border
and the templates (signPredDataLarge). The intra path is unchanged: blocks up to 32 x 32 take signPrediction as before
(storeRecon has one more branch, taken only where sign prediction keeps the prediction).
ASBT on the GPU (2026-10-23, agent lf; partial verification): JVET-AL0181's inter blocks code their residual at half the
width or height (TransformUnit::blocksResidual) and place it as a sub-block at the position the prediction's gradient
energy chose (InterPrediction::asbtCoordPred / asbtCoord; upsampleOneBlock(H): the other samples' residual 0).
recordAsbtBlock records such a block as up to three records: the parts before and after the sub-block (prediction only,
no residual) and the sub-block (its residual as a block of the residual's size: transforms, dequantization, sign
prediction as other inter blocks'), which waits for every block before it (DERIVE_PREFIX: the units it shares with the
other parts are its own in the owner maps). The sub-block's position and direction are the CPU's (tu.asbtPosx /
asbtPosy, chroma luma's halved: derived from the prediction, as the vectors are, so taken like them in partial
verification), its levels in the CPU's layout after invTransposeTransform. Sign prediction's border of a sub-block is
asbtBorderResi's / asbtBorderResiH's (SP_ASBT in its signPred data with the TU block's position, the direction, the
sub-block's position and the offset of the part before's prediction; intra_rec's spPredicted / spUse in both
signPrediction paths): on the side inside the block twice the prediction next to the sub-block minus the one beyond it
(with the reconstruction next to the block where the sub-block is one sample in), bitDepth - 1 at the picture's edge
(the CPU passes bitDepth - 1, not the mid level), useLeft / useTop by the TU block's position. prepareSignPred takes the
levels in blocksResidual's layout; chroma residual scaling's size test takes the TU block's size (a 2 x 2 sub-block of a
larger block is scaled: ldp_crowdrun_q27). Still the CPU's: ASBT of an IBC CU, decimated both ways, with joint CbCr,
with LFNST / NSPT, other chroma formats (none in the 50 streams). Results: "ASBT" 101 -> 0, every verified picture
identical (builds aglft9 and, with the scaling's size test, aglft10).
Inter CCCM on the GPU (2026-10-23, agent lf; partial verification): a chroma block of a TU with interCccm
(DERIVE_INTER_CCCM, 1 << 31, with DERIVE_INTER; its record at Tu::ibc: the TU's luma block, its prediction's offset in
Params::interPred, LMCS's forward mapping, the SPS's collocated chroma) predicts in intra_rec (intra_intercccm.glsl) as
InterPrediction::deriveInterCccmPrediction does: a model of 8 parameters (the six luma taps around the co-located luma
sample, five with collocated chroma, less the luma offset; the nonlinear term of the mean of the sample and the one
below it; the bias) fitted by intra_cccm.glsl's cccmSolve in solve2's inter mode (28 matrix bits up to 10 bits,
regularization 1: g_cccmInter) to the TU's luma prediction and the block's own chroma prediction, each less its offset
(computeOffset: the four corners' rounded mean), on the grid the chroma size subsamples; each sample then (3 x the model
applied to the TU's luma reconstruction + the inter prediction + 2) >> 2, clipped. DecCu hands over the inter chroma
prediction before the CPU blends it in place (interCccmPred); the block waits for every block before it (DERIVE_PREFIX:
the luma reconstruction). A CCP merge candidate of inter CCCM (CCP_TYPE_INTER_CCCM: an inter CU's TU model, or a
temporal one) predicts with xInterCccmApplyModelOffset's features (intra_cccm.glsl's g_cccmInterNs: the six taps of the
luma without subsampling, the nonlinear term, the bias) clipped to ClipPel's range alone; with merge adjustment or
decoder-derived fusion it stays the CPU's. Results (build aglft9, all 50 streams but ldp_crowdrun_q27's ASBT difference,
fixed before): "inter CCCM" 18 -> 0 (1 now "inter CCCM with ASBT": its luma prediction lies in several records), "CCP
merge with an inter CCCM candidate" 14 -> 0 (4 now "CCP merge adjustment or decoder-derived fusion with an inter CCCM
candidate"). Inter CCP merge (161 CUs) is still the CPU's.
ON mode with ALF (2026-10-22, agent st): with ALF on (JVET-AC0162, ALF's residual input) the ON mode output differed
from the CPU's on intra pictures the GPU reconstructed (test streams of runs/agents/st/enc encoded with the CTC intra
cfg, the tools still gated off: bis_alf with ALF / CCALF / ALF-CCCM alone differs, deblocking, SAO / CCSAO and BIF
alone are identical; with --GpuLoopFilter=0 and =3 alike). Cause: AdaptiveLoopFilter::copyResiData copies the
picture's luma residual buffer (cs.getResiBuf()), which DecCu writes as each block's piResi; DecCu's derive-only pass
returns before the residual, so ALF read a stale buffer (the previous picture's). Change: in ON mode with ALF in the
SPS intra_rec writes each luma sample's residual - the value it reconstructs with, piResi's - into a picture-sized
int16 plane (Params::resiPlane, new at the end of Params: 504 bytes), downloaded into the picture's residual buffer
with the reconstruction. Verification and partial verification are unchanged (no plane). Results (agst22 without,
agst23 with the change, --GpuRecon=3): bis_alf differed with --GpuLoopFilter=0 and =3, now identical with both;
cr_fore32, cr_park27, nocr_fore32 (all in-loop filters of the CTC cfg) identical; partial verification of 8 streams
identical.
ON mode's CPU fallback after a derive-only pass (2026-10-22, agent st): a picture that record() rejects in ON mode is
reconstructed by the CPU in a second pass over the parsed slice after the derive-only pass (DecSlice). That second pass
did not decode as without the first: with every gate of supported() lifted (a test build), streams whose pictures fell
back differed from the CPU decode, and so did the test streams of runs/agents/st/enc with a forced fallback. Causes,
each reproduced and removed: (1) the derive-only pass of chroma reordering (de070984c / befb9754e) wrote its placeholder
over pu.intraDir[1], the chroma mode as parsed (DM, DIMD, DBV or a candidate's index), which the second pass reads again
(DecCu::m_gpuParsedChroma keeps and restores it); (2) the derive-only pass marked the whole slice decoded (setDecomp),
so the second pass's availability, IntraTMP's and IBC's search ranges and IBC's vector checks saw undecoded areas
(cleared before it); (3) reconstructParsed did not reset the motion LUTs as the parsing loop does (it assumed no IBC),
and the second pass found the first's IntraTMP / IBC vectors in them (each pass starts from the slice's LUTs and resets
them where the loop does); (4) the rest of a derive-only pass after the GPU path gave up changed parsed state (IBC's
vectors in xDeriveCUMV; decompressCtu now stops there). The derive-only pass also ran into the CPU's checks on values it
derives from samples it has not reconstructed, once the gates of BVG-DIMD, MDIP, IntraTMP / IBC, EIP / SGPM, PDP and NN
are lifted: a BVG-DIMD or intra merge CU without a mode (255; MDIP's isIdxModeValid, getFinalIntraModeForTransform), the
DIMD modes of predictions it does not compute (EIP, IntraTMP, NN, MIP, regression SGPM, a DIMD with a vector, BVG-DIMD:
getTrTypes, getFinalIntraModeForTransform in the host's choice of the transforms), BVG-CCCM's candidates ("No valid BV
found"), the TIMD merge transform types getTrTypes derives after xReconIntraQT; now placeholders in range (the GPU
derives all of them), and an IntraTMP CU spans its motion (list 0, the CPU's vector 0) so that the later CUs' OBMC
records read its kind as the CPU's pass would (the dump comparison found an OBMC record that took a GPU IntraTMP
neighbour for an intra one). A check that still fails in a derive-only pass makes the picture the CPU's ("a check of the
CPU's derivation (function)"), not the decode's end. The ON summary lists the CPU's pictures by reason ("[gpu-intra]
pictures by the CPU"); ECM_GPU_DUMP writes the CU data's section sizes, dumpdiff.py (Tu of 105 words now) names the
section that differs. Results (Intel PC, agst41, --GpuRecon=3): the 50 streams identical with the CPU (the gates still
in place: all pictures the CPU's); the test streams cr_fore32, cr_park27, cr_crowd32, bis_alf identical, all pictures on
the GPU; with a debug switch forcing the fallback after a complete derive-only pass (agst33 before, agst34 after the fix
of (1)) cr_fore32, cr_park27, bis_alf went from different (chroma) to identical. With every gate lifted (test build
agst40): 37 of the 50 streams identical; the 13 others reconstruct their intra picture on the GPU, but their inter
pictures differ (ld_foreman_q32: pictures 0 and 1 identical, 2 on different): they read side information of the intra
picture that the GPU did not write back to the CPU (next paragraph).
ON mode's side information (2026-10-22, agent st): an intra picture the GPU reconstructs in ON mode leaves in the CPU's
CodingStructure what the CPU's derive-only pass derived from samples it had not reconstructed, and the later inter
pictures read four kinds of it from their reference pictures: the luma intra modes (IPM: spanIpmInfoInter reads the
reference unit's; an intra CU's from spanIpmInfoIntra / spanIpmInfoSgpm, a CU with a block vector's from the unit it
references, spanIpmInfoIBC), the motion of the CUs with a vector (IntraTMP, BVG-DIMD, EIP with a vector), the CCP models
(CS::saveTemporalCcpModel stores each intra PU's curCand for temporal CCP merge) and the EIP models
(CS::saveTemporalEipModel). With the gates lifted (agst40) 13 streams reconstructed their intra picture on the GPU and
then differed in their inter pictures (ld_foreman_q32 from picture 2 on). Change: sideInformation() takes them from the
GPU's outputs - the CUs' modes (cuMode: the mode the MPM lists read, which is the IPM for MPM, DIMD, OBIC, intra merge,
TIMD and TMRL CUs), their DimdOut (an SGPM CU's split and part modes, an IntraTMP CU's mv[0] and list - bv is the list's
entry its index, of fusion the fused set's first, selects: mv[0] may be a sub-sample vector -, a BVG-DIMD CU's first
vector, an EIP CU's pu.bv and its DIMD), the EIP models and the CCP records (cpuCcpWords' fields, the others kept;
CCCM's and GLM's models with the bit depth and mid value of CccmModel) - and writes them in decoding order with the
CPU's own spanning (spanMotionInfo for a vector, spanIpmInfoSgpm, the IPM buffer, curCand, eipModel). In verification of
an intra picture it compares them with the CPU's instead ("ON mode's side information: ..."), so that partial.py checks
what ON mode writes: the first versions differed in curCand's numParams (CCLM's 2) and the second model's ranges of one
model (fields the GPU does not have, now kept), in an IntraTMP CU's bv (sub-sample) and in eipDimdMode (the GPU's is the
block's, in DimdOut). ON mode now downloads the CU modes, DimdOut and the CCP records too. Results (Intel PC, agst44 /
agst45): partial verification of the 50 streams with the side information: 49 identical, ra_off4_q32 differs in its
intra picture by an NN CU of an SPS without DIMD (befb9754e, next paragraph); with the gates lifted (agst45,
--GpuRecon=3) the 15 streams that had a GPU picture identical: ld_foreman_q32, ldp_foreman_q32, ldp_wp_q32,
ld_parkjoy_q27, ra_10bit_q32, ra_dbkparams_q32, ra_internal8_q32, ra_nodbk_q32, ra_nolmcs_q32, ra_nosao_q32,
ra_off3_q32, ra_off5_q32, ra_long_q32 (3 GPU pictures), ai_crowdrun_q42, ai_fast_q32.
ON mode without supported()'s BVG-DIMD gate (2026-10-22, agent st): the sequence-level rejection of BVG-DIMD is gone;
BVG-DIMD CUs (CU_BVG since M65) and the BVG-CCCM blocks record() rejects ("BVG-CCCM") are the GPU's per CU, what the GPU
lacks rejected by record() (the picture the CPU's, through 44e11ef88's fallback; its side information through
7882788d3). Results (Intel PC, agst46, --GpuRecon=3 against the build's CPU decode, the 50 streams): 50 of 50 identical;
intra pictures on the GPU 0 -> 0 of 331 pictures; the CPU's intra pictures by their first reason: MDIP / MPM sorting 51;
IntraTMP / IBC 3; wavefronts 1.
ON mode without supported()'s MDIP / MPM sorting gate (2026-10-22, agent st): the sequence-level rejection of MDIP / MPM
sorting is gone; MDIP and MPM sorting (the MPM records of M57, derived by the CUs' nodes) are the GPU's per CU, what the
GPU lacks rejected by record() (the picture the CPU's, through 44e11ef88's fallback; its side information through
7882788d3). Results (Intel PC, agst47, --GpuRecon=3 against the build's CPU decode, the 50 streams): 50 of 50 identical;
intra pictures on the GPU 0 -> 0 of 331 pictures; the CPU's intra pictures by their first reason: IntraTMP / IBC 54;
wavefronts 1.
ON mode without supported()'s IntraTMP / IBC gate (2026-10-22, agent st): the sequence-level rejection of IntraTMP / IBC
is gone; IntraTMP CUs (CU_TMP, M62-M74; record() rejects LIC, fusion, FLM, OBMC cases the GPU lacks) and IBC CUs, which
ON mode takes per CU as every non-intra CU ("inter / IBC / palette CU": their vectors and prediction are the CPU's,
which the derive-only pass does not compute) are the GPU's per CU, what the GPU lacks rejected by record() (the picture
the CPU's, through 44e11ef88's fallback; its side information through 7882788d3). Results (Intel PC, agst48,
--GpuRecon=3 against the build's CPU decode, the 50 streams): 50 of 50 identical; intra pictures on the GPU 0 -> 0 of
331 pictures; the CPU's intra pictures by their first reason: EIP / SGPM 52; CCLM with LFNST / NSPT without DIMD 2;
wavefronts 1.
ON mode without supported()'s EIP / SGPM gate (2026-10-22, agent st): the sequence-level rejection of EIP / SGPM is
gone; EIP CUs (CU_EIP) and SGPM CUs (CU_SGPM; record() rejects regression SGPM and SGPM above 1024 samples) are the
GPU's per CU, what the GPU lacks rejected by record() (the picture the CPU's, through 44e11ef88's fallback; its side
information through 7882788d3). Results (Intel PC, agst49, --GpuRecon=3 against the build's CPU decode, the 50 streams):
50 of 50 identical; intra pictures on the GPU 0 -> 1 of 331 pictures; the CPU's intra pictures by their first reason:
PDP 47; NN intra prediction 4; CCLM with LFNST / NSPT without DIMD 2; wavefronts 1.
ON mode without supported()'s PDP gate (2026-10-22, agent st): the sequence-level rejection of PDP is gone; PDP (its MPM
lists and filters on the GPU; record() rejects what it lacks) are the GPU's per CU, what the GPU lacks rejected by
record() (the picture the CPU's, through 44e11ef88's fallback; its side information through 7882788d3). Results (Intel
PC, agst50, --GpuRecon=3 against the build's CPU decode, the 50 streams): 50 of 50 identical; intra pictures on the GPU
1 -> 1 of 331 pictures; the CPU's intra pictures by their first reason: NN intra prediction 51; CCLM with LFNST / NSPT
without DIMD 2; wavefronts 1.
ON mode without supported()'s NN intra prediction gate (2026-10-22, agent st): the sequence-level rejection of NN intra
prediction is gone; NN intra prediction (intra_nn.glsl; record() rejects the shapes without a model) are the GPU's per
CU, what the GPU lacks rejected by record() (the picture the CPU's, through 44e11ef88's fallback; its side information
through 7882788d3). Results (Intel PC, agst51, --GpuRecon=3 against the build's CPU decode, the 50 streams): 50 of 50
identical; intra pictures on the GPU 1 -> 17 of 331 pictures; the CPU's intra pictures by their first reason: inter /
IBC / palette CU 16; DBV with LM chroma fusion 10; CCLM with LFNST / NSPT without DIMD 2; BVG-CCCM 2; IntraTMP OBMC with
a neighbour of two vectors or inter 2; IntraTMP above 1024 samples 1; IntraTMP fusion / FLM above 10 bits 1; DBV with LM
chroma fusion, LIC or a flip 1.
Chroma reordering over an NN CU without DIMD (2026-10-22, agent st): partial verification of ra_off4_q32 (an SPS without
DIMD) differed in its intra picture: indicesRepresentationPnn of an NN CU, CPU planar, GPU 63, and from there a chroma
reordering CU's DM. befb9754e (chroma reordering) has every NN block of a dual tree with reordering derive the DIMD of
its prediction (DERIVE_PNN_DIMD) for that DM, but without DIMD in the SPS xPredIntraPnn's deriveIpmForTransform returns
planar at once (M86: the host sets planar and DC). Now no such derivation without DIMD, and the reordering CU takes the
host's mode then. Results (agst53): partial verification of ra_off4_q32, ra_off15_q32, ai_off15_q27, ra_off1_q32
identical; ON mode (--GpuRecon=3) on the 50 streams identical, 17 intra pictures on the GPU.
Integration of the agents' work after M105 (2026-10-23): lf's inter sign prediction above 32 x 32 (1a9d9d19b), ASBT
(86153b5cc) and inter CCCM (9886461dd), st's ON-mode commits 9c7de2340 .. 2de8ee5a9 (ALF's residual plane, the CPU's
pass after a derive-only pass, an intra picture's side information, the six gates removed, the chroma reordering DM
fix without DIMD; st's chroma reordering commit itself was already in as M95, f36a1b1ef) cherry-picked onto M105.
Partial verification (50 streams, build gpu56): 44 identical, ra_wpp_q32 skipped, five differed at inter CCCM chroma
blocks (ld_parkjoy_q27, ldp_crowdrun_q27, ra_foreman_q22, ra_parkjoy832_q32, ra_parkjoy_q37): intra_rec's inter CCCM
branch blends the block's prediction before the inter branch applies OBMC to it, which on lf's base was the CPU's
final prediction and with M96-M105 is the GPU's without OBMC yet. 3af02cfde: an inter CCCM CU keeps the CPU's
prediction (interMcGpu); with it those streams are identical (build gpu58). ON mode (--GpuRecon=3, md5 against the
build's CPU decoding, build gpu56): 50 of 50 identical, 17 intra pictures reconstructed on the GPU.
M106 (2026-10-23): CIIP on the GPU (not CIIP-PDPC, CIIP with LFNST / NSPT, LIC or inter CCCM, CUs of several transform
units; affine CIIP stays rejected). The inter prediction is the kind beneath CIIP (interKindBase) made by inter_mc,
with what CIIP changes in it: no BDOF (xPredInterBi's bioApplied) and the plain average whatever the BCW index
(xWeightedAverage without LIC) - found on the way (ra_foreman_q22: predictions off by 1-3 at angular CIIP CUs); OBMC
applies in intra_rec as for any CU. The intra prediction is the CPU's parameters of geneIntrainterPred's predIntraAng
(IntraPrediction::ciipSnapshot, new, per component; partial verification: the CPU's mode, CIIP-TIMD's included) put in
the inter block's Tu (ciipFields: the mode, its angle, filters, PDPC, fusion, PDP); Tu::ciip (new) carries
geneWeightedPred's blend: the neighbours' weights (the PUs left of the bottom-left and above the top-right sample,
intra or not) or TIMD's angular weights 6, 5, 3, 2 / 8 by quarter of the columns or rows, and LMCS's forward mapping
of the inter prediction. intra_rec: a CIIP block applies OBMC in the inter branch, then goes on to the intra path,
whose storeRecon blends (ciipBlend) before the residual (as an inter block's); intra_deps and the host's mirror give
it the intra prediction's references besides the inter block's dependencies. Results (50 streams, build gpu59): every
picture identical (ra_wpp_q32 skipped as before); CIIP 99 CUs predicted by the GPU; from the CPU: CIIP-PDPC 231, with
LFNST / NSPT 53, with LIC 32, OBMC with an intra neighbour 20, OBMC 8, with inter CCCM 1.
M107 (2026-10-23): GPM of two uni-prediction inter parts on the GPU (motionCompensationGeo's plain path:
weightedGeoBlk). CommonLib records what motionCompensationGeo did - the plain blend taken, split direction, blending
index, each part's motion, its LIC flag and models as its merge candidate set them, its non-local LIC flag
(InterPrediction::m_gpuGeoPlain / m_gpuGeoPart, gpuGeoPart; unconditional and at the end of the class: CommonLib is
built without ECM_VULKAN, so code or members under it there vanish or give the class two layouts - a first version
did, xSubPuMC faulted) - and DecCu hands it over (gpmParts). A GPM CU's block jobs carry MC_GPM, both parts' motion
and the component's weights (gpmWeights: xWeightedGeoBlk's walk of g_geoWeights); mcSampleBase predicts each part at
the 14-bit intermediate, applies a part's inherited LIC model where it is not the identity (the geo branch of
xPredInterBlk; found on the way, ld_parkjoy_q27: the CU no longer shows its parts' LIC after the prediction, and the
CPU skips an identity model, whose round trip rounds), and blends (w p0 + (32 - w) p1 + offset) >> shift; OBMC of the
CU follows as for any CU. Still the CPU's (gpmWhy): separate OBMC per part (JVET-AK0212, every slice but low-delay B),
regression-based blending, intra / IBC / affine / bi-prediction / DMVR parts, LIC of the CU, non-local LIC parts.
Results (50 streams, build gpm1): every picture identical (ra_wpp_q32 skipped as before); GPM 50 CUs predicted by the
GPU; from the CPU: GPM (separate OBMC) 761, (an affine part) 610, (an intra part) 449, (regression-based blending)
314, (a bi-prediction part) 45, (LIC) 18, (its OBMC) 9.
M108 (2026-10-23): CIIP-PDPC on the GPU (planar / DC CIIP CUs whose slice clips to the bit depth). predIntraAng with
ciipPDPC replaces the intra prediction by PDPC's term of the references alone - (wL left + wT top + 32) >> 6,
xIntraPredPlanarDcPdpc - and skips PDP; geneWeightedPred blends ((intra << 6) + (64 - wT - wL) inter + 32) >> 6,
clipped. Tu::ciip's new kind CIIP_EDGE: intra_rec's planar / DC branch computes the term, ciipBlend the blend;
ciipFields sets no PDP for it (found on the way, ld_foreman_q32 and three more: with PDP the prediction differed).
Results (50 streams, build gpm2, with M107): every picture identical (ra_wpp_q32 skipped as before; ai_foreman352_q22
lost the device in the parallel run and passed alone); CIIP 200 CUs predicted by the GPU (+101); from the CPU: CIIP
with LFNST / NSPT 117, with LIC 66, OBMC with an intra neighbour 47, OBMC 13, with inter CCCM 1.
Inter CCCM with OBMC (2026-10-23, agent lf; fixes 9886461dd): since M99 the GPU blends OBMC's band into an inter block's
prediction in place (the DERIVE_INTER branch), and the CPU's motion compensation does it before
deriveInterCccmPrediction blends; the inter CCCM branch now does the same first (interObmcBand, the DERIVE_INTER
branch's step for the block), so the model fits and the blend takes the OBMC'd chroma prediction (the luma's is the luma
block's, OBMC'd before by DERIVE_PREFIX). Found on the way (dec-gr-ccpmerge's 50-stream run on 2de8ee5a9):
ld_parkjoy_q27, ldp_crowdrun_q27, ra_foreman_q22, ra_parkjoy832_q32 and ra_parkjoy_q37 differed at inter CCCM chroma
blocks of OBMC CUs, by 1 to 3.
Inter CCP merge on the GPU (2026-10-23, agent lf; partial verification): an inter TU's chroma prediction blended with
the CCP merge candidate of the lowest template cost (JVET-AF0073: InterPrediction::deriveInterCcpMergePrediction,
IntraPrediction::selectCcpMergeCand, combineCcpAndInter; also JVET-AH0066's CU without residual,
interCcpMergeZeroRootCbfIdc). recordInterCu records the chroma blocks as inter blocks, then interCcpRecord turns each
into an intra CCP merge block of LM_CHROMA_IDX on the intra path (DERIVE_INTER cleared; the residual, its clipping and
chroma residual scaling the inter block's), marked DERIVE_INTER_BLEND (bit 31, DERIVE_INTER_CCCM's, without
DERIVE_INTER), waiting for every block before it (DERIVE_PREFIX). Its entry (interCcpEntry) is an intra merge entry's
with the index 0 (selectCcpMergeCand's first of the lowest cost is the stable sort's first), the CTU row's history as it
was before the CU (interCu snapshots it: the CPU adds the CU's own model after its prediction), and
getCCPModelCandidateList's sources for an inter CU: ccpMergeSources now puts, for an inter CU, the temporal candidates
moved by the CU's own vector to the collocated picture (its first list referring to it, else none) after the adjacent
positions and before the unshifted ones. In a low-delay slice its list starts with xAddOnTheFlyCalcCCPCands4InterBlk's
on-the-fly CCCM candidate: the block fits it into a record of its own (ccpInterFly: ccpFitModels' one model of the
reference samples alone; type 0 where every coefficient but the bias is 0, which the list then skips: ccpInterFlyFirst).
Before the merge it applies its CU's OBMC band to its inter prediction (interObmcBand), and storeRegion blends each
predicted sample with it ((wCc ccp + (4 - wCc) inter + 2) >> 2, clipped; wCc combineCcpAndInter's: 3, or 4 for
interCcpMerge without luma residual, or interCcpMergeZeroRootCbfIdc's weight), the inter prediction being the one before
the CPU's blend (DecCu's interChromaPred, also inter CCCM's, renamed from interCccmPred). The block writes no CCP
record: its TU's model stays the CPU's (cpuCuModels) for the later merges. Still the CPU's: inter CCP merge of an IBC
CU, of a CU of several TUs, with ASBT, with adjustment or decoder-derived fusion, other chroma formats (none in the 50
streams). Results (50 streams, builds aglcc1 and aglcc2): "inter CCP merge" 161 -> 0, every verified picture identical
(with the previous paragraph's fix; ra_wpp_q32 skipped as before).
Integration of lf's two commits with M106-M108 (2026-10-23): with the inter CCCM fix above, 5d1f0277a's exception
(interMcGpu left inter CCCM CUs to the CPU's prediction) goes. The 50-stream run of that (build int2) found
ld_foreman_q32, ldp_crowdrun_q27, ldp_foreman_q32 and ra_parkjoy832_q32 differing at inter CCP merge chroma blocks of
CIIP CUs (M106 and the inter CCP merge port were verified apart): the CPU blends the CCP prediction with the CIIP one,
intra_rec blends it with the inter prediction in storeRegion, before storeRecon's CIIP blend. ciipWhy now leaves CIIP
with inter CCP merge to the CPU (14 CUs in those four streams). Results: int2 every other picture identical
(ra_wpp_q32 skipped as before); the four streams identical with the fix (build int3, the commit's tree).
M109 (2026-10-23): GPM with separate OBMC on the GPU (JVET-AK0212: motionCompensationGeo predicts each part at its
final precision, applies the part's own subBlockOBMC - with m_geoOBMC: no skipObmcConditionByPixel, no sharp bi filter
(JVET-AI0094) for the neighbour bands and TM templates, TM only in low-delay slices - and blends weightedGeoBlkRounded,
(w a + (32 - w) b + 16) >> 5). CommonLib records the blend taken and each part's motion as its OBMC sees it, spanned
over the CU (m_gpuGeoSep, m_gpuGeoMi, m_gpuGeoObmc via gpuGeoObmc). The host plans each part's OBMC with that motion
put into the CU's motion buffer and back (withPartMotion: getSameNeigMotion compares the neighbours with it; an MMVD
part's usePreCheck runs are the same, the part's motion being the same over the CU) and makes one OBMC record per part
(OBMC_REC_GEO; its jobs with no pixel test - OBMC_J_INT_DIR 0 - and MC_NO_SHARP; TM's current template from the part's
motion). A block's job carries MC_GPM_SEP: mcSampleBase gives part 0's final prediction, the job writes part 1's into
the OBMC data region (MC_BDOF_TH); Tu::gpm (Tu now 109 words) points at a record of part 1's place, its OBMC record
and the block's weights; intra_rec's interObmcBand (now also the DERIVE_INTER branch's step) applies each part's OBMC
and blends. intra_deps and the host's mirror take the TM templates' dependencies from part 1's record where part 0 has
none. Found on the way: ra_crowdrun_q32 differed by 1 in OBMC bands next to bi-predicted neighbours (the sharp filter;
MC_NO_SHARP); ra_ctu256_q32, ra_rpr15x_q32 and ra_rpr2x_q32 (low-delay slices, TM) where part 0 had no segments (the
templates' dependencies). Still the CPU's: parts with LIC, DMVR or sub-block motion, bi-prediction parts, and the rest
gpmWhy names. Results (50 streams, build gsep2): every picture identical (ra_wpp_q32 skipped as before); GPM 406 CUs
predicted by the GPU (+355); from the CPU: GPM (an affine part) 613, (an intra part) 467, (regression-based blending)
324, (a bi-prediction part) 303, (LIC) 82, (OBMC with an intra neighbour) 47, (separate OBMC with a LIC part) 45.
CIIP's chroma residual scaling once (2026-10-23, agent lf; fixes M106): a CIIP chroma block with LMCS chroma residual
scaling ran crsSetup twice, in the DERIVE_INTER branch and in the intra path's preamble, and the second workgroup sum
(wgSum2 into s_red) added to the first: twice the luma average, another scale; the DERIVE_INTER branch now leaves a
CIIP block's to the preamble (found once OBMC with intra neighbours put a CIIP CU of coded chroma residual on the GPU:
ldp_foreman_q32 POC 8, a Cr block's residual 1 off). lf's commit also left an inter CCP merge CIIP CU to the CPU in
interCcpWhy, the same fix as the integration paragraph's ciipWhy one, which stays.
Hardware of M99-M109 (correction, 2026-10-23): their verification ran on this PC, a Core Ultra 7 265K with its Intel
Graphics iGPU (driver 32.0.101.8991); the commit messages 86e6c59f3 .. 30744a6ab name an i7-10700 with an RTX 3070,
copied from older messages. Re-check on NVIDIA and AMD.
OBMC with intra neighbours on the GPU (2026-10-23, agent lf; JVET-AJ0161; partial verification: the CPU's derived
modes): with DIMD in the SPS, subBlockOBMC blends a run of intra neighbour units (getSameNeigMotion's iState 1) where
getGradForOBMC gave the side's modes (m_modeGetCheck: calcGradForOBMC, a DIMD of the two rows / columns next to the
CU): per run of one mode (at most MAX_INTRA_SIZE long) within the side's angles (above DIA_IDX to VDIA_IDX, left beyond
DC up to DIA_IDX) an angular prediction of the strip (subBlockIntraForOBMC: predIntraAng of the 4 luma rows / columns
and the 2 chroma ones with isobmcMC - no reference filter nor Gaussian interpolation, no bilateral filter, PDPC only
for the side's first unit, which has both sides' references; after it xFillReferenceSamplesOBMC's one-sided references
and prepareRef's samples along the side beyond the corner; luma mapped back by LMCS's inverse), blended by
xSubblockOBMC as weightObmcBoundary's mode -1 (no skipObmcConditionByPixel). obmcPlan now passes these runs (rejecting
only other chroma formats than 4:2:0, and more than OBMC_MAX_SEG segments counting each intra unit);
obmcIntraSegs splits them by the modes obmcSideModes takes from calcGradForOBMC over the CPU's reconstruction (in
partial verification the CPU's modes; intra_obmc.glsl's obmcGradModes is IntraTMP's GPU port of it, needing the units'
records) and appends them to the CU's OBMC record after the motion segments (obmcJobs numbers those): head side |
OBMC_SEG_INTRA | mode << OBMC_SEG_MODE, start, length, data laid out as a motion segment's; record flags OBMC_REC_INTRA,
OBMC_REC_INTRA_INV (the picture's LMCS and the CTU flag), OBMC_REC_NO_PDPC, OBMC_REC_NO_REF_FILTER (the SPS's). On
the GPU each block with Tu::obmc predicts, before obmcModes, the intra strips of its component that overlap it
(intra_obmc.glsl's obmcInterIntra: IntraTMP's strip predictor, generalized to chroma - units of 2, linear
interpolation - and to any scratch place, obmcStripPredict) and writes its part into the segment's band data in
Params::interPred; obmcModes gives intra segments mode -1, obmcSample masks the side bit and reads their data
device-coherent. The strips use s_ref[0], the angular arrays (cleared after) and the scratch's start, so an inter CCP
merge block now applies its band (interObmcBand) at the start of the intra path, before its references, instead of
before ccpMergePrediction. Dependencies (intra_deps' obmcIntraDeps and the host's mirror, for every block with an OBMC
record of intra segments): per strip its arms (8x its sides) and prepareRef's samples. Results (50 streams, build
lfio2, with the previous paragraph's fixes): every verified picture identical (ra_wpp_q32 skipped as before); the
reasons "OBMC (OBMC with an intra neighbour)" 422, "affine (...)" 245, "LIC (...)" 150, "CIIP (...)" 47, "non-local
LIC (...)" 13 -> 0; inter CUs predicted by the GPU against gpm2 (which lacked inter CCP merge): OBMC 6432 -> 6867,
affine 1527 -> 1790, LIC 552 -> 739, CIIP 200 -> 247, non-local LIC 113 -> 126.
Integration with M109 (2026-10-23): GPM with separate OBMC next to an intra neighbour stays the CPU's (gpmSepPlan:
"separate OBMC with an intra neighbour", 45 CUs) - obmcPlan without its intra list no longer rejects intra runs, and
the first integration build took such CUs to the GPU without their strips (12 RA streams differed); with the fix the
50 streams are identical (build int4; ra_wpp_q32 skipped).
M110 (2026-10-23): the affine / SbTMVP sub-PU BDOF refinement (JVET-AF0159) is told to the host: DecCu's xReconInter
resets availableBdofRefinedMv right after the prediction, so affineGpu's test of it never held at record time (the
CUs it meant were caught by obmcPlan's "OBMC of sub-block motion" alone, as checkDoAffineBdofRefine forces their OBMC
flag); DecCu now calls subPuBdof where getDoAffineSubPuBdof holds, before the reset, and recordInterCu leaves those CUs
to the CPU ("affine (with BDOF)"). A first version also ported the inner sub-block OBMC of affine CUs where
checkDoAffineBdofRefine holds without the BDOF having run (luma alone, innerObmcSkipChroma; mvDiffTH only with DMVR):
no CU of the 50 streams reaches it, so it stays rejected, unported. Results (50 streams, build aob3): every picture
identical (ra_wpp_q32 skipped); the GPU's counts unchanged; "affine (OBMC of sub-block motion)" 697 and 12 of
"affine (other)" are now "affine (with BDOF)" 709 - the sub-PU BDOF port is what they need.
M111 (2026-10-23): GPM parts of bi-prediction on the GPU (JVET-AE0046) without BDOF, DMVR or LIC. CommonLib records
per part whether xPredInterBi applied BDOF (m_gpuBio, read by gpuGeoBio after the part's motionCompensation) and its BCW
index (m_gpuGeoPart grows to 24 ints). A bi-prediction part gets its own job record in pc.affine (MC_NLIC_AT + 24 + k:
the vectors clipped at the PU, BCW's weights, the lists out of boundary at the unclipped vectors, list 0 alone where
xCheckIdenticalMotion holds); inter_mc's gpmPart predicts it with mcPlain - mcSampleBase's uni / bi-prediction tail,
now a function of its own (GLSL has no recursion) - and, for the plain blend, takes it back to the intermediate as
convert2HighPrec does ((v << (14 - bd)) - 8192, unclipped); with separate OBMC it stays at its final precision. Results
(50 streams, build gpmb1): every picture identical (ra_wpp_q32 skipped; ai_foreman352_q22 alone after a device loss in
the parallel run); GPM 486 CUs predicted by the GPU (+71); the former "a bi-prediction part" 303 are now mostly "a DMVR
part" 224 (all in RA streams: their separate OBMC runs the inner sub-block OBMC over the part's DMVR motion) and "a
bi-prediction part with LIC" 4.
M112 (2026-10-23): explicit weighted prediction on the GPU for uni- / bi-prediction CUs and their OBMC (xPredInterBi:
xWeightedPredictionBi in a B slice of weighted bi-prediction where BCW is the default and neither BDOF nor DMVR applies -
addWeightBi, addWeightUni with one list -, xWeightedPredictionUni in a P slice of weighted prediction). wpRecord writes
per component getWpScaling's weights, offset and shift (with the intermediate's shiftNum) into pc.affine; a job with
MC_WP reads it at MC_BDOF_OUT, and mcPlain weighs the lists' 14-bit intermediates (mcPredict with bi, the sharp filter
too, as xPredInterUni with bi): weightUnidir (also for the weight 1 << log2WeightDenom, where noWeightUnidir gives the
same) or weightBidir, clipped, without the out-of-boundary masks. OBMC's neighbour bands get the neighbour's (its lists
and BCW index; xSubBlockMCFast leaves weighted prediction to xPredInterBi); the TM templates and the pixel test stay
unweighted as on the CPU. mcBdof gains xPredInterBi's biocheck0 / biocheck1. Still the CPU's with weighted prediction:
LIC, non-local LIC, DMVR, LIC or non-local LIC neighbours of OBMC, affine and GPM CUs. Results (50 streams, build wp1):
every picture identical (ra_wpp_q32 skipped; ai_foreman352_q22 lost the device twice and passed alone on a third run);
OBMC 6972 (+98) and CIIP 257 (+10) CUs predicted by the GPU, "OBMC (its base prediction)" 128 -> 19.
M113 (2026-10-23): explicit weighted prediction of affine CUs on the GPU: xPredInterBi weighs xPredAffineBlk's
intermediates of each list as of any CU (the identical-motion shortcut is off with weighted bi-prediction); the affine
record's AF_WP points at wpRecord's record and afSample applies M112's weightUnidir / weightBidir to afList's
intermediates (PROF in them), without the out-of-boundary masks. Not with LIC or non-local LIC. Results (50 streams,
build wp2): every picture identical (ra_wpp_q32 skipped; ai_crowdrun832_q27 lost the device in the parallel run and
passed alone); affine 1876 CUs predicted by the GPU (+83).
M114 (2026-10-23): affine CIIP on the GPU (JVET-AG0135: DecCu blends an affine CIIP CU with
geneWeightedCIIPAffinePred, the neighbours' weights alone - (wMerge inter + wIntra intra + 2) >> 2, luma's inter
forward-mapped -, whatever the intra mode). affineGpu takes a CIIP CU where ciipAffine holds and ciipWhy finds nothing
(not with CIIP-PDPC); ciipFields gives it CIIP_UNIFORM (M106's blend of those weights); the affine record's BCW
weights are the default for CIIP (xWeightedAverage's plain average). The CPU-reason tally names an affine CIIP CU's
blocker ("affine (CIIP: ...)"). Results (50 streams, build aciip1): every picture identical (ra_wpp_q32 skipped);
affine 1927 CUs predicted by the GPU (+51); "affine (CIIP)" 212 -> CIIP with LIC 60, with LFNST / NSPT 54, OBMC of
sub-block motion 13, inter CCP merge 4, inter CCCM 1, others 28 (non-affine CIIP, CIIP-PDPC, DMVR, weighted
prediction with LIC neighbours).
M115 (2026-10-23): GPM with an intra part on the GPU (JVET-Y0065, motionCompensationGeo's gpmIntraFlag path: the intra
part predIntraAng of a geometric MPM, chroma at the same mode; the inter part at its final precision -
roundToOutputBitdepth's roundBD of the intermediate, or as it is with separate OBMC -, after its own subBlockOBMC with
its motion, forward-mapped by LMCS; weightedGeoBlkRounded of the two). It is a CIIP variant on the GPU: CommonLib
saves the intra part's parameters (IntraPrediction::gpmSnapshot after each predIntraAng) and its mode (the part's
record's word 24, m_gpuGeoPart now 25 ints), and the blend taken (kind 3); a block's job (MC_GPM_INTRA, MC_GPM_I1 for
part 1, MC_GPM_FIN with separate OBMC) predicts the inter part; ciipFields fills the intra fields from the snapshot and
Tu::ciip gets CIIP_GPM (CIIP_GPM_I1 where the intra part is part 1) with Tu::gpm's record of the weights, so intra_rec
applies the inter part's OBMC (its record as the CU's, made with its motion: withPartMotion; the normal rules -
skipObmcConditionByPixel, TM - without separate OBMC) in the DERIVE_INTER branch, goes on to the intra path, and
ciipBlend weighs (w a + (32 - w) b + 16) >> 5 with the inter part forward-mapped (CIIP_FWD). gpmWhy applies ciipWhy's
limits (one TU, no LFNST / NSPT, LIC, inter CCCM / CCP) and leaves SGPM's intra parts to the CPU. Found on the way:
the GPM job's loop took a slot for the intra part's stale reference index (getRefPic's CHECK). Results (50 streams,
build gpmi1): every picture identical (ra_wpp_q32 skipped; ai_foreman352_q22 lost the device twice and passed alone);
GPM 612 CUs predicted by the GPU (+126); "an intra part" 467 -> SGPM 78, CIIP's limits 136 (LFNST 49, LIC 48, several
TUs 31, inter CCP 8), and CUs that now show their next reason (an affine part 613 -> 671, separate OBMC with an intra
neighbour 51 -> 100).
Sub-PU motion compensation on the GPU (2026-10-23, agent sb; motionCompensation's xSubPuMC: sub-block merge - SbTMVP
-, AMVP-SbTMVP and JVET-AF0159's affine / SbTMVP sub-PU BDOF refinement). CommonLib records xSubPuMC's predictions of
its last PU (InterPrediction::m_gpuSubPu through gpuSubPu, at the end of the class): per group of xGetSubPuGroupArea2D
its area, motion, m_bdofMvRefined after its xPredInterBi and whether bdofCond is off; per run of sub-PUs of one motion
its area, motion and xCheckIdenticalMotion. DecCu clears it before the CU's motionCompensation and hands it over after
(subPuMotion; partial verification: the CPU's motion, groups and runs); recordInterCu takes such a CU where subPuWhy
finds nothing missing (the record's PU and the predictions' areas covering it, no LIC, CIIP, DMVR, multi-hypothesis,
weighted prediction, resampling or SbTMVP with non-local LIC). M110's subPuBdof goes (the record says it). Each group
and run is an inter_mc.comp mode-2 job (DM_SUB) written at its place in the PU's prediction (dmOut with the PU's
stride: DM_OX, DM_OY, DM_OW, DM_OH), which the CU's blocks copy (MC_BDOF). A group replays xPredInterBi with m_subPuMC
and bdmvrRefine at its one vector pair through M98's functions with the m_subPuMC branches (dmSub): a side 4 modulo 8
(m_skipAffineFirstIterBdof) skips the first pass, its one iteration then predicting and refining the whole group;
else one prediction of the group, with BDOF where the CPU's bioApplied held (DM_BIO), else bi-prediction with BCW's
weights and nothing more. applyBiOptFlow with m_subPuMC: scaleBDOF 1 where a side of the block is 4 modulo 8 (not 4),
else 2; bioBlockDistTh without the QP term; the first pass writes the luma of the blocks it leaves in place (addAvg,
subBlockBiOptFlow); its offsets halved only from 256 samples; calcBIOParamSumHighPrecision_SSE's isSub mean (getMean of
the window's dI sum plus half its size, without the magnitude test); no noMeanRm4Sample / noMR; with non-local LIC no
offset at all (DM_NLIC; the model then on the PU's prediction, MC_NLIC, and on the inner OBMC's neighbour predictions,
as subBlockOBMC keeps the CU's altLM flag). xPredInterBiBDMVR2 runs once (i > 1 skipped), joins blocks by equal offsets
alone and skips those whose last offset is 0; xSubPuMC zeroes the offsets (not the last ones) before
xPredInterBiSubPuBDOF where bdofCond is off (DM_ZERO); xPredInterBiSubPuBDOF's scaleBDOF is 1 below 256 samples or
where a side is 4 modulo 8, and subBlockBiOptFlow's blocks follow both sides. A run is uni-prediction (DM_UNI; list 0
where xCheckIdenticalMotion holds) or bi-prediction with BCW's weights, clipped at the run. OBMC of these CUs:
obmcPlan's "OBMC of sub-block motion" goes for them, bSubMotion as subBlockOBMC has it (the inner sub-blocks at the
stored motion, the refined one where APPLY_AND_STORE), affine's innerObmcSkipChroma a record flag (OBMC_REC_SUB_LUMA:
intra_rec blends the inner sub-blocks of luma alone), and obmcJobs' inner motion threshold the half sample of JVET-AL0081
only with DMVR (subBlockOBMC's mvDiffTH). Results (50 streams, build agsb3; agsb4 only renames a reason, rerun on 5
streams): every picture identical (ra_wpp_q32 skipped as before); the GPU predicts "affine (sub-block merge)" 807 of
850, "affine (with BDOF)" 696 of 709, "sub-block motion" 123 of 130; from the CPU: weighted prediction 61 (ldp_wp_q32,
ra_wp_q32), non-local LIC with resampling 2 (counted as "affine (with inherited LIC or resampling)" before), OBMC with an
IBC neighbour 2.
Regression-based GPM blending on the GPU (2026-10-23, agent rg; JVET-AG0112, JVET-AJ0274; partial verification: the
CPU's parts and blend model): DecCu predicts a CU of geoBlendFlag by motionCompensationGeoBlend - a pair of the merge
list's candidates (getGeoBlendCand; with geoBlendTmFlag the candidates refined by deriveTMMv first) and a blend model
fitted to their templates (deriveBcwBlendingBiDir), each part predicted alone as motionCompensationGeo does (at the
intermediate precision; with separate OBMC, JVET-AK0212, at its final one after its own subBlockOBMC), blended by
weightedBlend: part 1's weight w at each luma position from the model (xWeightAffineBlk_SSE: p0 x + p1 y + p2 + offset
in 32 bits, shifted, saturated to 16 bits, clipped as min( max, max( w, min ) ); chroma takes the even luma positions),
((32 - w) p0 + w p1 + offset) >> shift, clipped (weightedBlendBlk, roundOutputBD as the geo blends). That is
xWeightedGeoBlk's / xWeightedGeoBlkRounded's blend with part 0's weight 32 - w (weightedBlendBlk's clip changes nothing
for weights within 0 .. 32 of predictions within range; gpmWhy checks min and max). motionCompensationGeoBlend now
calls M107 / M109 / M111's hooks (gpuGeoPart, gpuGeoBio, gpuGeoObmc; it resets their records first) and records its
blend (gpuGeoBlend: its kind - intra part, TM - and cu.blendModel as weightedBlend takes it, InterPrediction::
m_gpuGeoBlend, unconditional and at the end of the class; m_gpuGeoPlain / m_gpuGeoSep set as for the geo blends;
motionCompensationGeo clears it); DecCu's gpmParts hands it over and recordInterBlock writes those weights where
gpmWeights wrote the split's (gpmBlendWeights); no shader change, the rest is M107-M111's path and gpmWhy's checks.
Found on the way (ld_parkjoy_q27, getRefPic's CHECK): such a CU's PU keeps interDir 3 with the last part's motion, so
recordInterBlock's slot of the PU's own list 0 took reference index -1; a GPM block now takes no slot of the PU's motion
(the parts' replace it). Still the CPU's (gpmWhy, "GPM (regression-based blending: ...)"): an intra part (JVET-AK0101),
DMVR parts, LIC of the CU (the last part's flag, which OBMC's TM templates of the CU's motion take), separate OBMC with
a LIC part or an intra neighbour, bi-prediction parts with LIC, LIC parts with resampling. Results (50 streams, build
agrg1): every picture identical (ra_wpp_q32 skipped; ai_foreman352_q22 lost the device in the parallel run and passed
alone); "GPM (regression-based blending)" 324 -> GPU 101 (31 of them with TM, a rerun of the 26 streams with these CUs
after the tally's TM split); from the CPU: a DMVR part 92, an intra part 47, LIC 45, separate OBMC with a LIC part 22,
separate OBMC with an intra neighbour 12, a bi-prediction part with LIC 3, a LIC part with resampling 2. Every other
count as int5's.

M116 (2026-10-24): GPM DMVR parts on the GPU (bi-prediction parts with multi-pass DMVR - gpmDmvrRefinePart -, all in RA
streams, also in regression-based blending). CommonLib records after the part's prediction its refined vectors per
16 x 16 sub-PU (m_gpuGeoDmvr, gpuGeoBio; BDOF from xPredInterBiBDMVR too) and at its OBMC the CU's whole motion buffer
(m_gpuGeoMb, gpuGeoObmc). Each DMVR part gets an M98 DMVR job of its own (the part's lists, BCW, BDOF; DM_NOMR:
applyBiOptFlow takes noMR for every GPM CU, no mean removal in the refinement - without it ra_crowdrun_q32 and
ra_parkjoy832_q32 differed by a few levels), its record in pc.affine flagged MC_BDOF, which gpmPart reads at the final
precision (or as convert2HighPrec takes it). withPartMotion restores the part's motion buffer and puts the part's
motion and bdmvrRefine into the PU as subBlockOBMC sees them, so separate OBMC plans and makes the part's inner
sub-block OBMC (M100's mode-4 jobs, the half-sample threshold of the same direction) like the CPU; part 1's record
reaches every block where it has inner OBMC. Results (50 streams, build gpmd1): every picture identical (ra_wpp_q32
skipped); GPM 849 CUs predicted by the GPU (+237), regression-based blending 192 (+91); "a DMVR part" 240 and
regression's 90 -> 0.

M117 (2026-10-24): GPM affine parts on the GPU (affineGPM[0/1]).
gpuGeoBio records the affine part's fields and motion buffer (m_gpuGeoAff[2][17]); each affine part gets an affine
record (affineRecord on the CU with the part's motion, inside withPartMotion), which gpmPart reads (MC_AFFINE, record
index in MC_BDOF_OUT). The kind-2 blend takes gpmPartFinal: with separate OBMC the part as it is, else (MC_GPM_RBD, the
low-delay case) roundBD of its intermediate, as roundToOutputBitdepth. Inner sub-block OBMC of an affine part is luma
only (OBMC_REC_SUB_LUMA, innerObmcSkipChroma via m_gpuSubMotion / gpuGeoSub); obmcJob's skip test of part 1 reads
part 1 (OBMC_P1). Status (build gpma1, Intel PC): ra_crowdrun_q32 and ra_foreman_q22 identical; ld_foreman_q32
differed in 1 picture (POC 1, CU 128,128 32x16: the GPU +1 over all of part 0, a plain uni-L0 part, beside affine part
1) and ld_parkjoy_q27 in 2.
Continued on the RTX 3070 PC (2026-10-24), two causes:
- The +1: MC_GPM_RBD's rounded part left out the LIC model the part inherits from its merge candidate, which the geo
  blend (MC_GPM) applies on the intermediate (mcLicBi) - a model of scale 32 and offset -1 is a uniform +1.
  gpmPartAt now gives a part at the final precision (GPM_FINAL), at the intermediate with its LIC (GPM_INTER) or that
  rounded (GPM_ROUNDED, roundToOutputBitdepth) for every GPM path: the geo blend, the rounded blend's two parts, the
  inter part beside an intra part (that one had the same omission without separate OBMC).
- NVIDIA's driver could not compile inter_mc any more: M117's affine parts put afSample / afList (~100k SPIR-V
  instructions) into gpmPart, which had 34 inlined copies (gpmPartLic, gpmPartFinal, mcSampleBase's five callers);
  inter_mc grew from 0.70M instructions fully inlined (M116) to 3.88M, and the 3070's driver took over 40 GB and
  15 minutes for it (the decoder was stopped at the 60 GB budget; every pipeline creation after scan's waits for
  it). gpmPart now has one call site (gpmPartAt, called once in mcSampleBase in a loop over the parts with
  [[dont_unroll]]), and mcSample four (main takes the block's and GPM part 1's predictions in such a loop;
  obmcInnerJob's luma and chroma calls are one): 0.84M, compiled in ~4-7 minutes and ~11 GB (obmcJob's two calls in
  such a loop too would make 0.65M; dropped as a suspect of the device losses below, which turned out to be older).
  perf/gpu/inline_size.py shows a module's inlined size and the functions that make it.
Found on the way, on the 3070 PC's new streams (C:\Code\ECM\streams_ld):
- A GPM part's own OBMC (the rounded blend: each part's subBlockOBMC) runs on the CPU with the part's LIC in the CU
  (motionCompensationGeo's setMergeInfo for the part), and selectOBMCmode predicts the template of the CU's motion
  with it; obmcJobs took the CU's final LIC (the last part's). withPartMotion now puts the part's LIC flag and models
  (its record's GPM_LIC_MODELS) into the CU beside its motion (ld_parkjoy_q27 POC 1 and 2: a plain part with LIC beside
  an affine part, the GPU 1 low in the OBMC band).
- The host's temporal CCP merge candidates (temporalAt) read the collocated picture at positions the CPU never reads:
  the CPU stops at a full list, the host goes on, and a vector shift can move a later position above or left of the
  picture, where getCcpmIdxInfo's CHECK ends the decoder (ldp_parkjoy_q27, ra_parkjoy_q32, ra720_pj17 at M116). Such
  positions are skipped.
Results (RTX 3070 PC, build grn4 = slangopt without inlining, ECM_GPU_NO_CCP_LFNST=1, partial.py -j 2; M116 built the
same way as m116 from a detached worktree, C:\Code\ECM\wt-m116, its intra_rec module byte for byte the same):
- the 19 regular streams: no new difference. The ones that differ do at M116 too (intra: ai_long POC 5 NN's
  indicesRepresentationPnn, ld_md5 POC 0 scheduling data and POC 40 an SGPM CU's DIMD mode, ra_ip64_md5 an SGPM CU's
  MTSS LFNST mode); ra720_pj17 differs in 5 pictures, which M116 never reached (it ended at POC 0 on the CCP CHECK, as
  on ra_long; ld_cif33 lost the device under M116). Over the 16 streams both decoded whole: inter CUs predicted by the
  GPU 7155 -> 7484 (GPM 493 -> 822), by the CPU 995 -> 666 ("GPM (an affine part)" 431 -> 0; 44 with an intra part
  stay the CPU's).
- the inter streams of streams_ld: ld_foreman_q32, ldp_foreman_q32, ld_four_q32, ra_foreman_q32 identical;
  ld_parkjoy_q27 POC 1 and 2 identical now, POC 7 and 8 differ as at M116 (POC 7: an affine sub-block merge CU with
  OBMC, 128 x 128, GPU 104 for 98 - gone with OBMC or LIC CUs on the CPU; POC 8: a 4 x 4 AMVP-SbTMVP CU of a uni-L0
  run, the GPU exactly twice the CPU's value, and an OBMC CU with DMVR); ldp_parkjoy_q27 (3 pictures, the first an
  inter block 1 low) and ra_parkjoy_q32 (2) differ as at M116 (with the CCP fix in the m116 worktree);
  ld_foreman_q22 identical when it runs through.
- Device losses (VK_ERROR_DEVICE_LOST ~6 s into the decode, i.e. a hang reaching Windows' 2 s timeout) come and go:
  ld_foreman_q22 alone lost it in 2 of 5 runs with this build and 1 of 5 with M116; ld_cif33 under parallel load
  (with M116 too). Not a hang but a memory fault: fixed, see below.
- Intra pictures: most need LEVEL_CCP_LFNST's pipeline and stay unverified on this PC (ECM_GPU_NO_CCP_LFNST).
Fix after M117 (2026-10-24): endPicture runs a picture again when its dependency edges pass the scheduling buffer's
room, and the DMVR / xSubPuMC jobs' vector offsets (DM_MVS) were made absolute inside that loop - the second pass
added the job array's size again, and every DMVR / sub-PU job read other data as its vectors (ld_parkjoy_q27 POC 7
and 8: the 4 x 4 AMVP-SbTMVP CU at 232,16 had the right bdofOut after the first pass, other samples after the second).
They are made absolute once, before the loop: ld_parkjoy_q27 identical.
Fix (2026-10-25), the intermittent device losses above: OBMC's intra strips of an inter CU (intra_obmc.glsl's
obmcInterIntra) predict in the workgroup's scratch (tsBase, BLEND_ACC words per workgroup), which endPicture
allocates only when a recorded block needs it (blendBlocks). obmcIntraRecord did not set it, so a picture whose only
users were such strips wrote past the buffer (nvlddmkm event 153, a memory fault, not a timeout). It sets it now:
ld_foreman_q22 ran through 20 of 20 times (2-6 of 10 lost before). Found with VK_EXT_device_fault (ECM_GPU_FAULT=1:
the faulting address, and every buffer's address range at its creation) and a spin limit in intra_rec's queue wait
that ruled a hang out.
M118 (2026-10-25): more inter CUs on the GPU and three fixes.
- Fix: a non-local LIC CU's OBMC (getSameNeigMotion) compares a non-local LIC neighbour's model with the CU's
  altLMParaUnit, which held the CU's applied model while the CPU made its OBMC; after the reconstruction DecCu keeps
  that model in secAltLMParaUnit and puts another into altLMParaUnit, so obmcPlan, run after it (recordInterCu), saw
  another model and planned a segment the CPU had skipped (ldp_parkjoy_q27 POC 2: an 8 x 4 CU whose left neighbour had
  its motion and model, 1 low at its corner). obmcPlan now compares with the applied model.
- Fix: a deferred block's prediction (g_deferRecon: the DIMD of the prediction picks its transforms) of a 64-sample
  side - an SGPM CU of 16 x 64 or 64 x 16 - went into signPredictionLarge's area instead of s_spPred, where the DIMD and
  the reconstruction read it (every such SGPM CU: ldp_parkjoy_q27 POC 6 / 8, ra_ip64_md5, ra720_pj17, ld_md5 POC 40;
  the DIMD's mode even changed from run to run). Only sign-predicted blocks above 32 x 32 use that area now.
- GPM with separate OBMC and intra neighbours (JVET-AJ0161 strips of each part's subBlockOBMC): gpmSepPlan returns
  each part's runs of intra neighbours, each part's OBMC record gets its intra segments, and interObmcBand predicts a
  part's strips (obmcInterIntra of that record) before its OBMC.
- LIC with DMVR (JVET-AG0276): xPredInterBi predicts a LIC CU with bdmvrRefine plainly at the PU's vectors, which it
  set to DMVR's first sub-PU's (pu.mv, as the host reads it after the reconstruction), and subBlockOBMC skips the inner
  sub-blocks of LIC CUs. LIC with BDOF stays on the CPU: ported as xLicCompAdjBdof (the models on both lists' grown
  blocks, 8 x 8 BDOF blocks, the SAD threshold without the QP's shift) it was 1 off in a CU's last column (3 samples
  of ld_foreman_q32 POC 6, 64,128 32 x 32, x 31, y 4..6); the CPU's grown block at the right edge needs a closer look.
- A GPM part's LIC with separate OBMC: the part is predicted on its own at the final precision, where xPredInterBlk
  applies the inherited model to the final sample (gpmPartAt's GPM_FINAL now applies it).
- CIIP with LIC: xPredInterBlk / xLicCompAdj leave a CIIP CU's LIC off unless its first reference is the picture before
  it in a low-delay slice (disCond; DecCu then stores identity models, which bi-prediction's rounding through them
  would still change): licApplied() gives the GPU's jobs the same rule. Where LIC is on, xWeightedAverage takes the
  BCW weights (alwCond), else CIIP's plain average (ld_cif33 POC 16: an 8 x 4 CIIP CU with LIC and BCW index 3).
- CIIP with LFNST / NSPT (up to 32 samples a side): their mode is a DIMD of the CIIP-blended prediction
  (xDecodeInterTexture after geneWeightedPred), so such a block defers its reconstruction in the intra path
  (g_deferRecon), keeps the blended prediction in s_spPred and takes interTransformMode's operators over it.
- EIP in inter slices (not EIP merge, whose list has shifted temporal models).
- GPM CUs whose CU-level LIC flag is set (the last part's, motionCompensationGeo's setMergeInfo: the parts' own LIC
  comes from their records), GPM with an intra part of JVET-AN0093's lists (the part's mode comes in its record), and
  GPM with an affine part beside an intra part (gpmPartAt takes the affine part) - three exclusions of gpmWhy that
  partial verification showed unneeded.
- Development: the scheduling-data mismatch names the field and lists the dependents only one side has;
  ECM_GPU_NO_DERIVED_CCP leaves the pictures that need LEVEL_CCP_DERIVED's pipeline unverified (as ECM_GPU_NO_CCP_LFNST
  does LEVEL_CCP_LFNST's): its MPM pipeline went past the memory budget on 2026-10-25 (other programs held ~24 GB).
Results (RTX 3070, ECM_GPU_NO_CCP_LFNST=1 and ECM_GPU_NO_DERIVED_CCP=1, the 19 regular streams and the 8 of
streams_ld): every verified picture identical but ai_long's (NN, older); inter CUs on the GPU GPM 5555 -> 7099,
CIIP 684 -> 1189, LIC 3228 -> 3608, EIP CUs of inter slices 239 more. Left on the CPU (top): IBC with OBMC 970, an
intra merge neighbour of a kind the GPU lacks 605 (512 of them a neighbour recorded as a known mode CU, i.e. one from
the CPU: a cascade), CIIP with inter CCP merge 259 (+ affine 72, GPM 62), GPM CIIP with LIC 198, IntraTMP OBMC with
inter or two-vector neighbours 182, IBC bi-prediction 180, EIP merge in inter slices 180, LIC with DMVR + BDOF 151,
affine CIIP 140, CIIP with inter CCCM 126.
intra_rec's size (2026-10-25): the CCLM / GLM / CCCM luma helpers (cclmDownsample, glmGradient, cccmDownsample,
cccmFeature, LBCCP's cclmInsideSample) read their taps in loops of one call site ([[dont_unroll]],
GL_EXT_control_flow_attributes) instead of written-out taps: each is inlined hundreds of times, and every tap was a
copy of recoAt / cccmLumaOf / cclmPredSample. Inlined size (inline_size.py) 1.00M -> 0.68M for intra_rec's module,
1.21M -> 0.76M for the CCP-LFNST one; recoAt's copies 9098 -> 1024. The CCP-LFNST pipeline (8:6), which needed more
than the ~35 GB the budget left, compiles in 818 s and 18.7 GB, so ECM_GPU_NO_CCP_LFNST / NO_DERIVED_CCP are not
needed on the 3070 any more. Intra merge CUs whose neighbour has no node (a CU from the CPU, recorded CU_KNOWN) of the
merge's or DIMD's kind: mergeClass gives the CPU's parameters (cpuDimdParams, the values verification compares a
node's with) and endPicture uploads them into that CU's DimdOut, so the neighbour stays a candidate instead of sending
the merge CU, and in turn its own merge neighbours, to the CPU (512 of the 605 such CUs were that cascade, most in
pj720_ai / fp720_ai).
Also tried (2026-10-25), not kept: intra_rec with its 29 largest functions of several call sites marked DontInline
(perf/gpu/spirv_noinline.py; ECM_GPU_INTRA_REC_SPV): the MPM + derived CCP pipeline compiled in 19.9 GB instead of
more than the ~35 GB left under the budget, but the output differed (inter CCP merge chroma blocks) and the GPU time
grew by 20 %.
M119 (2026-10-25): the size cut and the merge feed above, three fixes of older bugs, eight ports, intra_rec in
slices and faster IntraTMP nodes.
- Fix: a motion run of IntraTMP's OBMC (intra_obmc.glsl's obmcSideOps) took its first unit's IBC LIC for the whole
  run; getSameNeigMotion ends a run where the LIC flag, scale or offset changes, so a run now extends only over units
  of the same LIC (ai_foreman / aiodd: an IntraTMP CU at 64,32, 32 x 4, whose units 5 and 6 had other models).
- EIP merge in inter slices (getNeiEipCands' temporal candidates): the collocated picture's models at the
  bottom-right / centre positions at five distances after the adjacent CUs' models, and the same shifted by the first
  adjacent inter neighbour's motion towards the collocated picture after the non-adjacent ones. The CPU's loops are one
  function now (IntraPrediction.cpp's eipColModels, getNeiEipCands calls it; its subpicture / picture bounds keep the
  CPU's unsigned comparison, which rules out positions left of or above the picture); eipTemporalModels lists them
  for the host, which gives each a picture EIP index of the CPU's model (eipModelIndex, uploaded like a CPU CU's) and
  a source of no CU record (-1: intra_deps waits for no node). The history is empty in an inter slice of separate
  trees, as tryHistEip has it.
- IBC with OBMC (subBlockOBMC of an IBC CU of luma alone - an intra slice's dual tree -, up to 32 x 32 and one TU): the
  IntraTMP OBMC record (bvObmcUnits, which addTmpObmcRecord now calls too), planned with the CU's own vector and
  LIC flag, scale and offset (licScale / licOffset of list 0's luma, what getSameNeigMotion compares), the block's
  prediction and its motion runs' copies in one loop, then tmpObmc's blends and the reconstruction (intra_ibccu.glsl's
  ibcBlock; the MPM pipelines, which hold intra_obmc.glsl, run the picture).
- IBC bi-prediction (JVET-AE0169): xPredInterBi's two copies of the picture at the intermediate precision (ibcPredict
  / ibcChromaPredict with hp: a copy shifted up by the headroom, one filter pass shifted down by the filter's
  precision less it, a second pass by the precision; ibcRound), averaged as xWeightedAverage averages; two equal
  vectors predict as one (xCheckIdenticalMotion). LIC, RR-IBC, BCW, weighted prediction, the IBC filter and OBMC beside
  it stay on the CPU.
- OBMC units of IntraTMP / IBC CUs next to an IBC CU of two vectors: their runs compare both vectors, both must be
  valid (checkIsIBCCandidateValidBi), the run's two copies are made at the intermediate precision and averaged at the
  blend; skipObmcConditionByPixel checks the first vector, then the second (an invalid one ends the checks).
- CIIP (and GPM with an intra part) whose chroma blends inter CCCM or inter CCP merge: DecCu blends CIIP's first
  (geneWeightedPred before xDecodeInterTexture's TU loop), so those blends read CIIP's prediction. A pre-pass record
  per chroma block (ciipChromaPrePass: a CIIP block without residual, CIIP_TO_PRED) blends CIIP's intra prediction
  into the inter one and writes the result where the TU's chroma block reads its inter prediction (Tu::ibc: that
  block's place in Params::interPred); that block then blends neither CIIP nor OBMC again and waits for the pre-pass
  (DERIVE_PREFIX, which inter CCCM / CCP merge blocks have). With inter CCCM the luma block keeps its CIIP blend in its
  inter prediction too (CIIP_KEEP_PRED), the luma prediction inter CCCM fits to (not forward-mapped again, as DecCu
  maps no CIIP prediction). Verification's first differing block skips the pre-pass records.
- GPM with an intra part whose CU-level LIC flag is set: as M118 found for GPM without one, the flag is the last
  part's and the inter part's LIC comes from its record, so gpmWhy no longer rejects it.
- GPM bi-prediction parts with LIC (not with BDOF or DMVR): xPredInterBlk's geo branch applies the part's inherited
  model to each list's intermediate where it is not the identity (nonTrivialModel), before BCW's average: the part's
  record (MC_BI | MC_GPM) goes through mcPlain with the part's models, which leaves an identity list as it is;
  gpmPartAt applies a part's LIC itself only to a uni-prediction part (one without a record of its own).
- OBMC of an inter CU next to an IBC neighbour (subBlockOBMC's MODE_IBC strip, xSubblockTMOBMC's mode 2): the segment
  (OBMC_SEG_IBC, with the neighbour's vector; no inter_mc job) is copied from the picture in intra_rec
  (intra_obmc.glsl's obmcInterIbc, from interObmcBand: the luma mapped back by LMCS's inverse, subBlockOBMC's lut),
  with skipObmcConditionByPixel's test at the vector clipped to the strip and rounded (not for GPM's separate OBMC);
  the CU's blocks wait for every block before them (DERIVE_PREFIX). An IBC neighbour of two vectors or LIC, and a CU
  of several TUs or ASBT (the test reads the whole strip's prediction before OBMC), stay on the CPU. interObmcBand
  now has one call site of the strips, the modes and the blend for GPM's parts and the plain case (intra_rec's
  inlined size 0.699M -> 0.690M with the IBC copies added).
- intra_rec in slices: a picture of the full intra tool set takes longer than Windows' GPU timeout (2 s, after which
  the device is lost; section 8), so intra_rec's dispatch ends after a budget of blocks and the host dispatches it
  again until every block is done. A workgroup counts each block it starts (sched counter 4) and stops when the
  budget is spent; a block it has taken over from a finished dependency (the chain's next one) goes back into the
  queue for the next slice. The budget follows the time per block measured so far (SLICE_NS, 250 ms per slice; at
  least 64 blocks), and a slice that finishes no block is an error.
- IntraTMP's node keeps the CU's own template (every cost compares it) in shared memory, in sign prediction's first
  stage buffer (s_spStage1, free during a node), and the sub-sample search's region in sign prediction's other buffers
  instead of reading the picture for each candidate.
- IntraTMP's list updates (invocation 0, after each chunk of 256 costs; section 8: most of a node's time) walk only the
  positions that may change a list, which the workgroup marks with the costs (s_spCost): in the refinement a cost below
  its type's threshold at the chunk's start (thresholds only fall), in the sparse search a position that is in a list
  after the chunk - the lists keep the first positions by cost, then order, so it ranks below the list's length among
  the list's entries and the chunk's positions (earlier ones that cost no more, later ones that cost less).
- The scratch (each workgroup's own part: IntraTMP's lists, CCCM's buffers, the blending sums, ...) is accessed with
  workgroup-scope coherence (WBufI32: workgroupcoherent) instead of device scope, so it stays in the SM's cache: the
  single-invocation loops over the lists no longer wait for L2 at each access.
- The OBMC record's layout (OBMC_REC_*, OBMC_SEG_*) is in intra_common.glsl for intra_rec and intra_deps: intra_deps
  read the segments with a stride of 4 words written out, which OBMC_SEG_MV's 6 broke during this work (dependencies
  of OBMC strips next to intra neighbours went missing).
- Fix: intra_deps' deduplication of a block's dependencies (g_uniq, used by addEdge alone) was moved into addEdge by
  the SPIR-V optimizer's private-to-local pass, which assumes inlining (ECM_SPIRV_INLINE=OFF turns it off): every
  call started with an uninitialized list, which dropped dependencies that matched its garbage and kept duplicates (an
  ISP sub-partition of ld_md5's first picture without its node, and pictures of fp720_ai / pj720_ai). Its list and
  count are one variable now, which main resets; perf/gpu/private_globals.py lists such variables of a module.
- Fix: an intra merge CU's IntraTMP neighbour names its CU in the merge record by the CodingStructure index
  (MERGE_NBR_TMP), not a CU record: both dependency analyses (the host's, intra_deps') read a record past the end
  there; such pairs add nothing now (the node waits for all before it, DERIVE_PREFIX).
- Development: ECM_GPU_DIFFS=1 lists the first differing block's samples (CPU / GPU prediction and residual);
  ECM_GPU_MIN_LEVEL=<n> runs every intra picture in a pipeline of at least that level. A scheduling mismatch names both
  blocks of the edge, their dependency counts and both sides' dependencies; the affine CUs from the CPU give the reason
  (affineWhy, which affineGpu now wraps).
Results (RTX 3070, build grn39, partial.py -j 1, every picture: no ECM_GPU_NO_* switch): identical ai720, aiodd,
ai_foreman, ai_md5, ai_parkjoy, pj720_ai, ld_cif33, ld_foreman, ld_md5, raodd, ra_foreman, ra_parkjoy, ra_long (257
pictures), ldp_foreman_q32, ld_foreman_q22, ld_foreman_q32, ld_four_q32, ra_foreman_q32. Differing, each in pictures
the switches skipped before (section 9, 000): ai_long 2 of 33 and ra720 1 of 3 (NN), fp720_ai 5 of 8 (chroma
reordering's DIMD, NN, SGPM's MTSS), ra_cif33 1 of 33 (SGPM's MTSS), CCP merge models in ra_ip64_md5 1 of 257,
ra720_pj17 5 of 17, ldp_parkjoy_q27 1 of 9, ld_parkjoy_q27 1 of 9, and ra_parkjoy_q32 1 of 17 (an OBIC CU's second
mode). Inter CUs on the GPU 42,423, on the CPU ~1,060 (top: LIC with DMVR 151, affine CIIP with LIC 142, GPM CIIP of
several TUs 130, GPM with an affine part with LIC 118, LIC with BDOF 109, multi-hypothesis 96, affine CIIP of
sub-block merge 90); intra CUs from the CPU ~600 (IBC bi-prediction with LIC / flip / BCW / OBMC 158, BVG-CCCM 89,
IntraTMP OBMC with an inter neighbour 69, an intra merge neighbour of a kind the GPU lacks 47).
M120 (2026-10-26): four fixes of older differences.
- Fix: regression SGPM's MTSS (intra_sgpm.glsl's sgpmMtss) skips the angular branch, as the CPU's does: a regression
  CU's prediction is its blend model's, not its angle's (ra_cif33 POC 32 CU 199, fp720_ai).
- Fix: an NN CU's LFNST / NSPT mode (mipTransformMode) starts its second mode at DC where no gradient counts, as
  xPredIntraPnn does (indicesRepresentationPnn of ai_long POC 5, ra720, fp720_ai).
- Fix: in an inter slice the CPU's OBIC counts an IntraTMP or MIP neighbour (a MIP one by its mipDimdMode), which the
  GPU's OBIC does not see: such CUs go to the CPU (ra_parkjoy_q32 POC 16 CU 8: obicMode[1] CPU 10 GPU 0).
- Fix: a chroma CU from the CPU whose CCP model its prediction sets (LM chroma fusion's, also beside DBV:
  DBV_CHROMA_IDX with isChromaFusion) had no record, as cpuKnownRecord ran before the model existed, so the CCP merge
  CUs after it lacked that neighbour (fp720_ai POC 2's merge fusion block at chroma 208,128: its non-adjacent
  candidate 9 from 264,72, so another pair won; the chroma reordering DIMD differences of fp720_ai and ra720 followed).
  cpuCuModels gives such a CU its record after the CPU's reconstruction (and the history entry, for an LM mode).
Results (RTX 3070, build grn55, partial.py -j 1, every picture of the 27 streams): identical but ra_ip64_md5 1 of 257
(decoder-derived CCP fusion, section 9), ra720_pj17 5 of 17 and ra_parkjoy_q32 1 of 17 (a CCP record's inside filter
flag) and ldp_parkjoy_q27 1 of 9 (a CCCM model of 2-row chroma blocks; its first difference moves between runs).
M121 (2026-10-26): six inter / IBC ports and a CCP record fix.
- LIC with BDOF (JVET-AG0276's xLicCompAdjBdof): the CU's LIC models on both lists' 14-bit intermediates of BDOF's
  grown block (mcLicBi in inter_mc.comp's bdofJob, the job's MC_LIC and its models at MC_BDOF_OUT), a LIC CU's
  sub-blocks min(8, W) x min(8, H) (applyBiOptFlow's scaleBDOF 2 with LIC) with a SAD threshold without the QP's term,
  the grown block's outer ring interpolated (xPredInterBlk's LIC branch predicts the PU grown by 3; without LIC the ring
  is a replication: the first build replicated it, and 1 - 3 samples on the CU's border rows and columns differed in
  9 streams). With multi-pass DMVR a LIC CU takes xPredInterBi's plain branch at the refined vectors, so "LIC with
  DMVR" is this case too. The luma block's job copies the BDOF prediction (no MC_LIC of its own); chroma is LIC
  bi-prediction.
- Non-local LIC with BDOF: motionCompensation applies the model after xPredInterBiSubPuBDOF, on the final prediction,
  as mcSample does on top of the BDOF copy.
- Affine CIIP with LIC: the models where licApplied (CIIP's condition), BCW's weights then (xWeightedAverage's alwCond),
  in affineRecord; uni-prediction where CIIP's condition leaves LIC out stays on the CPU (xPredAffineBlk applies the
  model it derives regardless, DecCu stores identity models).
- Multi-hypothesis prediction (xAddHypMC): the base prediction as its kind's (affine or interKindBase's), then each
  hypothesis a uni-prediction at its clipped vector (mhpRecord: slot, vector, half-sample flag, weight) added as
  addHypothesisAndClip adds it, in mcSample after non-local LIC (OBMC's skip test reads the same prediction); the
  record at the block job's MC_BDOF_TH, which only BDOF's and GPM's jobs use otherwise. Hypotheses with LIC and weighted
  prediction stay on the CPU.
- GPM bi-prediction parts with BDOF (gpuGeoBio): the part's luma by BDOF's sub-block jobs into pc.bdofOut (the part's
  record MC_BI | MC_BDOF: gpmPart reads its luma there, its chroma plainly), the vectors clipped at the grown position;
  not with the part's LIC.
- IBC bi-prediction with BCW: list 0's weight in IBC_REC_BI, the average addWeightedAvg's (addAvg's at 4 / 4).
- Fix: CU::saveCcInsideFilterFlagInCCP leaves a single tree's CU with luma ISP as it is, so its CCP record's inside
  filter flag stays 0 (LM_RECORD_NO_INSIDE, Tu::lm's sign bit; ra720_pj17, ra_parkjoy_q32).
Results (RTX 3070, build grn62, partial.py -j 1, every picture of the 27 streams): identical but ra_ip64_md5 1 of 257
(decoder-derived CCP fusion, section 9) and ra720_pj17 4 of 17 (CCP records whose first difference moves between
runs, section 9). Inter CUs predicted by the CPU (all streams' counts) 1,083 -> 716, by the GPU 42,423 -> 42,790.
M122 (2026-10-26): CCCM's sums from features computed once per template sample, two separate-tree fixes.
- cccmAccumulate: the template's samples in chunks of 64, each invocation computing one sample's M + 2 features and its
  class into s_spCoef (int16), then the invocations of the matrix's entries summing products from there, instead of
  every entry's invocations computing the features of every sample again. intra_rec's GPU time (median of 3, idle
  3070, against M121): aiodd 364.4 -> 355.6 ms (-2.4 %), ai720 2,406 -> 2,358 ms (-2.0 %).
- Fix: CCCM without subsampling reads the luma right of / below its block where the CU has separate trees
  (xCccmCreateLumaNoSubRef's cu.separateTree: also an inter slice's CU of JVET-AI0136's adaptive dual tree), not only in
  an intra picture's dual tree as with subsampling (LM_SEPARATE_TREE in Tu::lm of CCCM and CCP merge blocks).
- Fix: CCCM with subsampling reads it only in an intra slice's dual tree (xCccmCreateLumaRef's condition); the host
  took CS::isDualITree once per picture, which reads the slice's state of the region decoded last and so could be on
  in an inter picture (Params::dualTree).
Results (RTX 3070, build grn67, partial.py -j 1, every picture of the 27 streams): as M121's, identical but
ra_ip64_md5 1 of 257 and ra720_pj17 4 of 17 (section 9).
M123 (2026-10-26): GPM bi-prediction parts with BDOF and LIC. The part's luma as M121's LIC CUs with BDOF: the models
xLicCompAdjBdof inherits (useInheritedMode: the part's) on both lists' intermediates of BDOF's grown block, sub-blocks
of at most 8 x 8, the SAD threshold without the QP's term (the part's BDOF jobs: MC_LIC, the models in pc.affine); its
chroma every list's model, identities too (xLicCompAdjBdof transforms each list, where xPredInterBlk's geo branch leaves
an identity model out: no MC_GPM in the part's record). Results (build grn71): as M122's, and ldp_parkjoy_q27 1 of 9
again (its stream has no such part; the same shaders had it identical in M122's run: a race, section 9). Inter CUs
predicted by the CPU 716 -> 636.
M124 (2026-10-26): affine PUs with inherited LIC models, predicted without LIC as the CPU predicts them: xPredInterUni's
affine branch turns licFlag off for xPredAffineBlk and tests the model against licFlag while it is still off, so
nonTrivialModel is never true, and xLicCompAdj leaves inherited models to xPredInterBlk. affineRecord gives such a PU
no models; affineWhy's inherited LIC, affine CIIP's uni-prediction with inherited models and gpmWhy's affine part with
inherited models go (gpuGeoBio records the part's licInheritPara as m_gpuGeoAff[k][0]'s second bit; a part with a model
of its own stays on the CPU). Results (build grn72): as M123's (ldp_parkjoy_q27's race at another block). Inter CUs
predicted by the CPU 636 -> 517.
M125 (2026-10-26): affine CIIP's uni-prediction with LIC where CIIP's condition (disCond) leaves LIC out: xPredAffineBlk
derives the model from its templates and applies it anyway, DecCu stores identity models. DecCu hands the GPU the
models InterPrediction holds after the prediction (GpuIntraRec::derivedLic, before it overwrites the CU's);
affineRecord gives the PU those. Results (grn73 / grn74): identical but ra_ip64_md5 1 and ra720_pj17 4 (section 9).
Inter CUs predicted by the CPU 517 -> 402.
M126 (2026-10-26): IntraTMP's positions enumerated in parallel. The node's invocation 0 walked the sparse search's and
the refinement's positions alone (aiodd: 33k and 74k); now it only cuts each chunk's rows (a region's rows, a candidate
window's rows, a row too long for the chunk in pieces whose walk state the next chunk takes on), an invocation per row
walks its positions with the same decisions in the same order (tmpSparseRow, tmpRefineRow), and tmpChunkPack puts them
after each other in the rows' order. intra_rec's GPU time (median of 3, idle 3070, against M125's grn74): aiodd
366.1 -> 310.2 ms (-15.3 %), ai720 2,322.8 -> 2,054.4 ms (-11.6 %). Results (build grn82, every picture of the 27
streams): identical but ra_ip64_md5 1 and ra720_pj17 4 (section 9), and in one run ld_parkjoy_q27 1 of 9 (a CCCM
block without subsampling, Cr 4 x 2 at 100,46 of POC 8; identical in two more runs of grn82 and in one of grn89:
section 9's race).
M127 (2026-10-27): IntraTMP's lists keep their entries in place: a list holds an order of slot numbers and the entries
stay in their slots, so an insertion or an erasure moves one word per entry behind it, not TE (tlSlot, tlPlace).
GPU time (median of 5, idle, against M126): aiodd 324.7 -> 315.8 ms, ai720 2,058.1 -> 2,032.3 ms. Results (grn96,
with M128 - M130 on top): identical but ra_ip64_md5 1 and ra720_pj17 4 (section 9).
M128 (2026-10-27): CCP merge's CCCM candidates reuse the luma buffers of the candidate before where they hold the same
filters' samples (ccpLumaKey: with or without subsampling, several filters' mode, GLM with luma's pattern), each
sample moved by the luma offsets' difference instead of downsampled from the picture again (GLM's gradients, which
have no offset, stay). Phase timers (aiodd, 84 merge blocks): the candidates' model loads 20.7 -> 9.6 ms. GPU time
(median of 5): aiodd 315.8 -> 312.3 ms, ai720 2,032.3 -> 1,987.8 ms. Results as M127's. (Reading the history's and
the list's records ahead in parallel with signatures of the compared words measured no faster: left out.)
M129 (2026-10-27): IntraTMP's refinement walks only the positions that can enter its lists: a position below its
list's threshold must also have fewer than the list's length ranking before it - the list's entries that cost no
more, the chunk's positions of its type below the threshold that cost less or as much and come earlier (of a
clustered type only those not equal to one in the list or earlier) - which every invocation counts for its position
(tmpRefineRanks; not BVG-DIMD's, whose threshold also falls by the list's cheapest). GPU time (median of 5, grn89 ->
grn92): aiodd 312.1 -> 297.5 ms, ai720 1,987.3 -> 1,924.5 ms. Results as M127's.
M130 (2026-10-27): the block vector lists (getItmpMergeCandidate's, the cascade's rounds) read ahead by the whole
workgroup into the end of its scratch - each candidate's vector and validity, each cascade position's CU - and
invocation 0 only compares and appends in shared memory (bvNeighbourList, bvCascade: IntraTMP, TIMD and DIMD nodes).
GPU time (grn92 -> grn93): aiodd 297.5 -> 277.1 ms, ai720 1,924.5 -> 1,838.0 ms. Results (grn96, this state): as
M127's.
M131 (2026-10-27): DIMD's strongest modes ranked in parallel (dimdRanks: every mode counts those ranking before it),
which invocation 0 inserted one by one (dimdSelect). GPU time (grn100 -> grn98): aiodd 277.7 -> 269.0 ms (median of
7), ai720 1,862.2 -> 1,825.8 ms. Results (grn115, M131 - M134 together): identical but ra_ip64_md5 1 and ra720_pj17
4 (section 9).
M132 (2026-10-27): NN intra prediction's narrow layers (fewer outputs than invocations: the models' last layers) with
up to 8 lanes per output, partial sums in s_spCost. GPU time (grn98 -> grn99): ai720 1,825.8 -> 1,811.5 ms, aiodd
unchanged (269.0 / 269.8). aiodd's 163 NN blocks spend ~110 us each in the wide sparse layers 0 and 1 (640 - 1216
outputs of ~5 packs of 8 weights each): chains of loads; loading 4 packs at once and a path without the partial sums
there measured slower overall (section 7's ~1 % from structural changes), left out. Results as M131's.
M133 (2026-10-27): reference line availability (refLineAvailability's prefix walks) tested by the whole workgroup,
each unit by an invocation, each arm's prefix by atomicMin (lineAvailability, in intra_tmrl.glsl): TMRL nodes' lines,
TMRL blocks' lines, LM blocks' templates (cclmAvailability). GPU time (grn101 -> grn104): aiodd 271.6 -> 267.0 ms,
ai720 1,786.2 -> 1,770.7 ms. Results as M131's.
M134 (2026-10-27): CCP merge adjustment's six on-the-fly fits from three passes over the template: a fit of two models
takes both classes in one pass (cccmAccumulatePair), the single model over the same samples from their sums added
(cccmNextClass, cccmBothClasses; CCP_FIT_ORDER). GPU time (grn104 -> grn115): aiodd 268.9 -> 263.8 ms, ai720
unchanged. Results (grn115): identical but ra_ip64_md5 1 and ra720_pj17 4 (section 9). Since M125 (grn74 -> grn115,
median of 5, one session): aiodd 365.8 -> 251.9 ms (-31 %), ai720 2,334.9 -> 1,769.8 ms (-24 %).
M135 (2026-10-27): CIIP and GPM with an intra part of CUs of several TUs. The CPU predicts such a CU's intra part once
for the CU, then adds each TU's residual; the GPU's inter blocks are TUs. A pre-pass per component (ciipSplitPrePass:
recordInterBlock's record of the CU's area, its job undone, no residual or OBMC; CIIP_PRE_INTRA) keeps the CU's intra
prediction in its own place in Params::interPred; each TU block (CIIP_SPLIT: Tu::cuX .. cuH the CU's region, Tu::ibc
that place, the pre-pass its dependency in Tu::ispPrev) blends it into its inter prediction after OBMC's band by the
CU's weights at its position (ciipSplitBlend, ciipBlendAt), then reconstructs as an inter block (crsSetup there too).
Not yet with inter CCCM / CCP merge or ASBT. Results (grn121, with M136, before crsSetup's CIIP_SPLIT case): identical
but ra_ip64_md5 1 and ra720_pj17 4 (section 9), and in one run ldp_parkjoy_q27 1 of 9 (a CCP merge model of a 2 x 16
Cb block at 124,64; identical in a second run: section 9c's race). Inter CUs predicted by the CPU: GPM (CIIP of
several transform units) 130 -> 0 (110 on the GPU; 15 with inter CCCM / CCP merge, 2 with ASBT, 3 other reasons).
M136 (2026-10-27): CIIP of sub-PU motion (xSubPuMC's CUs: affine sub-block merge, SbTMVP), which subPuWhy left to the
CPU: within ciipWhy's limits they predict as other CIIP CUs - the PU's prediction and its OBMC (with bSubMotion's inner
sub-blocks), then the blocks' blend with the intra prediction. Results as M135's (grn121). Inter CUs predicted by the
CPU: affine (CIIP: sub-block merge) 90 -> 2, affine (CIIP: OBMC of sub-block motion) 36 -> 0.
M137 (2026-10-27): OBMC of IBC CUs of two vectors (JVET-AE0169). The prediction and OBMC's plan were there; what
differs is getSameNeigMotion's comparison with the CU's own motion (identicalMvOBMC: the same lists and vectors, then
BCW's index), which now takes the CU's second vector too (IBC_REC_OWN_MV2, obmcPlan's ownMv2: none for a CU of one
vector, a value no neighbour has where the CU's BCW is not the default). ibcWhy's composite bi-prediction reason is
split. Results (grn123, the streams with such CUs): fp720_ai and pj720_ai identical; their intra CUs from the CPU 279
-> 171 ("IBC bi-prediction with OBMC" 109 -> 0, one of them now "above 1024 samples").
M138 (2026-10-28): regression-based GPM blending with an intra part (JVET-AK0101) on M115's GPM-intra path: the inter
part's prediction with its OBMC, the intra part as CIIP's intra prediction with the CPU's mode, blended by the
regression model's weights (gpmBlendWeights; weightedBlend with roundOutputBD off rounds as weightedGeoBlkRounded).
JVET-AM0215's LMCS order: the blend in the original domain (the intra part inverse-mapped: CIIP_ORIG), forward-mapped
as an inter CU's prediction at the reconstruction, a deferred block's after its DIMD of the unmapped blend
(xDecodeInterTexture's deriveIpmForTransform reads the prediction before xDecodeInterTU maps it).
motionCompensationGeoBlend records the intra part's mode and parameters and the inter part's OBMC motion as
motionCompensationGeo does; gpmIntraPart() names both kinds. Not with inter CCCM / CCP merge. Results (27 streams,
grn154 / grn153 with ECM_GPU_FULL_PIPELINE): as before the series, identical but ra_ip64_md5 1 and ra720_pj17 4
(section 9). Inter CUs from the CPU: GPM (regression-based blending: an intra part) 60 -> 0.
M139 (2026-10-28): CIIP of several TUs with inter CCCM or CCP merge: the TU's chroma blocks blend the CU's intra
prediction (M135's pre-pass) into their inter prediction in place before their own blend (ciipSplitBlend in the inter
CCCM and the CCP merge paths), no chroma pre-pass; CIIP_SPLIT finds the pre-pass through Tu::ispPrev's record, as
those blends' records take Tu::ibc. Results (27 streams, grn154 / grn153 with ECM_GPU_FULL_PIPELINE): as before the
series, identical but ra_ip64_md5 1 and ra720_pj17 4 (section 9). Inter CUs from the CPU: 15 -> 0.
M140 (2026-10-28): CIIP with LFNST / NSPT above 32 samples a side through M135's split path (ciipSplit), whose blend
the inter path's DIMD reads in place (no s_spPred). Results (27 streams, grn154 / grn153 with ECM_GPU_FULL_PIPELINE):
as before the series, identical but ra_ip64_md5 1 and ra720_pj17 4 (section 9). Inter CUs from the CPU: 18 -> 0.
M141 (2026-10-28): OBMC of inter CUs with IBC neighbours of two vectors (default BCW, no weighted prediction):
obmcInterIbc's strip the average of the two copies (OBMC_SEG_IBC2, OBMC_SEG_MV2; the segment record 8 words),
skipObmcConditionByPixel per vector. No such CU in the 27 streams: the second copy is not exercised yet. Results (27
streams, grn154 / grn153 with ECM_GPU_FULL_PIPELINE): as before the series, identical but ra_ip64_md5 1 and ra720_pj17
4 (section 9).
M142 (2026-10-28): CIIP with non-local LIC (ciipWhy's rejection removed: the GPU's non-local LIC prediction, then
CIIP's blend). Results (27 streams, grn154 / grn153 with ECM_GPU_FULL_PIPELINE): as before the series, identical but
ra_ip64_md5 1 and ra720_pj17 4 (section 9). Inter CUs from the CPU: affine (CIIP: CIIP with LIC) 24 -> 0.
M143 (2026-10-28): SGPM CUs above 1024 samples: the blend weights after the sums of a 64 x 64 block (SGPM_BLOCK_W), a
deferred block above 32 x 32 samples keeps its prediction in the workgroup's scratch (DEFER_LARGE, deferredPred).
Results (27 streams, grn154 / grn153 with ECM_GPU_FULL_PIPELINE): as before the series, identical but ra_ip64_md5 1
and ra720_pj17 4 (section 9). Intra CUs from the CPU: 37 -> 0.
M144 (2026-10-28): TIMD / DIMD CUs predicting from a block vector with MTSS: MTSS's first try the DIMD of the
prediction's second mode (none for a TIMD CU whose modes do not blend; an OBIC blend's second OBIC mode), g_bvSecond.
Results (27 streams, grn154 / grn153 with ECM_GPU_FULL_PIPELINE): as before the series, identical but ra_ip64_md5 1
and ra720_pj17 4 (section 9). Intra CUs from the CPU: 41 -> 0 (and fp720_ai's intra merge CUs of such neighbours 19 ->
6).
M145 (2026-10-28): DBV with LM chroma fusion's first type (its CCCM blend applies to the DBV copy in storeRegion);
CFLM, IBC LIC and the flip stay on the CPU (their reasons apart). Results (27 streams, grn154 / grn153 with
ECM_GPU_FULL_PIPELINE): as before the series, identical but ra_ip64_md5 1 and ra720_pj17 4 (section 9). Intra CUs from
the CPU: 37 -> 26.
M146 (2026-10-28): OBMC of IntraTMP / IBC CUs with inter neighbours: per run (getSameNeigMotion's) an inter_mc OBMC
job (OBMC_INT_OUT: the luma strip and per list the copy at the rounded vector, obmcBandJob shared with the inter CUs'
jobs), the run an operation of tmpObmc (OBMC_OP_INTER: the skip test against the copies, the strip blended), LMCS
forward-mapped (OBMC_FLAG_FWD; subBlockOBMC makes an IntraTMP CU an IBC CU for its OBMC, so its lut and its mode 2 are
IBC's); inter_mc runs for such jobs alone too. calcGradForOBMC passes a neighbour of motion by its PU, a GPM PU with
an intra or IBC part by one unit (unitsToSkip; ra_long POC 184). Results (27 streams, grn157 with
ECM_GPU_FULL_PIPELINE): as before the series, identical but ra_ip64_md5 1 and ra720_pj17 4 (section 9). Intra CUs from
the CPU: IntraTMP OBMC with an inter neighbour 69 -> 0.
M147 (2026-10-28): OBMC of inter CUs with an IBC neighbour with LIC: the copy through the neighbour's list-0 model
(subBlockOBMC copies licScale / licOffset, xPredInterBlk's isobmcMC branch applies them, skip test copies too, before
LMCS's inverse): OBMC_SEG_IBC_LIC, the models after the record at OBMC_SEG_MV2. Two vectors with LIC stay on the CPU.
Results (27 streams, grn154 / grn153 with ECM_GPU_FULL_PIPELINE): as before the series, identical but ra_ip64_md5 1
and ra720_pj17 4 (section 9). Inter CUs from the CPU: 27 -> 3 (now their other cause, several TUs / ASBT).
GPU tooling (2026-10-28): ECM_GPU_FULL_PIPELINE=1 runs every picture in the largest intra_rec pipeline (MPM variant,
CCP-LFNST level; each holds those below), so a shader change compiles once for all streams (the 27 then verify in
about an hour) instead of once per variant and level a stream uses; a change's verification still runs without it.
M148 (2026-10-28): the last open differences (section 9's items a and c): ccpCccmModel sets g_cccmInterNs for a CCP
merge candidate of inter CCCM, and nothing reset it, so the next CCCM model a workgroup fitted (an intra CCCM block's,
a CFLM block's, decoder-derived CCP fusion's fits) took inter CCCM's features: the first difference moved with the
schedule (c), the derived fusion's CCCM costs came out several times too high (a). cccmPrediction, cflmArea and
ccpFitModels reset it. Found by dumping a block's luma buffer, area, offsets, template chroma (all the CPU's) and its
feature sums (inter CCCM's list) under ECM_GPU_GROUPS=1. Results (27 streams, grn167 with ECM_GPU_FULL_PIPELINE, M148
to M151): every picture identical. Before: identical but ra720_pj17 4 and ra_ip64_md5 1.
M149 (2026-10-28): BVG-CCCM (JVET-AE0100): the host records each position's attempts (bvgRecord: the luma PU's kind,
adjustChromaBv's vector or an IntraTMP CU's record, whose node finds it; the PU of an earlier position ends the
position), intra_cccm.glsl checks them (bvgCandidates, checkIsChromaBvCandidateValid from the owner map), copies each
candidate's luma with padding, Cb and Cr into the scratch (bvgFill: xBvgCccmCreateLumaRef's padding rules, the flip),
fits the plain 11-parameter models to the samples in the block luma's widened range, each picture position once
(bvgSample in cccmAccumulate; MMLM split at the block's mean luma), and predicts from the block's luma buffer. No CCP
record (predIntraCCCM sets no curCand). Its blocks wait for all before them (DERIVE_PREFIX; record() kept assigning
Tu::derive over it until now); a joint CbCr pair whose Cr block decides the signs stays on the CPU (a wait cycle
otherwise). Not with LFNST / NSPT, the inside filter, LM fusion or another CCCM kind. Results (27 streams, grn167 with
ECM_GPU_FULL_PIPELINE, M148 to M151): every picture identical. Intra CUs from the CPU: BVG-CCCM 89 -> 8 (LFNST 6,
joint CbCr 2).
M150 (2026-10-28): DBV with CFLM: the template's arms are copies at the DBV block's refined vector (W x 2 two rows up,
2 x H two columns left; the flip's rows / columns, the vector itself where the arm's is no valid vector for the Cb
block: xCflmCreateChromaPred's DBV branch, cflmDbvArm), the block's chroma feature its flipped copy, its prediction
flipped back before the reconstruction's flip; the host checks the CPU's arms (a PU of another CU beside the block)
are the picture's. Results (27 streams, grn167 with ECM_GPU_FULL_PIPELINE, M148 to M151): every picture identical.
Intra CUs from the CPU: DBV with CFLM 22 -> 0.
M151 (2026-10-28): OBIC with an NN neighbour: the NN CU's luma block derives the DIMD mode of its prediction
(DERIVE_PNN_DIMD) into its DimdOut, the neighbour's entry a negative sample count, the OBIC CU's node waiting for all
before it (DERIVE_PREFIX). Results (27 streams, grn167 with ECM_GPU_FULL_PIPELINE, M148 to M151): every picture
identical. Intra CUs from the CPU: OBIC with an NN neighbour 16 -> 0; with M148 to M151 253 -> 113.
M152 (2026-10-28): IntraTMP CUs above 1024 samples (up to 64 x 64): the prediction at DEFER_LARGE in the scratch
(tmpPredict, tmpLicApply and tmpObmc through deferredSet / deferredPred), the OBMC's copy before OBMC only the band it
blends (tmpBeforeAt), the sub-sample search's padded region above 2560 samples in the node's scratch at TS_FBUF
(g_tfScratch). Results (27 streams, grn170 with ECM_GPU_FULL_PIPELINE, M152 to M154): every picture identical. Intra
CUs from the CPU: IntraTMP above 1024 samples 11 -> 0.
M153 (2026-10-28): OBIC with inter neighbours: the host sums their units' modes (obicInterModes: the motion field's
getIpmInfo, a GPM CU's angle; 16 per unit, class 0) into one OBIC_INTER entry that obicHistogram adds. Also a host bug
it exposed: an OBIC CU rejected after taking its neighbour entries shifted the later CUs' Cu::obic against their OBIC
outputs (and the GPU wrote past them): entries now go in on success, the CPU's output at Cu::obic. Results (27
streams, grn170 with ECM_GPU_FULL_PIPELINE, M152 to M154): every picture identical. Intra CUs from the CPU: an inter
OBIC neighbour 16 -> 0 (14 of them now under their other neighbours' reasons: IntraTMP / MIP / EIP neighbours in inter
slices).
M154 (2026-10-28): inter CCCM with luma ASBT: the inter CCCM record lists the TU's luma records (one, or ASBT's
parts), icccmLuma reads from the part holding a sample; chroma ASBT stays on the CPU. Results (27 streams, grn170 with
ECM_GPU_FULL_PIPELINE, M152 to M154): every picture identical. Intra CUs from the CPU: inter CCCM with ASBT 14 -> 0;
with M152 to M154 113 -> 86.
M155 (2026-10-28): a chroma CU whose co-located luma CU is IBC takes the known record of the mode chroma DM reads
(getCoLocatedIntraLumaMode, syntax) as Cu::lumaCu, so chroma reordering (and its DBV entries) runs on the GPU. Results
(27 streams, grn171, M155 and M156): every picture identical. Intra CUs from the CPU: DBV (dual tree) 17 -> 0.
M156 (2026-10-28): OBIC with IntraTMP / MIP neighbours in inter slices as M151's NN ones (their block's DIMD of its
prediction in DimdOut, a MIP block now with DERIVE_MIP_DIMD; negative count; DERIVE_PREFIX), EIP neighbours by their
CU mode (eipDimdMode, written by their block). Results (27 streams, grn171): every picture identical. Intra CUs from
the CPU: 13 -> 0.
M157 (2026-10-28): chroma TMRL with IBC luma CUs at its five positions (their known mode records). Results (27
streams, grn172, M155 to M158): every picture identical. Intra CUs from the CPU: 6 -> 0.
M158 (2026-10-28): BVG-CCCM with LFNST / NSPT (cclmTransformMode's both components through M149's path). Results (27
streams, grn172): every picture identical. Intra CUs from the CPU: 6 -> 0 (one now under the joint CbCr reason).
M159 (2026-10-28): CCP merge adjustment and decoder-derived fusion with inter CCCM candidates (the check removed: the
adjustment's range and ccpCccmModel's features hold for them; M148's switch was the likely cause of the old
differences). Results (27 streams, grn173, M155 to M159): every picture identical. Intra CUs from the CPU: 13 -> 0;
with M155 to M159 86 -> 24 (DBV with IBC LIC 16, BVG-CCCM with joint CbCr 3, IntraTMP OBMC with an SGPM neighbour 2,
three of one CU each).
M160 (2026-10-29): DBV with IBC LIC: a single tree's DBV copy at a vector with a chroma fraction goes through the
IntraTMP CU's LIC model of its chroma component (tmpLicModel with a component: the luma check of the vector moved by a
chroma sample, the templates at the vector moved by one luma sample with the bilinear chroma filter,
ibcChromaPredict's new filter choice). Results (8 streams with such CUs, grn174): every picture identical; Results (27
streams, grn181 with ECM_GPU_FULL_PIPELINE, M160 to M168 and the block clock): every picture identical. Intra CUs from
the CPU: DBV with IBC LIC 16 -> 0.
M161 (2026-10-29): OBMC with an SGPM neighbour above 1024 samples: the table of SGPM's unit parts holds 8 words per
size and partition (SGPM_UNIT_WORDS: a 64 x 64 CU's 256 units). Results (27 streams, grn181 with
ECM_GPU_FULL_PIPELINE, M160 to M168 and the block clock): every picture identical. Intra CUs from the CPU: IntraTMP
OBMC with an SGPM neighbour 2 -> 0.
M162 (2026-10-29): multi-hypothesis over sub-block motion: subPuWhy rejects only what mhpWhy does (the jobs' MC_MHP
adds the hypotheses to xSubPuMC's PU prediction they read). Results (27 streams, grn181 with ECM_GPU_FULL_PIPELINE,
M160 to M168 and the block clock): every picture identical. Inter CUs from the CPU: multi-hypothesis (sub-block
motion) 9 -> 0.
M163 (2026-10-29): multi-hypothesis hypotheses with LIC: the CPU hands over the models xLocalIlluComp derives for them
(InterPrediction::m_gpuHypLic, GpuIntraRec::hypothesisLic), mhpRecord gives each hypothesis its per-component scale
and offset, mcHypotheses applies them (mcLic); m_gpuHypLic starts with the models the base prediction applied (the
hypotheses can leave another stored: an affine LIC base's 32 / 8 stored as 32 / -1, ld_cif33 POC 24), which the CU's
own LIC records take (cuLicModel). Results (27 streams, grn181 with ECM_GPU_FULL_PIPELINE, M160 to M168 and the block
clock): every picture identical. Inter CUs from the CPU: multi-hypothesis (a hypothesis with LIC) 6 -> 0.
M164 (2026-10-29): BVG-CCCM with joint CbCr whose Cr block decides the signs: such a Cr block (DERIVE_PREFIX, spCoded
the record before it: prefixBeforePair) waits for every record before its Cb block, which intra_rec's prefix release
starts it at. Results (27 streams, grn181 with ECM_GPU_FULL_PIPELINE, M160 to M168 and the block clock): every picture
identical. Intra CUs from the CPU: 3 -> 0.
M165 (2026-10-29): IBC CUs with OBMC above 1024 samples: ibcBlock keeps such a luma block's prediction in the scratch
(g_deferLarge, deferredSet / deferredPred), as M152's IntraTMP blocks. Results (27 streams, grn181 with
ECM_GPU_FULL_PIPELINE, M160 to M168 and the block clock): every picture identical. Intra CUs from the CPU: 1 -> 0.
M166 (2026-10-29): CIIP / GPM with an intra part of ciipSplit with ASBT: every record of an ASBT block blends the CU's
intra prediction at its own part (ciipSplitBlock per record). Results (27 streams, grn181 with ECM_GPU_FULL_PIPELINE,
M160 to M168 and the block clock): every picture identical. Inter CUs from the CPU: 2 -> 0.
M167 (2026-10-29): inter OBMC with an IBC neighbour over several TUs or ASBT parts: the OBMC record's word 9
(OBMC_REC_PARTS) names the CU's records on its band; the first one on a strip copies it and tests it against the
prediction of the record holding each sample (obmcStripFirst, obmcCuPred). Results (27 streams, grn181 with
ECM_GPU_FULL_PIPELINE, M160 to M168 and the block clock): every picture identical. Inter CUs from the CPU: 8 -> 0.
M168 (2026-10-29): LM chroma fusion in inter slices: MMLM's two CCLM models (not CCCM's), weights 2 : 2, the CCP
record CCLM | MMLM (fusionCclm in intra_rec, fitted from the references as the block's own mode has them - filtered
where it filters them: getPredictorPtr -; cclmDeps in intra_deps and the host's mirror). Results (27 streams, grn181
with ECM_GPU_FULL_PIPELINE, M160 to M168 and the block clock): every picture identical. Intra CUs from the CPU: 1 ->
0.
Block clock (2026-10-29, tooling): the CMake option ECM_GPU_BLOCK_CLOCK (off by default) makes intra_rec read the
device's realtime clock at each block's start and end (Params::blockClock) and the decoder print per picture the sums
per kind of record ('[gpu-intra] POC n block clock: ...'); perf/gpu/clocksum.py sums a build's partial-verification
logs per stream and kind. It replaces the temporary phase timers of section 8.
Block clock phases (2026-10-29, tooling): with the option, intra_rec's clockMark( k ) also marks the end of phase k
of the running record (up to 8 per record, after the starts and ends in the buffer), and the decoder prints per
picture the phases' sums per kind ('[gpu-intra] POC n phases of <kind>: 0 <ms>; ...', the last up to the record's
end). Marked: IntraTMP's node (setup, sparse search, the vector list's cascade and costs, clustering, refinement, then
the list), CCP merge blocks (the list, the block's own fits, the candidates' costs, the choice), NN blocks (the input,
the layers, the output).
M169 (2026-10-29): IntraTMP's chunks in shared memory without packing: a chunk's rows write their positions straight
into their slots (x, y, tag as int16 in s_angL[0], -1 marking a slot the row's walk left out), their costs in
s_angL[1] (tmpChunkClear zeroes both after the node), instead of the scratch with invocation 0 packing the rows'
positions (tmpChunkPack's serial loop and 3 barriers per chunk); the sparse search's ranks read the position's own
costs from registers; tmpLicModel keeps the scratch's start (TS_LIC). GPU time (27 streams, one session, grn181 ->
grn189): 58,420 -> 58,160 ms (-0.4 %), all-intra -1 to -2 % (fp720_ai 14,432 -> 14,223 ms, ai720 1,812 -> 1,790 ms),
the others within +-1 %. Results (grn189): every picture identical.
M170 (2026-10-29): IntraTMP's lists merged in parallel after each chunk (tmpListMerge) instead of invocation 0's
walk inserting the chunk's positions one by one: a position that enters a list has its place there after the chunk
in its rank (the list's entries and the chunk's positions before it, which tmpChunkCosts / tmpRefineRanks count
already), so each such position writes its entry there, each other place takes the list's next entry (its bit masks,
s_tmpTaken), and the lists go back in order, all three in one pass. The sparse search merges every chunk; the
refinement merges where no position that may enter repeats one in its list or before it in the chunk (a clustered
type's), else, and for BVG-DIMD (whose threshold also falls by the cheapest), it walks as before (s_tmpSerial).
GPU time (27 streams, one session, grn189 -> grn198): 58,254 -> 56,617 ms (-2.8 %); ra720 2,363 -> 2,158 ms (-9 %),
ld_foreman_q22 -8 %, fp720_ai 14,378 -> 13,605 ms (-5 %), ai720 1,803 -> 1,712 ms (-5 %); ra720_pj17 +1.4 %,
ld_parkjoy_q27 +0.9 %, the rest faster or within 1 %. Results (grn198): every picture identical. A
first version (grn196) put intra_rec's shared memory 180 bytes over the 49,152-byte limit with a word per place:
NVIDIA's driver created the pipeline and the device was lost on every stream; shared_size.py checks that now.
M171 (2026-10-29): IntraTMP's clustering in parallel: per vector of the list and list type, every invocation tests
an entry of the list (at the vector, or its window overlapping the vector's) and atomicMin finds the first (s_tmpHit:
the index and what it found, two slots by turn so that invocation 0 never resets the one still being read); invocation
0 then changes the list as before (replace, take in, or a new entry). The list's walk, entry by entry, was most of
the clustering (197 of the node's 1,157 ms on aiodd, ai720 and ra720). The vectors' costs come straight from the
cascade's chunk (TS_MCOST's copy gone). GPU time (27 streams, one session, grn198 -> grn205): 56,588 -> 52,918 ms
(-6.5 %), every stream faster: fp720_ai 13,610 -> 12,430 ms (-8.7 %), ai720 1,716 -> 1,567 ms (-8.7 %), ra720 2,155 ->
1,973 ms (-8.4 %), the long RA streams -0.5 to -1.2 %. Results (grn205): every picture identical. A first build lost
the device: its loop ran over
numList, which only invocation 0 has (the others' 0), so the invocations met different barriers - a loop with
barriers needs a bound every invocation shares (s_tmpCount here).
M172 (2026-10-29): the last CU from the CPU - "an OBIC neighbour that is not a recorded CU" (ra720, 1 CU) was an IBC
neighbour: obicNeighbours lists one whose motion field keeps intra modes, but deriveObicMode has no branch for an IBC
(or palette) CU, so it counts nothing; addObicNeighbours now leaves it out instead of looking for its record. Results
(27 streams, grn225 with ECM_GPU_FULL_PIPELINE): every picture identical, CUs from the CPU 1 -> 0 (none left on the
test streams). Host only: the shaders' SPIR-V is unchanged.
intra_rec's register mode (2026-10-29, after M172; the logs' "[gpu] pipeline intra_rec ... Register Count"): NVIDIA's
driver compiles it at 255 registers (M171, M172) or at 128 (two workgroups per SM), and nearly every change of some
size since landed on 128, costing ~4-7 % of the whole kernel (paired, ai720 ~1,550 -> ~1,600-1,670 ms). Not the
binary size (the sign-prediction merge below cut the inlined size 5.7 % and got 128), not SM packing (46 groups
instead of 32 change nothing in either mode). Changes measured in 128 mode against each other, so their own effect:
IntraTMP's sub-sample costs split over 8 invocations per candidate (-45 ms ai720), NN's wide layers two outputs per
invocation without barriers (-30 ms in 128 mode, neutral in 255), the CCP merge list's and history's comparisons as a
parallel bit matrix (neutral), signPrediction's three inter call sites merged into one (inlined size -41k
instructions, neutral), IntraTMP's sparse ranks over the positions below the threshold only (gathered by atomics or
ballots; +5 % against 255, its own effect not measured). Together the first two in 128 mode: 27 streams 52.22 ->
54.23 s (+3.9 % against M172's 255). Kept out until a way to keep 255 (or to make 128 as fast) is found.
Where the time goes before intra_rec (2026-10-29, grn231; each POC line now prints "before intra_rec x ms: motion a,
residuals b"): over the 27 streams GPU 51.95 s, of it inter_mc's motion compensation 12.1 s (23 %; ~90 % of the RA
streams' time), and of that BDOF with DMVR / sub-PU (modes 1 and 2, before their one barrier) most: ra_ip64_md5
4,161 ms against the blocks' 401 ms, ra_long 3,058 / 585, ra720_pj17 874 / 105, ra_cif33 728 / 68.
M173 (2026-10-29): DMVR PUs with BDOF in passes across the GPU. Timed apart (a barrier between them), BDOF's
sub-block jobs were 15 ms of ra_ip64_md5 and DMVR's PU jobs the rest: one 64-invocation workgroup per PU (one SM, two
warps) ran the first pass, two iterations and the last pass block after block - a 128 x 128 PU 66-103 ms, a 32 x 32
one 14 ms -, with every tile sample the 144 reads of the 12-tap 2D filter and the tiles in pc.scratch. Now a PU of at
most 128 x 128 (DM_PASSES, not xSubPuMC's) goes to inter_dmvr, a second module of inter_mc.comp (DMVR_PASSES): four
dispatches (the first pass's stripes, each iteration's rows, the last pass's rows; a barrier between, BDOF's and mode
2's jobs beside the last), a workgroup per stripe - the rows of a pass read and write only their own blocks' offsets -,
each run's tile in shared memory (int16, 20 KB with the horizontal stage; in a module of its own so that inter_mc's
other modes keep their occupancy), the filter's two stages apart (24 reads per sample), and in every run (both
modules) each BDOF block by a cluster of 8 invocations with clustered subgroup sums, without the barriers between
blocks. (The clusters alone, still in pc.scratch, were slower: ra_ip64_md5's motion 4.59 -> 5.88 s, likely the
uncoalesced reads of four blocks per warp.) GPU time (27 streams, grn231 -> grn235): 51,952 -> 43,658 ms (-16.0 %):
ra_ip64_md5 4,945 -> 1,343 ms (x3.68), ra_long 4,253 -> 1,582 (x2.69), ra_parkjoy_q32 964 -> 512, ra_foreman 221 ->
118, ld_md5 385 -> 260, ra_cif33 1,488 -> 1,060, ra720_pj17 4,031 -> 3,541; the AI streams unchanged. Results
(grn235): every picture identical. Left in mode 2: xSubPuMC's groups with BDOF (up to 128 x 128: one ra720_pj17
picture's 72 ms) and DMVR PUs without it (inter_dmvr's next step), its runs.
M174 (2026-10-29): inter_dmvr takes xSubPuMC's groups with BDOF (or m_skipAffineFirstIterBdof's) and the DMVR PUs
without BDOF, up to 128 x 128. A group by rows of BDOF blocks: the first pass's rows each predict their own rows of the
group's tile (dmTileRows: the rows they read, 3 around, the group's ring as in the whole tile) and refine their blocks
(dmRefineRows), the iteration's and the last pass's rows as a PU's (DM_ZERO's offsets 0 by the last pass's rows); a PU
without BDOF the first pass's stripes alone. Mode 2 keeps xSubPuMC's runs and larger jobs. Motion compensation
(grn235 -> grn239): ra720_pj17 446 -> 249 ms, ra_cif33 371 -> 140, ra_ip64_md5 977 -> 685, ra_long 984 -> 824. GPU time
(27 streams, one session each): 43,658 -> 42,850 ms (-1.8 %; the AI streams, untouched, ~0.8 % slower in that
session); paired (alternating, two rounds): ra720_pj17 3,546 / 3,562 -> 3,477 / 3,446 ms, ai720 1,564 / 1,552 ->
1,550 / 1,540. Results (grn239, 27 streams): every picture identical.
M175 (2026-10-29): intra_deps' passes 1 and 2 (the read ranges, availability and dependencies; timed apart with a
barrier after intra_resi, whose "residuals" timestamp had no barrier and so counted in "deps") took 2-6 ms per picture
even at 132 blocks: one invocation per block walked the owners along its reference lines (8x its sides), the areas
(templates, DIMD's regions, CCCM's) and the modes' read ranges one after the other, each new owner looked up in the
32-entry list in local memory. Now a subgroup per block: its lanes run the block alike (the first lane alone writes),
split those scans (addLaneDeps adds each step's owners in ascending order; availablePrefix's ballot finds an arm's
first unit not decoded) and hold the 32-entry list one entry per lane (a lookup is one subgroupAny). Only the
dependencies' order differs, so the duplicates beyond the list - the host compares them as sets. The list in local
memory alone kept most of the cost (ai_md5 2.5 -> 1-2 ms per pass); with it spread: 0.36 ms. Pictures of ~11,000
blocks keep their 5.7-6 ms (fp720_ai) or lose (pj720_ai's deps 110 -> 171 ms): 32 lanes per block cost throughput
there. Deps over the 27 streams 1,092 -> 557 ms; GPU time (grn239 -> grn244): 42,693 -> 42,017 ms (-1.6 %): ai_md5
658 -> 541 ms, ai_long 905 -> 795, ld_md5 248 -> 220, ra_ip64_md5 1,083 -> 996. Results (grn244, 27 streams): every
picture identical, the scheduling data the host's.
M176 (2026-10-29): the blocks' jobs (inter_mc's mode 0) in bands of rows. Timed apart (barriers between inter_mc's
segments), mode 0 was most of the motion compensation left (ra_long 590 ms, ra_ip64_md5 405, ld_md5 135, ld_cif33 106)
with only ~50 jobs per picture: a workgroup filters its block's samples 64 at a time, a 64 x 64 block 64 rounds of the
12-tap 2D filter. The host now splits a job of more than MC_BAND_SAMPLES (256) samples into bands of whole rows,
each a job of its own (the first in the job's record - OBMC's jobs name the CU's block jobs by index -, the others
after the jobs; MC_DST and MC_GPM_SEP's part 1 at the band's rows). Host only. GPU time (27 streams, grn244 ->
grn246): 42,017 -> 40,587 ms (-3.4 %): ld_md5 220 -> 103 ms (x2.14), ra_long 1,386 -> 834 (x1.66), ra_ip64_md5 996 ->
650 (x1.53), ld_cif33 495 -> 404 (x1.22). Results (grn246, 27 streams): every picture identical.
Tried after M176 and left out (2026-10-29): intra_deps' passes 1 and 2 with a cluster of 8 invocations per block
(clustered subgroup operations) instead of M175's subgroup: pj720_ai's deps 171 -> 145 ms, but ai_md5 24 -> 62,
ai_long 28 -> 79, ld_md5 8 -> 17 (the scans 4 times as long per block). To keep intra_rec at 255 registers: a branch
that never runs (a negative block count) holding 224 loaded values live at once, at the start of the block loop, on
the sign-prediction merge (a 128-register variant): still 128 - the driver spills such a peak rather than raising
the count. (The build after that experiment, with HEAD's intra_rec, compiled intra_rec again although runs/pipecache
names its files by the SPIR-V's hash; the cause was not found.) Twice as many workgroups at 128 registers (64,
ECM_GPU_GROUPS; two fit per SM): ai720 and fp720_ai unchanged - the loss is per block (spills), not parallelism.
intra_rec at 128 invocations (two workgroups per SM at 255 registers, so that 128 would gain the driver nothing): the
device was lost - some code takes an invocation per element of up to 256. The drivers' statistics: every 255-register
build has 10,672 bytes of local memory per invocation, the 128-register ones ~11,400 (the spills). The rest is likely
local arrays: the SPIR-V's function-scope arrays indexed at run time include glslang's "indexable" copies of constant
tables (obmcSample 1 KB, ibcChromaTap, ibcTap, coLumaDbvSample, cccmDownsample, glmGradient, eipTap*, mpm*, timd*,
applyModeParams) and Tu copies of 436 bytes. Within the 128-register mode a change moves the spills too: HEAD + CCP
merge's linear candidates' costs at once (grn249) ai720 1,651 ms against the sign-prediction merge's 1,670 (both
128), but with IntraTMP's fractional costs over 8 invocations and NN's wide layers added (grn251) 1,764. (Nsight
Graphics' GPU Trace with the real-time shader profiler would show where: it needs the driver's GPU performance
counters allowed, a setting for the user: ngfx.exe --no-timeout --activity "GPU Trace Profiler" --exe <decoder>
--real-time-shader-profiler --auto-export, launched by gpuslot.run; it compiles the pipelines again.)
Tried after M125 and left out (idle 3070, median of 5 or 7): IntraTMP's lists in the angular arrays (s_angA, a word
as two int16; paired: aiodd +19 %, ra720 +41 %, the clustering 6.6 times as long - likely the index division and the
two loads per word in invocation 0's walks, against the scratch's one load; with the division gone, M169's chunks
gain); the CCP merge candidates' costs grouped by their luma buffers' filters, each set built once (neutral: the
rebuilds are mostly each block's first); the CCP merge history's and list's records read ahead in parallel with
signatures (neutral); IntraTMP's
refinement ranks counting only the windows that overlap no other
(+3 % aiodd: the weaker filter sends more positions to the walk) or comparing only positions of overlapping windows
(+7 % aiodd); the sparse cascade read ahead (neutral); MDIP's modes ranked in parallel (neutral); the cascade's slots
loaded at once (+1.5 %); the block vector cascade read in two passes (neutral); NN layers with 4 packs loaded at once
and no partial sums for the wide layers (+1 % ai720). Section 7's codegen effect (~1 % from structural changes in
intra_rec) makes gains below ~2 % hard to see.

## 8. Numbers (i7-10700 8C/16T, Win11, GCC 16.2, RTX 3070 driver 616.92)

GPU kernel time, 24 M1/M2 pictures: 2845.5 ms (first kernel) -> ~55 ms (~52x; step table in DECODER.md).
Per 720p picture: M2 ~5 ms, M3 8.3 ms (park720) / 12.8 ms (four720); M3's angular BF gate reads the full
reference arrays, which roughly doubles the longest dependency chain (park720 1127 -> 1761, four720 1198 -> 2325;
without those reads 1081 / 1192). A restructure of the BF with subgroup-only reductions was slower (dropped).

Wall time, m3_park16 (16 x 720p intra, LTO, best of 3) at ddf77bcb:
CPU 1.51 s, GPU on 1.52 s (3 of 16 pictures still on the CPU while the device starts), CPU FT8 0.73 s.
GPU on with FT8 was 1.37 s at the build before ddf77bcb (GPU engaged in every context from the start): with frame
threads the GPU path is currently a net loss. 180ccabc (scheduling on the GPU) is not timed yet.

Profile of a GPU-on decode (non-LTO, m2_park16, share of all samples): parsing ~29 %, derive-only pass ~19 %
(MPM derivation getIntraMPMs -> getSpatialIpm -> getPURestricted 6.6 %, xIntraRecBlk 8.7 %), endPicture 14 %
(host dependency analysis ~7 % - now moved to the GPU by 180ccabc - and the GPU wait ~5 %), loop filter CPU work
9 % (even with LoopFilterDisable), Vulkan start-up 17 % (now hidden by background creation).

Intra pictures of the full tool set (2026-10-25, M119's work, RTX 3070): aiodd's one picture (CIF-sized, every
intra tool), whose whole decode takes the CPU 0.14 s, took 0.94 s of GPU time in intra_rec at M119's start, 761 ms
with its ports and slices (grn22), now 370 ms (grn39; median of 3, idle GPU); ai720 (720p: 0.45 s on the CPU) 4,427
-> 2,369 ms. A temporary per-block timer (the device's realtime clock
at each block's start and end) shows the blocks' durations adding up to the picture's GPU time: the picture runs along
one chain of dependencies, the other 31 workgroups idle. IntraTMP's nodes took two thirds of it (600 ms of aiodd's
940); timers of their phases (lid 0, after the barriers) put 470 ms in invocation 0's list updates (the refinement's
351, the sparse search's 117: per position global loads of the lists and positions, in a single invocation), 57 ms in
its enumeration of the positions, 26 ms in the clustering, and 3 ms in the costs themselves. The survivor filters and
the workgroup-scope scratch (section 7, M119) leave (aiodd, with the timers): IntraTMP nodes 147 ms (the enumeration
57, the list updates 46, the setup 15, the clustering 8), chroma blocks of a known mode (CCLM / CCCM / CCP kinds) 79 ms
(130 blocks), TIMD nodes 49 ms (190), luma blocks of a known mode 25 ms (193), MPM nodes 17 ms, TMRL nodes 15 ms,
SGPM nodes 13 ms. The pipeline's size does not matter: ai_md5's 33 pictures took 910 ms in their own pipelines and
918 ms all in the CCP-LFNST one (ECM_GPU_MIN_LEVEL). Unrolling the costs' loads (16 in flight per invocation) changed
little (the costs are few per node: ~1,100 positions).
Such pictures exceeded Windows' GPU timeout (2 s: VK_ERROR_DEVICE_LOST, nvlddmkm event 153, the device fault
report only instruction pointers): every 720p intra stream was lost on this PC once the CCP-LFNST pipeline compiled
within the budget (M118's 3070 runs skipped those pictures with ECM_GPU_NO_CCP_LFNST). intra_rec now runs in slices
(see section 7, M119).

## 9. Next steps (in the order I would take them)

000. Now (2026-10-26, after M121; RTX 3070 PC): every picture verifies (no ECM_GPU_NO_* switch needed). Open
   differences, all of them older (pictures the switches skipped before):
   a. Fixed by M148 (a CCP merge candidate's inter CCCM switch left set). It was: decoder-derived CCP fusion in an
      inter picture (ra_ip64_md5 POC 8, chroma 4,32 4x16: CPU type 16, GPU 1; also
      4x16 blocks of ra720_pj17 POC 4): the CPU's mode costs (decoderDerivedCcp) 250 for CCLM, 83 / 82 for CCCM
      without / with the inside filter, 237 / 251 for MMLM, 285 for GL-CCCM, then the CCCM pair (cost 79); the GPU's
      CCLM cost agrees (250) but its CCCM modes' are 461, 468, 479, 455 and 820. Its first fitted model (the two
      lines') equals the CPU's (all 7 parameters, area 12 x 22, 52 samples), yet its probes of that model's step are
      smoother (Cb above 472 472 472 471, left 470 .. 469 against the CPU's 468 .. and 466 .. 478): the model as the
      step reloads it from its record or the luma the step rebuilds differs.
   b. (Fixed by M121: a CCP record's inside filter flag.)
   c. Fixed by M148 (the same switch). It was: CCP records whose first difference moves between runs:
      ldp_parkjoy_q27 POC 8 (a CCCM model of a 2-row chroma
      block, no subsampling at 104,64 4x2, or GL-CCCM at 96,26 8x4; identical in M121's run), ra720_pj17 POCs 4, 8,
      10 and 12 (a CCCM / MDF-CCCM record's parameters or type, a CCP merge block): look for a race or for luma
      samples a block reads before it writes them (the shared luma buffer in the angular arrays) first. Also
      ld_parkjoy_q27 POC 8 in 1 of 4 runs (2026-10-26/27: M126's, M133's): its first block a Cr CCCM block without
      subsampling of 2 rows (100,46 4x2; 96,24 8x2, then the CCLM record at 96,26 8x4) - the same kind as above,
      identical in the other runs. And ldp_parkjoy_q27 POC 8 in 1 of 2 runs of grn121 (M135): a CCP merge model of a
      2 x 16 Cb block at 124,64 (CPU 2137, GPU -2995) - again a 2-sample-wide chroma block.
   d. Performance (section 8, M126 - M134): aiodd's GPU time per kind now (phase timers, under load): IntraTMP nodes
      ~90 ms (updates ~23, enumeration ~18, ranks ~10, setup ~10, clustering ~9), CCP merge blocks ~44 (candidates'
      template samples 12, model loads 10, fits ~4), TIMD nodes ~41 (the vector list ~13, the costs ~10), NN blocks ~30
      (layers 0 and 1: ~110 us per block, latency-bound loads of their sparse weights), MPM nodes ~19 (MDIP's list
      4, its costs 4, the end 5), SGPM ~18, TMRL ~14. Next: the CCP candidates' steps batched (one luma buffer per
      filter kind, the candidates' offsets applied in the features), the NN weights' loads, the MPM node's serial
      parts.
   e. Ports by count (M121's run): GPM CIIP of several TUs 130, GPM affine parts with LIC 118, affine CIIP with LIC
      of uni-prediction 115, affine CIIP of sub-block merge 90, GPM bi-prediction parts with BDOF and LIC 82 (as
      M121's LIC with BDOF, the part's inherited models), GPM regression blending with an intra part 60; intra: IBC
      bi-prediction with LIC / flip / OBMC 158, BVG-CCCM 89, IntraTMP OBMC with an inter neighbour 69.
00. (2026-10-24, the inter phase, after M117; on the RTX 3070 PC):
   a. Open differences of the 3070 PC's streams: ai_long POC 5 (NN's indicesRepresentationPnn). The others of
      M117's time are gone with M118 (the non-local LIC OBMC plan, SGPM CUs of a 64-sample side) and the device
      losses with the OBMC strips' scratch fix (section 7); the pictures ECM_GPU_NO_DERIVED_CCP / NO_CCP_LFNST skip
      are not verified on this PC.
      ECM_GPU_CPU_KINDS narrows a difference to a kind (sub-PU CUs go their own way, not through it). A bdofOut /
      buffer read-back beside verification's download (endPicture) shows what a job wrote.
   b. Done (2026-10-25): intra_rec's pipelines compile within the budget again (section 7: the luma helpers' taps in
      loops, 1.21M -> 0.76M inlined). Further cuts, if needed: decodedBefore's 342 copies (refLineAvailability, the
      IBC validity checks), cccmFeature's 33, glmGradient's 113.
   c. Then the CUs still on the CPU by count (M118's results list them): the intra merge cascade first (give a
      neighbour from the CPU its CPU-made DimdOut - the values verification compares - uploaded, so that it stays a
      candidate), then IBC / IntraTMP OBMC with inter neighbours (their band from inter_mc, the blend in intra_rec),
      CIIP with inter CCP merge, GPM CIIP with LIC, IBC bi-prediction, EIP merge's temporal models.

0. Earlier (2026-10-15, the intra phase; the user's order: all intra first, then inter), by the CPU CUs they free:
   a. Done (2026-10-16): M67, committed on dec-gpurecon from the WIP commit's diff (the branch
      dec-gpurecon-m67-wip keeps the WIP commit as it was pushed).
   a2. Done (2026-10-22, M77; found 2026-10-16): the two single-tree intra
      streams (`--DualITree=0`; CTC's intra slices are dual tree) differ in partial verification - ai_nodualtree_q27
      (a CCP merge chroma block inherits the wrong model type, CCLM instead of CCCM, first seen as NN's
      indicesRepresentationPnn) and ra_nodualtree_q32 (a luma MPM block's prediction, then DIMD chroma's second
      mode). ON mode leaves both to the CPU (identical output). Fix before single tree counts as supported.
   b. Done (2026-10-16): M68, IBC's interpolation (intra_ibc.glsl) and SGPM's fractional vectors.
   b2. Done (2026-10-16): M69, IntraTMP's sub-sample vectors.
   b3. Done (2026-10-21): M70, IntraTMP's OBMC. Before the next milestone: room in the CCP-LFNST module (26k IDs).
   c. Done (2026-10-22): M71-M73 IntraTMP's LIC, fusion and FLM (and OBMC of LIC CUs), M74 intra merge's IntraTMP
      neighbours. Left: "an intra merge neighbour of a kind the GPU lacks" (279),
      "IntraTMP OBMC with a neighbour with LIC" (102: the neighbour's IBC LIC of the strip).
   d. Regression SGPM (795: deriveSgpmBlending's least-squares blend model), MTSS of the remaining kinds (~200), DBV
      (~200), NN of split CUs (~150), BVG-CCCM (~70).
   e. ON mode on the CTC streams: chroma reordering / DBV (the sequence-level checks in supported()), then inter.
1. Time 180ccabc (paired grL builds, `walls.py` with `--GpuRecon=3`, CPU vs GPU, FT8; use m3_park60).
2. Done (2026-10-11): M4 dual tree, M5 LMCS, M6 dependent quantization and QCS, M7 joint CbCr, M8 explicit MTS,
   M9 transform skip and BDPCM, M10 MIP, M11 LFNST / NSPT (computed by the host until M15), M12 MRL, M13 ISP, M14 sign
   prediction, M15 LFNST / NSPT on the GPU (the VVC kernels, LFNSTExt=0, rejected), M16 sign prediction of LFNST /
   NSPT blocks. 2026-10-12: M17 intra modes derived on the GPU (tested with `ECM_GPU_MODES=1`), M18 the transforms of
   derived modes on the GPU, M19 DIMD (the m19d test set), M20 OBIC (the m20 and m20o test sets), M21 intra merge
   (the m19 set, encoded by the stock encoder with DIMD: OBIC and intra merge everywhere), M22 the transforms of DIMD
   CUs and MTSS (the m22 and m22o sets), M23 LFNST / NSPT of MIP (the m23 set), M24 TIMD (the m24 and m24s sets),
   M25 OBIC and intra merge with TIMD neighbours (the m25, m25o, m25s and m25os sets), M26 TIMD-SAD (the m26 and
   m26o sets), M27 OBIC and intra merge with TIMD-SAD neighbours (the m27 and m27o sets), M28 TIMD merge (the m28 and
   m28m sets). 2026-10-13: M29 TMRL of luma (the m29 set, m29n_fore32, m29s_screen32 / 22), M30 chroma TMRL (the m30
   set), M31 CCLM (the m31n set, m31a_fore32 / 22, m31c_fore32 / 22), M32 LFNST / NSPT of CCLM (the m31 set), M33
   GLM's gradient patterns (the m33 set, m33l_fore32 / 22 / 37, m33a_fore22, m33c_fore22), M34 CCCM (the m34 set,
   m34a_fore22, m34c_fore22), M35 GL-CCCM (the m35 set), M36 CCCM with several downsampling filters (the m36 set),
   M37 CCCM without subsampling (the m37 set, m37a_fore22, m37c_fore22), M38 CCCM with LFNST / NSPT (the m38 set),
   M39 GLM with luma (the m39 set, m39a_fore22, m39c_fore22), M40 LM chroma fusion of CCCM (the m40 and m40f sets),
   M41 CFLM (the m41 and m41f sets), M42 GLM with luma with LFNST / NSPT (the m42 set), M43 the SPS switches of the
   CCP tools (the m43 and m43c sets). 2026-10-14: M44 the LM CUs' CCP models, M45 those of LM chroma fusion CUs and
   the inside filter's flags (verified on the LM sets), M46 CCP merge (the m46 set, m46u_four720, h46_crowd), M47 CCP
   merge fusion with CCCM (the m47 set), M48 with DIMD chroma (the m48 set), M49 CCP merge adjustment (the m49 set),
   M50 decoder-derived CCP fusion (the m50 set), M51 decoder-derived CCP merge fusion (the m51 set), M52 CCP merge
   fusion with CCCM by a blend model (the m52 set). 2026-10-15: M53 CCP merge and decoder-derived fusion with LFNST /
   NSPT (the m53 set; intra_rec's second module), M54 its merge fusion with DIMD chroma, M55 PDP, M56 NN intra
   prediction, M57 the MPM lists of MPM sorting, MDIP and PDP, TIMD / TMRL on the regular streams, M58 EIP, M59-M61
   block vectors: TIMD, DIMD and OBIC with vectors, the scheduler's prefix waits (partial verification of the regular
   streams), M62 IntraTMP's search (its plain CUs; LIC, sub-sample vectors and OBMC need inter's LIC, interpolation
   and OBMC, fusion and FLM their models), M63 EIP with a vector, M64 intra merge with vectors, M65 BVG-DIMD, M66
   SGPM (its CUs without vector candidates or regression, up to 1024 samples). 2026-10-16: M67 SGPM's block vectors
   and their LIC models, M68 IBC's interpolation and SGPM's fractional vectors, M69 IntraTMP's sub-sample vectors.
   2026-10-21: M70 IntraTMP's OBMC.
   All 19 regular streams now stop at "chroma reordering / DBV" (the next sequence-level check; the block-level tools
   of step 4 follow; BVG-CCCM needs IntraTMP / IBC first).
3. Make the GPU path win on CPU time per picture: the derive-only pass (record during parsing or a leaner pass;
   speed up getSpatialIpm/getPURestricted, which also helps the CPU decoder), GPU behaviour with frame threads
   (8 contexts x 32 persistent workgroups oversubscribe the GPU; spinning workgroups; try fewer groups per context,
   more frame threads than cores to hide GPU waits, overlapping a picture's GPU work with the next picture's parse).
4. More tool levels. CTC turns on DIMD, TIMD, TMRL, MPM sorting, MDIP, PDP and chroma reordering together, and every
   one of them makes the intra modes depend on reconstructed samples (DIMD's modes are in every MPM list), so with
   CTC the GPU derives every mode (M17's nodes) and everything that depends on a mode moves to the GPU too (M18 did
   the transforms, M19 DIMD, M20 OBIC, M21 intra merge). ECM's encoder uses OBIC and intra merge whenever DIMD is on
   and has no switch for them; for tests of fewer tools, test encoders built with a local patch to IntraSearch.cpp
   that is not committed (`perf/gpu/enc_m1.ps1` with ECM_ENC_EXE):
   `bin\encNoOM` (m19d: ECM_TEST_NO_OBIC_MERGE skips both), `bin\encTest` (m20: ECM_TEST_NO_INTRA_MERGE;
   ECM_TEST_NO_OBIC skips OBIC), `bin\encTest2` (the same, and for m20o ECM_TEST_PREFER_OBIC, which skips plain DIMD
   where OBIC is available, for many more OBIC CUs) - the patch adds `&& !getenv(...)` to `testObic` /
   `testIntraMerge` and sets `m_skipDimdMode` / `m_skipDimdLfnstMtsPass` where OBIC is tested; `bin\encTest4` (m24)
   adds ECM_TEST_NO_TIMD_SAD (to `testTimdSad`, in estIntraPredLumaQT and reduceHadCandList); `bin\encTest7` (m25o)
   adds ECM_TEST_TIMD_CHECKER=<log2 cell size> (`testTimd`, `m_skipDimdMode` and `m_skipDimdLfnstMtsPass` off in the
   cells of the CU's top left where the cell indices' sum is odd; with ECM_TEST_TIMD_SPARSE in all cells but those
   whose indices are both multiples of 4) and ECM_TEST_FORCE_OBIC (an OBIC mode's RD cost in estIntraPredLumaQT divided
   by 4; xCheckRDCostIntra recomputes the CU's cost from its bits and distortion, so the partitioning sees the true
   cost); `bin\encTest8` (m26o) adds ECM_TEST_PREFER_TIMD_SAD (`testTimd` off where `testTimdSad`); `bin\encTest9`
   (m29) adds ECM_TEST_NO_CHROMA_TMRL (the `PU::hasChromaTmrl(pu) && ...getUseTmrl()` conditions of the chroma search
   false); `bin\encTest10` (m31) adds ECM_TEST_NO_GLM (the `PU::hasGlmFlag` conditions of the chroma search false) and
   ECM_TEST_NO_CHROMA_FUSION (its `PU::hasChromaFusionFlag` conditions false); `bin\encTest11` (m33) adds
   ECM_TEST_GLM_GRADIENT_ONLY (xFindBestGlmIdcSATD searches the gradient patterns only, not GLM with luma);
   `bin\encTest12` adds ECM_TEST_CCCM_PLAIN (the chroma searches' CCCM candidates without GL-CCCM's, the first
   downsampling filter only) and `bin\encTest13` (m34) extends it to LBCCP's search (its GL-CCCM candidate and its other
   downsampling filters; encTest12's streams still had them); `bin\encTest14` (m35) adds ECM_TEST_CCCM_NO_MDF (the
   downsampling-filter switches of ECM_TEST_CCCM_PLAIN alone: GL-CCCM stays; m36: none of them; m37: none, with
   `--CCCM=2`; m38: m37's with LFNST / NSPT; m39: m33's arguments without ECM_TEST_GLM_GRADIENT_ONLY; m42: m39's with
   LFNST / NSPT); `bin\encTest15`
   (m40) adds ECM_TEST_CHROMA_FUSION_LM_ONLY (the chroma fusion search's `uiFusionModeNum` 1: no CFLM) and
   `bin\encTest16` (m40f) ECM_TEST_PREFER_CHROMA_FUSION (a fusion candidate's RD cost divided by 4; m41 / m41f: its
   search with every type, without ECM_TEST_CHROMA_FUSION_LM_ONLY); `bin\encTest17` (m43) ECM_TEST_NO_CCP (the
   chroma searches' PU::hasNonLocalCCP, hasBvgCccmFlag and hasDecoderDerivedCCP conditions false); `bin\encTest18`
   (m46) ECM_TEST_NO_CCP_FUSION (the merge searches' fusion modes, `fusionModeNum` 0); `bin\encTest19` (m47)
   ECM_TEST_CCP_FUSION_CCCM (`fusionModeNum` 1: the modes none and CCCM; without it, m48: all three; m49 with the
   encoder's default of 6 adjustment candidates); `bin\encTest20` (m50) ECM_TEST_NO_DD_MERGE_FUSION (the merge searches'
   decoder-derived merge fusion lists and checks off, so decoder-derived CCP fusion CUs come without decoder-derived
   merge fusion ones); `bin\encTest21` (m52) ECM_TEST_CCP_FUSION_MODE (the SPS's ccpFusionMode 1 in all-intra too:
   EncLib.cpp's `sps.setCcpFusionMode( getenv( "ECM_TEST_CCP_FUSION_MODE" ) ? 1 : getIntraPeriod() == 1 ? 0 : 1 )`;
   m53 without ECM_TEST_CCP_FUSION_CCCM, with LFNST / NSPT).
   The user's order (2026-10-13): all intra first, then inter. Next chroma reordering / DBV, BVG-DIMD, MDIP / MPM
   sorting, IntraTMP / IBC (then BVG-CCCM), EIP / SGPM, PDP, NN
   intra, implicit MTS (not in CTC; its test set, the m24 arguments with `--MTS=0 --MTSImplicit=1`, is encoded:
   `streams_gpu\pending\imts_*`), deblocking on the GPU, then inter pictures.
   Intra merge's GPU time (m19: park720 152, four720 183 ms per picture, from 347 / 392 at first) has room: the
   candidate walk and the reference padding are single-invocation steps on the dependency chain. Each level: new
   test set with exactly that tool on, verify mode identical, ON mode identical (and `ECM_GPU_DUMP` +
   `perf/gpu/dumpdiff.py`: the records of both runs agree), canaries, commit. Speed of derived modes: with
   `ECM_GPU_MODES=1` the GPU time of four 720p pictures sets is 25.3 instead of 14.0 ms per picture (the union of the
   read ranges lengthens the dependency chains, the residuals of derived transforms are computed in the chain); exact
   ranges would need dependencies a node registers once it knows the mode (a dependents list per block, closed when
   the block finishes).
5. In-loop filters in ON mode on the GPU without a CPU round trip (deblocking: see dec-gpu-dbf-wip; ALF/SAO/... are
   on the GPU already via --GpuLoopFilter, but on the CPU's picture).
6. 1080p/4K verification and timing on long streams once sources are agreed with the user (three short 1080p
   streams verify identical since M12); check on other GPU makers (AMD, Intel) before any release.

## 10. Findings worth knowing

- Fixed 2026-10-16 (the Intel PC's corpus, `C:\Code\corpus`: 48 streams of the CTC tools with tool groups off):
  getArBvMergeCandidate's IntraTMP fusion branch `break`s out of its own loop only, so a fusion CU's cheapest vector
  can still enter a list that is full - the list then holds total + 1 (BV_OUT_FUSION; ai_foreman352_q22's TIMD
  vector); with DIMD off the MTSS mode list is empty and NN's transform mode is planar (host rejections "MTSS without
  DIMD", "NN intra prediction without DIMD"); a block sent to the CPU after its CU's EIP / MIP DIMD marker was set
  left the marker (cpuFillCu clears it); GLM with luma's CCP record took numParams, which glmModelToCcpParams never
  sets (GLM_NUM_PARAMS).
- Copy-Item keeps the source's time stamp, so `bin\<name>\DecoderApp.exe`'s time is the link time: a build whose
  link did not run (cut short) leaves the old exe and still prints "built". Check the exe's time after a shader change.

- RTX 3070: plain loads through `coherent`/`volatile` buffer references returned stale L1 data across workgroups;
  use `#pragma use_vulkan_memory_model` + GL_KHR_memory_scope_semantics. Persistent workgroups + an atomic ready
  queue are deadlock-free dependency scheduling.
- RTX 3070, also with the memory model: data passed between the invocations of one workgroup through a global buffer
  and a barrier was read stale by whole warps while another process loaded the GPU (intra_resi, fixed by shared
  memory). Pass data inside a workgroup through shared memory, and stress-test with `gpuload.py` running. Where it
  cannot fit (M18's in-chain residuals of up to 128 x 128 samples), store and load through a `coherent` buffer
  reference with `controlBarrier( gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsBuffer,
  gl_SemanticsAcquireRelease )` between: without `coherent` the memory model's accesses are private and no barrier
  makes them visible (likely intra_resi's old failure); 120 stress decodes of such blocks under load were identical.
  Code a kernel never runs still costs: M18's in-chain paths made intra_rec ~4 % slower on pictures that never take
  them (a per-sample test of a global flag in storeRecon cost as much again; folded into Tu::resiFrom), M19's DIMD
  blending ~15 % (spread over the loop around the prediction and the code inside it; no single part), M21's intra
  merge ~11 % on pictures with DIMD, a second call site of selectTransform and inChainResidual ~7 % (M23: one call
  site each instead), M31's CCLM 30 % on plain pictures. Since M19 intra_rec is built as several pipelines with
  specialization constants (`WITH_DIMD`, `WITH_DERIVED_TRANSFORMS`, `WITH_MERGE`, ..., `WITH_CCLM`, `WITH_CCCM`;
  `gpu::Kernel::create`'s specialization, GpuIntraRec.cpp's `IntraRecVariant` and `LmLevel`) and a picture runs the one without the
  features it does not use. The compiler is not monotonic
  here: the pipeline with DIMD alone was slower than the one with DIMD and the derived transforms in every case
  measured, so that one is not built. Measure each new variant.
- `supported()` / `record()` must reject every tool the kernels lack, including tools that only change a CPU code
  path the GPU replaces (dequantization, transforms, clipping): dependent quantization was not rejected until M6,
  and the reconstruction on decoded such streams wrong. With a new level, look through the CPU path it replaces for
  flags the GPU ignores, and encode a test set with each such flag on.
- The kernels' reference availability is "decoded earlier" (owner maps); the CPU's (`CodingStructure::
  getCURestricted`) also excludes other slices and tiles, with wavefronts the CTUs right of the current one in the
  row above, and with GDR the dirty area. Tiles, wavefronts and GDR were not rejected until 2026-10-11: a stream with
  tiles or WPP decoded wrong with the reconstruction on (test streams `fx_tiles_park720`, `fx_wpp_park720`, now
  skipped). Any new availability rule of the CPU must be mirrored or rejected.
- Vulkan objects are never created or destroyed while a stream records or has a submission in flight
  (`gpu::Context::inFlight()`), and the device is destroyed after decoding (`DecLib::releaseGpu` /
  `releaseGpuDevice`, called by DecApp): with the frame threads' contexts created in the background, NVIDIA's driver
  (616.92) now and then never signalled the fence of a completed submission, and a device left to the end of the
  process now and then hung its termination (2026-10-11, ~1 in 200 decodes of m13c_25 with 8 frame threads;
  DECODER.md). Once a fence signalled before its submission's commands had run (an all-zero picture): every
  submission now ends by writing its number into a host-visible word that `Stream::wait` checks (a late end is waited
  for, a missing one runs the commands of a repeatable stream again, else throws). Keep new GPU code to these rules,
  and test frame-thread changes with many repeated short decodes under load whose outputs are compared, not only once.
- DepQuant::getPredictedSigns (sign prediction) changes the TU's levels in place: the parsed signs of the sign area go
  to the positions sorted by level. Host code that calls it (GpuIntraRec::prepareSignPred) must give the TU its levels
  back, as the CPU calls it again when it verifies or reconstructs a picture the GPU path rejected. With the
  reconstruction on, the TUs keep the undecided signs (in the decoder only parsing, dequantization and the inverse
  transforms read coefficients; the loop filters read none).
- The scheduling allows a dependency on a later block: with joint CbCr whose Cr is coded, the Cr block decides the
  signs and writes the Cb block's residual (Tu::resiFrom). Statistics over the graph must not assume earlier blocks.
  Any new edge between the blocks of one CU can close a cycle with it: a first M17 version let a CU's later blocks wait
  for its first one (which derived the mode), and the Cb block (waiting for its Cr block) and the Cr block (waiting
  for the Cb block) hung the GPU (VK_ERROR_DEVICE_LOST). Edges to the earlier nodes only are cycle-free.
- `gpucheck.py --GpuRecon=2` compares the output, which in verification is the CPU's: it also requires the summary's
  "0 with differences" since 2026-10-12 (before, only the ON modes caught a wrong GPU reconstruction).
- PowerShell 5.1 drops an empty-string argument to a native program: `DecoderApp -b x.bin -o '' --GpuRecon=2` ran as
  `-o --GpuRecon=2`, a CPU decode into a file named `--GpuRecon=2` without a summary line. Decode through the Python
  scripts (`verify_all.py`, `gpucheck.py`), which pass `-o ""` intact.
- NVIDIA's shared shader cache (`%LOCALAPPDATA%\NVIDIA\GLCache`) holds 1 GB. At that limit (2026-10-13, after many
  canary builds) it stored no new pipelines for a while (later the driver pruned it to ~320 MB): every decode of a
  new build compiled all of intra_rec's pipelines, ~96 s per decoder start (six decodes of one canary build took ~90 s
  each). Since then intra_rec's variants are compiled on first use (`intraRecKernel`: ~19 s for a cold start that
  needs one), and the test scripts give the driver a cache of their own (`__GL_SHADER_DISK_CACHE_PATH`,
  `__GL_SHADER_DISK_CACHE_SIZE`, `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP`, which the Windows driver honors).
- Build time of a shader change (2026-10-13): ~15 s, of which glslang `-Os` on intra_rec.comp took 14.1 s (0.3 s
  without the optimizer, `-O` 12.2 s). Without the optimizer the driver's cold compile of a pipeline takes 30.7
  instead of ~19 s and the kernels run up to 18 % slower (2026-10-14: m19 96.2 against 81.8 ms per picture, m24 102.3
  against 93.4). Each shader depends on the includes glslang's depfile names (CMake 3.21+), not on all of them;
  moving the SPIR-V arrays out of GpuIntraRec.cpp gained nothing (its compile runs beside glslang's).
- SPIR-V's ID bound: past its limit (4 194 303) glslang prints "ID overflow. Try running compact-ids." and writes the
  unoptimized module, with exit code 0 - the build succeeds and the kernels get slower. `-Os` inlines every call site
  of every function, which brought intra_rec's bound to 3.72M at M48 (unoptimized: 40k); since 2026-10-14 the build
  uses `-O` (GpuLib's CMakeLists, GLSLANG_OPT): the same kernel speed (plain, m19, m24, m48 within noise) and cold
  pipeline compile (24 s plain, 108 s CCP merge: the same), bound 1.96M. Check the bound of a large change (word 3 of
  the module: glslang `-V --target-env vulkan1.3 -O` into a short path) and keep new code to one call site per large
  function. Since 2026-10-22 (the user's OK to install the Vulkan SDK's spirv-opt) the build with ECM_SPIRV_OPT runs
  glslang without its optimizer and then spirv-opt with glslang -O's pass list (glslang's SpvTools.cpp), a raised
  `--max-id-bound` and `--compact-ids` (GpuLib's spv_build.cmake, which also writes the header and the include
  depfile): the bound was dead IDs of the optimizer's passes - the finished modules hold ~280k / 335k IDs (M70's code:
  3.34M / 4.17M with glslang -O) - and the newer SPIRV-Tools take ~35 s instead of 160 s for intra_rec. The SPIR-V is
  not glslang -O's word for word (another SPIRV-Tools version); its corpus verification and kernel timing are in the
  next milestone's paragraph. Memory: glslang -O took 17-19 GB per intra_rec module (two at once per build), this
  recipe ~2 GB. Without ECM_SPIRV_OPT the old glslang -O recipe still builds, until a module passes 4.19M IDs.
- Buffer references (device addresses) have no robustness: a read past a plane's end loses the device
  (VK_ERROR_DEVICE_LOST, nvlddmkm event 153; all three decodes running at the time failed), and one past a row's end reads
  the next row unnoticed. To find such a read without more device losses (M31's first version: luma TIMD / TMRL blocks
  of 131-mode values from 67 on took the CCLM branch), a temporary check in `recoAt` that returns 0 out of the
  picture and records the block (`atomicMax` into the unused `counter[3]`, downloaded with the dependency count;
  `atomicCompSwap` of the coordinates into a Tu field that verification downloads) named the blocks and the positions
  in one decode (stubbing parts of the new code out had only narrowed it to the prediction step).
- The owner maps have 4x4 luma units; ISP sub-partitions 1 or 2 wide or high share them. The last block owning a
  unit must win everywhere (intra_deps pass 0 writes with atomicMax: plain parallel writes raced), and a block
  reading a shared unit only waits for that last owner, so the sub-partitions of a CU are chained (Tu::ispPrev;
  without the chain every m13 picture differed).
- Class layouts must not depend on `ECM_VULKAN`: the define reaches only the targets that link DecoderLib or GpuLib,
  not EncoderLib, and EncCu derives from DecCu. GPU members under `#if ECM_VULKAN` in DecCu made the encoder of every
  Vulkan build of dec-gpurecon crash at start-up (fixed 2026-10-11). Run the encoder suite on a Vulkan build too.
- The NVIDIA GPU drops to a low clock between pictures (kernels ~3x slower); power settings are off limits.
- `BilateralFilter` derives its LUTs in `create()`, not in the constructor.
- ECM intra reconstruction is raster-serial within a picture once intra/IBC CUs exist (JVET_AJ0172 IBC/IntraTMP
  reference area, BV-guided tools); parallelism comes from independent blocks, components and pictures.
- A dependent per-CU decision chain runs ~7x slower on one GPU workgroup than on one CPU core.
- Edge capacity of the GPU dependency graph: 8 per block at first, grows to 1.25x the largest seen; an overflow
  makes all kernels stand down and the host resubmits the picture (tested with capacity 16).
- With the reconstruction on, the CPU's derive-only pass derives DIMD (and every later sample-based mode) from samples
  it has not reconstructed, so its modes are garbage there and right in verification. A field the host takes from the
  CPU's mode of a block whose mode the GPU derives is therefore wrong only with the reconstruction on, which
  verification cannot see: M19's first ON run differed almost everywhere because the bilateral filter's LMCS flag came
  from the CPU's mode. Check every new level with `ECM_GPU_DUMP=<prefix>` on a verification and an ON decode and
  `perf/gpu/dumpdiff.py <prefix> <POCs>` (the records agree apart from the fields intra_rec derives). The CPU's
  parameter snapshot after DIMD / OBIC blending is the last blended mode's, not the main mode's (record() restates
  them). Garbage modes can also be out of range for the CPU's own later code: OBIC without a mode left -1, which
  getSpatialIpm writes an array with (DecCu keeps a placeholder in ON mode).
- Shared memory of intra_rec is nearly full (by the declarations, about 48 500 of the 49 152 bytes Vulkan gives a
  workgroup on the RTX 3070): OBIC's histogram reuses DIMD's, intra merge's candidates and costs live in s_spCost
  and s_tmp (free in a node).
- Verification reports the first difference by kind of check (scheduling, intra merge costs, DIMD / merge
  parameters, OBIC, block parameters, samples), not in decoding order: a node's output can differ only because an
  earlier block's samples did (M21's first mismatch was an MPM CU's DIMD, caused by an earlier OBIC block). Find the
  earliest differing block before chasing the reported one.
- A rare case no encoder setting reaches (OBIC's class -1 counted by another OBIC CU: ECM's encoder picks OBIC a few
  times per picture where intra merge is allowed) can be tested by forcing the rare value on the CPU and the GPU alike
  with temporary edits on streams rich in the surrounding tool: a counter on the CPU shows the case runs, verification
  must stay identical, and a canary on the GPU side must be reported.
- A double-precision expression of the CPU (xLocationdepBlending's clip ranges) can be reproduced exactly in integers
  when its inputs are small: `dimdScaledWeight` (checked exhaustively against IEEE doubles for inputs below 200).
