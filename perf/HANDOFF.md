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
