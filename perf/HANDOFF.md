# Handoff: continuing the ECM-20.0 speed-up on another computer

State on 2026-10-10. Read this first, then `perf/OPTIMIZATIONS.md` (every step with its measured gain) and the commit
messages (each has problem, cause, change, verification and limits).

## 1. Where things stand

- **Branches** on github.com/4KVCD/ECM-speedup: `perf` (default) and `merge` are both at b8e5f2ba and hold all kept
  work. Start new work from `perf` on a branch of your own.
  - `dev` has 3 commits from the i7-10700 that are not merged; see section 9.
  - The stock baseline is 6d442b2a: tag ECM-20.0 (3391a5f0) plus two GCC 16 build fixes.
- **Rule: every bitstream is byte-identical to stock ECM-20.0.** The md5 of every test bitstream is checked after every
  change. Nothing may change a decision of the encoder. The work is only about computing the same thing faster.
- **Results**, Core Ultra 7 265K, Windows 11, GCC 16.2:
  - Encoder, one core: **1.51x** (CPU time, 5-test suite, paired; per test 1.44-1.60x).
  - Encoder with `--ParallelSegments`: all-intra 33 pictures **17.0x** with 20 processes; random access 257 frames
    **5.9x** with 8 processes (wall-clock vs stock on one core).
  - Decoder: **1.19-1.30x**. It was never profiled or tuned on its own; the gains come from the shared CommonLib code.
- **Profile is flat**: about 96 % of encoding time is `EncCu::compressCtu` (the mode search). In the latest profile,
  25 % of self time is in 12 functions, 50 % in 47, 90 % in 331. Single changes now give 0.2-1 %.

## 2. Ground rules (the user's rules, as applied to this project)

- **Bit-exact**: compare the bitstream md5 of all 5 tests with `runs/results/ref.json` (produced by the stock build;
  see 3.5). Any difference means the change is wrong. Understand why; don't tune it away.
- **Check builds** for anything that is not obviously equivalent. Put the old code behind `#ifdef XXX_CHECK` next to
  the new code, compare the results on every call, `abort()` with a message on the first difference, and print the
  number of compared calls at exit (a static object's destructor). Build with `-DCMAKE_CXX_FLAGS=-DXXX_CHECK` and run
  the suite. A counter of 0 means the path was never exercised. Examples: `MODEBITS_CHECK`, `GEOSORT_CHECK`,
  `CCCMAPPLY_CHECK`, `PDPOPT_CHECK`, `DIMDANG_CHECK`, `SIGNSORT_CHECK`, `AFFINE_NARROW_CHECK` in the code.
- **Measure paired.** Run both builds of a test at the same time on a pair of identical cores, swap the cores every
  repetition, and use the median per-repetition CPU-time ratio (`bench.py ab`, see 4.3). Unpaired runs drift by ±2 %.
- **Commits**: one commit per kept step.
  - Subject: what changed, plus "(bit-exact)".
  - Body: problem, cause, change, effect, verification (counters, md5, hardware), limits, and the benchmark table.
  - End with the `Co-Authored-By:` line.
  - No `git stash`, no rebase, no force-push. Amend only commits that are not on GitHub yet; otherwise add a commit
    that names the one it fixes.
  - Work in a worktree, never in the main clone.
- **Pushing**: only with the user's explicit yes, every time.
- Keep the user's global CLAUDE.md on the new computer; this project follows it.

## 3. Setting up the new computer

### 3.1 Tools

| tool | version used | notes |
|---|---|---|
| GCC | WinLibs `gcc-16.2.0-mingw-w64ucrt-14.0.0-r2` (x86_64, UCRT, posix-seh) | unpack to `C:\Code\tools\mingw64` |
| CMake | 4.4.4 | `C:\Program Files\CMake` |
| Ninja | 1.13 (winget `Ninja-build.Ninja`) | |
| Python | 3.14 | bench and report scripts; no packages needed |
| Git, GitHub CLI | any | `gh auth login`; see 4.7 for pushing |

The build scripts dot-source `C:\Code\tools\env.ps1`. Copy `perf/env.example.ps1` there and adjust the paths.

### 3.2 Directory layout (the scripts assume it)

```
C:\Code\ECM          main clone (do not commit here); bench.py reads cfg files from C:\Code\ECM\cfg
C:\Code\opt          worktree for the work branch
C:\Code\stock        worktree at 6d442b2a (stock + GCC 16 fixes)
C:\Code\tools        env.ps1, mingw64\, bin\sprof.exe
C:\Code\seq          test clips (3.3)
C:\Code\bin\<name>   one folder per build (EncoderApp.exe, DecoderApp.exe)
C:\Code\build\<name> build trees
C:\Code\runs         results (runs\results\*.json), logs, profiles
```

```bash
git clone https://github.com/4KVCD/ECM-speedup.git C:/Code/ECM
cd C:/Code/ECM
git worktree add -b work C:/Code/opt origin/perf
git worktree add --detach C:/Code/stock 6d442b2a
```

### 3.3 Test clips

Get the sources from https://media.xiph.org/video/derf/y4m/: `foreman_cif.y4m`, `park_joy_1080p50.y4m`,
`FourPeople_1280x720_60.y4m`, `crowd_run_1080p50.y4m`. Only the first frames are used. Then:

```bash
python perf/y4m2yuv.py foreman_cif.y4m           C:/Code/seq/foreman_352x288.yuv     10
python perf/y4m2yuv.py foreman_cif.y4m           C:/Code/seq/foreman_176x144.yuv     10  176 144  88  72
python perf/y4m2yuv.py foreman_cif.y4m           C:/Code/seq/foreman_128x128_257.yuv 257 128 128 112  80
python perf/y4m2yuv.py park_joy_1080p50.y4m      C:/Code/seq/ParkJoy_256x256.yuv     10  256 256 832 412
python perf/y4m2yuv.py FourPeople_1280x720_60.y4m C:/Code/seq/FourPeople_256x256.yuv 10  256 256 512 232
python perf/y4m2yuv.py crowd_run_1080p50.y4m     C:/Code/seq/CrowdRun_256x256.yuv    10  256 256 832 412
```

md5 of the files on the 265K. All but CrowdRun were re-made from the sources above on 2026-10-10 and matched; the
CrowdRun source is no longer on the 265K:

| clip | md5 |
|---|---|
| foreman_352x288.yuv | e9196f8bb1a25503d000c5d8e92fdb4e |
| foreman_176x144.yuv | 0b4abe3ae59ab26f137d370d42e1c71d |
| foreman_128x128_257.yuv | dbf9b50e3491b269ae8512d68278fad6 |
| ParkJoy_256x256.yuv | 9da12de43694a2998ed2c746aa798a0a |
| FourPeople_256x256.yuv | 30ae8f86a7c50ef855a3f09db243f827 |
| CrowdRun_256x256.yuv | 868e66359ca196f59fd1a9737cdb0428 |

FourPeople and CrowdRun are used only to train PGO, so the benchmark never sees them.

### 3.4 Builds

```powershell
# stock reference: plain -O3 Release, static, no LTO, no -march. Every speedup in the logs is measured against this
# build. The stock tree has no perf/ folder, so use the pinned build script with a source path.
perf\build_pinned.ps1 stock C:\Code\stock
# current code, LTO, code for this CPU (use the -march value of the new CPU, e.g. native, znver4, znver5)
perf\build_pinned.ps1 base C:\Code\opt -Lto -DECM_MARCH=native
# profile-guided build: instrumented build, 6 training encodes (FourPeople/CrowdRun), optimized rebuild (~20 min)
perf\pgo.ps1 pgo1 -DECM_MARCH=native
```

- `perf/build.ps1 <name>` builds the tree it belongs to.
- `perf/build_pinned.ps1 <name> <tree>` builds any tree, optionally pinned to some cores (`-Affinity 0x...`) so that
  builds can run while pinned benchmarks use other cores.
- CMakeLists already adds what bit-exactness and MinGW need: `-ffp-contract=off` (RD costs are doubles; FMA
  contraction changes them), `-Wa,-muse-unaligned-vector-move` (GCC cannot realign the stack for AVX spills on Win64),
  `-fno-aggressive-loop-optimizations` (stock ALF code reads past an array bound that GCC 16 would exploit).
- `ECM_MARCH` empty means the stock SSE4.1 build. The SADL (intra NN) units always stay on the stock ISA.

### 3.5 Reference md5 (stock encoder, these clips)

```bash
python perf/bench.py run ref --exe C:/Code/bin/stock/EncoderApp.exe --jobs 5
python perf/bench.py run ref --exe C:/Code/bin/stock/EncoderApp.exe --tests ai_long,ra_long --jobs 2
```

With the same clips and compiler, these md5 should come out:

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

### 3.6 Which cores to measure on

`python perf/cores.py` prints the logical processors by efficiency class.
- Hybrid Intel: use the performance cores for `--cpus` (the 265K: 0,1,6,7,8,9,18,19) and build on the efficiency
  cores.
- AMD or other SMT CPUs: use one logical processor per physical core and leave the siblings idle while measuring.
- Never measure while builds, other benchmarks or interactive work run on the same cores. Chrome alone pushed
  ra_foreman to ±5 %.

## 4. Workflow

### 4.1 Profile

Build with symbols (`-DCMAKE_CXX_FLAGS=-g`, LTO keeps them) and build the sampler once:

```bash
g++ -O2 -std=c++17 perf/sprof/sprof.cpp -o C:/Code/tools/bin/sprof.exe -ldbghelp -lpsapi -lwinmm
```

`perf/prof.ps1 <build>` samples the 5 tests (each pinned to a core; edit `$pcores`) into `C:\Code\runs\prof\<build>`.
Reports:
- `sprof_report.py samples.txt EncoderApp.exe --top 60`: self and inclusive time.
  - `--callers F` / `--callees F` / `--callsites F`: who calls F, what F calls, from which lines.
  - `--lines F`: hot source lines of F.
- `sprof/incl.py exe file1,file2 func1,func2`: inclusive share of functions per test.
- `sprof/inl.py samples exe --grep FILE:LINE`: self time by innermost inlined line.
- `sprof/hotaddr.py exe FILE:LINE samples...`: disassembly around the hot addresses of a line.

The tools need `objdump`, `nm` and `addr2line` from the MinGW `bin` on PATH.

### 4.2 Change, check, measure

1. Find structural waste in the profile (section 6), not just a hot function.
2. Implement it in the worktree. For long edits, write a patch script (Python, reading and writing files) instead of
   hand edits, so a step can be re-applied and dropped cleanly. Never write C or Python with backslash escapes through
   bash heredocs or `python -c`: the shell eats them.
3. If the change is not trivially equivalent, run a check build over the suite: no abort, md5 equal, counter > 0.
4. Run LTO builds A/B (4.3). Keep the step if the total gain is clear of noise and no test loses.
5. Commit (format in section 2). Keep the dropped diff outside the repo for reference.

### 4.3 Benchmarks (`perf/bench.py`)

```bash
python perf/bench.py run NAME --exe C:/Code/bin/NAME/EncoderApp.exe --jobs 5 --cpus 0,1,6,7,8   # md5 + times vs ref
python perf/bench.py ab OLD NEW --reps 4 --cpus 0,1,6,7,8,9    # paired A/B, prints per-test median CPU ratios
python perf/bench.py run NAME --exe ... --tests ai_long "--extra=--ParallelSegments=20" --jobs 1   # no --cpus for parallel runs
```

- Noise on the 265K: A/A ±0.8 % per test, ±0.3 % total. Code layout alone moves single tests by several percent. On
  the i7-10700, a change that only touched GPM made all-intra 1.03-1.04x "faster" (all-intra never runs GPM) and
  ld_foreman 0.95x; judge a change by the total and by the tests that run the changed code.
- Gains measured on LTO builds shrink to about 0.6x of their size under PGO. Re-run PGO only at milestones, then run
  `bench.py ab stock pgoN --reps 4` for the end-to-end number.
- `--cpus` pins each encode to one core. A segment-parallel run would then put all its child processes on that one
  core, so never pass `--cpus` with `--ParallelSegments`.
- The CPU column counts only the top process, so segment-parallel runs show about 0 s. The `dev` branch fixes this
  with a job object.

### 4.4 Decoder

`python perf/decbench.py stock\DecoderApp.exe new\DecoderApp.exe outdir a.bin,b.bin --reps 6 --cpus 0,1`
- It decodes with both decoders at the same time and compares the YUV md5. The decoded YUV must be identical.
- It reports CPU time and the decoder's own timer, which excludes process start-up (about 0.2 s) but includes writing
  the YUV.
- Decodes of small bench streams take 0.3-1.4 s, so use the long or CIF streams (section 5) or many repetitions.

### 4.5 Segment-parallel

- Check: `bench.py run ... --tests ai_long,ra_long "--extra=--ParallelSegments=N"`. The md5 must equal the stock
  serial ref.
- Reordering only happens with more chunks than workers. To test it on random access, use `--SegmentLength=32` with
  fewer workers than intra periods; for example, 97 frames on 2 workers gave the serial md5.
- `perf/segtest.ps1` is the older chunk-and-parcat check.

### 4.6 Frame-level bound

`python perf/ra_critical_path.py encoder.log 32` takes the per-picture times ([ET]) of a serial random-access log. It
prints, per intra period, the work and the critical path through references and the same-layer chains of the temporal
CABAC store, i.e. the most that encoding pictures concurrently could gain.

### 4.7 Pushing from a non-interactive shell

The Git Credential Manager dialog cannot open there. With `gh` logged in:

```bash
git -c credential.helper= -c "credential.helper=!gh auth git-credential" push origin work:perf
```

## 5. Reference streams and numbers (265K)

| run | result |
|---|---|
| 5-test suite, stock vs 49fa654f+PGO (4 paired reps) | 1.512x CPU; ai 1.436/1.440, ld 1.604, ra 1.523/1.533 |
| ai_long serial: stock / PGO | 311.2 s / 210.7 s |
| ai_long, `--ParallelSegments` 8 / 20 | 34.9 s / 18.4 s |
| ra_long serial: stock / PGO | 2647.9 s / 1842.8 s |
| ra_long, 8 workers | 451.6 s |
| extra encodes identical to stock | AI QP 22 foreman CIF 1 frame `6d142bad…`; LD-P foreman 176x144 3 frames `445f299e…`; RA ParkJoy 416x240 crop (752,420) 4 frames QP 37 `0d1880f7…` |
| decoder, stock vs PGO | ra_cif 289 frames 1.22x, ld_cif 33 frames 1.19x, ra_long 1.30x |

The CIF decode streams were encoded by the optimized encoder only: foreman CIF 289 frames RA QP 32 with 19 workers,
33 frames LD QP 32.

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
- Inside a picture, CTUs are serial: the mode search of CTU n+1 needs the exact CABAC states and the reconstruction
  that CTU n leaves.
- Split subtrees share caches (`BestEncInfoCache`, `CacheBlkInfoCtrl`, IBC BV caches) and the picture's recon
  buffer, so a parallel split search cannot be exact.
- A thread pool for the IntraTMP scan was exact but 0 % faster (the scan is 2 % of the time). Busy-waiting workers
  also lower the main core's turbo.
- Between pictures, the temporal CABAC store (`Contexts.h`, `CtxStateStore`, key = temporal layer and slice QP,
  5 entries) and the AMVP-SbTMVP area statistics (`g_picAmvpSbTmvpEnabledArea`, keyed by layer) chain the pictures
  of each temporal layer. Low delay needs each previous picture after ALF, so it is fully serial.

## 7. Dropped experiments (do not retry as they were)

Dropped on the 265K, all bit-exact unless noted:
- Luma mode-bits memo extended to TMP/MIP/TIMD/EIP/SGPM/TMRL modes: 0.997x.
- Bypass-sign counting in codePredictedSigns: 1.001x.
- CCCM training-sample gather in AVX2 with compaction: 1.000x.
- Sign-prediction cost loop in AVX2: 1.001x.
- BDOF parameter sums via lane permutes: 0.999x / 0.997x.
- TMP joint SAD two rows per step: 0.993x.
- Bilateral filter with small blocks packed per AVX2 step: 1.001x.
- AVX2 DepQuant decision; AVX2 inverse transform on top of the zero skip; GEO std::sort of packed keys; inlined
  buffer accessors; a neighbour-availability shortcut: no gain.
- Thread pool for the IntraTMP sparse scan: exact, 0 %.
- ParallelSegments workers pinned to performance/efficiency cores: slower.

Dropped on the i7-10700: listed in `perf/OPTIMIZATIONS.md` (sparse context restore, MC caches, cached CU lookups,
wavefront check). The GPM carry-swap sort (1.00x) is logged on `dev`.

Probes that found little: intra CU re-evaluation duplicates (2-4 %); content-identical angular predictions (AI 1 %,
RA 18 % of samples, but hashing costs as much); chroma CCCM work repeated in later LFNST passes (about 0.2 %);
getCURestricted (38 % repeat one of the last 2 queries, but a memo costs what the walk saves).

## 8. What to try next

1. **Encode independent random-access pictures concurrently** (largest remaining multithreading gain, exact by
   construction if every shared state is handled).
   - Bound from `ra_critical_path.py` on ra_long: 1.27-1.80x per intra period. ra_long with 8 workers could go from
     451.6 s to about 250 s.
   - Needs per-picture encoder object stacks; the leftover `ENABLE_SPLIT_PARALLELISM` code in EncLib shows how
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
3. **Merge `dev`** if wanted (section 9).
4. **Single-thread leftovers**, each probably 0.2-0.5 %: CU re-initialization in `initSubStructure`, OBMC work reuse
   (`subBlockOBMC` is 6.9 % inclusive in ra_foreman), and items in section 7's probes.
5. On a new CPU: re-measure the build options (`-march`, PGO) and any kernel that depends on vector width.

## 9. Known issues

- `dev` (i7-10700) holds three commits not in `perf`:
  - 05b53351: the allocator with an extra canary build mode `-DFASTALLOC_CANARY=1` (guard bytes, poisoning).
  - df395834: `bench.py` counts the CPU time of all processes of a run through a job object.
  - f07bd2f1: log notes on the dropped GPM carry-swap sort, layout noise and CPU accounting.

  Merging conflicts in `FastAlloc.cpp`: take dev's file, which is perf's plus the canary mode, and confirm the
  default build gives the same md5.
- Stock ECM-20.0 does not compile with `ENABLE_HIGH_BITDEPTH=ON` or with `JVET_AG0061_INTER_LFNST_NSPT` set to 0.
  bd1be75c keeps the new code ready for both.
- Windows Python cannot open paths over 260 characters (for example the Claude scratchpad); keep work files under
  `C:\Code`.
- The main clone `C:\Code\ECM` on the 265K is not updated past 2ffd9ac3; pull before using it.
