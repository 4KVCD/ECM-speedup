# ECM-speedup

**The same bitstream, 1.5× faster on one core and up to 17× on twenty.**

ECM-speedup is the ECM-20.0 reference software with a faster encoder and decoder. Every change keeps the output
byte-for-byte identical to stock ECM-20.0: after each change, the md5 of the test bitstreams is compared with the stock
encoder's, and the decoder's output pictures with the stock decoder's.

| encoder | bitstream md5 | time | run on |
|---|---|---:|---|
| stock ECM-20.0 | `3072c03937f193a5ef4e467d5345773b` | 2647.9 s | 1 core |
| ECM-speedup | `3072c03937f193a5ef4e467d5345773b` | 1842.8 s | 1 core |
| ECM-speedup | `3072c03937f193a5ef4e467d5345773b` | 451.6 s | 8 processes |

*ra_long: foreman 128×128, 257 frames, random access, QP 32, intra period 32. Wall-clock time on a Core Ultra 7 265K.*

| | speedup | how it was measured |
|---|---:|---|
| Encoder, one core | **1.51×** | CPU time over the 5 test encodes (all-intra, random access, low delay), paired runs |
| Encoder, 20 processes | **17.0×** | all-intra, 33 pictures: 311.2 s on one stock core, 18.4 s with `--ParallelSegments=20` |
| Encoder, 8 processes | **5.9×** | random access, 257 frames in 8 intra periods: 2647.9 s down to 451.6 s |
| Decoder, one core | **1.22×** | random access, 352×288, 289 frames; 1.19–1.30× on three other streams |

## Results

### One core: every test 1.4–1.6× faster

Each test was encoded by both builds at the same time on two performance cores, four times with the cores swapped.
Times are the mean wall-clock time; the speedup is the median ratio of CPU times. All eight bitstreams per test matched
the stock md5.

| test | configuration | stock | ECM-speedup | speedup |
|---|---|---:|---:|---:|
| ai_foreman | all-intra · 352×288 · 1 frame · QP 32 | 49.5 s | 34.7 s | 1.44× |
| ai_parkjoy | all-intra · 256×256 · 2 frames · QP 27 | 74.6 s | 52.0 s | 1.44× |
| ld_foreman | low delay · 176×144 · 5 frames · QP 32 | 72.5 s | 45.2 s | 1.60× |
| ra_foreman | random access · 176×144 · 9 frames · QP 32 | 144.9 s | 95.4 s | 1.52× |
| ra_parkjoy | random access · 256×256 · 5 frames · QP 37 | 92.3 s | 60.0 s | 1.53× |

### Many cores: segment-parallel encoding, up to 17× on a 20-core desktop

With `--ParallelSegments=N` the encoder runs independent pieces of the sequence in N child processes and joins their
output into the bitstream a single encode produces. Random access splits at intra periods, all-intra at pictures.
Low-delay sequences have no such split points and stay on one core.

**ai_long** · foreman 128×128, 33 pictures, all-intra, QP 32

| run | wall-clock time | vs stock on 1 core |
|---|---:|---:|
| stock, 1 core | 311.2 s | |
| ECM-speedup, 1 core | 210.7 s | 1.48× |
| ECM-speedup, 8 processes | 34.9 s | 8.9× |
| ECM-speedup, 20 processes | 18.4 s | 17.0× |

**ra_long** · foreman 128×128, 257 frames, random access, QP 32

| run | wall-clock time | vs stock on 1 core |
|---|---:|---:|
| stock, 1 core | 2647.9 s | |
| ECM-speedup, 1 core | 1842.8 s | 1.44× |
| ECM-speedup, 8 processes | 451.6 s | 5.9× |

ra_long has 8 intra periods, so 8 processes is its maximum. In all-intra mode the most expensive pictures now start
first, ranked by the luma detail of the input pictures; that alone made the 20-process run 1.34× faster (24.6 s to
18.4 s).

### Decoder: the faster prediction and transform code is shared

Many changes sit in the code that the encoder and decoder share: interpolation filters, intra prediction, inverse
transforms. Decode time below is the decoder's own timer, which includes writing the decoded pictures; both decoders
ran at the same time on two performance cores, 6 to 20 times per stream. Every stream decodes to the same YUV md5 with
both decoders.

| stream | content | stock | ECM-speedup | speedup | decoded YUV |
|---|---|---:|---:|---:|---|
| ra_cif | random access · 352×288 · 289 frames · QP 32 | 7.420 s | 6.061 s | 1.22× | identical `00f7736c…` |
| ld_cif | low delay · 352×288 · 33 frames · QP 32 | 0.700 s | 0.589 s | 1.19× | identical `bc83d933…` |
| ra_long | random access · 128×128 · 257 frames · QP 32 | 1.166 s | 0.899 s | 1.30× | identical `9809e976…` |
| ra_97 | random access · 128×128 · 97 frames · QP 32 | 0.608 s | 0.487 s | 1.25× | identical `6fcf4a15…` |

## What changed: every mode is still tested; each test costs less

The encoder tests every mode stock ECM tests and makes the same decision every time. The work went into doing each test
with less computation: vector instructions where code ran one sample at a time, and removing copies, clears, sorts and
recomputation that did not change any result. The numbers in brackets are the measured speedup of the 5-test suite
when each change landed. They were measured on two machines (an i7-10700 and the Core Ultra 7 265K) against different
baselines, so they do not add up exactly to the totals above. [perf/OPTIMIZATIONS.md](perf/OPTIMIZATIONS.md) and the
commit messages have the details.

**Compiler and build** (about 1.15× together)
- Link-time optimization (+5.7 %)
- Profile-guided optimization, trained on clips outside the test set (FourPeople, CrowdRun) (+3.8–6.3 %)
- Code generated for the host CPU, with floating-point contraction turned off so every result stays identical (+3 %)

**Motion compensation and inter search**
- SIMD for the 1–2 sample and 16n+1 wide interpolation blocks that ran scalar, 56 % of filter calls (+1.8 %)
- AVX2 12-tap luma interpolation, 8 and 16 columns per step (+1.3 %)
- Pixel-level affine MC: one fused 12-tap 2-D filter call per 1–2 wide sub-block (+2.4 %)
- GPM: split-mode tables cleared row by row on first use instead of 1.6 MB per CU; branch-free, then AVX2 rank
  counting (+2.5 %)
- Multi-hypothesis search ordered through an index array; buffer views built only when used (+1.3 %)
- Inlined MV clipping; affine AMVP buffers cleared only where written (+1.0 %)

**Intra prediction**
- Long-tap angular interpolation specialized on the tap count, then 8 samples per step in SSE4.1 (all-intra +1 %; +0.3 %)
- Angular reference buffers kept zero between calls; SSE2 dot products for the CCCM statistics (+0.6 %)
- DIMD gradient histograms in AVX2 with a direct angle lookup (+0.9 %)
- CCCM: 7-tap model applied in AVX2, parameters in a fixed-size vector instead of the heap (+0.9 %)
- Matrix-based (PDP) and template-matching linear-filter prediction in AVX2 (+0.6 %)
- Mode-bit estimates replay a CU's mode flags instead of coding them again, 3 million calls per all-intra test
  (all-intra +1.2 %)
- Intra mode lists copied as plain memory (+0.5 %)

**Transform and quantization**
- Dependent-quantization trellis rebuilt in 9 steps: states as a structure of arrays, branch-free decisions, rate
  tables reused while their contexts are unchanged. The quantizer itself runs about 1.7× faster (+5 %)
- Inverse transforms skip all-zero coefficient groups (+0.7 %)
- NSPT/LFNST kernels in AVX2 without gathers (+0.4 %)
- Sign prediction: SSE2 hypothesis costs, coefficient lists ordered without std::stable_sort (all-intra +1 %; +0.3 %)
- Large transform and prediction work buffers moved off the stack (+1 %)

**Entropy-coder state and memory**
- CABAC context snapshots copy only the 32-context blocks that changed, tracked by version stamps (+3 %)
- A small-object allocator for the encoder's millions of short-lived allocations (+2 %)
- Large structures passed by reference (+1.5 %); short lists sorted without heap buffers (+1.2 %)
- Only the used entries of the history tables copied when a coding structure is set up (+0.5 %)

**Loop-filter encoders**
- ALF statistics: SSE accumulators, then a register-tiled AVX2 kernel per class (+1.8 %; +1.7 %)
- ALF clipping search: all trials of a pass solved at once in AVX2 lanes; repeated searches reused (+2.2 %; +0.5 %)
- ALF class merging with compact statistics and batched solves (+0.7 %; +0.8 %)
- CC-ALF and CCSAO statistics reused for unchanged inputs (+1.5 %)

**Parallel encoding** (`--ParallelSegments=N`)
- Random access: each intra period is encoded by its own process. When the parts are joined, the copy of the shared
  intra picture at each joint is dropped and POC LSBs are rebased, so the result equals one sequential encode.
- All intra: each picture is its own process and starts at its own POC with the state the full encode has at that
  picture.
- When there are more chunks than processes, the most expensive start first, ranked by the luma detail of their input
  pictures (correlation 0.95 with encoding time).

## How exactness is checked

- Every change: the five test encodes (two all-intra, two random access, one low delay) must reproduce the stock
  bitstream md5.
- Risky changes get a check build that runs the old and the new code side by side on every call and stops at the first
  difference, for example all 3 million mode-bit estimates of an all-intra test.
- Decoder: 13 streams decode to the same YUV md5 with the stock decoder and the ECM-speedup decoder.
- Configurations outside the test set: all-intra at QP 22, low delay with P slices, random access at 416×240 (not a
  multiple of the 128×128 CTU).
- Long sequences: ra_long and ai_long give the stock md5 both on one core and split into segments.
- Tried and dropped because they were slower or not exact: caches of motion-compensation results (34–80 % of calls
  repeat, but they are the cheap ones), clang ThinLTO (1.5 % faster on all-intra, crashes in inter coding), a thread
  pool inside a picture (exact, but 0 % faster).

## Limits: why one picture still runs on one core

Inside a picture, each 128×128 CTU starts its mode search from the exact CABAC state and the reconstruction that the
previous CTU left behind, so CTUs run one after another. About 96 % of encoding time is that search. Splitting it
across threads changes decisions, unless each thread does only a self-contained piece. The one tried, the intra
template-matching scan, is 2 % of the time and gained nothing.

Between pictures, ECM's temporal CABAC store links every picture of a temporal layer to the previous one, and a
low-delay picture needs the previous picture after its loop filters. Encoding independent random-access pictures at the
same time could still add 1.3–1.8× per intra period (critical-path analysis of ra_long). That is not built yet.

## Build and run

The options and configuration files are those of ECM-20.0; `--ParallelSegments` is the one addition.

```bash
# release build with link-time optimization and code for the host CPU
# (perf/pgo.ps1 shows the profile-guided build used for the measurements)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON -DECM_MARCH=native
cmake --build build -j --target EncoderApp DecoderApp

# random access, one process per intra period (here up to 8 at a time)
EncoderApp -c cfg/encoder_randomaccess_ecm.cfg -i in.yuv -wdt 1920 -hgt 1080 -fr 60 -f 600 -q 32 -b out.bin --ParallelSegments=8

# all-intra, one process per picture
EncoderApp -c cfg/encoder_intra_ecm.cfg -i in.yuv -wdt 1920 -hgt 1080 -fr 60 -f 600 -q 32 -b out.bin --ParallelSegments=20
```

## Measurement setup

Measured in October 2026 on an Intel Core Ultra 7 265K (8 performance and 12 efficiency cores), Windows 11, GCC 16.2
(WinLibs, UCRT), static builds. Stock is ECM-20.0 built with the same compiler; it needed two fixes to build with GCC 16.
Test clips are center crops of xiph.org derf sequences: foreman 352×288, 176×144 and 128×128, ParkJoy 256×256, at
QP 27–37 with the JVET ECM test configurations. Upstream ECM:
[vcgit.hhi.fraunhofer.de/ecm/ECM](https://vcgit.hhi.fraunhofer.de/ecm/ECM).

---

## ECM reference software (original README)

This software package is the reference software for Enhanced Compression Model (ECM). The reference software includes both encoder and decoder functionality.

Reference software is useful in aiding users of a video coding standard to establish and test conformance and interoperability, and to educate users and demonstrate the capabilities of the test model.

The software has been jointly developed by the ITU-T Video Coding Experts Group (VCEG, Question 6 of ITU-T Study Group 16) and the ISO/IEC Moving Picture Experts Group (MPEG, Working Group 11 of Subcommittee 29 of ISO/IEC Joint Technical Committee 1).

A software manual, which contains usage instructions, can be found in the "doc" subdirectory of this software package.

### Build instructions

The CMake tool is used to create platform-specific build files. 

Although CMake may be able to generate 32-bit binaries, **it is generally suggested to build 64-bit binaries**. 32-bit binaries are not able to access more than 2GB of RAM, which will not be sufficient for coding larger image formats. Building in 32-bit environments is not tested and will not be supported.

In order to use the NNLF interface, please activate the option using the cmake option EXTENSION_NNLF.
If using SADL, please initialize the submodule:
```bash
git submodule init
git submodule update
```

#### Build instructions for plain CMake (suggested)

**Note:** A working CMake installation is required for building the software.

CMake generates configuration files for the compiler environment/development environment on each platform. 
The following is a list of examples for Windows (MS Visual Studio), macOS (Xcode) and Linux (make).

Open a command prompt on your system and change into the root directory of this project.

Create a build directory in the root directory:
```bash
mkdir build 
```

Use one of the following CMake commands, based on your platform. Feel free to change the commands to satisfy
your needs.

**Windows Visual Studio 2015/17/19 64 Bit:**

Use the proper generator string for generating Visual Studio files, e.g. for VS 2015:

```bash
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
```

Then open the generated solution file in MS Visual Studio.

For VS 2017 use "Visual Studio 15 2017 Win64", for VS 2019 use "Visual Studio 16 2019".

Visual Studio 2019 also allows you to open the CMake directory directly. Choose "File->Open->CMake" for this option.

**macOS Xcode:**

For generating an Xcode workspace type:
```bash
cd build
cmake .. -G "Xcode"
```
Then open the generated work space in Xcode.

For generating Makefiles with optional non-default compilers, use the following commands:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9
```
In this example the brew installed GCC 9 is used for a release build.

**Linux**

For generating Linux Release Makefile:
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```
For generating Linux Debug Makefile:
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

Then type
```bash
make -j
```

For more details, refer to the CMake documentation: https://cmake.org/cmake/help/latest/

#### Build instructions for make

**Note:** The build instructions in this section require the make tool and Python to be installed, which are
part of usual Linux and macOS environments. See below for installation instruction for Python and GnuWin32 
on Windows.

Open a command prompt on your system and change into the root directory of this project.

To use the default system compiler simply call:
```bash
make all
```


**MSYS2 and MinGW (Windows)**

**Note:** Build files for MSYS MinGW were added on request. The build platform is not regularily tested and can't be supported. 

Open an MSYS MinGW 64-Bit terminal and change into the root directory of this project.

Call:
```bash
make all toolset=gcc
```

The following tools need to be installed for MSYS2 and MinGW:

Download CMake: http://www.cmake.org/ and install it.

Python and GnuWin32 are not mandatory, but they simplify the build process for the user.

python:    https://www.python.org/downloads/release/python-371/

gnuwin32:  https://sourceforge.net/projects/getgnuwin32/files/getgnuwin32/0.6.30/GetGnuWin32-0.6.3.exe/download

To use MinGW, install MSYS2: http://repo.msys2.org/distrib/msys2-x86_64-latest.exe

Installation instructions: https://www.msys2.org/

Install the needed toolchains:
```bash
pacman -S --needed base-devel mingw-w64-i686-toolchain mingw-w64-x86_64-toolchain git subversion mingw-w64-i686-cmake mingw-w64-x86_64-cmake
```

