# Profile-guided build: instrumented build -> training encodes -> optimized rebuild in the same tree.
#   perf\pgo.ps1 <name> [extra cmake -D args...]
# Training content (FourPeople and CrowdRun 256x256 crops) is deliberately different from the benchmark sequences:
#   C:\Code\seq\FourPeople_256x256.yuv: y4m2yuv.py FourPeople_1280x720_60.y4m <out> 10 256 256 512 232
#   C:\Code\seq\CrowdRun_256x256.yuv:   y4m2yuv.py crowd_run_1080p50.y4m <out> 10 256 256 832 412
#   (sources: media.xiph.org/video/derf/y4m; only the first frames are needed)
param( [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Name, [Parameter(ValueFromRemainingArguments=$true)][string[]]$CMakeArgs )   # Name: a plain name, the build directory is deleted
$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\..\..\tools\env.ps1"
$env:CCACHE_DISABLE = "1"   # profile flags + ccache: don't risk stale objects
$src   = Resolve-Path "$PSScriptRoot\.."
$build = "C:\Code\build\$Name"
$out   = "C:\Code\bin\$Name"
$train = "C:\Code\runs\pgo-train-$Name"
[string[]]$lto = @( "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW" )

function Build( [string]$flags )
{
  cmake -G Ninja -S $src -B $build -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON -Wno-deprecated @lto @CMakeArgs "-DCMAKE_CXX_FLAGS=$flags" "-DCMAKE_EXE_LINKER_FLAGS=$flags" 2>&1 | Out-Null
  if( $LASTEXITCODE ) { throw "configure failed" }
  cmake --build $build --target EncoderApp DecoderApp -j 16 2>&1 | Select-String -Pattern 'error|FAILED' | Select-Object -First 30
  if( $LASTEXITCODE ) { throw "build failed" }
}

# 1) instrumented build
Remove-Item -Recurse -Force $build -ErrorAction SilentlyContinue -Confirm:$false
Build "-fprofile-generate -fprofile-update=single"
$exe = "$src\bin\ninja\gcc-mingw-16.2\x86_64\release\EncoderApp.exe"

# 2) training (sequential: concurrent .gcda merging is not safe on MinGW)
New-Item -ItemType Directory -Force $train | Out-Null
# (CrowdRun added: trained on FourPeople alone, the high-texture ai_parkjoy test ran about 1 % slower than without PGO)
$runs = @(
  @( "encoder_intra_ecm.cfg",        "FourPeople_256x256.yuv", 60, 1, 37 ),
  @( "encoder_randomaccess_ecm.cfg", "FourPeople_256x256.yuv", 60, 5, 32 ),
  @( "encoder_lowdelay_ecm.cfg",     "FourPeople_256x256.yuv", 60, 3, 27 ),
  @( "encoder_intra_ecm.cfg",        "CrowdRun_256x256.yuv",   50, 1, 27 ),
  @( "encoder_randomaccess_ecm.cfg", "CrowdRun_256x256.yuv",   50, 5, 37 ),
  @( "encoder_lowdelay_ecm.cfg",     "CrowdRun_256x256.yuv",   50, 3, 32 ) )
foreach( $r in $runs )
{
  & $exe -c "$src\cfg\$($r[0])" -i "C:\Code\seq\$($r[1])" -wdt 256 -hgt 256 -fr $r[2] --InputBitDepth=8 -f $r[3] -q $r[4] -b "$train\t.bin" -o NUL | Select-String 'Total Time'
}

# 3) optimized build from the profile (same build tree, so .gcda paths match)
Build "-fprofile-use -fprofile-partial-training -Wno-missing-profile -Wno-error=coverage-mismatch"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item $exe, "$src\bin\ninja\gcc-mingw-16.2\x86_64\release\DecoderApp.exe" $out -Force
Write-Host "built $out"
