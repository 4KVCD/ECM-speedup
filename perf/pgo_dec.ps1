# Profile-guided build of the decoder: instrumented build -> training decodes -> optimized rebuild in the same tree.
#   perf\pgo_dec.ps1 <name> [extra cmake -D args...]
# The training streams are 9 of the 18 decbench streams (C:\Code\ECM\streams); the other 9 show whether a gain is specific
# to the training content. Only the DecoderApp of the result is meant to be used: its EncoderApp is optimized with a
# decoder profile.
param( [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Name, [Parameter(ValueFromRemainingArguments=$true)][string[]]$CMakeArgs )   # Name: a plain name, the build directory is deleted
$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\..\..\tools\env.ps1"
$env:CCACHE_DISABLE = "1"   # profile flags + ccache: don't risk stale objects
$src   = Resolve-Path "$PSScriptRoot\.."
$build = "C:\Code\ECM\build\$Name"
$out   = "C:\Code\ECM\bin\$Name"
$lto   = @( "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW" )

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
$exe = "$src\bin\ninja\gcc-mingw-16.2\x86_64\release\DecoderApp.exe"

# 2) training (sequential: concurrent .gcda merging is not safe on MinGW)
foreach( $s in "ai_parkjoy", "ai_md5", "aiodd", "ai_foreman", "ra_parkjoy", "ra_foreman", "raodd", "ld_foreman", "ld_md5" )
{
  & $exe -b "C:\Code\ECM\streams\$s.bin" -o NUL | Select-String 'Total Time'
}

# 3) optimized build from the profile (same build tree, so .gcda paths match)
Build "-fprofile-use -fprofile-partial-training -Wno-missing-profile -Wno-error=coverage-mismatch"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item "$src\bin\ninja\gcc-mingw-16.2\x86_64\release\EncoderApp.exe", $exe $out -Force
Write-Host "built $out"
