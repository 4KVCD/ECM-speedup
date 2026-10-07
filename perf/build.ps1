# usage: perf\build.ps1 <name> [-NoLto] [extra cmake -D args...]
# Configures C:\Code\build\<name> (link-time optimization unless -NoLto), builds EncoderApp+DecoderApp and copies the
# exes to C:\Code\bin\<name>. (ECM's CMake writes all builds to the same in-source bin/ folder, hence the copy.)
param( [Parameter(Mandatory=$true)][string]$Name, [switch]$NoLto, [Parameter(ValueFromRemainingArguments=$true)][string[]]$CMakeArgs )
$ErrorActionPreference = 'Continue'   # native-command stderr must not be fatal in PS 5.1; check exit codes instead
. "$PSScriptRoot\..\..\tools\env.ps1"
$src   = Resolve-Path "$PSScriptRoot\.."
$build = "C:\Code\build\$Name"
$out   = "C:\Code\bin\$Name"
$lto   = if( $NoLto ) { @( "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF" ) } else { @( "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW" ) }
cmake -G Ninja -S $src -B $build -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON -Wno-deprecated @lto @CMakeArgs 2>&1 | Out-Null
if( $LASTEXITCODE ) { throw "configure failed" }
cmake --build $build --target EncoderApp DecoderApp -j 16 2>&1 | Select-String -Pattern 'error|FAILED' | Select-Object -First 30
if( $LASTEXITCODE ) { throw "build failed" }
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item "$src\bin\ninja\gcc-mingw-16.2\x86_64\release\EncoderApp.exe", "$src\bin\ninja\gcc-mingw-16.2\x86_64\release\DecoderApp.exe" $out -Force
Write-Host "built $out"
