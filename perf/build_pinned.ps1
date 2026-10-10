# usage: perf\build_pinned.ps1 <name> <source tree> [-Lto] [-Affinity 0x3FC3C] [extra cmake -D args...]
# Configures C:\Code\build\<name> from any source tree (also trees without perf\, e.g. the stock worktree), builds
# EncoderApp + DecoderApp and copies the exes to C:\Code\bin\<name>.
# -Affinity restricts cmake, ninja and the compilers to the given logical processors, so that builds can run while
# pinned benchmarks use other cores. On the Core Ultra 7 265K, 0x3FC3C is the efficiency cores (logical CPUs 2-5,
# 10-17); perf\cores.py lists the cores of the machine.
param( [Parameter(Mandatory=$true)][string]$Name, [Parameter(Mandatory=$true)][string]$Src, [switch]$Lto,
       [string]$Affinity = "", [Parameter(ValueFromRemainingArguments=$true)][string[]]$CMakeArgs )
$ErrorActionPreference = 'Continue'
if( $Affinity )
{
  # children (cmake, ninja, gcc) inherit the affinity of this process
  [System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr][Convert]::ToInt64( $Affinity, 16 )
}
. "C:\Code\tools\env.ps1"
$build = "C:\Code\build\$Name"
$out   = "C:\Code\bin\$Name"
[string[]]$ltoArgs = if( $Lto ) { @( "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW" ) } else { @( "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF" ) }
$cfg = cmake -G Ninja -S $Src -B $build -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON -Wno-deprecated @ltoArgs @CMakeArgs 2>&1
if( $LASTEXITCODE ) { $cfg | Select-Object -Last 20; throw "configure failed" }
$log = cmake --build $build --target EncoderApp DecoderApp -j 12 2>&1
if( $LASTEXITCODE ) { $log | Select-String -Pattern 'error|FAILED' | Select-Object -First 30; throw "build failed" }
New-Item -ItemType Directory -Force $out | Out-Null
# ECM's CMake writes every build to the same in-source bin\ folder
$bin = Get-ChildItem "$Src\bin\ninja" -Recurse -Filter EncoderApp.exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Copy-Item $bin.FullName, ( Join-Path $bin.DirectoryName "DecoderApp.exe" ) $out -Force
Write-Host "built $out"
