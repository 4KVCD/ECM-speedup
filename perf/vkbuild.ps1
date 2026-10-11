# usage: perf\vkbuild.ps1 <name> [targets...] [-Lto] [-Configure] [-CMakeArgs @(...)]
# Development build of the GPU decoder: configures C:\Code\build\<name> from C:\Code\vk with the Vulkan decoder on,
# builds the given targets (default DecoderApp and VulkanTestApp) on the efficiency cores and copies the exes to
# C:\Code\bin\<name>.
param( [Parameter(Mandatory=$true)][string]$Name, [string[]]$Targets = @('DecoderApp', 'VulkanTestApp'),
       [switch]$Lto, [switch]$Configure, [string[]]$CMakeArgs = @() )
$ErrorActionPreference = 'Continue'
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]0x3FC3C
. "C:\Code\tools\env.ps1"
$src   = "C:\Code\vk"
$build = "C:\Code\build\$Name"
$out   = "C:\Code\bin\$Name"
if( $Configure -or -not ( Test-Path "$build\build.ninja" ) )
{
  [string[]]$ltoArgs = if( $Lto ) { @( "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW" ) } else { @( "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF" ) }
  [string[]]$vkArgs = @( "-DECM_VULKAN=ON" )
  if( -not $env:VULKAN_SDK )
  {
    $vkArgs += @( "-DVULKAN_HEADERS_DIR=C:/Code/tools/Vulkan-Headers/include", "-DGLSLANG_EXE=C:/Code/tools/glslang/bin/glslang.exe" )
  }
  $cfg = cmake -G Ninja -S $src -B $build -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON -DECM_MARCH=arrowlake -Wno-deprecated @ltoArgs @vkArgs @CMakeArgs 2>&1
  if( $LASTEXITCODE ) { $cfg | Select-Object -Last 30; throw "configure failed" }
}
$log = cmake --build $build --target @Targets -j 12 2>&1
if( $LASTEXITCODE ) { $log | Select-String -Pattern 'error|FAILED|ERROR' | Select-Object -First 40; throw "build failed" }
New-Item -ItemType Directory -Force $out | Out-Null
$bin = Get-ChildItem "$src\bin\ninja" -Recurse -Filter DecoderApp.exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1
foreach( $t in $Targets )
{
  $exe = Join-Path $bin.DirectoryName "$t.exe"
  if( Test-Path $exe ) { Copy-Item $exe $out -Force }
}
Write-Host "built $out"
