# QA of the commit to be pushed: CPU-only and Vulkan LTO builds, a libstdc++-assertions build; every stream in every
# decoder mode against the stock decoder, the GPU verification, the encoder suite
$ErrorActionPreference = 'Continue'
$sp  = $PSScriptRoot
$git = 'C:\Code\ECM\tools\git\cmd\git.exe'
"HEAD: " + ( & $git -C C:\Code\ECM\wt-dec log -1 --format='%h %s' ) + "  status lines: " + ( & $git -C C:\Code\ECM\wt-dec status --short | Measure-Object ).Count
$vk = @( "-DECM_VULKAN=ON", "-DECM_VULKAN_HEADERS=C:/Code/ECM/tools/vulkan-headers/Vulkan-Headers-vulkan-sdk-1.4.363.0/include", "-DECM_GLSLANG=C:/Code/ECM/tools/glslang/bin/glslang.exe" )
& "$sp\cleanlibs.ps1"
"== build qaCpu (LTO, CPU only)";            & C:\Code\ECM\wt-dec\perf\build.ps1 qaCpu 2>&1 | Select-Object -Last 2
& "$sp\cleanlibs.ps1"
"== build qaGpu (LTO, Vulkan)";              & C:\Code\ECM\wt-dec\perf\build.ps1 qaGpu @vk 2>&1 | Select-Object -Last 2
& "$sp\cleanlibs.ps1"
"== build qaChk (no LTO, Vulkan, _GLIBCXX_ASSERTIONS)"
& C:\Code\ECM\wt-dec\perf\build.ps1 qaChk -NoLto @vk "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_ASSERTIONS" 2>&1 | Select-Object -Last 2
Select-String -Path C:\Code\ECM\build\qaChk\build.ninja -Pattern '_GLIBCXX_ASSERTIONS' | Select-Object -First 1 | ForEach-Object { "assertions flag in build.ninja: yes" }
cd C:\Code\ECM\wt-dec\perf
$runs = @(
  @( 'qaCpu',      'qaCpu', '' ),
  @( 'qaGpuOn',    'qaGpu', '--GpuLoopFilter=1' ),
  @( 'qaGpuV',     'qaGpu', '--GpuLoopFilter=2' ),
  @( 'qaGpuFt8',   'qaGpu', '--GpuLoopFilter=1 --FrameThreads=8' ),
  @( 'qaGpuFt16',  'qaGpu', '--GpuLoopFilter=1 --FrameThreads=16' ),
  @( 'qaGpuSer',   'qaGpu', '--GpuLoopFilter=1 --FrameThreads=8 --FrameThreadsSerial=1' ),
  @( 'qaCpuFt8',   'qaCpu', '--FrameThreads=8' ),
  @( 'qaGpuSeg8',  'qaGpu', '--GpuLoopFilter=1 --ParallelSegments=8' ),
  @( 'qaChkOn',    'qaChk', '--GpuLoopFilter=1' ),
  @( 'qaChkV',     'qaChk', '--GpuLoopFilter=2' ),
  @( 'qaChkFt8',   'qaChk', '--GpuLoopFilter=1 --FrameThreads=8' ),
  @( 'qaChkCpu',   'qaChk', '' )
)
foreach( $r in $runs )
{
  $jobs = if( $r[2] -match 'FrameThreads|ParallelSegments' ) { 2 } else { 4 }
  "== decode $($r[0]): $($r[1]) $($r[2])"
  python decbench.py run $r[0] --exe "C:/Code/ECM/bin/$($r[1])/DecoderApp.exe" --extra="$($r[2])" --jobs $jobs | Where-Object { $_ -notmatch ' ok$|  ok ' } | Select-Object -Last 25
}
foreach( $v in 'qaGpuV', 'qaChkV' )
{
  $t = 0; $d = 0
  Get-ChildItem "C:\Code\ECM\runs\dec_$v\*.log" | ForEach-Object { $l = Select-String -Path $_.FullName -Pattern 'verification: (\d+) pictures.*?(\d+) with differences' | Select-Object -First 1; if( $l ) { $t += [int]$l.Matches[0].Groups[1].Value; $d += [int]$l.Matches[0].Groups[2].Value } else { "  no verification line: $($_.Name)" } }
  "$v verification: $t pictures, $d with differences"
}
"assertion / error lines in the decoder logs:"
Select-String -Path C:\Code\ECM\runs\dec_qa*\*.log -Pattern 'Assertion|assert|ERROR|error|exception|terminate' | Select-Object -First 20 | ForEach-Object { "  $($_.Path): $($_.Line)" }
"== encoder suite (qaCpu)"
python bench.py run qaCpuEnc --exe C:/Code/ECM/bin/qaCpu/EncoderApp.exe 2>&1 | Select-Object -Last 8
"QA done"
