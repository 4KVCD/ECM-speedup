# usage: perf\prof.ps1 <build name under C:\Code\bin> [tests...]
# Samples the benchmark tests (perf/bench.py settings) with sprof (perf\sprof\sprof.cpp, built to
# C:\Code\tools\bin\sprof.exe) in parallel, each pinned to one core of $pcores. Build the encoder with symbols
# (-DCMAKE_CXX_FLAGS=-g) for readable reports. Output: C:\Code\runs\prof\<name>\<test>.txt, read with
# perf\sprof\sprof_report.py, incl.py, inl.py, hotaddr.py.
param( [Parameter(Mandatory=$true)][string]$Name, [string[]]$Tests = @('ai_foreman','ai_parkjoy','ra_foreman','ra_parkjoy','ld_foreman') )
$exe = "C:\Code\bin\$Name\EncoderApp.exe"
$cfg = 'C:\Code\opt\cfg'
$out = "C:\Code\runs\prof\$Name"
New-Item -ItemType Directory -Force $out | Out-Null
$def = @{
  ai_foreman = @('encoder_intra_ecm.cfg',        'foreman_352x288.yuv', 352, 288, 30, 1, 32)
  ai_parkjoy = @('encoder_intra_ecm.cfg',        'ParkJoy_256x256.yuv', 256, 256, 50, 2, 27)
  ra_foreman = @('encoder_randomaccess_ecm.cfg', 'foreman_176x144.yuv', 176, 144, 30, 9, 32)
  ra_parkjoy = @('encoder_randomaccess_ecm.cfg', 'ParkJoy_256x256.yuv', 256, 256, 50, 5, 37)
  ld_foreman = @('encoder_lowdelay_ecm.cfg',     'foreman_176x144.yuv', 176, 144, 30, 5, 32)
}
# the performance cores of the Core Ultra 7 265K; set the cores of this machine (perf\cores.py)
$pcores = @(0, 1, 6, 7, 8, 9, 18, 19)
$procs = @()
$i = 0
foreach( $t in $Tests )
{
  $d = $def[$t]
  $args = @("$out\$t.txt", $exe, '-c', "$cfg\$($d[0])", '-i', "C:\Code\seq\$($d[1])", '-wdt', $d[2], '-hgt', $d[3], '-fr', $d[4],
            '--InputBitDepth=8', '-f', $d[5], '-q', $d[6], '-b', "$out\$t.bin", '-o', 'NUL')
  $p = Start-Process -FilePath C:\Code\tools\bin\sprof.exe -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput "$out\$t.log"
  $p.ProcessorAffinity = [IntPtr](1 -shl $pcores[$i % $pcores.Count])   # the encoder child inherits it if set early enough
  $procs += $p
  $i++
}
$procs | Wait-Process
Get-ChildItem $out\*.txt | Select-Object Name, Length
