# Checks that encoding intra-period chunks independently and concatenating them with parcat
# gives a bitstream byte-identical to one full-sequence encode.
#   segtest.ps1 -Exe C:\Code\ECM\bin\ref\EncoderApp.exe -Parcat C:\Code\ECM\bin\ref\parcat.exe
param( [string]$Exe = "C:\Code\ECM\bin\ref\EncoderApp.exe", [string]$Parcat = "C:\Code\ECM\bin\ref\parcat.exe",
       [string]$Yuv = "C:\Code\ECM\seq\foreman_128x128.yuv", [int]$W = 128, [int]$H = 128, [int]$Frames = 65,
       [int]$Period = 32, [int]$QP = 32, [string]$Cfg = "C:\Code\ECM\cfg\encoder_randomaccess_ecm.cfg",
       [string]$Out = "C:\Code\ECM\runs\segtest" )
New-Item -ItemType Directory -Force $Out | Out-Null
$common = @( "-c", $Cfg, "-i", $Yuv, "-wdt", $W, "-hgt", $H, "-fr", 30, "--InputBitDepth=8", "-q", $QP, "-o", "NUL" )
$jobs = @()
$jobs += Start-Process -FilePath $Exe -ArgumentList ($common + @("-f", $Frames, "-b", "$Out\full.bin")) -RedirectStandardOutput "$Out\full.log" -NoNewWindow -PassThru
$chunks = @()
for( $s = 0; $s -lt $Frames - 1; $s += $Period )
{
  $n = [Math]::Min( $Period + 1, $Frames - $s )
  $bin = "$Out\chunk_$s.bin"
  $chunks += $bin
  $jobs += Start-Process -FilePath $Exe -ArgumentList ($common + @("-fs", $s, "-f", $n, "-b", $bin, "--FirstValidFrame=0", "--LastValidFrame=$($Frames-1)")) -RedirectStandardOutput "$Out\chunk_$s.log" -NoNewWindow -PassThru
}
$jobs | Wait-Process
& $Parcat @chunks "$Out\cat.bin" | Out-Null
$a = (Get-FileHash "$Out\full.bin" -Algorithm MD5).Hash
$b = (Get-FileHash "$Out\cat.bin" -Algorithm MD5).Hash
"full: $a  $((Get-Item "$Out\full.bin").Length) bytes"
"cat : $b  $((Get-Item "$Out\cat.bin").Length) bytes"
if( $a -eq $b ) { "IDENTICAL" } else { "DIFFERENT" }
