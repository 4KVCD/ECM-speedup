# enc_long.ps1 <name> <yuv> <w> <h> <segments> <framesPerSegment> [extra encoder args...]: encodes an all-intra test
# stream as parallel segments (each starts with an IDR, -fs skips the frames before it) and concatenates them
param( [string]$Name, [string]$Yuv, [int]$W, [int]$H, [int]$Segs, [int]$Per, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Extra )
$sp = $PSScriptRoot
$jobs = @()
for( $i = 0; $i -lt $Segs; $i++ )
{
  $seg = "${Name}_seg$i"
  $jobs += Start-Job -ScriptBlock { param( $sp, $seg, $Yuv, $W, $H, $Per, $fs, $Extra )
    & "$sp\enc_m1.ps1" $seg $Yuv $W $H $Per -fs $fs --TemporalSubsampleRatio=1 @Extra | Select-Object -First 1
  } -ArgumentList $sp, $seg, $Yuv, $W, $H, $Per, ( $i * $Per ), $Extra
}
$jobs | Wait-Job | Receive-Job
$out = [IO.File]::Create( "C:\Code\streams_gpu\$Name.bin" )
for( $i = 0; $i -lt $Segs; $i++ )
{
  $b = [IO.File]::ReadAllBytes( "C:\Code\streams_gpu\${Name}_seg$i.bin" )
  $out.Write( $b, 0, $b.Length )
}
$out.Close()
"wrote C:\Code\streams_gpu\$Name.bin"
