# enc_set.ps1 <prefix> [extra encoder args...]: the GPU-reconstruction test set for one tool level (enc_m1.ps1 base)
param( [string]$Prefix, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Extra )
$sp = $PSScriptRoot
$jobs = @(
  @( "${Prefix}_fore32",   'C:\Code\seq\foreman_352x288.yuv',     352,  288, 4, '32' ),
  @( "${Prefix}_fore22",   'C:\Code\seq\foreman_352x288.yuv',     352,  288, 2, '22' ),
  @( "${Prefix}_fore37",   'C:\Code\seq\foreman_352x288.yuv',     352,  288, 2, '37' ),
  @( "${Prefix}_park720",  'C:\Code\seq\ParkJoy_1280x720.yuv',    1280, 720, 2, '32' ),
  @( "${Prefix}_four720",  'C:\Code\seq\FourPeople_1280x720.yuv', 1280, 720, 2, '27' )
)
foreach( $j in $jobs )
{
  $t = Get-Date
  $r = & "$sp\enc_m1.ps1" $j[0] $j[1] $j[2] $j[3] $j[4] --TemporalSubsampleRatio=1 -q $j[5] @Extra | Select-Object -First 1
  "{0,-16} {1}  {2:N0} s" -f $j[0], $r, ( (Get-Date) - $t ).TotalSeconds
}
