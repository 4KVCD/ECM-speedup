# enc_m12_edges.ps1: the small MRL test streams m12e_* (crops of the test sequences: many blocks at the picture's
# edges) and m12s_* (synthetic content on which the encoder picks far reference lines), with the M12 set's arguments
$sp  = $PSScriptRoot
$seq = 'C:\Code\ECM\seq'
$m12 = @( '--TemporalSubsampleRatio=1', '--IntraToolControlMode=1', '--IntraPredBf=1', '--DualITree=1', '--LMCSEnable=1',
          '--DepQuant=2', '--QCS=1', '--JointCbCr=1', '--MTS=1', '--MTSExt=1', '--TransformSkip=1', '--ChromaTS=1', '--BDPCM=1',
          '--IntraLFNSTISlice=1', '--LFNSTExt=1', '--NSPT=1', '--MRL=1' )
# sources: crop.py <in> <w> <h> <x> <y> <cw> <ch> <frames> <out>, gen_mrl_edge.py <out> <w> <h> <frames> <seed> [p16]
python "$sp\crop.py" "$seq\foreman_352x288.yuv" 352 288 144 64 64 64 8 "$seq\foreCrop_64x64.yuv"
python "$sp\crop.py" "$seq\foreman_352x288.yuv" 352 288 0 200 64 64 16 "$seq\foreCrop2_64x64.yuv"
python "$sp\crop.py" "$seq\ParkJoy_1280x720.yuv" 1280 720 600 300 128 64 8 "$seq\parkCrop_128x64.yuv"
python "$sp\crop.py" "$seq\FourPeople_1280x720.yuv" 1280 720 400 200 64 64 8 "$seq\fourCrop_64x64.yuv"
python "$sp\crop.py" "$seq\screen_352x288.yuv" 352 288 0 0 64 64 4 "$seq\screenCrop0_64x64.yuv"
python "$sp\crop.py" "$seq\screen_352x288.yuv" 352 288 160 96 64 64 4 "$seq\screenCrop1_64x64.yuv"
python "$sp\gen_mrl_edge.py" "$seq\mrlEdge_128x64.yuv" 128 64 8 1
python "$sp\gen_mrl_edge.py" "$seq\mrlEdge2_128x64.yuv" 128 64 8 2 p16
$jobs = @(
  @( 'm12e_fore27',   'foreCrop_64x64.yuv',     64,  64,  8, '27' ),
  @( 'm12e_fore37',   'foreCrop_64x64.yuv',     64,  64,  8, '37' ),
  @( 'm12e_fore2_32', 'foreCrop2_64x64.yuv',    64,  64, 16, '32' ),
  @( 'm12e_park32',   'parkCrop_128x64.yuv',    128, 64,  8, '32' ),
  @( 'm12e_four27',   'fourCrop_64x64.yuv',     64,  64,  8, '27' ),
  @( 'm12e_scr32',    'screen_352x288.yuv',     352, 288, 4, '32' ),
  @( 'm12e_scr0_27',  'screenCrop0_64x64.yuv',  64,  64,  4, '27' ),
  @( 'm12e_scr1_37',  'screenCrop1_64x64.yuv',  64,  64,  4, '37' ),
  @( 'm12s_edge27',   'mrlEdge_128x64.yuv',     128, 64,  8, '27' ),
  @( 'm12s_edge37',   'mrlEdge_128x64.yuv',     128, 64,  8, '37' ),
  @( 'm12s_edgeB22',  'mrlEdge2_128x64.yuv',    128, 64,  8, '22' ),
  @( 'm12s_edgeB32',  'mrlEdge2_128x64.yuv',    128, 64,  8, '32' )
)
foreach( $j in $jobs )
{
  $r = & "$sp\enc_m1.ps1" $j[0] "$seq\$($j[1])" $j[2] $j[3] $j[4] @m12 -q $j[5] | Select-Object -First 1
  "{0,-14} {1}" -f $j[0], $r
}
