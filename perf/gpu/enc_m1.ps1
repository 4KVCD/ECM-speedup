# M1 test stream for the GPU reconstruction: intra only, ECM tools off (plain angular/planar/DC, DCT-II, scalar
# quantization, no in-loop filters); usage: enc_m1.ps1 <name> <yuv> <w> <h> <frames> [extra encoder args...]
param( [string]$Name, [string]$Yuv, [int]$W, [int]$H, [int]$Frames, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Extra )
$enc = 'C:\Code\ECM\bin\qaCpu\EncoderApp.exe'
$out = "C:\Code\ECM\streams_gpu"
New-Item -ItemType Directory -Force $out | Out-Null
$off = @(
  '--MTS=0', '--MTSExt=0', '--IntraLFNSTISlice=0', '--IntraLFNSTPBSlice=0', '--InterLFNST=0', '--NSPT=0', '--LFNSTExt=0', '--ISP=0', '--MIP=0', '--MRL=0', '--SBT=0',
  '--DIMD=0', '--BvgDIMD=0', '--TIMD=0', '--TIMDMerge=0', '--IntraTMP=0', '--IBC=0', '--IntraNN=0', '--PDP=0',
  '--LMChroma=0', '--CCCM=0', '--BvgCccm=0', '--CCPMerge=0', '--DDCCPFusion=0', '--EIP=0', '--SGPM=0', '--TMRL=0',
  '--MPMSorting=0', '--MDIP=0', '--ChromaReordering=0', '--IntraDBV=0', '--EnableTMnoninterTools=0',
  '--DepQuant=0', '--NumSignPred=0', '--NumSignPredOverrideByQP=0 0', '--JointCbCr=0', '--TransformSkip=0', '--ChromaTS=0', '--BDPCM=0',
  '--LMCSEnable=0', '--DualITree=0', '--SAO=0', '--ALF=0', '--CCALF=0', '--ALFCCCM=0', '--BIF=0', '--ChromaBIF=0',
  '--CCSAO=0', '--LoopFilterDisable=1', '--InloopOffsetRefineFlag=0', '--IntraCMD=0', '--QCS=0'
)
& $enc -c C:\Code\ECM\wt-dec\cfg\encoder_intra_ecm.cfg -i $Yuv -wdt $W -hgt $H -fr 30 -f $Frames -q 32 -b "$out\$Name.bin" -o NUL @off @Extra > "$out\$Name.enclog" 2> "$out\$Name.encerr"
"exit $LASTEXITCODE"
Get-Content "$out\$Name.encerr" | Select-Object -First 5
Select-String -Path "$out\$Name.enclog" -Pattern 'TOOL CFG|POC' | Select-Object -First 8 | ForEach-Object { $_.Line }
