# crlf.ps1 <worktree>: converts every modified or untracked text file of the worktree to CRLF line endings (no BOM)
param( [string]$Root )
$git = 'C:\Code\tools\git\cmd\git.exe'
$enc = New-Object Text.UTF8Encoding( $false )
$files = & $git -C $Root status --porcelain --untracked-files=all | ForEach-Object { $_.Substring( 3 ) } |
  Where-Object { $_ -match '\.(cpp|h|comp|glsl|txt|py|md|ps1|cmake)$' -or $_ -match 'CMakeLists\.txt$' }
foreach( $f in $files )
{
  $p = Join-Path $Root $f
  if( -not ( Test-Path $p -PathType Leaf ) ) { continue }
  $t = [IO.File]::ReadAllText( $p )
  $n = ( $t -replace "`r`n", "`n" ) -replace "`n", "`r`n"
  if( $n -ne $t ) { [IO.File]::WriteAllText( $p, $n, $enc ); "CRLF: $f" }
}
