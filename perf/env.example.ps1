# Copy to C:\Code\tools\env.ps1 and adjust the paths. Dot-sourced by perf\build.ps1, perf\build_pinned.ps1 and
# perf\pgo.ps1: puts the build toolchain first on PATH.
#   GCC 16.2 (WinLibs gcc-16.2.0-mingw-w64ucrt-14.0.0-r2, UCRT posix-seh) in C:\Code\tools\mingw64, CMake, Ninja
#   (winget Ninja-build.Ninja), Python 3.14.
$ninjaDir = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter 'Ninja-build.Ninja_*' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$env:Path = ( @(
  "C:\Code\tools\mingw64\bin",
  "C:\Program Files\CMake\bin",
  $ninjaDir,
  "$env:LOCALAPPDATA\Programs\Python\Python314",
  "C:\Program Files\Git\cmd"
) | Where-Object { $_ } ) -join ';' | ForEach-Object { "$_;$env:Path" }
$env:CC  = "gcc"
$env:CXX = "g++"
