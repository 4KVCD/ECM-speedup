# removes the in-source libraries and decoder of the worktree, which every build directory shares, so that the next
# build relinks them from its own objects
Get-ChildItem C:\Code\wt-gr\lib\ninja\gcc-mingw-16.2\x86_64\release -Filter *.a | Remove-Item -Force
$exe = 'C:\Code\wt-gr\bin\ninja\gcc-mingw-16.2\x86_64\release\DecoderApp.exe'
if( Test-Path $exe ) { Remove-Item -Force $exe }
