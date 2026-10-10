# builds an LTO (or -NoLto) Vulkan decoder build C:\Code\bin\<name> via perf\build.ps1
param( [Parameter(Mandatory=$true)][string]$Name, [switch]$NoLto )
$a = @( $Name ); if( $NoLto ) { $a += '-NoLto' }
& C:\Code\wt-dec\perf\build.ps1 @a -DECM_VULKAN=ON "-DECM_VULKAN_HEADERS=C:/Code/tools/vulkan-headers/Vulkan-Headers-vulkan-sdk-1.4.363.0/include" "-DECM_GLSLANG=C:/Code/tools/glslang/bin/glslang.exe"
