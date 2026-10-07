# minimum required cmake version
cmake_minimum_required( VERSION 3.5 FATAL_ERROR )

# project name
if( EXTENSION_360_VIDEO )
  project( NextSoftware360 )
else()
  project( NextSoftware )
endif()

# use ccache
find_program( CCACHE_FOUND ccache )
if( CCACHE_FOUND )
  message( STATUS "ccache found. using it." )
  set_property( GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache )
  set_property( GLOBAL PROPERTY RULE_LAUNCH_LINK ccache )
endif()

# set default CMAKE_BUILD_TYPE to Release if not set
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
  if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    set( USE_ADDRESS_SANITIZER OFF CACHE BOOL "Compiles with -sanitize=address and links to libasan" )
  endif()
endif()
set( EXTENSION_NNLF OFF CACHE BOOL "If EXTENSION_NNLF is on, NNLF interfaces are added. Please use git submodule init/update to get SADL library." )

set( EXTENSION_360_VIDEO OFF CACHE BOOL "If EXTENSION_360_VIDEO is on, 360Lib will be added" )
set( EXTENSION_HDRTOOLS OFF CACHE BOOL "If EXTENSION_HDRTOOLS is on, HDRLib will be added" )

if (DEFINED ENABLE_TRACING)
  set( ENABLE_TRACING OFF CACHE BOOL "ENABLE_TRACING will be set to this value" )
endif()
if (DEFINED ENABLE_HIGH_BITDEPTH)
  set( ENABLE_HIGH_BITDEPTH OFF CACHE BOOL "ENABLE_HIGH_BITDEPTH will be set to this value" )
endif()

if( CMAKE_COMPILER_IS_GNUCC )
  set( BUILD_STATIC OFF CACHE BOOL "Build static executables" )
endif()

# set c++17
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

# compile everything position independent (even static libraries)
set( CMAKE_POSITION_INDEPENDENT_CODE TRUE )

# set verbose compile options
#set( CMAKE_VERBOSE_MAKEFILE ON )

# use folders in IDEs for projects (e.g. lib sample app test)
set_property( GLOBAL PROPERTY USE_FOLDERS ON )

# Include a utility module providing functions, macros, and settings
include( ${CMAKE_SOURCE_DIR}/cmake/CMakeBuild/cmake/modules/BBuildEnv.cmake )

# Enable warnings for some generators and toolsets.
# bb_enable_warnings( gcc warnings-as-errors -Wno-sign-compare )
# bb_enable_warnings( gcc -Wno-unused-variable )
# bb_enable_warnings( gcc-4.8 warnings-as-errors -Wno-unused-variable )
# for gcc 8.2:
bb_enable_warnings( gcc warnings-as-errors -Wno-sign-compare)

if( XCODE )
  bb_enable_warnings( clang warnings-as-errors
                            -Wno-deprecated-declarations
                            -Wno-unknown-attributes
                            -Wno-pessimizing-move
                            -Wno-absolute-value
                            -Wno-unused-const-variable
                            -Wno-unused-command-line-argument )
else()
  bb_enable_warnings( clang warnings-as-errors
                            -Wno-unknown-attributes
                            -Wno-pessimizing-move
                            -Wno-absolute-value
                            -Wno-unused-const-variable )
endif()
#bb_enable_warnings( clang warnings-as-errors )

# enable warnings
bb_enable_warnings( msvc warnings-as-errors "/wd4996" )

# enable sse4.1 build for all source files for gcc and clang
if( (UNIX OR MINGW) AND NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64") )
  add_compile_options( "-msse4.1" )
  # perf: target ISA for all (non-dispatched) code, e.g. x86-64-v3 (AVX2/BMI2/FMA), "native", or "" for stock.
  # Measured: x86-64-v3 is bit-exact but gives no speedup (hot code is hand-written SIMD), so stock is the default.
  set( ECM_MARCH "" CACHE STRING "-march value for all source files (empty: stock SSE4.1 build)" )
  if( ECM_MARCH )
    add_compile_options( "-march=${ECM_MARCH}" )
  endif()
  if( MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    # Win64 GCC cannot realign the stack for 32-byte AVX spills (GCC bug 54412): emit unaligned vector moves
    add_compile_options( "-Wa,-muse-unaligned-vector-move" )
  endif()
  # never contract a*b+c into FMA: RD costs are doubles and must stay bit-exact with the reference build
  add_compile_options( "-ffp-contract=off" )
endif()

if (EXTENSION_NNLF)
    add_compile_definitions(JVET_AL0228_NNLF_INTERFACES=1)
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/sadl)
        message(FATAL_ERROR "SADL library needed for build. Execute command:\ngit submodule init; git submodule update\nin order to retrieve, or unzip the archive in the sadl directory.")
    endif()
    set(PATH_TO_DIRECTORY_ROOT_SADL "${CMAKE_SOURCE_DIR}/sadl")
endif()
# enable parallel build for Visual Studio
if( MSVC )
  add_compile_options( "/MP" )
  add_compile_options( "/EHsc" )
endif()

# set address sanitizer compiler arguments
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
  if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
    if( USE_ADDRESS_SANITIZER )
      # add compile options
      add_compile_options( "-fsanitize=address" )
      add_link_options( "-fsanitize=address" )
    endif()
  endif()

  if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0 )
    add_compile_options( "-fabi-version=6" )
  endif()

  # EncAdaptiveLoopFilter.cpp indexes m_chromaScaleIdxFinal[4][1] with altIdx up to 7; keep newer GCC
  # from exploiting that UB (and from failing the build with -Werror) so behaviour matches the reference
  add_compile_options( "-fno-aggressive-loop-optimizations" )
  # newer GCC emits false-positive maybe-uninitialized warnings at higher -march levels; keep them non-fatal
  add_compile_options( "-Wno-error=maybe-uninitialized" )
endif()


# modify .lldbinit for lldb custom data formatters
if( XCODE )
  set( LLDB_INSTALL_ROOT "$ENV{HOME}/.lldb.d" )
  set( LLDBINIT_FILE "$ENV{HOME}/.lldbinit" )
  set( ENABLE_LLDBINIT_UPDATE ON )
  # add custom target to install LLDB files.
  add_subdirectory( "lldb" )
endif()

# add needed subdirectories
add_subdirectory( "source/Lib/CommonLib" )
add_subdirectory( "source/Lib/CommonAnalyserLib" )
if( EXTENSION_360_VIDEO )
  add_subdirectory( "source/Lib/Lib360" )
  add_subdirectory( "source/Lib/AppEncHelper360" )
endif()
if ( EXTENSION_HDRTOOLS )
  add_subdirectory( "source/Lib/HDRLib")
endif()
add_subdirectory( "source/Lib/DecoderAnalyserLib" )
add_subdirectory( "source/Lib/DecoderLib" )
add_subdirectory( "source/Lib/EncoderLib" )
add_subdirectory( "source/Lib/Utilities" )

add_subdirectory( "source/App/DecoderAnalyserApp" )
add_subdirectory( "source/App/DecoderApp" )
add_subdirectory( "source/App/EncoderApp" )
add_subdirectory( "source/App/SEIRemovalApp" )
add_subdirectory( "source/App/Parcat" )
add_subdirectory( "source/App/StreamMergeApp" )
add_subdirectory( "source/App/BitstreamExtractorApp" )
add_subdirectory( "source/App/CabacTraining" )
if( EXTENSION_360_VIDEO )
  add_subdirectory( "source/App/utils/360ConvertApp" )
endif()

