# spv_build.cmake: one shader into a header with its SPIR-V words (const uint32_t <VAR>[]), with spirv-opt as the
# optimizer. glslang compiles without its own optimizer (and writes the include depfile), spirv-opt runs glslang -O's
# pass list with a raised ID limit and compacts the IDs at the end: glslang's optimizer fails past SPIR-V's 4.19M ID
# bound although the finished module needs far fewer (intra_rec: 2.4M allocated while optimizing, 0.28M after
# compacting), and the newer SPIRV-Tools of spirv-opt runs ~5x faster. Run by GpuLib's CMakeLists.txt with
#   -DGLSLANG= -DSPIRV_OPT= -DSHADER= -DOUT= -DVAR= -DINCLUDE= -DDEFINES=<comma-separated -DX>
foreach( V GLSLANG SPIRV_OPT SHADER OUT VAR INCLUDE )
  if( NOT DEFINED ${V} )
    message( FATAL_ERROR "spv_build.cmake: ${V} not set" )
  endif()
endforeach()
string( REPLACE "," ";" DEFINES "${DEFINES}" )
# INLINE 0: no --inline-entry-points-exhaustive (the driver gets the functions; an experiment on its compile memory)
set( INLINE_PASS --inline-entry-points-exhaustive )
if( DEFINED INLINE AND NOT INLINE )
  set( INLINE_PASS )
endif()
set( RAW "${OUT}.raw.spv" )
set( OPT "${OUT}.spv" )
execute_process( COMMAND "${GLSLANG}" -V --target-env vulkan1.3 ${DEFINES} "-I${INCLUDE}" --depfile "${OUT}.raw.d"
                         -o "${RAW}" "${SHADER}"
                 RESULT_VARIABLE RC OUTPUT_VARIABLE LOG ERROR_VARIABLE LOG )
if( RC )
  message( FATAL_ERROR "glslang ${SHADER}:\n${LOG}" )
endif()
# glslang -O's passes (glslang's SpvTools.cpp, SpvToolsTransform without optimizeSize), then --compact-ids
execute_process( COMMAND "${SPIRV_OPT}" --target-env=vulkan1.3 --max-id-bound=67108863
                         --wrap-opkill --eliminate-dead-branches --merge-return ${INLINE_PASS}
                         --eliminate-dead-functions --scalar-replacement --convert-local-access-chains
                         --eliminate-local-single-block --eliminate-local-single-store --simplify-instructions
                         --eliminate-dead-code-aggressive --vector-dce --eliminate-dead-inserts
                         --eliminate-dead-code-aggressive --eliminate-dead-branches --merge-blocks
                         --eliminate-local-multi-store --if-conversion --simplify-instructions
                         --eliminate-dead-code-aggressive --vector-dce --eliminate-dead-inserts --interpolate-fixup
                         --eliminate-dead-code-aggressive --cfg-cleanup --compact-ids
                         "${RAW}" -o "${OPT}"
                 RESULT_VARIABLE RC OUTPUT_VARIABLE LOG ERROR_VARIABLE LOG )
if( RC OR NOT LOG STREQUAL "" )
  message( FATAL_ERROR "spirv-opt ${SHADER}:\n${LOG}" )
endif()
# the words as glslang's --vn writes them (little-endian bytes -> 32-bit words, 8 per line)
file( READ "${OPT}" HEX HEX )
string( REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1," WORDS "${HEX}" )
string( REGEX REPLACE "((0x........,)(0x........,)(0x........,)(0x........,)(0x........,)(0x........,)(0x........,)(0x........,))"
        "\\1\n" WORDS "${WORDS}" )
file( WRITE "${OUT}.tmp" "#pragma once\nconst uint32_t ${VAR}[] = {\n${WORDS}\n};\n" )
file( RENAME "${OUT}.tmp" "${OUT}" )
# the depfile names the header (glslang's names the raw module)
file( READ "${OUT}.raw.d" DEPS )
string( FIND "${DEPS}" ": " COLON )
string( SUBSTRING "${DEPS}" ${COLON} -1 DEPS )
file( WRITE "${OUT}.d" "${OUT}${DEPS}" )
