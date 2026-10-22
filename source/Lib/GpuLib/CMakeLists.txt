# library: Vulkan compute backend of the decoder
set( LIB_NAME GpuLib )

if( NOT ECM_VULKAN_HEADERS OR NOT EXISTS "${ECM_VULKAN_HEADERS}/vulkan/vulkan.h" )
  message( FATAL_ERROR "ECM_VULKAN: set ECM_VULKAN_HEADERS to the Vulkan-Headers include directory (with vulkan/vulkan.h)" )
endif()
if( NOT ECM_GLSLANG OR NOT EXISTS "${ECM_GLSLANG}" )
  message( FATAL_ERROR "ECM_VULKAN: set ECM_GLSLANG to the glslang executable" )
endif()
if( ECM_SPIRV_OPT AND NOT EXISTS "${ECM_SPIRV_OPT}" )
  message( FATAL_ERROR "ECM_VULKAN: ECM_SPIRV_OPT (${ECM_SPIRV_OPT}) does not exist" )
endif()

file( GLOB SRC_FILES "*.cpp" )
file( GLOB INC_FILES "*.h" )
file( GLOB SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp" )
file( GLOB SHADER_INCLUDES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl" )

# every compute shader becomes a header with its SPIR-V words (const uint32_t spv_<name>[]); glslang's depfile names
# the includes each one reads, so a change to one include recompiles only the shaders that read it (CMake 3.21 has
# DEPFILE for every generator; before, each shader depends on all includes). glslang's optimizer matters (without it
# the kernels run up to 18 % slower); -O gives the speed and cold pipeline compile of -Os with half the SPIR-V IDs
# (intra_rec 1.96M against 3.72M of the 4.19M a module may have: -Os inlines every call site). With ECM_SPIRV_OPT (the
# Vulkan SDK's spirv-opt) glslang compiles without its optimizer and spirv-opt runs the same passes with a raised ID
# limit and compacts the IDs (spv_build.cmake): no ID budget any more (intra_rec 0.28M IDs), ~5x faster
option( ECM_SPIRV_INLINE "with ECM_SPIRV_OPT: inline every call (glslang -O's pass list)" ON )
set( GLSLANG_OPT -O CACHE STRING "glslang's optimization option for the shaders (-O; empty: none)" )
set( SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/spv" )
file( MAKE_DIRECTORY "${SPV_DIR}" )
set( SPV_HEADERS )
# spv_header( <name> <shader> [macros...] ): the header spv/<name>.spv.h of <shader> compiled with the macros defined
# (const uint32_t spv_<name>[]), appended to SPV_HEADERS
function( spv_header NAME SHADER )
  set( OUT "${SPV_DIR}/${NAME}.spv.h" )
  set( DEFINES )
  foreach( MACRO ${ARGN} )
    list( APPEND DEFINES "-D${MACRO}" )
  endforeach()
  string( REPLACE ";" "," DEFINES_ARG "${DEFINES}" )   # (a ; would split the argument)
  if( ECM_SPIRV_OPT AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.21 )
    add_custom_command( OUTPUT "${OUT}"
                        COMMAND "${CMAKE_COMMAND}" "-DGLSLANG=${ECM_GLSLANG}" "-DSPIRV_OPT=${ECM_SPIRV_OPT}"
                                "-DSHADER=${SHADER}" "-DOUT=${OUT}" "-DVAR=spv_${NAME}"
                                "-DINCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/shaders" "-DDEFINES=${DEFINES_ARG}" "-DINLINE=${ECM_SPIRV_INLINE}"
                                -P "${CMAKE_CURRENT_SOURCE_DIR}/spv_build.cmake"
                        DEPENDS "${SHADER}" "${CMAKE_CURRENT_SOURCE_DIR}/spv_build.cmake"
                        DEPFILE "${OUT}.d"
                        COMMENT "glslang + spirv-opt ${NAME}" VERBATIM )
  elseif( CMAKE_VERSION VERSION_GREATER_EQUAL 3.21 )
    add_custom_command( OUTPUT "${OUT}"
                        COMMAND "${ECM_GLSLANG}" -V --target-env vulkan1.3 ${GLSLANG_OPT} ${DEFINES}
                                -I${CMAKE_CURRENT_SOURCE_DIR}/shaders --vn spv_${NAME} --depfile "${OUT}.d" -o "${OUT}"
                                "${SHADER}"
                        DEPENDS "${SHADER}"
                        DEPFILE "${OUT}.d"
                        COMMENT "glslang ${NAME}" VERBATIM )
  else()
    add_custom_command( OUTPUT "${OUT}"
                        COMMAND "${ECM_GLSLANG}" -V --target-env vulkan1.3 ${GLSLANG_OPT} ${DEFINES}
                                -I${CMAKE_CURRENT_SOURCE_DIR}/shaders --vn spv_${NAME} -o "${OUT}" "${SHADER}"
                        DEPENDS "${SHADER}" ${SHADER_INCLUDES}
                        COMMENT "glslang ${NAME}" VERBATIM )
  endif()
  set( SPV_HEADERS ${SPV_HEADERS} "${OUT}" PARENT_SCOPE )
endfunction()
foreach( SHADER ${SHADER_FILES} )
  get_filename_component( NAME ${SHADER} NAME_WE )
  spv_header( ${NAME} "${SHADER}" )
endforeach()
# intra_rec's second module: the pipelines of the pictures with CCP merge or decoder-derived CCP fusion blocks with
# LFNST / NSPT, whose second merge prediction adds ~1M IDs that the other pipelines' module does without
spv_header( intra_rec_lfnst "${CMAKE_CURRENT_SOURCE_DIR}/shaders/intra_rec.comp" CCP_LFNST_MODULE )

add_library( ${LIB_NAME} STATIC ${SRC_FILES} ${INC_FILES} ${SPV_HEADERS} ${SHADER_FILES} ${SHADER_INCLUDES} )
target_include_directories( ${LIB_NAME} PUBLIC . "${ECM_VULKAN_HEADERS}" PRIVATE "${SPV_DIR}" )
target_compile_definitions( ${LIB_NAME} PUBLIC ECM_VULKAN=1 VK_NO_PROTOTYPES )
target_link_libraries( ${LIB_NAME} CommonLib )
if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
  target_link_libraries( ${LIB_NAME} ${CMAKE_DL_LIBS} )
endif()

source_group( "Shaders" FILES ${SHADER_FILES} ${SHADER_INCLUDES} )
set_target_properties( ${LIB_NAME} PROPERTIES FOLDER lib )
