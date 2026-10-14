# library: Vulkan compute backend of the decoder
set( LIB_NAME GpuLib )

if( NOT ECM_VULKAN_HEADERS OR NOT EXISTS "${ECM_VULKAN_HEADERS}/vulkan/vulkan.h" )
  message( FATAL_ERROR "ECM_VULKAN: set ECM_VULKAN_HEADERS to the Vulkan-Headers include directory (with vulkan/vulkan.h)" )
endif()
if( NOT ECM_GLSLANG OR NOT EXISTS "${ECM_GLSLANG}" )
  message( FATAL_ERROR "ECM_VULKAN: set ECM_GLSLANG to the glslang executable" )
endif()

file( GLOB SRC_FILES "*.cpp" )
file( GLOB INC_FILES "*.h" )
file( GLOB SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp" )
file( GLOB SHADER_INCLUDES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl" )

# every compute shader becomes a header with its SPIR-V words (const uint32_t spv_<name>[]); glslang's depfile names
# the includes each one reads, so a change to one include recompiles only the shaders that read it (CMake 3.21 has
# DEPFILE for every generator; before, each shader depends on all includes). glslang's optimizer matters (without it
# the kernels run up to 18 % slower); -O gives the speed and cold pipeline compile of -Os with half the SPIR-V IDs
# (intra_rec 1.96M against 3.72M of the 4.19M a module may have: -Os inlines every call site)
set( GLSLANG_OPT -O )
set( SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/spv" )
file( MAKE_DIRECTORY "${SPV_DIR}" )
set( SPV_HEADERS )
foreach( SHADER ${SHADER_FILES} )
  get_filename_component( NAME ${SHADER} NAME_WE )
  set( OUT "${SPV_DIR}/${NAME}.spv.h" )
  if( CMAKE_VERSION VERSION_GREATER_EQUAL 3.21 )
    add_custom_command( OUTPUT "${OUT}"
                        COMMAND "${ECM_GLSLANG}" -V --target-env vulkan1.3 ${GLSLANG_OPT}
                                -I${CMAKE_CURRENT_SOURCE_DIR}/shaders --vn spv_${NAME} --depfile "${OUT}.d" -o "${OUT}"
                                "${SHADER}"
                        DEPENDS "${SHADER}"
                        DEPFILE "${OUT}.d"
                        COMMENT "glslang ${NAME}.comp" VERBATIM )
  else()
    add_custom_command( OUTPUT "${OUT}"
                        COMMAND "${ECM_GLSLANG}" -V --target-env vulkan1.3 ${GLSLANG_OPT}
                                -I${CMAKE_CURRENT_SOURCE_DIR}/shaders --vn spv_${NAME} -o "${OUT}" "${SHADER}"
                        DEPENDS "${SHADER}" ${SHADER_INCLUDES}
                        COMMENT "glslang ${NAME}.comp" VERBATIM )
  endif()
  list( APPEND SPV_HEADERS "${OUT}" )
endforeach()

add_library( ${LIB_NAME} STATIC ${SRC_FILES} ${INC_FILES} ${SPV_HEADERS} ${SHADER_FILES} ${SHADER_INCLUDES} )
target_include_directories( ${LIB_NAME} PUBLIC . "${ECM_VULKAN_HEADERS}" PRIVATE "${SPV_DIR}" )
target_compile_definitions( ${LIB_NAME} PUBLIC ECM_VULKAN=1 VK_NO_PROTOTYPES )
target_link_libraries( ${LIB_NAME} CommonLib )
if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
  target_link_libraries( ${LIB_NAME} ${CMAKE_DL_LIBS} )
endif()

source_group( "Shaders" FILES ${SHADER_FILES} ${SHADER_INCLUDES} )
set_target_properties( ${LIB_NAME} PROPERTIES FOLDER lib )
