# executable: unit tests of the GPU decoder kernels against the CPU code
set( EXE_NAME VulkanTestApp )

file( GLOB SRC_FILES "*.cpp" )
file( GLOB INC_FILES "*.h" )

add_executable( ${EXE_NAME} ${SRC_FILES} ${INC_FILES} )

if( CMAKE_COMPILER_IS_GNUCC AND BUILD_STATIC )
  set( ADDITIONAL_LIBS ${ADDITIONAL_LIBS} -static -static-libgcc -static-libstdc++ )
endif()

target_link_libraries( ${EXE_NAME} CommonLib VulkanLib ${ADDITIONAL_LIBS} )
set_target_properties( ${EXE_NAME} PROPERTIES FOLDER app LINKER_LANGUAGE CXX )
