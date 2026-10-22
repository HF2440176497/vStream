cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED VSTREAM_ROOT_PATH)
  set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

include(${VSTREAM_ROOT_PATH}/cmake/FindFFmpeg.cmake)
if(NOT FFMPEG_FOUND)
  message(FATAL_ERROR "cuda_init_repro requires FFmpeg. Set FFMPEG_ROOT_DIR.")
endif()

add_executable(cuda_init_repro main.cpp)

target_include_directories(cuda_init_repro PRIVATE
  ${FFMPEG_INCLUDE_DIR}
)

target_link_libraries(cuda_init_repro PRIVATE
  CUDA::cudart
  Threads::Threads
  ${FFMPEG_LIBRARIES}
)

vstream_platform_apply(cuda_init_repro)

if(VSTREAM_PACKAGE AND VSTREAM_PACKAGE_INCLUDE_TOOLS)
  set_target_properties(cuda_init_repro PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
  )
  install(TARGETS cuda_init_repro RUNTIME DESTINATION bin)
endif()
