

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
  set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    CACHE INTERNAL "Top-level directory of this project")
endif()

project(tools_ffmpeg_demo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(VSTREAM_TOOLS_DEMO_FFMPEG_3RDPARTY_INCLUDE "")
set(VSTREAM_TOOLS_DEMO_FFMPEG_3RDPARTY_LIBS "")

set(VSTREAM_TOOLS_DEMO_FFMPEG_SOURCES
  encode_demo.cpp
)


# ---[ FFmpeg
include(${VSTREAM_ROOT_PATH}/cmake/FindFFmpeg.cmake)
if(FFMPEG_FOUND)
  list(APPEND VSTREAM_TOOLS_DEMO_FFMPEG_3RDPARTY_INCLUDE ${FFMPEG_INCLUDE_DIR})
  list(APPEND VSTREAM_TOOLS_DEMO_FFMPEG_3RDPARTY_LIBS ${FFMPEG_LIBRARIES})
  if(FFMPEG_ROOT_DIR)
    list(APPEND VSTREAM_TOOLS_DEMO_FFMPEG_3RDPARTY_LIBS ${VFFMPEG_ROOT_DIR}/lib)
  endif()
  message(STATUS "----- FFmpeg found Includes: ${FFMPEG_INCLUDE_DIR} Libraries: ${FFMPEG_LIBRARIES}")
else()
  message(FATAL_ERROR "FFmpeg not found! Please ensure FFmpeg is installed and set FFMPEG_ROOT_DIR")
endif()

# 单文件对应单个可执行文件的模式
foreach(demo_source ${VSTREAM_TOOLS_DEMO_FFMPEG_SOURCES})
  get_filename_component(demo_name ${demo_source} NAME_WE)
  add_executable(${demo_name} ${demo_source})
  target_include_directories(${demo_name}
   PRIVATE
    ${VSTREAM_TOOLS_DEMO_FFMPEG_3RDPARTY_INCLUDE}
  )
  target_link_libraries(${demo_name}
   PRIVATE
    ${VSTREAM_TOOLS_DEPENDENCIES}
    ${VSTREAM_TOOLS_3RDPARTY_LIBS}
    ${VSTREAM_TOOLS_DEMO_FFMPEG_3RDPARTY_LIBS}
  )
endforeach()



