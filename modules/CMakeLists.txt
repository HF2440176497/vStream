cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(VSTREAM_MODULES_3RDPARTY_INCLUDE "")
set(VSTREAM_MODULES_3RDPARTY_LIBS "")
set(VSTREAM_MODULES_DEPENDENCIES "")

# ---[ gflags
include(${VSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
  message(STATUS "Using installed GFlags for modules unittest")
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${GFLAGS_INCLUDE_DIRS})
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${GFLAGS_LIBRARIES})
else()
  message(FATAL_ERROR "GFlags not found! Please ensure GFlags is installed and set GFLAGS_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ glog
include(${VSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
if(GLOG_FOUND)
  message(STATUS "Using installed GLog for modules unittest")
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${GLOG_INCLUDE_DIRS})
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${GLOG_LIBRARIES})
else()
  message(FATAL_ERROR "GLog not found! Please ensure GLog is installed and set GLOG_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ libyuv
if(DEFINED VSTREAM_HAS_BUILT_LIBYUV AND VSTREAM_HAS_BUILT_LIBYUV)
  if(TARGET yuv)  # 根据 libyuv 内的 cmake 文件确定
    list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS yuv)
  elseif(TARGET yuv_shared)
    list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS yuv_shared)
  else()
    message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
  endif()
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${VSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
else()
  message(FATAL_ERROR "libyuv has not built. Set BUILD_LIBYUV=ON in root CMakeLists.txt to build")
endif()

# ---[ FFmpeg
include(${VSTREAM_ROOT_PATH}/cmake/FindFFmpeg.cmake)
if(FFMPEG_FOUND)
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${FFMPEG_INCLUDE_DIR})
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${FFMPEG_LIBRARIES})
  if(FFMPEG_ROOT_DIR)
    list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${VFFMPEG_ROOT_DIR}/lib)
  endif()
  message(STATUS "----- FFmpeg found Includes: ${FFMPEG_INCLUDE_DIR} Libraries: ${FFMPEG_LIBRARIES}")
else()
  message(FATAL_ERROR "FFmpeg not found! Please ensure FFmpeg is installed and set FFMPEG_ROOT_DIR")
endif()

# ---[ opencv
include(${VSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${OpenCV_INCLUDE_DIRS})
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

# ---[ nlohmann/json
list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${VSTREAM_ROOT_PATH}/3rdparty/nlohmann-json/include)

# ---[ framework
include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${VSTREAM_ROOT_PATH})
if(HAVE_FRAMEWORK_TARGET)
  list(APPEND VSTREAM_MODULES_DEPENDENCIES vstream_core)
else()
  message(FATAL_ERROR "framework target not found")
endif()

if(VSTREAM_BUILD_TESTS)
  add_definitions(-DVSTREAM_UNIT_TEST)
endif()

option(VSTREAM_MODULES_BUILD_SOURCE "build module source" ON)
option(VSTREAM_MODULES_BUILD_SINK "build module sink" ON)
option(VSTREAM_MODULES_BUILD_INFERENCE "build module inference" ON)
option(VSTREAM_MODULES_BUILD_OSD "build module osd" ON)
option(VSTREAM_MODULES_BUILD_VA "build stream_va" ON)

set(module_list "")
if(VSTREAM_MODULES_BUILD_SOURCE)
  list(APPEND module_list source)
endif()

if(VSTREAM_MODULES_BUILD_SINK)
  list(APPEND module_list sink)
endif()

if(VSTREAM_MODULES_BUILD_INFERENCE)
  list(APPEND module_list inference)
endif()

if(VSTREAM_MODULES_BUILD_OSD)
  list(APPEND module_list osd)
endif()

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB util_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/*.cpp)

if(VSTREAM_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  file(GLOB util_cuda_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cu)
  list(APPEND util_src ${util_cuda_src})
  
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS 
    CUDA::cudart
    CUDA::nvml
  )
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS         
    CUDA::npps
    CUDA::nppc
    CUDA::nppial
    CUDA::nppicc
    CUDA::nppidei
    CUDA::nppif
    CUDA::nppig
    CUDA::nppim
    CUDA::nppist
    CUDA::nppisu
    CUDA::nppitc
  )
    
  message(STATUS "VSTREAM_USE_CUDA enabled - adding CUDA sources")
endif()

list(APPEND srcs ${util_src})

if(VSTREAM_USE_CUDA)
  include(${VSTREAM_ROOT_PATH}/cmake/FindTensorRT.cmake)
  if(TENSORRT_FOUND)
    list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${TENSORRT_INCLUDE_DIR})
    list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${TENSORRT_LIBRARIES})
  endif()
endif()

# 递归添加模块的源文件: 要求每个 module 目录都是 include src 目录结构
foreach(module ${module_list})
  file(GLOB ${module}_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${module}/src/*.cpp)

  if (VSTREAM_USE_CUDA)
    file(GLOB ${module}_cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${module}/src/cuda/*.cpp)
    list(APPEND ${module}_srcs ${${module}_cuda_srcs})
  endif()

  list(APPEND srcs ${${module}_srcs})
endforeach()

# 前后处理模块
set(proc_common_srcs "")
file(GLOB proc_common_pre_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/common/preproc/*.cpp)
file(GLOB proc_common_post_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/common/postproc/*.cpp)
list(APPEND proc_common_srcs ${proc_common_pre_src} ${proc_common_post_src})

list(APPEND srcs ${proc_common_srcs})

if (VSTREAM_USE_CUDA)
  set(proc_cuda_srcs "")
  file(GLOB proc_cuda_pre_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/cuda/preproc/*.cpp)
  file(GLOB proc_cuda_post_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/cuda/postproc/*.cpp)
  list(APPEND proc_cuda_srcs ${proc_cuda_pre_src} ${proc_cuda_post_src})
  list(APPEND srcs ${proc_cuda_srcs})
endif()


if(VSTREAM_MODULES_BUILD_VA)
  add_library(vstream_va SHARED ${srcs})
  target_include_directories(vstream_va 
   PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/util/include
   PRIVATE
    ${VSTREAM_MODULES_3RDPARTY_INCLUDE}
  )

  foreach(module ${module_list})
    target_include_directories(vstream_va 
     PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/${module}/include
    )
  endforeach()

  target_link_libraries(vstream_va
   PUBLIC
    ${VSTREAM_MODULES_DEPENDENCIES} 
   PRIVATE
    ${VSTREAM_MODULES_3RDPARTY_LIBS})  # vstream_core
  message(STATUS "vstream_va enabled")
endif()

if(VSTREAM_BUILD_TESTS)
  add_subdirectory(unittest)
endif()