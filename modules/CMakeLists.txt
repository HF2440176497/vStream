cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(MODULES_3RDPARTY_LIBS "")
set(MODULES_DEPENDENCIES "")

# ---[ gflags
include(${VSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
  message(STATUS "Using installed GFlags for modules unittest")
  include_directories(${GFLAGS_INCLUDE_DIRS})
  list(APPEND MODULES_3RDPARTY_LIBS ${GFLAGS_LIBRARIES})
else()
  message(FATAL_ERROR "GFlags not found! Please ensure GFlags is installed and set GFLAGS_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ glog
include(${VSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
if(GLOG_FOUND)
  message(STATUS "Using installed GLog for modules unittest")
  include_directories(${GLOG_INCLUDE_DIRS})
  list(APPEND MODULES_3RDPARTY_LIBS ${GLOG_LIBRARIES})
else()
  message(FATAL_ERROR "GLog not found! Please ensure GLog is installed and set GLOG_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ nlohmann/json
include_directories(${VSTREAM_ROOT_PATH}/3rdparty/nlohmann-json/include)

# ---[ framework
include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${VSTREAM_ROOT_PATH})
if(HAVE_FRAMEWORK_TARGET)
  include_directories(${VSTREAM_ROOT_PATH}/framework/core/include)
  list(APPEND MODULES_DEPENDENCIES vstream_core)
else()
  message(FATAL_ERROR "framework target not found")
endif()

if(build_tests)
  add_definitions(-DUNIT_TEST)
endif()

option(build_source "build module source" ON)
option(build_inference "build module inference" ON)
option(build_va "build stream_va" ON)

# ---[ opencv
include(${VSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND MODULES_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

# ---[ FFmpeg
include(${VSTREAM_ROOT_PATH}/cmake/FindFFmpeg.cmake)
if(FFMPEG_FOUND)
  list(APPEND MODULES_3RDPARTY_LIBS ${FFMPEG_LIBRARIES})
  include_directories(${FFMPEG_INCLUDE_DIR})
  if(FFMPEG_ROOT_DIR)
    link_directories(${FFMPEG_ROOT_DIR}/lib)
  endif()
  message(STATUS "----- FFmpeg found Includes: ${FFMPEG_INCLUDE_DIR} Libraries: ${FFMPEG_LIBRARIES}")
else()
  message(FATAL_ERROR "FFmpeg not found! Please ensure FFmpeg is installed and set FFMPEG_ROOT_DIR")
endif()

# ---[ libyuv
if(DEFINED has_built_libyuv AND has_built_libyuv)
  if(TARGET yuv)  # 根据 libyuv 内的 cmake 文件确定
    list(APPEND MODULES_3RDPARTY_LIBS yuv)
    list(APPEND MODULES_DEPENDENCIES yuv)
    # message(STATUS "----- Using libyuv static library: yuv")
  elseif(TARGET yuv_shared)
    list(APPEND MODULES_3RDPARTY_LIBS yuv_shared)
    list(APPEND MODULES_DEPENDENCIES yuv_shared)
    # message(STATUS "----- Using libyuv shared library: yuv_shared")
  else()
    message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
  endif()
  include_directories(${VSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
else()
  message(FATAL_ERROR "libyuv has not built. Set BUILD_LIBYUV=ON in root CMakeLists.txt to build")
endif()

set(module_list "")
if(build_source)
  list(APPEND module_list source)
endif()

if(build_inference)
  list(APPEND module_list inference)
endif()

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB util_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/*.cpp)

if(VSTREAM_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  file(GLOB util_cuda_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cu)
  list(APPEND util_src ${util_cuda_src})
  
  list(APPEND MODULES_3RDPARTY_LIBS 
    CUDA::cudart
    CUDA::nvml
  )
  list(APPEND MODULES_3RDPARTY_LIBS         
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
    list(APPEND MODULES_3RDPARTY_LIBS ${TENSORRT_LIBRARIES})
    include_directories(${TENSORRT_INCLUDE_DIR})
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
file(GLOB proc_common_pre_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/preproc/*.cpp)
file(GLOB proc_common_post_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/postproc/*.cpp)
list(APPEND proc_common_srcs ${proc_common_pre_src} ${proc_common_post_src})

list(APPEND srcs ${proc_common_srcs})

if (VSTREAM_USE_CUDA)
  set(proc_cuda_srcs "")
  file(GLOB proc_cuda_pre_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/cuda/preproc/*.cpp)
  file(GLOB proc_cuda_post_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/cuda/postproc/*.cpp)
  list(APPEND proc_cuda_srcs ${proc_cuda_pre_src} ${proc_cuda_post_src})
  list(APPEND srcs ${proc_cuda_srcs})
endif()


if(build_va)
  add_library(vstream_va SHARED ${srcs})
  target_include_directories(vstream_va PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/util/include
  )

  foreach(module ${module_list})
    target_include_directories(vstream_va PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${module}/include
    )
  endforeach()

  add_dependencies(vstream_va ${MODULES_DEPENDENCIES})
  target_link_libraries(vstream_va ${MODULES_DEPENDENCIES} ${MODULES_3RDPARTY_LIBS} rt dl pthread)
  add_custom_target(va DEPENDS vstream_va)
  # set_target_properties(vstream_va PROPERTIES LINK_FLAGS_RELEASE -s)
  message(STATUS "vstream_va enabled - targets: vstream_va")
endif()

if(build_tests)
  add_subdirectory(unittest)
endif()