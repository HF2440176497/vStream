

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        CACHE INTERNAL "Top-level directory of this project")
endif()

project(tools_trt_demo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(TOOLS_DEMO_TRT_3RDPARTY_LIBS "")
set(TOOLS_DEMO_TRT_DEPENDENCIES "")


find_package(CUDAToolkit REQUIRED)
list(APPEND TOOLS_DEMO_TRT_3RDPARTY_LIBS 
  CUDA::cudart
  CUDA::nvml
)

set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT root directory")
endif()

# trt include
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include include/x86_64-linux-gnu
)

# trt lib
find_library(TENSORRT_LIB nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

find_library(TENSORRT_ONNXPARSER_LIB nvonnxparser
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)


if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR "TensorRT not found! Please set TENSORRT_ROOT environment variable or cmake option.")
else()
    list(APPEND TOOLS_DEMO_TRT_3RDPARTY_LIBS 
      ${TENSORRT_LIB}
      ${TENSORRT_ONNXPARSER_LIB}
      ${TENSORRT_PLUGIN_LIB}
    )
endif()

# Source files
# 每个源文件生成一个可执行文件
set(DEMO_TRT_SOURCES
  builder.cpp
)

foreach(demo_source ${DEMO_TRT_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    add_executable(${demo_name} ${demo_source})
    target_include_directories(${demo_name} PRIVATE 
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_libraries(${demo_name} 
        ${TOOLS_DEMO_TRT_3RDPARTY_LIBS}
        rt
        dl
        pthread
    )
endforeach()
