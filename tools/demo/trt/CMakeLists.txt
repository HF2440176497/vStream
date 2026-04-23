

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
  set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    CACHE INTERNAL "Top-level directory of this project")
endif()

project(tools_trt_demo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(VSTREAM_TOOLS_DEMO_TRT_3RDPARTY_INCLUDE "")
set(VSTREAM_TOOLS_DEMO_TRT_3RDPARTY_LIBS "")

find_package(CUDAToolkit REQUIRED)
list(APPEND VSTREAM_TOOLS_DEMO_TRT_3RDPARTY_LIBS 
  CUDA::cudart
  CUDA::nvml
)

if(NOT DEFINED TENSORRT_FOUND OR NOT TENSORRT_FOUND)
  include(${VSTREAM_ROOT_PATH}/cmake/FindTensorRT.cmake)
endif()

if(TENSORRT_FOUND)
  list(APPEND VSTREAM_TOOLS_DEMO_TRT_3RDPARTY_INCLUDE ${TENSORRT_INCLUDE_DIR})
  list(APPEND VSTREAM_TOOLS_DEMO_TRT_3RDPARTY_LIBS ${TENSORRT_LIBRARIES})
else()
  message(FATAL_ERROR "TensorRT not found. Can not complie trt_demo")
endif()

# Source files
# 每个源文件生成一个可执行文件
set(VSTREAM_TOOLS_DEMO_TRT_SOURCES trans_model.cpp)

foreach(demo_source ${VSTREAM_TOOLS_DEMO_TRT_SOURCES})
  get_filename_component(demo_name ${demo_source} NAME_WE)
  add_executable(${demo_name} ${demo_source})
  target_include_directories(${demo_name}
   PRIVATE 
    ${VSTREAM_TOOLS_3RDPARTY_INCLUDE}  # from tools/CMakeLists.txt
    ${VSTREAM_TOOLS_DEMO_TRT_3RDPARTY_INCLUDE}
  )
  target_link_libraries(${demo_name} 
   PUBLIC
    ${VSTREAM_TOOLS_DEPENDENCIES}
   PRIVATE
    ${VSTREAM_TOOLS_3RDPARTY_LIBS}
    ${VSTREAM_TOOLS_DEMO_TRT_3RDPARTY_LIBS}
  )
endforeach()
