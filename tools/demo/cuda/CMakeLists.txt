
# 单独编译 demo/cuda 下的源文件，临时验证

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        CACHE INTERNAL "Top-level directory of this project")
endif()

project(tools_cuda_demo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(TOOLS_DEMO_CUDA_3RDPARTY_LIBS "")
set(TOOLS_DEMO_CUDA_DEPENDENCIES "")

# ---[ CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
list(APPEND TOOLS_DEMO_CUDA_3RDPARTY_LIBS 
  CUDA::cudart
  CUDA::nvml
)
list(APPEND TOOLS_DEMO_CUDA_3RDPARTY_LIBS         
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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# ---[ opencv
include(${VSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND TOOLS_DEMO_CUDA_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}")

# ---[ libyuv
if(NOT DEFINED has_built_libyuv OR NOT has_built_libyuv)
  add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/libyuv ${CMAKE_CURRENT_BINARY_DIR}/lib/)
  set(has_built_libyuv TRUE)
endif()
if(TARGET yuv)
  list(APPEND TOOLS_DEMO_CUDA_3RDPARTY_LIBS yuv)
  list(APPEND TOOLS_DEMO_CUDA_DEPENDENCIES yuv)
  # message(STATUS "----- Using libyuv static library: yuv")
elseif(TARGET yuv_shared)
  list(APPEND TOOLS_DEMO_CUDA_3RDPARTY_LIBS yuv_shared)
  list(APPEND TOOLS_DEMO_CUDA_DEPENDENCIES yuv_shared)
  # message(STATUS "----- Using libyuv shared library: yuv_shared")
else()
  message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
endif()
include_directories(${VSTREAM_ROOT_PATH}/3rdparty/libyuv/include)

# Source files
set(DEMO_CUDA_SOURCES
  inspect_mem.cpp
  transfmt_cuda_npp_demo.cu
  transfmt_cuda_kernel_demo.cu
)

foreach(demo_source ${DEMO_CUDA_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    cuda_add_executable(${demo_name} ${demo_source})
    add_dependencies(${demo_name} ${TOOLS_DEMO_CUDA_DEPENDENCIES})
    target_link_libraries(${demo_name} 
      ${TOOLS_DEMO_CUDA_3RDPARTY_LIBS}
      rt
      dl
      pthread
    )
endforeach()
