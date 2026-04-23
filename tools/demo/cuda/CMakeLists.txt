
# 单独编译 demo/cuda 下的源文件，临时验证

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
  set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    CACHE INTERNAL "Top-level directory of this project")
endif()

project(tools_cuda_demo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(VSTREAM_TOOLS_DEMO_CUDA_3RDPARTY_LIBS "")

# ---[ CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
list(APPEND VSTREAM_TOOLS_DEMO_CUDA_3RDPARTY_LIBS 
  CUDA::cudart
  CUDA::nvml
)
list(APPEND VSTREAM_TOOLS_DEMO_CUDA_3RDPARTY_LIBS 
  CUDA::cudart
  CUDA::nvml
)
list(APPEND VSTREAM_TOOLS_DEMO_CUDA_3RDPARTY_LIBS         
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

# Source files
set(VSTREAM_TOOLS_DEMO_CUDA_SOURCES
  inspect_mem.cpp
  transfmt_cuda_npp_demo.cu
  transfmt_cuda_kernel_demo.cu
)

foreach(demo_source ${VSTREAM_TOOLS_DEMO_CUDA_SOURCES})
  get_filename_component(demo_name ${demo_source} NAME_WE)
  cuda_add_executable(${demo_name} ${demo_source})
  target_include_directories(${demo_name} 
   PRIVATE 
    ${VSTREAM_TOOLS_3RDPARTY_INCLUDE}
  )
  target_link_libraries(${demo_name}
   PUBLIC
    ${VSTREAM_TOOLS_DEPENDENCIES}
   PRIVATE
    ${VSTREAM_TOOLS_3RDPARTY_LIBS}
    ${VSTREAM_TOOLS_DEMO_CUDA_3RDPARTY_LIBS}
  )
endforeach()
