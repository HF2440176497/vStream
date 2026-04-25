
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "Please create a separate directory for build files.")
endif()

cmake_minimum_required(VERSION 3.13)

if(UNIX)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupLibraryPaths_ubuntu.cmake)
else()
  message(FATAL_ERROR "only support UNIX platform")
endif()

project(vStream LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform options
option(VSTREAM_USE_CUDA "Enable NVIDIA CUDA platform support" ON)

if (VSTREAM_USE_CUDA)
  add_definitions(-DVSTREAM_USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
endif()

set(VSTREAM_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "Top-level directory of this project")

# 单元测试
option(VSTREAM_BUILD_TESTS "build unit tests" ON)

# 模块
option(VSTREAM_BUILD_MODULES "build modules" ON)
option(VSTREAM_BUILD_PYTHON_API "build python api" ON)
option(VSTREAM_BUILD_TOOLS "build tools" ON)

# 第三方库 默认都为 ON
option(VSTREAM_BUILD_LIBYUV "build libyuv" ON)
option(VSTREAM_BUILD_PYBIND11 "build pybind11" ON)

# 静态库 and 动态库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VSTREAM_ROOT_PATH}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VSTREAM_ROOT_PATH}/lib)

# 可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${VSTREAM_ROOT_PATH}/bin)

# ---[ libyuv
if (VSTREAM_BUILD_LIBYUV)
  set(LIBYUV_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(LIBYUV_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(LIBYUV_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/libyuv)
  set(VSTREAM_HAS_BUILT_LIBYUV TRUE)
endif()

# ---[ pybind11
if (VSTREAM_BUILD_PYBIND11)
  message(STATUS "Pybind11 building from 3rdparty/pybind11")
  set(PYBIND11_PYTHON_VERSION 3)
  add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/pybind11)
  set(VSTREAM_HAS_BUILT_PYBIND11 TRUE)
endif()

# 当使用系统安装的 gtest，VSTREAM_HAS_BUILT_GTEST = FALSE
if(VSTREAM_BUILD_TESTS)
  add_definitions(-DVSTREAM_UNIT_TEST)
  # ---[ gtest
  include(${VSTREAM_ROOT_PATH}/cmake/FindGtest.cmake)
  if(GTEST_FOUND)
    message(STATUS "Found installed GTest (include: ${GTEST_INCLUDE_DIRS}, lib: ${GTEST_LIBRARIES})")
    set(VSTREAM_HAS_BUILT_GTEST FALSE)
  else()
    message(STATUS "GTest not found, building from 3rdparty/googletest")
    add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/googletest)
    set(VSTREAM_HAS_BUILT_GTEST TRUE)
  endif()
endif()

add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/backward-cpp)

# ------ vstream all parts
add_subdirectory(framework)
if(VSTREAM_BUILD_MODULES)
  add_subdirectory(modules)
endif()
if (VSTREAM_BUILD_PYTHON_API)
  add_subdirectory(${VSTREAM_ROOT_PATH}/python)
endif()
if(VSTREAM_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
