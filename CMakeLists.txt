

if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "Please create a separate directory for build files.")
endif()

cmake_minimum_required(VERSION 3.13)

if(WIN32)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupLibraryPaths_win.cmake)
endif()

if(UNIX)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupLibraryPaths_ubuntu.cmake)
endif()

project(vStream LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_ASAN "Enable AddressSanitizer" ON)

if(ENABLE_ASAN)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
endif()

# Platform options
option(VSTREAM_USE_CUDA "Enable NVIDIA CUDA platform support" ON)

if (VSTREAM_USE_CUDA)
  add_definitions(-DVSTREAM_USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
endif()

set(VSTREAM_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "Top-level directory of this project")

option(build_modules "build modules" ON)
option(build_tests "build unit tests" ON)
option(build_libyuv "build libyuv" ON)
option(build_tools "build tools" ON)
option(build_python_api "build python api" OFF)


# 静态库 and 动态库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${VSTREAM_ROOT_PATH}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${VSTREAM_ROOT_PATH}/lib)

# 可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${VSTREAM_ROOT_PATH}/bin)

# add macro definitions
if(build_tests)
  add_definitions(-DUNIT_TEST)
endif()

if (build_libyuv)
  set(LIBYUV_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(LIBYUV_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(LIBYUV_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libyuv)
  set(has_built_libyuv TRUE)
endif()

if(build_tests)
  # ---[ gtest
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGtest.cmake)
  
  if(GTEST_FOUND)
    message(STATUS "Found installed GTest (include: ${GTEST_INCLUDE_DIRS}, library: ${GTEST_LIBRARIES})")
    set(has_built_gtest TRUE)
  else()
    message(STATUS "GTest not found, building from 3rdparty/googletest")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest)
    set(has_built_gtest TRUE)
  endif()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/backward-cpp)

add_subdirectory(framework)
if(build_modules)
  add_subdirectory(modules)
endif()
if(build_tools)
  add_subdirectory(tools)
endif()

