
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "Please create a separate directory for build files.")
endif()

cmake_minimum_required(VERSION 3.16)

if(UNIX)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupLibraryPaths_ubuntu.cmake)
else()
  message(FATAL_ERROR "only support UNIX platform")
endif()

project(vStream LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type: Debug or Release" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform options
option(VSTREAM_USE_CUDA "Enable NVIDIA CUDA platform support" ON)
option(VSTREAM_USE_ROCKCHIP "Enable Rockchip RKMPP platform support" OFF)

if (VSTREAM_USE_CUDA)
  add_definitions(-DVSTREAM_USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
endif()

if (VSTREAM_USE_ROCKCHIP)
  add_definitions(-DVSTREAM_USE_ROCKCHIP)
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

# 打包相关选项
option(VSTREAM_PACKAGE "Enable CPack packaging support" OFF)
option(VSTREAM_PACKAGE_BUNDLE_DEPS
       "Bundle third-party deps into <prefix>/dev/"
       OFF)
option(VSTREAM_PACKAGE_INCLUDE_TOOLS
       "Include tools in the package" OFF)
option(VSTREAM_PACKAGE_INCLUDE_TESTS
       "Include unit-test executables in the package" OFF)


# Release 构建类型编译器优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "C++ flags for Release" FORCE)
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "C flags for Release" FORCE)
  endif()
  if(VSTREAM_USE_CUDA AND CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "CUDA flags for Release" FORCE)
  endif()
endif()

# Python 路径配置
set(VSTREAM_PYTHON_ROOT_DIR "/usr/local/python3.13"
    CACHE PATH "Custom Python root directory (e.g. /usr/local/python3.13). If not set, use system default.")
set(VSTREAM_PYTHON_EXECUTABLE "/usr/local/bin/python3.13"
    CACHE FILEPATH "Custom Python executable (e.g. /usr/local/python3.13/bin/python3.13). If not set, use system default.")

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

  # install libyuv（随 vstream 打包，装到 lib/）
  if(VSTREAM_PACKAGE)
    if(TARGET yuv_shared)
      install(TARGETS yuv_shared
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
      )
    elseif(TARGET yuv)
      install(TARGETS yuv
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
      )
    endif()
  endif()
endif()

# ---[ pybind11
if (VSTREAM_BUILD_PYBIND11)
  message(STATUS "Pybind11 building from 3rdparty/pybind11")

  # 如果用户指定了自定义 Python 路径，则强制使用
  if(VSTREAM_PYTHON_ROOT_DIR)
    set(Python_ROOT_DIR "${VSTREAM_PYTHON_ROOT_DIR}" CACHE PATH "Python root directory" FORCE)
    set(Python3_ROOT_DIR "${VSTREAM_PYTHON_ROOT_DIR}" CACHE PATH "Python3 root directory" FORCE)
    message(STATUS "Using custom Python_ROOT_DIR: ${VSTREAM_PYTHON_ROOT_DIR}")
  endif()
  if(VSTREAM_PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE "${VSTREAM_PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable" FORCE)
    set(Python3_EXECUTABLE "${VSTREAM_PYTHON_EXECUTABLE}" CACHE FILEPATH "Python3 executable" FORCE)
    set(PYTHON_EXECUTABLE "${VSTREAM_PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable" FORCE)
    message(STATUS "Using custom Python_EXECUTABLE: ${VSTREAM_PYTHON_EXECUTABLE}")
  endif()

  set(PYBIND11_PYTHON_VERSION 3)
  add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/pybind11)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)

  message(STATUS "VSTREAM_BUILD_PYBIND11: using Python ")
  message(STATUS "  Interpreter  : ${Python3_EXECUTABLE}")
  message(STATUS "  Version      : ${Python3_VERSION}")
  message(STATUS "  Include dir  : ${Python3_INCLUDE_DIRS}")
  message(STATUS "  Library      : ${Python3_LIBRARIES}")

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

# ============================================================
# 打包相关：RPATH 与 CPack
# ============================================================
# RPATH 只覆盖 vstream 自身库（同级 lib/ 目录），第三方依赖交给 ld.so.conf.d 自适应排序

if(VSTREAM_PACKAGE)
  set(CMAKE_SKIP_BUILD_RPATH OFF)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)

  set(CMAKE_INSTALL_RPATH "$ORIGIN")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

  # 默认安装前缀（可被 -DCMAKE_INSTALL_PREFIX=... 覆盖）
  if(NOT CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "/usr/local/vstream" CACHE PATH "vStream install prefix" FORCE)
  endif()

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Packaging.cmake)
endif()
