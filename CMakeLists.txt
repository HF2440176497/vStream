

cmake_minimum_required(VERSION 3.22)

if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "Please create a separate directory for build files.")
endif()

if(NOT UNIX)
  message(FATAL_ERROR "only support UNIX platform")
endif()

set(VSTREAM_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "Top-level directory of this project")

# ============================================================
# 版本唯一来源：仓库根目录 VERSION 文件（单行，如 1.0.0）
# 发版时只改该文件，然后重新执行 cmake 配置即可
# ============================================================
set(VSTREAM_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
if(NOT EXISTS "${VSTREAM_VERSION_FILE}")
  message(FATAL_ERROR "VERSION file not found: ${VSTREAM_VERSION_FILE}")
endif()
file(STRINGS "${VSTREAM_VERSION_FILE}" VSTREAM_VERSION LIMIT_COUNT 1)
string(STRIP "${VSTREAM_VERSION}" VSTREAM_VERSION)
if(NOT VSTREAM_VERSION MATCHES "^[0-9][A-Za-z0-9.+~-]*$")
  message(FATAL_ERROR
    "Invalid version '${VSTREAM_VERSION}' in ${VSTREAM_VERSION_FILE} (expected e.g. 1.0.0)")
endif()
message(STATUS "vStream version: ${VSTREAM_VERSION}")

# ============================================================
# 平台选择：唯一入口，须在 project() 之前
#   - 只解析平台 ID（-DVSTREAM_PLATFORM / host 默认）并做白名单校验
#   - 不加载平台文件：加载动作见 vstream_platform_load()
# ============================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolvePlatform.cmake)

project(vStream LANGUAGES C CXX)

# 组件开关
option(VSTREAM_BUILD_MODULES "build modules" ON)
option(VSTREAM_BUILD_TOOLS "build tools" ON)

# ============================================================
# 平台配置：唯一执行副作用的地方，须在 project() 之后
#   - 加载 cmake/platforms/<id>.cmake（此时工具链的 RK_* 已可用）
#   - enable_language / find_package / 编译标准 / vstream_platform 目标
# ============================================================
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PlatformBase.cmake)
vstream_platform_load()
vstream_platform_validate()
vstream_platform_configure()

# 第三方库路径默认值（FFMPEG_ROOT_DIR / OPENCV_ROOT_DIR）
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupLibraryPaths_ubuntu.cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type: Debug or Release" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# CMAKE_CXX_STANDARD / CMAKE_CUDA_STANDARD / CUDA 架构由平台契约提供
# vstream_platform_configure() 设置


# 单元测试
option(VSTREAM_BUILD_TESTS "build unit tests" ON)

# Python 绑定
option(VSTREAM_BUILD_PYTHON_API "build python api" ON)

# 默认都为 ON
option(VSTREAM_BUILD_LIBYUV "build libyuv" ON)
option(VSTREAM_BUILD_PYBIND11 "build pybind11" ON)

# Release 构建是否保留调试符号（-g），默认关闭：
# 开启后 backward-cpp + libdw 可将崩溃堆栈解析到函数名/文件/行号，产物体积增大；
option(VSTREAM_RELEASE_WITH_DEBUG_INFO "keep debug symbols (-g) in Release builds" OFF)

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
  if(VSTREAM_RELEASE_WITH_DEBUG_INFO)
    set(VSTREAM_RELEASE_FLAGS "-O2 -DNDEBUG -g")
    message(STATUS "Release build: keeping debug symbols (-g) for crash stacktraces")
  else()
    set(VSTREAM_RELEASE_FLAGS "-O2 -DNDEBUG")
    message(STATUS "Release build: no debug symbols, crash stacktraces will show addresses only")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${VSTREAM_RELEASE_FLAGS}" CACHE STRING "C++ flags for Release" FORCE)
    set(CMAKE_C_FLAGS_RELEASE "${VSTREAM_RELEASE_FLAGS}" CACHE STRING "C flags for Release" FORCE)
  endif()
  if(VSTREAM_HAS_CUDA AND CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    set(CMAKE_CUDA_FLAGS_RELEASE "${VSTREAM_RELEASE_FLAGS}" CACHE STRING "CUDA flags for Release" FORCE)
  endif()
endif()

# Python 路径配置
set(VSTREAM_PYTHON_ROOT_DIR "/usr/local/python3.13"
    CACHE PATH "Custom Python root directory (e.g. /usr/local/python3.13). If not set, use system default.")
set(VSTREAM_PYTHON_EXECUTABLE "/usr/local/bin/python3.13"
    CACHE FILEPATH "Custom Python executable (e.g. /usr/local/python3.13/bin/python3.13). If not set, use system default.")

# ============================================================
# 打包相关：安装前缀与 RPATH
# 必须在任何 target 创建之前设置 —— 
# CMAKE_INSTALL_RPATH 只在 target 创建时 用于初始化该 target 的 INSTALL_RPATH 属性
# ============================================================
if(VSTREAM_PACKAGE)
  # 默认安装前缀（可被 -DCMAKE_INSTALL_PREFIX=... 覆盖）
  if(NOT CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
    set(CMAKE_INSTALL_PREFIX "/usr/local/vstream" CACHE PATH "vStream install prefix" FORCE)
  endif()

  # RPATH 只覆盖 vstream 自身库（同级 lib/ 目录），第三方依赖交给 ld.so.conf.d 自适应排序
  set(CMAKE_SKIP_BUILD_RPATH OFF)
  # 交叉编译（rockchip）时： toolchain 里的 CMAKE_BUILD_WITH_INSTALL_RPATH 设置 TRUE
  # 使构建目录产物可直接上板运行；原生构建则用 build RPATH
  if(NOT CMAKE_CROSSCOMPILING)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
  endif()
  # $ORIGIN           → 覆盖装在 lib/ 的库（找同目录的兄弟库）
  # $ORIGIN/../lib    → 覆盖装在 bin/ 的可执行文件（库在上一级 lib/）
  # 两者同时给出：找不到就跳过，对 lib/ 下的库无副作用。
  # 注意：CMAKE_INSTALL_RPATH 是分号列表，CMake 写进 ELF 时会转成 ':' 分隔。
  set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
endif()

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
# 在 build.sh 设置默认值 ON
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

# ---[ gflags / glog：统一从 3rdparty 随主构建
# 共享库编译到 lib/，链接 in-tree target，随包安装到 lib/（不依赖系统安装）
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupThirdPartyInTree.cmake)

# ---[ gtest：统一从 3rdparty 随主构建（静态库，仅供单测链接，不安装）
if(VSTREAM_BUILD_TESTS)
  add_definitions(-DVSTREAM_UNIT_TEST)
  set(INSTALL_GTEST OFF CACHE BOOL "googletest: install" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "googletest: build self tests" FORCE)
  add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/googletest)
  set(VSTREAM_HAS_BUILT_GTEST TRUE)
  message(STATUS "GTest building from 3rdparty/googletest")
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
# 导出平台变量：供 install(CODE) 阶段的脚本（打包依赖收集）读取
# 必须在所有可能修改 VSTREAM_PLATFORM_* 的步骤之后
# ============================================================
vstream_export_platform_vars()

# ============================================================
# 打包相关：CPack（RPATH 与安装前缀已在 target 创建之前设置）
# ============================================================
if(VSTREAM_PACKAGE)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Packaging.cmake)
endif()
