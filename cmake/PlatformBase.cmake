# =============================================================================
#
# 平台文件（cmake/platforms/<id>.cmake）只 set 变量；
# 本文件负责把变量变成 find_package / enable_language / target_* 等动作。
#
# 提供：
#   vstream_platform_validate()      契约完整性校验
#   vstream_platform_configure()     语言 / 依赖 / vstream_platform 目标（project() 之后）
#   vstream_platform_apply(<tgt>)    编译定义 + include
#   vstream_platform_link(<tgt>)     链接平台库
#   vstream_platform_sources(<out> <module_dir>)  按平台收集源码
#   vstream_export_platform_vars()   导出 build/platform-vars.cmake（供 install 阶段使用）
# =============================================================================

include_guard(GLOBAL)

# ---- 契约必备项：由平台文件全部显式赋值（允许为空）----
set(VSTREAM_CONTRACT_KEYS
  ID DESC LANGUAGES DEFINES CXX_STANDARD CUDA_STANDARD CUDA_ARCHITECTURES
  SRC_SUBDIRS TOOL_SUBDIRS TEST_SUBDIRS
  PACKAGES FIND_MODULES LIBS INCLUDE_DIRS
  FFMPEG_ROOT OPENCV_ROOT
  DEFAULT_ENCODER DEFAULT_DECODER
  DEP_SEARCH_DIRS DEP_EXCLUDE_EXTRA EXTRA_BUNDLE_DIRS)

# -----------------------------------------------------------------------------
# 契约校验
# -----------------------------------------------------------------------------
function(vstream_platform_validate)
  set(_missing "")
  foreach(_key ${VSTREAM_CONTRACT_KEYS})
    if(NOT DEFINED VSTREAM_PLATFORM_${_key})
      list(APPEND _missing "VSTREAM_PLATFORM_${_key}")
    endif()
  endforeach()
  if(_missing)
    message(FATAL_ERROR
      "平台文件 cmake/platforms/${VSTREAM_PLATFORM}.cmake 缺少契约项：\n  ${_missing}\n"
      "契约清单见 cmake_platform_refactor_plan.md 第 2.1 节")
  endif()

  if(NOT VSTREAM_PLATFORM_ID STREQUAL VSTREAM_PLATFORM)
    message(FATAL_ERROR
      "平台文件 ID (${VSTREAM_PLATFORM_ID}) 与文件名 (${VSTREAM_PLATFORM}) 不一致")
  endif()
endfunction()

# -----------------------------------------------------------------------------
# 平台配置：必须在 project() 之后调用
# 用 macro 而非 function —— 需要把 CMAKE_* 与聚合后的依赖写回目录作用域
# -----------------------------------------------------------------------------
macro(vstream_platform_configure)
  # ---- 平台与工具链一致性：捕捉缓存残留（x86 工具链 + rockchip 平台）----
  if(VSTREAM_PLATFORM STREQUAL "rockchip"
     AND CMAKE_SYSTEM_PROCESSOR AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(FATAL_ERROR
      "VSTREAM_PLATFORM=rockchip 需要 aarch64 工具链，"
      "当前 CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}。\n"
      "若为缓存残留，请清理构建目录，或显式指定 --platform")
  endif()

  # ---- 是否需要解析后端依赖 ----
  # 只有 modules / tools 真正编译后端代码
  set(_vp_need_backend FALSE)
  if(VSTREAM_BUILD_MODULES OR VSTREAM_BUILD_TOOLS)
    set(_vp_need_backend TRUE)
  endif()

  if(_vp_need_backend)
    foreach(_vp_lang ${VSTREAM_PLATFORM_LANGUAGES})
      enable_language(${_vp_lang})
    endforeach()
  endif()

  # ---- 编译标准 ----
  if(VSTREAM_PLATFORM_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD ${VSTREAM_PLATFORM_CXX_STANDARD})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()
  if(VSTREAM_PLATFORM_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${VSTREAM_PLATFORM_CUDA_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  if(VSTREAM_PLATFORM_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${VSTREAM_PLATFORM_CUDA_ARCHITECTURES})
  endif()

  if(_vp_need_backend)
    # ---- find_package（CMake 内置）----
    foreach(_vp_pkg ${VSTREAM_PLATFORM_PACKAGES})
      find_package(${_vp_pkg} REQUIRED)
    endforeach()

    # ---- 自定义 Find 模块：统一输出 <PKG>_FOUND / _INCLUDE_DIRS / _LIBRARIES ----
    foreach(_vp_mod ${VSTREAM_PLATFORM_FIND_MODULES})
      include(${VSTREAM_ROOT_PATH}/cmake/Find${_vp_mod}.cmake)
      if(NOT ${_vp_mod}_FOUND)
        message(FATAL_ERROR
          "平台 ${VSTREAM_PLATFORM_ID} 依赖的 ${_vp_mod} 未找到。\n"
          "请检查安装或通过 CMAKE_PREFIX_PATH / ${_vp_mod}_ROOT_DIR 指定路径")
      endif()
      list(APPEND VSTREAM_PLATFORM_INCLUDE_DIRS ${${_vp_mod}_INCLUDE_DIRS})
      list(APPEND VSTREAM_PLATFORM_LIBS ${${_vp_mod}_LIBRARIES})

      # 把库所在目录并入打包期的依赖搜索目录 “VSTREAM_PLATFORM_DEP_SEARCH_DIRS”
      if(${_vp_mod}_LIBRARY)
        get_filename_component(_vp_libdir "${${_vp_mod}_LIBRARY}" DIRECTORY)
        if(IS_DIRECTORY "${_vp_libdir}")
          list(APPEND VSTREAM_PLATFORM_DEP_SEARCH_DIRS "${_vp_libdir}")
        endif()
      endif()
    endforeach()
  endif()

  # ---- 平台依赖的聚合目标 ----
  if(NOT TARGET vstream_platform)
    add_library(vstream_platform INTERFACE)
  endif()
  if(VSTREAM_PLATFORM_DEFINES)
    target_compile_definitions(vstream_platform INTERFACE ${VSTREAM_PLATFORM_DEFINES})
  endif()
  if(VSTREAM_PLATFORM_INCLUDE_DIRS)
    target_include_directories(vstream_platform INTERFACE ${VSTREAM_PLATFORM_INCLUDE_DIRS})
  endif()
  # 仅在确实解析过依赖时才挂库：否则 CUDA::cudart 等 imported target 不存在，
  # 会在 generate 阶段报 "target not found"（--no-modules --no-tools 场景）
  if(_vp_need_backend AND VSTREAM_PLATFORM_LIBS)
    target_link_libraries(vstream_platform INTERFACE ${VSTREAM_PLATFORM_LIBS})
  endif()

  # ---- 供 Release 编译选项等处使用的派生标志 ----
  if("CUDA" IN_LIST VSTREAM_PLATFORM_LANGUAGES)
    set(VSTREAM_PLATFORM_HAS_CUDA TRUE)
  else()
    set(VSTREAM_PLATFORM_HAS_CUDA FALSE)
  endif()
endmacro()

# -----------------------------------------------------------------------------
# 编译定义 + include（不链接库）
# 用于只编译工程源码、不需要后端库的目标（framework / python 等）
# -----------------------------------------------------------------------------
function(vstream_platform_apply _target)
  if(NOT TARGET ${_target})
    message(FATAL_ERROR "vstream_platform_apply: 目标 ${_target} 不存在")
  endif()
  if(VSTREAM_PLATFORM_DEFINES)
    target_compile_definitions(${_target} PRIVATE ${VSTREAM_PLATFORM_DEFINES})
  endif()
  if(VSTREAM_PLATFORM_INCLUDE_DIRS)
    target_include_directories(${_target} PRIVATE ${VSTREAM_PLATFORM_INCLUDE_DIRS})
  endif()
endfunction()

# -----------------------------------------------------------------------------
# 链接平台库
# -----------------------------------------------------------------------------
function(vstream_platform_link _target)
  if(NOT TARGET ${_target})
    message(FATAL_ERROR "vstream_platform_link: 目标 ${_target} 不存在")
  endif()
  if(TARGET vstream_platform)
    target_link_libraries(${_target} PRIVATE vstream_platform)
  endif()
endfunction()

# -----------------------------------------------------------------------------
# 按平台收集源码：<dir>/src/*.cpp + <dir>/src/<sub>/*.cpp|*.cu
# <sub> 遍历 VSTREAM_PLATFORM_SRC_SUBDIRS（如 cuda, rockchip）
# -----------------------------------------------------------------------------
function(vstream_platform_sources _out_var _dir)
  file(GLOB _vp_srcs "${_dir}/src/*.cpp")
  foreach(_sub ${VSTREAM_PLATFORM_SRC_SUBDIRS})
    if(IS_DIRECTORY "${_dir}/src/${_sub}")
      file(GLOB _vp_sub_srcs
        "${_dir}/src/${_sub}/*.cpp"
        "${_dir}/src/${_sub}/*.cu")
      list(APPEND _vp_srcs ${_vp_sub_srcs})
    endif()
  endforeach()
  set(${_out_var} ${_vp_srcs} PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# 导出平台变量到 build/platform-vars.cmake
# -----------------------------------------------------------------------------
macro(vstream_export_platform_vars)
  get_cmake_property(_vp_all_vars VARIABLES)
  set(_vp_out "# generated by cmake/PlatformBase.cmake at configure time\n")
  string(APPEND _vp_out "# do not edit; consumed by install(CODE) scripts\n")
  foreach(_vp_var ${_vp_all_vars})
    if(_vp_var MATCHES "^VSTREAM_PLATFORM_")
      string(REPLACE "\\" "\\\\" _vp_val "${${_vp_var}}")
      string(REPLACE "\"" "\\\"" _vp_val "${_vp_val}")
      string(APPEND _vp_out "set(${_vp_var} \"${_vp_val}\")\n")
    endif()
  endforeach()
  # 非平台变量：显式白名单，避免把整个工程变量表写进文件
  foreach(_vp_var VSTREAM_INTREE_GFLAGS_GLOG)
    if(DEFINED ${_vp_var})
      string(APPEND _vp_out "set(${_vp_var} \"${${_vp_var}}\")\n")
    endif()
  endforeach()
  file(WRITE "${CMAKE_BINARY_DIR}/platform-vars.cmake" "${_vp_out}")
  message(STATUS "Platform vars exported: ${CMAKE_BINARY_DIR}/platform-vars.cmake")
endmacro()
