# =============================================================================
# cmake/ResolvePlatform.cmake
# 平台 ID 解析 —— 全工程唯一的平台"流程控制"入口（在 project() 之前 include）
#
# 职责边界：
#   - 只做变量逻辑：解析 ID、白名单校验、定位平台文件
#   - 不加载平台文件（加载在 cmake/PlatformBase.cmake 的 vstream_platform_load()，
#     即 project() 之后 —— 平台文件可能引用工具链设置的 RK_* 变量）
#   - 不做 find_package / enable_language / add_compile_definitions
#     （那些在 cmake/PlatformBase.cmake 的 configure 阶段执行）
#
# 平台 ID：cuda | rockchip | cpu
# 来源优先级：-D / cache 的 VSTREAM_PLATFORM  >  host 默认（cuda）
# 交叉编译（指定了 CMAKE_TOOLCHAIN_FILE）时必须显式传入平台 ID。
# =============================================================================


# ---- 解析平台 ID ----
if(VSTREAM_PLATFORM)
  # 已由 -D 或上次 configure 的 cache 指定
elseif(CMAKE_TOOLCHAIN_FILE)
  message(FATAL_ERROR
    "指定了 CMAKE_TOOLCHAIN_FILE 但未指定平台，无法推断目标后端。\n"
    "请显式传入: -DVSTREAM_PLATFORM=<cuda|rockchip|cpu>")
else()
  set(VSTREAM_PLATFORM "cuda")
endif()

# ---- 白名单 + 平台文件存在性 ----
set(_vs_known_platforms cuda rockchip cpu)
if(NOT VSTREAM_PLATFORM IN_LIST _vs_known_platforms)
  message(FATAL_ERROR
    "未知平台 '${VSTREAM_PLATFORM}'。受支持的平台：${_vs_known_platforms}\n"
    "对应文件：cmake/platforms/<id>.cmake")
endif()

set(_vs_platform_file "${CMAKE_CURRENT_LIST_DIR}/platforms/${VSTREAM_PLATFORM}.cmake")
if(NOT EXISTS "${_vs_platform_file}")
  message(FATAL_ERROR "平台文件不存在：${_vs_platform_file}")
endif()

# ---- 固化平台 ID，保证同一构建目录内多次 configure 一致 ----
set(VSTREAM_PLATFORM "${VSTREAM_PLATFORM}" CACHE STRING "vStream target platform id" FORCE)

# ---- 记录平台文件路径，此处只校验存在性，不加载 ----
# 加载推迟到 project() 之后：CMAKE_TOOLCHAIN_FILE 在 project() 时才被读取，
# 平台文件中的 RK_SYSROOT / RK_FFMPEG_ROOT / RK_OPENCV_ROOT 等来自工具链，
# 若在 project() 之前 include，首轮 configure 会全部取到空值。
set(VSTREAM_PLATFORM_FILE "${_vs_platform_file}"
    CACHE INTERNAL "Resolved platform data file (loaded after project())")
