# =============================================================================
# cmake/ResolvePlatform.cmake
# 平台 ID 解析 —— 全工程唯一的平台"流程控制"入口（在 project() 之前 include）
#
# 职责边界：
#   - 只做变量逻辑：解析 ID、拒绝废弃参数、白名单校验、加载平台文件
#   - 不做 find_package / enable_language / add_compile_definitions
#     （那些在 cmake/PlatformBase.cmake 的 configure 阶段执行）
#
# 平台 ID：cuda | rockchip | cpu
# 来源优先级：-D / cache 的 VSTREAM_PLATFORM  >  host 默认（cuda）
# 交叉编译（指定了 CMAKE_TOOLCHAIN_FILE）时必须显式传入平台 ID。
# =============================================================================

# ---- 1) 拒绝已移除的旧参数 ----
# 不再做"旧参数映射"：静默兼容会让忘记改脚本的 CI 用 -DVSTREAM_USE_ROCKCHIP=ON
# 悄悄编出 cuda 产物。这里选择响亮失败。
if(DEFINED VSTREAM_USE_CUDA OR DEFINED VSTREAM_USE_ROCKCHIP)
  message(FATAL_ERROR
    "VSTREAM_USE_CUDA / VSTREAM_USE_ROCKCHIP 已移除，请改用单一平台 ID：\n"
    "  -DVSTREAM_PLATFORM=<cuda|rockchip|cpu>\n"
    "若该变量来自旧构建目录的缓存残留，请删除构建目录后重新配置。")
endif()

# ---- 2) 解析平台 ID ----
if(VSTREAM_PLATFORM)
  # 已由 -D 或上次 configure 的 cache 指定
elseif(CMAKE_TOOLCHAIN_FILE)
  message(FATAL_ERROR
    "指定了 CMAKE_TOOLCHAIN_FILE 但未指定平台，无法推断目标后端。\n"
    "请显式传入: -DVSTREAM_PLATFORM=<cuda|rockchip|cpu>")
else()
  set(VSTREAM_PLATFORM "cuda")
endif()

# ---- 3) 白名单 + 平台文件存在性 ----
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

# ---- 4) 固化平台 ID，保证同一构建目录内多次 configure 一致 ----
set(VSTREAM_PLATFORM "${VSTREAM_PLATFORM}" CACHE STRING "vStream target platform id" FORCE)

# ---- 5) 加载平台文件（纯数据：只 set 变量，无任何副作用）----
include("${_vs_platform_file}")

message(STATUS "Platform: ${VSTREAM_PLATFORM_ID} (${VSTREAM_PLATFORM_DESC})")
