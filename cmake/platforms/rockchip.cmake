# =============================================================================
# cmake/platforms/rockchip.cmake
# 平台：瑞芯微 RKMPP（硬解硬编）+ RKNN（NPU 推理）
#
# 纯数据契约：本文件只 set 变量，不调用任何命令。
# 路径默认值来自 toolchain 暴露的 RK_* 变量（-D 或环境变量覆盖）。
# =============================================================================

set(VSTREAM_PLATFORM_ID    "rockchip")
set(VSTREAM_PLATFORM_DESC  "Rockchip RKMPP + RKNN")

# ---- 语言与编译 ----
# 交叉编译不启用 CUDA 语言
set(VSTREAM_PLATFORM_LANGUAGES "")
set(VSTREAM_PLATFORM_DEFINES
  VSTREAM_USE_ROCKCHIP
  VSTREAM_PLATFORM_CUDA=0
  VSTREAM_PLATFORM_ROCKCHIP=1)
set(VSTREAM_PLATFORM_CXX_STANDARD      "17")
set(VSTREAM_PLATFORM_CUDA_STANDARD     "")
set(VSTREAM_PLATFORM_CUDA_ARCHITECTURES "")

# ---- 源码子目录 ----
set(VSTREAM_PLATFORM_SRC_SUBDIRS  "rockchip")
set(VSTREAM_PLATFORM_TOOL_SUBDIRS "")
set(VSTREAM_PLATFORM_TEST_SUBDIRS "")

# ---- 依赖 ----
set(VSTREAM_PLATFORM_PACKAGES     "")
set(VSTREAM_PLATFORM_FIND_MODULES "RKNN")
set(VSTREAM_PLATFORM_LIBS         "")
set(VSTREAM_PLATFORM_INCLUDE_DIRS "")

# ---- 第三方路径默认值（SDK 提供的预编译树）----
set(VSTREAM_PLATFORM_FFMPEG_ROOT "${RK_FFMPEG_ROOT}")
set(VSTREAM_PLATFORM_OPENCV_ROOT "${RK_OPENCV_ROOT}")

# ---- 策略默认值 ----
set(VSTREAM_PLATFORM_DEFAULT_ENCODER "h264_rkmpp")
set(VSTREAM_PLATFORM_DEFAULT_DECODER "rkmpp")

# ---- 打包：目标侧库搜索目录（不在构建机默认搜索路径中，必须显式补充）----
set(VSTREAM_PLATFORM_DEP_SEARCH_DIRS
  "${RK_FFMPEG_ROOT}/lib"
  "${RK_FFMPEG_ROOT}/lib/aarch64-linux-gnu"
  "${RK_OPENCV_ROOT}/lib"
  "${RK_OPENCV_ROOT}/lib/aarch64-linux-gnu"
  "${RK_SYSROOT}/lib/aarch64-linux-gnu"
  "${RK_SYSROOT}/usr/lib"
  "${RK_SYSROOT}/usr/lib/aarch64-linux-gnu")

# librknnrt.so 已由 BundleThirdPartyDeps 显式打包，按名排除防重复
set(VSTREAM_PLATFORM_DEP_EXCLUDE_EXTRA "^librknnrt\\.so")

# rknpu2 运行库随包分发；目录在 configure 阶段由 FindRKNN 的 RKNN_LIBRARY 反推，
# 这里留空表示"无额外硬编码目录"
set(VSTREAM_PLATFORM_EXTRA_BUNDLE_DIRS "")
