# =============================================================================
# cmake/platforms/cpu.cmake
#
# =============================================================================

set(VSTREAM_PLATFORM_ID    "cpu")
set(VSTREAM_PLATFORM_DESC  "CPU only (no accelerator backend)")

# ---- 语言与编译 ----
set(VSTREAM_PLATFORM_LANGUAGES "")
set(VSTREAM_PLATFORM_DEFINES
  VSTREAM_PLATFORM_CUDA=0
  VSTREAM_PLATFORM_ROCKCHIP=0)
set(VSTREAM_PLATFORM_CXX_STANDARD      "17")
set(VSTREAM_PLATFORM_CUDA_STANDARD     "")
set(VSTREAM_PLATFORM_CUDA_ARCHITECTURES "")

# ---- 源码子目录：无平台专属源码 ----
set(VSTREAM_PLATFORM_SRC_SUBDIRS  "")
set(VSTREAM_PLATFORM_TOOL_SUBDIRS "")
set(VSTREAM_PLATFORM_TEST_SUBDIRS "")

# ---- 依赖：无后端依赖 ----
set(VSTREAM_PLATFORM_PACKAGES     "")
set(VSTREAM_PLATFORM_FIND_MODULES "")
set(VSTREAM_PLATFORM_LIBS         "")
set(VSTREAM_PLATFORM_INCLUDE_DIRS "")

# ---- 第三方路径默认值 ----
set(VSTREAM_PLATFORM_FFMPEG_ROOT "/usr/local/ffmpeg")
set(VSTREAM_PLATFORM_OPENCV_ROOT "")

# ---- 策略默认值 ----
set(VSTREAM_PLATFORM_DEFAULT_ENCODER "")
set(VSTREAM_PLATFORM_DEFAULT_DECODER "cpu")

# ---- 打包 ----
set(VSTREAM_PLATFORM_DEP_SEARCH_DIRS   "")
set(VSTREAM_PLATFORM_DEP_EXCLUDE_EXTRA "")
set(VSTREAM_PLATFORM_EXTRA_BUNDLE_DIRS "")
