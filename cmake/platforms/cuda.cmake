# =============================================================================
# cmake/platforms/cuda.cmake
# 平台：NVIDIA CUDA + TensorRT
#
# 纯数据契约：本文件只 set 变量，不调用任何命令。
# 契约清单与消费方见 cmake_platform_refactor_plan.md 第 2.1 节。
# =============================================================================

set(VSTREAM_PLATFORM_ID    "cuda")
set(VSTREAM_PLATFORM_DESC  "NVIDIA CUDA + TensorRT")

# ---- 语言与编译 ----
set(VSTREAM_PLATFORM_LANGUAGES "CUDA")
set(VSTREAM_PLATFORM_DEFINES
  VSTREAM_USE_CUDA
  VSTREAM_PLATFORM_CUDA=1
  VSTREAM_PLATFORM_ROCKCHIP=0)
set(VSTREAM_PLATFORM_CXX_STANDARD      "17")
set(VSTREAM_PLATFORM_CUDA_STANDARD     "17")
# 默认覆盖桌面卡（86: sm_86, 89: sm_89, 120: Blackwell）。
if(DEFINED VSTREAM_CUDA_ARCHITECTURES)
  set(VSTREAM_PLATFORM_CUDA_ARCHITECTURES "${VSTREAM_CUDA_ARCHITECTURES}")
else()
  set(VSTREAM_PLATFORM_CUDA_ARCHITECTURES "86;89;120")
endif()

# ---- 源码子目录（modules/<m>/src/<sub>、proc/<sub>/、unittest/<sub>/）----
set(VSTREAM_PLATFORM_SRC_SUBDIRS  "cuda")
set(VSTREAM_PLATFORM_TOOL_SUBDIRS "trt")
set(VSTREAM_PLATFORM_TEST_SUBDIRS "cuda")

# ---- 依赖 ----
set(VSTREAM_PLATFORM_PACKAGES     "CUDAToolkit")
set(VSTREAM_PLATFORM_FIND_MODULES "TensorRT")
set(VSTREAM_PLATFORM_LIBS
  CUDA::cudart
  CUDA::nvml
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
  CUDA::nppitc)
set(VSTREAM_PLATFORM_INCLUDE_DIRS "")

# ---- 第三方路径默认值 ----
set(VSTREAM_PLATFORM_FFMPEG_ROOT "/usr/local/ffmpeg")
set(VSTREAM_PLATFORM_OPENCV_ROOT "")

# ---- 策略默认值（供生成头使用）----
set(VSTREAM_PLATFORM_DEFAULT_ENCODER "h264_nvenc")
set(VSTREAM_PLATFORM_DEFAULT_DECODER "cuda")

# ---- 打包：依赖搜索目录 / 排除规则 / 额外打包目录 ----
set(VSTREAM_PLATFORM_DEP_SEARCH_DIRS     "")
set(VSTREAM_PLATFORM_DEP_EXCLUDE_EXTRA   "")
set(VSTREAM_PLATFORM_EXTRA_BUNDLE_DIRS   "")
