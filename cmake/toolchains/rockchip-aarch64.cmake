# =============================================================================
# cmake/toolchains/rockchip-aarch64.cmake
# 瑞芯微 RK3576 (aarch64) 交叉编译工具链文件
#
# 用法（推荐）:
#   ./build.sh --rockchip
# 或手动:
#   cmake -B build-rk \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/rockchip-aarch64.cmake \
#     -DVSTREAM_USE_ROCKCHIP=ON -DVSTREAM_USE_CUDA=OFF ..
#
# 路径优先级: -D 传入 > 环境变量 > 默认值
#   RK_SYSROOT      RK SDK rootfs sysroot
#   RK_FFMPEG_ROOT  ffmpeg-rockchip 安装树（含 rkmpp 硬编解码器）
#   RK_PYTHON_ROOT  交叉编译 Python 安装树（可选，Python API 用）
#   CROSS_PREFIX    交叉工具链前缀
# =============================================================================

# ---- 路径参数 ----
if(NOT DEFINED RK_SYSROOT)
  if(DEFINED ENV{RK_SYSROOT})
    set(RK_SYSROOT "$ENV{RK_SYSROOT}")
  else()
    set(RK_SYSROOT "ubuntu/armv8a-linux")
  endif()
endif()
set(RK_SYSROOT "${RK_SYSROOT}" CACHE PATH "Rockchip SDK rootfs sysroot")

if(NOT DEFINED RK_FFMPEG_ROOT)
  if(DEFINED ENV{RK_FFMPEG_ROOT})
    set(RK_FFMPEG_ROOT "$ENV{RK_FFMPEG_ROOT}")
  else()
    set(RK_FFMPEG_ROOT "ubuntu/ffmpeg-rockchip")
  endif()
endif()
set(RK_FFMPEG_ROOT "${RK_FFMPEG_ROOT}" CACHE PATH "ffmpeg-rockchip install tree")

set(RK_PYTHON_ROOT "" CACHE PATH "Cross-compiled Python install tree (optional, for Python API)")

# ---- 目标平台声明（CMAKE_CROSSCOMPILING 由此自动置位）----
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CROSS_PREFIX "aarch64-linux-gnu-" CACHE STRING "Cross toolchain prefix")

set(CMAKE_C_COMPILER   ${CROSS_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_PREFIX}g++)
set(CMAKE_AR           ${CROSS_PREFIX}ar)
set(CMAKE_STRIP        ${CROSS_PREFIX}strip)

# ---- 搜索根：库/头文件/包只在 sysroot、ffmpeg-rockchip（及 Python）安装树内查找 ----
set(CMAKE_SYSROOT ${RK_SYSROOT})
list(APPEND CMAKE_FIND_ROOT_PATH ${RK_SYSROOT} ${RK_FFMPEG_ROOT})
if(RK_PYTHON_ROOT)
  list(APPEND CMAKE_FIND_ROOT_PATH ${RK_PYTHON_ROOT})
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ---- pkg-config 指向目标侧 ----
set(ENV{PKG_CONFIG_SYSROOT_DIR} ${RK_SYSROOT})
set(ENV{PKG_CONFIG_LIBDIR} "${RK_SYSROOT}/usr/lib/pkgconfig:${RK_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${RK_SYSROOT}/usr/share/pkgconfig")
set(ENV{PKG_CONFIG_PATH} "${RK_FFMPEG_ROOT}/lib/pkgconfig")

# ---- RPATH 相对化：build 目录产物可直接上板运行 ----
# 注：VSTREAM_PACKAGE=ON 时主 CMakeLists 会覆盖为 $ORIGIN（库同级目录），语义一致
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

message(STATUS "=== Rockchip aarch64 toolchain ===")
message(STATUS "  SYSROOT     : ${RK_SYSROOT}")
message(STATUS "  FFmpeg root : ${RK_FFMPEG_ROOT}")
message(STATUS "  Python root : ${RK_PYTHON_ROOT}")
message(STATUS "  Compiler    : ${CMAKE_CXX_COMPILER}")
