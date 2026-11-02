#!/bin/bash
set -euo pipefail

# =============================================================================
# scripts/build-rk-dev.sh — RK3576 开发板交叉编译入口
#
# 封装 ./build.sh --platform rockchip，负责：
#   1. 解析/校验交叉编译环境（编译器、sysroot、FFmpeg、OpenCV、RKNN）
#   2. 把厂家 SDK 的路径转成 build.sh 认识的 RK_* 环境变量
#   3. 调用 build.sh 完成配置 + 编译（固定不含 Python 绑定）
#
# 依赖的目录布局（本机交叉编译容器实测）：
#   SDK 根          /workspace/rk3576_linux6.1_release/ubuntu
#   sysroot         <SDK>/armv8a-linux
#   OpenCV          <sysroot>/usr/lib/aarch64-linux-gnu/（含 cmake/opencv4/OpenCVConfig.cmake）
#   FFmpeg          <SDK>/ffmpeg-rockchip/build/install/usr/local/ffmpeg-rockchip
#   RKNN runtime    仓库内 3rdparty/rknpu2
#
# 用法:
#   scripts/build-rk-dev.sh                     # Release + 调试符号，跳过单测
#   scripts/build-rk-dev.sh --debug             # Debug 构建
#   scripts/build-rk-dev.sh --clean             # 先清空 build-rk 再编
#   scripts/build-rk-dev.sh --install-dir /tmp/vstream-rk
#   scripts/build-rk-dev.sh --dry-run           # 只打印将要执行的命令
#
# 若进入容器后尚未 source 厂家环境脚本，可用 --env 指定：
#   scripts/build-rk-dev.sh --env /workspace/rk3576_linux6.1_release/ubuntu/environment-setup-xxx
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ========== 路径：先取环境变量，未设置的稍后按 SDK 根推导 ==========
SDK_DIR="${RK_SDK_DIR:-/workspace/rk3576_linux6.1_release/ubuntu}"
SYSROOT="${RK_SYSROOT:-}"
FFMPEG_ROOT="${RK_FFMPEG_ROOT:-}"
OPENCV_ROOT="${RK_OPENCV_ROOT:-}"
RKNN_DIR="${PROJECT_ROOT}/3rdparty/rknpu2"
CROSS_PREFIX="${CROSS_PREFIX:-aarch64-linux-gnu-}"

# ========== 构建选项 ==========
BUILD_TYPE="Release"
RELEASE_DEBUG="ON"      # Release 保留 -g，崩溃堆栈可解析到函数名/行号
ENABLE_TESTS="OFF"      # 交叉编译产物无法在 host 运行，默认不编
ENABLE_PYTHON="0"       # 默认关闭；用 --python 开启（需先跑 build-python-rk.sh）
PYTHON_ROOT="${VSTREAM_PYTHON_ROOT_DIR:-}"
PYTHON_EXEC="${VSTREAM_PYTHON_EXECUTABLE:-}"
JOBS="$(nproc 2>/dev/null || echo 8)"
BUILD_DIR=""
CLEAN="0"
INSTALL_DIR=""
ENABLE_PACKAGE="0"
RUN_CPACK="0"
ENV_FILE=""
DRY_RUN="0"

# ========== 帮助 ==========
usage() {
    local code="${1:-0}"
    cat <<EOF
用法: $0 [选项]

构建选项:
  -t, --build-type TYPE     Debug | Release          (默认: Release)
      --debug               等价 -t Debug
      --release             等价 -t Release
      --debug-symbols       Release 保留 -g           (默认开启)
      --no-debug-symbols    Release 去掉 -g，产物更小
  -j, --jobs N              并行任务数                (默认: \$(nproc))
  -b, --build-dir DIR       构建目录                  (默认: <repo>/build-rk)
      --clean               构建前清空构建目录
      --tests               编译单元测试                (默认不编)
      --python              编译 Python 绑定             (默认关闭)
      --python-root DIR     目标侧 Python 安装树（含 include/ 与 lib/）
      --python-exec PATH    宿主机上同版本 python（构建期解释器）

路径覆盖（默认值见脚本头部）:
      --sdk DIR             SDK 根目录
      --sysroot DIR         目标 rootfs（RK_SYSROOT）
      --ffmpeg DIR          ffmpeg-rockchip 安装前缀（含 include/ 与 lib/）
      --opencv DIR          OpenCV 前缀（含 cmake/opencv4/OpenCVConfig.cmake）
      --cross-prefix P      交叉工具链前缀            (默认: aarch64-linux-gnu-)

其他:
      --env FILE            source 厂家环境脚本后再构建（未 source 时可用）
      --install-dir DIR     编译后安装到该目录（便于 scp 到板端）
      --package             启用 CPack 打包
      --cpack               打包并生成 .deb（隐含 --package）
      --dry-run             只打印解析结果与 build.sh 命令，不执行
  -h, --help                显示本帮助

说明:
  - 默认不编译 Python 绑定；需要时先跑 scripts/build-python-rk.sh 交叉编译 CPython，
    再用 --python（或设置 VSTREAM_PYTHON_ROOT_DIR / VSTREAM_PYTHON_EXECUTABLE）
  - 实际构建由 ./build.sh --platform rockchip 完成；本脚本只做环境准备与参数映射

示例:
  # 进入容器 → source 厂家环境 → 一条命令构建
  source /workspace/rk3576_linux6.1_release/ubuntu/environment-setup-xxx
  scripts/build-rk-dev.sh

  # 清空重编并安装到暂存目录，再整体 scp
  scripts/build-rk-dev.sh --clean --install-dir /tmp/vstream-rk
  scp -r /tmp/vstream-rk/* root@<board>:/usr/local/vstream/
EOF
    exit "${code}"
}

die() { echo "错误: $*" >&2; exit 1; }
warn() { echo "警告: $*" >&2; }

need_arg() {
    [[ $# -ge 2 ]] || die "选项 $1 缺少参数"
}

# ========== 参数解析 ==========
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--build-type)    need_arg "$@"; BUILD_TYPE="$2"; shift 2 ;;
        --debug)            BUILD_TYPE="Debug"; shift ;;
        --release)          BUILD_TYPE="Release"; shift ;;
        --debug-symbols)    RELEASE_DEBUG="ON"; shift ;;
        --no-debug-symbols) RELEASE_DEBUG="OFF"; shift ;;
        -j|--jobs)          need_arg "$@"; JOBS="$2"; shift 2 ;;
        -b|--build-dir)     need_arg "$@"; BUILD_DIR="$2"; shift 2 ;;
        --clean)            CLEAN="1"; shift ;;
        --tests)            ENABLE_TESTS="ON"; shift ;;
        --python)           ENABLE_PYTHON="1"; shift ;;
        --python-root)      need_arg "$@"; PYTHON_ROOT="$2"; shift 2 ;;
        --python-exec)      need_arg "$@"; PYTHON_EXEC="$2"; shift 2 ;;
        --sdk)              need_arg "$@"; SDK_DIR="$2"; shift 2 ;;
        --sysroot)          need_arg "$@"; SYSROOT="$2"; shift 2 ;;
        --ffmpeg)           need_arg "$@"; FFMPEG_ROOT="$2"; shift 2 ;;
        --opencv)           need_arg "$@"; OPENCV_ROOT="$2"; shift 2 ;;
        --cross-prefix)     need_arg "$@"; CROSS_PREFIX="$2"; shift 2 ;;
        --env)              need_arg "$@"; ENV_FILE="$2"; shift 2 ;;
        --install-dir)      need_arg "$@"; INSTALL_DIR="$2"; shift 2 ;;
        --package)          ENABLE_PACKAGE="1"; shift ;;
        --cpack)            ENABLE_PACKAGE="1"; RUN_CPACK="1"; shift ;;
        --dry-run)          DRY_RUN="1"; shift ;;
        -h|--help)          usage 0 ;;
        *)                  die "未知选项 '$1'（用 --help 查看用法）" ;;
    esac
done

case "${BUILD_TYPE}" in
    Debug|Release) ;;
    *) die "构建类型必须是 Debug 或 Release，当前: ${BUILD_TYPE}" ;;
esac

# 未显式指定的子路径跟随最终的 SDK 根推导（--sdk 生效）
: "${SYSROOT:=${SDK_DIR}/armv8a-linux}"
: "${FFMPEG_ROOT:=${SDK_DIR}/ffmpeg-rockchip/build/install/usr/local/ffmpeg-rockchip}"
: "${OPENCV_ROOT:=${SYSROOT}/usr}"
: "${BUILD_DIR:=${PROJECT_ROOT}/build-rk}"

# ========== 1. 交叉编译环境 ==========
# 优先使用 --env；否则编译器不在 PATH 时，在 SDK 根下自动探测唯一的环境脚本
if [[ -n "${ENV_FILE}" ]]; then
    [[ -f "${ENV_FILE}" ]] || die "环境脚本不存在: ${ENV_FILE}"
elif ! command -v "${CROSS_PREFIX}g++" >/dev/null 2>&1 && [[ -d "${SDK_DIR}" ]]; then
    _candidates=()
    while IFS= read -r -d '' _f; do _candidates+=("${_f}"); done \
        < <(find "${SDK_DIR}" -maxdepth 1 -type f -name 'environment*' -print0 2>/dev/null)
    if [[ ${#_candidates[@]} -eq 1 ]]; then
        ENV_FILE="${_candidates[0]}"
    fi
fi

if [[ -n "${ENV_FILE}" ]]; then
    echo "source 环境脚本: ${ENV_FILE}"
    # 厂家脚本可能引用未定义变量、ROS setup 可能返回非零；临时关闭严格模式
    set +eu
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set -eu
fi

command -v "${CROSS_PREFIX}g++" >/dev/null 2>&1 \
    || die "未找到交叉编译器 ${CROSS_PREFIX}g++。
请在容器内先 source 厂家环境脚本，或用 --env <脚本路径> 指定：
  source ${SDK_DIR}/environment-setup-xxx"

# ========== 2. 路径校验 ==========
[[ -d "${SYSROOT}" ]] || die "sysroot 不存在: ${SYSROOT}
用 --sysroot 指定，或确认 SDK 路径（当前 --sdk=${SDK_DIR}）"

if [[ ! -f "${FFMPEG_ROOT}/include/libavcodec/avcodec.h" ]]; then
    die "FFmpeg 头文件缺失: ${FFMPEG_ROOT}/include/libavcodec/avcodec.h
用 --ffmpeg 指向 ffmpeg-rockchip 的安装前缀（应含 include/ 与 lib/）"
fi
compgen -G "${FFMPEG_ROOT}/lib/libavcodec.so*" >/dev/null \
    || die "FFmpeg 库缺失: ${FFMPEG_ROOT}/lib/libavcodec.so*
交叉编译的 ffmpeg-rockchip 应已 make install 到该前缀"

[[ -f "${RKNN_DIR}/lib/librknnrt.so" ]] \
    || die "RKNN runtime 缺失: ${RKNN_DIR}/lib/librknnrt.so（应随仓库提供）"

# OpenCV 走 sysroot 的 Debian 布局；找不到只告警，最终以 CMake 的报错为准
OPENCV_CONFIG=""
for _c in \
    "${OPENCV_ROOT}/lib/aarch64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake" \
    "${OPENCV_ROOT}/usr/lib/aarch64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake" \
    "${OPENCV_ROOT}/lib/cmake/opencv4/OpenCVConfig.cmake" \
    "${OPENCV_ROOT}/usr/lib/cmake/opencv4/OpenCVConfig.cmake" \
    "${OPENCV_ROOT}/cmake/opencv4/OpenCVConfig.cmake" ; do
    if [[ -f "${_c}" ]]; then OPENCV_CONFIG="${_c}"; break; fi
done

if [[ -z "${OPENCV_CONFIG}" ]]; then
    warn "未在 ${OPENCV_ROOT} 下找到 OpenCVConfig.cmake，CMake 配置可能失败。
  已尝试:
    ${OPENCV_ROOT}/lib/aarch64-linux-gnu/cmake/opencv4/
    ${OPENCV_ROOT}/usr/lib/aarch64-linux-gnu/cmake/opencv4/
  若 OpenCV 不在 sysroot，请用 --opencv <前缀> 指定。"
fi

# ========== 2.5 Python 绑定（可选）==========
if [[ "${ENABLE_PYTHON}" == "1" ]]; then
    # 未指定时，尝试使用 build-python-rk.sh 的默认产物目录
    if [[ -z "${PYTHON_ROOT}" ]]; then
        for _p in "${PROJECT_ROOT}"/output/python-aarch64/usr/local/python3.*; do
            if [[ -d "${_p}" ]]; then PYTHON_ROOT="${_p}"; break; fi
        done
    fi
    [[ -n "${PYTHON_ROOT}" && -d "${PYTHON_ROOT}" ]] || die "未指定目标侧 Python 安装树。
先运行 scripts/build-python-rk.sh 交叉编译 CPython，或用 --python-root 指定。
（默认会查找 ${PROJECT_ROOT}/output/python-aarch64/usr/local/python3.*）"

    _pyinc=""
    for _h in "${PYTHON_ROOT}"/include/python*/Python.h; do
        if [[ -f "${_h}" ]]; then _pyinc="${_h}"; break; fi
    done
    [[ -n "${_pyinc}" ]] || die "Python 头文件缺失：${PYTHON_ROOT}/include/python*/Python.h"

    _pylib=""
    for _l in "${PYTHON_ROOT}"/lib/libpython3*.so; do
        if [[ -f "${_l}" ]]; then _pylib="${_l}"; break; fi
    done
    [[ -n "${_pylib}" ]] || die "libpython 缺失：${PYTHON_ROOT}/lib/libpython3*.so
交叉编译时请保留 --enable-shared"

    if [[ -z "${PYTHON_EXEC}" ]]; then
        for _c in python3.13 python3; do
            if command -v "${_c}" >/dev/null 2>&1; then PYTHON_EXEC="$(command -v "${_c}")"; break; fi
        done
    fi
    [[ -n "${PYTHON_EXEC}" && -x "${PYTHON_EXEC}" ]] || die "未找到宿主机构建期解释器。
用 --python-exec 指定一个与目标版本一致的 python。"
fi

# ========== 3. 构建目录缓存检查 ==========
CACHE_FILE="${BUILD_DIR}/CMakeCache.txt"
if [[ -f "${CACHE_FILE}" && "${CLEAN}" == "0" ]]; then
    _cached_platform="$(grep -E '^VSTREAM_PLATFORM:STRING=' "${CACHE_FILE}" | cut -d= -f2- || true)"
    _cached_sysroot="$(grep -E '^RK_SYSROOT:PATH=' "${CACHE_FILE}" | cut -d= -f2- || true)"
    _cached_ffmpeg="$(grep -E '^RK_FFMPEG_ROOT:PATH=' "${CACHE_FILE}" | cut -d= -f2- || true)"
    if [[ "${_cached_platform}" != "rockchip" ]]; then
        warn "构建目录 ${BUILD_DIR} 上次不是 rockchip 平台（VSTREAM_PLATFORM=${_cached_platform:-未设置}），建议加 --clean"
    elif [[ -n "${_cached_sysroot}" && "${_cached_sysroot}" != "${SYSROOT}" ]]; then
        warn "构建目录缓存的 sysroot 与本次不一致：
  缓存: ${_cached_sysroot}
  本次: ${SYSROOT}
  建议加 --clean 重新配置"
    elif [[ -n "${_cached_ffmpeg}" && "${_cached_ffmpeg}" != "${FFMPEG_ROOT}" ]]; then
        warn "构建目录缓存的 FFmpeg 前缀与本次不一致，建议加 --clean"
    fi
fi

# ========== 4. 组装 build.sh 参数 ==========
BUILD_ARGS=( --platform rockchip -t "${BUILD_TYPE}" -j "${JOBS}" )

if [[ "${ENABLE_PYTHON}" == "1" ]]; then
    BUILD_ARGS+=( --python )
else
    BUILD_ARGS+=( --no-python )
fi

if [[ "${ENABLE_TESTS}" == "ON" ]]; then
    BUILD_ARGS+=( --tests )
else
    BUILD_ARGS+=( --no-tests )
fi

if [[ "${RELEASE_DEBUG}" == "ON" ]]; then
    BUILD_ARGS+=( --release-debug )
else
    BUILD_ARGS+=( --no-release-debug )
fi

if [[ "${CLEAN}" == "1" ]]; then
    BUILD_ARGS+=( --clean )
fi

BUILD_ARGS+=( -b "${BUILD_DIR}" )

if [[ -n "${INSTALL_DIR}" ]]; then
    BUILD_ARGS+=( --install --install-prefix "${INSTALL_DIR}" )
fi

if [[ "${ENABLE_PACKAGE}" == "1" ]]; then
    BUILD_ARGS+=( --package )
fi
if [[ "${RUN_CPACK}" == "1" ]]; then
    BUILD_ARGS+=( --cpack )
fi

# build.sh 从环境变量读取这几个路径并转成 -DRK_*
export RK_SYSROOT="${SYSROOT}"
export RK_FFMPEG_ROOT="${FFMPEG_ROOT}"
export RK_OPENCV_ROOT="${OPENCV_ROOT}"
export CROSS_PREFIX

if [[ "${ENABLE_PYTHON}" == "1" ]]; then
    export RK_PYTHON_ROOT="${PYTHON_ROOT}"
    export VSTREAM_PYTHON_ROOT_DIR="${PYTHON_ROOT}"
    export VSTREAM_PYTHON_EXECUTABLE="${PYTHON_EXEC}"
fi

# ========== 5. 打印解析结果 ==========
echo "============================================"
echo "  RK3576 交叉编译配置"
echo "============================================"
printf "  %-14s: %s\n" "SDK           " "${SDK_DIR}"
printf "  %-14s: %s\n" "Sysroot       " "${SYSROOT}"
printf "  %-14s: %s\n" "FFmpeg        " "${FFMPEG_ROOT}"
printf "  %-14s: %s\n" "OpenCV root   " "${OPENCV_ROOT}"
printf "  %-14s: %s\n" "OpenCV config " "${OPENCV_CONFIG:-<未找到>}"
printf "  %-14s: %s\n" "RKNN          " "${RKNN_DIR}/lib/librknnrt.so"
printf "  %-14s: %s\n" "Toolchain     " "${CROSS_PREFIX}"
printf "  %-14s: %s\n" "Build Type    " "${BUILD_TYPE}"
printf "  %-14s: %s\n" "Debug Symbols " "${RELEASE_DEBUG}"
printf "  %-14s: %s\n" "Unit Tests    " "${ENABLE_TESTS}"
if [[ "${ENABLE_PYTHON}" == "1" ]]; then
    printf "  %-14s: %s\n" "Python Root   " "${PYTHON_ROOT}"
    printf "  %-14s: %s\n" "Python Exec   " "${PYTHON_EXEC}"
else
    printf "  %-14s: %s\n" "Python API    " "OFF"
fi
printf "  %-14s: %s\n" "Build Dir     " "${BUILD_DIR}"
echo "============================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo ""
    echo "[dry-run] 将执行:"
    echo "  RK_SYSROOT=${SYSROOT} \\"
    echo "  RK_FFMPEG_ROOT=${FFMPEG_ROOT} \\"
    echo "  RK_OPENCV_ROOT=${OPENCV_ROOT} \\"
    echo "  ${PROJECT_ROOT}/build.sh ${BUILD_ARGS[*]}"
    exit 0
fi

# ========== 6. 构建 ==========
echo ""
"${PROJECT_ROOT}/build.sh" "${BUILD_ARGS[@]}"

# ========== 7. 收尾提示 ==========
echo ""
echo "============================================"
echo "  构建完成（rockchip / ${BUILD_TYPE}）"
echo "  产物: ${PROJECT_ROOT}/lib  与  ${PROJECT_ROOT}/bin"
echo ""
echo "  打包部署（可选）:"
echo "    RK_SYSROOT=${SYSROOT} \\"
echo "    RK_FFMPEG_ROOT=${FFMPEG_ROOT} \\"
echo "    ${PROJECT_ROOT}/scripts/collect-rk-deps.sh"
echo "============================================"
