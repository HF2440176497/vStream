#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# ---------- 默认值 ----------
BUILD_TYPE="Debug"
PLATFORM="cuda"
BUILD_DIR=""            # 空 = 按平台取默认目录
BUILD_DIR_GIVEN="0"     # 用户是否显式指定 -b
ENABLE_TESTS="ON"
ENABLE_PYTHON_API="ON"
ENABLE_TOOLS="ON"
JOBS=$(nproc 2>/dev/null || echo 4)
CLEAN_BUILD="OFF"
RELEASE_DEBUG_INFO="OFF"

# 打包相关默认值
ENABLE_PACKAGE="OFF"
ENABLE_BUNDLE_DEPS="OFF"
INSTALL_PREFIX="/usr/local/vstream"
RUN_CPACK="OFF"
RUN_INSTALL="OFF"

KNOWN_PLATFORMS="cuda rockchip cpu"

# ---------- 帮助信息 ----------
usage() {
    local exit_code="${1:-0}"
    cat <<EOF
用法: $0 [选项]

平台:
  --platform ID                     目标平台 (默认: cuda)
                                      cuda     NVIDIA CUDA + TensorRT（host 或 Jetson）
                                      rockchip 瑞芯微 RKMPP + RKNN（aarch64 交叉编译）
                                      cpu      无加速后端，纯 CPU 路径

                                     rockchip 会自动启用 cmake/toolchains/rockchip-aarch64.cmake
                                     并默认使用 build-rk/ 目录。可用环境变量覆盖工具链路径:
                                     RK_SYSROOT / RK_FFMPEG_ROOT / RK_OPENCV_ROOT / RK_PYTHON_ROOT / CROSS_PREFIX

构建:
  -t, --build-type TYPE             构建类型: Debug 或 Release (默认: Debug)
  -b, --build-dir DIR               CMake 构建目录
                                     默认按平台取: cuda -> build/，rockchip -> build-rk/，cpu -> build-cpu/
  -j, --jobs N                      并行编译任务数 (默认: \$(nproc))
  --clean                           构建前先清理构建目录

  --release-debug / --no-release-debug
                                    Release 构建是否保留调试符号(-g) (默认: --no-release-debug)
                                    开启后崩溃堆栈(backward-cpp + libdw)可解析到
                                    函数名/文件/行号，产物体积会增大

组件:
  --tests / --no-tests              启用/禁用单元测试 (默认: --tests)
  --python / --no-python            启用/禁用 Python API (默认: --python)
  --tools / --no-tools              启用/禁用工具构建 (默认: --tools)

打包:
  --package / --no-package          启用/禁用 CPack DEB 打包 (默认: --no-package)
  --bundle-deps / --no-bundle-deps  自包含模式: 把第三方库打包到 dev/ (默认: --no-bundle-deps)
  --install-prefix PREFIX           安装前缀 (默认: /usr/local/vstream)
  --cpack                           配置/编译完成后执行 cpack -G DEB 生成 .deb
  --install                         配置/编译完成后执行 cmake --install 到指定前缀

  -h, --help                        显示此帮助信息

示例:
  # Debug 构建 (默认平台 cuda)
  $0

  # Release 构建，禁用测试和 Python API
  $0 -t Release --no-tests --no-python

  # 纯 CPU 平台（无需 CUDA）
  $0 --platform cpu -t Release

  # 指定构建目录和并行任务数
  $0 -t Release -b build_release -j 8 --no-tests

  # 清理后重新构建
  $0 --clean -t Release

  # Release 构建并保留调试符号
  $0 -t Release --release-debug --no-tests

  # 精简 DEB 打包 (Release, 不含第三方依赖, 由基础镜像提供)
  $0 -t Release --package --no-bundle-deps --cpack

  # 自包含 DEB 打包 (Release)
  $0 -t Release --package --bundle-deps --tools --cpack

  # RK3576 交叉编译（需先 source RK SDK 环境）
  $0 --platform rockchip -t Release --package --bundle-deps --cpack

  # RK 交叉编译 + Python API（需先交叉编译好 Python）
  RK_PYTHON_ROOT=/workspace/python-aarch64/python3.13 $0 --platform rockchip -t Release --python

  # 只安装到本地前缀, 不打包
  $0 -t Release --install --install-prefix /opt/vstream
EOF
    exit "${exit_code}"
}

# ---------- 小工具 ----------
die() {
    echo "错误: $*" >&2
    exit 2
}

# 校验"需要参数"的选项：$1 为选项名，$@ 为当前剩余参数
require_arg() {
    local opt="$1"
    if [[ $# -lt 2 ]]; then
        die "选项 ${opt} 缺少参数"
    fi
}

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            require_arg "$@"
            case "$2" in
                cuda|rockchip|cpu) PLATFORM="$2" ;;
                *) die "未知平台 '$2'（受支持: ${KNOWN_PLATFORMS// / | }）" ;;
            esac
            shift 2
            ;;
        -t|--build-type)
            require_arg "$@"
            BUILD_TYPE="$2"
            shift 2
            ;;
        -b|--build-dir)
            require_arg "$@"
            BUILD_DIR="$2"
            BUILD_DIR_GIVEN="1"
            shift 2
            ;;
        -j|--jobs)
            require_arg "$@"
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        --release-debug)
            RELEASE_DEBUG_INFO="ON"
            shift
            ;;
        --no-release-debug)
            RELEASE_DEBUG_INFO="OFF"
            shift
            ;;
        --tests)
            ENABLE_TESTS="ON"
            shift
            ;;
        --no-tests)
            ENABLE_TESTS="OFF"
            shift
            ;;
        --python)
            ENABLE_PYTHON_API="ON"
            shift
            ;;
        --no-python)
            ENABLE_PYTHON_API="OFF"
            shift
            ;;
        --tools)
            ENABLE_TOOLS="ON"
            shift
            ;;
        --no-tools)
            ENABLE_TOOLS="OFF"
            shift
            ;;
        --package)
            ENABLE_PACKAGE="ON"
            shift
            ;;
        --no-package)
            ENABLE_PACKAGE="OFF"
            shift
            ;;
        --bundle-deps)
            ENABLE_BUNDLE_DEPS="ON"
            shift
            ;;
        --no-bundle-deps)
            ENABLE_BUNDLE_DEPS="OFF"
            shift
            ;;
        --install-prefix)
            require_arg "$@"
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --cpack)
            RUN_CPACK="ON"
            shift
            ;;
        --install)
            RUN_INSTALL="ON"
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        # 已移除的旧参数：给出明确的替代写法，避免用户猜
        --cuda)
            die "选项 --cuda 已移除，请改用 --platform cuda"
            ;;
        --no-cuda)
            die "选项 --no-cuda 已移除，请改用 --platform cpu"
            ;;
        --rockchip)
            die "选项 --rockchip 已移除，请改用 --platform rockchip"
            ;;
        *)
            die "未知选项 '$1'（用 --help 查看用法）"
            ;;
    esac
done

# ---------- 构建目录：按平台取默认值 ----------
if [[ "${BUILD_DIR_GIVEN}" == "0" ]]; then
    case "${PLATFORM}" in
        cuda)     BUILD_DIR="${PROJECT_ROOT}/build" ;;
        rockchip) BUILD_DIR="${PROJECT_ROOT}/build-rk" ;;
        cpu)      BUILD_DIR="${PROJECT_ROOT}/build-cpu" ;;
    esac
fi

# 相对路径转绝对路径
BUILD_DIR="$(cd "$(dirname "${BUILD_DIR}")" 2>/dev/null && pwd)/$(basename "${BUILD_DIR}")" || {
    BUILD_DIR="$(cd "${PROJECT_ROOT}" && pwd)/$(basename "${BUILD_DIR}")"
}

# ---------- 打印配置 ----------
echo "============================================"
echo "  Build Configuration"
echo "============================================"
printf "  %-14s: %s\n" "Project Root  " "${PROJECT_ROOT}"
printf "  %-14s: %s\n" "Platform      " "${PLATFORM}"
printf "  %-14s: %s\n" "Build Dir     " "${BUILD_DIR}"
printf "  %-14s: %s\n" "Build Type    " "${BUILD_TYPE}"
printf "  %-14s: %s\n" "Release Debug " "${RELEASE_DEBUG_INFO}"
printf "  %-14s: %s\n" "Parallel Jobs " "${JOBS}"
printf "  %-14s: %s\n" "Clean Build   " "${CLEAN_BUILD}"
printf "  %-14s: %s\n" "Unit Tests    " "${ENABLE_TESTS}"
printf "  %-14s: %s\n" "Python API    " "${ENABLE_PYTHON_API}"
printf "  %-14s: %s\n" "Build Tools   " "${ENABLE_TOOLS}"
printf "  %-14s: %s\n" "CPack Package " "${ENABLE_PACKAGE}"
printf "  %-14s: %s\n" "Bundle Deps   " "${ENABLE_BUNDLE_DEPS}"
printf "  %-14s: %s\n" "Install Prefix" "${INSTALL_PREFIX}"
printf "  %-14s: %s\n" "Run cpack     " "${RUN_CPACK}"
printf "  %-14s: %s\n" "Run install   " "${RUN_INSTALL}"
echo "============================================"

# ---------- 清理 ----------
if [[ "${CLEAN_BUILD}" == "ON" ]] && [[ -d "${BUILD_DIR}" ]]; then
    echo "清理构建目录: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

# ---------- CMake 配置 ----------
mkdir -p "${BUILD_DIR}"

CMAKE_OPTIONS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    -DVSTREAM_RELEASE_WITH_DEBUG_INFO="${RELEASE_DEBUG_INFO}"
    -DVSTREAM_PLATFORM="${PLATFORM}"
    -DVSTREAM_BUILD_TESTS="${ENABLE_TESTS}"
    -DVSTREAM_BUILD_TOOLS="${ENABLE_TOOLS}"
    -DVSTREAM_BUILD_PYTHON_API="${ENABLE_PYTHON_API}"
    -DVSTREAM_BUILD_PYBIND11="${ENABLE_PYTHON_API}"
    -DVSTREAM_PACKAGE="${ENABLE_PACKAGE}"
    -DVSTREAM_PACKAGE_BUNDLE_DEPS="${ENABLE_BUNDLE_DEPS}"
    -DVSTREAM_PACKAGE_INCLUDE_TOOLS="${ENABLE_TOOLS}"
    -DVSTREAM_PACKAGE_INCLUDE_TESTS="${ENABLE_TESTS}"
)

# Python 路径透传：
# - 未设置时使用 CMakeLists.txt 中的默认值（默认指向 /usr/local/python3.13）
# - 交叉编译时：VSTREAM_PYTHON_ROOT_DIR 应当指向 aarch64 安装树；
#   VSTREAM_PYTHON_EXECUTABLE 应当指向构建机上（x86）同版本的 Python
[[ -n "${VSTREAM_PYTHON_ROOT_DIR:-}" ]]   && CMAKE_OPTIONS+=(-DVSTREAM_PYTHON_ROOT_DIR="${VSTREAM_PYTHON_ROOT_DIR}")
[[ -n "${VSTREAM_PYTHON_EXECUTABLE:-}" ]] && CMAKE_OPTIONS+=(-DVSTREAM_PYTHON_EXECUTABLE="${VSTREAM_PYTHON_EXECUTABLE}")

# rockchip：交叉工具链 + 路径透传
# 平台 ID 已由 -DVSTREAM_PLATFORM 传入，toolchain 只负责编译器/sysroot/find_root
if [[ "${PLATFORM}" == "rockchip" ]]; then
    CMAKE_OPTIONS+=(
        -DCMAKE_TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/toolchains/rockchip-aarch64.cmake"
    )
    [[ -n "${RK_SYSROOT:-}" ]]     && CMAKE_OPTIONS+=(-DRK_SYSROOT="${RK_SYSROOT}")
    [[ -n "${RK_FFMPEG_ROOT:-}" ]] && CMAKE_OPTIONS+=(-DRK_FFMPEG_ROOT="${RK_FFMPEG_ROOT}")
    [[ -n "${RK_OPENCV_ROOT:-}" ]] && CMAKE_OPTIONS+=(-DRK_OPENCV_ROOT="${RK_OPENCV_ROOT}")
    [[ -n "${RK_PYTHON_ROOT:-}" ]] && CMAKE_OPTIONS+=(-DRK_PYTHON_ROOT="${RK_PYTHON_ROOT}")
    [[ -n "${CROSS_PREFIX:-}" ]]   && CMAKE_OPTIONS+=(-DCROSS_PREFIX="${CROSS_PREFIX}")
    true  # 保证 if 块返回 0（set -e 下条件行为空时短路返回非零）
fi

echo ""
echo "运行 CMake 配置..."
echo "cmake -S ${PROJECT_ROOT} -B ${BUILD_DIR} ${CMAKE_OPTIONS[*]}"

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" "${CMAKE_OPTIONS[@]}"

echo ""
echo "CMake 配置完成，开始编译..."

# ---------- 编译 ----------
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${JOBS}"

echo ""
echo "============================================"
echo "  编译完成"
echo "  平台    : ${PLATFORM}"
echo "  构建目录: ${BUILD_DIR}"
echo "============================================"

# ---------- 安装（可选） ----------
if [[ "${RUN_INSTALL}" == "ON" ]]; then
    echo ""
    echo "开始安装到前缀: ${INSTALL_PREFIX} ..."
    cmake --install "${BUILD_DIR}"
    echo "安装完成"
fi

# ---------- 打包（可选） ----------
if [[ "${RUN_CPACK}" == "ON" ]]; then
    echo ""
    echo "开始运行 CPack 生成 DEB 包..."
    (cd "${BUILD_DIR}" && cpack)
    echo ""
    echo "DEB 包生成完成，位置:"
    ls -lh "${BUILD_DIR}"/*.deb 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  全部完成"
echo "  平台    : ${PLATFORM}"
echo "  构建目录: ${BUILD_DIR}"
if [[ "${RUN_CPACK}" == "ON" ]]; then
    echo "  DEB 包  : ${BUILD_DIR}/*.deb"
fi
if [[ "${RUN_INSTALL}" == "ON" ]]; then
    echo "  安装前缀: ${INSTALL_PREFIX}"
fi
echo "============================================"
