#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# ---------- 默认值 ----------
BUILD_TYPE="Debug"
BUILD_DIR="${PROJECT_ROOT}/build"
ENABLE_CUDA="ON"
ENABLE_TESTS="ON"
ENABLE_PYTHON_API="ON"
ENABLE_TOOLS="ON"
JOBS=$(nproc 2>/dev/null || echo 4)
CLEAN_BUILD="OFF"

# 打包相关默认值
ENABLE_PACKAGE="OFF"
ENABLE_BUNDLE_DEPS="OFF"
INSTALL_PREFIX="/usr/local/vstream"
RUN_CPACK="OFF"
RUN_INSTALL="OFF"

# ---------- 帮助信息 ----------
usage() {
    cat <<EOF
用法: $0 [选项]

选项:
  -t, --build-type TYPE             构建类型: Debug 或 Release (默认: Debug)
  -b, --build-dir DIR               CMake 构建目录 (默认: \${PROJECT_ROOT}/build)
  -j, --jobs N                      并行编译任务数 (默认: \$(nproc))
  --clean                           构建前先清理构建目录

  --cuda / --no-cuda                启用/禁用 CUDA 支持 (默认: --cuda)
  --tests / --no-tests              启用/禁用单元测试 (默认: --tests)
  --python / --no-python            启用/禁用 Python API (默认: --python)
  --tools / --no-tools              启用/禁用工具构建 (默认: --tools)

  --package / --no-package          启用/禁用 CPack DEB 打包 (默认: --no-package)
  --bundle-deps / --no-bundle-deps  自包含模式: 把第三方库打包到 dev/ (默认: --no-bundle-deps)
  --install-prefix PREFIX           安装前缀 (默认: /usr/local/vstream)
  --cpack                           配置/编译完成后执行 cpack -G DEB 生成 .deb
  --install                         配置/编译完成后执行 cmake --install 到指定前缀

  -h, --help                        显示此帮助信息

示例:
  # Debug 构建 (默认)
  $0

  # Release 构建，禁用测试和 Python API
  $0 -t Release --no-tests --no-python

  # Release 构建，指定构建目录和并行任务数
  $0 -t Release -b build_release -j 8 --no-tests

  # 清理后重新构建
  $0 --clean -t Release

  # 自包含 DEB 打包 (Release, 含工具, 不含测试)
  $0 -t Release --package --bundle-deps --tools --cpack

  # 精简 DEB 打包 (Release, 不含第三方依赖, 由基础镜像提供)
  $0 -t Release --package --no-bundle-deps --cpack

  # 只安装到本地前缀, 不打包
  $0 -t Release --install --install-prefix /opt/vstream
EOF
    exit 0
}

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        --cuda)
            ENABLE_CUDA="ON"
            shift
            ;;
        --no-cuda)
            ENABLE_CUDA="OFF"
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
            usage
            ;;
        *)
            echo "未知选项: $1"
            usage
            ;;
    esac
done

# 将相对路径的 BUILD_DIR 转为绝对路径
BUILD_DIR="$(cd "$(dirname "${BUILD_DIR}")" 2>/dev/null && pwd)/$(basename "${BUILD_DIR}")" || {
    BUILD_DIR="$(cd "${PROJECT_ROOT}" && pwd)/$(basename "${BUILD_DIR}")"
}

# ---------- 打印配置 ----------
echo "============================================"
echo "  Build Configuration"
echo "============================================"
printf "  %-14s: %s\n" "Project Root  " "${PROJECT_ROOT}"
printf "  %-14s: %s\n" "Build Dir     " "${BUILD_DIR}"
printf "  %-14s: %s\n" "Build Type    " "${BUILD_TYPE}"
printf "  %-14s: %s\n" "Parallel Jobs " "${JOBS}"
printf "  %-14s: %s\n" "Clean Build   " "${CLEAN_BUILD}"
printf "  %-14s: %s\n" "CUDA Support  " "${ENABLE_CUDA}"
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
    -DVSTREAM_USE_CUDA="${ENABLE_CUDA}"
    -DVSTREAM_BUILD_TESTS="${ENABLE_TESTS}"
    -DVSTREAM_BUILD_TOOLS="${ENABLE_TOOLS}"
    -DVSTREAM_PACKAGE="${ENABLE_PACKAGE}"
    -DVSTREAM_PACKAGE_BUNDLE_DEPS="${ENABLE_BUNDLE_DEPS}"
    -DVSTREAM_PACKAGE_INCLUDE_TOOLS="${ENABLE_TOOLS}"
    -DVSTREAM_PACKAGE_INCLUDE_TESTS="${ENABLE_TESTS}"
)

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
echo "  编译成功完成!"
echo "  构建目录: ${BUILD_DIR}"
echo "============================================"

# ---------- 安装（可选） ----------
if [[ "${RUN_INSTALL}" == "ON" ]]; then
    echo ""
    echo "开始安装到前缀: ${INSTALL_PREFIX} ..."
    cmake --install "${BUILD_DIR}"
    echo "安装完成。"
fi

# ---------- 打包（可选） ----------
if [[ "${RUN_CPACK}" == "ON" ]]; then
    echo ""
    echo "开始运行 CPack 生成 DEB 包..."
    (cd "${BUILD_DIR}" && cpack -G DEB -V)
    echo ""
    echo "DEB 包生成完成，位置:"
    ls -lh "${BUILD_DIR}"/*.deb 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  全部完成"
echo "  构建目录: ${BUILD_DIR}"
if [[ "${RUN_CPACK}" == "ON" ]]; then
    echo "  DEB 包  : ${BUILD_DIR}/*.deb"
fi
if [[ "${RUN_INSTALL}" == "ON" ]]; then
    echo "  安装前缀: ${INSTALL_PREFIX}"
fi
echo "============================================"