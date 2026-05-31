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

# ---------- 帮助信息 ----------
usage() {
    cat <<EOF
用法: $0 [选项]

选项:
  -t, --build-type TYPE      构建类型: Debug 或 Release (默认: Debug)
  -b, --build-dir DIR        CMake 构建目录 (默认: \${PROJECT_ROOT}/build)
  -j, --jobs N               并行编译任务数 (默认: \$(nproc))
  --clean                    构建前先清理构建目录

  --cuda / --no-cuda         启用/禁用 CUDA 支持 (默认: --cuda)
  --tests / --no-tests       启用/禁用单元测试 (默认: --tests)
  --python / --no-python     启用/禁用 Python API (默认: --python)
  --tools / --no-tools       启用/禁用工具构建 (默认: --tools)

  -h, --help                 显示此帮助信息

示例:
  # Debug 构建 (默认)
  $0

  # Release 构建，禁用测试和 Python API
  $0 -t Release --no-tests --no-python

  # Release 构建，指定构建目录和并行任务数
  $0 -t Release -b build_release -j 8 --no-tests

  # 清理后重新构建
  $0 --clean -t Release
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
echo "  vStream 构建配置"
echo "============================================"
echo "  项目根目录     : ${PROJECT_ROOT}"
echo "  构建目录       : ${BUILD_DIR}"
echo "  构建类型       : ${BUILD_TYPE}"
echo "  并行任务数     : ${JOBS}"
echo "  清理构建       : ${CLEAN_BUILD}"
echo "  CUDA 支持      : ${ENABLE_CUDA}"
echo "  单元测试       : ${ENABLE_TESTS}"
echo "  Python API     : ${ENABLE_PYTHON_API}"
echo "  工具构建       : ${ENABLE_TOOLS}"
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
    -DVSTREAM_USE_CUDA="${ENABLE_CUDA}"
    -DVSTREAM_BUILD_TESTS="${ENABLE_TESTS}"
    -DVSTREAM_BUILD_PYTHON_API="${ENABLE_PYTHON_API}"
    -DVSTREAM_BUILD_TOOLS="${ENABLE_TOOLS}"
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
echo "  构建成功完成!"
echo "  构建目录: ${BUILD_DIR}"
echo "============================================"