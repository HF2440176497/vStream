#!/bin/bash
set -euo pipefail

# =============================================================================
# collect-rk-deps.sh — RK(aarch64) 快速部署打包通道（tar.gz）
#
# 功能：
#   1. 收集 vStream 构建产物（lib/*.so、bin/* 可执行）
#   2. 用 aarch64-linux-gnu-readelf 递归提取 NEEDED 依赖
#   3. 从 sysroot / ffmpeg-rockchip 安装树 / 仓库内 rknpu2 复制依赖（保留软链）
#   4. 产出 package/{bin,lib}，打 tar.gz 供板端解压直接运行
#
# 用法（先完成 ./build.sh --rockchip 交叉编译）:
#   scripts/collect-rk-deps.sh
#   可用环境变量覆盖: RK_SYSROOT / RK_FFMPEG_ROOT / OUTPUT_DIR
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ========== 用户配置区 ==========
SYSROOT="${RK_SYSROOT:-${SDK_QMAKE_SYSROOT:-ubuntu/armv8a-linux}}"
FFMPEG_INSTALL_PATH="${RK_FFMPEG_ROOT:-ubuntu/ffmpeg-rockchip}"
RKNN_LIB_DIR="${PROJECT_ROOT}/3rdparty/rknpu2/lib"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/package}"
PKG_NAME="vstream-rk"

# vStream 构建产物（lib/ 由顶层 CMakeLists 统一输出）
APP_LIB_DIR="${PROJECT_ROOT}/lib"
APP_BIN_DIR="${PROJECT_ROOT}/bin"

# 目标板自带的系统库，不打包
EXCLUDE_LIBS=(
    "ld-linux-aarch64.so.1"
    "libc.so.6"
    "libpthread.so.0"
    "libm.so.6"
    "libdl.so.2"
    "librt.so.1"
    "libgcc_s.so.1"
    "libstdc++.so.6"
    "libresolv.so.2"
    "libcrypt.so.1"
    "libutil.so.1"
    "libanl.so.1"
)
# ==============================

if ! command -v aarch64-linux-gnu-readelf &>/dev/null; then
    echo "错误: 未找到 aarch64-linux-gnu-readelf，请确保交叉编译工具链已安装并在 PATH 中"
    exit 1
fi

for _root in "${SYSROOT}" "${FFMPEG_INSTALL_PATH}"; do
    if [ -z "${_root}" ] || [ ! -d "${_root}" ]; then
        echo "错误: 路径无效: ${_root}（可用 RK_SYSROOT / RK_FFMPEG_ROOT 环境变量覆盖）"
        exit 1
    fi
done
SYSROOT=$(realpath "${SYSROOT}")
echo "使用 Sysroot: ${SYSROOT}"
echo "FFmpeg root : ${FFMPEG_INSTALL_PATH}"

PKG_LIB_DIR="${OUTPUT_DIR}/lib"
PKG_BIN_DIR="${OUTPUT_DIR}/bin"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${PKG_LIB_DIR}" "${PKG_BIN_DIR}"

declare -A processed_libs

# ------------------------------------------------------------
# 递归处理一个二进制文件：提取 NEEDED，把依赖复制到 PKG_LIB_DIR
# ------------------------------------------------------------
process_binary() {
    local bin="$1"
    if [ -z "$bin" ] || [ ! -f "$bin" ]; then
        return
    fi

    local real_bin
    real_bin=$(readlink -f "$bin")

    if ! aarch64-linux-gnu-readelf -h "$real_bin" &>/dev/null; then
        return
    fi

    local needed_libs
    needed_libs=$(aarch64-linux-gnu-readelf -d "$real_bin" 2>/dev/null | \
        grep "NEEDED" | sed 's/.*\[\(.*\)\].*/\1/') || true

    for libname in $needed_libs; do
        if [[ -n "${processed_libs[$libname]:-}" ]]; then
            continue
        fi
        processed_libs["$libname"]=1

        local skip=0
        for exclude in "${EXCLUDE_LIBS[@]}"; do
            if [[ "$libname" == "$exclude" ]]; then
                skip=1
                break
            fi
        done
        if [ $skip -eq 1 ]; then
            continue
        fi

        # 已随包的 vstream 自身库（lib/ 目录）不再收集
        if [ -f "${PKG_LIB_DIR}/${libname}" ]; then
            continue
        fi

        local lib_path=""
        local search_dirs=(
            "$SYSROOT/usr/local/lib"
            "$FFMPEG_INSTALL_PATH/lib"
            "$RKNN_LIB_DIR"
            "$SYSROOT/usr/lib/aarch64-linux-gnu"
            "$SYSROOT/usr/lib"
            "$SYSROOT/lib/aarch64-linux-gnu"
            "$SYSROOT/lib"
        )
        for dir in "${search_dirs[@]}"; do
            if [ -f "$dir/$libname" ] || [ -L "$dir/$libname" ]; then
                lib_path="$dir/$libname"
                break
            fi
        done

        if [ -z "$lib_path" ]; then
            echo "  [MISSING] 未找到: $libname"
            continue
        fi

        # 安全检查：解析后的最终路径必须仍在允许的根目录内（防软链逃逸）
        local real_lib
        real_lib=$(readlink -f "$lib_path")
        local safe_root=0
        for allowed_root in "$SYSROOT" "$FFMPEG_INSTALL_PATH" "$(realpath "$RKNN_LIB_DIR")"; do
            if [[ "$real_lib" == "$allowed_root"* ]]; then
                safe_root=1
                break
            fi
        done
        if [ $safe_root -eq 0 ]; then
            echo "  [WARN] 库路径逃逸允许目录，跳过: $lib_path -> $real_lib"
            continue
        fi

        echo "  [COPY] $libname  (来自 $lib_path)"

        # cp -P 保留符号链接结构，防止运行时按具体链接名查找失败
        cp -P "$lib_path" "${PKG_LIB_DIR}/"

        if [ -L "$lib_path" ]; then
            local target
            target=$(readlink -f "$lib_path")
            if [ -f "$target" ] && [ ! -f "${PKG_LIB_DIR}/$(basename "$target")" ]; then
                cp -P "$target" "${PKG_LIB_DIR}/"
            fi
        fi

        process_binary "$real_lib"
    done
}

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
echo "开始收集 vStream RK 部署包..."

if [ ! -d "${APP_LIB_DIR}" ]; then
    echo "错误: 构建产物目录不存在: ${APP_LIB_DIR}（请先 ./build.sh --rockchip）"
    exit 1
fi

# 1. vStream 自身库
shopt -s nullglob
_vstream_libs=("${APP_LIB_DIR}"/*.so*)
if [ ${#_vstream_libs[@]} -eq 0 ]; then
    echo "错误: ${APP_LIB_DIR} 下没有 .so 产物，请先 ./build.sh --rockchip"
    exit 1
fi
for _lib in "${_vstream_libs[@]}"; do
    cp -P "$_lib" "${PKG_LIB_DIR}/"
    echo "[APP ] $(basename "$_lib")"
done

# 2. 可执行文件（测试/工具，可能为空）
_vstream_bins=("${APP_BIN_DIR}"/*)
if [ ${#_vstream_bins[@]} -gt 0 ]; then
    for _bin in "${_vstream_bins[@]}"; do
        [ -f "$_bin" ] || continue
        cp "$_bin" "${PKG_BIN_DIR}/"
        echo "[APP ] bin/$(basename "$_bin")"
    done
fi
shopt -u nullglob

# 3. 递归收集依赖
echo "开始递归收集依赖库..."
for _lib in "${PKG_LIB_DIR}"/*.so*; do
    [ -e "$_lib" ] || continue
    process_binary "$_lib"
done
if [ -d "${PKG_BIN_DIR}" ] && [ "$(ls -A "${PKG_BIN_DIR}" 2>/dev/null)" ]; then
    for _bin in "${PKG_BIN_DIR}"/*; do
        process_binary "$_bin"
    done
fi

# 4. 打 tar.gz
TARBALL="${OUTPUT_DIR}/../${PKG_NAME}.tar.gz"
(
    cd "${OUTPUT_DIR}"
    tar -czf "${TARBALL}" .
)
lib_count=$(find "${PKG_LIB_DIR}" -type f -o -type l | wc -l)
echo ""
echo "✅ 依赖收集完成！"
echo "   库文件数量: ${lib_count}"
echo "   打包目录  : ${OUTPUT_DIR}"
echo "   部署包    : ${TARBALL}"
echo ""
echo "板端运行："
echo "   tar xzf ${PKG_NAME}.tar.gz -C /opt/vstream-rk && cd /opt/vstream-rk"
echo "   export LD_LIBRARY_PATH=\$PWD/lib:\$LD_LIBRARY_PATH"
echo "   ./bin/<可执行文件>"
