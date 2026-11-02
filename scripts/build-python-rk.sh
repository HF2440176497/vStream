#!/bin/bash
set -euo pipefail

# =============================================================================
# scripts/build-python-rk.sh — 为 RK3576(aarch64) 交叉编译 CPython
#
# 与 x86 手动编译的差异（都是交叉编译的硬约束，不是选择）：
#   1. 去掉 --enable-optimizations
#      PGO 需要「运行」刚编出来的解释器来收集 profile；交叉产物是 aarch64，
#      在 x86 容器里跑不了（除非配 qemu-user 模拟，代价大且 profile 不准）。
#   2. 加 --host / --build
#      让 autoconf 进入交叉模式，否则它会用宿主编译器并试图执行目标程序。
#   3. 加 --with-build-python=<宿主同版本 python>
#      Python 3.11+ 在构建期需要运行解释器来生成冻结模块（freeze_modules），
#      交叉构建必须提供一个能在宿主机跑的同版本 python。
#   4. 用 --sysroot 指向厂家 rootfs
#      容器里的 aarch64-linux-gnu-gcc 默认 sysroot 是 /usr/aarch64-linux-gnu，
#      不是 SDK 的 rootfs；不加 --sysroot 会链到宿主的 glibc。
#   5. 保留 --enable-shared
#      pybind11 的 Development.Embed 需要 libpythonX.Y.so。
#
# 用法:
#   scripts/build-python-rk.sh --download              # 下载并编译（需联网）
#   scripts/build-python-rk.sh --tarball Python-3.13.7.tar.xz
#   scripts/build-python-rk.sh --srcdir /path/Python-3.13.7
#   scripts/build-python-rk.sh --dry-run               # 只看将执行的配置
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ========== 默认值 ==========
PYVER="3.13.7"
PYMAJOR=""
PREFIX=""                       # 默认 /usr/local/python<major>
DESTDIR="${PROJECT_ROOT}/output/python-aarch64"
WORK_DIR="${PROJECT_ROOT}/build-python"
SRC_DIR=""
TARBALL=""
DO_DOWNLOAD="0"
BUILD_PYTHON=""
JOBS="$(nproc 2>/dev/null || echo 8)"
CLEAN="0"
DRY_RUN="0"

SDK_DIR="${RK_SDK_DIR:-/workspace/rk3576_linux6.1_release/ubuntu}"
SYSROOT="${RK_SYSROOT:-}"
CROSS_PREFIX="${CROSS_PREFIX:-aarch64-linux-gnu-}"
HOST_TRIPLET="aarch64-linux-gnu"

usage() {
    local code="${1:-0}"
    cat <<EOF
用法: $0 [选项]

源码来源（三选一）:
      --tarball FILE        Python-X.Y.Z.tar.xz 路径
      --srcdir DIR          已解压的源码目录
      --download            从 python.org 下载（需要联网）

版本与安装:
      --version X.Y.Z       目标版本                  (默认: ${PYVER})
      --prefix PATH         目标机安装前缀            (默认: /usr/local/python<major.minor>)
      --destdir DIR         暂存根（DESTDIR）         (默认: <repo>/output/python-aarch64)
      --work-dir DIR        构建工作目录              (默认: <repo>/build-python)

交叉编译:
      --build-python PATH   宿主机上同版本的 python（3.11+ 交叉构建必需）
                            默认自动探测 python3.13 / /usr/local/bin/python3.13
      --sdk DIR             SDK 根目录
      --sysroot DIR         目标 rootfs
      --cross-prefix P      工具链前缀                (默认: aarch64-linux-gnu-)

其他:
  -j, --jobs N              并行任务数                (默认: \$(nproc))
      --clean               清空工作目录后重来
      --dry-run             只打印配置，不执行
  -h, --help                显示本帮助

产物:
  <destdir><prefix>/lib/libpython<major.minor>.so
  <destdir><prefix>/include/python<major.minor>/Python.h

编译完成后，用以下环境变量让 vStream 构建 Python 绑定:
  export VSTREAM_PYTHON_ROOT_DIR=<destdir><prefix>
  export VSTREAM_PYTHON_EXECUTABLE=<build python>
  scripts/build-rk-dev.sh --python
EOF
    exit "${code}"
}

die() { echo "错误: $*" >&2; exit 1; }
warn() { echo "警告: $*" >&2; }
need_arg() { [[ $# -ge 2 ]] || die "选项 $1 缺少参数"; }

# ========== 参数解析 ==========
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)      need_arg "$@"; PYVER="$2"; shift 2 ;;
        --prefix)       need_arg "$@"; PREFIX="$2"; shift 2 ;;
        --destdir)      need_arg "$@"; DESTDIR="$2"; shift 2 ;;
        --work-dir)     need_arg "$@"; WORK_DIR="$2"; shift 2 ;;
        --tarball)      need_arg "$@"; TARBALL="$2"; shift 2 ;;
        --srcdir)       need_arg "$@"; SRC_DIR="$2"; shift 2 ;;
        --download)     DO_DOWNLOAD="1"; shift ;;
        --build-python) need_arg "$@"; BUILD_PYTHON="$2"; shift 2 ;;
        --sdk)          need_arg "$@"; SDK_DIR="$2"; shift 2 ;;
        --sysroot)      need_arg "$@"; SYSROOT="$2"; shift 2 ;;
        --cross-prefix) need_arg "$@"; CROSS_PREFIX="$2"; shift 2 ;;
        -j|--jobs)      need_arg "$@"; JOBS="$2"; shift 2 ;;
        --clean)        CLEAN="1"; shift ;;
        --dry-run)      DRY_RUN="1"; shift ;;
        -h|--help)      usage 0 ;;
        *)              die "未知选项 '$1'（用 --help 查看用法）" ;;
    esac
done

# major.minor，例如 3.13
PYMM="$(echo "${PYVER}" | cut -d. -f1,2)"
: "${PYMAJOR:=${PYMM}}"
: "${PREFIX:=/usr/local/python${PYMM}}"
: "${SYSROOT:=${SDK_DIR}/armv8a-linux}"

[[ "${PYVER}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || die "版本号格式应为 X.Y.Z，当前: ${PYVER}"

# ========== 1. 环境检查 ==========
command -v "${CROSS_PREFIX}gcc" >/dev/null 2>&1 \
    || die "未找到交叉编译器 ${CROSS_PREFIX}gcc。
请在容器内先 source 厂家环境脚本，或用 --cross-prefix 指定前缀。"

[[ -d "${SYSROOT}" ]] || die "sysroot 不存在: ${SYSROOT}（用 --sysroot 指定）"
[[ -f "${SYSROOT}/usr/include/stdio.h" ]] \
    || die "sysroot 不像一个 rootfs（缺少 usr/include/stdio.h）: ${SYSROOT}"

# 宿主同版本 python（3.11+ 交叉构建必需）
if [[ -z "${BUILD_PYTHON}" ]]; then
    for _cand in "python${PYMM}" "/usr/local/bin/python${PYMM}" "python3"; do
        if command -v "${_cand}" >/dev/null 2>&1; then
            _v="$("${_cand}" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)"
            if [[ "${_v}" == "${PYMM}" ]]; then BUILD_PYTHON="$(command -v "${_cand}")"; break; fi
        fi
    done
fi

if [[ -z "${BUILD_PYTHON}" ]]; then
    die "未找到宿主机上的 python${PYMM}。
交叉构建需要它来生成冻结模块（--with-build-python）。
请安装同版本 python，或用 --build-python <路径> 指定。"
fi

_actual="$("${BUILD_PYTHON}" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[[ "${_actual}" == "${PYMM}" ]] \
    || die "--build-python 版本不匹配：需要 ${PYMM}，实际 ${_actual}（${BUILD_PYTHON}）"

# ========== 2. 源码准备 ==========
mkdir -p "${WORK_DIR}"
[[ "${CLEAN}" == "1" ]] && rm -rf "${WORK_DIR:?}/Python-${PYVER}" "${WORK_DIR:?}/build-aarch64"

if [[ -z "${SRC_DIR}" ]]; then
    if [[ -z "${TARBALL}" ]]; then
        for _t in "./Python-${PYVER}.tar.xz" "${PROJECT_ROOT}/Python-${PYVER}.tar.xz" \
                  "${PROJECT_ROOT}/third_party/Python-${PYVER}.tar.xz"; do
            [[ -f "${_t}" ]] && { TARBALL="${_t}"; break; }
        done
    fi
    if [[ -z "${TARBALL}" && "${DO_DOWNLOAD}" == "1" ]]; then
        TARBALL="${WORK_DIR}/Python-${PYVER}.tar.xz"
        if [[ ! -f "${TARBALL}" ]]; then
            echo "下载 https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tar.xz ..."
            if command -v curl >/dev/null 2>&1; then
                curl -fL --retry 3 -o "${TARBALL}" \
                    "https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tar.xz"
            elif command -v wget >/dev/null 2>&1; then
                wget -O "${TARBALL}" \
                    "https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tar.xz"
            else
                die "需要 curl 或 wget 才能下载源码"
            fi
        fi
    fi
    [[ -n "${TARBALL}" && -f "${TARBALL}" ]] \
        || die "未找到源码包 Python-${PYVER}.tar.xz。
请用 --tarball 指定，或 --download 联网下载，或 --srcdir 指向已解压目录。"
    SRC_DIR="${WORK_DIR}/Python-${PYVER}"
    if [[ ! -d "${SRC_DIR}" ]]; then
        echo "解压 ${TARBALL} -> ${WORK_DIR}"
        tar -xf "${TARBALL}" -C "${WORK_DIR}"
    fi
fi

[[ -x "${SRC_DIR}/configure" ]] || die "源码目录无效（缺少可执行 configure）: ${SRC_DIR}"

# ========== 3. 交叉编译环境变量 ==========
export CC="${CROSS_PREFIX}gcc"
export CXX="${CROSS_PREFIX}g++"
export AR="${CROSS_PREFIX}ar"
export RANLIB="${CROSS_PREFIX}ranlib"
export STRIP="${CROSS_PREFIX}strip"
export READELF="${CROSS_PREFIX}readelf"

export CPPFLAGS="--sysroot=${SYSROOT}"
export CFLAGS="--sysroot=${SYSROOT} -O2 -fPIC"
export CXXFLAGS="${CFLAGS}"
# $ORIGIN 用反斜杠转义，保证原样传给链接器（板端 bin/python 能找到同级 lib/libpython）
export LDFLAGS="--sysroot=${SYSROOT} -Wl,-rpath,\$ORIGIN/../lib"

# 目标侧 pkg-config：让 configure 从 sysroot 里找 openssl/zlib/sqlite 等
export PKG_CONFIG_SYSROOT_DIR="${SYSROOT}"
export PKG_CONFIG_LIBDIR="${SYSROOT}/usr/lib/pkgconfig:${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${SYSROOT}/usr/share/pkgconfig"
export PKG_CONFIG_PATH="${SYSROOT}/usr/local/lib/pkgconfig"

# autoconf 的交叉探测缓存：避免因无法运行目标程序而误判
export ac_cv_file__dev_ptmx=yes
export ac_cv_file__dev_ptc=no
export ac_cv_buggy_getaddrinfo=no

BUILD_DIR="${WORK_DIR}/build-aarch64"
mkdir -p "${BUILD_DIR}"

# ========== 4. 组装 configure 参数 ==========
# --build 必须是「构建机」三元组（宿主编译器），不能用交叉编译器的 -dumpmachine
# —— 后者返回的是目标三元组，会让 autoconf 认为 host==build，从而不走交叉分支
BUILD_TRIPLET="$(gcc -dumpmachine 2>/dev/null || cc -dumpmachine 2>/dev/null || echo x86_64-pc-linux-gnu)"

CONF_ARGS=(
    --host="${HOST_TRIPLET}"
    --build="${BUILD_TRIPLET}"
    --prefix="${PREFIX}"
    --enable-shared
    --with-ensurepip=no
)

# 按源码实际支持的选项动态追加，兼容不同小版本
_help="$("${SRC_DIR}/configure" --help 2>/dev/null || true)"
if echo "${_help}" | grep -q -- '--with-build-python'; then
    CONF_ARGS+=( --with-build-python="${BUILD_PYTHON}" )
else
    warn "该版本 configure 不支持 --with-build-python（3.11 以下），交叉构建可能失败"
fi
if echo "${_help}" | grep -q -- '--disable-test-modules'; then
    CONF_ARGS+=( --disable-test-modules )
fi
if echo "${_help}" | grep -q -- '--with-system-ffi'; then
    CONF_ARGS+=( --with-system-ffi )
fi

# ========== 5. 打印配置 ==========
echo "============================================"
echo "  交叉编译 CPython ${PYVER} -> aarch64"
echo "============================================"
printf "  %-14s: %s\n" "Source        " "${SRC_DIR}"
printf "  %-14s: %s\n" "Build dir     " "${BUILD_DIR}"
printf "  %-14s: %s\n" "Sysroot       " "${SYSROOT}"
printf "  %-14s: %s\n" "Cross prefix  " "${CROSS_PREFIX}"
printf "  %-14s: %s\n" "Build python  " "${BUILD_PYTHON}"
printf "  %-14s: %s\n" "Prefix        " "${PREFIX}"
printf "  %-14s: %s\n" "DESTDIR       " "${DESTDIR}"
printf "  %-14s: %s\n" "Jobs          " "${JOBS}"
echo "--------------------------------------------"
echo "  configure ${CONF_ARGS[*]}"
echo "============================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] 未执行任何构建命令"
    exit 0
fi

# ========== 6. 配置 / 编译 / 安装 ==========
cd "${BUILD_DIR}"
"${SRC_DIR}/configure" "${CONF_ARGS[@]}" 2>&1 | tee configure.log | tail -40

echo ""
echo "开始编译（-j${JOBS}）..."
make -j"${JOBS}"

echo ""
echo "安装到 ${DESTDIR}${PREFIX} ..."
make install DESTDIR="${DESTDIR}"

# ========== 7. 产物校验 ==========
PYLIB="${DESTDIR}${PREFIX}/lib/libpython${PYMM}.so"
PYINC="${DESTDIR}${PREFIX}/include/python${PYMM}/Python.h"

[[ -f "${PYLIB}" ]] || die "未找到 ${PYLIB}（--enable-shared 是否生效？）"
[[ -f "${PYINC}" ]] || die "未找到 ${PYINC}"

if command -v "${CROSS_PREFIX}readelf" >/dev/null 2>&1; then
    _machine="$("${CROSS_PREFIX}readelf" -h "${PYLIB}" | grep -o 'AArch64' | head -1 || true)"
    [[ "${_machine}" == "AArch64" ]] \
        || die "${PYLIB} 不是 aarch64 产物，检查是否用了错误的编译器"
fi

echo ""
echo "============================================"
echo "  构建完成"
echo "  libpython : ${PYLIB}"
echo "  headers   : ${DESTDIR}${PREFIX}/include/python${PYMM}"
echo ""
echo "  给 vStream 用（二选一）:"
echo "    1) 环境变量方式:"
echo "       export VSTREAM_PYTHON_ROOT_DIR=${DESTDIR}${PREFIX}"
echo "       export VSTREAM_PYTHON_EXECUTABLE=${BUILD_PYTHON}"
echo "       ${PROJECT_ROOT}/scripts/build-rk-dev.sh --python"
echo ""
echo "    2) 直接传参:"
echo "       ${PROJECT_ROOT}/scripts/build-rk-dev.sh --python \\"
echo "         --python-root ${DESTDIR}${PREFIX} \\"
echo "         --python-exec ${BUILD_PYTHON}"
echo "============================================"
