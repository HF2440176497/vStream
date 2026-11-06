#!/usr/bin/env bash
# =============================================================================
# 离线转换入口：编译 ov_trans_model -> ONNX 转 IR -> 预热 model cache -> 冒烟验证
#
#   bash tools/openvino_demo/convert.sh [onnx_path] [out.xml] [cache_dir]
#   SHAPE=1,3,640,640 DEVICE=CPU bash tools/openvino_demo/convert.sh
#
# 多输入模型：SHAPE 用分号分隔多条（index 或 tensor name 均可）
#   SHAPE='data=3,150,200,1;seq_len=3' bash tools/openvino_demo/convert.sh
#   SHAPE='0=1,3,640,640;1=1,1,256' bash tools/openvino_demo/convert.sh
#
# 注意：先不带 --shape 跑一次 ov_trans_model，看 [1] 打印的输入 index/name/shape；
#       缓存由 runtime 自动管理，OV 版本/设备/配置变化时会自动重建。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO_DIR="${ROOT}/tools/openvino_demo"
BUILD_DIR="${ROOT}/build/ov_demo"

cmake -S "${DEMO_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

ONNX="${1:-${ROOT}/bin/onnx/yolov5_sheath_detection.onnx}"
OUT_XML="${2:-${ROOT}/bin/model/openvino/yolov5_sheath_detection.xml}"
CACHE_DIR="${3:-$(dirname "${OUT_XML}")/cache}"
SHAPE="${SHAPE:-1,3,640,640}"
DEVICE="${DEVICE:-CPU}"

# 分号分隔的多条 --shape，逐条传给 ov_trans_model
IFS=';' read -r -a SHAPE_LIST <<< "${SHAPE}"
SHAPE_ARGS=()
for s in "${SHAPE_LIST[@]}"; do
  if [[ -n "${s}" ]]; then
    SHAPE_ARGS+=(--shape "${s}")
  fi
done

mkdir -p "$(dirname "${OUT_XML}")" "${CACHE_DIR}"

echo "=== 离线转换 + 预热 model cache: ${ONNX} -> ${OUT_XML} ==="
"${BUILD_DIR}/ov_trans_model" "${ONNX}" "${OUT_XML}" \
  "${SHAPE_ARGS[@]}" \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}"

echo "=== 冒烟验证: 命中离线生成的缓存 ==="
"${BUILD_DIR}/ov_hello" "${OUT_XML}" "${DEVICE}" 5 --cache-dir "${CACHE_DIR}"

echo "=== 对比: 禁用缓存（观察首次编译耗时）==="
"${BUILD_DIR}/ov_hello" "${OUT_XML}" "${DEVICE}" 5 --no-cache

echo "=== 完成，部署侧命令 ==="
echo "  ${BUILD_DIR}/ov_hello ${OUT_XML} ${DEVICE} 10 --cache-dir ${CACHE_DIR}"
echo "  demo 不传 --cache-dir 时默认使用 IR 同目录的 ov_cache/（检测到 EXPORT_IMPORT 能力才开启）"
