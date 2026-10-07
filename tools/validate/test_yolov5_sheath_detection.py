# -*- coding: utf-8 -*-
"""
离线验证脚本
    model_name   : yolov5_sheath_detection
    preproc_name : Pre_YOLO_CPU_v2
    postproc_name: Post_YOLOv5_CPU
    postproc cfg : yolov5_sheath_detection.json

模型实测
    input  = [1, 3, 640, 640]
    output = [1, 25200, 17]  ->  num_classes = 17 - 5 = 12

用法示例
    # 1) 最小验证：加载 + 元信息
    python test_yolov5_sheath_detection.py --info-only

    # 2) 完整链路
    python test_yolov5_sheath_detection.py --image ../../bin/image.png

    # 3) 指定模型+后处理配置
    
    python test_yolov5_sheath_detection.py \
        --model /path/yolov5_sheath_detection.engine \
        --postproc-config /path/yolov5_sheath_detection.json

    # 4) 跳过 benchmark / 关闭结果保存
    python test_yolov5_sheath_detection.py --no-benchmark --no-save
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import cv2

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


if os.path.join(PROJECT_ROOT, "lib") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib"))

try:
    import vstream
except ImportError:
    print("[ERROR] 无法 import vstream，请先构建 Python API 并设置环境变量：")
    print("  ./build.sh --python")
    print("  source python/test/env.sh   # 设置 LD_LIBRARY_PATH / PYTHONPATH")
    sys.exit(1)


DEFAULT_MODEL = os.path.join(
    PROJECT_ROOT, "bin", "model", "20260625", "yolov5_sheath_detection.engine")

DEFAULT_POSTPROC_CONFIG = os.path.join(
    SCRIPT_DIR, "yolov5_sheath_detection.json")

DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "bin", "image.png")


PREPROC_NAME = "Pre_YOLO_CPU_v2"
POSTPROC_NAME = "Post_YOLOv5_CPU"

# 模型类别数：来自 output [1, 25200, 17] => 17 - 5
NUM_CLASSES = 12


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def print_separator(title: str):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def create_synthetic_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """合成一张 BGR 测试图（无真实样本时兜底）。"""
    img = np.full((height, width, 3), (60, 90, 120), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (320, 320), (0, 255, 0), 3)
    cv2.rectangle(img, (520, 180), (860, 520), (0, 0, 255), -1)
    cv2.circle(img, (1020, 400), 90, (255, 0, 0), -1)
    cv2.putText(img, "synthetic", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return img


def ensure_postproc_config(path: str, num_classes: int) -> str:
    """
    保证后处理配置存在。

    Post_YOLOv5_CPU::Init 读取的 json 结构
        {
          "classes": { "0": {"name": "...", "threshold": 0.5}, ... },
          "max_boxes_num": 300,          # 可选
          "nms_iou_threshold": 0.45      # 可选
        }
    """
    if path and os.path.exists(path):
        print(f"[INFO] 使用后处理配置: {path}")
        return path

    cfg = {
        "classes": {
            str(i): {"name": f"class_{i}", "threshold": 0.5}
            for i in range(num_classes)
        },
        "max_boxes_num": 300,
        "nms_iou_threshold": 0.45,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[WARN] 未找到后处理配置，已生成模板: {path}")
    print(f"[WARN] 类别名称为占位符 class_0..class_{num_classes - 1}，"
          f"请按实际业务替换 name 与 threshold")
    return path


def load_class_names(config_path: str) -> dict:
    """从后处理配置读取 id -> name 映射（ModelValidator 不回填 class_name）。"""
    names = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes", {})
        for k, v in classes.items():
            names[int(k)] = v.get("name", f"class_{k}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 解析类别配置失败: {e}")
    return names


def draw_detections(image: np.ndarray, detections, class_names: dict) -> np.ndarray:
    """
    在图上画框并保存。

    注意：ValidatorDetection 的 x/y/w/h 是「原图像素坐标」(左上角 + 宽高)，
    而非头文件注释所写的 normalized。postproc 用 DataFrame 的宽高做反 letterbox，
    输出即为原图像素。这里可直接使用。
    """
    canvas = image.copy()
    palette = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
        (0, 255, 128), (255, 128, 0), (128, 255, 0), (60, 60, 60),
    ]
    for det in detections:
        x, y, w, h = int(det.x), int(det.y), int(det.w), int(det.h)
        color = palette[det.class_id % len(palette)]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        name = det.class_name or class_names.get(det.class_id, f"cls{det.class_id}")
        label = f"{det.class_id}:{name} {det.score:.2f}"
        cv2.putText(canvas, label, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


# --------------------------------------------------------------------------- #
# 步骤 1：加载模型 & 打印元信息
# --------------------------------------------------------------------------- #
def step_load_and_info(validator) -> bool:
    print_separator("Step 1: 加载模型 & 元信息")

    if not validator.load():
        print("[FAIL] 模型加载失败，请检查 --model 路径 / device_type / TensorRT 环境")
        return False
    if not validator.is_loaded():
        print("[FAIL] is_loaded() 为 False")
        return False
    print("[OK] 模型加载成功")

    info = validator.get_model_info()
    print(f"  model_path : {info.model_path}")
    print(f"  device     : {info.device_type} (id={info.device_id})")
    print(f"  batch_size : {info.batch_size}")
    print(f"  input HW C : {info.width} x {info.height} x {info.channel}")
    print(f"  inputs ({len(info.inputs)}):")
    for i, t in enumerate(info.inputs):
        print(f"    [{i}] name={t.name} shape={t.shape} dtype={t.dtype}")
    print(f"  outputs ({len(info.outputs)}):")
    for i, t in enumerate(info.outputs):
        print(f"    [{i}] name={t.name} shape={t.shape} dtype={t.dtype}")

    # 交叉校验：期望 [1, 25200, 5+12]
    if info.outputs:
        out_shape = info.outputs[0].shape
        if len(out_shape) == 3:
            nc = out_shape[2] - 5
            print(f"  -> 推断类别数 = {out_shape[2]} - 5 = {nc}")
            if nc != NUM_CLASSES:
                print(f"  [WARN] 与脚本常量 NUM_CLASSES={NUM_CLASSES} 不一致，"
                      f"请同步更新！")
    if (info.width, info.height, info.channel) != (640, 640, 3):
        print(f"  [WARN] 期望输入 640x640x3，实际 "
              f"{info.width}x{info.height}x{info.channel}")
    return True


# --------------------------------------------------------------------------- #
# 步骤 2：原始张量推理（无前后处理，验证 engine 本身可跑通）
# --------------------------------------------------------------------------- #
def step_raw_infer(validator):
    print_separator("Step 2: 原始张量推理（无前/后处理）")
    info = validator.get_model_info()

    inputs = []
    for t in info.inputs:
        count = int(np.prod(t.shape))
        inputs.append(np.random.rand(count).astype(np.float32))
        print(f"  输入 '{t.name}': shape={t.shape}, 元素数={count}")

    t0 = time.time()
    outputs = validator.infer(inputs)
    dt = (time.time() - t0) * 1000.0

    if not outputs:
        print("[FAIL] infer() 返回空（模型未加载或输入尺寸不匹配）")
        return
    print(f"  [OK] {len(outputs)} 个输出张量, 耗时 {dt:.2f} ms")
    for i, out in enumerate(outputs):
        arr = np.asarray(out)
        flag = ""
        if np.any(np.isnan(arr)):
            flag += " [含 NaN!]"
        if np.any(np.isinf(arr)):
            flag += " [含 Inf!]"
        print(f"    output[{i}]: size={arr.size} min={arr.min():.4f} "
              f"max={arr.max():.4f} mean={arr.mean():.4f}{flag}")


# --------------------------------------------------------------------------- #
# 步骤 3：端到端（前处理 -> 推理 -> 后处理 -> 检测框）
# --------------------------------------------------------------------------- #
def step_run_e2e(validator, image: np.ndarray, config_path: str,
                 class_names: dict, save: bool):
    print_separator("Step 3: 端到端 image -> preproc -> infer -> postproc")

    # ModelValidator.run_e2e 会把 params 透传给 Postproc::Init。
    # config_file 用绝对路径即可（GetPathRelativeToTheJSONFile 对绝对路径直接返回）。
    postproc_params = {"config_file": os.path.abspath(config_path)}

    print(f"  image        : {image.shape[1]}x{image.shape[0]} (WxH, BGR)")
    print(f"  preproc      : {PREPROC_NAME}")
    print(f"  postproc     : {POSTPROC_NAME}")
    print(f"  postproc cfg : {postproc_params['config_file']}")

    result = validator.run_e2e(
        np.ascontiguousarray(image),
        PREPROC_NAME,
        POSTPROC_NAME,
        postproc_params=postproc_params,
    )

    if result.error:
        print(f"  [ERROR] {result.error}")
        print("  排查：1) config_file 是否存在且为 {\"classes\":{...}} 结构；"
              "2) preproc/postproc 名是否与编译进库的插件一致")
        return result

    print(f"  [OK] {len(result.detections)} 个检测框, 端到端耗时 "
          f"{result.latency_ms:.2f} ms")
    for i, det in enumerate(result.detections[:20]):
        name = det.class_name or class_names.get(det.class_id, "")
        print(f"    det[{i}]: id={det.class_id} name='{name}' "
              f"score={det.score:.4f} "
              f"bbox=[x={det.x:.1f}, y={det.y:.1f}, w={det.w:.1f}, h={det.h:.1f}]")
    if len(result.detections) > 20:
        print(f"    ... 其余 {len(result.detections) - 20} 个略")

    # 类别分布统计
    if result.detections:
        hist = {}
        for d in result.detections:
            hist[d.class_id] = hist.get(d.class_id, 0) + 1
        dist = ", ".join(
            f"{class_names.get(k, k)}({k}):{v}" for k, v in sorted(hist.items()))
        print(f"  类别分布: {dist}")

    if save and result.detections:
        out_path = os.path.join(SCRIPT_DIR, "_result.jpg")
        canvas = draw_detections(image, result.detections, class_names)
        cv2.imwrite(out_path, canvas)
        print(f"  可视化已保存: {out_path}")

    return result


# --------------------------------------------------------------------------- #
# 步骤 4：性能基准
# --------------------------------------------------------------------------- #
def step_benchmark(validator, image: np.ndarray, config_path: str,
                   warmup: int, runs: int):
    print_separator(f"Step 4: Benchmark (warmup={warmup}, runs={runs})")

    postproc_params = {"config_file": os.path.abspath(config_path)}
    results = validator.benchmark(
        np.ascontiguousarray(image),
        PREPROC_NAME,
        POSTPROC_NAME,
        postproc_params=postproc_params,
        warmup_runs=warmup,
        test_runs=runs,
        batch_sizes=[1],
    )
    if not results:
        print("  [WARN] 无基准结果（可能 RunE2E 全部失败）")
        return
    for r in results:
        print(f"  batch={r.batch_size}: avg={r.avg_ms:.2f}ms min={r.min_ms:.2f}ms "
              f"max={r.max_ms:.2f}ms p99={r.p99_ms:.2f}ms "
              f"fps={r.fps:.1f} errors={r.error_count}")
        print("  提示：这是「单张图串行 E2E」吞吐，不含 batching，"
              "低于 pipeline 实际吞吐属正常。")



def main():
    parser = argparse.ArgumentParser(
        description="yolov5_sheath_detection 离线验证 (ModelValidator)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"engine 完整路径 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "rockchip"],
                        help="设备类型 (默认: cuda)")
    parser.add_argument("--device-id", type=int, default=0, help="设备号 (默认: 0)")
    parser.add_argument("--input-ordered-index", type=int, default=0,
                        help="主输入张量下标 (默认: 0)")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE,
                        help="测试图路径；为空或不存在则用合成图")
    parser.add_argument("--postproc-config", type=str, default=DEFAULT_POSTPROC_CONFIG,
                        help="后处理类别配置 json；不存在则自动生成模板")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help=f"生成模板配置时的类别数 (默认: {NUM_CLASSES})")
    parser.add_argument("--info-only", action="store_true",
                        help="只加载并打印模型信息，不做推理")
    parser.add_argument("--with-raw-infer", action="store_true",
                        help="额外执行一次随机张量推理")
    parser.add_argument("--no-benchmark", action="store_true", help="跳过 benchmark")
    parser.add_argument("--no-save", action="store_true", help="不保存可视化结果")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark 预热次数")
    parser.add_argument("--runs", type=int, default=50, help="benchmark 测试次数")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] 模型不存在: {args.model}")
        sys.exit(2)

    config_path = ensure_postproc_config(args.postproc_config, args.num_classes)
    class_names = load_class_names(config_path)

    validator = vstream.ModelValidator(
        args.model, args.device, args.device_id, args.input_ordered_index)

    if not step_load_and_info(validator):
        sys.exit(1)
    if args.info_only:
        print("\n[INFO] --info-only，结束。")
        return

    # 准备图像
    if args.image and os.path.exists(args.image):
        image = cv2.imread(args.image)
        if image is None:
            print(f"[ERROR] 无法读取图片: {args.image}")
            sys.exit(1)
        print(f"\n[INFO] 使用图片: {args.image} ({image.shape[1]}x{image.shape[0]})")
    else:
        image = create_synthetic_image()
        print("\n[INFO] 使用合成图 1280x720（无真实样本时仅验证链路是否跑通）")

    if args.with_raw_infer:
        step_raw_infer(validator)

    step_run_e2e(validator, image, config_path, class_names, save=not args.no_save)

    if not args.no_benchmark:
        step_benchmark(validator, image, config_path, args.warmup, args.runs)

    print("\n" + "=" * 68)
    print("  验证流程结束")
    print("=" * 68)


if __name__ == "__main__":
    main()
