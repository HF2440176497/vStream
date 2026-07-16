# -*- coding: utf-8 -*-
"""
ModelValidator — 独立模型验证

无需搭建 Pipeline，即可验证模型的：
  1. 加载与元信息检查 (tensor shapes, dtypes)
  2. 原始张量推理 (raw tensor in/out, 无前后处理)
  3. 端到端推理 (image -> preproc -> infer -> postproc -> detections)
  4. 性能基准测试 (latency, p99, fps)

用法:
    python test_model_validator.py \
        --model /path/to/yolov8s.engine \
        --device cuda \
        --image /path/to/test.jpg \
        --postproc-config /path/to/yolo_coco.json

    # 仅检查模型信息 (不推理)
    python test_model_validator.py --model /path/to/model.engine --info-only

    # 使用合成图像 (无需真实图片)
    python test_model_validator.py --model /path/to/model.engine --device cuda
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2

try:
    import vstream
except ImportError:
    print("[ERROR] Cannot import vstream. Build the Python module first:")
    print("  ./build.sh --python")
    sys.exit(1)


def create_synthetic_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a synthetic BGR test image with some shapes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 100, 150)  # BGR background
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 3)
    cv2.rectangle(img, (500, 200), (800, 500), (0, 0, 255), -1)
    cv2.circle(img, (1000, 400), 80, (255, 0, 0), -1)
    return img


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


#  Test 1: Load model & inspect metadata
def test_load_and_info(validator: vstream.ModelValidator):
    print_separator("Step 1: Load Model & Inspect Metadata")

    if not validator.load():
        print("[FAIL] Model load failed.")
        return False

    assert validator.is_loaded(), "is_loaded() should be True after load()"
    print("[OK] Model loaded successfully.")

    info = validator.get_model_info()
    print(f"  Model path : {info.model_path}")
    print(f"  Device     : {info.device_type} (id={info.device_id})")
    print(f"  Batch size : {info.batch_size}")
    print(f"  Input shape: {info.width}x{info.height}x{info.channel}")
    print(f"  Inputs     : {len(info.inputs)}")
    for i, t in enumerate(info.inputs):
        print(f"    [{i}] name={t.name}, shape={t.shape}, dtype={t.dtype}")
    print(f"  Outputs    : {len(info.outputs)}")
    for i, t in enumerate(info.outputs):
        print(f"    [{i}] name={t.name}, shape={t.shape}, dtype={t.dtype}")

    return True


#  Test 2: Raw tensor inference
def test_raw_infer(validator: vstream.ModelValidator):
    print_separator("Step 2: Raw Tensor Inference (no preproc/postproc)")

    info = validator.get_model_info()

    # Build random float32 inputs matching tensor shapes
    inputs = []
    for t in info.inputs:
        count = 1
        for d in t.shape:
            count *= d
        data = np.random.rand(count).astype(np.float32)
        inputs.append(data)
        print(f"  Input '{t.name}': shape={t.shape}, elements={count}")

    outputs = validator.infer(inputs)

    if not outputs:
        print("[FAIL] Inference returned empty results.")
        return

    print(f"  [OK] Got {len(outputs)} output tensor(s):")
    for i, out in enumerate(outputs):
        arr = np.asarray(out)
        print(f"    Output[{i}]: size={arr.size}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}, "
              f"mean={arr.mean():.4f}")
        # Check for NaN/Inf
        if np.any(np.isnan(arr)):
            print(f"    [WARN] Output[{i}] contains NaN!")
        if np.any(np.isinf(arr)):
            print(f"    [WARN] Output[{i}] contains Inf!")


#  Test 3: End-to-end inference
def test_run_e2e(validator: vstream.ModelValidator, image: np.ndarray,
                 preproc_name: str, postproc_name: str,
                 postproc_config: str = ""):
    print_separator("Step 3: End-to-End (image -> preproc -> infer -> postproc)")

    postproc_params = {}
    if postproc_config:
        postproc_params["config_file"] = postproc_config

    print(f"  Image       : {image.shape[1]}x{image.shape[0]} (WxH)")
    print(f"  Preproc     : {preproc_name}")
    print(f"  Postproc    : {postproc_name}")
    if postproc_config:
        print(f"  Postproc cfg: {postproc_config}")

    result = validator.run_e2e(
        image, preproc_name, postproc_name,
        postproc_params=postproc_params
    )

    if result.error:
        print(f"  [ERROR] {result.error}")
        if not postproc_config:
            print("  (hint: provide --postproc-config for postproc label names)")
        return

    print(f"  [OK] {len(result.detections)} detections in {result.latency_ms:.2f} ms")
    for i, det in enumerate(result.detections[:20]):
        print(f"    det[{i}]: class={det.class_id}, name='{det.class_name}', "
              f"score={det.score:.4f}, bbox=[{det.x:.3f}, {det.y:.3f}, "
              f"{det.w:.3f}, {det.h:.3f}]")
    if len(result.detections) > 20:
        print(f"    ... and {len(result.detections) - 20} more")


#  Test 4: Benchmark
def test_benchmark(validator: vstream.ModelValidator, image: np.ndarray,
                   preproc_name: str, postproc_name: str,
                   postproc_config: str = "",
                   warmup: int = 10, runs: int = 50):
    print_separator(f"Step 4: Benchmark (warmup={warmup}, runs={runs})")

    postproc_params = {}
    if postproc_config:
        postproc_params["config_file"] = postproc_config

    results = validator.benchmark(
        image, preproc_name, postproc_name,
        postproc_params=postproc_params,
        warmup_runs=warmup,
        test_runs=runs,
        batch_sizes=[1]
    )

    for r in results:
        print(f"  Batch={r.batch_size}: "
              f"avg={r.avg_ms:.2f}ms, "
              f"min={r.min_ms:.2f}ms, "
              f"max={r.max_ms:.2f}ms, "
              f"p99={r.p99_ms:.2f}ms, "
              f"fps={r.fps:.1f}, "
              f"errors={r.error_count}")



#  Test 5: Error path tests (no model needed)
def test_error_paths():
    print_separator("Step 0: Error Path Tests (no model needed)")

    # Construct with non-existent model
    v = vstream.ModelValidator("/nonexistent/model.engine", "cpu", 0)
    assert not v.is_loaded(), "Should not be loaded before load()"
    print("  [OK] Constructed validator, not loaded")

    # Load should fail
    ok = v.load()
    assert not ok, "Load should fail for non-existent model"
    print("  [OK] Load() returned False for non-existent model")

    # GetModelInfo should return empty info
    info = v.get_model_info()
    assert info.batch_size == 0
    assert len(info.inputs) == 0
    print("  [OK] get_model_info() returns empty before load")

    # Infer should return empty
    outputs = v.infer([np.zeros(1, dtype=np.float32)])
    assert len(outputs) == 0
    print("  [OK] infer() returns empty before load")

    # RunE2E should return error
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    result = v.run_e2e(dummy_img, "Pre_YOLO_CPU_v2", "Post_YOLOv8_CPU_v2")
    assert result.error, "run_e2e should have error before load"
    print(f"  [OK] run_e2e() returns error: '{result.error}'")


def main():
    parser = argparse.ArgumentParser(
        description="vStream ModelValidator — standalone model validation"
    )
    parser.add_argument("--model", type=str, default="",
                        help="Path to model file (.engine / .rknn)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "rockchip"],
                        help="Device type (default: cuda)")
    parser.add_argument("--device-id", type=int, default=0,
                        help="Device ID (default: 0)")
    parser.add_argument("--image", type=str, default="",
                        help="Path to test image (BGR). If empty, uses synthetic image.")
    parser.add_argument("--postproc-config", type=str, default="",
                        help="Path to postproc config JSON (e.g. yolo_coco.json)")
    parser.add_argument("--preproc", type=str, default="Pre_YOLO_CPU_v2",
                        help="Preproc class name (default: Pre_YOLO_CPU_v2)")
    parser.add_argument("--postproc", type=str, default="Post_YOLOv8_CPU_v2",
                        help="Postproc class name (default: Post_YOLOv8_CPU_v2)")
    parser.add_argument("--info-only", action="store_true",
                        help="Only load model and print info, skip inference")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip benchmark step")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Benchmark warmup runs (default: 10)")
    parser.add_argument("--runs", type=int, default=50,
                        help="Benchmark test runs (default: 50)")
    args = parser.parse_args()

    # --- Error path tests (always run, no model needed) ---
    test_error_paths()

    if not args.model:
        print("\n[INFO] No --model provided. Exiting after error path tests.")
        print("[INFO] To test with a real model, run:")
        print("  python test_model_validator.py --model /path/to/model.engine --device cuda")
        return

    validator = vstream.ModelValidator(
        args.model, args.device, args.device_id, 0
    )

    # --- Step 1: Load & info ---
    if not test_load_and_info(validator):
        print("\n[FAIL] Cannot continue without a loaded model.")
        sys.exit(1)

    if args.info_only:
        print("\n[INFO] --info-only mode, skipping inference tests.")
        return

    if args.image and os.path.exists(args.image):
        image = cv2.imread(args.image)
        if image is None:
            print(f"[ERROR] Cannot read image: {args.image}")
            sys.exit(1)
        print(f"\n[INFO] Using image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    else:
        image = create_synthetic_image()
        print("[INFO] Using synthetic test image (1280x720)")

    # --- Step 2: Raw tensor inference ---
    test_raw_infer(validator)

    # --- Step 3: End-to-end ---
    test_run_e2e(validator, image, args.preproc, args.postproc,
                 args.postproc_config)

    # --- Step 4: Benchmark ---
    if not args.no_benchmark:
        test_benchmark(validator, image, args.preproc, args.postproc,
                       args.postproc_config, args.warmup, args.runs)

    print("\n" + "=" * 60)
    print("  All validation steps completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
