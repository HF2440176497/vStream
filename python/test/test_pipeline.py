# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import threading
from datetime import datetime

import numpy as np
import cv2

import vstream


def get_timestamp_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def create_test_image(height: int = 480, width: int = 640) -> np.ndarray:
    """创建一张测试用的 BGR 图像（numpy uint8 数组）。"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (128, 64, 32)  # BGR
    return image


def create_pipeline_json(output_path: str, mode: str = "image") -> None:
    """
    构造一个简单的 pipeline JSON 配置文件。

    mode:
      - "image" : DataSource -> ProcessOne (图像文件输入)
      - "send"  : DataSource -> DecodeQueue (外部 Send 输入)
    """
    if mode == "image":
        config = {
            "profiler_config": {
                "enable_profile": False
            },
            "decoder": {
                "parallelism": 1,
                "max_input_queue_size": 20,
                "class_name": "cnstream::DataSource",
                "next_modules": ["osd"],
                "custom_params": {
                    "output_type": "cpu",
                    "decoder_type": "cpu",
                    "file_path": "image.png",
                    "frame_rate": "30"
                }
            },
            "osd": {
                "parallelism": 1,
                "max_input_queue_size": 20,
                "class_name": "cnstream::DecodeQueue",
                "next_modules": [],
                "custom_params": {
                    "queue_size": "30"
                }
            }
        }
    elif mode == "send":
        config = {
            "profiler_config": {
                "enable_profile": False
            },
            "decoder": {
                "parallelism": 1,
                "max_input_queue_size": 20,
                "class_name": "cnstream::DataSource",
                "next_modules": ["osd"],
                "custom_params": {
                    "output_type": "cpu",
                    "decoder_type": "cpu"
                }
            },
            "osd": {
                "parallelism": 1,
                "max_input_queue_size": 20,
                "class_name": "cnstream::DecodeQueue",
                "next_modules": [],
                "custom_params": {
                    "queue_size": "30"
                }
            }
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. 测试 Pipeline 基本接口
# ---------------------------------------------------------------------------
def test_pipeline_basic():
    print("=" * 60)
    print("TEST: Pipeline basic interface")
    print("=" * 60)

    pipeline = vstream.Pipeline("test_pipeline")
    print(f"Pipeline name: {pipeline.get_name()}")
    assert pipeline.get_name() == "test_pipeline"

    # is_running 在未启动时应为 False
    assert not pipeline.is_running()
    print("PASS: Pipeline basic interface")


# ---------------------------------------------------------------------------
# 2. 测试 DataSource + ImageHandler 流水线（参考 test_source.cpp）
# ---------------------------------------------------------------------------
def test_image_source_pipeline():
    print("=" * 60)
    print("TEST: Image source pipeline")
    print("=" * 60)

    json_path = "pipeline_source_base.json"
    create_pipeline_json(json_path, mode="image")

    pipeline = vstream.Pipeline("image_pipeline")
    ok = pipeline.build_pipeline_by_json_file(json_path)
    assert ok, "Build pipeline failed"
    print("Pipeline built successfully")

    # 获取 DataSource 模块
    source_module = pipeline.get_source_module("decoder")
    assert source_module is not None, "get_source_module('decoder') returned None"
    print(f"Source module type: {type(source_module)}")

    # 创建 ImageHandler
    stream_id = "channel-1"
    image_handler = vstream.ImageHandler(source_module, stream_id)
    assert image_handler is not None
    print(f"ImageHandler created, stream_id={image_handler.get_stream_id()}")

    # 启动流水线
    ok = pipeline.start()
    assert ok, "Pipeline start failed"
    print("Pipeline started")
    assert pipeline.is_running()

    # 添加源流
    ret = source_module.add_source(image_handler)
    assert ret == 0, f"AddSource failed, ret={ret}"
    print("ImageHandler added to source module")

    # 运行一段时间
    time.sleep(2)

    # 停止 handler 并关闭
    image_handler.stop()
    image_handler.close()
    print("ImageHandler stopped and closed")

    # 停止流水线
    pipeline.stop()
    print("Pipeline stopped")
    print("PASS: Image source pipeline")


# ---------------------------------------------------------------------------
# 3. 测试 DataSource + SendHandler + DecodeQueue（参考 test_send.cpp）
# ---------------------------------------------------------------------------
def test_send_pipeline():
    print("=" * 60)
    print("TEST: Send source pipeline with DecodeQueue")
    print("=" * 60)

    json_path = "pipeline_source_send.json"
    create_pipeline_json(json_path, mode="send")

    pipeline = vstream.Pipeline("send_pipeline")
    ok = pipeline.build_pipeline_by_json_file(json_path)
    assert ok, "Build pipeline failed"
    print("Pipeline built successfully")

    # 获取 DataSource 模块
    source_module = pipeline.get_source_module("decoder")
    assert source_module is not None
    print(f"Source module type: {type(source_module)}")

    # 创建 SendHandler
    stream_id = "channel-1"
    send_handler = vstream.SendHandler(source_module, stream_id)
    assert send_handler is not None
    print(f"SendHandler created, stream_id={send_handler.get_stream_id()}")

    # 获取 DecodeQueue 模块（消费端）
    osd_module = pipeline.get_output_module("osd")
    assert osd_module is not None
    print(f"OutputModule module type: {type(osd_module)}")

    # 添加源流并启动流水线
    ret = source_module.add_source(send_handler)
    assert ret == 0, f"AddSource failed, ret={ret}"

    ok = pipeline.start()
    assert ok, "Pipeline start failed"
    print("Pipeline started")

    running = True
    send_count = 0
    receive_count = 0

    def send_thread():
        nonlocal send_count
        while running:
            pts = get_timestamp_ms()
            frame_id_s = str(send_count)
            test_image = create_test_image(480, 640)
            ok = send_handler.send(pts, frame_id_s, test_image)
            if ok != 0:
                print(f"Warning: send returned {ok}")
            send_count += 1
            time.sleep(0.02)  # 50 fps

    def receive_thread():
        nonlocal receive_count
        while running:
            # get_data 返回 (ok, data)
            ok, data = osd_module.get_data(wait_ms=10)
            if not ok:
                time.sleep(0.01)
                continue
            receive_count += 1
            if receive_count % 20 == 0:
                print(f"Received {receive_count} frames, latest frame_id_s={data.frame_id_s}")

    t_send = threading.Thread(target=send_thread)
    t_recv = threading.Thread(target=receive_thread)
    t_send.start()
    t_recv.start()

    # 运行 5 秒
    time.sleep(5)

    running = False
    t_send.join()
    t_recv.join()

    print(f"Total sent: {send_count}, total received: {receive_count}")

    # note: 可以直接 pipeline stop
    # send_handler.stop()
    # send_handler.close()
    # print("SendHandler stopped and closed")

    pipeline.stop()
    print("Pipeline stopped")
    print("PASS: Send source pipeline with DecodeQueue")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("vstream Python binding validation script")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print()

    test_pipeline_basic()
    print()

    try:
        test_image_source_pipeline()
    except Exception as e:
        print(f"SKIPPED/FAILED test_image_source_pipeline: {e}")
    print()

    try:
        test_send_pipeline()
    except Exception as e:
        print(f"SKIPPED/FAILED test_send_pipeline: {e}")
    print()

    print("=" * 60)
    print("All tests finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
