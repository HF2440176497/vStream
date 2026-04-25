# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import threading
from datetime import datetime

import numpy as np
import cv2

# 手动添加路径，例如：
sys.path.insert(0, "../../lib")
sys.path.insert(0, "../../python/test")

import vstream


class MyPythonModule(vstream.Module):
    """
    在 Python 中继承 vstream.Module，实现自定义处理逻辑。
    对应 C++ 中的 Pybind11ModuleV<Pybind11Module>。
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.frame_count = 0
        self.last_frame = None
        self.has_save_frame = False

    def open(self, params: dict) -> bool:
        print(f"[{self.get_name()}] open called with params={params}")
        return True

    def close(self):
        print(f"[{self.get_name()}] close called, processed {self.frame_count} frames")

    def process(self, data) -> int:
        data_frame = data.get_data_frame()
        cur_image = data_frame.get_frame()
        cv2.putText(cur_image, f"Frame {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 保存上一次 data 取出并修改的 frame
        if self.last_frame is not None and not self.has_save_frame:
            cv2.imwrite("last_frame.jpg", self.last_frame)
            self.has_save_frame = True

        self.last_frame = cur_image

        self.frame_count += 1
        if self.frame_count % 50 == 0:
            print(f"[{self.get_name()}] processed {self.frame_count} frames")
        return 0

    def on_eos(self, stream_id: str):
        print(f"[{self.get_name()}] on_eos: {stream_id}")


def test_python_module():
    print("=" * 60)
    print("TEST: Python custom module")
    print("=" * 60)

    # 构造一个使用 pyclass_name 指向 Python 类的 pipeline JSON
    config = {
        "profiler_config": {
            "enable_profile": False
        },
        "decoder": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::DataSource",
            "next_modules": ["py_module"],
            "custom_params": {
                "output_type": "cpu",
                "decoder_type": "cpu"
            }
        },
        "py_module": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::PyModule",
            "next_modules": ["osd"],
            "custom_params": {
                "pyclass_name": "test_module.MyPythonModule"
            }
        },
        "osd": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::DecodeQueue",
            "next_modules": [],
            "custom_params": {
                "queue_size": "100"
            }
        }
    }

    json_path = "pipeline_python_module.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    pipeline = vstream.Pipeline("py_module_pipeline")
    ok = pipeline.build_pipeline_by_json_file(json_path)
    assert ok, "Build pipeline failed"
    print("Pipeline built successfully")

    # 获取模块
    source_module = pipeline.get_source_module("decoder")
    assert source_module is not None

    stream_id = "channel-1"
    send_handler = vstream.SendHandler(source_module, stream_id)
    assert send_handler is not None

    osd_module = pipeline.get_output_module("osd")
    assert osd_module is not None

    ret = source_module.add_source(send_handler)
    assert ret == 0

    ok = pipeline.start()
    assert ok, "Pipeline start failed"
    print("Pipeline started")

    test_image = create_test_image(480, 640)
    cv2.imwrite("image.png", test_image)

    running = True
    send_count = 0
    receive_count = 0

    def send_thread():
        nonlocal send_count
        while running:
            pts = get_timestamp_ms()
            frame_id_s = str(send_count)
            send_handler.send(pts, frame_id_s, test_image.copy())
            send_count += 1
            time.sleep(0.02)

    def receive_thread():
        nonlocal receive_count
        while running:
            ok, data = osd_module.get_data(wait_ms=10)
            if not ok:
                time.sleep(0.01)
                continue
            receive_count += 1
            if receive_count % 20 == 0:
                print(f"[Receive] count={receive_count}, frame_id_s={data.frame_id_s}")

    t_send = threading.Thread(target=send_thread)
    t_recv = threading.Thread(target=receive_thread)
    t_send.start()
    t_recv.start()

    time.sleep(3)

    running = False
    t_send.join()
    t_recv.join()

    print(f"Total sent: {send_count}, total received: {receive_count}")

    send_handler.stop()
    send_handler.close()
    pipeline.stop()
    print("PASS: Python custom module")


if __name__ == "__main__":
    test_python_module()
