# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import threading
from datetime import datetime

# 获取当前脚本所在目录，添加到 sys.path 以便 C++ 能 import 到 test_module
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

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


class MyPythonModule(vstream.Module):
    """
    在 Python 中继承 vstream.Module，实现自定义处理逻辑。
    对应 C++ 中的 Pybind11ModuleV<Pybind11Module>。
    """

    def __init__(self, name: str):
        vstream.Module.__init__(self, name)
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
        cur_image = data_frame.get_image()

        # 保存 （1） 上一次 data 取出并修改的 frame, （2） 当前 data 取出的 frame
        # 两者应该不同 last_frame 是存在标记的
        if self.last_frame is not None and not self.has_save_frame:
            cv2.imwrite(f"py_last_frame.jpg", self.last_frame)
            cv2.imwrite(f"py_cur_frame.jpg", cur_image)
            self.has_save_frame = True

        cv2.putText(cur_image, f"Frame {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
            "next_modules": ["sink"],
            "custom_params": {
                "pyclass_name": "test_module.MyPythonModule"
            }
        },
        "sink": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::DataSink",
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
    source = pipeline.get_data_source("decoder")
    assert source is not None

    stream_id = "channel-1"
    send_handler = vstream.SendHandler(source, stream_id)
    assert send_handler is not None

    sink = pipeline.get_data_sink("sink")
    assert sink is not None

    queue_handler = vstream.QueueHandler(sink, stream_id)
    assert queue_handler is not None

    ret = source.add_source(send_handler)
    assert ret == 0

    ret = sink.add_sink(queue_handler)
    assert ret == 0, f"AddSink failed, ret={ret}"

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
            send_handler.send(pts, frame_id_s, test_image)
            send_count += 1
            time.sleep(0.02)

    def receive_thread():
        nonlocal receive_count
        while running:
            ok, data = queue_handler.get_data(wait_ms=10)
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
