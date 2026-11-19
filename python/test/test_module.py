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

source_module_name = "source"
sink_module_name = "sink"

stream_id = "channel-4"

def get_timestamp_ms() -> int:
    return int(datetime.now().timestamp() * 1000)

def create_test_image(height: int = 480, width: int = 640) -> np.ndarray:
    """创建一张测试用的 BGR 图像"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (128, 64, 32)  # BGR
    return image


class MyPythonModule(vstream.Module):
    """
    在 Python 中继承 vstream.Module 实现自定义处理逻辑。
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
        "source": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::DataSource",
            "next_modules": ["py_module"],
            "custom_params": {
                "config_file": "data_source.json"
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
                "config_file": "data_sink.json"
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
    source = pipeline.get_data_source(source_module_name)
    assert source is not None

    send_handler = vstream.SendHandler(source, stream_id)
    assert send_handler is not None

    sink = pipeline.get_data_sink(sink_module_name)
    assert sink is not None

    queue_handler = vstream.QueueHandler(sink, stream_id)
    assert queue_handler is not None

    ok = pipeline.start()
    assert ok, "Pipeline start failed"
    print("Pipeline started")

    ret = source.add_source(send_handler)
    assert ret == 0

    ret = sink.add_sink(queue_handler)
    assert ret == 0, f"AddSink failed, ret={ret}"

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

    time.sleep(10)

    running = False
    t_send.join()
    t_recv.join()

    print(f"Total sent: {send_count}, total received: {receive_count}")

    send_handler.stop()
    send_handler.close()
    pipeline.stop()
    print("PASS: Python custom module")


# ---------------------------------------------------------------------------
# 扇出 + 派生图示例：演示推荐的安全写法
#
# 拓扑 source -> [py_writer, py_reader] 同一帧同时进两个分支。
# 若写分支直接在 get_image() 返回的共享数组上原地改图，就会与并行的读分支产生数据竞争。
# 安全写法是：只在私有副本上修改，再经 set_model_input_image
# 发布为派生图——派生图是 collection 里一块独立缓冲，不与原图共享内存。
# 下游 C++ 推理前处理经 GetModelInputImage 读取这块派生图。
# ---------------------------------------------------------------------------

# 写分支在副本上填充的纯色方块（BGR），以及采样点
_EDIT_COLOR = (0, 255, 0)
_SAMPLE = (30, 30)  # (row, col)，落在方块内部
_BG_COLOR = (128, 64, 32)  # create_test_image 的背景色


class SafeDerivedWriter(vstream.Module):
    """扇出中的写分支：改私有副本 + 发布派生图，绝不原地修改共享原图。"""

    processed = 0

    def __init__(self, name: str):
        vstream.Module.__init__(self, name)

    def open(self, params: dict) -> bool:
        return True

    def close(self):
        pass

    def process(self, data) -> int:
        df = data.get_data_frame()
        if df is None:
            return 0
        img = df.get_image()          # 零拷贝读原图（共享内存，只读）
        edited = img.copy()           # 关键：私有副本，不碰共享内存
        cv2.rectangle(edited, (10, 10), (60, 60), _EDIT_COLOR, -1)

        # 发布为帧级派生图（独立缓冲）。若只想喂给某个模型，传 model_name="xxx"
        data.set_model_input_image(edited, model_name="")

        # 自证：派生图带上了修改，而原图保持干净
        derived = data.get_model_input_image(model_name="")
        assert tuple(int(v) for v in derived[_SAMPLE]) == _EDIT_COLOR
        assert tuple(int(v) for v in img[_SAMPLE]) == _BG_COLOR

        SafeDerivedWriter.processed += 1
        return 0

    def on_eos(self, stream_id: str):
        pass


class SafeOriginalReader(vstream.Module):
    """扇出中的读分支：并发读原图，统计是否被写分支污染（应恒为 0）。"""

    checked = 0
    violations = 0

    def __init__(self, name: str):
        vstream.Module.__init__(self, name)

    def open(self, params: dict) -> bool:
        return True

    def close(self):
        pass

    def process(self, data) -> int:
        df = data.get_data_frame()
        if df is None:
            return 0
        img = df.get_image()          # 读原图，与写分支并发
        px = tuple(int(v) for v in img[_SAMPLE])
        if px != _BG_COLOR:
            SafeOriginalReader.violations += 1
        SafeOriginalReader.checked += 1
        return 0

    def on_eos(self, stream_id: str):
        pass


def test_python_module_fanout_derived_image():
    print("=" * 60)
    print("TEST: Python module fan-out + derived image (safe write)")
    print("=" * 60)

    config = {
        "profiler_config": {"enable_profile": False},
        "source": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::DataSource",
            "next_modules": ["py_writer", "py_reader"],
            "custom_params": {"config_file": "data_source.json"}
        },
        "py_writer": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::PyModule",
            "next_modules": ["sink"],
            "custom_params": {"pyclass_name": "test_module.SafeDerivedWriter"}
        },
        "py_reader": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::PyModule",
            "next_modules": [],
            "custom_params": {"pyclass_name": "test_module.SafeOriginalReader"}
        },
        "sink": {
            "parallelism": 1,
            "max_input_queue_size": 20,
            "class_name": "cnstream::DataSink",
            "next_modules": [],
            "custom_params": {"config_file": "data_sink.json"}
        }
    }

    json_path = "pipeline_python_fanout.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    pipeline = vstream.Pipeline("py_fanout_pipeline")
    assert pipeline.build_pipeline_by_json_file(json_path), "Build pipeline failed"

    source = pipeline.get_data_source(source_module_name)
    assert source is not None
    send_handler = vstream.SendHandler(source, stream_id)
    sink = pipeline.get_data_sink(sink_module_name)
    assert sink is not None
    queue_handler = vstream.QueueHandler(sink, stream_id)

    assert pipeline.start(), "Pipeline start failed"
    assert source.add_source(send_handler) == 0
    assert sink.add_sink(queue_handler) == 0

    running = True
    send_count = 0
    receive_count = 0

    def send_thread():
        nonlocal send_count
        while running:
            send_handler.send(get_timestamp_ms(), str(send_count), create_test_image(480, 640))
            send_count += 1
            time.sleep(0.02)

    def receive_thread():
        nonlocal receive_count
        while running:
            ok, _ = queue_handler.get_data(wait_ms=10)
            if ok:
                receive_count += 1
            else:
                time.sleep(0.01)

    t_send = threading.Thread(target=send_thread)
    t_recv = threading.Thread(target=receive_thread)
    t_send.start()
    t_recv.start()
    time.sleep(5)
    running = False
    t_send.join()
    t_recv.join()

    send_handler.stop()
    send_handler.close()
    pipeline.stop()

    # C++ 侧 import 的是 test_module 模块（与 __main__ 是同一文件的两个模块对象），
    # 统计累加在该模块的类属性上，这里 import 回来读取（sys.modules 缓存，同一对象）
    import test_module as tm
    print(f"sent={send_count}, received={receive_count}, "
          f"writer_processed={tm.SafeDerivedWriter.processed}, "
          f"reader_checked={tm.SafeOriginalReader.checked}, "
          f"reader_violations={tm.SafeOriginalReader.violations}")

    assert tm.SafeDerivedWriter.processed > 0, "writer never ran"
    assert tm.SafeOriginalReader.checked > 0, "reader never ran"
    assert tm.SafeOriginalReader.violations == 0, "original image was polluted in fan-out!"
    print("PASS: fan-out safe write via derived image")


if __name__ == "__main__":
    test_python_module()
    test_python_module_fanout_derived_image()
