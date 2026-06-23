# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import threading
from datetime import datetime

import cv2

import vstream


stream_id_image_push = "channel-1"
stream_id_video_push = "channel-2"
stream_id_send_queue = "channel-4"

source_module_name = "source"
inference_module_name = "inference"
sink_module_name = "sink"

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def get_timestamp_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


def test_pipeline_basic():
    print("=" * 60)
    print("TEST: Pipeline basic interface")
    print("=" * 60)

    pipeline = vstream.Pipeline("pipeline")
    print(f"Pipeline name: {pipeline.get_name()}")
    assert pipeline.get_name() == "pipeline"

    # is_running 在未启动时应为 False
    assert not pipeline.is_running()


def test_send_queue_pipeline():
    """
    测试 send_queue_pipeline
    """
    print("=" * 60)
    print("TEST: Send-Queue pipeline")
    print("=" * 60)

    json_path = "pipeline.json"

    pipeline = vstream.Pipeline(f"pipeline_{stream_id_send_queue}")
    ok = pipeline.build_pipeline_by_json_file(json_path)
    assert ok, "Build pipeline failed"
    print("Pipeline built successfully")

    # 获取 DataSource 模块
    source = pipeline.get_data_source(source_module_name)
    assert source is not None
    print(f"Source module type: {type(source)}")

    sink = pipeline.get_data_sink(sink_module_name)
    assert sink is not None
    print(f"DataSink module type: {type(sink)}")

    send_handler = vstream.SendHandler(source, stream_id_send_queue)
    assert send_handler is not None
    print(f"SendHandler created, stream_id={send_handler.get_stream_id()}")

    # 创建 QueueHandler
    queue_handler = vstream.QueueHandler(sink, stream_id_send_queue)
    assert queue_handler is not None
    print(f"QueueHandler created, stream_id={queue_handler.get_stream_id()}")

    ok = pipeline.start()
    assert ok, "Pipeline start failed"
    print("Pipeline started")
    assert pipeline.is_running()

    ret = source.add_source(send_handler)
    assert ret == 0, f"AddSource failed, ret={ret}"
    print("SendHandler added to source module")

    ret = sink.add_sink(queue_handler)
    assert ret == 0, f"AddSink failed, ret={ret}"
    print("QueueHandler added to sink module")

    send_image = cv2.imread("image.png")

    running = True
    send_count = 0
    receive_count = 0

    def send_thread():
        nonlocal send_count
        while running:
            pts = get_timestamp_ms()
            frame_id_s = str(send_count)
            ok = send_handler.send(pts, frame_id_s, send_image)
            if ok != 0:
                print(f"Warning: send returned {ok}")
            send_count += 1
            time.sleep(0.02)  # 50 fps

    def receive_thread():
        nonlocal receive_count
        while running:
            # get_data 返回 (ok, data)
            ok, data = queue_handler.get_data(wait_ms=10)
            if not ok:
                time.sleep(0.01)
                continue
            receive_count += 1
            if receive_count % 10 == 0:
                print(f'receive {data}')
                print(f"Received {receive_count} frames, send_count: {send_count}")

                # 检查并保存 data 顶层成员（防御式获取）
                data_members = [m for m in dir(data) if not m.startswith('_')]
                print(f"data members: {data_members}")

                result = getattr(data, 'result', None)
                timestamp = getattr(data, 'timestamp', None)
                frame_id_s = getattr(data, 'frame_id_s', None)
                objects_json = getattr(data, 'objects_json', None)
                objects = getattr(data, 'objects', [])
                print(f"  result={result}, timestamp={timestamp}, frame_id_s={frame_id_s}")
                print(f"  objects count={len(objects)}, objects_json={objects_json}")

                # 检查并保存每个 obj 的成员（防御式获取）
                for idx, obj in enumerate(objects):
                    obj_members = [m for m in dir(obj) if not m.startswith('_')]
                    print(f"  obj[{idx}] members: {obj_members}")

                    obj_id = getattr(obj, 'id', -1)
                    obj_score = getattr(obj, 'score', 0.0)
                    obj_type = getattr(obj, 'type', '')
                    obj_bboxs = getattr(obj, 'bboxs', [])
                    obj_classes = getattr(obj, 'classes', [])
                    obj_attributes = getattr(obj, 'attributes', [])

                    print(f"    id={obj_id}, score={obj_score}, type={obj_type}")
                    print(f"    bboxs={obj_bboxs}")
                    print(f"    classes={obj_classes}")
                    print(f"    attributes={obj_attributes}")

                    # 遍历 attributes，保存 OCR 等识别结果
                    for attr_key, attr in obj_attributes:
                        attr_name = getattr(attr, 'name', '')
                        attr_score = getattr(attr, 'score', 0.0)
                        attr_id = getattr(attr, 'id', -1)
                        attr_value = getattr(attr, 'value', -1)
                        print(f"    attr key={attr_key}, name={attr_name}, "
                              f"score={attr_score}, id={attr_id}, value={attr_value}")

    t_send = threading.Thread(target=send_thread)
    t_recv = threading.Thread(target=receive_thread)
    t_send.start()
    t_recv.start()

    time.sleep(10)

    running = False
    t_send.join()
    t_recv.join()

    print(f"Total sent: {send_count}, total received: {receive_count}")

    pipeline.stop()
    print("Pipeline stopped")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("vstream Python binding validation script")
    print(f"Python version: {sys.version}")
    print()

    test_pipeline_basic()
    print()

    try:
        test_send_queue_pipeline()
    except Exception as e:
        print(f"SKIPPED/FAILED test_send_queue_pipeline: {e}")
    print()

    print("=" * 60)
    print("All tests finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
