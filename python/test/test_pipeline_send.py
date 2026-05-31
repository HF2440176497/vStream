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

key_source = "source"
key_inference = "inference"
key_sink = "sink"

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
    source = pipeline.get_data_source(key_source)
    assert source is not None
    print(f"Source module type: {type(source)}")

    sink = pipeline.get_data_sink(key_sink)
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
            if receive_count % 20 == 0:
                print(f"Received {receive_count} frames, latest frame_id_s={data.frame_id_s}")

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
