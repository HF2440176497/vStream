# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import threading
from datetime import datetime

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


# ---------------------------------------------------------------------------
# 1. image push
# ---------------------------------------------------------------------------
def test_image_push_pipeline():
    print("=" * 60)
    print("TEST: Image push pipeline")
    print("=" * 60)

    json_path = "pipeline2.json"

    pipeline = vstream.Pipeline(f"pipeline_{stream_id_image_push}")
    ok = pipeline.build_pipeline_by_json_file(json_path)
    assert ok, "Build pipeline failed"
    print("Pipeline built successfully")

    # 获取 DataSource 模块
    source = pipeline.get_data_source(source_module_name)
    assert source is not None, f"get_data_source('{source_module_name}') returned None"
    print(f"Source module type: {type(source)}")

    sink = pipeline.get_data_sink(sink_module_name)
    assert sink is not None, f"get_data_sink('{sink_module_name}') returned None"
    print(f"Sink module type: {type(sink)}")

    # 创建 ImageHandler
    image_handler = vstream.ImageHandler(source, stream_id_image_push)
    assert image_handler is not None
    print(f"ImageHandler created, stream_id={image_handler.get_stream_id()}")

    push_handler = vstream.PushHandler(sink, stream_id_image_push)
    assert push_handler is not None
    print(f"PushHandler created, stream_id={push_handler.get_stream_id()}")

    ok = pipeline.start()
    assert ok, "Pipeline start failed"
    print("Pipeline started")
    assert pipeline.is_running()

    ret = source.add_source(image_handler)
    assert ret == 0, f"AddSource failed, ret={ret}"
    print("ImageHandler added to source module")

    ret = sink.add_sink(push_handler)
    assert ret == 0, f"AddSink failed, ret={ret}"
    print("PushHandler added to sink module")

    # 运行一段时间
    time.sleep(60)

    # 停止流水线
    pipeline.stop()
    print("Pipeline stopped")
    print("PASS: Image push pipeline")


# ---------------------------------------------------------------------------
# 2. video push + queue
# ---------------------------------------------------------------------------
def test_video_push_pipeline():
    print("=" * 60)
    print("TEST: Video push pipeline")
    print("=" * 60)

    json_path = "pipeline.json"

    pipeline = vstream.Pipeline(f"pipeline_{stream_id_video_push}")
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

    # 创建 PullHandler
    pull_handler = vstream.PullHandler(source, stream_id_video_push)
    assert pull_handler is not None
    print(f"PullHandler created, stream_id={pull_handler.get_stream_id()}")

    push_handler = vstream.PushHandler(sink, stream_id_video_push)
    assert push_handler is not None
    print(f"PushHandler created, stream_id={push_handler.get_stream_id()}")

    ok = pipeline.start()
    assert ok, "Pipeline start failed"
    print("Pipeline started")
    assert pipeline.is_running()

    # 添加源流并启动流水线
    ret = source.add_source(pull_handler)
    assert ret == 0, f"AddSource failed, ret={ret}"
    print("PullHandler added to source module")

    ret = sink.add_sink(push_handler)
    assert ret == 0, f"AddSink failed, ret={ret}"
    print("PushHandler added to sink module")

    time.sleep(180)

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

    # try:
    #     test_image_push_pipeline()
    # except Exception as e:
    #     print(f"SKIPPED/FAILED test_image_push_pipeline: {e}")
    # print()

    try:
        test_video_push_pipeline()
    except Exception as e:
        print(f"SKIPPED/FAILED test_video_push_pipeline: {e}")
    print()

    print("=" * 60)
    print("All tests finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
