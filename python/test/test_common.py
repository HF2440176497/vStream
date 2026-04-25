# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import threading
from datetime import datetime

import numpy as np

# 手动添加路径，例如：
sys.path.insert(0, "../../lib")
import vstream

def test_data_structures():
    print("=" * 60)
    print("TEST: Data structures")
    print("=" * 60)

    # FrameInfo
    frame_info = vstream.FrameInfo("channel-1", eos=False)
    assert frame_info.stream_id == "channel-1"
    assert not frame_info.is_eos()
    print("FrameInfo OK")

    # DataFrame
    data_frame = vstream.DataFrame()
    assert data_frame.get_planes() == 0
    print("DataFrame OK")

    # InferObjs / InferObject / InferBoundingBox / InferAttr
    infer_objs = vstream.InferObjs()
    obj = vstream.InferObject()
    obj.id = 1
    obj.track_id = 100
    obj.score = 0.95
    obj.bbox = vstream.InferBoundingBox(10.0, 20.0, 30.0, 40.0)
    infer_objs.push_back(obj)
    assert len(infer_objs.objs) == 1
    print("InferObjs OK")

    # class_infos / obj_in / output_data (OSD 相关结构)
    cls = vstream.class_infos()
    cls.id = 0
    cls.model_name = "model"
    cls.id_name = "person"
    cls.score = 0.9
    cls.value = 0.85
    print(f"class_infos: id={cls.id}, name={cls.id_name}")

    obj_in = vstream.obj_in()
    obj_in.track_id = "100"
    obj_in.score = 0.88
    obj_in.bboxs = [10, 20, 30, 40]
    obj_in.classes = [cls]
    print(f"obj_in: track_id={obj_in.track_id}, bboxs={obj_in.bboxs}")

    out_data = vstream.output_data()
    out_data.result = 0
    out_data.timestamp = get_timestamp_ms()
    out_data.frame_id_s = "channel-1"
    out_data.objects = [obj_in]
    print(f"output_data: result={out_data.result}, frame_id_s={out_data.frame_id_s}")

    # SendFrame
    img = create_test_image(64, 64)
    send_frame = vstream.SendFrame(pts=200, frame_id_s="200", image=img)
    assert send_frame.pts == 200
    assert send_frame.frame_id_s == "200"
    assert send_frame.image.shape == (64, 64, 3)
    print("SendFrame OK")

    print("PASS: Data structures")


if __name__ == "__main__":
    test_data_structures()
    