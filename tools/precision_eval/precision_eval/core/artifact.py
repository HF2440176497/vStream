# -*- coding: utf-8 -*-
"""通用张量产物 I/O —— 与模型类型无关。

"产物" = <case_id>.bin（raw float32）+ <case_id>.json
这是跨环境（本机 ↔ 部署容器）传递的唯一媒介，所以格式稳定且自描述

目录约定：
    corpus/         输入张量（评估起点，模型无关）
    ref/            ONNX 参考输出
    engine_<tag>/   引擎输出
"""

import hashlib
import json
import os
import time

import numpy as np

SCHEMA_VERSION = 1


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def artifact_path(dir_path, case_id, suffix=".bin"):
    return os.path.join(dir_path, case_id + suffix)


def write_artifact(out_dir, case_id, array, meta):
    """
    创建 <case_id>.bin + <case_id>.json，返回完整元信息
    """
    os.makedirs(out_dir, exist_ok=True)
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))

    bin_path = artifact_path(out_dir, case_id)
    arr.tofile(bin_path)

    m = dict(meta)
    m["schema_version"] = SCHEMA_VERSION
    m["case_id"] = case_id
    m["data_file"] = os.path.basename(bin_path)
    m["shape"] = list(arr.shape)
    m["dtype"] = "float32"
    m["num_elements"] = int(arr.size)
    m["sha256"] = sha256_file(bin_path)
    dump_json(artifact_path(out_dir, case_id, ".json"), m)
    return m


def read_artifact(dir_path, case_id, shape=None):
    """
    读回 <case_id>.bin + <case_id>.json，返回 (ndarray, meta)

    默认按 json 里记录的 shape 还原；传入 shape 可覆盖（用于比对形状不一致的情况）
    """
    meta = load_json(artifact_path(dir_path, case_id, ".json"))
    arr = np.fromfile(artifact_path(dir_path, case_id), dtype=np.float32)
    target = shape if shape is not None else meta.get("shape")
    if target:
        expected = int(np.prod(target))
        # arr.shape 实际类型
        if expected != arr.size:
            raise ValueError(
                "case %s 元素数不匹配：文件 %d, 元信息 %r 期望 %d"
                % (case_id, arr.size, target, expected)
            )
        arr = arr.reshape(target)
    return arr, meta


def list_cases(dir_path):
    """
    列出目录下同时具备 .bin 与 .json 的 case_id
    """
    if not os.path.isdir(dir_path):
        return []
    out = []
    for name in sorted(os.listdir(dir_path)):
        if not name.endswith(".bin"):
            continue
        cid = name[:-4]
        if os.path.exists(artifact_path(dir_path, cid, ".json")):
            out.append(cid)
    return out


def new_provenance(**kw):
    """
    构造统一的 provenance 字段，便于排查"结果是用哪个模型跑出来的"
    """
    p = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    p.update(kw)
    return p
