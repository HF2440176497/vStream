# -*- coding: utf-8 -*-
"""
通用数值指标
"""

import numpy as np


def numeric_compare(ref, eng):
    """通用数值层对比。

    返回 dict，全部为 python 原生类型（便于直接 json 序列化）。
    """
    ref = np.asarray(ref, dtype=np.float32)
    eng = np.asarray(eng, dtype=np.float32)
    if ref.shape != eng.shape:
        return {
            "shape_match": False,
            "ref_shape": list(ref.shape),
            "eng_shape": list(eng.shape),
        }

    d = eng.astype(np.float64) - ref.astype(np.float64)
    abs_d = np.abs(d)
    ref_norm = float(np.linalg.norm(ref.astype(np.float64)))
    eng_norm = float(np.linalg.norm(eng.astype(np.float64)))

    denom = ref_norm * eng_norm
    cosine = float(np.dot(ref.ravel().astype(np.float64),
                          eng.ravel().astype(np.float64)) / denom) if denom > 0 else float("nan")

    return {
        "shape_match": True,
        "numel": int(ref.size),
        "max_abs": float(abs_d.max()),
        "mean_abs": float(abs_d.mean()),
        "rms": float(np.sqrt(np.mean(d ** 2))),
        "rel_l2": float(np.linalg.norm(d) / (ref_norm + 1e-12)),
        "cosine": cosine,
        "ref_absmax": float(np.abs(ref).max()),
        "eng_absmax": float(np.abs(eng).max()),
    }


def topk_stats(logits):
    """
    通用 logits 统计：top1 / top1 值 / top2 值 / margin

    对任意分类式输出都适用
    logits: [T, C] 或 [1, T, C]
    """
    a = np.asarray(logits, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError("expect logits rank 2 or 3, got %r" % (a.shape,))

    top1 = a.argmax(axis=1)
    top1_val = a.max(axis=1)
    if a.shape[1] > 1:
        top2_val = np.partition(a, -2, axis=1)[:, -2]
    else:
        top2_val = np.full(a.shape[0], -np.inf, dtype=np.float32)
    return top1, top1_val, top2_val, (top1_val - top2_val)


def format_sci(x):
    return "%.3e" % x
