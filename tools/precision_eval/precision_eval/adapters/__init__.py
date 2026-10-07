# -*- coding: utf-8 -*-
"""适配器注册表。

单一事实来源 ../model_types.json

对外接口：
    list_adapters()            列出全部模型类型（含 steps）
    get_adapter(name, params)  按名实例化适配器
"""

import importlib
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)
REGISTRY_PATH = os.path.join(_PKG_DIR, "model_types.json")


def load_registry(path=None):
    p = path or REGISTRY_PATH
    if not os.path.exists(p):
        raise IOError("找不到模型注册表: %s" % p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "adapters" not in data:
        raise ValueError("注册表缺少 'adapters' 字段: %s" % p)
    return data


def list_adapters(path=None):
    """
    返回 [{'name','description','steps','required_inputs','models'}...]，按名字排序。
    """
    data = load_registry(path)
    models = data.get("models", {})
    out = []
    for name, entry in sorted(data["adapters"].items()):
        item = {
            "name": name,
            "description": entry.get("description", ""),
            "steps": entry.get("steps", []),
            "required_inputs": entry.get("required_inputs", []),
            "module": entry.get("module", ""),
            "class": entry.get("class", ""),
            "models": sorted([m for m, v in models.items() if v.get("adapter") == name]),
        }
        out.append(item)
    return out


def get_adapter(name, params=None, path=None):
    """
    按名字实例化适配器
    """
    data = load_registry(path)
    entry = data["adapters"].get(name)
    if entry is None:
        avail = ", ".join(sorted(data["adapters"].keys()))
        raise KeyError("未知的模型类型 '%s'。可用: %s" % (name, avail))

    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])
    return cls(params or {})


def steps_for(name, path=None):
    """
    返回适配器需要跑的步骤
    """
    data = load_registry(path)
    entry = data["adapters"].get(name)
    if entry is None:
        return []
    return entry.get("steps", [])
