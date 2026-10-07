# -*- coding: utf-8 -*-
"""
通用适配器

只提供数值层对比
"""

from .base import ModelAdapter


class GenericAdapter(ModelAdapter):
    name = "generic"
    description = "任意模型：只做输入张量与数值层对比，不做任何解码。"
    required_inputs = ["onnx", "engine"]

    provides_preprocess = False
    provides_decision = False
