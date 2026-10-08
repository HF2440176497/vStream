# -*- coding: utf-8 -*-
"""
precision_eval —— 模型部署精度评估

分层：
  core/      通用层：张量产物 I/O、数值指标、推理后端。与模型类型完全无关。
  adapters/  模型特定层：预处理、解码、决策层指标。按模型类型插拔。

新增一个模型类型 = 新增一个 adapter + 在 model_types.yaml 登记，core 不动。
"""

__version__ = "0.2.0"
