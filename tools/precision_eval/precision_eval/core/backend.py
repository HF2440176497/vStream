# -*- coding: utf-8 -*-
"""推理后端 —— 与模型类型无关。

两个后端，接口一致：
    OnnxRuntimeBackend  本机可用（只要装了 onnxruntime）
    TrtBackend          部署容器可用（走 vStream 的 ModelValidator）

统一约定：run(array_nd) -> ndarray
  - 输入是已按正确 shape 组织好的 ndarray
  - 输出是模型的第 0 个输出张量

这里刻意不引入任何 OCR / 检测相关的概念。
"""

import numpy as np


class Backend:
    name = "base"

    def input_shape(self):
        raise NotImplementedError

    def output_shape(self):
        raise NotImplementedError

    def run(self, array_nd):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class OnnxRuntimeBackend(Backend):

    name = "onnxruntime"

    def __init__(self, model_path, input_index=0, output_index=0, providers=None, verbose=False):
        import onnxruntime as ort

        self.model_path = model_path
        self._input_index = input_index
        self._output_index = output_index
        so = ort.SessionOptions()
        if not verbose:
            so.log_severity_level = 3
        if providers:
            self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        else:
            self.sess = ort.InferenceSession(model_path, sess_options=so,
                                             providers=["CPUExecutionProvider"])

        self._inputs = self.sess.get_inputs()
        self._outputs = self.sess.get_outputs()
        self._in_name = self._inputs[self._input_index].name
        self._in_shape = list(self._inputs[self._input_index].shape)
        self._out_shape = list(self._outputs[self._output_index].shape)

    def input_shape(self):
        return self._in_shape

    def output_shape(self):
        return self._out_shape

    def run(self, array_nd):
        arr = np.ascontiguousarray(array_nd, dtype=np.float32)
        outs = self.sess.run(None, {self._in_name: arr})
        return np.asarray(outs[self._output_index], dtype=np.float32)

    def providers(self):
        return list(self.sess.get_providers())

    def describe(self):
        return {
            "backend": self.name,
            "model_path": self.model_path,
            "providers": self.providers(),
            "input_names": [i.name for i in self._inputs],
            "input_shapes": [list(i.shape) for i in self._inputs],
            "output_names": [o.name for o in self._outputs],
            "output_shapes": [list(o.shape) for o in self._outputs],
        }


class TrtBackend(Backend):
    """
    TensorRT 引擎后端
    """

    name = "tensorrt"

    def __init__(self, engine_path, device="cuda", device_id=0, input_index=0):
        try:
            import libs.vstream as vstream
        except ImportError as e:
            raise ImportError(
                "无法 import vstream。请确认 vStream 以 VSTREAM_BUILD_PYTHON_API=ON 构建，"
                "且 PYTHONPATH 指向绑定所在目录。原始错误: %s" % e
            )

        self.engine_path = engine_path
        self.device = device
        self.device_id = device_id
        self._v = vstream.ModelValidator(engine_path, device, device_id, input_index)
        if not self._v.load():
            raise RuntimeError("引擎加载失败: %s" % engine_path)

        info = self._v.get_model_info()
        self._info = info
        self._in_shape = list(info.inputs[input_index].shape)
        self._out_shape = list(info.outputs[0].shape)
        self._input_index = input_index

    def input_shape(self):
        return self._in_shape

    def output_shape(self):
        return self._out_shape

    def run(self, array_nd):
        arr = np.ascontiguousarray(array_nd, dtype=np.float32)
        outs = self._v.infer([arr.ravel()])
        if not outs:
            raise RuntimeError("ModelValidator.infer 返回空结果")
        return np.asarray(outs[0], dtype=np.float32).reshape(self._out_shape)

    def describe(self):
        info = self._info
        return {
            "backend": self.name,
            "engine_path": self.engine_path,
            "device": info.device_type,
            "device_id": info.device_id,
            "batch_size": info.batch_size,
            "input_names": [t.name for t in info.inputs],
            "input_shapes": [t.shape for t in info.inputs],
            "output_names": [t.name for t in info.outputs],
            "output_shapes": [t.shape for t in info.outputs],
        }
