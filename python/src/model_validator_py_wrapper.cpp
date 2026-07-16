/*************************************************************************
 * Copyright (C) [2024] by vStream. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *************************************************************************/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "common_wrapper.hpp"
#include "model_validator.hpp"

namespace py = pybind11;

namespace cnstream {

/* ----------------------------------------------------------------------- *
 *  Struct bindings
 * ----------------------------------------------------------------------- */

static void BindModelTensorInfo(const py::module& m) {
  py::class_<ModelTensorInfo>(m, "ModelTensorInfo")
      .def_readwrite("name", &ModelTensorInfo::name)
      .def_readwrite("shape", &ModelTensorInfo::shape)
      .def_readwrite("dtype", &ModelTensorInfo::dtype)
      .def("__repr__", [](const ModelTensorInfo& t) {
        std::string s = "ModelTensorInfo(name='" + t.name + "', shape=[";
        for (size_t i = 0; i < t.shape.size(); ++i) {
          if (i) s += ",";
          s += std::to_string(t.shape[i]);
        }
        s += "], dtype='" + t.dtype + "')";
        return s;
      });
}

static void BindModelInfo(const py::module& m) {
  py::class_<ModelInfo>(m, "ModelInfo")
      .def_readwrite("inputs", &ModelInfo::inputs)
      .def_readwrite("outputs", &ModelInfo::outputs)
      .def_readwrite("batch_size", &ModelInfo::batch_size)
      .def_readwrite("channel", &ModelInfo::channel)
      .def_readwrite("height", &ModelInfo::height)
      .def_readwrite("width", &ModelInfo::width)
      .def_readwrite("device_type", &ModelInfo::device_type)
      .def_readwrite("device_id", &ModelInfo::device_id)
      .def_readwrite("model_path", &ModelInfo::model_path)
      .def("__repr__", [](const ModelInfo& info) {
        return "ModelInfo(model='" + info.model_path + "', device=" + info.device_type +
               ", batch=" + std::to_string(info.batch_size) +
               ", input=" + std::to_string(info.width) + "x" +
               std::to_string(info.height) + "x" + std::to_string(info.channel) +
               ", inputs=" + std::to_string(info.inputs.size()) +
               ", outputs=" + std::to_string(info.outputs.size()) + ")";
      });
}

static void BindValidatorDetection(const py::module& m) {
  py::class_<ValidatorDetection>(m, "ValidatorDetection")
      .def_readwrite("class_id", &ValidatorDetection::class_id)
      .def_readwrite("class_name", &ValidatorDetection::class_name)
      .def_readwrite("score", &ValidatorDetection::score)
      .def_readwrite("x", &ValidatorDetection::x)
      .def_readwrite("y", &ValidatorDetection::y)
      .def_readwrite("w", &ValidatorDetection::w)
      .def_readwrite("h", &ValidatorDetection::h)
      .def("__repr__", [](const ValidatorDetection& d) {
        return "Detection(class=" + std::to_string(d.class_id) +
               ", name='" + d.class_name + "'" +
               ", score=" + std::to_string(d.score) +
               ", bbox=[" + std::to_string(d.x) + "," +
               std::to_string(d.y) + "," + std::to_string(d.w) + "," +
               std::to_string(d.h) + "])";
      });
}

static void BindE2EResult(const py::module& m) {
  py::class_<E2EResult>(m, "E2EResult")
      .def_readwrite("detections", &E2EResult::detections)
      .def_readwrite("latency_ms", &E2EResult::latency_ms)
      .def_readwrite("error", &E2EResult::error)
      .def("__repr__", [](const E2EResult& r) {
        if (!r.error.empty()) {
          return "E2EResult(error='" + r.error + "')";
        }
        return "E2EResult(detections=" + std::to_string(r.detections.size()) +
               ", latency_ms=" + std::to_string(r.latency_ms) + ")";
      });
}

static void BindBenchmarkResult(const py::module& m) {
  py::class_<BenchmarkResult>(m, "BenchmarkResult")
      .def_readwrite("batch_size", &BenchmarkResult::batch_size)
      .def_readwrite("avg_ms", &BenchmarkResult::avg_ms)
      .def_readwrite("min_ms", &BenchmarkResult::min_ms)
      .def_readwrite("max_ms", &BenchmarkResult::max_ms)
      .def_readwrite("p99_ms", &BenchmarkResult::p99_ms)
      .def_readwrite("fps", &BenchmarkResult::fps)
      .def_readwrite("error_count", &BenchmarkResult::error_count)
      .def("__repr__", [](const BenchmarkResult& r) {
        return "BenchmarkResult(bs=" + std::to_string(r.batch_size) +
               ", avg=" + std::to_string(r.avg_ms) + "ms" +
               ", p99=" + std::to_string(r.p99_ms) + "ms" +
               ", fps=" + std::to_string(r.fps) +
               ", errors=" + std::to_string(r.error_count) + ")";
      });
}

/* ----------------------------------------------------------------------- *
 *  ModelValidator class binding
 * ----------------------------------------------------------------------- */

void ModelValidatorWrapper(const py::module& m) {
  // Bind structs first
  BindModelTensorInfo(m);
  BindModelInfo(m);
  BindValidatorDetection(m);
  BindE2EResult(m);
  BindBenchmarkResult(m);

  py::class_<ModelValidator, std::shared_ptr<ModelValidator>>(m, "ModelValidator",
      R"doc(
Standalone model validation tool — load model, run inference, test
preproc/postproc, benchmark performance, all without a Pipeline.

Usage:
    validator = vstream.ModelValidator("/path/to/model.engine", "cuda", 0)
    if not validator.load():
        print("Load failed")
        return
    info = validator.get_model_info()
    print(info)
    result = validator.run_e2e(image, "Pre_YOLO_CPU_v2", "Post_YOLOv8_CPU_v2",
                                postproc_params={"config_file": "yolo_coco.json"})
    for det in result.detections:
        print(det)
)doc")
      .def(py::init([](const std::string& model_path,
                       const std::string& device_type,
                       int device_id,
                       int input_ordered_index) {
             return std::make_shared<ModelValidator>(
                 model_path, device_type, device_id, input_ordered_index);
           }),
           py::arg("model_path"),
           py::arg("device_type") = "cpu",
           py::arg("device_id") = 0,
           py::arg("input_ordered_index") = 0)

      .def("load", [](ModelValidator& self) -> bool {
        py::gil_scoped_release release;
        return self.Load();
      }, "Load model and allocate buffers. Returns False on failure.")

      .def("is_loaded", &ModelValidator::IsLoaded,
           "Whether model is loaded and valid.")

      .def("get_model_info", &ModelValidator::GetModelInfo,
           "Get model metadata (shapes, dtypes, names).")

      /* ---- raw tensor inference ---- */
      .def("infer", [](ModelValidator& self,
                       std::vector<py::array_t<float, py::array::c_style>> inputs)
                        -> std::vector<py::array_t<float>> {
        // Convert numpy arrays to vector<vector<float>>
        std::vector<std::vector<float>> cpp_inputs;
        cpp_inputs.reserve(inputs.size());
        for (auto& arr : inputs) {
          py::buffer_info buf = arr.request();
          float* ptr = static_cast<float*>(buf.ptr);
          cpp_inputs.emplace_back(ptr, ptr + buf.size);
        }
        // Release GIL during inference
        py::gil_scoped_release release;
        auto outputs = self.Infer(cpp_inputs);
        py::gil_scoped_acquire acquire;

        // 值拷贝返回
        std::vector<py::array_t<float>> result;
        result.reserve(outputs.size());
        for (auto& out : outputs) {
          result.push_back(py::array_t<float>(out.size(), out.data()));
        }
        return result;
      }, py::arg("inputs"),
         R"doc(Raw tensor inference — no preproc/postproc.

Args:
    inputs: List of numpy float32 arrays, one per input tensor.
            Each array's size must match InputShape(i).DataCount().

Returns:
    List of numpy float32 arrays, one per output tensor.
    Empty list if model not loaded or inference failed.
)doc")

      .def("run_e2e", [](ModelValidator& self,
                         py::array_t<uint8_t> image,
                         const std::string& preproc_name,
                         const std::string& postproc_name,
                         const std::map<std::string, std::string>& preproc_params,
                         const std::map<std::string, std::string>& postproc_params)
                          -> E2EResult {
        cv::Mat mat = ArrayToMat(image);
        // Release GIL during E2E
        py::gil_scoped_release release;
        return self.RunE2E(mat, preproc_name, postproc_name,
                           preproc_params, postproc_params);
      }, py::arg("image"),
         py::arg("preproc_name"),
         py::arg("postproc_name"),
         py::arg("preproc_params") = std::map<std::string, std::string>{},
         py::arg("postproc_params") = std::map<std::string, std::string>{},
         R"doc(End-to-end: image -> preproc -> infer -> postproc -> detections.

Args:
    image: BGR image as numpy uint8 array (H, W, 3).
    preproc_name: Registered preproc class name (e.g. "Pre_YOLO_CPU_v2").
    postproc_name: Registered postproc class name (e.g. "Post_YOLOv8_CPU_v2").
    preproc_params: Custom params for Preproc::Init.
    postproc_params: Custom params for Postproc::Init
                     (e.g. {"config_file": "yolo_coco.json"}).

Returns:
    E2EResult with detections, latency_ms, and optional error string.
)doc")

      .def("benchmark", [](ModelValidator& self,
                            py::array_t<uint8_t> image,
                            const std::string& preproc_name,
                            const std::string& postproc_name,
                            const std::map<std::string, std::string>& preproc_params,
                            const std::map<std::string, std::string>& postproc_params,
                            int warmup_runs,
                            int test_runs,
                            std::vector<int> batch_sizes)
                             -> std::vector<BenchmarkResult> {
        cv::Mat mat = ArrayToMat(image);
        py::gil_scoped_release release;
        return self.Benchmark(mat, preproc_name, postproc_name,
                              preproc_params, postproc_params,
                              warmup_runs, test_runs, batch_sizes);
      }, py::arg("image"),
         py::arg("preproc_name"),
         py::arg("postproc_name"),
         py::arg("preproc_params") = std::map<std::string, std::string>{},
         py::arg("postproc_params") = std::map<std::string, std::string>{},
         py::arg("warmup_runs") = 10,
         py::arg("test_runs") = 100,
         py::arg("batch_sizes") = std::vector<int>{1},
         R"doc(Benchmark E2E latency over multiple runs.

Args:
    image: BGR image as numpy uint8 array (H, W, 3).
    preproc_name: Registered preproc class name.
    postproc_name: Registered postproc class name.
    preproc_params: Custom params for Preproc::Init.
    postproc_params: Custom params for Postproc::Init.
    warmup_runs: Number of warmup iterations (not measured). Default 10.
    test_runs: Number of measured iterations. Default 100.
    batch_sizes: Batch sizes to test. Currently only [1] supported.

Returns:
    List of BenchmarkResult, one per batch size.
)doc");
}

}  // namespace cnstream
