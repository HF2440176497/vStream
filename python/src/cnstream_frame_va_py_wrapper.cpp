/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common_wrapper.hpp"
#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "data_source_param.hpp"

namespace py = pybind11;

namespace cnstream {

extern std::shared_ptr<py::class_<FrameInfo, std::shared_ptr<FrameInfo>>> g_py_frame_info;

std::shared_ptr<DataFrame> GetDataFrame(std::shared_ptr<FrameInfo> frame) {
  if (!frame->collection.HasValue(kDataFrameTag)) {
    return nullptr;
  }
  return frame->collection.Get<std::shared_ptr<DataFrame>>(kDataFrameTag);
}

std::shared_ptr<InferObjs> GetInferObjects(std::shared_ptr<FrameInfo> frame) {
  if (!frame->collection.HasValue(kInferObjsTag)) {
    return nullptr;
  }
  return frame->collection.Get<std::shared_ptr<InferObjs>>(kInferObjsTag);
}

/**
 * @brief DataFrame wrapper.
 */
void DataFrameWrapper(const py::module &m) {
  py::class_<DataFrame, std::shared_ptr<DataFrame>>(m, "DataFrame")
    .def(py::init([]() {
      return std::make_shared<DataFrame>();
    }))
    .def("get_planes", &DataFrame::GetPlanes)
    .def("get_plane_bytes", &DataFrame::GetPlaneBytes)
    .def("get_bytes", &DataFrame::GetBytes)
    .def("get_image", [](std::shared_ptr<DataFrame> data_frame) {
      cv::Mat image = data_frame->GetImage();  // BGR format
      return MatToArray(image);
    })
    .def("has_image", &DataFrame::HasImage)
    .def("data", [](const DataFrame& data_frame, int plane_idx) {
        int planes = data_frame.GetPlanes();
        if (plane_idx < 0 || plane_idx >= planes) {
          throw py::index_error("plane_idx out of range: " + std::to_string(plane_idx) +
                                ", planes=" + std::to_string(planes));
        }
        const auto& mem = data_frame.data_[plane_idx];
        if (!mem) {
          throw py::value_error("plane " + std::to_string(plane_idx) + " not allocated");
        }
        const void* ptr = mem->GetCpuData();
        size_t size = mem->GetSize();
        if (!ptr || size == 0) {
          return py::array_t<uint8_t>();
        }
        // 返回该平面的 CPU 字节（一维只读视图）；reference_internal 保证数组存活期间 DataFrame 不被释放
        return py::array_t<uint8_t>(
            {static_cast<py::ssize_t>(size)},
            {static_cast<py::ssize_t>(sizeof(uint8_t))},
            static_cast<const uint8_t*>(ptr));
    }, py::return_value_policy::reference_internal)
    .def("get_frame_id", &DataFrame::GetFrameId)  // uint64_t
    .def("get_fmt", &DataFrame::GetFmt)
    .def("get_width", &DataFrame::GetWidth)
    .def("get_height", &DataFrame::GetHeight);

  py::enum_<DataFormat>(m, "DataFormat")
    .value("INVALID", DataFormat::INVALID)
    .value("PIXEL_FORMAT_YUV420_NV21", DataFormat::PIXEL_FORMAT_YUV420_NV21)
    .value("PIXEL_FORMAT_YUV420_NV12", DataFormat::PIXEL_FORMAT_YUV420_NV12)
    .value("PIXEL_FORMAT_BGR24", DataFormat::PIXEL_FORMAT_BGR24)
    .value("PIXEL_FORMAT_RGB24", DataFormat::PIXEL_FORMAT_RGB24)
    .value("PIXEL_FORMAT_ARGB32", DataFormat::PIXEL_FORMAT_ARGB32)
    .value("PIXEL_FORMAT_ABGR32", DataFormat::PIXEL_FORMAT_ABGR32)
    .value("PIXEL_FORMAT_RGBA32", DataFormat::PIXEL_FORMAT_RGBA32)
    .value("PIXEL_FORMAT_BGRA32", DataFormat::PIXEL_FORMAT_BGRA32);

  py::enum_<DevType>(m, "DevType")
    .value("INVALID", DevType::INVALID)
    .value("CPU", DevType::CPU)
    .value("CUDA", DevType::CUDA)
    .value("ROCKCHIP", DevType::ROCKCHIP);

  py::class_<DevContext>(m, "DevContext")
    .def(py::init())
    .def_readwrite("device_type", &DevContext::device_type)
    .def_readwrite("device_id", &DevContext::device_id);
}

void InferObjsWrapper(const py::module &m) {
  py::class_<InferObjs, std::shared_ptr<InferObjs>>(m, "InferObjs")
      .def(py::init([]() {
        return std::make_shared<InferObjs>();
      }))
      .def_property("objs", [](std::shared_ptr<InferObjs> objs_holder) {
          std::lock_guard<std::mutex> lck(objs_holder->mutex_);
          return objs_holder->objs_;
      }, [](std::shared_ptr<InferObjs> objs_holder, std::vector<std::shared_ptr<InferObject>> objs) {
          std::lock_guard<std::mutex> lck(objs_holder->mutex_);
          objs_holder->objs_ = objs;
      })
      .def("push_back", [](std::shared_ptr<InferObjs> objs_holder, std::shared_ptr<InferObject> obj) {
          std::lock_guard<std::mutex> lck(objs_holder->mutex_);
          objs_holder->objs_.push_back(obj);
      });


  py::class_<InferBoundingBox, std::shared_ptr<InferBoundingBox>>(m, "InferBoundingBox")
      .def(py::init([]() {
        return std::make_shared<InferBoundingBox>();
      }))
      .def(py::init([](float x, float y, float w, float h) {
        auto bbox = std::make_shared<InferBoundingBox>();
        bbox->x = x;
        bbox->y = y;
        bbox->w = w;
        bbox->h = h;
        return bbox;
      }))
      .def_readwrite("x", &InferBoundingBox::x)
      .def_readwrite("y", &InferBoundingBox::y)
      .def_readwrite("w", &InferBoundingBox::w)
      .def_readwrite("h", &InferBoundingBox::h);


  py::class_<InferAttr, std::shared_ptr<InferAttr>>(m, "InferAttr")
      .def(py::init([]() {
        return std::make_shared<InferAttr>();
      }))
      .def(py::init([](int id, int value, float score) {
        auto attr = std::make_shared<InferAttr>();
        attr->id = id;
        attr->value = value;
        attr->score = score;
        return attr;
      }))
      .def_readwrite("id", &InferAttr::id)
      .def_readwrite("value", &InferAttr::value)
      .def_readwrite("score", &InferAttr::score)
      .def_readwrite("name", &InferAttr::name);


  py::class_<InferObjectInfo, std::shared_ptr<InferObjectInfo>>(m, "InferObjectInfo")
      .def(py::init([]() {
        return std::make_shared<InferObjectInfo>();
      }))
      .def_readwrite("id", &InferObjectInfo::id)
      .def_readwrite("model_name", &InferObjectInfo::model_name)
      .def_readwrite("name", &InferObjectInfo::name)
      .def_readwrite("score", &InferObjectInfo::score)
      .def_readwrite("value", &InferObjectInfo::value);


  py::class_<InferObject, std::shared_ptr<InferObject>>(m, "InferObject")
      .def(py::init([]() {
        return std::make_shared<InferObject>();
      }))
      .def_readwrite("id", &InferObject::id)
      .def_readwrite("track_id", &InferObject::track_id)
      .def_readwrite("score", &InferObject::score)
      .def_readwrite("bbox", &InferObject::bbox)
      .def_readwrite("model_name", &InferObject::model_name)
      .def_readwrite("name", &InferObject::name)
      .def_readwrite("area", &InferObject::area)
      .def_readwrite("classes", &InferObject::classes)
      .def_readwrite("key_points", &InferObject::key_points)
      .def("add_attribute", [](std::shared_ptr<InferObject> obj, const std::string& key, const InferAttr& attr) {
        obj->AddAttribute(key, attr);
      })
      .def("get_attribute", &InferObject::GetAttribute)
      .def("add_extra_attribute", &InferObject::AddExtraAttribute)
      .def("add_extra_attributes", &InferObject::AddExtraAttributes)
      .def("get_extra_attribute", &InferObject::GetExtraAttribute)
      .def("remove_extra_attribute", &InferObject::RemoveExtraAttribute)
      .def("get_extra_attributes", &InferObject::GetExtraAttributes)
      .def("add_feature", &InferObject::AddFeature)
      .def("get_feature", &InferObject::GetFeature)
      .def("get_features", &InferObject::GetFeatures);
}

/**
 * @brief ModelInputImage（派生图）wrapper。
 *
 * Collection 内部以 std::any 做类型擦除，无法通用映射到 Python；
 * 这里暴露真正与 C++ 互通的 typed tag，使 Python 模块能为下游推理写入派生输入图。
 */
void ModelInputImageWrapper(const py::module &m) {
  py::class_<ModelInputImage, std::shared_ptr<ModelInputImage>>(m, "ModelInputImage")
      .def(py::init([]() {
        return std::make_shared<ModelInputImage>();
      }))
      .def_property("image",
          [](const ModelInputImage &self) { return MatToArray(self.image); },
          [](ModelInputImage &self, py::array_t<uint8_t> img) { self.image = ArrayToMat(img); })
      .def_readwrite("cur_offset_x", &ModelInputImage::cur_offset_x)
      .def_readwrite("cur_offset_y", &ModelInputImage::cur_offset_y)
      .def_readwrite("cur_width", &ModelInputImage::cur_width)
      .def_readwrite("cur_height", &ModelInputImage::cur_height)
      .def_readwrite("cur_scale_x", &ModelInputImage::cur_scale_x)
      .def_readwrite("cur_scale_y", &ModelInputImage::cur_scale_y)
      .def_readwrite("rotation", &ModelInputImage::rotation)
      .def_readwrite("derived", &ModelInputImage::derived);
}

void FrameVaWrapper(const py::module &m) {
  DataFrameWrapper(m);
  InferObjsWrapper(m);
  ModelInputImageWrapper(m);

  g_py_frame_info->def("get_data_frame", &GetDataFrame);
  g_py_frame_info->def("get_infer_objects", &GetInferObjects);

  // 派生图读取：优先级与 GetModelInputImage 一致（模块级 > 帧级 > 原图），
  // 但逐级判空，避免缺 tag / 空派生图 / 无 DataFrame（如 EOS）时触发框架 LOGF
  g_py_frame_info->def("get_model_input_image",
      [](std::shared_ptr<FrameInfo> frame, const std::string &model_name) {
        if (!model_name.empty()) {
          std::string mtag = ModelInputImageTagForModel(model_name);
          if (frame->collection.HasValue(mtag)) {
            auto d = frame->collection.Get<ModelInputImagePtr>(mtag);
            if (d && !d->image.empty()) return MatToArray(d->image);
          }
        }
        if (frame->collection.HasValue(kModelInputImageTag)) {
          auto d = frame->collection.Get<ModelInputImagePtr>(kModelInputImageTag);
          if (d && !d->image.empty()) return MatToArray(d->image);
        }
        if (frame->collection.HasValue(kDataFrameTag)) {
          auto df = frame->collection.Get<DataFramePtr>(kDataFrameTag);
          if (df) return MatToArray(df->GetImage());
        }
        return py::array();
      }, py::arg("model_name") = "");

  // 派生图写入：model_name 非空写模块级 tag，否则写帧级 tag
  g_py_frame_info->def("set_model_input_image",
      [](std::shared_ptr<FrameInfo> frame, py::array_t<uint8_t> image, const std::string &model_name) {
        auto derived = std::make_shared<ModelInputImage>();
        derived->image = ArrayToMat(image);
        if (derived->image.channels() != 3) {
          throw std::runtime_error("model input image must be 3-channel BGR");
        }
        derived->cur_width = derived->image.cols;
        derived->cur_height = derived->image.rows;
        std::string tag = model_name.empty() ? std::string(kModelInputImageTag)
                                             : ModelInputImageTagForModel(model_name);
        frame->collection.Set<ModelInputImagePtr>(tag, derived);
      }, py::arg("image"), py::arg("model_name") = "");

  // 需要控制坐标还原元信息（offset/scale/rotation）时，直接读写 ModelInputImage 结构
  g_py_frame_info->def("get_model_input_image_info",
      [](std::shared_ptr<FrameInfo> frame, const std::string &model_name) -> std::shared_ptr<ModelInputImage> {
        std::string tag = model_name.empty() ? std::string(kModelInputImageTag)
                                             : ModelInputImageTagForModel(model_name);
        if (!frame->collection.HasValue(tag)) return nullptr;
        return frame->collection.Get<ModelInputImagePtr>(tag);
      }, py::arg("model_name") = "");

  g_py_frame_info->def("set_model_input_image_info",
      [](std::shared_ptr<FrameInfo> frame, std::shared_ptr<ModelInputImage> info, const std::string &model_name) {
        std::string tag = model_name.empty() ? std::string(kModelInputImageTag)
                                             : ModelInputImageTagForModel(model_name);
        frame->collection.Set<ModelInputImagePtr>(tag, info);
      }, py::arg("info"), py::arg("model_name") = "");
}

}  //  namespace cnstream
