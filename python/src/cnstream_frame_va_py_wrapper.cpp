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
          return data_frame.data_[plane_idx].get();
      }, py::return_value_policy::reference_internal)
      .def_readwrite("frame_id", &DataFrame::frame_id_)  // uint64_t
      .def_readwrite("fmt", &DataFrame::fmt_)
      .def_readwrite("width", &DataFrame::width_)
      .def_readwrite("height", &DataFrame::height_)

      // getter and setter for stride
      .def_property("stride",
        [](const DataFrame& data_frame) {
          py::array_t<int> result(FRAME_MAX_PLANES);
          py::buffer_info buf = result.request();
          memcpy(buf.ptr, data_frame.stride_, FRAME_MAX_PLANES * sizeof(int));
          return result;
        },
        // eg: data_frame.stride = new_strides
        [](std::shared_ptr<DataFrame> data_frame, py::array_t<int> strides) {
            py::buffer_info strides_buf = strides.request();
            int size = std::min(static_cast<int>(strides_buf.size), FRAME_MAX_PLANES);
            memcpy(data_frame->stride_, strides_buf.ptr, size * sizeof(int));
        });

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

  py::class_<DevContext>(m, "DevContext")
      .def(py::init())
      .def_readwrite("device_type", &DevContext::device_type)
      .def_readwrite("device_id", &DevContext::device_id);

  py::enum_<DevContext::DevType>(m, "DevType")
    .value("INVALID", DevContext::DevType::INVALID)
    .value("CPU", DevContext::DevType::CPU)
    .value("CUDA", DevContext::DevType::CUDA);
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
      .def_readwrite("score", &InferAttr::score);


  py::class_<InferObject, std::shared_ptr<InferObject>>(m, "InferObject")
      .def(py::init([]() {
        return std::make_shared<InferObject>();
      }))
      .def_readwrite("id", &InferObject::id)
      .def_readwrite("track_id", &InferObject::track_id)
      .def_readwrite("score", &InferObject::score)
      .def_readwrite("bbox", &InferObject::bbox)

      // 单独为 py 提供的一个容器
      .def("get_py_collection", [](std::shared_ptr<InferObject> obj) {
          if (!obj->collection.HasValue("py_collection")) {
            obj->collection.Add("py_collection", py::dict());
          }
          return obj->collection.Get<py::dict>("py_collection");
      })
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

void FrameVaWrapper(const py::module &m) {
  DataFrameWrapper(m);
  InferObjsWrapper(m);

  g_py_frame_info->def("get_data_frame", &GetDataFrame);
  g_py_frame_info->def("get_infer_objects", &GetInferObjects);
}

}  //  namespace cnstream
