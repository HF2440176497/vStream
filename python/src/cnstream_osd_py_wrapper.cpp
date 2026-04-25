
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cnstream_pipeline.hpp"
#include "cnstream_source.hpp"
#include "decode_queue.hpp"
#include "output_module.hpp"
#include "data_common.hpp"
#include "common_wrapper.hpp"

#include <memory>
#include <string>

namespace py = pybind11;

namespace cnstream {

namespace detail {

/**
 * @brief PyOutputModule 是 OutputModule 的 trampoline 类
 * @details 允许 Python 继承 OutputModule 并 override 虚函数
 */
class PyOutputModule : public OutputModule {
 public:
  using OutputModule::OutputModule;

  bool Open(ModuleParamSet params) override {
    PYBIND11_OVERRIDE_PURE(
        bool,
        OutputModule,
        open,
        params);
  }

  void Close() override {
    PYBIND11_OVERRIDE_PURE(
        void,
        OutputModule,
        close);
  }

  int Process(std::shared_ptr<FrameInfo> data) override {
    PYBIND11_OVERRIDE_PURE(
        int,
        OutputModule,
        process,
        data);
  }

  /**
   * @brief 获取输出数据
   * @param data 输出数据
   * @param wait_ms 等待时间，单位毫秒
   * @return true 成功获取数据
   */
  bool GetData(s_output_data& data, int wait_ms) override {
    PYBIND11_OVERRIDE_PURE(
        bool,
        OutputModule,
        get_data,
        data,
        wait_ms);
  }
};  // class PyOutputModule

}  // namespace detail

void OsdModuleWrapper(py::module &m) {

  py::class_<s_class_infos>(m, "class_infos")
      .def(py::init<>())
      .def_readwrite("id", &s_class_infos::id)
      .def_readwrite("model_name", &s_class_infos::model_name)
      .def_readwrite("id_name", &s_class_infos::id_name)
      .def_readwrite("score", &s_class_infos::score)
      .def_readwrite("value", &s_class_infos::value);

  py::class_<s_obj_in>(m, "obj_in")
      .def(py::init<>())
      .def_readwrite("track_id", &s_obj_in::track_id)
      .def_readwrite("score", &s_obj_in::score)
      .def_readwrite("bboxs", &s_obj_in::bboxs)
      .def_readwrite("feature", &s_obj_in::feature)
      .def_readwrite("classes", &s_obj_in::classes)
      .def_readwrite("str_id", &s_obj_in::str_id)
      .def_readwrite("model_name", &s_obj_in::model_name);

  py::class_<s_output_data>(m, "output_data")
      .def(py::init<>())
      .def_readwrite("result", &s_output_data::result)
      .def_readwrite("timestamp", &s_output_data::timestamp)
      .def_readwrite("frame_id_s", &s_output_data::frame_id_s)
      .def_readwrite("objects", &s_output_data::objects)
      .def_readwrite("objects_dict", &s_output_data::objects_dict)
      .def_readwrite("objects_json", &s_output_data::objects_json)
      .def_property("image_dict",
        [](const s_output_data& data) {
          py::dict dict;
          for (auto& [key, mat] : data.image_dict) {
            dict[key.c_str()] = MatToArray(const_cast<cv::Mat&>(mat));
          }
          return dict;
        },
        // setter 重新赋值 image_dict
        [](s_output_data& data, py::dict dict) {
          data.image_dict.clear();
          for (auto item : dict) {
            std::string key = py::str(item.first).cast<std::string>();
            py::array_t<uint8_t> arr = item.second.cast<py::array_t<uint8_t>>();
            data.image_dict[key] = ArrayToMat(arr);
          }
        });


  py::class_<OutputModule, std::shared_ptr<OutputModule>, detail::PyOutputModule>(m, "OutputModule")
      .def(py::init<const std::string&>())
      .def("open", &OutputModule::Open)
      .def("close", &OutputModule::Close)
      .def("process", &OutputModule::Process)
      // for py: ret, data = osd.get_data(wait_ms=10)
      .def("get_data",
        [](OutputModule& self, int wait_ms) {
          s_output_data data;
          bool ok = self.GetData(data, wait_ms);
          return std::make_pair(ok, data);
        },
        py::arg("wait_ms") = 0);

  // <class, holder, base>
  py::class_<DecodeQueue, OutputModule, std::shared_ptr<DecodeQueue>>(m, "DecodeQueue")
      .def(py::init<const std::string&>());
}

}  // namespace cnstream
