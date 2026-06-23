
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cnstream_pipeline.hpp"
#include "cnstream_source.hpp"
#include "data_sink.hpp"
#include "data_common.hpp"
#include "common_wrapper.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace cnstream {

namespace detail {

class PySinkHandler : public SinkHandler {
 public:
  using SinkHandler::SinkHandler;
  bool Open() override {
    PYBIND11_OVERRIDE_PURE(
      bool,
      SinkHandler,
      open);
  }
  void Close() override {
    PYBIND11_OVERRIDE_PURE(
      void,
      SinkHandler,
      close);
  }
};

class PySinkModule : public SinkModule {
 public:
  using SinkModule::SinkModule;
  bool Open(ModuleParamSet params) override {
    PYBIND11_OVERRIDE_PURE(
      bool,
      SinkModule,
      open,
      params);
  }
  void Close() override {
    PYBIND11_OVERRIDE_PURE(
      void,
      SinkModule,
      close);
  }
};

}  // namespace detail

void SinkModuleWrapper(py::module &m) {

  py::class_<s_class_infos>(m, "class_infos")
      .def(py::init<>())
      .def_readwrite("id", &s_class_infos::id)
      .def_readwrite("model_name", &s_class_infos::model_name)
      .def_readwrite("name", &s_class_infos::name)
      .def_readwrite("score", &s_class_infos::score)
      .def_readwrite("value", &s_class_infos::value)
      .def("__repr__", [](const s_class_infos& info) {
        std::ostringstream oss;
        oss << info;
        return oss.str();
      })
      .def("__dir__", [](const s_class_infos&) {
        return std::vector<std::string>{"id", "model_name", "name", "score", "value"};
      });

  py::class_<s_attr_info>(m, "attr_info")
      .def(py::init<>())
      .def_readwrite("id", &s_attr_info::id)
      .def_readwrite("value", &s_attr_info::value)
      .def_readwrite("score", &s_attr_info::score)
      .def_readwrite("name", &s_attr_info::name)
      .def("__repr__", [](const s_attr_info& info) {
        std::ostringstream oss;
        oss << info;
        return oss.str();
      })
      .def("__dir__", [](const s_attr_info&) {
        return std::vector<std::string>{"id", "value", "score", "name"};
      });

  py::class_<s_obj_in>(m, "obj_in")
      .def(py::init<>())
      .def_readwrite("id", &s_obj_in::id)
      .def_readwrite("track_id", &s_obj_in::track_id)
      .def_readwrite("score", &s_obj_in::score)
      .def_readwrite("bboxs", &s_obj_in::bboxs)
      .def_readwrite("feature", &s_obj_in::feature)
      .def_readwrite("classes", &s_obj_in::classes)
      .def_readwrite("model_name", &s_obj_in::model_name)
      .def_readwrite("key_points", &s_obj_in::key_points)
      .def_readwrite("type", &s_obj_in::type)
      .def_readwrite("attributes", &s_obj_in::attributes)
      .def("__repr__", [](const s_obj_in& obj) {
        std::ostringstream oss;
        oss << obj;
        return oss.str();
      })
      .def("__dir__", [](const s_obj_in&) {
        return std::vector<std::string>{
            "id", "track_id", "score", "bboxs", "feature", "classes",
            "model_name", "key_points", "type", "attributes"};
      });

  py::class_<s_output_data>(m, "output_data")
      .def(py::init<>())
      .def_readwrite("result", &s_output_data::result)
      .def_readwrite("timestamp", &s_output_data::timestamp)
      .def_readwrite("frame_id_s", &s_output_data::frame_id_s)
      .def_readwrite("objects", &s_output_data::objects)
      .def_readwrite("objects_dict", &s_output_data::objects_dict)
      .def_readwrite("objects_json", &s_output_data::objects_json)
      .def("__repr__", [](const s_output_data& data) {
        std::ostringstream oss;
        oss << data;
        return oss.str();
      })
      .def("__dir__", [](const s_output_data&) {
        return std::vector<std::string>{
            "result", "timestamp", "frame_id_s", "objects",
            "objects_dict", "objects_json", "image_dict"};
      })
      .def_property("image_dict",
        [](const s_output_data& data) {
          py::dict dict;
          for (auto& [key, mat] : data.image_dict) {
            dict[key.c_str()] = MatToArray(mat);
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

  py::class_<SinkModule, std::shared_ptr<SinkModule>, detail::PySinkModule>(m, "SinkModule")
      .def(py::init<const std::string&>())
      .def("open", &SinkModule::Open)
      .def("close", &SinkModule::Close)
      .def("add_sink", &SinkModule::AddSink)
      .def("get_sink_handler", &SinkModule::GetSinkHandler)
      .def("remove_sink", 
        [](SinkModule *sink, std::shared_ptr<SinkHandler> handler, bool force) {
          return sink->RemoveSink(handler, force);
        },
        py::arg("handler"), py::arg("force") = false,
        py::call_guard<py::gil_scoped_release>())
      .def("remove_sink", 
        [](SinkModule *sink, const std::string& stream_id, bool force) {
          return sink->RemoveSink(stream_id, force);
        },
        py::arg("stream_id"), py::arg("force") = false,
        py::call_guard<py::gil_scoped_release>())
      .def("remove_sinks", 
        &SinkModule::RemoveSinks, py::arg("force") = false,
        py::call_guard<py::gil_scoped_release>());

  py::class_<DataSink, std::shared_ptr<DataSink>, SinkModule>(m, "DataSink")
      .def(py::init<const std::string&>())
      .def("open", &DataSink::Open)
      .def("close", &DataSink::Close)
      .def("check_param_set", &DataSink::CheckParamSet);

  py::class_<SinkHandler, std::shared_ptr<SinkHandler>, detail::PySinkHandler>(m, "SinkHandler")
      .def(py::init<SinkModule *, const std::string&>())
      .def("open", &SinkHandler::Open)
      .def("close", &SinkHandler::Close)
      .def("stop", &SinkHandler::Stop)
      .def("get_stream_id", &SinkHandler::GetStreamId);

  py::class_<QueueHandler, std::shared_ptr<QueueHandler>, SinkHandler>(m, "QueueHandler")
      .def(py::init([](DataSink *module, const std::string& stream_id) {
        auto queue_handler = QueueHandler::Create(module, stream_id);
        if (!queue_handler) {
          return std::shared_ptr<QueueHandler>(nullptr);
        }
        return std::dynamic_pointer_cast<QueueHandler>(queue_handler);
      }), py::arg("module"), py::arg("stream_id"))
      .def("open", &QueueHandler::Open)
      .def("close", &QueueHandler::Close)
      .def("stop", &QueueHandler::Stop)
      .def("get_data",
        [](QueueHandler& self, int wait_ms) {
          s_output_data data;
          bool ok = self.GetData(data, wait_ms);
          return std::make_pair(ok, data);
        },
        py::arg("wait_ms") = 0,
        py::call_guard<py::gil_scoped_release>());

  py::class_<PushHandler, std::shared_ptr<PushHandler>, SinkHandler>(m, "PushHandler")
      .def(py::init([](DataSink *module, const std::string& stream_id) {
        auto push_handler = PushHandler::Create(module, stream_id);
        if (!push_handler) {
          return std::shared_ptr<PushHandler>(nullptr);
        }
        return std::dynamic_pointer_cast<PushHandler>(push_handler);
      }), py::arg("module"), py::arg("stream_id"))
      .def("open", &PushHandler::Open)
      .def("close", &PushHandler::Close)
      .def("stop", &PushHandler::Stop);
}

}  // namespace cnstream
