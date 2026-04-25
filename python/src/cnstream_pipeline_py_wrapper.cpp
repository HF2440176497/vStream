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

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cnstream_pipeline.hpp"
#include "cnstream_source.hpp"
#include "output_module.hpp"

#include <memory>
#include <string>

namespace py = pybind11;

namespace cnstream {

/**
 * @brief 只绑定必要的接口
 */
void PipelineWrapper(py::module &m) {

  py::enum_<StreamMsgType>(m, "StreamMsgType")
      .value("eos_msg", StreamMsgType::EOS_MSG)
      .value("error_msg", StreamMsgType::ERROR_MSG)
      .value("stream_err_msg", StreamMsgType::STREAM_ERR_MSG)
      .value("frame_err_msg", StreamMsgType::FRAME_ERR_MSG);

  py::class_<StreamMsg>(m, "StreamMsg")
      .def_readwrite("type", &StreamMsg::type)
      .def_readwrite("stream_id", &StreamMsg::stream_id)
      .def_readwrite("module_name", &StreamMsg::module_name)
      .def_readwrite("pts", &StreamMsg::pts);

  py::class_<Pipeline>(m, "Pipeline")
      .def(py::init<std::string>())
      .def("get_name", &Pipeline::GetName)
      .def("build_pipeline", static_cast<bool (Pipeline::*)(const CNGraphConfig &)>(&Pipeline::BuildPipeline))
      .def("build_pipeline_by_json_file", &Pipeline::BuildPipelineByJSONFile)
      .def("start", &Pipeline::Start)
      .def("stop", &Pipeline::Stop, py::call_guard<py::gil_scoped_release>())
      .def("is_running", &Pipeline::IsRunning)
      .def("get_source_module",
           [](Pipeline *pipeline, const std::string &module_name) {
              auto* module = pipeline->GetModule(module_name);
              if (!module) return static_cast<SourceModule *>(nullptr);
              return dynamic_cast<SourceModule *>(module);
           },
           py::return_value_policy::reference)
      .def("get_output_module", [](Pipeline *pipeline, const std::string &module_name) {
              auto* module = pipeline->GetModule(module_name);
              if (!module) return static_cast<OutputModule *>(nullptr);
              return dynamic_cast<OutputModule *>(module);
           },
           py::return_value_policy::reference)
      .def("get_module",
           [](Pipeline *pipeline, const std::string &module_name) {
              auto* module = pipeline->GetModule(module_name);
              if (!module) return static_cast<Module *>(nullptr);
              return dynamic_cast<Module *>(module);
           },
           py::return_value_policy::reference)
      .def("get_module_config", &Pipeline::GetModuleConfig)
      .def("is_profiling_enabled", &Pipeline::IsProfilingEnabled)
      .def("is_root_node", &Pipeline::IsRootNode);
}

}  // namespace cnstream

