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

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include "cnstream_frame.hpp"
#include "cnstream_source.hpp"

namespace py = pybind11;

namespace cnstream {

std::shared_ptr<py::class_<FrameInfo, std::shared_ptr<FrameInfo>>> g_py_frame_info;

void FrameInfoWrapper(const py::module &m) {
  g_py_frame_info = std::make_shared<py::class_<FrameInfo, std::shared_ptr<FrameInfo>>>(m, "FrameInfo");

  (*g_py_frame_info)
      .def(py::init([](std::string stream_id, bool eos = false) {
        auto frame_info = FrameInfo::Create(stream_id, eos);
        return frame_info;
      }), py::arg().noconvert(), py::arg("eos") = false)
      .def("is_eos", &FrameInfo::IsEos)
      .def("is_removed", &FrameInfo::IsRemoved)
      .def("is_invalid", &FrameInfo::IsInvalid)

      // 获取固定的 collection 的 py_collection value
      .def("get_py_collection", [] (std::shared_ptr<FrameInfo> frame) -> py::dict {
          frame->collection.AddIfNotExists("py_collection", std::shared_ptr<py::dict>(new py::dict(), [] (py::dict* t) {
            // py::dict destruct in c++ thread without gil resource
            // this is important to get gil when delete a py::dict.
            py::gil_scoped_acquire gil;
            delete t;
          }));
          // copy constructor
          auto py_collection = *(frame->collection.Get<std::shared_ptr<py::dict>>("py_collection"));
          return py_collection;
      })
      .def_readwrite("stream_id", &FrameInfo::stream_id)
      .def_readwrite("timestamp", &FrameInfo::timestamp);
}

}  //  namespace cnstream
