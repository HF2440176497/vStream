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

#include <memory>
#include <string>
#include <vector>

#include "cnstream_source.hpp"
#include "data_source.hpp"
#include "common_wrapper.hpp"

namespace py = pybind11;

namespace cnstream {

void DataHandlerWrapper(const py::module &m) {

  py::enum_<OutputType>(m, "OutputType")
      .value("output_cpu", OutputType::OUTPUT_CPU)
      .value("output_cuda", OutputType::OUTPUT_CUDA);

  py::enum_<DecoderType>(m, "DecoderType")
      .value("decoder_cpu", DecoderType::DECODER_CPU)
      .value("decoder_cuda", DecoderType::DECODER_CUDA)
      .value("decoder_rkmpp", DecoderType::DECODER_RKMPP);

  py::class_<DataSourceParam>(m, "DataSourceParam")
      .def(py::init());

  py::class_<DataSource, std::shared_ptr<DataSource>, SourceModule>(m, "DataSource")
      .def(py::init<const std::string&>())
      .def("open", &DataSource::Open)
      .def("close", &DataSource::Close)
      .def("check_param_set", &DataSource::CheckParamSet)
      .def("get_source_param", &DataSource::GetSourceParam);

  py::class_<ImageHandler, std::shared_ptr<ImageHandler>, SourceHandler>(m, "ImageHandler")
      .def(py::init([](DataSource *module, const std::string &stream_id) {
        auto image_handler = ImageHandler::Create(module, stream_id);
        if (!image_handler)
          return std::shared_ptr<ImageHandler>();
        return std::dynamic_pointer_cast<ImageHandler>(image_handler);
      }), py::arg("module"), py::arg("stream_id"))
      .def("open", &ImageHandler::Open)
      .def("stop", &ImageHandler::Stop)
      .def("close", &ImageHandler::Close);

  py::class_<PullHandler, std::shared_ptr<PullHandler>, SourceHandler>(m, "PullHandler")
      .def(py::init([](DataSource *module, const std::string &stream_id) {
        auto pull_handler = PullHandler::Create(module, stream_id);
        if (!pull_handler)
          return std::shared_ptr<PullHandler>();
        return std::dynamic_pointer_cast<PullHandler>(pull_handler);
      }), py::arg("module"), py::arg("stream_id"))
      .def("open", &PullHandler::Open)
      .def("stop", &PullHandler::Stop)
      .def("close", &PullHandler::Close);


  py::class_<SendFrame>(m, "SendFrame")
      .def(py::init([]() {
        return SendFrame{};
      }))
      .def(py::init([](uint64_t pts, const std::string& frame_id_s, py::array_t<uint8_t> image) {
        SendFrame frame;
        frame.pts = pts;
        frame.frame_id_s = frame_id_s;
        frame.image = ArrayToMat(image);  // deep copy
        return frame;
      }), py::arg("pts"), py::arg("frame_id_s"), py::arg("image"))
      .def_readwrite("pts", &SendFrame::pts)
      .def_readwrite("frame_id_s", &SendFrame::frame_id_s)
      .def_property("image",
        [](const SendFrame& frame) {
          return MatToArray(frame.image);
        },
        [](SendFrame& frame, py::array_t<uint8_t> image) {
          frame.image = ArrayToMat(image);  // deep copy
        });

  py::class_<SendHandler, std::shared_ptr<SendHandler>, SourceHandler>(m, "SendHandler")
      .def(py::init([](DataSource *module, const std::string &stream_id) {
        auto send_handler = SendHandler::Create(module, stream_id);
        if (!send_handler)
          return std::shared_ptr<SendHandler>();
        return std::dynamic_pointer_cast<SendHandler>(send_handler);
      }), py::arg("module"), py::arg("stream_id"))
      .def("open", &SendHandler::Open)
      .def("stop", &SendHandler::Stop)
      .def("close", &SendHandler::Close)
      .def("send", [](SendHandler& self, uint64_t pts, const std::string& frame_id_s, py::array_t<uint8_t> image) {
        cv::Mat mat = ArrayToMat(image);  // must hold GIL for numpy buffer access
        py::gil_scoped_release release;
        return self.Send(pts, frame_id_s, mat);
      }, py::arg("pts"), py::arg("frame_id_s"), py::arg("image"))
      .def("send_frame", [](SendHandler& self, const SendFrame& frame) {
        return self.Send(frame);
      }, py::arg("frame"),
         py::call_guard<py::gil_scoped_release>());


}  // DataHandlerWrapper

}  // namespace cnstream
