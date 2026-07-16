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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

extern "C" {
  #include <libavutil/log.h>
}
#include "cnstream_ffmpeg_logging.hpp"

namespace py = pybind11;

namespace cnstream {

void FrameInfoWrapper(const py::module&);
void FrameVaWrapper(const py::module&);
void ModuleWrapper(py::module &);
void SourceModuleWrapper(const py::module &);
void PipelineWrapper(py::module &);
void DataHandlerWrapper(const py::module &);
void SinkModuleWrapper(py::module &);
void ModelValidatorWrapper(const py::module &);

PYBIND11_MODULE(vstream, m) {
  m.doc() = "vstream python api";

  SetFFmpegLogLevel(AV_LOG_WARNING);
  m.def("set_ffmpeg_log_level", &SetFFmpegLogLevel,
        "Set the FFmpeg log level (e.g., AV_LOG_ERROR=16, AV_LOG_WARNING=24, AV_LOG_INFO=32, AV_LOG_DEBUG=48)");

  FrameInfoWrapper(m);
  FrameVaWrapper(m);
  ModuleWrapper(m);
  SourceModuleWrapper(m);
  PipelineWrapper(m);
  DataHandlerWrapper(m);
  SinkModuleWrapper(m);
  ModelValidatorWrapper(m);
}

}  // namespace cnstream

