/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
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

#ifndef MODULES_INFERENCE_SRC_BATCHING_DONE_STAGE_HPP_
#define MODULES_INFERENCE_SRC_BATCHING_DONE_STAGE_HPP_

#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <cassert>

#include "profiler/module_profiler.hpp"
#include "memop.hpp"
#include "memop_factory.hpp"
#include "infer_task.hpp"
#include "infer_resource.hpp"

#include "model_loader.hpp"

namespace cnstream {

class Postproc;
class ObjPostproc;
class CpuInputResource;
class CpuOutputResource;
class NetInputResource;
class NetOutputResource;

class InferTask;
class FrameInfo;
class InferObject;

struct AutoSetDone {
  explicit AutoSetDone(const std::shared_ptr<std::promise<void>>& p,
                       std::shared_ptr<FrameInfo> data)
      : p_(p), data_(data) {}
  ~AutoSetDone() {
    p_->set_value();
  }
  std::shared_ptr<std::promise<void>> p_;
  std::shared_ptr<FrameInfo> data_;
};  // struct AutoSetDone

using BatchingDoneInput = std::vector<std::pair<std::shared_ptr<FrameInfo>, std::shared_ptr<AutoSetDone>>>;

class BatchingDoneStage {
 public:
  BatchingDoneStage() = default;
  BatchingDoneStage(ModelLoader* model, uint32_t batchsize, int device_id)
      : model_(model), batchsize_(batchsize), device_id_(device_id) {}
  virtual ~BatchingDoneStage() {}
  virtual std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) = 0;

  void SetDumpResizedImageDir(const std::string &dir) {
    dump_resized_image_dir_ = dir;
  }

  void SetSavingInputData(const bool& saving_infer_input, const std::string& module_name) {
    saving_infer_input_ = saving_infer_input;
    module_name_ = module_name;
  }

  void SetProfiler(ModuleProfiler* profiler) {
    profiler_ = profiler;
  }

#ifdef UNIT_TEST
 public:
#else
 protected:
#endif
  ModuleProfiler* profiler_ = nullptr;
  bool saving_infer_input_ = false;
  std::string module_name_ = "";
  std::string dump_resized_image_dir_ = "";
  ModelLoader* model_;
  uint32_t batchsize_ = 0;
  int device_id_ = 0;
};  // class BatchingDoneStage


class H2DBatchingDoneStage : public BatchingDoneStage {
 public:
  H2DBatchingDoneStage(ModelLoader* model,
                       uint32_t batchsize,
                       int device_id,
                       std::shared_ptr<CpuInputResource> cpu_input_res, 
                       std::shared_ptr<NetInputResource> net_input_res)
      : BatchingDoneStage(model, batchsize, device_id), 
      cpu_input_res_(cpu_input_res), 
      net_input_res_(net_input_res) {
        if (model) {
          memop_ = MemOpFactory::Instance().CreateMemOp(model->GetDeviceType(), model->GetDeviceId());
        }
      }
  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override;
 private:
  std::shared_ptr<CpuInputResource> cpu_input_res_;
  std::shared_ptr<NetInputResource> net_input_res_;
  std::shared_ptr<MemOp> memop_;
};  // class H2DBatchingDoneStage


class InferBatchingDoneStage : public BatchingDoneStage {
 public:
  InferBatchingDoneStage(ModelLoader* model,
                         uint32_t batchsize,
                         int device_id,
                         std::shared_ptr<NetInputResource> net_input_res,
                         std::shared_ptr<NetOutputResource> net_output_res);
  ~InferBatchingDoneStage();
  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override;
 private:
  std::shared_ptr<NetInputResource> net_input_res_;
  std::shared_ptr<NetOutputResource> net_output_res_;
};  // class InferBatchingDoneStage

class D2HBatchingDoneStage : public BatchingDoneStage {
 public:
  D2HBatchingDoneStage(ModelLoader* model,
                       uint32_t batchsize,
                       int device_id,
                       std::shared_ptr<NetOutputResource> net_output_res,
                       std::shared_ptr<CpuOutputResource> cpu_output_res)
      : BatchingDoneStage(model, batchsize, device_id), 
      net_output_res_(net_output_res), 
      cpu_output_res_(cpu_output_res) {
        if (model) {
          memop_ = MemOpFactory::Instance().CreateMemOp(model->GetDeviceType(), model->GetDeviceId());
        }
      }

  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override;

 private:
  std::shared_ptr<NetOutputResource> net_output_res_;
  std::shared_ptr<CpuOutputResource> cpu_output_res_;
  std::shared_ptr<MemOp> memop_;
};  // class D2HBatchingDoneStage

class PostprocessingBatchingDoneStage : public BatchingDoneStage {
 public:
  PostprocessingBatchingDoneStage(ModelLoader* model,
                                  uint32_t batchsize,
                                  int device_id, 
                                  std::shared_ptr<Postproc> postprocessor,
                                  std::shared_ptr<CpuOutputResource> cpu_output_res)
      : BatchingDoneStage(model, batchsize, device_id), 
      postprocessor_(postprocessor), 
      cpu_output_res_(cpu_output_res) {}

  PostprocessingBatchingDoneStage(ModelLoader* model,
                                  uint32_t batchsize,
                                  int device_id, 
                                  std::shared_ptr<Postproc> postprocessor,
                                  std::shared_ptr<NetOutputResource> net_output_res)
      : BatchingDoneStage(model, batchsize, device_id), 
      postprocessor_(postprocessor), 
      net_output_res_(net_output_res) {}

  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override;
  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos,
                                                       const std::shared_ptr<CpuOutputResource> &cpu_output_res);
  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos,
                                                       const std::shared_ptr<NetOutputResource> &net_output_res);
 private:
  std::shared_ptr<Postproc> postprocessor_ = nullptr;
  std::shared_ptr<CpuOutputResource> cpu_output_res_ = nullptr;
  std::shared_ptr<NetOutputResource> net_output_res_ = nullptr;
};  // class PostprocessingBatchingDoneStage


class ObjPostprocessingBatchingDoneStage : public BatchingDoneStage {
 public:
  ObjPostprocessingBatchingDoneStage(ModelLoader* model, uint32_t batchsize, int device_id,
                                     std::shared_ptr<ObjPostproc> postprocessor,
                                     std::shared_ptr<CpuOutputResource> cpu_output_res)
      : BatchingDoneStage(model, batchsize, device_id), postprocessor_(postprocessor), cpu_output_res_(cpu_output_res) {}

  ObjPostprocessingBatchingDoneStage(ModelLoader* model, uint32_t batchsize, int device_id,
                                     std::shared_ptr<ObjPostproc> postprocessor,
                                     std::shared_ptr<NetOutputResource> net_output_res)
      : BatchingDoneStage(model, batchsize, device_id), postprocessor_(postprocessor), net_output_res_(net_output_res) {}

  std::vector<std::shared_ptr<InferTask>> BatchingDone(const BatchingDoneInput& finfos) override { return {}; }

  std::vector<std::shared_ptr<InferTask>> ObjBatchingDone(const BatchingDoneInput& finfos,
                                                          const std::vector<std::shared_ptr<InferObject>>& objs);
  std::vector<std::shared_ptr<InferTask>> ObjBatchingDone(const BatchingDoneInput& finfos,
                                                          const std::vector<std::shared_ptr<InferObject>>& objs,
                                                          const std::shared_ptr<CpuOutputResource> &cpu_output_res);
  std::vector<std::shared_ptr<InferTask>> ObjBatchingDone(const BatchingDoneInput& finfos,
                                                          const std::vector<std::shared_ptr<InferObject>>& objs,
                                                          const std::shared_ptr<NetOutputResource> &net_output_res);

 private:
  std::shared_ptr<ObjPostproc> postprocessor_;
  std::shared_ptr<CpuOutputResource> cpu_output_res_ = nullptr;
  std::shared_ptr<NetOutputResource> net_output_res_ = nullptr;
};  // class ObjPostprocessingBatchingDoneStage

}  // namespace cnstream


#endif  // MODULES_INFERENCE_SRC_BATCHING_DONE_STAGE_HPP_
