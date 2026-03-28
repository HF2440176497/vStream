/*************************************************************************
 * Copyright (C) [2024] by TensorRT Adapter. All rights reserved
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

#include "infer_engine.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>

namespace cnstream {


InferEngine::InferEngine(const InferOptions& options)
    : model_(options.model()),
      preprocessor_(options.preprocessor()),
      postprocessor_(options.postprocessor()),
      obj_preprocessor_(options.obj_preprocessor()),
      obj_postprocessor_(options.obj_postprocessor()),
      batchsize_(options.batchsize()),
      batching_timeout_(options.batching_timeout()),
      dev_id_(options.dev_id()),
      batching_by_obj_(options.batching_by_obj()),
      module_name_(options.module_name()),
      error_func_(options.error_handler()) {
  cudaSetDevice(dev_id_);

  thread_pool_ = std::make_shared<InferThreadPool>();
  thread_pool_->SetErrorHandleFunc(error_func);
  thread_pool_->Init(dev_id_, batchsize * 3 + 4);

  cpu_input_res_ = std::make_shared<CpuInputResource>(model_, batchsize_);
  cpu_output_res_ = std::make_shared<CpuOutputResource>(model_, batchsize_);
  net_input_res_ = std::make_shared<NetInputResource>(model_, batchsize_);
  net_output_res_ = std::make_shared<NetOutputResource>(model_, batchsize_);

  cpu_input_res_->Init();
  cpu_output_res_->Init();
  net_input_res_->Init();
  net_output_res_->Init();

  StageAssemble();

  running_ = true;
}

InferEngine::~InferEngine() {
  running_ = false;
  cv_.notify_all();

  if (timeout_thread_.joinable()) {
    timeout_thread_.join();
  }

  if (thread_pool_) {
    thread_pool_->Destroy();
  }

  if (cpu_input_res_) cpu_input_res_->Destroy();
  if (cpu_output_res_) cpu_output_res_->Destroy();
  if (net_input_res_) net_input_res_->Destroy();
  if (net_output_res_) net_output_res_->Destroy();
}

/**
 * note: 暂时严格按照 prec - h2d - infer - d2h - postproc 顺序
 */
void InferEngine::StageAssemble() {
  bool cpu_preprocessing = (!batching_by_obj_ && preprocessor_) || (batching_by_obj_ && obj_preprocessor_);

  if (cpu_preprocessing) {
    if (batching_by_obj_) {
      obj_batching_stage_ = std::make_shared<CpuPreprocessingObjBatchingStage>(model_, batchsize_, obj_preprocessor_,
                                                                               cpu_input_res_);
    } else {
      batching_stage_ =
          std::make_shared<CpuPreprocessingBatchingStage>(model_, batchsize_, preprocessor_, cpu_input_res_);
    }

    auto h2d_stage = std::make_shared<H2DBatchingDoneStage>(model_, batchsize_, dev_id_, cpu_input_res_, net_input_res_);
    batching_done_stages_.push_back(h2d_stage);
  }

  auto infer_stage =
      std::make_shared<InferBatchingDoneStage>(model_, batchsize_, dev_id_, net_input_res_, net_output_res_);
  batching_done_stages_.push_back(infer_stage);

  auto d2h_stage =
      std::make_shared<D2HBatchingDoneStage>(model_, batchsize_, dev_id_, net_output_res_, cpu_output_res_);
  batching_done_stages_.push_back(d2h_stage);

  if (batching_by_obj_) {
      obj_postproc_stage_ = std::make_shared<ObjPostprocessingBatchingDoneStage>(model_, batchsize_, dev_id_,
                                                                                 obj_postprocessor_, cpu_output_res_);
  } else {
    auto postproc_stage =
        std::make_shared<PostprocessingBatchingDoneStage>(model_, batchsize_, dev_id_, postprocessor_, cpu_output_res_);
    batching_done_stages_.push_back(postproc_stage);
  }
}

InferEngine::ResultWaitingCard InferEngine::FeedData(void* frame_info) {
  std::lock_guard<std::mutex> lk(mtx_);
  cached_frame_cnt_++;

  auto ret_promise = std::make_shared<std::promise<void>>();
  ResultWaitingCard card(ret_promise);
  auto auto_set_done = std::make_shared<AutoSetDone>(ret_promise, frame_info);

  if (batching_by_obj_) {
    batched_finfos_.push_back(std::make_pair(frame_info, auto_set_done));

    if (batched_finfos_.size() == batchsize_) {
      BatchingDone();
    }
  } else {
    InferTaskSptr task = batching_stage_->Batching(frame_info);
    thread_pool_->SubmitTask(task);

    batched_finfos_.push_back(std::make_pair(frame_info, auto_set_done));

    if (batched_finfos_.size() == batchsize_) {
      BatchingDone();
    }
  }

  return card;
}

void InferEngine::BatchingDone() {
  cached_frame_cnt_ = 0;

  if (batching_by_obj_) {
    obj_batching_stage_->Reset();
  } else {
    batching_stage_->Reset();
  }

  if (!batched_finfos_.empty()) {
    for (auto& stage : batching_done_stages_) {
      auto tasks = stage->BatchingDone(batched_finfos_);
      thread_pool_->SubmitTask(tasks);
    }

    if (batching_by_obj_) {
      auto tasks = obj_postproc_stage_->ObjBatchingDone(batched_finfos_, batched_objs_);
      thread_pool_->SubmitTask(tasks);
      batched_objs_.clear();
    }

    batched_finfos_.clear();
  }
}

}  // namespace cnstream
