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
      obj_filter_(options.obj_filter()),
      dump_resized_image_dir_(options.dump_resized_image_dir()),
      batching_timeout_(options.batching_timeout()),
      device_id_(options.device_id()),
      batching_by_obj_(options.batching_by_obj()),
      module_name_(options.module_name()),
      error_func_(options.error_handler()) {

  batchsize_ = model_->get_batch_size();

  thread_pool_ = std::make_shared<InferThreadPool>();
  thread_pool_->SetErrorHandleFunc(error_func_);
  thread_pool_->Init(device_id_, batchsize_ * 3 + 4);

  cpu_input_res_ = std::make_shared<CpuInputResource>(model_);
  cpu_output_res_ = std::make_shared<CpuOutputResource>(model_);
  net_input_res_ = std::make_shared<NetInputResource>(model_);
  net_output_res_ = std::make_shared<NetOutputResource>(model_);

  cpu_input_res_->Init();
  cpu_output_res_->Init();
  net_input_res_->Init();
  net_output_res_->Init();

  StageAssemble();
  timeout_helper_.SetTimeout(batching_timeout_);

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
 * TODO: 暂时严格按照 prec - h2d - infer - d2h - postproc 顺序
 * 前处理暂时只有 CpuPreprocessing
 * 后处理 PostprocessingBatchingDoneStage 通过重载来区分是否是 CPU 的后处理
 */
void InferEngine::StageAssemble() {

  if (batching_by_obj_) {
    obj_batching_stage_ = std::make_shared<CpuPreprocessingObjBatchingStage>(model_, batchsize_, obj_preprocessor_,
                                                                            cpu_input_res_);
  } else {
    batching_stage_ =
        std::make_shared<CpuPreprocessingBatchingStage>(model_, batchsize_, preprocessor_, cpu_input_res_);
  }

  auto h2d_stage = std::make_shared<H2DBatchingDoneStage>(model_, batchsize_, device_id_, cpu_input_res_, net_input_res_);
  batching_done_stages_.push_back(h2d_stage);

  auto infer_stage =
      std::make_shared<InferBatchingDoneStage>(model_, batchsize_, device_id_, net_input_res_, net_output_res_);
  batching_done_stages_.push_back(infer_stage);

  auto d2h_stage =
      std::make_shared<D2HBatchingDoneStage>(model_, batchsize_, device_id_, net_output_res_, cpu_output_res_);
  batching_done_stages_.push_back(d2h_stage);

  if (batching_by_obj_) {
      obj_postproc_stage_ = std::make_shared<ObjPostprocessingBatchingDoneStage>(model_, batchsize_, device_id_,
                                                                                 obj_postprocessor_, cpu_output_res_);
  } else {
    auto postproc_stage =
        std::make_shared<PostprocessingBatchingDoneStage>(model_, batchsize_, device_id_, postprocessor_, cpu_output_res_);
    batching_done_stages_.push_back(postproc_stage);
  }
}

/**
 * @note: timeout_helper_ 保护 FeedData 不会被中断
 */
InferEngine::ResultWaitingCard InferEngine::FeedData(std::shared_ptr<FrameInfo> frame_info) {

  timeout_helper_.LockOperator();
  cached_frame_cnt_++;  // 表示当前 batch 内正在处理的 frame 的计数

  auto ret_promise = std::make_shared<std::promise<void>>();
  ResultWaitingCard card(ret_promise);
  auto auto_set_done = std::make_shared<AutoSetDone>(ret_promise, frame_info);  // destructor will set done
  ret_promise.reset();  // only use once

  if (batching_by_obj_) {

    if (!frame_info->collection.HasValue(kInferObjsTag)) {
      timeout_helper_.UnlockOperator();
      return card;
    }
    // objs_holder: std::vector<inferobjptr>, mutex
    InferObjsPtr objs_holder = frame_info->collection.Get<InferObjsPtr>(kInferObjsTag);
    objs_holder->mutex_.lock();
    std::vector<std::shared_ptr<InferObject>> objs = objs_holder->objs_;
    objs_holder->mutex_.unlock();

    for (int obj_idx = 0; obj_idx < objs.size(); ++obj_idx) {
      auto& obj = objs[obj_idx];  // shared_ptr<InferObject>

      if (obj_filter_) {
        if (!obj_filter_->Filter(frame_info, obj)) continue;
      }

      InferTaskSptr task = obj_batching_stage_->Batching(frame_info, obj);
      thread_pool_->SubmitTask(task);

      batched_finfos_.push_back(std::make_pair(frame_info, auto_set_done));
      batched_objs_.push_back(obj);

      if (batched_finfos_.size() == batchsize_) {
        BatchingDone();
        timeout_helper_.Reset(nullptr);
      } else {
        // TODO: 若 objs 数量非 batch_size 整数倍，存在超时丢弃
        // obj model 最好 batch_size == 1
        timeout_helper_.Reset([this]() -> void { BatchingDone(); });
      }
    }  // end for objs
    
    if (cached_frame_cnt_ >= batchsize_) {
      BatchingDone();
      timeout_helper_.Reset(nullptr);
    }

  } else {  // batching_by_obj_ = false

    // 对于前处理，task 封装对单张图像的操作
    InferTaskSptr task = batching_stage_->Batching(frame_info);
    thread_pool_->SubmitTask(task);
    batched_finfos_.push_back(std::make_pair(frame_info, auto_set_done));

    if (batched_finfos_.size() == batchsize_) {
      BatchingDone();
      timeout_helper_.Reset(nullptr);
    } else {
      timeout_helper_.Reset([this]() -> void { BatchingDone(); });
    }
  }
  timeout_helper_.UnlockOperator();
  return card;
}

/**
 * TODO: 简便起见，强制提交时 需要长度满足 batchsize_
 */
void InferEngine::ForceBatchingDone() {
  if (batched_finfos_.size() == batchsize_) {
    BatchingDone();
  }
}

// 正常调用条件：batched_finfos_.size == batch_size_
// 超时情况：Force submit, batched_finfos_ 数量不定
void InferEngine::BatchingDone() {
  cached_frame_cnt_ = 0;

  // obj_batching_stage_ 和 obj_postproc_stage_ 分别是前后处理

  // reset batch_idx
  if (batching_by_obj_) {  // params: object_infer
    obj_batching_stage_->Reset();
  } else {
    batching_stage_->Reset();
  }

  // TODO: 有可能超时，暂定丢弃
  if (!batched_finfos_.empty() && batched_finfos_.size() != batchsize_) {
    batched_finfos_.clear();
    return;
  }

  // h2d, infer, d2h, post(not obj)
  if (!batched_finfos_.empty()) {
    for (auto& stage : batching_done_stages_) {

      // note: 查看各派生类的实现 tasks 长度 == 1，一个 batch 提交为一个 task
      auto tasks = stage->BatchingDone(batched_finfos_);
      thread_pool_->SubmitTask(tasks);
    }

    // post(obj)
    if (batching_by_obj_) {
      auto tasks = obj_postproc_stage_->ObjBatchingDone(batched_finfos_, batched_objs_);
      thread_pool_->SubmitTask(tasks);
      batched_objs_.clear();
    }

    batched_finfos_.clear();
  }
}

}  // namespace cnstream
