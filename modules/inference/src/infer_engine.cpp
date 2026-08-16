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
      postproc_on_device_(options.postproc_on_device()),
      batching_by_obj_(options.batching_by_obj()),
      module_name_(options.module_name()),
      error_func_(options.error_handler()),
      profiler_(options.profiler()),
      options_(options) {

  batchsize_ = model_->get_batch_size();

  if (batching_by_obj_ && batchsize_ != 1) {
    LOGE(INFER) << "[" << module_name_ << "] obj model requires batch_size == 1, but got "
                << batchsize_ << ". This may cause objects to be discarded.";
  }

  StageAssemble();

  // 异步流水线下，需按池深度扩容线程池
  const uint32_t async_depth = input_res_ ? input_res_->GetResPoolSize() : 1;
  const uint32_t extra_threads = async_depth > 1 ? async_depth : 0;
  thread_pool_ = std::make_shared<InferThreadPool>();
  thread_pool_->SetErrorHandleFunc(error_func_);
  thread_pool_->Init(batchsize_ * 3 + 4 + extra_threads);

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

  // 通用资源句柄是实际持有者；具体资源指针可能与其别名，避免重复
  if (input_res_) input_res_->Destroy();
  if (output_res_ && output_res_ != input_res_) output_res_->Destroy();

  if (cpu_input_res_ && cpu_input_res_ != input_res_) cpu_input_res_->Destroy();
  if (cpu_output_res_ && cpu_output_res_ != output_res_) cpu_output_res_->Destroy();
  if (net_input_res_ && net_input_res_ != input_res_) net_input_res_->Destroy();
  if (net_output_res_ && net_output_res_ != output_res_) net_output_res_->Destroy();
}

/**
 * 按设备类型选择 PipelineStrategy 组装流水线。
 * - CUDA/TRT：prec -> H2D -> infer -> D2H -> postproc
 * - CPU / host-visible：prec -> infer -> postproc（跳过 H2D/D2H）
 */
void InferEngine::StageAssemble() {
  if (!model_) {
    LOGE(INFER) << "InferEngine: model is null";
    return;
  }
  auto strategy = PipelineStrategy::Create(model_->GetDeviceType());
  if (!strategy) {
    LOGE(INFER) << "InferEngine: no pipeline strategy for device type "
                << DevType2Str(model_->GetDeviceType());
    return;
  }

  PipelineConfig config = strategy->Build(model_, options_);

  batching_stage_ = config.batching_stage;
  obj_batching_stage_ = config.obj_batching_stage;
  batching_done_stages_ = std::move(config.batching_done_stages);
  obj_postproc_stage_ = config.obj_postproc_stage;

  input_res_ = config.input_res;
  output_res_ = config.output_res;

  cpu_input_res_ = config.cpu_input_res;
  cpu_output_res_ = config.cpu_output_res;
  net_input_res_ = config.net_input_res;
  net_output_res_ = config.net_output_res;
}

/**
 * @note: timeout_helper_ 保护 FeedData 不会被中断
 */
InferEngine::ResultWaitingCard InferEngine::FeedData(std::shared_ptr<FrameInfo> frame_info) {

  timeout_helper_.LockOperator();

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

    // note: objs size not fixed
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
        timeout_helper_.Reset([this]() -> void { BatchingDone(); });
      }
    }  // end for objs

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
  timeout_helper_.LockOperator();
  if (batched_finfos_.size() == batchsize_) {
    BatchingDone();
    timeout_helper_.Reset(nullptr);
  }
  timeout_helper_.UnlockOperator();
}

// 正常调用：batched_finfos_.size == batch_size_
// 超时触发：批次不完整时使用 pad 策略，保证已有的帧不丢失
void InferEngine::BatchingDone() {
  // reset batch_idx
  if (batching_by_obj_) {
    obj_batching_stage_->Reset();
  } else {
    batching_stage_->Reset();
  }

  if (!batched_finfos_.empty() && batched_finfos_.size() != batchsize_) {
    // 使用最后一帧重复填充
    auto last_finfo = batched_finfos_.back();
    auto last_obj = batching_by_obj_ && !batched_objs_.empty() ? batched_objs_.back() : nullptr;
    while (batched_finfos_.size() < batchsize_) {
      auto pad_promise = std::make_shared<std::promise<void>>();
      auto pad_auto_set_done = std::make_shared<AutoSetDone>(pad_promise, last_finfo.first);
      batched_finfos_.push_back(std::make_pair(last_finfo.first, pad_auto_set_done));
      if (batching_by_obj_) {
        batched_objs_.push_back(last_obj);
      }
    }
  }

  // h2d, infer, d2h, post(not obj)
  if (!batched_finfos_.empty()) {
    for (auto& stage : batching_done_stages_) {

      // note: 查看各派生类的实现 tasks 长度 == 1，一个 batch 提交为一个 task
      auto tasks = stage->BatchingDone(batched_finfos_);
      thread_pool_->SubmitTask(tasks);
    }

    // post(obj)
    if (batching_by_obj_ && obj_postproc_stage_) {
      auto tasks = obj_postproc_stage_->ObjBatchingDone(batched_finfos_, batched_objs_);
      thread_pool_->SubmitTask(tasks);
      batched_objs_.clear();
    }

    batched_finfos_.clear();
  }
  return;
}

}  // namespace cnstream
