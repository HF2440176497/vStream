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

#include "data_sink.hpp"
#include "cnstream_collection.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"
#include "util/cnstream_queue.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace cnstream {

class QueueHandlerImpl {
 public:
  explicit QueueHandlerImpl(DataSink *module, SinkHandler *handler)
      : module_(module), stream_id_(handler->GetStreamId()) {}

  bool Open() {
    LOGI(SINK) << "[" << stream_id_ << "]: QueueHandlerImpl Open, queue_size=" << queue_size_;
    running_.store(true);
    queue_ = std::make_unique<ThreadSafeQueue<s_output_data>>(queue_size_);
    return true;
  }

  void Stop() {
    LOGI(SINK) << "[" << stream_id_ << "]: QueueHandlerImpl Stop";
    running_.store(false);
  }

  void Close() {
    LOGI(SINK) << "[" << stream_id_ << "]: QueueHandlerImpl Close";
    running_.store(false);
    if (queue_) {
      queue_->Stop();
      queue_.reset();
    }
  }

  /**
   * @brief Extracts output data from FrameInfo and pushes it into the internal queue.
   * 外界需要主动调用 GetData 获取 QueueHandler 组装的数据
   */
  int Process(const std::shared_ptr<FrameInfo> frame_info) {
    if (!running_.load()) {
      LOGW(SINK) << "[" << stream_id_ << "]: QueueHandler not running, skip frame";
      return -1;
    }

    if (frame_info->IsInvalid() || frame_info->IsRemoved()) {
      LOGW(SINK) << "[" << stream_id_ << "]: frame_info has problems";
      return -1;
    }

    LOGI(SINK) << "[" << stream_id_ << "]: OnFrame timestamp=" << frame_info->timestamp;

    s_output_data data;

    if (frame_info->collection.HasValue(cnstream::kDataFrameTag)) {
      auto img_data = frame_info->collection.Get<cnstream::DataFramePtr>(cnstream::kDataFrameTag);
      data.image_dict[output_constants::key_original_image] = img_data->GetImage();
    }

    data.frame_id_s = frame_info->frame_id_s;
    data.timestamp = frame_info->timestamp;

    InferObjsPtr objs_holder = nullptr;
    if (frame_info->collection.HasValue(cnstream::kInferObjsTag)) {
      objs_holder = frame_info->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
    } else {
      LOGW(SINK) << "[" << stream_id_ << "]: frame_info has no infer objs";
      data.result = -1;
      return 0;  // 没有推理结果也返回0，表示处理完成
    }

    {
      std::lock_guard<std::mutex> lk(objs_holder->mutex_);
      for (auto re : objs_holder->objs_) {
        s_obj_in data_obj;
        data_obj.str_id = re->id;
        data_obj.track_id = re->track_id;
        data_obj.score = re->score;
        data_obj.model_name = re->model_name;

        std::vector<int> bboxs;
        bboxs.push_back(re->bbox.x);
        bboxs.push_back(re->bbox.y);
        bboxs.push_back(re->bbox.w);
        bboxs.push_back(re->bbox.h);
        data_obj.bboxs = bboxs;
        data.objects.push_back(data_obj);
      }
    }

    data.result = 0;
    Push(data);
    return 0;
  }

  /**
   * 直接调用 queue_ 非阻塞式推送
   */
  bool Push(const s_output_data& data) {
    if (!queue_) {
      return false;
    }
    return queue_->Push(data);
  }

  bool GetData(s_output_data& data, int wait_ms) {
    if (!queue_) {
      return false;
    }
    if (wait_ms < 0) {
      queue_->WaitAndPop(data);
      if (data.result != 0) {
        return false;
      }
      return true;
    } else if (wait_ms == 0) {
      return queue_->TryPop(data);
    } else {
      std::chrono::milliseconds timeout(wait_ms);
      return queue_->WaitAndTryPop(data, timeout);
    }
    return false;
  }

  s_output_data GetData(int wait_ms) {
    s_output_data data;
    data.result = -1;
    if (GetData(data, wait_ms)) {
      return data;
    }
    return s_output_data();
  }

  DataSink *module_ = nullptr;
  std::string stream_id_;
  std::atomic<bool> running_{false};
  ModuleParamSet param_set_;
  uint32_t queue_size_ = 20;
  std::unique_ptr<ThreadSafeQueue<s_output_data>> queue_;
};

std::shared_ptr<SinkHandler> QueueHandler::Create(DataSink *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SINK) << "[" << stream_id << "]: module is null";
    return nullptr;
  }
  return std::shared_ptr<SinkHandler>(new QueueHandler(module, stream_id));
}

QueueHandler::QueueHandler(DataSink *module, const std::string &stream_id)
    : SinkHandler(module, stream_id) {
  impl_ = new QueueHandlerImpl(module, this);
}

QueueHandler::~QueueHandler() {
  Close();
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

bool QueueHandler::Open() {
  if (!module_) {
    LOGE(SINK) << "[" << stream_id_ << "]: module_ null";
    return false;
  }
  if (!impl_) {
    LOGE(SINK) << "[" << stream_id_ << "]: Queue handler open failed, no memory left";
    return false;
  }
  return impl_->Open();
}

void QueueHandler::Close() {
  if (impl_) {
    impl_->Close();
  }
}

void QueueHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

int QueueHandler::Process(const std::shared_ptr<FrameInfo> data) {
  if (!impl_) {
    return -1;
  }
  return impl_->Process(data);
}

void QueueHandler::RegisterHandlerParams() {
  param_register_.Register(key_queue_size, "Size of the output queue. Default is 20.");
}


bool QueueHandler::CheckHandlerParams(const ModuleParamSet& params) {
  if (params.find(key_queue_size) != params.end()) {
    int size = std::stoi(params.at(key_queue_size));
    if (size <= 0) {
      LOGE(SINK) << "[" << stream_id_ << "]: queue_size must be positive";
      return false;
    }
  }
  return true;
}

bool QueueHandler::SetHandlerParams(const ModuleParamSet& params) {
  if (impl_) {
    impl_->param_set_ = params;
    if (params.find(key_queue_size) != params.end()) {
      impl_->queue_size_ = static_cast<uint32_t>(std::stoi(params.at(key_queue_size)));
    } else {
      impl_->queue_size_ = 20;
    }
  }  // end if (impl_)
  return true;
}

bool QueueHandler::GetData(s_output_data& data, int wait_ms) {
  if (!impl_) {
    return false;
  }
  return impl_->GetData(data, wait_ms);
}

s_output_data QueueHandler::GetData(int wait_ms) {
  if (!impl_) {
    return s_output_data();
  }
  return impl_->GetData(wait_ms);
}

}  // namespace cnstream
