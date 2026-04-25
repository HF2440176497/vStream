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
#include <bitset>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include "cnstream_source.hpp"
#include "cnstream_eventbus.hpp"
#include "cnstream_pipeline.hpp"


namespace cnstream {

// #ifdef VSTREAM_UNIT_TEST
// 
// static std::mutex stream_idx_lock;
// static std::map<std::string, uint32_t> stream_idx_map;
// static std::bitset<MAX_STREAM_NUM> stream_bitset(0);
// 
// static uint32_t _GetStreamIndex(const std::string &stream_id) {
//   std::lock_guard<std::mutex>  guard(stream_idx_lock);
//   auto search = stream_idx_map.find(stream_id);
//   if (search != stream_idx_map.end()) {
//     return search->second;
//   }
//   for (uint32_t i = 0; i < GetMaxStreamNumber(); i++) {
//     if (!stream_bitset[i]) {
//       stream_bitset.set(i);
//       stream_idx_map[stream_id] = i;
//       return i;
//     }
//   }
//   return INVALID_STREAM_IDX;
// }
// 
// static int _ReturnStreamIndex(const std::string &stream_id) {
//   std::lock_guard<std::mutex>  guard(stream_idx_lock);
//   auto search = stream_idx_map.find(stream_id);
//   if (search == stream_idx_map.end()) {
//     return -1;
//   }
//   uint32_t stream_idx = search->second;
//   if (stream_idx >= GetMaxStreamNumber()) {
//     return -1;
//   }
//   stream_bitset.reset(stream_idx);
//   stream_idx_map.erase(search);
//   return 0;
// }
// 
// #endif

uint32_t SourceModule::GetStreamIndex(const std::string &stream_id) {
  RwLockReadGuard guard(container_lock_);
  if (container_) return container_->GetStreamIndex(stream_id);
  return INVALID_STREAM_IDX;
}

void SourceModule::ReturnStreamIndex(const std::string &stream_id) {
  RwLockReadGuard guard(container_lock_);
  if (container_) container_->ReturnStreamIndex(stream_id);
}

int SourceModule::AddSource(std::shared_ptr<SourceHandler> handler) {
  if (!handler) {
    LOGE(CORE) << "handler is null";
    return -1;
  }
  // param_set_ set in DataSource::Open
  if (!handler->CheckHandlerParams(param_set_)) {
    LOGE(CORE) << "handler check params failed";
    return -1;
  }
  if (!handler->SetHandlerParams(param_set_)) {
    LOGE(CORE) << "handler set params failed";
    return -1;
  }
  std::string stream_id = handler->GetStreamId();
  std::unique_lock<std::mutex> lock(mutex_);
  if (source_map_.find(stream_id) != source_map_.end()) {
    LOGE(CORE) << "[" << stream_id << "]: " << "Duplicate stream_id";
    return -1;
  }
  if (source_map_.size() >= GetMaxStreamNumber()) {
    LOGW(CORE) << "[" << stream_id << "]: "
               << " doesn't add to pipeline because of maximum limitation: " << GetMaxStreamNumber();
    return -1;
  }
  SetStreamRemoved(stream_id, false);
  LOGI(CORE) << "[" << handler->GetStreamId() << "]: " << "Stream opening...";
  if (handler->Open() != true) {
    LOGE(CORE) << "[" << stream_id << "]: " << "stream Open failed";
    return -1;
  }
  source_map_[stream_id] = handler;
  LOGI(CORE) << "Add stream success, stream id : [" << stream_id << "]";
  return 0;
}

int SourceModule::RemoveSource(std::shared_ptr<SourceHandler> handler, bool force) {
  if (!handler) {
    return -1;
  }
  return RemoveSource(handler->GetStreamId(), force);
}

std::shared_ptr<SourceHandler> SourceModule::GetSourceHandler(const std::string &stream_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (source_map_.find(stream_id) == source_map_.cend()) {
    return nullptr;
  }
  return source_map_[stream_id];
}

// int SourceModule::RemoveSource(const std::string &stream_id, bool force) {
//   LOGI(CORE) << "Begin to remove stream, stream id : [" << stream_id << "]";
//   SetStreamRemoved(stream_id, force);
//   // Close handler first
//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     auto iter = source_map_.find(stream_id);
//     if (iter == source_map_.end()) {
//       LOGW(CORE) << "stream named [" << stream_id << "] does not exist\n";
//       return 0;
//     }
//     LOGI(CORE) << "[" << stream_id << "]: "
//                << "Stream closing...";
//     iter->second->Close();
//     LOGI(CORE) << "[" << stream_id << "]: "
//                << "Stream close done";
//   }
//   // wait for eos reached
//   CheckStreamEosReached(stream_id, force);
//   SetStreamRemoved(stream_id, false);
//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     auto iter = source_map_.find(stream_id);
//     if (iter == source_map_.end()) {
//       LOGW(CORE) << "source does not exist\n";
//       return 0;  // 认为是成功的
//     }
//     source_map_.erase(iter);
//   }
//   LOGI(CORE) << "Finish removing stream, stream id : [" << stream_id << "]";
//   return 0;
// }

/**
 * @brief 调用 handler 的 Close 函数
 * 根据更改后的 SetStreamRemoved 函数，其用来表示清理状态
 * force: 仅仅表示是否等待 eos FrameInfo 的析构
 */
int SourceModule::RemoveSource(const std::string &stream_id, bool force) {
  LOGI(CORE) << "Begin to remove stream, stream id : [" << stream_id << "]";
  SetStreamRemoved(stream_id, true);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = source_map_.find(stream_id);
    if (iter == source_map_.end()) {
      LOGW(CORE) << "[" << stream_id << "]: source does not exist";
      return 0;
    }
    LOGI(CORE) << "[" << stream_id << "]: stream closing...";
    iter->second->Close();
    LOGI(CORE) << "[" << stream_id << "]: stream close done";
  }
  bool ret = CheckStreamEosReached(stream_id, force);
  if (!ret) {
    LOGW(CORE) << "[" << stream_id << "]: check stream eos, return false";
  }
  SetStreamRemoved(stream_id, false);  // 设置清除完成
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = source_map_.find(stream_id);
    if (iter == source_map_.end()) {
      LOGW(CORE) << "[" << stream_id << "]: source does not exist";
      return 0;  // 认为是成功的
    }
    source_map_.erase(iter);
  }
  LOGI(CORE) << "Finish removing stream, stream id : [" << stream_id << "]";
  return 0;
}


// int SourceModule::RemoveSources(bool force) {
//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     for (auto &iter : source_map_) {
//       SetStreamRemoved(iter.first, force);
//     }
//   }
//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     for (auto &iter : source_map_) {
//       iter.second->Stop();
//     }
//     for (auto &iter : source_map_) {
//       iter.second->Close();
//     }
//   }
//   {
//     std::unique_lock<std::mutex> lock(mutex_);
//     for (auto &iter : source_map_) {
//       CheckStreamEosReached(iter.first, force);
//       SetStreamRemoved(iter.first, false);
//     }
//     source_map_.clear();
//   }
//   return 0;
// }

/**
 * 直接复用 RemoveSource 函数
 */
int SourceModule::RemoveSources(bool force) {
  LOGI(CORE) << "Begin to remove all streams, force: " << std::boolalpha << force;
  std::vector<std::string> stream_ids;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto &iter : source_map_) {
      stream_ids.push_back(iter.first);
    }
  }
  for (const auto &stream_id : stream_ids) {
    LOGD(CORE) << "remove source stream_id: [" << stream_id << "]";
    RemoveSource(stream_id, force);
  }
  LOGI(CORE) << "Finish removing all streams";
  return 0;
}

/**
 * 调用处：SourceHandler::SendData
 */
bool SourceModule::SendData(const std::shared_ptr<FrameInfo> data) {
  if (!data->IsEos() && IsStreamRemoved(data->stream_id)) {
    return false;
  }
#ifdef VSTREAM_UNIT_TEST
  LOGD(SourceModule) << "SendData, stream_id: " << data->stream_id << ", ts: " << data->timestamp << std::endl;
#endif
  return TransmitData(data);
}

}  // namespace cnstream
