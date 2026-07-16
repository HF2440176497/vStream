
#include <memory>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include "cnstream_sink.hpp"
#include "cnstream_pipeline.hpp"

namespace cnstream {

int SinkModule::AddSink(std::shared_ptr<SinkHandler> handler) {
  if (!handler) {
    LOGE(CORE) << "handler is null";
    return -1;
  }
  // Reject AddSink if pipeline is stopping or not running
  Pipeline* pipeline = GetContainer();
  if (pipeline && (pipeline->IsStopping() || !pipeline->IsRunning())) {
    LOGE(CORE) << "[" << handler->GetStreamId() << "]: "
               << "AddSink rejected, pipeline is "
               << (pipeline->IsStopping() ? "stopping" : "not running");
    return -1;
  }
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

  // Double-check stopping_ after acquiring lock to prevent race
  if (pipeline && pipeline->IsStopping()) {
    LOGE(CORE) << "[" << stream_id << "]: "
               << "AddSink rejected, pipeline is stopping (post-lock check)";
    return -1;
  }

  auto it = sink_map_.find(stream_id);
  if (it != sink_map_.end()) {
    for (const auto& existing : it->second) {
      if (existing.get() == handler.get()) {
        LOGE(CORE) << "[" << stream_id << "]: " << "Duplicate handler";
        return -1;
      }
    }
  }
  LOGI(CORE) << "[" << stream_id << "]: " << "Sink opening...";
  if (handler->Open() != true) {
    LOGE(CORE) << "[" << stream_id << "]: " << "sink Open failed";
    return -1;
  }
  sink_map_[stream_id].push_back(handler);
  LOGI(CORE) << "Add sink success, stream id : [" << stream_id << "]";
  return 0;
}

int SinkModule::RemoveSink(std::shared_ptr<SinkHandler> handler, bool /*force*/) {
  if (!handler) {
    return -1;
  }
  std::string stream_id = handler->GetStreamId();
  LOGI(CORE) << "Begin to remove sink handler, stream id : [" << stream_id << "]";
  std::shared_ptr<SinkHandler> target;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = sink_map_.find(stream_id);
    if (it != sink_map_.end()) {
      auto& vec = it->second;
      for (auto vit = vec.begin(); vit != vec.end(); ++vit) {
        if (vit->get() == handler.get()) {
          target = *vit;
          vec.erase(vit);
          break;
        }
      }
      if (vec.empty()) {
        sink_map_.erase(it);
      }
    }
  }
  if (target) {
    LOGI(CORE) << "[" << stream_id << "]: sink closing...";
    target->Stop();
    target->Close();
    LOGI(CORE) << "[" << stream_id << "]: sink close done";
  } else {
    LOGW(CORE) << "[" << stream_id << "]: handler not found";
  }
  return 0;
}

std::vector<std::shared_ptr<SinkHandler>> SinkModule::GetSinkHandlers(const std::string &stream_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = sink_map_.find(stream_id);
  if (it == sink_map_.cend()) {
    return {};
  }
  return it->second;
}

int SinkModule::RemoveSink(const std::string &stream_id, bool /*force*/) {
  LOGI(CORE) << "Begin to remove sinks, stream id : [" << stream_id << "]";
  std::vector<std::shared_ptr<SinkHandler>> handlers;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = sink_map_.find(stream_id);
    if (iter == sink_map_.end()) {
      LOGW(CORE) << "[" << stream_id << "]: sink does not exist";
      return 0;
    }
    handlers = std::move(iter->second);
    sink_map_.erase(iter);
  }
  for (auto& handler : handlers) {
    if (handler) {
      LOGI(CORE) << "[" << stream_id << "]: sink closing...";
      handler->Stop();
      handler->Close();
      LOGI(CORE) << "[" << stream_id << "]: sink close done";
    }
  }
  LOGI(CORE) << "Finish removing sinks, stream id : [" << stream_id << "]";
  return 0;
}

int SinkModule::RemoveSinks(bool /*force*/) {
  LOGI(CORE) << "Begin to remove all sinks";
  std::vector<std::pair<std::string, std::vector<std::shared_ptr<SinkHandler>>>> all;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto &iter : sink_map_) {
      all.emplace_back(iter.first, iter.second);
    }
    sink_map_.clear();
  }
  for (auto &p : all) {
    for (auto &h : p.second) {
      if (h) {
        LOGD(CORE) << "remove sink stream_id: [" << p.first << "]";
        h->Stop();
        h->Close();
      }
    }
  }
  LOGI(CORE) << "Finish removing all sinks";
  return 0;
}

/**
 * @return 返回语义同 Process 函数，return 0 表示成功
 */
int SinkModule::DispatchData(const std::shared_ptr<FrameInfo> data) {
  if (!data) {
    return -1;
  }
  std::vector<std::shared_ptr<SinkHandler>> handlers;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = sink_map_.find(data->stream_id);
    if (iter == sink_map_.end()) {
      LOGW(CORE) << "No sink handler for stream [" << data->stream_id << "]";
      return 0;  // 可能未添加该流的 sink handler
    }
    handlers = iter->second;
  }
  int ret = 0;
  for (auto& handler : handlers) {
    if (handler) {
      int r = handler->Process(data);
      if (r != 0) ret = r;
    }
  }
  return ret;
}

}  // namespace cnstream
