
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
  if (sink_map_.find(stream_id) != sink_map_.end()) {
    LOGE(CORE) << "[" << stream_id << "]: " << "Duplicate stream_id";
    return -1;
  }
  LOGI(CORE) << "[" << stream_id << "]: " << "Sink opening...";
  if (handler->Open() != true) {
    LOGE(CORE) << "[" << stream_id << "]: " << "sink Open failed";
    return -1;
  }
  sink_map_[stream_id] = handler;
  LOGI(CORE) << "Add sink success, stream id : [" << stream_id << "]";
  return 0;
}

int SinkModule::RemoveSink(std::shared_ptr<SinkHandler> handler, bool force) {
  if (!handler) {
    return -1;
  }
  return RemoveSink(handler->GetStreamId(), force);
}

std::shared_ptr<SinkHandler> SinkModule::GetSinkHandler(const std::string &stream_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (sink_map_.find(stream_id) == sink_map_.cend()) {
    return nullptr;
  }
  return sink_map_[stream_id];
}

int SinkModule::RemoveSink(const std::string &stream_id, bool force) {
  LOGI(CORE) << "Begin to remove sink, stream id : [" << stream_id << "]";
  std::shared_ptr<SinkHandler> handler;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = sink_map_.find(stream_id);
    if (iter == sink_map_.end()) {
      LOGW(CORE) << "[" << stream_id << "]: sink does not exist";
      return 0;
    }
    handler = iter->second;
    sink_map_.erase(iter);
  }
  if (handler) {
    LOGI(CORE) << "[" << stream_id << "]: sink closing...";
    handler->Stop();
    handler->Close();
    LOGI(CORE) << "[" << stream_id << "]: sink close done";
  }
  LOGI(CORE) << "Finish removing sink, stream id : [" << stream_id << "]";
  return 0;
}

int SinkModule::RemoveSinks(bool force) {
  LOGI(CORE) << "Begin to remove all sinks, force: " << std::boolalpha << force;
  std::vector<std::string> stream_ids;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto &iter : sink_map_) {
      stream_ids.push_back(iter.first);
    }
  }
  for (const auto &stream_id : stream_ids) {
    LOGD(CORE) << "remove sink stream_id: [" << stream_id << "]";
    RemoveSink(stream_id, force);
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
  std::shared_ptr<SinkHandler> handler;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = sink_map_.find(data->stream_id);
    if (iter == sink_map_.end()) {
      LOGW(CORE) << "No sink handler for stream [" << data->stream_id << "]";
      return 0;  // 可能未添加该流的 sink handler
    }
    handler = iter->second;
  }
  if (!handler) {
    return -1;
  }
  return handler->Process(data);
}

}  // namespace cnstream
