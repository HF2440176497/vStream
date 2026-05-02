
#include "data_sink.hpp"
#include "cnstream_logging.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace cnstream {

class PushHandlerImpl {
 public:
  explicit PushHandlerImpl(DataSink *module, SinkHandler *handler)
      : module_(module), stream_id_(handler->GetStreamId()) {}

  bool Open() {
    LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerImpl Open";
    running_.store(true);
    return true;
  }

  void Stop() {
    LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerImpl Stop";
    running_.store(false);
  }

  void Close() {
    LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerImpl Close";
    running_.store(false);
  }

  /**
   * TODO: 目前仅实现原图推流，后续实现指定图像、特定标框逻辑等业务逻辑
   */
  int Process(const std::shared_ptr<FrameInfo> data) {
    LOGI(SINK) << "[" << stream_id_ << "]: Process frame, pts=" << data->timestamp;



    return 0;
  }

  DataSink *module_ = nullptr;
  std::string stream_id_;
  std::atomic<bool> running_{false};
  ModuleParamSet param_set_;
};

std::shared_ptr<SinkHandler> PushHandler::Create(DataSink *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SINK) << "[" << stream_id << "]: module is null";
    return nullptr;
  }
  return std::shared_ptr<SinkHandler>(new PushHandler(module, stream_id));
}

PushHandler::PushHandler(DataSink *module, const std::string &stream_id)
    : SinkHandler(module, stream_id) {
  impl_ = new PushHandlerImpl(module, this);
}

PushHandler::~PushHandler() {
  Close();
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

bool PushHandler::Open() {
  if (!module_) {
    LOGE(SINK) << "[" << stream_id_ << "]: module_ null";
    return false;
  }
  if (!impl_) {
    LOGE(SINK) << "[" << stream_id_ << "]: Push handler open failed, no memory left";
    return false;
  }
  return impl_->Open();
}

void PushHandler::Close() {
  if (impl_) {
    impl_->Close();
  }
}

void PushHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

int PushHandler::Process(const std::shared_ptr<FrameInfo> data) {
  if (!impl_) {
    return -1;
  }
  return impl_->Process(data);
}

void PushHandler::RegisterHandlerParams() {
  param_register_.Register(key_output_url, "Target URL for push stream (rtmp/rtsp).");
}

/**
 * @param params 来自 DataSink 的参数
 */
bool PushHandler::CheckHandlerParams(const ModuleParamSet& params) {
  if (params.find(key_output_url) == params.end()) {
    LOGE(SINK) << "[" << stream_id_ << "]: push output_url not set";
    return false;
  }
  return true;
}

/**
 * @brief CheckHandlerParams SetHandlerParams 是在 AddSink 调用的
 */
bool PushHandler::SetHandlerParams(const ModuleParamSet& params) {
  if (impl_) {
    impl_->param_set_ = params;
  }
  return true;
}

}  // namespace cnstream
