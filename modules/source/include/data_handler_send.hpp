

#ifndef MODULES_SOURCE_HANDLER_SEND_HPP_
#define MODULES_SOURCE_HANDLER_SEND_HPP_

#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cnstream_logging.hpp"
#include "data_handler_util.hpp"
#include "data_source.hpp"
#include "data_source_param.hpp"

namespace cnstream {

/**
 * @brief Send 队列容量参数名
 * 注意：符号名与 data_sink.hpp 中的 key_queue_size 区分，避免重定义
 */
inline const std::string key_send_queue_size = "queue_size";

/**
 * @brief 发送图片
 * 提供发送接口，发送到队列，消费者不断取出向下游输送
 */
class SendHandlerImpl: public SourceRender {

  struct MatBufRef : public IDecBufRef {
    explicit MatBufRef(void* data) : data_(data) {}
    ~MatBufRef() override {
      delete[] static_cast<uint8_t*>(data_);
    }
    void* data_;
  };

 friend class SendHandler;

 public:
  explicit SendHandlerImpl(DataSource *module, SourceHandler *handler)
      : SourceRender(handler), module_(module), stream_id_(handler->GetStreamId()) {}

  SendRet Push(const SendFrame& send_frame, int wait_ms);
  bool Open();
  void Close();
  void Stop();
  void Loop();
  /** 解析 param_set_ 中的 queue_size 并重建队列，须在 Open 之前调用 */
  void SetupQueue();

public:
  void OnEndFrame();
  std::shared_ptr<FrameInfo> OnDecodeFrame(DecodeFrame* frame);

public:
  bool IsRunning() const { return running_; }

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  std::atomic<bool> running_{false};
  std::unique_ptr<ThreadSafeQueue<SendFrame>> image_queue_{
      std::make_unique<ThreadSafeQueue<SendFrame>>(40)};
  uint32_t queue_size_ = 40;

  std::thread thread_;  // consumer thread
  DataSource *module_;
  std::string stream_id_;
};

}  // namespace cnstream

#endif

