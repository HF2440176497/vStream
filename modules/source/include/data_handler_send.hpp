

#ifndef MODULES_SOURCE_HANDLER_SEND_HPP_
#define MODULES_SOURCE_HANDLER_SEND_HPP_

#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cnstream_logging.hpp"
#include "data_handler_util.hpp"
#include "data_source.hpp"
#include "data_source_param.hpp"


namespace cnstream {

/**
 * @brief 发送图片
 * 提供发送接口，发送到队列，消费者不断取出向下游输送
 */
class SendHandlerImpl: public SourceRender {

  struct SendFrame {
    uint64_t pts;
    std::string frame_id_s;
    cv::Mat image;
  };

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
  
  bool Push(const SendFrame& send_frame);
  bool Open();
  void Close();
  void Stop();
  void Loop();

public:
  void OnEndFrame();
  std::shared_ptr<FrameInfo> OnDecodeFrame(DecodeFrame* frame);

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  std::atomic<bool> running_{false};
  ThreadSafeQueue<SendFrame> image_queue_{20};

  std::thread thread_;  // consumer thread
  DataSource *module_;
  std::string stream_id_;
};

}  // namespace cnstream

#endif

