
/**
 * 提供生产者-消费者的接口，读取图片
 */

#ifndef MODULES_SOURCE_HANDLER_IMAGE_QUEUE_HPP_
#define MODULES_SOURCE_HANDLER_IMAGE_QUEUE_HPP_

#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cnstream_logging.hpp"
#include "data_handler_util.hpp"  // SourceRender
#include "data_source.hpp"  // DataSource
#include "data_source_param.hpp"  // DataSourceParam


namespace cnstream {

class ImageHandlerImpl: public SourceRender {
  struct MatBufRef : public IDecBufRef {
    explicit MatBufRef(void* data) : data_(data) {}
    ~MatBufRef() override {
      delete[] static_cast<uint8_t*>(data_);
    }
    void* data_;
  };

 friend class ImageHandler;

 public:
  explicit ImageHandlerImpl(DataSource *module, SourceHandler *handler)
      : SourceRender(handler), module_(module), stream_id_(handler->GetStreamId()) {}
  
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

  std::string image_path_;
  int frame_rate_ = 10;

  cv::Mat image_;
  std::thread thread_;  // consumer thread
  DataSource *module_;
  std::string stream_id_;
  int frame_index_ = 0;
};

}  // namespace cnstream

#endif
