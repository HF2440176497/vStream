/**
 * @brief 提供生产者-消费者的接口，读取视频流（支持 NVIDIA CUDA 硬件解码）
 */

#ifndef MODULES_SOURCE_HANDLER_VIDEO_HPP_
#define MODULES_SOURCE_HANDLER_VIDEO_HPP_

#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cnstream_logging.hpp"
#include "data_handler_util.hpp"
#include "data_source.hpp"
#include "data_source_param.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
}

namespace cnstream {

class PullHandlerIm : public SourceRender {
  friend class PullHandler;
  
 public:
  struct MatBufRef : public IDecBufRef {
    explicit MatBufRef(void* data) : data_(data) {}
    ~MatBufRef() override {
      delete[] static_cast<uint8_t*>(data_);
    }
    void* data_;
  };

  struct MatBufRefNV12 : public IDecBufRef {
    MatBufRefNV12(void* y_data, void* uv_data) : y_data_(y_data), uv_data_(uv_data) {}
    ~MatBufRefNV12() override {
      delete[] static_cast<uint8_t*>(y_data_);
      delete[] static_cast<uint8_t*>(uv_data_);
    }
    void* y_data_;
    void* uv_data_;
  };

  explicit PullHandlerIm(DataSource *module, SourceHandler *handler)
      : SourceRender(handler), module_(module), stream_id_(handler->GetStreamId()) {}
  virtual ~PullHandlerIm() = default;

  bool Open();
  void Close();
  void Stop();
  void Loop();

 public:
  void OnEndFrame();
  std::shared_ptr<FrameInfo> OnDecodeFrame(DecodeFrame* frame);

 protected:
  virtual int codec_init() = 0;
  virtual void clean_up();

  virtual int decode_write() = 0;
  virtual bool SupportHWDevice() { return true; }
  virtual void ConfigureOutputType() {}

  int input_format_init();
  
 public:
  bool IsRunning() const { return running_; }

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 protected:
#endif
  std::atomic<bool> running_{false};
  std::thread        thread_;

  int         interval_    = 0;
  int         device_id_   = -1;
  std::string stream_url_;
  int         frame_rate_  = 10;

  DataSource *module_;
  std::string stream_id_;

  AVFormatContext    *ifmt_ctx_      = nullptr;
  AVDictionary       *ifmt_opts_     = nullptr;
  int                 video_index_   = -1;

  AVBufferRef        *hw_device_ctx_ = nullptr;
  enum AVHWDeviceType device_type_   = AV_HWDEVICE_TYPE_NONE;
  
  AVCodec           *codec_        = nullptr;
  AVCodecContext    *codec_ctx_    = nullptr;
  AVCodecParameters *codecpar_     = nullptr;
  AVPacket           pkt_;

  OutputType output_type_ = OutputType::OUTPUT_CPU;
};

class PullHandlerImCPU : public PullHandlerIm {
 public:
  using PullHandlerIm::PullHandlerIm;

 protected:
  int codec_init() override;
  int decode_write() override;

 private:
  std::shared_ptr<FrameInfo> ProcessFrame(AVFrame *p_frame, int &ret);
};

}  // namespace cnstream

#ifdef VSTREAM_USE_CUDA
#include "cuda/data_handler_pull_cuda.hpp"
#elif defined(VSTREAM_USE_ROCKCHIP)
#include "rockchip/data_handler_pull_rk.hpp"
#endif

#endif