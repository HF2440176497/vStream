/**
 * @brief 提供生产者-消费者的接口，读取视频流（支持 NVIDIA CUDA 硬件解码）
 */

#ifndef MODULES_SOURCE_HANDLER_VIDEO_HPP_
#define MODULES_SOURCE_HANDLER_VIDEO_HPP_

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

class VideoHandlerImpl : public SourceRender {
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

  friend class VideoHandler;

 public:
  explicit VideoHandlerImpl(DataSource *module, SourceHandler *handler)
      : SourceRender(handler), module_(module), stream_id_(handler->GetStreamId()) {}

  bool Open();
  void Close();
  void Stop();
  void Loop();

 public:
  void OnEndFrame();
  std::shared_ptr<FrameInfo> OnDecodeFrame(DecodeFrame* frame);

 private:
  bool support_hwdevice();
  int init_hwdevice_conf();
  int hw_decoder_init();
  int codec_init();
  int input_format_init();
  int convert_frame_init();
  int decode_write();
  void clean_up();

 private:
  static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  std::atomic<bool> running_{false};
  std::thread thread_;

  std::string stream_url_;
  int frame_rate_ = 10;

  DataSource *module_;
  std::string stream_id_;

  AVFormatContext *ifmt_ctx_ = nullptr;
  AVDictionary *ifmt_opts_ = nullptr;
  int video_index_ = -1;

  AVFrame *s_frame_ = nullptr;  // sws_scale 待转换帧
  AVFrame *cv_frame_ = nullptr;  // sws_scale 转换后的帧
  uint8_t *cv_buf_ = nullptr;

  enum AVHWDeviceType device_type_ = AV_HWDEVICE_TYPE_NONE;
  AVBufferRef *hw_device_ctx_ = nullptr;

  AVCodec *codec_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  AVCodecParameters *codecpar_ = nullptr;
  AVPacket pkt_;
  struct SwsContext *sws_ctx_ = nullptr;

  std::string type_name_ = "cuda";

  int device_id_ = -1;
  OutputType output_type_ = OutputType::OUTPUT_CPU;
};

}  // namespace cnstream

#endif
