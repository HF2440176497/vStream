#ifndef MODULES_SOURCE_HANDLER_VIDEO_CUDA_HPP_
#define MODULES_SOURCE_HANDLER_VIDEO_CUDA_HPP_

#ifdef VSTREAM_USE_CUDA

#include "cuda/cuda_check.hpp"

#include "data_handler_video.hpp"

namespace cnstream {

class VideoHandlerImplCUDA : public VideoHandlerImpl {
 public:
  using VideoHandlerImpl::VideoHandlerImpl;
  ~VideoHandlerImplCUDA() {
    if (src_stream_) {
      cudaStreamDestroy(static_cast<cudaStream_t>(src_stream_));
      src_stream_ = nullptr;
    }
  }

 protected:
  int codec_init() override;
  int decode_write() override;
  bool SupportHWDevice() override;
  void ConfigureOutputType() override;

 private:
  bool support_hwdevice();
  int init_hwdevice_conf();
  int hw_decoder_init();
  static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);
  std::shared_ptr<FrameInfo> ProcessFrameCPU(AVFrame *p_frame, AVFrame *sw_frame, int &ret);
  std::shared_ptr<FrameInfo> ProcessFrameCUDA(AVFrame *p_frame, int &ret);

  std::string type_name_ = "cuda";
};

}  // namespace cnstream

#endif  // VSTREAM_USE_CUDA

#endif