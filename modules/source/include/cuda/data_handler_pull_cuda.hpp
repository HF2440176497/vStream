#ifndef MODULES_SOURCE_HANDLER_PULL_CUDA_HPP_
#define MODULES_SOURCE_HANDLER_PULL_CUDA_HPP_

#include "cuda/cuda_check.hpp"
#include "data_handler_pull.hpp"

namespace cnstream {

class PullHandlerImCUDA : public PullHandlerIm {
 public:
  using PullHandlerIm::PullHandlerIm;
  ~PullHandlerImCUDA() override = default;

 protected:
  int codec_init() override;
  int decode_write() override;
  void clean_up() override;  // 释放 src_stream_ 后再走基类
  bool SupportHWDevice() override;
  void ConfigureOutputType() override;

 private:
  bool support_hwdevice();
  int init_hwdevice_conf();
  int hw_decoder_init();
  static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);
  std::shared_ptr<FrameInfo> ProcessFrameCUDA(AVFrame *p_frame, int &ret);

  enum AVPixelFormat hw_pix_fmt_ = AV_PIX_FMT_NONE;
  std::string type_name_ = "cuda";
};

}  // namespace cnstream


#endif