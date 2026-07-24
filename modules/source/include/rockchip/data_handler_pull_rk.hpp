#ifndef MODULES_SOURCE_HANDLER_PULL_RK_HPP_
#define MODULES_SOURCE_HANDLER_PULL_RK_HPP_

#include "data_handler_pull.hpp"

namespace cnstream {

class PullHandlerImRK : public PullHandlerIm {
 public:
  using PullHandlerIm::PullHandlerIm;
  ~PullHandlerImRK() override = default;

 protected:
  int codec_init() override;
  int decode_write() override;
  void clean_up() override;
  bool SupportHWDevice() override;
  void ConfigureOutputType() override;

 private:
  bool support_hwdevice();
  int init_hwdevice_conf();
  int hw_decoder_init();
  static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);
  std::shared_ptr<FrameInfo> ProcessFrameRKMPP(AVFrame *p_frame, int &ret);
  static const char* pickDrmDevice();
  void precheckDeviceNodes();

  std::string type_name_ = "rkmpp";
  AVFrame *sw_frame_ = nullptr;  // hw(DRM_PRIME) -> sw(NV12) 中转帧
};

}  // namespace cnstream

#endif  // MODULES_SOURCE_HANDLER_PULL_RK_HPP_
