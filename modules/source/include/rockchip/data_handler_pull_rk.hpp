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
  bool SupportHWDevice() override;
  void ConfigureOutputType() override;

 private:
  bool support_hwdevice();
  // 检查解码器是否声明支持指定后端(HW_DEVICE_CTX 方式且输出 DRM_PRIME)
  bool CheckHwConfig(enum AVHWDeviceType type);
  // 依序尝试 rkmpp -> drm 后端创建硬件设备，全部失败返回 -1
  int hw_decoder_init();
  static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);

  std::shared_ptr<FrameInfo> ProcessFrameRKMPP(AVFrame *p_frame);
  static const char* pickDrmDevice();
  void precheckDeviceNodes();

  std::string type_name_ = "rkmpp";

  // 连续单帧处理失败计数：超过阈值判定流与硬解通路不兼容，终止拉流
  static constexpr uint32_t kMaxFrameErrorCnt = 64;
  uint32_t frame_error_cnt_ = 0;
};

}  // namespace cnstream

#endif  // MODULES_SOURCE_HANDLER_PULL_RK_HPP_
