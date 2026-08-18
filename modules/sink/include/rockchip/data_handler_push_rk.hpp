#ifndef MODULES_SINK_HANDLER_PUSH_RK_HPP_
#define MODULES_SINK_HANDLER_PUSH_RK_HPP_

#include "data_handler_push.hpp"

namespace cnstream {

/**
 * @brief Rockchip RKMPP 硬件编码推流处理器
 *
 *   - 使用 h264_rkmpp 硬件编码器
 *   - 通过 AV_HWDEVICE_TYPE_RKMPP（或 DRM）建立硬件设备上下文
 *   - 硬件帧格式为 AV_PIX_FMT_DRM_PRIME，软件帧格式为 AV_PIX_FMT_NV12
 *   - 使用 sws_scale 将 CPU BGR/RGB 转换到 NV12，再通过
 *     av_hwframe_transfer_data 上传到 DRM_PRIME 硬件帧后送入编码器
 */
class PushHandlerImRK : public PushHandlerIm {
 public:
  using PushHandlerIm::PushHandlerIm;

 protected:
  bool InitDeviceCtx() override;
  void CleanDeviceCtx() override;
  bool SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) override;

 private:
  /**
   * @brief 将 CPU 内存的 BGR/RGB 帧经 sws_scale 转到 NV12，
   *        再上传到 DRM_PRIME 硬件帧后送入编码器。
   */
  bool SendFrameFromCpu(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts);

  /**
   * @brief 取编码器声明的 HW_DEVICE_CTX，无声明返回 nullptr。
   */
  static const AVCodecHWConfig* GetHwDeviceConfig(const AVCodec* codec);

  /**
   * @brief 用指定后端(rkmpp/drm)创建设备/帧上下文并绑定到编码器，
   *        含预分配 hw_frame；失败时清理自身部分状态并返回 false
   */
  bool CreateRkHwContext(AVHWDeviceType type);

  /** 选一个可访问的 DRM 设备节点，无则返回 nullptr */
  static const char* PickDrmDevice();
};

}  // namespace cnstream

#endif  // MODULES_SINK_HANDLER_PUSH_RK_HPP_
