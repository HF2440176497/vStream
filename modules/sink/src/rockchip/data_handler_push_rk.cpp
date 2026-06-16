#include "data_handler_push.hpp"
#include "rockchip/data_handler_push_rk.hpp"

#include "cnstream_logging.hpp"
#include "cnstream_frame_va.hpp"
#include "data_common.hpp"
#include "memop.hpp"
#include "memop_factory.hpp"

#include <memory>
#include <opencv2/opencv.hpp>

namespace cnstream {

AVHWDeviceType PushHandlerImRK::DetectRkDeviceType() {
  // 优先使用 rkmpp 后端；若 FFmpeg 未编译 rkmpp，再回退到 drm。
  static const char* backend_names[] = {"rkmpp", "drm"};
  for (auto name : backend_names) {
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name(name);
    if (type != AV_HWDEVICE_TYPE_NONE) {
      LOGI(SINK) << "[" << stream_id_ << "]: RK HW device backend: " << name
                 << " (" << av_hwdevice_get_type_name(type) << ")";
      return type;
    }
  }
  LOGE(SINK) << "[" << stream_id_ << "]: rkmpp/drm HW device not found."
             << " Check `ffmpeg -hwaccels` for available backends.";
  return AV_HWDEVICE_TYPE_NONE;
}

bool PushHandlerImRK::InitDeviceCtx() {
  // 1. 探测 Rockchip 硬件后端
  AVHWDeviceType dev_type = DetectRkDeviceType();
  if (dev_type == AV_HWDEVICE_TYPE_NONE) {
    return false;
  }

  // 2. 创建硬件设备上下文
  int ret = av_hwdevice_ctx_create(&ctx_.hw_device_ctx, dev_type,
                                   nullptr, nullptr, 0);
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwdevice_ctx_create (rk) failed: "
               << errbuf << " (" << ret << ")";
    return false;
  }

  // 3. 分配硬件帧上下文：硬件帧为 DRM_PRIME，软件帧为 NV12
  ctx_.hw_frames_ctx = av_hwframe_ctx_alloc(ctx_.hw_device_ctx);
  if (!ctx_.hw_frames_ctx) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwframe_ctx_alloc failed";
    return false;
  }

  AVHWFramesContext* hw_frames = reinterpret_cast<AVHWFramesContext*>(ctx_.hw_frames_ctx->data);
  hw_frames->format            = AV_PIX_FMT_DRM_PRIME;
  hw_frames->sw_format         = AV_PIX_FMT_NV12;
  hw_frames->width             = width_;
  hw_frames->height            = height_;
  hw_frames->initial_pool_size = 20;

  ret = av_hwframe_ctx_init(ctx_.hw_frames_ctx);
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwframe_ctx_init failed: "
               << errbuf << " (" << ret << ")";
    return false;
  }

  // 4. 把硬件上下文挂到编码器上
  // 在结束时统一解除引用
  ctx_.codec_ctx->hw_device_ctx = av_buffer_ref(ctx_.hw_device_ctx);
  ctx_.codec_ctx->hw_frames_ctx = av_buffer_ref(ctx_.hw_frames_ctx);
  // 编码器看到的 pix_fmt 是硬件侧格式
  ctx_.codec_ctx->pix_fmt = AV_PIX_FMT_DRM_PRIME;

  // 5. 预分配一个硬件帧（用于 upload）
  ctx_.hw_frame = av_frame_alloc();
  if (!ctx_.hw_frame) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_alloc (hw_frame) failed";
    return false;
  }
  ret = av_hwframe_get_buffer(ctx_.hw_frames_ctx, ctx_.hw_frame, 0);
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwframe_get_buffer failed: "
               << errbuf << " (" << ret << ")";
    return false;
  }

  LOGI(SINK) << "[" << stream_id_ << "]: RK hardware encoder initialized: "
             << "hw_format=" << av_get_pix_fmt_name(hw_frames->format)
             << ", sw_format=" << av_get_pix_fmt_name(hw_frames->sw_format);
  return true;
}

void PushHandlerImRK::CleanDeviceCtx() {
  if (ctx_.hw_frame)      { av_frame_free(&ctx_.hw_frame); }
  if (ctx_.hw_frames_ctx) { av_buffer_unref(&ctx_.hw_frames_ctx); }
  if (ctx_.hw_device_ctx) { av_buffer_unref(&ctx_.hw_device_ctx); }
}

/**
 * @brief 针对 RKNN 平台，只能通过 sws_scale 转换
 */
bool PushHandlerImRK::SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) {
  return SendFrameFromCpu(frame, src_pix_fmt, pts);
}

bool PushHandlerImRK::SendFrameFromCpu(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) {
  if (!hw_ctx_initialized_.load()) {
    hw_ctx_initialized_.store(true);
  }
  const int src_width  = frame->GetWidth();
  const int src_height = frame->GetHeight();

#ifdef VSTREAM_UNIT_TEST
  const int src_stride = frame->GetStride(0);
  if (src_pix_fmt == AV_PIX_FMT_RGB24 || src_pix_fmt == AV_PIX_FMT_BGR24) {
    if (src_stride != GetStride_8U_C3(src_width)) {
      LOGE(SINK) << "[" << stream_id_ << "]: src_stride != GetStride_8U_C3(src_width)";
      return false;
    }
  }
#endif

  // 1. CPU BGR/RGB -> NV12 (sw_frame，CPU 侧)
  EnsureSwsContext(src_pix_fmt, src_width, src_height);
  int ret = av_frame_make_writable(ctx_.sw_frame);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_make_writable (sw_frame) failed";
    return false;
  }

  const uint8_t* src_data = static_cast<const uint8_t*>(frame->data_[0]->GetCpuData());
  int src_stride = frame->GetStride(0);
  sws_scale(ctx_.sws_ctx,
            &src_data, &src_stride,
            0, src_height,
            ctx_.sw_frame->data, ctx_.sw_frame->linesize);
  ctx_.sw_frame->pts = pts;

  // 2. NV12 (CPU) -> DRM_PRIME (硬件帧)
  ret = av_hwframe_transfer_data(ctx_.hw_frame, ctx_.sw_frame, 0);
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwframe_transfer_data (cpu->rk) failed: "
               << errbuf << " (" << ret << ")";
    return false;
  }
  ctx_.hw_frame->pts = pts;

  return EncodeFrame(ctx_.hw_frame);
}

}  // namespace cnstream
