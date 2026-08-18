#include "data_handler_push.hpp"
#include "rockchip/data_handler_push_rk.hpp"

#include "cnstream_logging.hpp"
#include "cnstream_frame_va.hpp"
#include "data_common.hpp"
#include "memop.hpp"
#include "memop_factory.hpp"

#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <opencv2/opencv.hpp>

namespace cnstream {

const AVCodecHWConfig* PushHandlerImRK::GetHwDeviceConfig(const AVCodec* codec) {
  for (int i = 0;; i++) {
    const AVCodecHWConfig* cfg = avcodec_get_hw_config(codec, i);
    if (!cfg) break;
    if (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
      return cfg;
    }
  }
  return nullptr;
}

const char* PushHandlerImRK::PickDrmDevice() {
  static const char* kCandidates[] = {
    "/dev/dri/card0",
    "/dev/dri/card1",
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    nullptr
  };
  for (int i = 0; kCandidates[i]; ++i) {
    if (access(kCandidates[i], R_OK | W_OK) == 0) {
      return kCandidates[i];
    }
  }
  return nullptr;
}

bool PushHandlerImRK::CreateRkHwContext(AVHWDeviceType type) {
  const char* type_name = av_hwdevice_get_type_name(type);
  // rkmpp 后端内部自己 open("/dev/mpp_service")，设备路径对它无效；
  // drm 后端才需要 DRM 设备节点路径
  const bool is_rkmpp = type_name && std::string(type_name) == "rkmpp";
  const char* device = is_rkmpp ? nullptr : PickDrmDevice();
  if (!is_rkmpp && !device) {
    LOGW(SINK) << "[" << stream_id_ << "]: no accessible DRM device node, skip backend "
               << (type_name ? type_name : "unknown");
    return false;
  }

  auto release_partial = [this] {
    if (ctx_.hw_frames_ctx) av_buffer_unref(&ctx_.hw_frames_ctx);
    if (ctx_.hw_device_ctx) av_buffer_unref(&ctx_.hw_device_ctx);
  };
  char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};

  // 1. 创建硬件设备上下文
  int ret = av_hwdevice_ctx_create(&ctx_.hw_device_ctx, type, device, nullptr, 0);
  if (ret < 0) {
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGW(SINK) << "[" << stream_id_ << "]: av_hwdevice_ctx_create (" << type_name
               << ") failed: " << errbuf << " (" << ret << ")";
    release_partial();
    return false;
  }

  // 2. 分配硬件帧上下文：硬件帧为 DRM_PRIME，软件帧为 NV12
  AVBufferRef* frames_ref = av_hwframe_ctx_alloc(ctx_.hw_device_ctx);
  if (!frames_ref) {
    LOGW(SINK) << "[" << stream_id_ << "]: av_hwframe_ctx_alloc (" << type_name << ") failed";
    release_partial();
    return false;
  }
  AVHWFramesContext* hw_frames = reinterpret_cast<AVHWFramesContext*>(frames_ref->data);
  hw_frames->format            = AV_PIX_FMT_DRM_PRIME;
  hw_frames->sw_format         = AV_PIX_FMT_NV12;
  hw_frames->width             = width_;
  hw_frames->height            = height_;
  hw_frames->initial_pool_size = 20;

  ret = av_hwframe_ctx_init(frames_ref);
  if (ret < 0) {
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGW(SINK) << "[" << stream_id_ << "]: av_hwframe_ctx_init (" << type_name
               << ") failed: " << errbuf << " (" << ret << ")";
    av_buffer_unref(&frames_ref);
    release_partial();
    return false;
  }

  // 3. 预分配一个硬件帧（用于 upload），全部成功后才绑定编码器，避免半初始化状态
  ctx_.hw_frame = av_frame_alloc();
  if (!ctx_.hw_frame) {
    LOGW(SINK) << "[" << stream_id_ << "]: av_frame_alloc (hw_frame) failed";
    av_buffer_unref(&frames_ref);
    release_partial();
    return false;
  }
  ret = av_hwframe_get_buffer(frames_ref, ctx_.hw_frame, 0);
  if (ret < 0) {
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGW(SINK) << "[" << stream_id_ << "]: av_hwframe_get_buffer (" << type_name
               << ") failed: " << errbuf << " (" << ret << ")";
    av_frame_free(&ctx_.hw_frame);
    av_buffer_unref(&frames_ref);
    release_partial();
    return false;
  }

  // 4. 绑定到编码器上下文：设备、帧上下文、硬件侧像素格式
  ctx_.hw_frames_ctx = frames_ref;  // 接管引用
  ctx_.codec_ctx->hw_device_ctx = av_buffer_ref(ctx_.hw_device_ctx);
  ctx_.codec_ctx->hw_frames_ctx = av_buffer_ref(ctx_.hw_frames_ctx);
  ctx_.codec_ctx->pix_fmt = AV_PIX_FMT_DRM_PRIME;

  LOGI(SINK) << "[" << stream_id_ << "]: RK HW context ready: backend=" << type_name
             << ", device=" << (device ? device : "<auto>")
             << ", hw_format=" << av_get_pix_fmt_name(AV_PIX_FMT_DRM_PRIME)
             << ", sw_format=" << av_get_pix_fmt_name(AV_PIX_FMT_NV12);
  return true;
}

bool PushHandlerImRK::InitDeviceCtx() {
  // 候选后端：编码器 hw config 声明的类型优先, 之后按 rkmpp -> drm 顺序补充回退项
  const AVCodec* codec = ctx_.codec_ctx->codec;
  if (!codec) {
    LOGE(SINK) << "[" << stream_id_ << "]: codec_ctx has no codec";
    return false;
  }

  std::vector<AVHWDeviceType> candidates;
  const AVCodecHWConfig* declared = GetHwDeviceConfig(codec);
  if (declared) {
    LOGI(SINK) << "[" << stream_id_ << "]: encoder " << codec->name
               << " declares HW device type: " << av_hwdevice_get_type_name(declared->device_type);
    candidates.push_back(declared->device_type);
  }
  for (const char* name : {"rkmpp", "drm"}) {
    AVHWDeviceType t = av_hwdevice_find_type_by_name(name);
    if (t == AV_HWDEVICE_TYPE_NONE) continue;
    bool dup = false;
    for (AVHWDeviceType c : candidates) {
      if (c == t) { dup = true; break; }
    }
    if (!dup) candidates.push_back(t);
  }
  if (candidates.empty()) {
    LOGE(SINK) << "[" << stream_id_ << "]: no RK HW backend (rkmpp/drm) available."
               << " Check `ffmpeg -hwaccels` for available backends.";
    return false;
  }

  for (AVHWDeviceType type : candidates) {
    if (CreateRkHwContext(type)) {
      return true;
    }
    LOGW(SINK) << "[" << stream_id_ << "]: backend "
               << av_hwdevice_get_type_name(type) << " failed, try next backend";
  }

  LOGE(SINK) << "[" << stream_id_ << "]: all RK HW backends failed."
             << " Checklist: 1) /dev/mpp_service exists"
             << " 2) /dev/dri/card* /dev/dri/renderD* accessible"
             << " 3) process permission (video/render group)";
  return false;
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
  const int src_width  = frame->GetWidth();
  const int src_height = frame->GetHeight();

#ifdef VSTREAM_UNIT_TEST
  const int _src_stride = frame->GetStride(0);
  if (src_pix_fmt == AV_PIX_FMT_RGB24 || src_pix_fmt == AV_PIX_FMT_BGR24) {
    if (_src_stride != GetStride_8U_C3(src_width)) {
      LOGE(SINK) << "[" << stream_id_ << "]: _src_stride != GetStride_8U_C3(src_width)";
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
  // 释放上一帧的硬件缓冲区引用（编码器可能仍持有引用），
  // 从池中获取新的空闲缓冲区，避免写入编码器正在读取的缓冲区。
  av_frame_unref(ctx_.hw_frame);
  ret = av_hwframe_get_buffer(ctx_.hw_frames_ctx, ctx_.hw_frame, 0);
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwframe_get_buffer failed: "
               << errbuf << " (" << ret << ")";
    return false;
  }
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
