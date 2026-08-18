#include "data_source.hpp"
#include "rockchip/data_handler_pull_rk.hpp"

#include "data_source_param.hpp"
#include "cnstream_source.hpp"

#include <memory>
#include <sstream>
#include <unordered_map>
#include <cstring>
#include <unistd.h>

namespace cnstream {

namespace {
// 持有 av_hwframe_transfer_data 输出的软件帧(NV12)，plane 内存直接交给 DecodeFrame 引用
struct AVFrameBufRef : public IDecBufRef {
  explicit AVFrameBufRef(AVFrame* frame) : frame_(frame) {}
  ~AVFrameBufRef() override { av_frame_free(&frame_); }
  AVFrame* frame_;
};
}  // namespace

bool PullHandlerImRK::support_hwdevice() {
  precheckDeviceNodes();
  const char* backend_names[] = {"rkmpp", "drm"};
  for (auto name : backend_names) {
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name(name);
    if (type != AV_HWDEVICE_TYPE_NONE) {
      device_type_ = type;
      type_name_ = name;
      LOGI(SOURCE) << "[" << stream_id_ << "]: HW device type: " << type_name_
                   << " (" << av_hwdevice_get_type_name(type) << ")";
      return true;
    }
  }
  LOGE(SOURCE) << "[" << stream_id_ << "]: rkmpp/drm HW device not found."
               << " Check `ffmpeg -hwaccels` for available backends.";
  return false;
}

enum AVPixelFormat PullHandlerImRK::get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
    if (*p == AV_PIX_FMT_DRM_PRIME) {
      return *p;
    }
  }
  LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to find HW surface format.";
  return AV_PIX_FMT_NONE;
}

bool PullHandlerImRK::CheckHwConfig(enum AVHWDeviceType type) {
  for (int i = 0;; i++) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(codec_, i);
    if (!config) {
      return false;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
        config->device_type == type) {
      if (config->pix_fmt != AV_PIX_FMT_DRM_PRIME) {
        LOGW(SOURCE) << "[" << stream_id_ << "]: " << codec_->name
                     << " does not support DRM_PRIME with device type "
                     << av_hwdevice_get_type_name(type);
        return false;
      }
      return true;
    }
  }
}

int PullHandlerImRK::hw_decoder_init() {
  // 依序尝试 rkmpp -> drm 后端
  static const char* backend_names[] = {"rkmpp", "drm"};
  char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
  for (auto name : backend_names) {
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name(name);
    if (type == AV_HWDEVICE_TYPE_NONE) {
      LOGW(SOURCE) << "[" << stream_id_ << "]: HW backend '" << name
                   << "' not registered, skip";
      continue;
    }
    if (!CheckHwConfig(type)) {
      LOGW(SOURCE) << "[" << stream_id_ << "]: decoder " << codec_->name
                   << " does not declare support for backend '" << name << "', skip";
      continue;
    }
    // rkmpp 后端内部自己 open("/dev/mpp_service"), device 对它无效传 nullptr;
    // drm 后端才需要 DRM 设备节点路径.
    const bool is_rkmpp = (std::string(name) == "rkmpp");
    const char *device = is_rkmpp ? nullptr : pickDrmDevice();
    if (!is_rkmpp && !device) {
      LOGW(SOURCE) << "[" << stream_id_ << "]: no accessible DRM device node, skip backend "
                   << name;
      continue;
    }
    int err = av_hwdevice_ctx_create(&hw_device_ctx_, type, device, nullptr, 0);
    if (err < 0) {
      av_strerror(err, errbuf, sizeof(errbuf));
      LOGW(SOURCE) << "[" << stream_id_ << "]: create HW device (" << name
                   << ") failed: " << errbuf << " (" << err << "), try next backend";
      if (hw_device_ctx_) av_buffer_unref(&hw_device_ctx_);
      continue;
    }
    device_type_ = type;
    type_name_ = name;
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    codec_ctx_->pix_fmt       = AV_PIX_FMT_DRM_PRIME;
    LOGI(SOURCE) << "[" << stream_id_ << "]: HW device created: " << name
                 << " (device=" << (device ? device : "<auto>") << ")";
    return 0;
  }
  LOGE(SOURCE) << "[" << stream_id_ << "]: all RK HW backends (rkmpp/drm) failed."
               << " Checklist: 1) /dev/mpp_service exists (ls -l /dev/mpp_service)"
               << " 2) /dev/dri/card* /dev/dri/renderD* accessible"
               << " 3) ffmpeg -hwaccels shows rkmpp/drm"
               << " 4) process permission (video/render group)";
  return -1;
}

const char* PullHandlerImRK::pickDrmDevice() {
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

void PullHandlerImRK::precheckDeviceNodes() {
  const char* paths[] = {
    "/dev/mpp_service",
    "/dev/dri/card0",
    "/dev/dri/card1",
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
  };
  for (auto p : paths) {
    if (access(p, F_OK) == 0) {
      bool rw = (access(p, R_OK | W_OK) == 0);
      LOGI(SOURCE) << "[" << stream_id_ << "]: device node " << p
                   << " (perm: " << (rw ? "rw" : "ro/none") << ")";
    } else {
      LOGW(SOURCE) << "[" << stream_id_ << "]: device node " << p << " (not found)";
    }
  }
}

int PullHandlerImRK::codec_init() {
  int ret = 0;
  AVStream* video_stream = ifmt_ctx_->streams[video_index_];

  static std::unordered_map<enum AVCodecID, std::string> codeid_name_table = {
    {AV_CODEC_ID_H264, "h264_rkmpp"},
    {AV_CODEC_ID_HEVC, "hevc_rkmpp"},
    {AV_CODEC_ID_VP9,  "vp9_rkmpp"},
    {AV_CODEC_ID_AV1,  "av1_rkmpp"},
  };

  auto it = codeid_name_table.find(video_stream->codecpar->codec_id);
  if (it == codeid_name_table.end()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Codec " << avcodec_get_name(video_stream->codecpar->codec_id)
                 << " not supported by RKMPP, fallback to CPU decode";
    return -1;
  }
  codec_ = const_cast<AVCodec*>(avcodec_find_decoder_by_name(it->second.c_str()));
  if (!codec_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: RKMPP codec " << it->second << " not found";
    return -1;
  }
  LOGI(SOURCE) << "[" << stream_id_ << "]: Using decoder: " << codec_->name;

  codec_ctx_ = avcodec_alloc_context3(codec_);
  if (!codec_ctx_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avcodec_alloc_context3 failed";
    return -1;
  }

  if ((ret = avcodec_parameters_to_context(codec_ctx_, video_stream->codecpar)) < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avcodec_parameters_to_context failed: " << ret;
    return ret;
  }

  codec_ctx_->pkt_timebase = video_stream->time_base;

  codec_ctx_->get_format = get_hw_format;
  if ((ret = hw_decoder_init()) < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: hw_decoder_init failed";
    return ret;
  }

  AVDictionary* decOpts = nullptr;
  av_dict_set(&decOpts, "extra_hw_frames", "8", 0);
  ret = avcodec_open2(codec_ctx_, codec_, &decOpts);
  av_dict_free(&decOpts);
  if (ret < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avcodec_open2 failed: " << ret;
    return ret;
  }

  return 0;
}

bool PullHandlerImRK::SupportHWDevice() {
  return support_hwdevice();
}

void PullHandlerImRK::ConfigureOutputType() {
  if (output_type_ != OutputType::OUTPUT_CPU) {
    LOGW(SOURCE) << "VSTREAM_USE_RK ON: force output type to CPU (NV12)";
  }
  output_type_ = OutputType::OUTPUT_CPU;
}

int PullHandlerImRK::decode_write() {
  int ret = avcodec_send_packet(codec_ctx_, pkt_);
  if (ret == AVERROR(EAGAIN)) {
    // 清空解码器输出缓冲区
    AVFrame *drain_frame = nullptr;
    while (running_.load()) {
      drain_frame = av_frame_alloc();
      if (!drain_frame) return -1;
      ret = avcodec_receive_frame(codec_ctx_, drain_frame);
      if (ret == 0) {
        // 处理帧（注意：这里是清空过程中产生的帧，可能是之前输入的输出）
        auto data = ProcessFrameRKMPP(drain_frame);
        av_frame_free(&drain_frame);
        if (data && module_ && handler_) {
          handler_->SendData(data);
        } else if (!data) {
          LOGW(SOURCE) << "[" << stream_id_ << "]: ProcessFrameRKMPP failed during drain, skip frame";
        }
      } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_frame_free(&drain_frame);
        break;
      } else {
        av_frame_free(&drain_frame);
        return ret;
      }
    }  // while
    // 重新尝试发送当前包
    ret = avcodec_send_packet(codec_ctx_, pkt_);
  }
  if (ret < 0 && ret != AVERROR_EOF) {
    LOGE(SOURCE) << "send_packet error: " << ret;
    return ret;
  }

  while (running_.load()) {
    AVFrame *p_frame = av_frame_alloc();
    if (!p_frame) return -1;
    ret = avcodec_receive_frame(codec_ctx_, p_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_frame_free(&p_frame);
      return 0;
    } else if (ret < 0) {
      av_frame_free(&p_frame);
      LOGE(SOURCE) << "receive_frame error: " << ret;
      return ret;
    }
    auto data = ProcessFrameRKMPP(p_frame);
    av_frame_free(&p_frame);
    if (!data) {
      // 单帧处理失败
      ++frame_error_cnt_;
      if (frame_error_cnt_ >= kMaxFrameErrorCnt) {
        LOGE(SOURCE) << "[" << stream_id_ << "]: " << frame_error_cnt_
                     << " consecutive frame errors, stream incompatible with RKMPP path, stop";
        return -1;
      }
      LOGW(SOURCE) << "[" << stream_id_ << "]: ProcessFrameRKMPP failed, skip frame (cnt="
                   << frame_error_cnt_ << "/" << kMaxFrameErrorCnt << ")";
      continue;
    }
    frame_error_cnt_ = 0;
    if (!module_ || !handler_) {
      LOGE(SOURCE) << "module_ or handler_ is null";
      return -1;
    }
    handler_->SendData(data);
  }
  return 0;
}

/**
 * DRM_PRIME -> 系统内存 NV12，plane 直接引用 AVFrame 缓冲区(零额外拷贝)，
 * 生命周期由 AVFrameBufRef 管理，下游 libyuv 转换直接按 linesize 步长读取。
 * 任何单帧失败返回 nullptr，由调用方跳过该帧。
 */
std::shared_ptr<FrameInfo> PullHandlerImRK::ProcessFrameRKMPP(AVFrame *p_frame) {
  if (!p_frame) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: p_frame is null";
    return nullptr;
  }
  if (p_frame->format != AV_PIX_FMT_DRM_PRIME) {
    LOGW(SOURCE) << "[" << stream_id_ << "]: frame format " << p_frame->format
                 << " is not DRM_PRIME, skip";
    return nullptr;
  }

  AVFrame* sw = av_frame_alloc();
  if (!sw) {
    LOGW(SOURCE) << "[" << stream_id_ << "]: av_frame_alloc failed, skip";
    return nullptr;
  }
  int err = av_hwframe_transfer_data(sw, p_frame, 0);
  if (err < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(err, errbuf, sizeof(errbuf));
    LOGW(SOURCE) << "[" << stream_id_ << "]: av_hwframe_transfer_data failed: " << errbuf
                 << " (" << err << "), skip";
    av_frame_free(&sw);
    return nullptr;
  }
  if (sw->format != AV_PIX_FMT_NV12) {
    LOGW(SOURCE) << "[" << stream_id_ << "]: sw frame format " << sw->format
                 << " is not NV12, skip";
    av_frame_free(&sw);
    return nullptr;
  }

  DecodeFrame frame(sw->height, sw->width, DataFormat::PIXEL_FORMAT_YUV420_NV12);
  frame.device_type = DevType::CPU;
  frame.device_id = device_id_;
  frame.planeNum = 2;
  frame.pts = p_frame->pts;
  frame.plane[0] = sw->data[0];
  frame.plane[1] = sw->data[1];
  frame.stride[0] = sw->linesize[0];
  frame.stride[1] = sw->linesize[1];
  frame.buf_ref = std::make_unique<AVFrameBufRef>(sw);

  return OnDecodeFrame(&frame);
}

}  // namespace cnstream
