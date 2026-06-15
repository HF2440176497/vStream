#include "data_source.hpp"
#include "rk/data_handler_pull_rk.hpp"

#include "data_source_param.hpp"
#include "cnstream_source.hpp"

#include <memory>
#include <sstream>
#include <unordered_map>
#include <cstring>

namespace cnstream {

static enum AVPixelFormat hw_pix_fmt_rk;

bool PullHandlerImRK::support_hwdevice() {
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
               << " Check ffmpeg -hwaccels for available backends.";
  return false;
}

enum AVPixelFormat PullHandlerImRK::get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
    if (*p == hw_pix_fmt_rk) {
      return *p;
    }
  }
  LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to find HW surface format.";
  return AV_PIX_FMT_NONE;
}

int PullHandlerImRK::init_hwdevice_conf() {
  for (int i = 0;; i++) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(codec_, i);
    if (!config) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: " << codec_->name
                   << " does not support device type "
                   << av_hwdevice_get_type_name(device_type_);
      return -1;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
        config->device_type == device_type_) {
      if (config->pix_fmt != AV_PIX_FMT_DRM_PRIME) {
        LOGE(SOURCE) << "[" << stream_id_ << "]: " << codec_->name
                     << " DRM_PRIME pix_fmt not supported";
        return -1;
      }
      hw_pix_fmt_rk = config->pix_fmt;
      return 0;
    }
  }
  return -1;
}

int PullHandlerImRK::hw_decoder_init() {
  int err = 0;
  // rkmpp 后端内部自己 open("/dev/mpp_service"), device 对它无效传 nullptr;
  // drm 后端才需要 DRM 设备节点路径.
  const char *device = (type_name_ == "rkmpp") ? nullptr : nullptr;
  if ((err = av_hwdevice_ctx_create(&hw_device_ctx_, device_type_,
                                     device, nullptr, 0)) < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to create HW device ("
                 << type_name_ << "): " << err;
    return err;
  }
  LOGI(SOURCE) << "[" << stream_id_ << "]: HW device created: " << type_name_;
  codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  codec_ctx_->pix_fmt       = AV_PIX_FMT_DRM_PRIME;
  return err;
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

  if ((ret = init_hwdevice_conf()) != 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: init_hwdevice_conf failed";
    return ret;
  }

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

  // 分配 hw->sw 中转帧
  sw_frame_ = av_frame_alloc();
  if (!sw_frame_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: av_frame_alloc(sw_frame_) failed";
    return -1;
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

void PullHandlerImRK::clean_up() {
  if (sw_frame_) {
    av_frame_free(&sw_frame_);
    sw_frame_ = nullptr;
  }
  PullHandlerIm::clean_up();
}


int PullHandlerImRK::decode_write() {
  int ret = avcodec_send_packet(codec_ctx_, &pkt_);
  if (ret == AVERROR(EAGAIN)) {
    // 清空解码器输出缓冲区
    AVFrame *drain_frame = nullptr;
    while (running_.load()) {
      drain_frame = av_frame_alloc();
      if (!drain_frame) return -1;
      ret = avcodec_receive_frame(codec_ctx_, drain_frame);
      if (ret == 0) {
        // 处理帧（注意：这里是清空过程中产生的帧，可能是之前输入的输出）
        auto data = ProcessFrameRKMPP(drain_frame, ret);
        if (data && module_ && handler_) {
            handler_->SendData(data);
        }
        av_frame_free(&drain_frame);
      } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_frame_free(&drain_frame);
        break;
      } else {
        av_frame_free(&drain_frame);
        return ret;
      }
    }  // while
    // 重新尝试发送当前包
    ret = avcodec_send_packet(codec_ctx_, &pkt_);
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
    auto data = ProcessFrameRKMPP(p_frame, ret);
    av_frame_free(&p_frame);
    if (!data || ret != 0) {
      LOGE(SOURCE) << "ProcessFrameRKMPP failed";
      return -1;
    }
    if (!module_ || !handler_) {
      LOGE(SOURCE) << "module_ or handler_ is null";
      return -1;
    }
    handler_->SendData(data);
  }
  av_frame_free(&p_frame);
  return 0;
}

/**
 * DRM_PRIME → NV12 (系统内存) → DecodeFrame
 */
std::shared_ptr<FrameInfo> PullHandlerImRK::ProcessFrameRKMPP(AVFrame *p_frame, int &ret) {
  if (!p_frame) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: p_frame is null";
    return nullptr;
  }
  if (p_frame->format != AV_PIX_FMT_DRM_PRIME) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: p_frame format not DRM_PRIME: " << p_frame->format;
    ret = -1;
    return nullptr;
  }

  // 1) DRM_PRIME → 系统内存 (NV12)
  // 每次首先减少引用计数
  av_frame_unref(sw_frame_);
  int err = av_hwframe_transfer_data(sw_frame_, p_frame, 0);
  if (err < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: av_hwframe_transfer_data failed: " << err;
    ret = -1;
    return nullptr;
  }

  if (sw_frame_->format != AV_PIX_FMT_NV12) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: sw_frame format=" << sw_frame_->format << " NV12 expected";
    return nullptr;
  }

  int width  = p_frame->width;
  int height = p_frame->height;
  int src_y_stride  = sw_frame_->linesize[0];
  int src_uv_stride = sw_frame_->linesize[1];

  // 2) 拷贝 NV12 数据到自有缓冲区
  size_t y_size  = static_cast<size_t>(width) * height;
  size_t uv_size = static_cast<size_t>(width) * height / 2;

  uint8_t* y_buffer  = new (std::nothrow) uint8_t[y_size];
  uint8_t* uv_buffer = new (std::nothrow) uint8_t[uv_size];
  if (!y_buffer || !uv_buffer) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to allocate NV12 buffer";
    delete[] y_buffer;
    delete[] uv_buffer;
    ret = -1;
    return nullptr;
  }

  for (int i = 0; i < height; ++i) {
    std::memcpy(y_buffer + i * width, sw_frame_->data[0] + i * src_y_stride, width);
  }
  for (int i = 0; i < height / 2; ++i) {
    std::memcpy(uv_buffer + i * width, sw_frame_->data[1] + i * src_uv_stride, width);
  }

  // 3) 构造 DecodeFrame
  DecodeFrame frame(height, width, DataFormat::PIXEL_FORMAT_YUV420_NV12);
  frame.device_type = DevType::CPU;
  frame.device_id = device_id_;
  frame.planeNum = 2;
  frame.pts = p_frame->pts;
  frame.plane[0] = y_buffer;
  frame.plane[1] = uv_buffer;
  frame.stride[0] = width;
  frame.stride[1] = width;
  frame.buf_ref = std::make_unique<PullHandlerIm::MatBufRefNV12>(y_buffer, uv_buffer);

  return OnDecodeFrame(&frame);
}

}  // namespace cnstream