
#include "data_source.hpp"
#include "cuda/data_handler_pull_cuda.hpp"

#include "data_source_param.hpp"
#include "cnstream_source.hpp"

#include <memory>
#include <sstream>
#include <unordered_map>

namespace cnstream {

static enum AVPixelFormat hw_pix_fmt;

bool PullHandlerImCUDA::support_hwdevice() {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name(type_name_.c_str());
  if (type == AV_HWDEVICE_TYPE_NONE) {
    LOGE(SOURCE) << "Device type: " << type_name_ << " is not supported.";
    return false;
  }
  device_type_ = type;
  return true;
}

enum AVPixelFormat PullHandlerImCUDA::get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  const enum AVPixelFormat *p;
  for (p = pix_fmts; *p != -1; p++) {
    if (*p == hw_pix_fmt) {
      return *p;
    }
  }
  LOGE(SOURCE) << "Failed to find HW surface format.";
  return AV_PIX_FMT_NONE;
}

int PullHandlerImCUDA::init_hwdevice_conf() {
  for (int i = 0;; i++) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(codec_, i);
    if (!config) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: " << codec_->name << " does not support device type "
                   << av_hwdevice_get_type_name(device_type_);
      return -1;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == device_type_) {
      if (config->pix_fmt != AV_PIX_FMT_CUDA) {
        LOGE(SOURCE) << "[" << stream_id_ << "]: " << codec_->name << " AV_PIX_FMT_CUDA pix_fmt not supported";
        return -1;
      }
      hw_pix_fmt = config->pix_fmt;
      return 0;
    }
  }
  return -1;
}

int PullHandlerImCUDA::hw_decoder_init() {
  int err = 0;
  if (device_id_ < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Invalid device ID";
    return -1;
  }
  std::string device_str = std::to_string(device_id_);
  if ((err = av_hwdevice_ctx_create(&hw_device_ctx_, device_type_, device_str.c_str(), NULL, 0)) < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to create specified HW device: " << err;
    return err;
  }
  this->codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  return err;
}

int PullHandlerImCUDA::codec_init() {
  int ret = 0;
  AVStream* video_stream = ifmt_ctx_->streams[video_index_];

  static std::unordered_map<enum AVCodecID, std::string> codeid_name_table = {
    {AV_CODEC_ID_H264, "h264_cuvid"},
    {AV_CODEC_ID_HEVC, "hevc_cuvid"},
    {AV_CODEC_ID_VP8, "vp8_cuvid"},
    {AV_CODEC_ID_VP9, "vp9_cuvid"},
    {AV_CODEC_ID_AV1, "av1_cuvid"},
  };

  auto it = codeid_name_table.find(video_stream->codecpar->codec_id);
  if (it == codeid_name_table.end()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Codec name not found, fallback to CPU decode";
    return -1;
  }
  codec_ = const_cast<AVCodec*>(avcodec_find_decoder_by_name(it->second.c_str()));
  if (!codec_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Codec not found";
    return -1;
  }
  if ((ret = init_hwdevice_conf()) != 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: init_hwdevice_conf error";
    return ret;
  }

  codec_ctx_ = avcodec_alloc_context3(codec_);
  if (!codec_ctx_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avcodec_alloc_context error";
    return -1;
  }

  if ((ret = avcodec_parameters_to_context(codec_ctx_, video_stream->codecpar)) < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avcodec_parameters_to_context error: " << ret;
    return ret;
  }

  codec_ctx_->pkt_timebase = video_stream->time_base;

  codec_ctx_->get_format = get_hw_format;
  if ((ret = hw_decoder_init()) < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: hw_decoder_init error";
    return ret;
  }

  if ((ret = avcodec_open2(codec_ctx_, codec_, NULL)) < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to open codec: " << ret;
    return ret;
  }

  if (!src_stream_) {
    CHECK_CUDA_RUNTIME(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&src_stream_)));
  }

  return 0;
}

bool PullHandlerImCUDA::SupportHWDevice() {
  return support_hwdevice();
}

void PullHandlerImCUDA::ConfigureOutputType() {
  if (output_type_ == OutputType::OUTPUT_CPU) {
    LOGW(SOURCE) << "VSTREAM_USE_CUDA ON: force output type to CUDA";
    output_type_ = OutputType::OUTPUT_CUDA;
  }
}

int PullHandlerImCUDA::decode_write() {
  int ret = 0;
  AVFrame *p_frame = nullptr;
  AVFrame *sw_frame = nullptr;

  while ((ret = avcodec_send_packet(codec_ctx_, &pkt_)) == AVERROR(EAGAIN)) {
    AVFrame* drain_frame = av_frame_alloc();
    if (!drain_frame) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: av_frame_alloc alloc drain_frame failed";
      return -1;
    }
    ret = avcodec_receive_frame(codec_ctx_, drain_frame);
    if (ret == 0) {
      std::shared_ptr<FrameInfo> data = nullptr;
      if (output_type_ == OutputType::OUTPUT_CPU) {
        AVFrame* sw_drain = av_frame_alloc();
        if (sw_drain) {
          data = ProcessFrameCPU(drain_frame, sw_drain, ret);
          av_frame_free(&sw_drain);
        }
      } else {
        data = ProcessFrameCUDA(drain_frame, ret);
      }
      if (ret == 0 && data && module_ && handler_) {
        handler_->SendData(data);
      }
    }
    av_frame_free(&drain_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      ret = 0;
      break;
    }
    if (ret < 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: avcodec_receive_frame error during drain: " << ret;
      return ret;
    }
  }

  if (ret < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avcodec_send_packet error: " << ret;
    return ret;
  }

  while (running_.load()) {
    if (!(p_frame = av_frame_alloc())) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: av_frame_alloc error";
      ret = -1;
      break;
    }

    if (output_type_ == OutputType::OUTPUT_CPU) {
      if (!(sw_frame = av_frame_alloc())) {
        LOGE(SOURCE) << "[" << stream_id_ << "]: av_frame_alloc alloc sw_frame failed";
        av_frame_free(&p_frame);
        ret = -1;
        break;
      }
    }

    ret = avcodec_receive_frame(codec_ctx_, p_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_frame_free(&p_frame);
      av_frame_free(&sw_frame);
      return 0;
    } else if (ret < 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: Error during decoding: " << ret;
      break;
    }

    std::shared_ptr<FrameInfo> data = nullptr;

    if (output_type_ == OutputType::OUTPUT_CPU) {
      data = ProcessFrameCPU(p_frame, sw_frame, ret);
    } else if (output_type_ == OutputType::OUTPUT_CUDA) {
      data = ProcessFrameCUDA(p_frame, ret);
    } else {
      LOGF(SOURCE) << "[" << stream_id_ << "]: Unsupported output type: " << static_cast<int>(output_type_);
      ret = -1;
      break;
    }

    if (!data || ret != 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: ProcessFrame failed, ret = " << ret;
      ret = -1;
      break;
    }
    if (!module_ || !handler_) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: module_ or handler_ is null";
      ret = -1;
      break;
    }

    handler_->SendData(data);

    av_frame_free(&p_frame);
    av_frame_free(&sw_frame);
  }

  av_frame_free(&p_frame);
  av_frame_free(&sw_frame);
  return ret;
}

/**
 * 解码帧回传到 CPU 内存
 */
std::shared_ptr<FrameInfo> PullHandlerImCUDA::ProcessFrameCPU(AVFrame *p_frame, AVFrame *sw_frame, int &ret) {
  if (!p_frame) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: p_frame is null";
    return nullptr;
  }
  if (p_frame->format == hw_pix_fmt) {
    if ((ret = av_hwframe_transfer_data(sw_frame, p_frame, 0)) < 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: Error transferring the data: " << ret;
      return nullptr;
    }
  } else {
    LOGE(SOURCE) << "[" << stream_id_ << "]: p_frame format not supported: " << p_frame->format;
    return nullptr;
  }

  if (!sw_frame) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: sw_frame is null";
    return nullptr;
  }

  DataFormat nv_fmt = DataFormat::INVALID;
  if (sw_frame->format == AV_PIX_FMT_NV12) {
    nv_fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;
  } else if (sw_frame->format == AV_PIX_FMT_NV21) {
    nv_fmt = DataFormat::PIXEL_FORMAT_YUV420_NV21;
  } else {
    LOGE(SOURCE) << "[" << stream_id_ << "]: sw_frame format not supported: " << sw_frame->format;
    ret = -1;
    return nullptr;
  }

  DecodeFrame frame(sw_frame->height, sw_frame->width, nv_fmt);
  frame.device_type = DevType::CPU;
  frame.device_id = -1;
  frame.planeNum = 2;
  frame.pts = sw_frame->pts;

  int width = sw_frame->width;
  int height = sw_frame->height;
  int src_y_stride = sw_frame->linesize[0];
  int src_uv_stride = sw_frame->linesize[1];
  size_t y_size = static_cast<size_t>(width) * height;
  size_t uv_size = static_cast<size_t>(width) * height / 2;

  uint8_t* y_buffer = new (std::nothrow) uint8_t[y_size];
  uint8_t* uv_buffer = new (std::nothrow) uint8_t[uv_size];
  if (!y_buffer || !uv_buffer) {
    LOGE(SOURCE) << "Failed to allocate memory for frame data";
    delete[] y_buffer;
    delete[] uv_buffer;
    ret = -1;
    return nullptr;
  }

  for (int i = 0; i < height; ++i) {
    memcpy(y_buffer + i * width, sw_frame->data[0] + i * src_y_stride, width);
  }
  for (int i = 0; i < height / 2; ++i) {
    memcpy(uv_buffer + i * width, sw_frame->data[1] + i * src_uv_stride, width);
  }

  frame.plane[0] = y_buffer;
  frame.plane[1] = uv_buffer;
  frame.stride[0] = width;
  frame.stride[1] = width;
  frame.buf_ref = std::make_unique<MatBufRefNV12>(y_buffer, uv_buffer);

  return OnDecodeFrame(&frame);
}

std::shared_ptr<FrameInfo> PullHandlerImCUDA::ProcessFrameCUDA(AVFrame *p_frame, int &ret) {
  if (!p_frame) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: p_frame is null";
    return nullptr;
  }
  if (p_frame->format != AV_PIX_FMT_CUDA) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: p_frame format not supported: " << p_frame->format;
    ret = -1;
    return nullptr;
  }

  DecodeFrame frame(p_frame->height, p_frame->width, DataFormat::PIXEL_FORMAT_YUV420_NV12);
  frame.device_type = DevType::CUDA;
  frame.device_id = device_id_;
  frame.planeNum = 2;
  frame.pts = p_frame->pts;

  frame.plane[0] = p_frame->data[0];
  frame.plane[1] = p_frame->data[1];
  frame.stride[0] = p_frame->linesize[0];
  frame.stride[1] = p_frame->linesize[1];

  if (frame.stride[0] != frame.stride[1]) {
    LOGW(SOURCE) << "stride[0] != stride[1]: " << frame.stride[0] << " != " << frame.stride[1];
  }

  return OnDecodeFrame(&frame);
}

}  // namespace cnstream