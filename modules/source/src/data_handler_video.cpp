#include "cnstream_source.hpp"
#include "data_handler_video.hpp"
#include "data_source.hpp"
#include "data_source_param.hpp"

#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

namespace cnstream {

std::shared_ptr<SourceHandler> VideoHandler::Create(DataSource *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SOURCE) << "[" << stream_id << "]: module_ null";
    return nullptr;
  }
  return std::shared_ptr<VideoHandler>(new VideoHandler(module, stream_id));
}

VideoHandler::VideoHandler(DataSource *module, const std::string &stream_id)
    : SourceHandler(module, stream_id) {
#ifdef VSTREAM_USE_CUDA
  impl_ = new VideoHandlerImplCUDA(module, this);
#else
  impl_ = new VideoHandlerImplCPU(module, this);
#endif
}

VideoHandler::~VideoHandler() {
  Close();
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

bool VideoHandler::Open() {
  if (!module_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: module_ null";
    return false;
  }
  if (!impl_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Video handler open failed, no memory left";
    return false;
  }
  if (stream_index_ == INVALID_STREAM_IDX) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Invalid stream_idx";
    return false;
  }
  return impl_->Open();
}

void VideoHandler::Close() {
  if (impl_) {
    impl_->Close();
  }
}

void VideoHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

void VideoHandler::RegisterHandlerParams() {
}

bool VideoHandler::CheckHandlerParams(const ModuleParamSet& params) {
  DataSource* ds = dynamic_cast<DataSource*>(module_);
  ModuleParamSet stream_params;
  const ModuleParamSet* check_params = &params;
  if (ds) {
    stream_params = ds->GetStreamParams(stream_id_);
    if (!stream_params.empty()) {
      check_params = &stream_params;
    }
  }
  if (check_params->find(key_input_url) == check_params->end()) {
    LOGE(SOURCE) << "[VideoHandler] stream_url is required";
    return false;
  }
  return true;
}

bool VideoHandler::SetHandlerParams(const ModuleParamSet& params) {
  if (!impl_) {
    return false;
  }
  DataSource* ds = dynamic_cast<DataSource*>(module_);
  if (ds) {
    ModuleParamSet stream_params = ds->GetStreamParams(stream_id_);
    if (!stream_params.empty()) {
      impl_->param_set_ = stream_params;
      return true;
    }
  }
  impl_->param_set_ = params;
  return true;
}

static int interrupt_cb(void *ctx) {
    auto *self = static_cast<VideoHandlerImpl*>(ctx);
    return !self->IsRunning() ? 1 : 0;
}

int VideoHandlerImpl::input_format_init() {
  int ret = 0;
  ret = avformat_network_init();
  if (ret != 0) {
    LOGE(SOURCE) << "avformat_network_init failed: " << ret;
    return ret;
  }

  ifmt_ctx_ = avformat_alloc_context();
  if (!ifmt_ctx_) {
    LOGE(SOURCE) << "avformat_alloc_context error";
    return -1;
  }
  ifmt_ctx_->interrupt_callback.callback = interrupt_cb;
  ifmt_ctx_->interrupt_callback.opaque = this;

  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "buffer_size", "1024000", 0);
  av_dict_set(&opts, "max_delay", "400000", 0);
  av_dict_set(&opts, "stimeout", "20000000", 0);
  av_dict_set(&opts, "rtsp_transport", "tcp", 0);

  ret = avformat_open_input(&ifmt_ctx_, stream_url_.c_str(), NULL, &opts);
  av_dict_free(&opts);
  if (ret != 0) {
    LOGE(SOURCE) << "avformat_open_input error: " << ret;
    return ret;
  }
  // ifmt_ctx_->max_analyze_duration = 20 * AV_TIME_BASE;
  ret = avformat_find_stream_info(ifmt_ctx_, nullptr);
  if (ret < 0) {
    LOGE(SOURCE) << "avformat_find_stream_info error: " << ret;
    return ret;
  }

  for (unsigned int i = 0; i < ifmt_ctx_->nb_streams; ++i) {
    AVCodecParameters* codec_par = ifmt_ctx_->streams[i]->codecpar;
    if (codec_par->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_index_ = i;
      break;
    }
  }

  if (video_index_ < 0) {
    LOGE(SOURCE) << "Failed to find video stream";
    return -1;
  }

  return 0;
}

void VideoHandlerImpl::clean_up() {
  av_frame_free(&s_frame_);
  if (sws_ctx_) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
    codec_ctx_ = nullptr;
  }
  if (ifmt_ctx_) {
    avformat_close_input(&ifmt_ctx_);
    ifmt_ctx_ = nullptr;
  }
  if (hw_device_ctx_) {
    av_buffer_unref(&hw_device_ctx_);
    hw_device_ctx_ = nullptr;
  }
  avformat_network_deinit();
}

bool VideoHandlerImpl::Open() {
  if (!module_) {
    LOGE(SOURCE) << "Video: module_ is null";
    return false;
  }
  if (param_set_.find(key_device_id) != param_set_.end()) {
    device_id_ = std::stoi(param_set_.at(key_device_id));
  }
  if (param_set_.find(key_output_type) != param_set_.end()) {
    std::string out_type = param_set_.at(key_output_type);
    auto it = param_output_map.find(out_type);
    if (it != param_output_map.end()) {
      output_type_ = it->second;
    }
  }
  if (param_set_.find(key_interval) != param_set_.end()) {
    interval_ = std::stoi(param_set_.at(key_interval));
  }
  LOGI(SOURCE) << "Video: device_id=" << device_id_
               << ", output=" << static_cast<int>(output_type_);

  stream_url_ = param_set_.at(key_input_url);
  if (stream_url_.empty()) {
    LOGE(SOURCE) << "Video: url is empty";
    return false;
  }
  if (param_set_.find(key_frame_rate) != param_set_.end()) {
    frame_rate_ = std::stoi(param_set_.at(key_frame_rate));
  }

  ConfigureOutputType();

  running_.store(true);
  thread_ = std::thread(&VideoHandlerImpl::Loop, this);
  return true;
}

void VideoHandlerImpl::Stop() {
  if (running_.load()) {
    running_.store(false);
  }
}

void VideoHandlerImpl::Close() {
  Stop();
  if (thread_.joinable()) {
    thread_.join();
  }
  clean_up();
}

void VideoHandlerImpl::Loop() {
  if (!SupportHWDevice()) {
    LOGE(SOURCE) << "Video: hardware device not supported";
    OnEndFrame();
    running_.store(false);
    return;
  }

  if (input_format_init() < 0) {
    LOGE(SOURCE) << "input_format_init failed";
    OnEndFrame();
    running_.store(false);
    return;
  }

  if (codec_init() < 0) {
    LOGE(SOURCE) << "codec_init failed";
    OnEndFrame();
    running_.store(false);
    return;
  }

  FrController controller(frame_rate_);
  if (frame_rate_ > 0) controller.Start();

  while (running_.load()) {
    int ret = av_read_frame(ifmt_ctx_, &pkt_);
    if (ret < 0) {
      LOGE(SOURCE) << "av_read_frame error";
      break;
    }
    if (pkt_.stream_index != video_index_) {
      av_packet_unref(&pkt_);
      continue;
    }
    ret = decode_write();
    if (ret < 0) {
      LOGE(SOURCE) << "decode_write error";
      break;
    }
    av_packet_unref(&pkt_);
    if (frame_rate_ > 0) {
      controller.Control();
    }
  }
  OnEndFrame();
}

std::shared_ptr<FrameInfo> VideoHandlerImpl::OnDecodeFrame(DecodeFrame* frame) {
  if (!frame) {
    LOGE(SOURCE) << "OnDecodeFrame: frame is null";
    return nullptr;
  }
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo();
  if (!data) {
    LOGE(SOURCE) << "OnDecodeFrame: failed to create FrameInfo.";
    return nullptr;
  }
  data->timestamp = frame->pts;
  if (!frame->valid) {
    data->flags = static_cast<size_t>(DataFrameFlag::FRAME_FLAG_INVALID);
    SendFrameInfo(data);
    return nullptr;
  }
  int ret = SourceRender::Process(data, frame, frame_id_++, src_stream_);
  if (ret < 0) {
    LOGE(SOURCE) << "OnDecodeFrame: failed to setup data frame.";
    return nullptr;
  }
  return data;
}

void VideoHandlerImpl::OnEndFrame() {
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo(true);
  if (!data) {
    LOGE(SOURCE) << "OnEndFrame: failed to create FrameInfo.";
    return;
  }
  SendFrameInfo(data);
  LOGI(SOURCE) << "OnEndFrame: send end frame.";
}

int VideoHandlerImplCPU::codec_init() {
  int ret = 0;
  AVStream* video_stream = ifmt_ctx_->streams[video_index_];

  codec_ = const_cast<AVCodec*>(avcodec_find_decoder(video_stream->codecpar->codec_id));
  if (!codec_) {
    LOGE(SOURCE) << "Codec not found";
    return -1;
  }

  codec_ctx_ = avcodec_alloc_context3(codec_);
  if (!codec_ctx_) {
    LOGE(SOURCE) << "avcodec_alloc_context error";
    return -1;
  }

  if ((ret = avcodec_parameters_to_context(codec_ctx_, video_stream->codecpar)) < 0) {
    LOGE(SOURCE) << "avcodec_parameters_to_context error: " << ret;
    return ret;
  }

  codec_ctx_->pkt_timebase = video_stream->time_base;

  if ((ret = avcodec_open2(codec_ctx_, codec_, NULL)) < 0) {
    LOGE(SOURCE) << "Failed to open codec: " << ret;
    return ret;
  }
  return 0;
}

int VideoHandlerImplCPU::decode_write() {
  int ret = 0;
  AVFrame *p_frame = nullptr;

  while ((ret = avcodec_send_packet(codec_ctx_, &pkt_)) == AVERROR(EAGAIN)) {
    AVFrame* drain_frame = av_frame_alloc();
    if (!drain_frame) {
      LOGE(SOURCE) << "av_frame_alloc alloc drain_frame failed";
      return -1;
    }
    ret = avcodec_receive_frame(codec_ctx_, drain_frame);
    if (ret == 0) {
      std::shared_ptr<FrameInfo> data = nullptr;
      data = ProcessFrame(drain_frame, ret);
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
      LOGE(SOURCE) << "avcodec_receive_frame error during drain: " << ret;
      return ret;
    }
  }

  if (ret < 0) {
    LOGE(SOURCE) << "avcodec_send_packet error: " << ret;
    return ret;
  }

  while (running_.load()) {
    if (!(p_frame = av_frame_alloc())) {
      LOGE(SOURCE) << "av_frame_alloc error";
      ret = -1;
      break;
    }

    ret = avcodec_receive_frame(codec_ctx_, p_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_frame_free(&p_frame);
      return 0;
    } else if (ret < 0) {
      LOGE(SOURCE) << "Error during decoding: " << ret;
      break;
    }

    std::shared_ptr<FrameInfo> data = nullptr;
    data = ProcessFrame(p_frame, ret);

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
  }

  av_frame_free(&p_frame);
  return ret;
}

std::shared_ptr<FrameInfo> VideoHandlerImplCPU::ProcessFrame(AVFrame *p_frame, int &ret) {
  s_frame_ = p_frame;

  if (!s_frame_) {
    LOGE(SOURCE) << "Video: s_frame_ is null";
    return nullptr;
  }

  DataFormat nv_fmt = DataFormat::INVALID;
  if (s_frame_->format == AV_PIX_FMT_NV12) {
    nv_fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;
  } else if (s_frame_->format == AV_PIX_FMT_NV21) {
    nv_fmt = DataFormat::PIXEL_FORMAT_YUV420_NV21;
  } else {
    LOGE(SOURCE) << "Video: s_frame_ format not supported: " << s_frame_->format;
    ret = -1;
    return nullptr;
  }

  DecodeFrame frame(s_frame_->height, s_frame_->width, nv_fmt);
  frame.device_type = DevType::CPU;
  frame.planeNum = 2;
  frame.pts = s_frame_->pts;

  int width = s_frame_->width;
  int height = s_frame_->height;
  int src_y_stride = s_frame_->linesize[0];
  int src_uv_stride = s_frame_->linesize[1];
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
    memcpy(y_buffer + i * width, s_frame_->data[0] + i * src_y_stride, width);
  }
  for (int i = 0; i < height / 2; ++i) {
    memcpy(uv_buffer + i * width, s_frame_->data[1] + i * src_uv_stride, width);
  }

  frame.plane[0] = y_buffer;
  frame.plane[1] = uv_buffer;
  frame.stride[0] = width;
  frame.stride[1] = width;
  frame.buf_ref = std::make_unique<MatBufRefNV12>(y_buffer, uv_buffer);

  return OnDecodeFrame(&frame);
}

}  // namespace cnstream