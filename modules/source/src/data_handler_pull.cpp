#include "cnstream_source.hpp"
#include "data_handler_pull.hpp"
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

static DecoderType ResolveDecoderType(DataSource *module, const std::string &stream_id) {
#ifdef VSTREAM_USE_CUDA
  DecoderType fallback = DecoderType::DECODER_CUDA;
#elif defined(VSTREAM_USE_ROCKCHIP)
  DecoderType fallback = DecoderType::DECODER_ROCKCHIP;
#else
  DecoderType fallback = DecoderType::DECODER_CPU;
#endif

  if (!module) return fallback;

  ModuleParamSet module_params = module->GetSourceParam().param_set_;
  ModuleParamSet stream_params = module->GetStreamParams(stream_id);
  const ModuleParamSet* candidates = !stream_params.empty() ? &stream_params : &module_params;

  auto it = candidates->find(key_decoder_type);
  if (it == candidates->end()) return fallback;
  auto mit = param_decoder_map.find(it->second);
  if (mit == param_decoder_map.end()) {
    LOGW(SOURCE) << "[" << stream_id << "]: unknown decoder_type '" << it->second
                 << "', fallback to default";
    return fallback;
  }
  return mit->second;
}

std::shared_ptr<SourceHandler> PullHandler::Create(DataSource *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SOURCE) << "[" << stream_id << "]: module_ null";
    return nullptr;
  }
  DecoderType decoder_type = ResolveDecoderType(module, stream_id);
  return std::shared_ptr<PullHandler>(new PullHandler(module, stream_id, decoder_type));
}

PullHandler::PullHandler(DataSource *module, const std::string &stream_id, DecoderType decoder_type)
    : SourceHandler(module, stream_id) {
#ifdef VSTREAM_USE_ROCKCHIP
  if (decoder_type == DecoderType::DECODER_RKMPP) {
    impl_ = new PullHandlerImRK(module, this);
    return;
  }
#endif
#ifdef VSTREAM_USE_CUDA
  if (decoder_type == DecoderType::DECODER_CUDA) {
    impl_ = new PullHandlerImCUDA(module, this);
  } else {
    impl_ = new PullHandlerImCPU(module, this);
  }
#else
  (void)decoder_type;
  impl_ = new PullHandlerImCPU(module, this);
#endif
}

PullHandler::~PullHandler() {
  Close();
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

bool PullHandler::Open() {
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

void PullHandler::Close() {
  if (impl_) {
    impl_->Close();
  }
}

void PullHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

void PullHandler::RegisterHandlerParams() {
}

bool PullHandler::CheckHandlerParams(const ModuleParamSet& params) {
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
    LOGE(SOURCE) << "[" << stream_id_ << "]: stream_url is required";
    return false;
  }
  return true;
}

bool PullHandler::SetHandlerParams(const ModuleParamSet& params) {
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
    auto *self = static_cast<PullHandlerIm*>(ctx);
    return !self->IsRunning() ? 1 : 0;
}

int PullHandlerIm::input_format_init() {
  int ret = 0;
  ret = avformat_network_init();
  if (ret != 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avformat_network_init failed: " << ret;
    return ret;
  }

  ifmt_ctx_ = avformat_alloc_context();
  if (!ifmt_ctx_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avformat_alloc_context error";
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
    LOGE(SOURCE) << "[" << stream_id_ << "]: avformat_open_input error: " << ret;
    return ret;
  }
  // ifmt_ctx_->max_analyze_duration = 20 * AV_TIME_BASE;
  ret = avformat_find_stream_info(ifmt_ctx_, nullptr);
  if (ret < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avformat_find_stream_info error: " << ret;
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
    LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to find video stream";
    return -1;
  }

  return 0;
}

void PullHandlerIm::clean_up() {
  av_packet_unref(&pkt_);
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

bool PullHandlerIm::Open() {
  if (!module_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: module_ is null";
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
  LOGI(SOURCE) << "[" << stream_id_ << "]: device_id=" << device_id_
               << ", output=" << static_cast<int>(output_type_);

  stream_url_ = param_set_.at(key_input_url);
  if (stream_url_.empty()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: url is empty";
    return false;
  }
  if (param_set_.find(key_frame_rate) != param_set_.end()) {
    frame_rate_ = std::stoi(param_set_.at(key_frame_rate));
  }

  ConfigureOutputType();

  running_.store(true);
  thread_ = std::thread(&PullHandlerIm::Loop, this);
  return true;
}

void PullHandlerIm::Stop() {
  if (running_.load()) {
    running_.store(false);
  }
}

void PullHandlerIm::Close() {
  Stop();
  if (thread_.joinable()) {
    thread_.join();
  }
  clean_up();
}

void PullHandlerIm::Loop() {
  if (!SupportHWDevice()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: hardware device not supported";
    OnEndFrame();
    running_.store(false);
    return;
  }

  if (input_format_init() < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: input_format_init failed";
    OnEndFrame();
    running_.store(false);
    return;
  }

  if (codec_init() < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: codec_init failed";
    OnEndFrame();
    running_.store(false);
    return;
  }

  FrController controller(frame_rate_);
  if (frame_rate_ > 0) controller.Start();

  while (running_.load()) {
    int ret = av_read_frame(ifmt_ctx_, &pkt_);
    if (ret < 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: av_read_frame failed: " << ret;
      break;
    }
    if (pkt_.stream_index != video_index_) {
      av_packet_unref(&pkt_);
      continue;
    }
    ret = decode_write();
    if (ret < 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: decode_write failed: " << ret;
      break;
    }
    av_packet_unref(&pkt_);
    if (frame_rate_ > 0) {
      controller.Control();
    }
  }
  OnEndFrame();
}

std::shared_ptr<FrameInfo> PullHandlerIm::OnDecodeFrame(DecodeFrame* frame) {
  if (!frame) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: OnDecodeFrame: frame is null";
    return nullptr;
  }
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo();
  if (!data) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: OnDecodeFrame: failed to create FrameInfo.";
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
    LOGE(SOURCE) << "[" << stream_id_ << "]: OnDecodeFrame: failed to setup data frame.";
    return nullptr;
  }
  return data;
}

void PullHandlerIm::OnEndFrame() {
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo(true);
  if (!data) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: OnEndFrame: failed to create FrameInfo.";
    return;
  }
  SendFrameInfo(data);
  LOGI(SOURCE) << "[" << stream_id_ << "]: OnEndFrame: send end frame.";
}

int PullHandlerImCPU::codec_init() {
  int ret = 0;
  AVStream* video_stream = ifmt_ctx_->streams[video_index_];

  codec_ = const_cast<AVCodec*>(avcodec_find_decoder(video_stream->codecpar->codec_id));
  if (!codec_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Codec not found";
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
    LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to open codec: " << ret;
    return ret;
  }
  return 0;
}


int PullHandlerImCPU::decode_write() {
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
        auto data = ProcessFrame(drain_frame, ret);
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
    auto data = ProcessFrame(p_frame, ret);
    av_frame_free(&p_frame);
    if (!data || ret != 0) {
      LOGE(SOURCE) << "ProcessFrame failed";
      return -1;
    }
    if (!module_ || !handler_) {
      LOGE(SOURCE) << "module_ or handler_ is null";
      return -1;
    }
    handler_->SendData(data);
  }
  return 0;
}


std::shared_ptr<FrameInfo> PullHandlerImCPU::ProcessFrame(AVFrame *p_frame, int &ret) {
  if (!p_frame) {
    LOGE(SOURCE) << "ProcessFrame: p_frame is null";
    return nullptr;
  }

  DataFormat nv_fmt = DataFormat::INVALID;
  if (p_frame->format == AV_PIX_FMT_NV12) {
    nv_fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;
  } else if (p_frame->format == AV_PIX_FMT_NV21) {
    nv_fmt = DataFormat::PIXEL_FORMAT_YUV420_NV21;
  } else {
    LOGE(SOURCE) << "ProcessFrame: p_frame format not supported: " << p_frame->format;
    ret = -1;
    return nullptr;
  }

  DecodeFrame frame(p_frame->height, p_frame->width, nv_fmt);
  frame.device_type = DevType::CPU;
  frame.planeNum = 2;
  frame.pts = p_frame->pts;

  int width = p_frame->width;
  int height = p_frame->height;
  int src_y_stride = p_frame->linesize[0];
  int src_uv_stride = p_frame->linesize[1];
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
    memcpy(y_buffer + i * width, p_frame->data[0] + i * src_y_stride, width);
  }
  for (int i = 0; i < height / 2; ++i) {
    memcpy(uv_buffer + i * width, p_frame->data[1] + i * src_uv_stride, width);
  }

  frame.plane[0] = y_buffer;
  frame.plane[1] = uv_buffer;
  frame.stride[0] = width;
  frame.stride[1] = width;
  frame.buf_ref = std::make_unique<MatBufRefNV12>(y_buffer, uv_buffer);

  return OnDecodeFrame(&frame);
}

}  // namespace cnstream