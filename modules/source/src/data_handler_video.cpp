#include "cnstream_source.hpp"
#include "data_handler_video.hpp"

#include <memory>
#include <unordered_map>

#include "data_source_param.hpp"

namespace cnstream {

static enum AVPixelFormat hw_pix_fmt;

std::shared_ptr<SourceHandler> VideoHandler::Create(DataSource *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SOURCE) << "[" << stream_id << "]: module_ null";
    return nullptr;
  }
  return std::shared_ptr<VideoHandler>(new VideoHandler(module, stream_id));
}

VideoHandler::VideoHandler(DataSource *module, const std::string &stream_id)
    : SourceHandler(module, stream_id) {
  impl_ = new VideoHandlerImpl(module, this);
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
  param_register_.Register(KEY_STREAM_URL, "URL of the video stream (rtsp/rtmp/file).");
  param_register_.Register(KEY_FRAME_RATE, "Framerate for video playback. Default is 25.");
}

bool VideoHandler::CheckHandlerParams(const ModuleParamSet& params) {
  if (params.find(KEY_STREAM_URL) == params.end()) {
    LOGE(SOURCE) << "[VideoHandler] stream_url is required";
    return false;
  }
  return true;
}

bool VideoHandler::SetHandlerParams(const ModuleParamSet& params) {
  if (impl_) {
    impl_->param_set_ = params;
  }
  return true;
}

bool VideoHandlerImpl::support_hwdevice() {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name(this->type_name_.c_str());
  if (type == AV_HWDEVICE_TYPE_NONE) {
    LOGE(SOURCE) << "Device type: " << type_name_ << " is not supported.";
    return false;
  }
  this->device_type_ = type;
  return true;
}

enum AVPixelFormat VideoHandlerImpl::get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  const enum AVPixelFormat *p;
  for (p = pix_fmts; *p != -1; p++) {
    if (*p == hw_pix_fmt) {
      return *p;
    }
  }
  LOGE(SOURCE) << "Failed to get HW surface format.";
  return AV_PIX_FMT_NONE;
}

/**
 * 作为 codec_init 的其中一步，保存支持的 HW 像素格式
 * 之后调用 hw_decoder_init 将信息附加到 codec_ctx_ 中
 */
int VideoHandlerImpl::init_hwdevice_conf() {
  for (int i = 0;; i++) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(this->codec_, i);
    if (!config) {
      LOGE(SOURCE) << "Decoder " << codec_->name << " does not support device type "
                   << av_hwdevice_get_type_name(device_type_);
      return -1;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == device_type_) {
      if (config->pix_fmt != AV_PIX_FMT_CUDA) {
        LOGE(SOURCE) << "Decoder " << codec_->name << " AV_PIX_FMT_CUDA pix_fmt not supported";
        return -1;
      }
      hw_pix_fmt = config->pix_fmt;
      return 0;
    }
  }
  return -1;
}

int VideoHandlerImpl::input_format_init() {
  int ret = 0;
  ret = avformat_network_init();
  if (ret != 0) {
    LOGE(SOURCE) << "avformat_network_init failed: " << ret;
    return ret;
  }

  this->ifmt_ctx_ = avformat_alloc_context();
  if (!this->ifmt_ctx_) {
    LOGE(SOURCE) << "avformat_alloc_context error";
    return -1;
  }

  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "buffer_size", "1024000", 0);
  av_dict_set(&opts, "max_delay", "200000", 0);
  av_dict_set(&opts, "stimeout", "20000000", 0);
  av_dict_set(&opts, "rtsp_transport", "tcp", 0);

  ret = avformat_open_input(&ifmt_ctx_, stream_url_.c_str(), NULL, &opts);
  av_dict_free(&opts);
  if (ret != 0) {
    LOGE(SOURCE) << "avformat_open_input error: " << ret;
    return ret;
  }

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

int VideoHandlerImpl::codec_init() {
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
    LOGE(SOURCE) << "Codec name not found, fallback to CPU decoder";
    this->codec_ = const_cast<AVCodec*>(avcodec_find_decoder(video_stream->codecpar->codec_id));
  } else {
    this->codec_ = const_cast<AVCodec*>(avcodec_find_decoder_by_name(it->second.c_str()));
  }

  if (!this->codec_) {
    LOGE(SOURCE) << "Codec not found";
    return -1;
  }

  if ((ret = init_hwdevice_conf()) != 0) {
    LOGE(SOURCE) << "init_hwdevice_conf error";
    return ret;
  }

  this->codec_ctx_ = avcodec_alloc_context3(this->codec_);
  if (!this->codec_ctx_) {
    LOGE(SOURCE) << "avcodec_alloc_context error";
    return -1;
  }

  if ((ret = avcodec_parameters_to_context(this->codec_ctx_, video_stream->codecpar)) < 0) {
    LOGE(SOURCE) << "avcodec_parameters_to_context error: " << ret;
    return ret;
  }

  this->codec_ctx_->pkt_timebase = video_stream->time_base;
  this->codec_ctx_->get_format = get_hw_format;

  if ((ret = hw_decoder_init()) < 0) {
    LOGE(SOURCE) << "hw_decoder_init error";
    return ret;
  }

  if ((ret = avcodec_open2(this->codec_ctx_, this->codec_, NULL)) < 0) {
    LOGE(SOURCE) << "Failed to open codec: " << ret;
    return ret;
  }

  return 0;
}

int VideoHandlerImpl::hw_decoder_init() {
  int err = 0;
  if (device_id_ < 0) {
    LOGE(SOURCE) << "Invalid device ID";
    return -1;
  }
  std::string device_str = std::to_string(device_id_);
  if ((err = av_hwdevice_ctx_create(&hw_device_ctx_, device_type_, device_str.c_str(), NULL, 0)) < 0) {
    LOGE(SOURCE) << "Failed to create specified HW device: " << err;
    return err;
  }
  this->codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  return err;
}

/**
 * 定义将 CUDA 格式的解码帧转换为 CPU 上的 BGR24 格式
 * cv_buf_ 是为 cv_frame_ 分配的内存，用于存储转换后的帧数据
 */
int VideoHandlerImpl::convert_frame_init() {
  if (!cv_frame_) {
    cv_frame_ = av_frame_alloc();
    if (!cv_frame_) {
      LOGE(SOURCE) << "cv_frame av_frame_alloc error";
      return -1;
    }
  }

  auto dst_width = codec_ctx_->width;
  auto dst_height = codec_ctx_->height;
  auto dst_pix_fmt = AV_PIX_FMT_BGR24;
  auto dst_img_size = av_image_get_buffer_size(dst_pix_fmt, dst_width, dst_height, 1);

  cv_buf_ = (uint8_t*)av_malloc(dst_img_size * sizeof(uint8_t));
  av_image_fill_arrays(cv_frame_->data, cv_frame_->linesize, cv_buf_, 
                      dst_pix_fmt, dst_width, dst_height, 1);
  return 0;
}

int VideoHandlerImpl::decode_write() {
  int ret = 0;
  AVFrame *p_frame = nullptr;
  AVFrame *sw_frame = nullptr;

  ret = avcodec_send_packet(codec_ctx_, &pkt_);
  if (ret < 0) {
    LOGE(SOURCE) << "avcodec_send_packet error: " << ret;
    return ret;
  }

  while (running_.load()) {
    if (!(p_frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc())) {
      LOGE(SOURCE) << "av_frame_alloc error";
      ret = -1;
      break;
    }

    ret = avcodec_receive_frame(codec_ctx_, p_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_frame_free(&p_frame);
      av_frame_free(&sw_frame);
      return 0;
    } else if (ret < 0) {
      LOGE(SOURCE) << "Error during decoding: " << ret;
      break;
    }

    std::shared_ptr<FrameInfo> data = nullptr;

    if (output_type_ == OutputType::OUTPUT_CPU) {
      if (p_frame->format == hw_pix_fmt) {
        if ((ret = av_hwframe_transfer_data(sw_frame, p_frame, 0)) < 0) {
          LOGE(SOURCE) << "Error transferring the data to system memory: " << ret;
          av_frame_free(&p_frame);
          av_frame_free(&sw_frame);
          break;
        }
        s_frame_ = sw_frame;
      } else {
        LOGE(SOURCE) << "VideoHandlerImpl: p_frame format not supported: " << p_frame->format;
        break;
      }
      if (!s_frame_) {
        LOGE(SOURCE) << "VideoHandlerImpl: s_frame_ is null";
        break;
      }

      DataFormat nv_fmt = DataFormat::INVALID;
      if (s_frame_->format == AV_PIX_FMT_NV12) {
        nv_fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;
      } else if (s_frame_->format == AV_PIX_FMT_NV21) {
        nv_fmt = DataFormat::PIXEL_FORMAT_YUV420_NV21;
      } else {
        LOGE(SOURCE) << "VideoHandlerImpl: s_frame_ format not supported: " << s_frame_->format;
        ret = -1;
        break;
      }

      // 传输到 CPU 的格式 NV12 or NV21
      DecodeFrame frame(s_frame_->height, s_frame_->width, nv_fmt);
      frame.device_type = DevType::CPU;
      frame.planeNum = 2;
      frame.pts = s_frame_->pts;

      int width = s_frame_->width;
      int height = s_frame_->height;
      size_t y_size = width * height;  // width * height 
      size_t uv_size = width * height / 2;  // width * height / 2

      uint8_t* y_buffer = new (std::nothrow) uint8_t[y_size];
      uint8_t* uv_buffer = new (std::nothrow) uint8_t[uv_size];
      if (!y_buffer || !uv_buffer) {
        LOGE(SOURCE) << "Failed to allocate memory for frame data";
        delete[] y_buffer;
        delete[] uv_buffer;
        ret = -1;
        break;
      }
      // src: according to linesize, dst: according to width
      for (int i = 0; i < height; ++i) {
        memcpy(y_buffer + i * width, s_frame_->data[0] + i * s_frame_->linesize[0], width);
      }
      for (int i = 0; i < height / 2; ++i) {
        memcpy(uv_buffer + i * width, s_frame_->data[1] + i * s_frame_->linesize[1], width);
      }

      frame.plane[0] = y_buffer;
      frame.plane[1] = uv_buffer;
      frame.stride[0] = width;
      frame.stride[1] = width;
      frame.buf_ref = std::make_unique<MatBufRefNV12>(y_buffer, uv_buffer);  // 后续可能 zero-copy，因此需要创建 buf_ref

      data = OnDecodeFrame(&frame);

    } else if (output_type_ == OutputType::OUTPUT_CUDA) {

      if (p_frame->format != AV_PIX_FMT_CUDA) {
        LOGE(SOURCE) << "VideoHandlerImpl: p_frame format not supported: " << p_frame->format;
        ret = -1;
        break;
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
        LOGW(SOURCE) << "VideoHandlerImpl: stride[0] != stride[1]: " << frame.stride[0] << " != " << frame.stride[1];
      }

      // 后续需要创建，拷贝到 data 的内存，因此不设置 buf_ref
      data = OnDecodeFrame(&frame);
    } else {
      LOGF(SOURCE) << "VideoHandler: nsupported output type: " << static_cast<int>(output_type_);
      ret = -1;
      break;
    }  // end if (output_type_ == OutputType::OUTPUT_CPU)

    if (!module_ || !handler_) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: module_ or handler_ is null";
      ret = -1;
      break;
    }

    if (!data) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: data is null";
      ret = -1;
      break;
    }

    handler_->SendData(data);
    av_frame_free(&p_frame);
    av_frame_free(&sw_frame);

  }  // end while (running_.load())

  av_frame_free(&p_frame);
  av_frame_free(&sw_frame);
  return ret;
}

void VideoHandlerImpl::clean_up() {
  av_frame_free(&s_frame_);
  av_frame_free(&cv_frame_);
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
}

bool VideoHandlerImpl::Open() {
  stream_url_ = param_set_.at(KEY_STREAM_URL);
  if (stream_url_.empty()) {
    LOGE(SOURCE) << "VideoHandlerImpl: stream_url is empty";
    return false;
  }

  if (param_set_.find(KEY_FRAME_RATE) != param_set_.end()) {
    frame_rate_ = std::stoi(param_set_.at(KEY_FRAME_RATE));
  }

  if (module_) {
    DataSourceParam source_param = module_->GetSourceParam();
    device_id_ = source_param.device_id_;
    output_type_ = source_param.output_type_;
    LOGI(SOURCE) << "VideoHandlerImpl: device_id=" << device_id_ 
                 << ", output_type=" << static_cast<int>(output_type_);
  }

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
  if (!support_hwdevice()) {
    LOGE(SOURCE) << "VideoHandlerImpl: hardware device not supported";
    OnEndFrame();
    return;
  }

  if (input_format_init() < 0) {
    LOGE(SOURCE) << "VideoHandlerImpl: input_format_init failed";
    OnEndFrame();
    return;
  }

  if (codec_init() < 0) {
    LOGE(SOURCE) << "VideoHandlerImpl: codec_init failed";
    OnEndFrame();
    return;
  }

  if (output_type_ == OutputType::OUTPUT_CPU) {
    if (convert_frame_init() < 0) {
      LOGE(SOURCE) << "VideoHandlerImpl: convert_frame_init failed";
      OnEndFrame();
      return;
    }
  }

  FrController controller(frame_rate_);
  if (frame_rate_ > 0) controller.Start();

  while (running_.load()) {
    int ret = av_read_frame(ifmt_ctx_, &pkt_);
    if (ret < 0) {
      LOGE(SOURCE) << "VideoHandlerImpl: av_read_frame error";
      break;
    }
    if (pkt_.stream_index != video_index_) {
      av_packet_unref(&pkt_);
      continue;
    }
    ret = decode_write();
    if (ret < 0) {
      LOGE(SOURCE) << "VideoHandlerImpl: decode_write error";
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
    LOGW(SOURCE) << "[VideoHandlerImpl] OnDecodeFrame function frame is nullptr.";
    return nullptr;
  }
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo();
  if (!data) {
    LOGW(SOURCE) << "[VideoHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return nullptr;
  }
  data->timestamp = frame->pts;
  if (!frame->valid) {
    data->flags = static_cast<size_t>(DataFrameFlag::FRAME_FLAG_INVALID);
    this->SendFrameInfo(data);
    return nullptr;
  }
  int ret = SourceRender::Process(data, frame, frame_id_++);
  if (ret < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: SetupDataFrame function, failed to setup data frame.";
    return nullptr;
  }
  return data;
}

void VideoHandlerImpl::OnEndFrame() {
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo(true);
  if (!data) {
    LOGW(SOURCE) << "[VideoHandlerImpl] OnEndFrame function, failed to create FrameInfo.";
    return;
  }
  this->SendFrameInfo(data);
  LOGI(SOURCE) << "[VideoHandlerImpl] OnEndFrame function, send end frame.";
}

}  // namespace cnstream
