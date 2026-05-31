#include "data_handler_push.hpp"
#include "cnstream_logging.hpp"

#include <opencv2/opencv.hpp>

namespace cnstream {

bool PushHandlerImpl::Open() {
  LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerImpl Open";

  if (param_set_.find(key_output_url) == param_set_.end()) {
    LOGE(SINK) << "[" << stream_id_ << "]: output_url not set";
    return false;
  }
  output_url_   = param_set_.at(key_output_url);

  fps_          = GetIntParam(param_set_, key_output_fps).value_or(fps_);
  width_        = GetIntParam(param_set_, key_output_width).value_or(width_);
  height_       = GetIntParam(param_set_, key_output_height).value_or(height_);
  bitrate_kbps_ = GetIntParam(param_set_, key_output_bitrate).value_or(bitrate_kbps_);
  codec_name_   = GetStrParam(param_set_, key_output_codec).value_or(codec_name_);
  device_id_    = GetIntParam(param_set_, key_output_device_id).value_or(device_id_);

  mark_render_ = GetIntParam(param_set_, key_mark_enable).value_or(0) != 0;
  if (mark_render_) {
    mark_config_.draw_label = GetIntParam(param_set_, key_mark_label).value_or(0) != 0;
    mark_config_.draw_score = GetIntParam(param_set_, key_mark_score).value_or(0) != 0;
  }

  running_.store(true);
  return true;
}

void PushHandlerImpl::Stop() {
  LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerImpl Stop";
  running_.store(false);
}

void PushHandlerImpl::Close() {
  LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerImpl Close";
  running_.store(false);
  ClearStream();
}

int PushHandlerImpl::Process(const std::shared_ptr<FrameInfo> data) {
  if (!IsRunning()) {
    LOGW(SINK) << "[" << stream_id_ << "]: PushHandler not running, skip frame";
    return -1;
  }
  if (data->IsInvalid() || data->IsRemoved()) {
    LOGW(SINK) << "[" << stream_id_ << "]: frame has problems";
    return -1;
  }
  if (data->IsEos()) {
    LOGI(SINK) << "[" << stream_id_ << "]: received EOS";
    return 0;
  }
  if (!ControlFps()) {
    return 0;
  }

  DataFramePtr frame = nullptr;
  if (data->collection.HasValue(cnstream::kDataFrameTag)) {
    frame = data->collection.Get<cnstream::DataFramePtr>(cnstream::kDataFrameTag);
  }
  if (!frame) {
    LOGW(SINK) << "[" << stream_id_ << "]: no DataFrame in frame info, skip";
    return -1;
  }
  auto it = kAvFmtMap.find(frame->GetFmt());
  if (it == kAvFmtMap.end()) {
    LOGE(SINK) << "[" << stream_id_ << "]: unsupported pixel format "
               << static_cast<int>(frame->GetFmt());
    return -1;
  }

  if (mark_render_ && data->collection.HasValue(cnstream::kInferObjsTag)) {
    if (!render_) {
      render_ = MarkRender::Create(frame->GetCtx().device_type);
    }
    auto objs = data->collection.Get<cnstream::InferObjsPtr>(cnstream::kInferObjsTag);
    if (objs && !objs->objs_.empty()) {
      render_->Render(frame, objs, mark_config_);
    }
  }

  std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
  if (IsRunning()) {
    if (!stream_initialized_) {
      if (!InitStream()) {
        LOGE(SINK) << "[" << stream_id_ << "]: InitStream failed";
        return -1;
      }
    }
    if (!SendDataFrame(frame, it->second)) {
      LOGE(SINK) << "[" << stream_id_ << "]: SendDataFrame failed";
      return -1;
    }
  }
  return 0;
}

bool PushHandlerImpl::InitStream() {
  AVOutputFormat* fmt = const_cast<AVOutputFormat*>(
      av_guess_format(GetFormatFromUrl(output_url_).c_str(), nullptr, nullptr));
  if (!fmt) {
    LOGE(SINK) << "[" << stream_id_ << "]: Unknown format";
    return false;
  }

  int ret = avformat_alloc_output_context2(&ctx_.fmt_ctx, nullptr, fmt->name, output_url_.c_str());
  if (ret < 0 || !ctx_.fmt_ctx) {
    LOGE(SINK) << "[" << stream_id_ << "]: avformat_alloc_output_context2 failed";
    return false;
  }

  const AVCodec* codec = FindEncoder();
  if (!codec) {
    LOGE(SINK) << "[" << stream_id_ << "]: encoder not found";
    return false;
  }
  ctx_.stream = avformat_new_stream(ctx_.fmt_ctx, nullptr);
  if (!ctx_.stream) {
    LOGE(SINK) << "[" << stream_id_ << "]: avformat_new_stream failed";
    return false;
  }
  ctx_.codec_ctx = avcodec_alloc_context3(codec);
  if (!ctx_.codec_ctx) {
    LOGE(SINK) << "[" << stream_id_ << "]: avcodec_alloc_context3 failed";
    return false;
  }
  ctx_.codec_ctx->codec_id   = codec->id;
  ctx_.codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  ctx_.codec_ctx->width      = width_;
  ctx_.codec_ctx->height     = height_;
  ctx_.codec_ctx->time_base  = {1, fps_};
  ctx_.codec_ctx->framerate  = {fps_, 1};
  ctx_.codec_ctx->bit_rate   = bitrate_kbps_ * 1000;
  ctx_.codec_ctx->gop_size   = fps_;
  ctx_.codec_ctx->max_b_frames = 1;
  ctx_.codec_ctx->pix_fmt    = kEncoderPixFmt;
  ctx_.stream->time_base     = ctx_.codec_ctx->time_base;

  if (!InitDeviceCtx()) {
    LOGE(SINK) << "[" << stream_id_ << "]: InitDeviceCtx failed";
    return false;
  }
  if (ctx_.fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
    ctx_.codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  ret = avcodec_open2(ctx_.codec_ctx, codec, nullptr);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: avcodec_open2 failed";
    return false;
  }
  ret = avcodec_parameters_from_context(ctx_.stream->codecpar, ctx_.codec_ctx);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: avcodec_parameters_from_context failed";
    return false;
  }
  if (!(ctx_.fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&ctx_.fmt_ctx->pb, output_url_.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
      LOGE(SINK) << "[" << stream_id_ << "]: avio_open failed";
      return false;
    }
  }
  ret = avformat_write_header(ctx_.fmt_ctx, nullptr);
  ctx_.header_written = true;
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: avformat_write_header failed";
    return false;
  }
  if (!InitSwsFrame()) {
    LOGE(SINK) << "[" << stream_id_ << "]: InitSwsFrame failed";
    return false;
  }
  LOGI(SINK) << "[" << stream_id_ << "]: Stream initialized, url=" << output_url_
             << " res=" << width_ << "x" << height_ << " fps=" << fps_;
  stream_initialized_ = true;
  return true;
}

const AVCodec* PushHandlerImpl::FindEncoder() {
  if (!codec_name_.empty()) {
    return avcodec_find_encoder_by_name(codec_name_.c_str());
  }
#ifdef VSTREAM_USE_CUDA
  return avcodec_find_encoder_by_name(kDefaultEncoder);
#else
  return avcodec_find_encoder(AV_CODEC_ID_H264);
#endif
}

bool PushHandlerImpl::ReinitStream(int device_id) {
  std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
  ClearStream();
  device_id_ = device_id;
  return InitStream();
}

bool PushHandlerImpl::InitSwsFrame() {
  ctx_.sws_frame = av_frame_alloc();
  ctx_.sws_frame->format = kSwsPixFmt;
  ctx_.sws_frame->width  = width_;
  ctx_.sws_frame->height = height_;
  int ret = av_frame_get_buffer(ctx_.sws_frame, 0);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_get_buffer failed";
    return false;
  }
  return true;
}

bool PushHandlerImpl::SendFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt) {
  const int src_width  = frame->GetWidth();
  const int src_height = frame->GetHeight();
  EnsureSwsContext(src_pix_fmt, src_width, src_height);

  int ret = av_frame_make_writable(ctx_.sws_frame);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_make_writable failed";
    return false;
  }

  const uint8_t* src_data = static_cast<const uint8_t*>(frame->data_[0]->GetCpuData());
  int src_stride = frame->GetStride(0);

  sws_scale(ctx_.sws_ctx,
            &src_data, &src_stride,
            0, src_height,
            ctx_.sws_frame->data, ctx_.sws_frame->linesize);

  ctx_.sws_frame->pts = ComputePts();

  AVFrame* enc_frame = ctx_.sws_frame;
  if (!enc_frame) return false;
  return EncodeFrame(enc_frame);
}

void PushHandlerImpl::EnsureSwsContext(AVPixelFormat src_pix_fmt, int src_width, int src_height) {
  if (ctx_.sws_ctx && src_pix_fmt_ == src_pix_fmt
      && sws_src_width_ == src_width && sws_src_height_ == src_height) {
    return;
  }
  ctx_.sws_ctx = sws_getCachedContext(
    ctx_.sws_ctx,
    src_width, src_height, src_pix_fmt,
    ctx_.codec_ctx->width, ctx_.codec_ctx->height, kSwsPixFmt,
    SWS_BILINEAR, nullptr, nullptr, nullptr);
  src_pix_fmt_    = src_pix_fmt;
  sws_src_width_  = src_width;
  sws_src_height_ = src_height;
}

bool PushHandlerImpl::SendFrameCpuFallback(const DataFramePtr& frame, AVPixelFormat src_pix_fmt) {
  auto src_data = static_cast<const uint8_t*>(frame->data_[0]->GetCpuData());
  int src_stride = frame->GetStride(0);

  EnsureSwsContext(src_pix_fmt, frame->GetWidth(), frame->GetHeight());
  int ret = av_frame_make_writable(ctx_.sws_frame);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_make_writable failed";
    return false;
  }
  sws_scale(ctx_.sws_ctx,
            &src_data, &src_stride,
            0, frame->GetHeight(),
            ctx_.sws_frame->data, ctx_.sws_frame->linesize);

  ctx_.sws_frame->pts = ComputePts();
  AVFrame* enc_frame = ctx_.sws_frame;
  if (!enc_frame) return false;
  return EncodeFrame(enc_frame);
}

bool PushHandlerImpl::EncodeFrame(AVFrame* frame) {
  int ret;
  while ((ret = avcodec_send_frame(ctx_.codec_ctx, frame)) == AVERROR(EAGAIN)) {
    AVPacket* drain_pkt = av_packet_alloc();
    ret = avcodec_receive_packet(ctx_.codec_ctx, drain_pkt);
    if (ret == 0) {
      av_packet_rescale_ts(drain_pkt, ctx_.codec_ctx->time_base, ctx_.stream->time_base);
      drain_pkt->stream_index = ctx_.stream->index;
      int write_ret = av_interleaved_write_frame(ctx_.fmt_ctx, drain_pkt);
      if (write_ret < 0) {
        char errbuf[1024];
        av_strerror(write_ret, errbuf, sizeof(errbuf));
        LOGE(SINK) << "[" << stream_id_ << "]: av_interleaved_write_frame error during drain: " << errbuf;
        av_packet_free(&drain_pkt);
        return false;
      }
    }
    av_packet_free(&drain_pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      ret = 0;
      break;
    }
    if (ret < 0) {
      LOGE(SINK) << "[" << stream_id_ << "]: avcodec_receive_packet error during drain";
      return false;
    }
  }
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: avcodec_send_frame error: " << ret;
    return false;
  }

  AVPacket* pkt = av_packet_alloc();
  while (IsRunning()) {
    ret = avcodec_receive_packet(ctx_.codec_ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      break;
    }
    if (ret < 0) {
      LOGE(SINK) << "[" << stream_id_ << "]: avcodec_receive_packet error: " << ret;
      av_packet_free(&pkt);
      return false;
    }
    av_packet_rescale_ts(pkt, ctx_.codec_ctx->time_base, ctx_.stream->time_base);
    pkt->stream_index = ctx_.stream->index;
    ret = av_interleaved_write_frame(ctx_.fmt_ctx, pkt);
    if (ret < 0) {
      char errbuf[1024];
      av_strerror(ret, errbuf, sizeof(errbuf));
      std::string error_msg(errbuf);
      LOGE(SINK) << "[" << stream_id_ << "]: av_interleaved_write_frame error: " << error_msg;
      av_packet_free(&pkt);
      return false;
    }
  }
  av_packet_free(&pkt);
  return true;
}

void PushHandlerImpl::ClearStream() {
  AVPacket* pkt = nullptr;
  std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
  if (ctx_.codec_ctx && ctx_.stream) {
    avcodec_send_frame(ctx_.codec_ctx, nullptr);
    pkt = av_packet_alloc();
    while (avcodec_receive_packet(ctx_.codec_ctx, pkt) == 0) {
      av_packet_rescale_ts(pkt, ctx_.codec_ctx->time_base, ctx_.stream->time_base);
      pkt->stream_index = ctx_.stream->index;
      av_interleaved_write_frame(ctx_.fmt_ctx, pkt);
    }
  }
  if (pkt) av_packet_free(&pkt);
  if (ctx_.header_written) {
      av_write_trailer(ctx_.fmt_ctx);
  }
  CleanDeviceCtx();
  if (ctx_.sws_frame) av_frame_free(&ctx_.sws_frame);
  if (ctx_.sws_ctx)   sws_freeContext(ctx_.sws_ctx);
  if (ctx_.codec_ctx) avcodec_free_context(&ctx_.codec_ctx);
  if (ctx_.fmt_ctx) {
    if (!(ctx_.fmt_ctx->oformat->flags & AVFMT_NOFILE))
      avio_closep(&ctx_.fmt_ctx->pb);
    avformat_free_context(ctx_.fmt_ctx);
  }
  ctx_ = StreamContext();
  stream_initialized_ = false;
  LOGI(SINK) << "[" << stream_id_ << "]: Stream clean done";
}

int64_t PushHandlerImpl::ComputePts() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
      now - push_start_time_).count();
  return elapsed_us * fps_ / 1000000;
}

bool PushHandlerImpl::ControlFps() {
  auto now = std::chrono::steady_clock::now();
  if (first_frame_) {
    push_start_time_ = now;
    fps_stat_start_time_ = now;
    last_push_time_ = now;
    first_frame_ = false;
    fps_stat_frame_count_++;
    return true;
  }

  auto elapsed = now - last_push_time_;
  auto frame_interval = std::chrono::microseconds(1000000 / fps_);
  if (elapsed < frame_interval) {
    return false;
  }
  last_push_time_ = now;

  fps_stat_frame_count_++;
  if (fps_stat_frame_count_ >= kFpsStatInterval) {
    auto stat_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - fps_stat_start_time_).count();
    if (stat_elapsed_ms > 0) {
      double actual_fps = fps_stat_frame_count_ * 1000.0 / stat_elapsed_ms;
      LOGI(SINK) << "[" << stream_id_ << "]: Actual FPS = " << actual_fps
                 << " (frames=" << fps_stat_frame_count_
                 << ", duration=" << stat_elapsed_ms << "ms)";
    }
    fps_stat_frame_count_ = 0;
    fps_stat_start_time_ = now;
  }
  return true;
}


bool PushHandlerImplCPU::SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt) {
  auto dev_type = frame->GetCtx().device_type;
  if (dev_type == DevType::CPU) {
    return SendFrame(frame, src_pix_fmt);
  }
  LOGE(SINK) << "[" << stream_id_ << "]: unknown device type " << DevType2Str(dev_type);
  return true;
}

std::shared_ptr<SinkHandler> PushHandler::Create(DataSink *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SINK) << "[" << stream_id << "]: module is null";
    return nullptr;
  }
  return std::shared_ptr<SinkHandler>(new PushHandler(module, stream_id));
}

PushHandler::PushHandler(DataSink *module, const std::string &stream_id)
    : SinkHandler(module, stream_id) {
#ifdef VSTREAM_USE_CUDA
  impl_ = new PushHandlerImplCUDA(module, this);
#else
  impl_ = new PushHandlerImplCPU(module, this);
#endif
}

PushHandler::~PushHandler() {
  Close();
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

bool PushHandler::Open() {
  if (!module_) {
    LOGE(SINK) << "[" << stream_id_ << "]: module_ null";
    return false;
  }
  if (!impl_) {
    LOGE(SINK) << "[" << stream_id_ << "]: Push handler open failed, no memory left";
    return false;
  }
  if (!impl_->Open()) {
    impl_->Close();
    return false;
  }
  return true;
}

void PushHandler::Close() {
  if (impl_) {
    impl_->Close();
  }
}

void PushHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

int PushHandler::Process(const std::shared_ptr<FrameInfo> data) {
  if (!data) {
    return -1;
  }
  if (data->IsEos() || data->IsInvalid()) {
    return 0;
  }
  if (!impl_) {
    return -1;
  }
  return impl_->Process(data);
}

void PushHandler::RegisterHandlerParams() {
}

/**
 * 验证相关参数的存在
 * @param params 来自 DataSink 的参数
 */
bool PushHandler::CheckHandlerParams(const ModuleParamSet& params) {
  DataSink* ds = dynamic_cast<DataSink*>(module_);
  ModuleParamSet stream_params;
  const ModuleParamSet* check_params = &params;
  if (ds) {
    stream_params = ds->GetStreamParams(stream_id_);
    if (!stream_params.empty()) {
      check_params = &stream_params;
    }
  }
  if (check_params->find(key_output_url) == check_params->end()) {
    LOGE(SINK) << "[" << stream_id_ << "]: push output_url not set";
    return false;
  }
  if (check_params->find(key_output_fps) == check_params->end()) {
    LOGE(SINK) << "[" << stream_id_ << "]: push output_fps not set";
    return false;
  }
  if (check_params->find(key_output_width) == check_params->end()) {
    LOGE(SINK) << "[" << stream_id_ << "]: push output_width not set";
    return false;
  }
  if (check_params->find(key_output_height) == check_params->end()) {
    LOGE(SINK) << "[" << stream_id_ << "]: push output_height not set";
    return false;
  }
  return true;
}

/**
 * @brief CheckHandlerParams SetHandlerParams 是在 AddSink 调用的
 */
bool PushHandler::SetHandlerParams(const ModuleParamSet& params) {
  if (!impl_) {
    return false;
  }
  DataSink* ds = dynamic_cast<DataSink*>(module_);
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

}  // namespace cnstream