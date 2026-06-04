#include "data_handler_push.hpp"
#include "cnstream_logging.hpp"
#include "data_converter.hpp"

#include <opencv2/opencv.hpp>
#include <cmath>

namespace cnstream {

bool PushHandlerIm::Open() {
  LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerIm Open";

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
  encode_thread_ = std::thread(&PushHandlerIm::EncodeWorkerLoop, this);
  return true;
}

void PushHandlerIm::Stop() {
  LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerIm Stop";
  running_.store(false);
  encode_queue_.Stop();
  if (encode_thread_.joinable()) {
    encode_thread_.join();
  }
}

void PushHandlerIm::Close() {
  LOGI(SINK) << "[" << stream_id_ << "]: PushHandlerIm Close";
  EncoderTask eos;
  eos.is_eos = true;
  encode_queue_.Push(eos);
  running_.store(false);
  encode_queue_.Stop();
  if (encode_thread_.joinable()) {
    encode_thread_.join();
  }
  ClearStream();
}

int PushHandlerIm::Process(const std::shared_ptr<FrameInfo> data) {
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

  SaveOriFrame(data);

  if (mark_render_ && data->collection.HasValue(cnstream::kInferObjsTag)) {
    if (!render_) {
      render_ = MarkRender::Create(frame->GetCtx().device_type);
    }
    auto objs = data->collection.Get<cnstream::InferObjsPtr>(cnstream::kInferObjsTag);
    if (objs && !objs->objs_.empty()) {
      render_->Render(frame, objs, mark_config_);
    }
  }

  EncoderTask task;
  task.frame   = frame;
  task.src_fmt = it->second;
  task.pts     = ComputePts();
  task.is_eos  = false;

  if (!encode_queue_.Push(task)) {
    LOGW(SINK) << "[" << stream_id_ << "]: encode queue full, dropping frame";
    return 0;
  }
  pts_count_++;
  return 0;
}

bool PushHandlerIm::InitStream() {
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
  ctx_.codec_ctx->max_b_frames = 0;
  ctx_.codec_ctx->pix_fmt    = kEncoderPixFmt;
  ctx_.stream->time_base = {1, 1000};  // ms

  if (!InitDeviceCtx()) {
    LOGE(SINK) << "[" << stream_id_ << "]: InitDeviceCtx failed";
    return false;
  }
  if (ctx_.fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
    ctx_.codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  auto fps_str = std::to_string(fps_);
  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "keyint", fps_str.c_str(), 0);
  av_dict_set(&opts, "min-keyint", fps_str.c_str(), 0);
  av_dict_set(&opts, "scenecut", "0", 0);  // 关闭场景切换检测，避免生成额外关键帧

  ret = avcodec_open2(ctx_.codec_ctx, codec, &opts);
  av_dict_free(&opts);
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

const AVCodec* PushHandlerIm::FindEncoder() {
  if (!codec_name_.empty()) {
    return avcodec_find_encoder_by_name(codec_name_.c_str());
  }
#ifdef VSTREAM_USE_CUDA
  return avcodec_find_encoder_by_name(kDefaultEncoder);
#else
  return avcodec_find_encoder(AV_CODEC_ID_H264);
#endif
}

bool PushHandlerIm::ReinitStream(int device_id) {
  std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
  ClearStream();
  device_id_ = device_id;
  return InitStream();
}

bool PushHandlerIm::InitSwsFrame() {
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

bool PushHandlerIm::SendFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) {
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

  ctx_.sws_frame->pts = pts;

  AVFrame* enc_frame = ctx_.sws_frame;
  if (!enc_frame) return false;
  return EncodeFrame(enc_frame);
}

void PushHandlerIm::EnsureSwsContext(AVPixelFormat src_pix_fmt, int src_width, int src_height) {
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

bool PushHandlerIm::SendFrameFb(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) {
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

  ctx_.sws_frame->pts = pts;
  AVFrame* enc_frame = ctx_.sws_frame;
  if (!enc_frame) return false;
  return EncodeFrame(enc_frame);
}

bool PushHandlerIm::EncodeFrame(AVFrame* frame) {
  int ret;
  while ((ret = avcodec_send_frame(ctx_.codec_ctx, frame)) == AVERROR(EAGAIN)) {
    if (!DrainPackets()) {
      return false;
    }
  }
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "] avcodec_send_frame error: " << ret;
    return false;
  }
  return DrainPackets();
}

bool PushHandlerIm::DrainPackets() {
  while (IsRunning()) {
    AVPacket* pkt = av_packet_alloc();
    int ret = avcodec_receive_packet(ctx_.codec_ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_packet_free(&pkt);
      break;
    }
    if (ret < 0) {
      LOGE(SINK) << "[" << stream_id_ << "] avcodec_receive_packet error: " << ret;
      av_packet_free(&pkt);
      return false;
    }
    av_packet_rescale_ts(pkt, ctx_.codec_ctx->time_base, ctx_.stream->time_base);
    pkt->stream_index = ctx_.stream->index;
    pkt->duration = av_rescale_q(pkt->duration, ctx_.codec_ctx->time_base, ctx_.stream->time_base);

    int write_ret = av_interleaved_write_frame(ctx_.fmt_ctx, pkt);
    av_packet_free(&pkt);

    if (write_ret < 0) {
      char errbuf[AV_ERROR_MAX_STRING_SIZE];
      av_strerror(write_ret, errbuf, sizeof(errbuf));
      LOGE(SINK) << "[" << stream_id_ << "] av_interleaved_write_frame error: " << errbuf;
      return false;
    }
  }
  return true;
}

bool PushHandlerIm::FlushEncoder() {
  int ret = avcodec_send_frame(ctx_.codec_ctx, nullptr);
  if (ret < 0 && ret != AVERROR_EOF) {
    LOGE(SINK) << "[" << stream_id_ << "] FlushEncoder send_frame error: " << ret;
    return false;
  }
  return DrainPackets();
}

// void PushHandlerIm::FlushEncoder() {
//   std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
//   if (!ctx_.codec_ctx || !ctx_.stream) return;
//   int ret = avcodec_send_frame(ctx_.codec_ctx, nullptr);
//   if (ret < 0 && ret != AVERROR_EOF) {
//     LOGE(SINK) << "[" << stream_id_ << "]: flush send_frame failed";
//     return;
//   }
//   AVPacket* pkt = av_packet_alloc();
//   while (avcodec_receive_packet(ctx_.codec_ctx, pkt) == 0) {
//     av_packet_rescale_ts(pkt, ctx_.codec_ctx->time_base, ctx_.stream->time_base);
//     pkt->stream_index = ctx_.stream->index;
//     av_interleaved_write_frame(ctx_.fmt_ctx, pkt);
//     av_packet_free(&pkt);
//   }
//   av_packet_free(&pkt);
// }

void PushHandlerIm::EncodeWorkerLoop() {
  LOGI(SINK) << "[" << stream_id_ << "]: EncodeWorkerLoop started";
  EncoderTask task;
  while (IsRunning()) {
    if (!encode_queue_.WaitAndTryPop(task, std::chrono::milliseconds(100))) {
      if (!IsRunning()) break;
      continue;
    }
    if (task.is_eos) {
      FlushEncoder();
      break;
    }
    std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
    if (!stream_initialized_) {
      if (!InitStream()) {
        LOGE(SINK) << "[" << stream_id_ << "]: InitStream failed in worker";
        continue;
      }
    }
    if (!SendDataFrame(task.frame, task.src_fmt, task.pts)) {
      LOGE(SINK) << "[" << stream_id_ << "]: SendDataFrame failed in worker";
    }
  }
  LOGI(SINK) << "[" << stream_id_ << "]: EncodeWorkerLoop exited";
}

void PushHandlerIm::ClearStream() {
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

int64_t PushHandlerIm::ComputePts() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
      now - push_start_time_).count();
  // int64_t pts = av_rescale_q(elapsed_us, {1, 1000000}, {1, fps_});
  int64_t pts = std::llround(static_cast<double>(elapsed_us) * fps_ / 1000000.0);
  if (pts <= last_pts_) {
    LOGW(SINK) << "[" << stream_id_ << "]: pts <= last_pts_ " << pts << " <= " << last_pts_;
    pts = last_pts_ + 1;
  }
  last_pts_ = pts;
  return pts;
}

// 存在的问题：last_push_time_ 取 now 时，时间间隔可能不均匀
// bool PushHandlerIm::ControlFps() {
//   auto now = std::chrono::steady_clock::now();
//   if (first_frame_) {
//     push_start_time_ = now;
//     fps_stat_start_time_ = now;
//     last_push_time_ = now;
//     first_frame_ = false;
//     fps_stat_frame_count_++;
//     return true;
//   }
//   auto frame_interval = std::chrono::microseconds(1000000 / fps_);
//   auto next_expected = last_push_time_ + frame_interval;
//   if (now < next_expected) {
//     return false;
//   }
//   if (now > next_expected + frame_interval) {
//     last_push_time_ = now;
//   } else {
//     last_push_time_ = next_expected;
//   }
//   fps_stat_frame_count_++;
//   if (fps_stat_frame_count_ >= kFpsStatInterval) {
//     auto stat_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
//         now - fps_stat_start_time_).count();
//     if (stat_elapsed_ms > 0) {
//       double actual_fps = fps_stat_frame_count_ * 1000.0 / stat_elapsed_ms;
//       LOGI(SINK) << "[" << stream_id_ << "]: Actual FPS = " << actual_fps
//                  << " (frames=" << fps_stat_frame_count_
//                  << ", duration=" << stat_elapsed_ms << "ms)";
//     }
//     fps_stat_frame_count_ = 0;
//     fps_stat_start_time_ = now;
//   }
//   return true;
// }

bool PushHandlerIm::ControlFps() {
  auto now = std::chrono::steady_clock::now();
  if (first_frame_) {
      push_start_time_ = now;
      next_frame_time_ = now;
      last_push_time_ = now;
      fps_stat_start_time_ = now;
      first_frame_ = false;
      fps_stat_frame_count_ = 1;
      return true;
  }
  auto frame_interval = std::chrono::microseconds(1000000 / fps_);
  if (now < next_frame_time_) {
      return false;
  }
  next_frame_time_ = std::max(now, next_frame_time_) + frame_interval;
  last_push_time_ = now;
  fps_stat_frame_count_++;
  if (fps_stat_frame_count_ >= kFpsStatInterval) {
      auto stat_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - fps_stat_start_time_).count();
      if (stat_elapsed_ms > 0) {
          double actual_fps = fps_stat_frame_count_ * 1000.0 / stat_elapsed_ms;
          LOGI(SINK) << "Actual FPS = " << actual_fps;
      }
      fps_stat_frame_count_ = 0;
      fps_stat_start_time_ = now;
  }
  return true;
}

bool PushHandlerImCPU::SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) {
  auto dev_type = frame->GetCtx().device_type;
  if (dev_type == DevType::CPU) {
    return SendFrame(frame, src_pix_fmt, pts);
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
  impl_ = new PushHandlerImCUDA(module, this);
#else
  impl_ = new PushHandlerImCPU(module, this);
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