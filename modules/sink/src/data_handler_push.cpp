#include "data_handler_push.hpp"
#include "cnstream_logging.hpp"
#include "data_converter.hpp"

#include <algorithm>
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

  output_preset_           = GetStrParam(param_set_, key_output_preset);
  output_tune_             = GetStrParam(param_set_, key_output_tune);
  output_profile_          = GetStrParam(param_set_, key_output_profile);
  output_gop_              = GetIntParam(param_set_, key_output_gop);
  output_timeout_ms_       = GetIntParam(param_set_, key_output_timeout_ms).value_or(output_timeout_ms_);
  output_tcp_nodelay_      = GetIntParam(param_set_, key_output_tcp_nodelay).value_or(output_tcp_nodelay_);
  output_send_buffer_size_ = GetIntParam(param_set_, key_output_send_buffer_size).value_or(output_send_buffer_size_);

  mark_render_ = GetIntParam(param_set_, key_mark_enable).value_or(0) != 0;
  if (mark_render_) {
    mark_config_.draw_label = GetIntParam(param_set_, key_mark_label).value_or(0) != 0;
    mark_config_.draw_score = GetIntParam(param_set_, key_mark_score).value_or(0) != 0;

    auto mark_filter = GetStrParam(param_set_, key_mark_filter);
    if (mark_filter && !mark_filter->empty()) {
      if (!mark_config_.ParseMarkFilter(*mark_filter)) {
        LOGE(SINK) << "[" << stream_id_ << "]: invalid mark '" << *mark_filter
                   << "', mark will be disabled";
        mark_render_ = false;
      } else if (!mark_config_.filter_model_ids.empty()) {
        LOGI(SINK) << "[" << stream_id_ << "]: mark filter enabled, " << *mark_filter;
      }
    }
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
  first_frame_ = true;
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
  first_frame_ = true;
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
  task.enqueue_time = std::chrono::steady_clock::now();

  if (!encode_queue_.Push(task)) {
    LOGW(SINK) << "[DEBUG-B] encode queue full, dropping frame stream_id=" << stream_id_
               << " queue_size=" << encode_queue_.Size();
    return 0;
  }

  return 0;
}

bool PushHandlerIm::InitStream() {

  ClearStream();

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
  const int gop_size = output_gop_.value_or(fps_);
  ctx_.codec_ctx->bit_rate       = bitrate_kbps_ * 1000;
  ctx_.codec_ctx->rc_min_rate    = bitrate_kbps_ * 1000;
  ctx_.codec_ctx->rc_max_rate    = bitrate_kbps_ * 1000;
  ctx_.codec_ctx->rc_buffer_size = bitrate_kbps_ * 1000;
  ctx_.codec_ctx->gop_size       = gop_size;
  ctx_.codec_ctx->max_b_frames   = 0;
  ctx_.codec_ctx->pix_fmt        = kEncoderPixFmt;
  ctx_.stream->time_base = {1, 1000};  // ms

  if (!InitDeviceCtx()) {
    LOGE(SINK) << "[" << stream_id_ << "]: InitDeviceCtx failed";
    return false;
  }
  if (ctx_.fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
    ctx_.codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  std::string gop_str     = std::to_string(gop_size);
  std::string bitrate_str = std::to_string(bitrate_kbps_);
  AVDictionary* opts = nullptr;

  const std::string codec_name = codec->name;

  if (codec_name == "libx264") {
    av_dict_set(&opts, "preset", output_preset_.value_or("veryfast").c_str(), 0);
    av_dict_set(&opts, "tune", output_tune_.value_or("zerolatency").c_str(), 0);
    av_dict_set(&opts, "profile", output_profile_.value_or("baseline").c_str(), 0);
    av_dict_set(&opts, "keyint", gop_str.c_str(), 0);
    av_dict_set(&opts, "min-keyint", gop_str.c_str(), 0);
    av_dict_set(&opts, "scenecut", "0", 0);  // 关闭场景切换检测，避免生成额外关键帧
    av_dict_set(&opts, "force-cfr", "1", 0);
    av_dict_set(&opts, "vbv-maxrate", bitrate_str.c_str(), 0);
    av_dict_set(&opts, "vbv-bufsize", bitrate_str.c_str(), 0);
    av_dict_set(&opts, "nal-hrd", "cbr", 0);  // 强制 CBR，暗场/静态画面维持码率
  
  } else if (codec_name == "h264_nvenc" || codec_name == "nvenc_h264") {
    av_dict_set(&opts, "preset", output_preset_.value_or("p4").c_str(), 0);
    av_dict_set(&opts, "tune", output_tune_.value_or("ll").c_str(), 0);
    av_dict_set(&opts, "profile", output_profile_.value_or("baseline").c_str(), 0);
    av_dict_set(&opts, "rc", "cbr", 0);
    av_dict_set(&opts, "cbr", "1", 0);
    av_dict_set(&opts, "zerolatency", "1", 0);
    av_dict_set(&opts, "g", gop_str.c_str(), 0);

  } else if (codec_name == "h264_rkmpp" || codec_name == "hevc_rkmpp") {
    av_dict_set(&opts, "rc_mode", "cbr", 0);
    av_dict_set(&opts, "g", gop_str.c_str(), 0);
    if (output_profile_) {
      av_dict_set(&opts, "profile", output_profile_->c_str(), 0);
    }
    
  } else {
    // 通用 H.264 编码器仅设置 GOP，避免未知私有选项导致失败
    av_dict_set(&opts, "g", gop_str.c_str(), 0);
  }

  ret = avcodec_open2(ctx_.codec_ctx, codec, &opts);
  av_dict_free(&opts);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: avcodec_open2 failed";
    return false;
  }
  // 检查 SPS/PPS extradata（RTMP/FLV 推流关键）：
  // 硬件编码器有时不在 extradata 中生成 SPS/PPS，
  // 需从首包 side_data 提取，否则服务器无法建立解码上下文。
  if (ctx_.codec_ctx->extradata && ctx_.codec_ctx->extradata_size > 0) {
    LOGI(SINK) << "[" << stream_id_ << "]: SPS/PPS extradata: "
               << ctx_.codec_ctx->extradata_size << " bytes";
  } else {
    LOGW(SINK) << "[" << stream_id_ << "]: extradata is empty, "
               << "Hardware encoder may not have generated SPS/PPS, "
               << "RTMP/FLV streaming may fail.";
  }
  ret = avcodec_parameters_from_context(ctx_.stream->codecpar, ctx_.codec_ctx);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: avcodec_parameters_from_context failed";
    return false;
  }
  if (!(ctx_.fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    AVDictionary* avio_opts = nullptr;
    av_dict_set(&avio_opts, "tcp_nodelay", std::to_string(output_tcp_nodelay_).c_str(), 0);
    av_dict_set(&avio_opts, "send_buffer_size", std::to_string(output_send_buffer_size_).c_str(), 0);
    // 设置连接/读写超时（微秒），避免 avio_open2 在服务器不可达时无限阻塞
    av_dict_set(&avio_opts, "rw_timeout",
                std::to_string(static_cast<int64_t>(output_timeout_ms_) * 1000).c_str(), 0);
    ret = avio_open2(&ctx_.fmt_ctx->pb, output_url_.c_str(), AVIO_FLAG_WRITE, nullptr, &avio_opts);
    av_dict_free(&avio_opts);
    if (ret < 0) {
      char errbuf[AV_ERROR_MAX_STRING_SIZE];
      av_strerror(ret, errbuf, sizeof(errbuf));
      LOGE(SINK) << "[" << stream_id_ << "]: avio_open2 failed: " << errbuf
                 << " (url=" << output_url_ << ", timeout=" << output_timeout_ms_ << "ms)";
      return false;
    }
  }
  ret = avformat_write_header(ctx_.fmt_ctx, nullptr);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: avformat_write_header failed";
    return false;
  }
  ctx_.header_written = true;
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
  if (kDefaultEncoder.empty()) {
    return avcodec_find_encoder(AV_CODEC_ID_H264);
  }
  return avcodec_find_encoder_by_name(kDefaultEncoder.c_str());
}

bool PushHandlerIm::ReinitStream(int device_id) {
  std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
  ClearStream();
  device_id_ = device_id;
  return InitStream();
}

bool PushHandlerIm::InitSwsFrame() {
  ctx_.sw_frame = av_frame_alloc();
  ctx_.sw_frame->format = kSwsPixFmt;
  ctx_.sw_frame->width  = width_;
  ctx_.sw_frame->height = height_;
  int ret = av_frame_get_buffer(ctx_.sw_frame, 0);
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

  int ret = av_frame_make_writable(ctx_.sw_frame);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_make_writable failed";
    return false;
  }

  const uint8_t* src_data = static_cast<const uint8_t*>(frame->data_[0]->GetCpuData());
  int src_stride = frame->GetStride(0);

  sws_scale(ctx_.sws_ctx,
            &src_data, &src_stride,
            0, src_height,
            ctx_.sw_frame->data, ctx_.sw_frame->linesize);

  ctx_.sw_frame->pts = pts;

  AVFrame* enc_frame = ctx_.sw_frame;
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
  int ret = av_frame_make_writable(ctx_.sw_frame);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_make_writable failed";
    return false;
  }
  sws_scale(ctx_.sws_ctx,
            &src_data, &src_stride,
            0, frame->GetHeight(),
            ctx_.sw_frame->data, ctx_.sw_frame->linesize);

  ctx_.sw_frame->pts = pts;
  AVFrame* enc_frame = ctx_.sw_frame;
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
      last_write_network_error_ = (write_ret == AVERROR(ETIMEDOUT) ||
                                   write_ret == AVERROR(EIO) ||
                                   write_ret == AVERROR(ECONNRESET) ||
                                   write_ret == AVERROR(EPIPE) ||
                                   write_ret == AVERROR(ECONNREFUSED));
      LOGE(SINK) << "[" << stream_id_ << "] av_interleaved_write_frame error: " << errbuf
                 << (last_write_network_error_ ? " (network error)" : "");
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

bool PushHandlerIm::TryReconnect() {
  std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
  auto now = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - last_reconnect_time_).count();

  // 超过较长间隔后重置重连计数，允许新一轮重连
  if (elapsed_ms >= kReconnectIntervalMs * kMaxReconnectAttempts) {
    reconnect_attempts_ = 0;
  }

  if (reconnect_attempts_ >= kMaxReconnectAttempts) {
    LOGE(SINK) << "[" << stream_id_ << "]: max reconnect attempts reached, stop retrying";
    return false;
  }

  reconnect_attempts_++;
  last_reconnect_time_ = now;
  LOGI(SINK) << "[" << stream_id_ << "]: reconnect attempt " << reconnect_attempts_
             << "/" << kMaxReconnectAttempts;

  ClearStream();
  return InitStream();
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
  // 陈旧帧丢弃阈值：排队时长超过该阈值的帧视为已过期，直接丢弃。
  // 作用：网络阻塞恢复后，积压在队列里的旧帧不会以突发方式排空再次打满
  // 网络/播放器，从而打断“阻塞->积压->突发排空->再阻塞”的自维持循环。
  // 阈值取 3 个帧间隔，与 ControlFps 的 max_lag 一致。
  const int64_t stale_threshold_us =
      static_cast<int64_t>(1000000) / fps_ * 3;
  while (IsRunning()) {
    if (!encode_queue_.WaitAndTryPop(task, std::chrono::milliseconds(100))) {
      if (!IsRunning()) break;
      continue;
    }
    if (task.is_eos) {
      FlushEncoder();
      break;
    }
    // 丢弃陈旧积压帧(不经过编码/推流，开销极低，可快速清空积压)
    auto queue_age_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - task.enqueue_time).count();
    if (queue_age_us > stale_threshold_us) {
      LOGW(SINK) << "[" << stream_id_ << "]: drop stale queued frame pts="
                 << task.pts << " age_us=" << queue_age_us
                 << " threshold_us=" << stale_threshold_us;
      continue;
    }
    std::lock_guard<std::recursive_mutex> lk(stream_mtx_);
    if (!stream_initialized_) {
      if (!InitStream()) {
        LOGE(SINK) << "[" << stream_id_ << "]: InitStream failed in worker, "
                   << "will retry after " << kReconnectIntervalMs << "ms backoff";
        // 退避等待，避免对推流服务器高频重连
        auto retry_deadline = std::chrono::steady_clock::now()
                            + std::chrono::milliseconds(kReconnectIntervalMs);
        while (IsRunning() && std::chrono::steady_clock::now() < retry_deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        continue;
      }
    }
    if (!SendDataFrame(task.frame, task.src_fmt, task.pts)) {
      LOGE(SINK) << "[" << stream_id_ << "]: SendDataFrame failed in worker";
      if (last_write_network_error_) {
        last_write_network_error_ = false;
        if (!TryReconnect()) {
          LOGE(SINK) << "[" << stream_id_ << "]: TryReconnect failed, will retry on next frame";
        }
      }
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
  if (ctx_.sw_frame) av_frame_free(&ctx_.sw_frame);
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
  // 以墙钟生成 PTS：pts = 实际经过时间 * fps。
  // codec time_base={1,fps_}，故 pts 直接换算回真实秒数。
  // 这样：
  //   - 到达率 < fps 时，被接受的帧稀疏到达，PTS 间隔 > 1，反映真实到达速率
  //     （不会人为把慢速源压缩成 fps 速率播放）；
  //   - 到达率 > fps 时，ControlFps 已将接受节流到 fps，PTS 间隔 ≈ 1，反映 fps 速率；
  //   - 网络阻塞期间被 ControlFps 丢弃的帧不进入此函数，PTS 自然体现真实间隙，
  //     直播播放器可据此感知延迟并追赶。
  auto now = std::chrono::steady_clock::now();
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
      now - push_start_time_).count();
  int64_t pts = std::llround(static_cast<double>(elapsed_us) * fps_ / 1000000.0);
  if (pts <= last_pts_) {
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

// bool PushHandlerIm::ControlFps() {
//   auto now = std::chrono::steady_clock::now();
//   static thread_local std::chrono::steady_clock::time_point last_arrival =
//       std::chrono::steady_clock::now();
//   auto arrival_interval_us =
//       std::chrono::duration_cast<std::chrono::microseconds>(now - last_arrival).count();
//   last_arrival = now;
//   if (first_frame_) {
//       push_start_time_ = now;
//       last_push_time_ = now;
//       fps_stat_start_time_ = now;
//       token_bucket_last_update_ = now;
//       token_bucket_tokens_ = kTokenBucketBurstSize - 1;
//       first_frame_ = false;
//       fps_stat_frame_count_ = 1;
//       return true;
//   }

//   // Token bucket: add tokens based on elapsed time, cap at burst size
//   auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
//       now - token_bucket_last_update_).count();
//   double tokens_to_add = static_cast<double>(elapsed_us) * fps_ / 1000000.0;
//   token_bucket_tokens_ = std::min(token_bucket_tokens_ + tokens_to_add, kTokenBucketBurstSize);
//   token_bucket_last_update_ = now;

//   LOGD(SINK) << "Control check: stream_id=" << stream_id_
//              << " arrival_interval_us=" << arrival_interval_us
//              << " tokens=" << token_bucket_tokens_;

//   if (token_bucket_tokens_ < 1.0) {
//       LOGW(SINK) << "frame dropped: stream_id=" << stream_id_
//                  << " arrival_interval_us=" << arrival_interval_us
//                  << " tokens=" << token_bucket_tokens_;
//       return false;
//   }

//   token_bucket_tokens_ -= 1.0;
//   last_push_time_ = now;
//   fps_stat_frame_count_++;

//   LOGD(SINK) << "frame accepted: stream_id=" << stream_id_
//              << " arrival_interval_us=" << arrival_interval_us
//              << " tokens_after=" << token_bucket_tokens_;

//   if (fps_stat_frame_count_ >= kFpsStatInterval) {
//       auto stat_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
//           now - fps_stat_start_time_).count();
//       if (stat_elapsed_ms > 0) {
//           double actual_fps = fps_stat_frame_count_ * 1000.0 / stat_elapsed_ms;
//           LOGI(SINK) << "Actual FPS = " << actual_fps;
//       }
//       fps_stat_frame_count_ = 0;
//       fps_stat_start_time_ = now;
//   }
//   return true;
// }

bool PushHandlerIm::ControlFps() {
  using clock = std::chrono::steady_clock;
  using us = std::chrono::microseconds;
  auto now = clock::now();

  // 目标帧间隔 (us)
  const int64_t frame_interval_us = static_cast<int64_t>(1000000) / fps_;
  auto frame_interval = us(frame_interval_us);
  const int64_t min_spacing_us = frame_interval_us / 2;

  if (first_frame_) {
    push_start_time_ = now;
    fps_stat_start_time_ = now;
    fps_stat_frame_count_ = 1;
    last_push_time_ = now + frame_interval;
    first_frame_ = false;
    return true;
  }

  fps_stat_frame_count_++;
  if (fps_stat_frame_count_ >= kFpsStatInterval) {
    auto stat_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - fps_stat_start_time_).count();
    if (stat_elapsed_ms > 0) {
      double input_fps = fps_stat_frame_count_ * 1000.0 / stat_elapsed_ms;
      LOGI(SINK) << "FPS stats stream_id=" << stream_id_
                 << " input_fps=" << input_fps
                 << " target_fps=" << fps_
                 << " window_ms=" << stat_elapsed_ms;
    }
    fps_stat_frame_count_ = 0;
    fps_stat_start_time_ = now;
  }

  uint32_t queue_size = encode_queue_.Size();

  // 推进下一帧调度时间点(仅在接受路径调用)。
  // 旧实现的队列满分支直接 return 冻结 last_push_time_，阻塞解除后 now 远大于
  // last_push_time_ 必然触发 reset 大跳变，reset 后积压帧突发排空再次打满网络，
  // 形成“阻塞->冻结->reset 突发->再阻塞”的自维持循环。现队列满分支改为漂移到
  // now，接受路径用本函数维持 fps 节奏并在大间隔后 reset。
  constexpr int kMaxFrameLag = 3;
  auto max_lag = us(frame_interval_us * kMaxFrameLag);
  auto advance_schedule = [&] {
    auto ideal_next = last_push_time_ + frame_interval;
    auto earliest_next = now + us(min_spacing_us);
    if (now > last_push_time_ + max_lag) {
      last_push_time_ = now;
    } else {
      last_push_time_ = std::max(ideal_next, earliest_next);
    }
  };

  // 队列接近满：丢帧，并将调度漂移到当前时间。
  if (queue_size + 5 >= kEncodeQueueSize) {
    LOGW(SINK) << "frame dropped (queue full) stream_id=" << stream_id_
               << " queue_size=" << queue_size;
    last_push_time_ = now;
    return false;
  }

  // 过早到达：严格丢弃，保证输出不超过 fps 上限。
  if (now < last_push_time_) {
    int64_t wait_remaining_us = std::chrono::duration_cast<us>(
        last_push_time_ - now).count();
    LOGW(SINK) << "frame dropped stream_id=" << stream_id_
               << " wait_remaining_us=" << wait_remaining_us;
    return false;
  }

  // 接受当前帧并推进调度
  advance_schedule();
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
#elif defined(VSTREAM_USE_ROCKCHIP)
  impl_ = new PushHandlerImRK(module, this);
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