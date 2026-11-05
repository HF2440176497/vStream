#include "cnstream_source.hpp"
#include "data_handler_pull.hpp"
#include "data_source.hpp"
#include "data_source_param.hpp"

#include <chrono>
#include <memory>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
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

namespace {

// 断流重连策略，与 sink 侧 PushHandlerIm::TryReconnect 保持一致
constexpr int kMaxReconnectAttempts = 3;
constexpr int kReconnectIntervalMs  = 1600;
// av_read_frame 连续返回 EAGAIN 的上限：每次退避 5ms，约 3s 无数据即判定断流
constexpr int kMaxReadAgain = 600;
// 一次连接至少稳定输出数据这么久，才认为它"成功"并清零重连计数
constexpr int64_t kStableStreamMs = 30000;
// input_format_init 的特殊返回码：配置问题（没有视频轨），重试无意义。
// 取正值以区别于 FFmpeg 的负值错误码（AVERROR(ENOENT) 恰好也是 -2）。
constexpr int kErrNoVideoStream = 1;

bool IsNetworkUrl(const std::string &url) {
  static const char *kPrefixes[] = {"rtsp://", "rtsps://", "rtmp://", "rtmps://",
                                    "http://", "https://", "udp://",  "rtp://",
                                    "tcp://",  "srt://"};
  for (const char *prefix : kPrefixes) {
    if (url.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

}  // namespace

int PullHandlerIm::input_format_init() {
  int ret = 0;
  // 断流重连会多次进入本函数，而 avformat_network_init/deinit 必须一一配对，
  // 因此整个 handler 生命周期内只初始化一次，deinit 见 clean_up。
  if (!network_ready_) {
    ret = avformat_network_init();
    if (ret != 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: avformat_network_init failed: " << ret;
      return ret;
    }
    network_ready_ = true;
  }

  ifmt_ctx_ = avformat_alloc_context();
  if (!ifmt_ctx_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: avformat_alloc_context error";
    return -1;
  }
  ifmt_ctx_->interrupt_callback.callback = interrupt_cb;
  ifmt_ctx_->interrupt_callback.opaque = this;

  AVDictionary* opts = nullptr;
  if (stream_url_.rfind("rtsp://", 0) == 0) {
    av_dict_set(&opts, "buffer_size", "1024000", 0);
    av_dict_set(&opts, "max_delay", "400000", 0);
    // FFmpeg 5.0 起 RTSP 的 stimeout 已更名为 timeout(微秒)
    av_dict_set(&opts, "timeout", "20000000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
  }
  // 通用读写超时(微秒)，对 RTMP/HTTP 等 TCP 协议生效：
  // 服务器不可达时 avformat_open_input 不会无限阻塞
  av_dict_set(&opts, "rw_timeout", "20000000", 0);

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

  // 重连场景下必须复位，否则上一次的索引会被误用
  video_index_ = -1;
  for (unsigned int i = 0; i < ifmt_ctx_->nb_streams; ++i) {
    AVCodecParameters* codec_par = ifmt_ctx_->streams[i]->codecpar;
    if (codec_par->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_index_ = i;
      break;
    }
  }

  if (video_index_ < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Failed to find video stream";
    return kErrNoVideoStream;
  }

  return 0;
}

void PullHandlerIm::teardown_input() {
  // 先释放解码器上下文：它内部持有 hw_device_ctx_ 的一份引用
  if (codec_ctx_) {
    avcodec_free_context(&codec_ctx_);
  }
  codec_ = nullptr;  // 仅是编解码器注册表里的静态对象，不负责释放
  if (hw_device_ctx_) {
    // 重连时若不 unref，下一次 hw_decoder_init 会直接覆盖成员导致引用泄漏
    av_buffer_unref(&hw_device_ctx_);
  }
  if (ifmt_ctx_) {
    avformat_close_input(&ifmt_ctx_);
  }
  video_index_ = -1;
}

void PullHandlerIm::clean_up() {
  bool expected = false;
  if (!cleaned_up_.compare_exchange_strong(expected, true)) {
    return;
  }
  LOGI(SOURCE) << "[" << stream_id_ << "]: clean_up start";
  if (pkt_) {
    av_packet_free(&pkt_);
  }
  teardown_input();
  if (network_ready_) {
    avformat_network_deinit();
    network_ready_ = false;
  }
  LOGI(SOURCE) << "[" << stream_id_ << "]: clean_up done";
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
  is_network_url_ = IsNetworkUrl(stream_url_);
  if (param_set_.find(key_frame_rate) != param_set_.end()) {
    frame_rate_ = std::stoi(param_set_.at(key_frame_rate));
  }

  ConfigureOutputType();

  closed_.store(false);
  cleaned_up_.store(false);
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
  bool expected = false;
  if (!closed_.compare_exchange_strong(expected, true)) {
    while (!cleaned_up_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return;
  }
  LOGI(SOURCE) << "[" << stream_id_ << "]: Close start";
  Stop();
  if (thread_.joinable()) {
    thread_.join();
  }
  clean_up();
  LOGI(SOURCE) << "[" << stream_id_ << "]: Close done";
}

bool PullHandlerIm::WaitBeforeReconnect(int attempt, const char *stage) {
  if (attempt > kMaxReconnectAttempts) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: " << stage << " failed "
                 << kMaxReconnectAttempts << " times consecutively, give up";
    return false;
  }
  // 线性退避：1.6s / 3.2s / 4.8s，避免对刚重启的服务器高频重连
  const int wait_ms = kReconnectIntervalMs * attempt;
  LOGW(SOURCE) << "[" << stream_id_ << "]: " << stage << " failed (attempt " << attempt << "/"
               << kMaxReconnectAttempts << "), retry in " << wait_ms << "ms";
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
  while (running_.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return running_.load();
}

PullHandlerIm::ReadResult PullHandlerIm::read_loop(FrController *controller, bool *got_data) {
  int consecutive_again = 0;
  while (running_.load()) {
    int ret = av_read_frame(ifmt_ctx_, pkt_);
    if (ret < 0) {
      if (ret == AVERROR(EAGAIN)) {
        // 暂无数据（非阻塞/缓冲为空）：小睡后重试，连续过久判定断流
        if (++consecutive_again > kMaxReadAgain) {
          LOGW(SOURCE) << "[" << stream_id_ << "]: no data for about "
                       << (kMaxReadAgain * 5 / 1000) << "s, treat as broken stream";
          return ReadResult::kBroken;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      consecutive_again = 0;

      if (ret == AVERROR_EXIT) {
        // 仅由 interrupt_callback 触发
        LOGI(SOURCE) << "[" << stream_id_ << "]: read aborted by stop request";
        return ReadResult::kStopped;
      }
      if (ret == AVERROR_EOF) {
        // 网络流 EOF 视为对端关闭连接（RTSP TEARDOWN / 推流端断开），可重连；
        // 本地文件 EOF 才是正常结束，不能重连（否则会无限循环播放）。
        if (is_network_url_) {
          LOGW(SOURCE) << "[" << stream_id_ << "]: stream ended by peer";
          return ReadResult::kBroken;
        }
        LOGI(SOURCE) << "[" << stream_id_ << "]: input EOF";
        return ReadResult::kEof;
      }

      char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
      av_strerror(ret, errbuf, sizeof(errbuf));
      LOGE(SOURCE) << "[" << stream_id_ << "]: av_read_frame failed: " << errbuf << " (" << ret
                   << ")";
      return is_network_url_ ? ReadResult::kBroken : ReadResult::kFatal;
    }
    consecutive_again = 0;

    if (pkt_->stream_index != video_index_) {
      av_packet_unref(pkt_);
      continue;
    }
    ret = decode_write();
    av_packet_unref(pkt_);
    if (ret < 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: decode_write failed: " << ret;
      // 网络流重连后解码器是全新的，可能是瞬时问题；本地文件则直接放弃
      return is_network_url_ ? ReadResult::kBroken : ReadResult::kFatal;
    }
    if (got_data) *got_data = true;
    if (controller && frame_rate_ > 0) {
      controller->Control();
    }
  }  // while (running_.load())
  return ReadResult::kStopped;
}

void PullHandlerIm::Loop() {
  if (!SupportHWDevice()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: hardware device not supported";
    OnEndFrame();
    running_.store(false);
    return;
  }

  // 外层循环用于控制重连
  // attempt 记录"连续失败次数"：只有一次连接稳定出过数据（≥kStableStreamMs）才清零，
  // WaitBeforeReconnect 内检查 attempt, 如果超过最大次数，外层循环应放弃；如果没有超过，外层循环会 continue 尝试重连
  // 如果连续次数断开，稳定连接未达到时长 kStableStreamMs，则 attempt 不清零，最终会导致失败
  int attempt = 0;
  while (running_.load()) {
    const int init_ret = input_format_init();
    if (init_ret == kErrNoVideoStream) {  // 直接结束
      teardown_input();
      break;
    }
    if (init_ret < 0) {
      teardown_input();
      if (!WaitBeforeReconnect(++attempt, "open input")) break;
      continue;
    }
    if (codec_init() < 0) {
      teardown_input();
      if (!WaitBeforeReconnect(++attempt, "init decode")) break;
      continue;
    }
    if (attempt > 0) {
      LOGI(SOURCE) << "[" << stream_id_ << "]: stream re-established after " << attempt
                   << " failed attempt(s)";
    }
    frame_error_cnt_ = 0;

    FrController controller(frame_rate_);
    if (frame_rate_ > 0) controller.Start();

    bool got_data = false;
    const auto read_start = std::chrono::steady_clock::now();
    const ReadResult result = read_loop(&controller, &got_data);
    teardown_input();

    if (!running_.load()) break;  // 主动关闭
    // 只有"确实出过数据且稳定跑了一段时间"才算一次成功连接，据此清零失败计数；
    const auto streamed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - read_start).count();
    if (got_data && streamed_ms >= kStableStreamMs) {
      attempt = 0;
    }
    if (result == ReadResult::kEof || result == ReadResult::kFatal ||
        result == ReadResult::kStopped) {
      break;
    }
    if (!WaitBeforeReconnect(++attempt, "decode")) break;
  }

  LOGI(SOURCE) << "[" << stream_id_ << "]: Loop done";
  OnEndFrame();
  // 线程即将退出，确保 IsRunning() 不再上报 true（也让残留的阻塞 IO 立刻中断）
  running_.store(false);
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
    LOGW(SOURCE) << "[" << stream_id_ << "]: OnDecodeFrame: invalid frame.";
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

/**
 * @brief override function
 */
int PullHandlerImCPU::decode_write() {
  // 送包：EAGAIN 表示"解码器输出未取空"，应先取帧再重送。
  // ReceiveAndDeliver 会把可取的帧一次性取完，正常一轮即可送入；
  // 这里仍加一个上限，防止异常解码器把这里变成死循环。
  constexpr int kMaxSendRetry = 8;
  int ret = 0;
  int send_retry = 0;
  while ((ret = avcodec_send_packet(codec_ctx_, pkt_)) == AVERROR(EAGAIN)) {
    if (++send_retry > kMaxSendRetry) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: send_packet keeps returning EAGAIN after "
                   << kMaxSendRetry << " retries, abort";
      return AVERROR_EXTERNAL;
    }
    const int drain_ret = ReceiveAndDeliver();
    if (drain_ret < 0) return drain_ret;
  }
  // send 在 while 条件内，到此处说明解码器被排空
  if (ret < 0 && ret != AVERROR_EOF) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGE(SOURCE) << "[" << stream_id_ << "]: send_packet error: " << errbuf << " (" << ret << ")";
    return ret;
  }
  return ReceiveAndDeliver();
}


int PullHandlerImCPU::ReceiveAndDeliver() {
  AVFrame *p_frame = av_frame_alloc();
  if (!p_frame) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: av_frame_alloc failed";
    return AVERROR(ENOMEM);
  }

  int ret = 0;
  while (running_.load()) {
    ret = avcodec_receive_frame(codec_ctx_, p_frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      ret = 0;
      break;
    }
    if (ret < 0) {
      char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
      av_strerror(ret, errbuf, sizeof(errbuf));
      LOGE(SOURCE) << "[" << stream_id_ << "]: receive_frame error: " << errbuf << " (" << ret
                   << ")";
      break;
    }

    int frame_ret = 0;
    auto data = ProcessFrame(p_frame, frame_ret);
    // ProcessFrame 内部已把像素拷出，因此这里复用同一个 AVFrame
    av_frame_unref(p_frame);

    if (!data || frame_ret != 0) {
      if (++frame_error_cnt_ >= kMaxFrameErrorCnt) {
        LOGE(SOURCE) << "[" << stream_id_ << "]: " << frame_error_cnt_
                     << " consecutive frame errors, stop pulling this stream";
        ret = AVERROR_INVALIDDATA;
        break;
      }
      LOGW(SOURCE) << "[" << stream_id_ << "]: drop undecodable frame (cnt=" << frame_error_cnt_
                   << "/" << kMaxFrameErrorCnt << ")";
      continue;
    }
    frame_error_cnt_ = 0;

    if (!module_ || !handler_) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: module_ or handler_ is null";
      ret = AVERROR(EINVAL);
      break;
    }
    handler_->SendData(data);
  }

  av_frame_free(&p_frame);
  return ret;
}


std::shared_ptr<FrameInfo> PullHandlerImCPU::ProcessFrame(AVFrame *p_frame, int &ret) {
  if (!p_frame) {
    LOGE(SOURCE) << "ProcessFrame: p_frame is null";
    return nullptr;
  }

  // 硬件解码器输出 NV12/NV21，而 FFmpeg 软解（h264/hevc/mpeg2...）恒输出
  // AV_PIX_FMT_YUV420P（全范围是 YUVJ420P，10bit 是 YUV420P10LE，4:2:2/4:4:4 另算），
  // 这里统一转成 NV12
  AVPixelFormat src_fmt = static_cast<AVPixelFormat>(p_frame->format);
  const AVFrame *src_frame = p_frame;
  if (src_fmt != AV_PIX_FMT_NV12 && src_fmt != AV_PIX_FMT_NV21) {
    src_frame = ToNV12(p_frame);
    if (!src_frame) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: ProcessFrame: convert to NV12 failed, src_fmt="
                   << av_get_pix_fmt_name(src_fmt);
      ret = -1;
      return nullptr;
    }
    src_fmt = AV_PIX_FMT_NV12;
  }

  const DataFormat nv_fmt = (src_fmt == AV_PIX_FMT_NV12)
                                ? DataFormat::PIXEL_FORMAT_YUV420_NV12
                                : DataFormat::PIXEL_FORMAT_YUV420_NV21;

  DecodeFrame frame(src_frame->height, src_frame->width, nv_fmt);
  frame.device_type = DevType::CPU;
  frame.planeNum = 2;
  frame.pts = p_frame->pts;

  const int width  = src_frame->width;
  const int height = src_frame->height;
  const int src_y_stride  = src_frame->linesize[0];
  const int src_uv_stride = src_frame->linesize[1];
  if (width <= 0 || height <= 0 || src_y_stride <= 0 || src_uv_stride <= 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: ProcessFrame: invalid frame geometry, w=" << width
                 << " h=" << height << " y_stride=" << src_y_stride
                 << " uv_stride=" << src_uv_stride;
    ret = -1;
    return nullptr;
  }

  // NV12 的 UV 平面高度为 ceil(height/2)，奇数高度下不能直接 height/2，
  // 否则最后一个色度行既不被拷贝，av_frame 的 buffer 也可能比分配的小。
  const int uv_height = (height + 1) / 2;
  const size_t y_size  = static_cast<size_t>(width) * height;
  const size_t uv_size = static_cast<size_t>(width) * uv_height;

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
    memcpy(y_buffer + i * width, src_frame->data[0] + i * src_y_stride, width);
  }
  for (int i = 0; i < uv_height; ++i) {
    memcpy(uv_buffer + i * width, src_frame->data[1] + i * src_uv_stride, width);
  }

  frame.plane[0] = y_buffer;
  frame.plane[1] = uv_buffer;
  frame.stride[0] = width;
  frame.stride[1] = width;
  frame.buf_ref = std::make_unique<MatBufRefNV12>(y_buffer, uv_buffer);

  return OnDecodeFrame(&frame);
}

const AVFrame *PullHandlerImCPU::ToNV12(AVFrame *p_frame) {
  const int width  = p_frame->width;
  const int height = p_frame->height;
  if (width <= 0 || height <= 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: ToNV12: invalid frame size " << width << "x" << height;
    return nullptr;
  }

  if (!nv12_frame_) {
    nv12_frame_ = av_frame_alloc();
    if (!nv12_frame_) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: ToNV12: av_frame_alloc failed";
      return nullptr;
    }
  }

  // 尺寸变化（分辨率切换）或首次使用时重建 NV12 输出缓冲
  const bool need_alloc = nv12_frame_->width != width || nv12_frame_->height != height ||
                          nv12_frame_->format != AV_PIX_FMT_NV12 || !nv12_frame_->buf[0];
  if (need_alloc) {
    av_frame_unref(nv12_frame_);
    nv12_frame_->format = AV_PIX_FMT_NV12;
    nv12_frame_->width  = width;
    nv12_frame_->height = height;
    // get_buffer 内部会分配缓冲区，因此在重建时 需要先调用 av_frame_unref
    int alloc_ret = av_frame_get_buffer(nv12_frame_, 0);
    if (alloc_ret < 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: ToNV12: av_frame_get_buffer failed: " << alloc_ret;
      av_frame_unref(nv12_frame_);
      return nullptr;
    }
  } else {
    int w_ret = av_frame_make_writable(nv12_frame_);
    if (w_ret < 0) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: ToNV12: av_frame_make_writable failed: "
                   << w_ret;
      return nullptr;
    }
  }

  const AVPixelFormat src_fmt = static_cast<AVPixelFormat>(p_frame->format);
  if (!sws_ctx_ || sws_src_w_ != width || sws_src_h_ != height || sws_src_fmt_ != src_fmt) {
    sws_ctx_ = sws_getCachedContext(sws_ctx_, width, height, src_fmt,
                                    width, height, AV_PIX_FMT_NV12,
                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: ToNV12: sws_getCachedContext failed, src_fmt="
                   << av_get_pix_fmt_name(src_fmt);
      return nullptr;
    }
    sws_src_w_   = width;
    sws_src_h_   = height;
    sws_src_fmt_ = src_fmt;
    LOGI(SOURCE) << "[" << stream_id_ << "]: software decode pixel format "
                 << av_get_pix_fmt_name(src_fmt) << " -> NV12, " << width << "x" << height;
  }

  const int scaled = sws_scale(sws_ctx_, p_frame->data, p_frame->linesize, 0, height,
                               nv12_frame_->data, nv12_frame_->linesize);
  if (scaled <= 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: ToNV12: sws_scale failed: " << scaled;
    return nullptr;
  }
  return nv12_frame_;
}

void PullHandlerImCPU::ReleaseConvertResources() {
  if (sws_ctx_) {
    sws_freeContext(sws_ctx_);
    sws_ctx_ = nullptr;
  }
  if (nv12_frame_) {
    // av_frame_free 内部先 av_frame_unref 再释放 AVFrame 结构体，并把指针置空
    av_frame_free(&nv12_frame_);
  }
  sws_src_w_   = 0;
  sws_src_h_   = 0;
  sws_src_fmt_ = AV_PIX_FMT_NONE;
}

void PullHandlerImCPU::clean_up() {
  ReleaseConvertResources();
  PullHandlerIm::clean_up();
}

PullHandlerImCPU::~PullHandlerImCPU() {
  // 正常路径下 PullHandler::~PullHandler 会先调用 Close() -> clean_up()，这里已无资源可放；
  // 兜底覆盖"Open 失败后直接析构 / 从未 Close 就销毁"的场景。
  // 注意只释放本类独占的转换资源，不触发基类的 FFmpeg 全局清理（那部分由 clean_up 管）。
  ReleaseConvertResources();
}

}  // namespace cnstream