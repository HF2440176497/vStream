#ifndef MODULES_SINK_HANDLER_PUSH_HPP_
#define MODULES_SINK_HANDLER_PUSH_HPP_

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cnstream_logging.hpp"
#include "cnstream_frame_va.hpp"
#include "data_common.hpp"
#include "data_sink.hpp"
#include "mark_render.hpp"
#include "memop.hpp"
#include "util/cnstream_queue.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
#include <libavutil/error.h>
}

#ifdef VSTREAM_USE_CUDA
static constexpr AVPixelFormat kEncoderPixFmt = AV_PIX_FMT_CUDA;
static constexpr AVPixelFormat kSwsPixFmt     = AV_PIX_FMT_NV12;
static std::string kDefaultEncoder = "h264_nvenc";

#elif defined(VSTREAM_USE_ROCKCHIP)
static constexpr AVPixelFormat kEncoderPixFmt = AV_PIX_FMT_DRM_PRIME;
static constexpr AVPixelFormat kSwsPixFmt     = AV_PIX_FMT_NV12;
static std::string kDefaultEncoder = "h264_rkmpp";

#else
static constexpr AVPixelFormat kEncoderPixFmt = AV_PIX_FMT_YUV420P;
static constexpr AVPixelFormat kSwsPixFmt     = AV_PIX_FMT_YUV420P;
static std::string kDefaultEncoder;
#endif

namespace cnstream {

struct StreamContext {
  AVFormatContext* fmt_ctx   = nullptr;
  AVCodecContext*  codec_ctx = nullptr;
  AVStream*        stream    = nullptr;
  SwsContext*      sws_ctx   = nullptr;
  AVFrame*         sw_frame = nullptr;
  uint64_t         frame_idx = 0;
  bool             header_written = false;

#ifdef VSTREAM_USE_CUDA
  AVBufferRef*     hw_device_ctx = nullptr;
  AVBufferRef*     hw_frames_ctx = nullptr;
  AVFrame*         hw_frame      = nullptr;

#elif defined(VSTREAM_USE_ROCKCHIP)
  AVBufferRef*     hw_device_ctx = nullptr;
  AVBufferRef*     hw_frames_ctx = nullptr;
  AVFrame*         hw_frame      = nullptr;

#endif
};

static std::string GetFormatFromUrl(const std::string& url) {
  if (url.find("rtmp://") == 0) return "flv";
  if (url.find("rtsp://") == 0) return "rtsp";
  if (url.find("http://") == 0 || url.find("https://") == 0) return "mpegts";
  return "flv";
}

static const std::unordered_map<DataFormat, AVPixelFormat> kAvFmtMap = {
  {DataFormat::PIXEL_FORMAT_RGB24, AV_PIX_FMT_RGB24},
  {DataFormat::PIXEL_FORMAT_BGR24, AV_PIX_FMT_BGR24},
};

struct EncoderTask {
  DataFramePtr  frame;
  AVPixelFormat src_fmt = AV_PIX_FMT_RGB24;
  int64_t       pts = 0;
  bool          is_eos = false;
  // 入队时刻，用于 EncodeWorkerLoop 丢弃网络阻塞期间积压的陈旧帧，
  // 避免阻塞恢复后积压帧以突发方式排空再次打满网络/播放器。
  std::chrono::steady_clock::time_point enqueue_time{};
};

static constexpr uint32_t kEncodeQueueSize = 20;

class PushHandlerIm {
  friend class PushHandler;

 public:
  explicit PushHandlerIm(DataSink *module, SinkHandler *handler)
      : module_(module), stream_id_(handler->GetStreamId()) {}
  virtual ~PushHandlerIm() { Close(); }

  static std::optional<int> GetIntParam(const ModuleParamSet& m, const std::string& key) {
    auto it = m.find(key);
    return it != m.end() ? std::optional<int>(std::stoi(it->second)) : std::nullopt;
  }

  static std::optional<std::string> GetStrParam(const ModuleParamSet& m, const std::string& key) {
    auto it = m.find(key);
    return it != m.end() ? std::optional<std::string>(it->second) : std::nullopt;
  }

  bool Open();
  void Stop();
  void Close();
  bool IsRunning() const { return running_.load(); }
  int Process(const std::shared_ptr<FrameInfo> data);

 protected:
  bool InitStream();
  const AVCodec* FindEncoder();
  bool ReinitStream(int device_id);
  virtual bool InitDeviceCtx() { return true; }
  bool InitSwsFrame();
  virtual void CleanDeviceCtx() {}
  virtual bool SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) = 0;
  bool SendFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts);
  void EnsureSwsContext(AVPixelFormat src_pix_fmt, int src_width, int src_height);
  bool SendFrameFb(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts);
  bool EncodeFrame(AVFrame* frame);
  void ClearStream();
  // void FlushEncoder();
  bool FlushEncoder();
  bool DrainPackets();
  bool TryReconnect();

  void EncodeWorkerLoop();
  int64_t ComputePts();
  bool ControlFps();

  DataSink *module_ = nullptr;
  std::string stream_id_;
  std::atomic<bool> running_{false};
  ModuleParamSet param_set_;

  std::string output_url_;
  int device_id_ = -1;
  int fps_ = 20;
  int width_ = 640;
  int height_ = 480;
  int bitrate_kbps_ = 1000;
  std::string codec_name_;

  std::optional<std::string> output_preset_;
  std::optional<std::string> output_tune_;
  std::optional<std::string> output_profile_;
  std::optional<int>         output_gop_;
  int output_timeout_ms_        = 10000;
  int output_tcp_nodelay_       = 1;
  int output_send_buffer_size_  = 262144;

  StreamContext ctx_;
  std::recursive_mutex stream_mtx_;
  int64_t last_pts_ = -1;

  AVPixelFormat src_pix_fmt_ = AV_PIX_FMT_RGB24;
  int sws_src_width_  = 0;
  int sws_src_height_ = 0;
  std::atomic<bool> hw_ctx_initialized_{false};

  bool stream_initialized_ = false;

  std::chrono::steady_clock::time_point push_start_time_;
  std::chrono::steady_clock::time_point last_push_time_;
  bool first_frame_ = true;

  // Token bucket for fps control, allows short bursts while keeping long-term average <= fps_
  static constexpr double kTokenBucketBurstSize = 8.0;
  double token_bucket_tokens_ = 0.0;
  std::chrono::steady_clock::time_point token_bucket_last_update_;

  std::unique_ptr<MarkRender> render_;
  MarkConfig mark_config_;
  bool mark_render_ = false;

  static constexpr int kFpsStatInterval = 100;
  std::chrono::steady_clock::time_point fps_stat_start_time_;
  uint64_t fps_stat_frame_count_ = 0;

  ThreadSafeQueue<EncoderTask> encode_queue_{kEncodeQueueSize};
  std::thread encode_thread_;

  static constexpr int kMaxReconnectAttempts = 3;
  static constexpr int64_t kReconnectIntervalMs = 1000;
  int reconnect_attempts_ = 0;
  std::chrono::steady_clock::time_point last_reconnect_time_;
  bool last_write_network_error_ = false;
};

class PushHandlerImCPU : public PushHandlerIm {
 public:
  using PushHandlerIm::PushHandlerIm;

 protected:
  bool SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) override;
};

}  // namespace cnstream

#ifdef VSTREAM_USE_CUDA
#include "cuda/data_handler_push_cuda.hpp"

#elif defined(VSTREAM_USE_ROCKCHIP)
#include "rockchip/data_handler_push_rk.hpp"

#endif

#endif