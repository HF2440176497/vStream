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
static constexpr const char* kDefaultEncoder = "h264_nvenc";
#else
static constexpr AVPixelFormat kEncoderPixFmt = AV_PIX_FMT_YUV420P;
static constexpr AVPixelFormat kSwsPixFmt     = AV_PIX_FMT_YUV420P;
static constexpr const char* kDefaultEncoder = nullptr;
#endif

namespace cnstream {

struct StreamContext {
  AVFormatContext* fmt_ctx   = nullptr;
  AVCodecContext*  codec_ctx = nullptr;
  AVStream*        stream    = nullptr;
  SwsContext*      sws_ctx   = nullptr;
  AVFrame*         sws_frame = nullptr;
  uint64_t         frame_idx = 0;
  bool             header_written = false;
#ifdef VSTREAM_USE_CUDA
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
};

static constexpr uint32_t kEncodeQueueSize = 20;

class PushHandlerImpl {
  friend class PushHandler;

 public:
  explicit PushHandlerImpl(DataSink *module, SinkHandler *handler)
      : module_(module), stream_id_(handler->GetStreamId()) {}
  virtual ~PushHandlerImpl() { Close(); }

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

  StreamContext ctx_;
  std::recursive_mutex stream_mtx_;
  int64_t last_pts_ = -1;
  int64_t pts_count_ = 0;

  AVPixelFormat src_pix_fmt_ = AV_PIX_FMT_RGB24;
  int sws_src_width_  = 0;
  int sws_src_height_ = 0;
  std::atomic<bool> hw_ctx_initialized_{false};

  bool stream_initialized_ = false;

  std::chrono::steady_clock::time_point push_start_time_;
  std::chrono::steady_clock::time_point last_push_time_;
  std::chrono::steady_clock::time_point next_frame_time_;
  bool first_frame_ = true;

  std::unique_ptr<MarkRender> render_;
  MarkConfig mark_config_;
  bool mark_render_ = false;

  static constexpr int kFpsStatInterval = 100;
  std::chrono::steady_clock::time_point fps_stat_start_time_;
  uint64_t fps_stat_frame_count_ = 0;

  ThreadSafeQueue<EncoderTask> encode_queue_{kEncodeQueueSize};
  std::thread encode_thread_;
};

class PushHandlerImplCPU : public PushHandlerImpl {
 public:
  using PushHandlerImpl::PushHandlerImpl;

 protected:
  bool SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) override;
};

}  // namespace cnstream

#ifdef VSTREAM_USE_CUDA
#include "cuda/data_handler_push_cuda.hpp"

#endif

#endif