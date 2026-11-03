/**
 * @brief 提供生产者-消费者的接口，读取视频流（支持 NVIDIA CUDA 硬件解码）
 */

#ifndef MODULES_SOURCE_HANDLER_VIDEO_HPP_
#define MODULES_SOURCE_HANDLER_VIDEO_HPP_

#include <atomic>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cnstream_logging.hpp"
#include "data_handler_util.hpp"
#include "data_source.hpp"
#include "data_source_param.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
}

namespace cnstream {

class PullHandlerIm : public SourceRender {
  friend class PullHandler;
  
 public:
  struct MatBufRef : public IDecBufRef {
    explicit MatBufRef(void* data) : data_(data) {}
    ~MatBufRef() override {
      delete[] static_cast<uint8_t*>(data_);
    }
    void* data_;
  };

  struct MatBufRefNV12 : public IDecBufRef {
    MatBufRefNV12(void* y_data, void* uv_data) : y_data_(y_data), uv_data_(uv_data) {}
    ~MatBufRefNV12() override {
      delete[] static_cast<uint8_t*>(y_data_);
      delete[] static_cast<uint8_t*>(uv_data_);
    }
    void* y_data_;
    void* uv_data_;
  };

  explicit PullHandlerIm(DataSource *module, SourceHandler *handler)
      : SourceRender(handler), module_(module), stream_id_(handler->GetStreamId()) {
    pkt_ = av_packet_alloc();
  }
  virtual ~PullHandlerIm() = default;

  bool Open();
  void Close();
  void Stop();
  void Loop();

 public:
  void OnEndFrame();
  std::shared_ptr<FrameInfo> OnDecodeFrame(DecodeFrame* frame);

 protected:
  virtual int codec_init() = 0;
  virtual void clean_up();

  virtual int decode_write() = 0;
  virtual bool SupportHWDevice() { return true; }
  virtual void ConfigureOutputType() {}

  int input_format_init();

  // 释放"一次连接"相关的资源（demuxer + 解码器 + 硬件设备上下文），
  // 供断流重连与 final clean_up 共用。可重入。
  void teardown_input();

  // 重连退避等待（可被 Stop 打断）。返回 false 表示已停止或连续失败超上限。
  bool WaitBeforeReconnect(int attempt, const char *stage);

  // 读包-解码-投递主循环。返回值决定 Loop() 下一步动作。
  enum class ReadResult {
    kStopped,  // 外部 Stop 或运行标志已清除
    kEof,      // 输入正常结束（文件读完 / 对端正常关闭）
    kBroken,   // 流中断（网络错误、对端断开、解码器不可恢复错误），可尝试重连
    kFatal     // 参数/下游致命错误，不再重试
  };
  ReadResult read_loop(FrController *controller, bool *got_data);

  // 连续单帧处理失败计数阈值：超过即判定该流与本条解码通路不兼容，终止拉流
  static constexpr uint32_t kMaxFrameErrorCnt = 64;
  uint32_t frame_error_cnt_ = 0;

 public:
  bool IsRunning() const { return running_; }

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 protected:
#endif
  std::atomic<bool> running_{false};
  std::atomic<bool> closed_{false};
  std::atomic<bool> cleaned_up_{false};
  std::thread        thread_;

  int         interval_    = 0;
  int         device_id_   = -1;
  std::string stream_url_;
  int         frame_rate_  = 10;

  DataSource *module_;
  std::string stream_id_;

  // 网络栈初始化状态：avformat_network_init/deinit 必须严格一一配对，
  // 重连会多次进入 input_format_init，故用该标志保证一个 handler 只 init 一次。
  bool network_ready_ = false;
  // URL 是否为网络流：决定 av_read_frame 返回 EOF 时是"正常结束"还是"断流需重连"
  bool is_network_url_ = false;

  AVFormatContext    *ifmt_ctx_      = nullptr;
  AVDictionary       *ifmt_opts_     = nullptr;
  int                 video_index_   = -1;

  AVBufferRef        *hw_device_ctx_ = nullptr;
  enum AVHWDeviceType device_type_   = AV_HWDEVICE_TYPE_NONE;
  
  AVCodec           *codec_        = nullptr;
  AVCodecContext    *codec_ctx_    = nullptr;
  AVCodecParameters *codecpar_     = nullptr;
  AVPacket          *pkt_ = nullptr;

  OutputType output_type_ = OutputType::OUTPUT_CPU;
};

class PullHandlerImCPU : public PullHandlerIm {
 public:
  using PullHandlerIm::PullHandlerIm;
  ~PullHandlerImCPU() override;

 protected:
  int codec_init() override;
  int decode_write() override;
  void clean_up() override;

 private:
  std::shared_ptr<FrameInfo> ProcessFrame(AVFrame *p_frame, int &ret);
  // 取出解码器当前可给出的所有帧并投递；复用同一个 AVFrame，单帧失败只跳过。
  // 返回 <0 表示解码器/下游出现不可恢复错误。
  int ReceiveAndDeliver();
  // 软解输出的像素格式与硬件解码器不同（软解恒为 YUV420P/YUVJ420P/YUV422P/YUV444P/10bit 等），
  // 统一用 sws 转成 NV12，保证下游只有 NV12/NV21 两条通路。
  // 返回值只是"借用"：指向成员 nv12_frame_（或调用方的 p_frame），
  // 调用方不得保存该指针，下一次 ToNV12 会覆盖其内容。失败返回 nullptr。
  const AVFrame *ToNV12(AVFrame *p_frame);

  // sws_ctx_ / nv12_frame_ 的唯一释放点，可重入（释放后置空）
  void ReleaseConvertResources();

  SwsContext   *sws_ctx_     = nullptr;
  int           sws_src_w_   = 0;
  int           sws_src_h_   = 0;
  AVPixelFormat sws_src_fmt_ = AV_PIX_FMT_NONE;
  AVFrame      *nv12_frame_  = nullptr;
};

}  // namespace cnstream

#ifdef VSTREAM_USE_CUDA
#include "cuda/data_handler_pull_cuda.hpp"
#elif defined(VSTREAM_USE_ROCKCHIP)
#include "rockchip/data_handler_pull_rk.hpp"
#endif

#endif