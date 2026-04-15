

#ifndef MODULES_SOURCE_HANDLER_UTIL_HPP_
#define MODULES_SOURCE_HANDLER_UTIL_HPP_

#include <thread>
#include <chrono>

#include "cnstream_source.hpp"  // SourceHandler
#include "cnstream_frame_va.hpp"  // DataFrame

namespace cnstream {

class SourceRender {
 public:
  explicit SourceRender(SourceHandler *handler) : handler_(handler) {}
  virtual ~SourceRender() = default;

  virtual bool CreateInterrupt() { return interrupt_.load(); }

  // invoked by: HandlerImpl::OnDecodeFrame
  std::shared_ptr<FrameInfo> CreateFrameInfo(bool eos = false) {
    std::shared_ptr<FrameInfo> data;
    if (handler_ == nullptr) {
      LOGF(SOURCE) << "CreateFrameInfo: handler_ is nullptr";
      return nullptr;
    }
    while (true) {
      data = handler_->CreateFrameInfo(eos);
      if (data != nullptr) break;
      if (CreateInterrupt()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
    auto frame = std::make_shared<DataFrame>();
    if (!frame) {
      return nullptr;
    }
    auto inferobjs = std::make_shared<InferObjs>();
    if (!inferobjs) {
      return nullptr;
    }
    auto inferdata =  std::make_shared<InferData>();
    if (!inferdata) {
      return nullptr;
    }
    data->collection.Add(kDataFrameTag, frame);
    data->collection.Add(kInferObjsTag, inferobjs);
    data->collection.Add(kInferDataTag, inferdata);
    return data;
  }

  void SendFlowEos() {
    if (eos_sent_) return;
    auto data = CreateFrameInfo(true);
    if (!data) {
      LOGE(SOURCE) << "[" << handler_->GetStreamId() << "]: "
                   << "SendFlowEos: Create DataFrame failed";
      return;
    }
    LOGI(SOURCE) << "[" << handler_->GetStreamId() << "]: " << "Send EOS frame info";
    SendFrameInfo(data);
    eos_sent_ = true;
  }

  /**
   * 不借助 Pipeline::TaskLoop 线程循环，直接向下游的 module 传输
   * 适用于直接发送 EOS 帧的情况
   */
  bool SendFrameInfo(std::shared_ptr<FrameInfo> data) {
    return handler_->SendData(data);
  }

 public:
  static int Process(std::shared_ptr<FrameInfo> frame_info,
                     DecodeFrame *frame, uint64_t frame_id);
 
 protected:
  SourceHandler *handler_;
  ModuleParamSet param_set_;  // from SourceModule param_set_
  bool eos_sent_ = false;
  std::atomic<bool> interrupt_{false};
  uint64_t frame_count_ = 0;
  uint64_t frame_id_ = 0;
};  // class SourceRender


/**********************************************************************
 * @brief FrController is used to control the frequency of sending data.
 ***********************************************************************/
class FrController {
 public:
  FrController() {}
  explicit FrController(uint32_t frame_rate) : frame_rate_(frame_rate) {
    delay_ = 1000.0 / frame_rate_;
  }
  void Start() { start_ = std::chrono::steady_clock::now(); }
  
  void Control() {
    if (0 == frame_rate_) return;  // 不进行限制
    
    end_ = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end_ - start_;
    auto gap = delay_ - diff.count() - time_gap_;
    if (gap > 0) {
      std::chrono::duration<double, std::milli> dura(gap);
      std::this_thread::sleep_for(dura);
      time_gap_ = 0;
    } else {  // 保留这一次的超时时间
      time_gap_ = -gap;
    }
    Start();
  }
  inline uint32_t GetFrameRate() const { return frame_rate_; }
  inline void SetFrameRate(uint32_t frame_rate) { frame_rate_ = frame_rate; }

 private:
  double delay_ = 0;
  uint32_t frame_rate_ = 0;
  double time_gap_ = 0;
  std::chrono::time_point<std::chrono::steady_clock> start_, end_;
};  // class FrController


}  // namespace cnstream

#endif  // _CNSTREAM_SOURCE_HANDLER_UTIL_HPP_