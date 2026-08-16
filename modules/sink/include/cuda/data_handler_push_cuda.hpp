#ifndef MODULES_SINK_HANDLER_PUSH_CUDA_HPP_
#define MODULES_SINK_HANDLER_PUSH_CUDA_HPP_

#include <cuda_runtime.h>

namespace cnstream {

class PushHandlerImCUDA : public PushHandlerIm {
 public:
  using PushHandlerIm::PushHandlerIm;

 protected:
  bool InitDeviceCtx() override;
  void CleanDeviceCtx() override;
  bool SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts) override;

 private:
  bool SendFrameCuda(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts);
  bool SendFrameToCuda(const DataFramePtr& frame, AVPixelFormat src_pix_fmt, int64_t pts);

  cudaStream_t sink_stream_ = nullptr;
};

}  // namespace cnstream

#endif