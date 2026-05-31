#ifndef MODULES_SINK_HANDLER_PUSH_CUDA_HPP_
#define MODULES_SINK_HANDLER_PUSH_CUDA_HPP_

#include <cuda_runtime.h>

namespace cnstream {

class PushHandlerImplCUDA : public PushHandlerImpl {
 public:
  using PushHandlerImpl::PushHandlerImpl;

 protected:
  bool InitDeviceCtx() override;
  void CleanDeviceCtx() override;
  bool SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt) override;

 private:
  bool SendFrameCuda(const DataFramePtr& frame, AVPixelFormat src_pix_fmt);
  bool SendFrameToCuda(const DataFramePtr& frame, AVPixelFormat src_pix_fmt);

  cudaStream_t sink_stream_ = nullptr;
};

}  // namespace cnstream

#endif