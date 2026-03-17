#ifndef BATCHING_STAGE_HPP_
#define BATCHING_STAGE_HPP_

#include <memory>

#include "infer_task.hpp"
#include "infer_resource.hpp"


class IOBatchingStage {
 public:
  IOBatchingStage(uint32_t batchsize, std::shared_ptr<IOResource> output_res)
      : batchsize_(batchsize), output_res_(output_res) {}
  virtual ~IOBatchingStage() {}
  std::shared_ptr<InferTask> Batching(std::shared_ptr<FrameInfo> finfo);
  void ProcessOneFrame(std::shared_ptr<FrameInfo> finfo, uint32_t bidx, IOResValue& value);
  void Reset() { batch_idx_ = 0; }

 private:
  uint32_t batchsize_;
  uint32_t batch_idx_ = 0;
  std::shared_ptr<IOResource> output_res_;
};


#endif  // BATCHING_STAGE_HPP_