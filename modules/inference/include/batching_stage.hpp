#ifndef BATCHING_STAGE_HPP_
#define BATCHING_STAGE_HPP_

#include <memory>

#include "infer_resource.hpp"


namespace cnstream {

class FrameInfo;
class InferTask;
struct IOResValue;
class IOResource;
class NetInputResource;
class CpuInputResource;
class Preproc;  // ObjPreproc see obj_batching_stage


class BatchingStage {
 public:
  BatchingStage(ModelLoader* model, uint32_t batchsize) : model_(model), batchsize_(batchsize) {}
  virtual ~BatchingStage() {}
  virtual std::shared_ptr<InferTask> Batching(std::shared_ptr<FrameInfo> finfo) = 0;
  virtual void Reset() {}

 protected:
  ModelLoader* model_;
  uint32_t batchsize_ = 0;
};  // class BatchingStage


class IOBatchingStage : public BatchingStage {
 public:
  IOBatchingStage(ModelLoader* model, uint32_t batchsize, std::shared_ptr<IOResource> output_res)
      : BatchingStage(model, batchsize), output_res_(output_res) {}
  virtual ~IOBatchingStage() {}
  std::shared_ptr<InferTask> Batching(std::shared_ptr<FrameInfo> finfo) override;
  void Reset() override { batch_idx_ = 0; }

 protected:
  virtual void ProcessOneFrame(std::shared_ptr<FrameInfo> finfo, uint32_t batch_idx, const IOResValue& value) = 0;

 private:
  using BatchingStage::batchsize_;
  uint32_t batch_idx_ = 0;
  std::shared_ptr<IOResource> output_res_ = NULL;
};  // class IOBatchingStage


class CpuPreprocessingBatchingStage : public IOBatchingStage {
 public:
  CpuPreprocessingBatchingStage(ModelLoader* model, uint32_t batchsize,
                                std::shared_ptr<Preproc> preprocessor, std::shared_ptr<CpuInputResource> cpu_input_res);
  ~CpuPreprocessingBatchingStage();

 private:
  void ProcessOneFrame(std::shared_ptr<FrameInfo> finfo, uint32_t batch_idx, const IOResValue& value) override;
  std::shared_ptr<Preproc> preprocessor_;
};  // class CpuPreprocessingBatchingStage


}  // namespace cnstream



#endif  // BATCHING_STAGE_HPP_