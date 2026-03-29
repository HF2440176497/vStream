
#include <memory>
#include <iostream>
#include <chrono>
#include <thread>

#include "infer_task.hpp"
#include "batching_stage.hpp"


std::shared_ptr<InferTask> IOBatchingStage::Batching(std::shared_ptr<FrameInfo> finfo) {
  bool reserve_ticket = false;
  if (batch_idx_ + 1 == batchsize_) {
    // ready to next batch, do not reserve resource ticket.
    reserve_ticket = false;
  } else {
    // in one batch, reserve resource ticket to parallel.
    reserve_ticket = true;
  }

  // reserve_ticket = false 对应当前是 batch 的最后一个 frame; 
  // 下一个会 PickUpTicket 会创建新的 ticket, 否则就会一直保留 output_res_ 的 ticket, 
  // 这样同一批次内的 output_res_ 返回对应相同 promise 的 shared_future
  // 同一 batch 内的可以进行并发
  QueuingTicket ticket = output_res_->PickUpTicket(reserve_ticket);
  auto bidx = batch_idx_;

  std::shared_ptr<InferTask> task = std::make_shared<InferTask>([this, ticket, finfo, bidx]() -> int {
    QueuingTicket t = ticket;
    IOResValue value = this->output_res_->WaitResourceByTicket(&t);
    this->ProcessOneFrame(finfo, bidx, value);
    this->output_res_->DeallingDone();
    return 0;
  });
  task->task_msg = "IOBatchingStage, bidx: " + std::to_string(bidx);
  batch_idx_ = (batch_idx_ + 1) % batchsize_;
  return task;
}


CpuPreprocessingBatchingStage::CpuPreprocessingBatchingStage(ModelLoader* model,
                                                             uint32_t batchsize, std::shared_ptr<Preproc> preprocessor,
                                                             std::shared_ptr<CpuInputResource> cpu_input_res)
    : IOBatchingStage(model, batchsize, cpu_input_res), preprocessor_(preprocessor) {}

CpuPreprocessingBatchingStage::~CpuPreprocessingBatchingStage() {}

void CpuPreprocessingBatchingStage::ProcessOneFrame(std::shared_ptr<FrameInfo> finfo, uint32_t batch_idx,
                                                    const IOResValue& value) {
  std::vector<float*> net_inputs;

  // 单张图片的处理，net_inputs.size == input tensor num
  for (auto it : value.datas) {
    net_inputs.push_back(reinterpret_cast<float*>(it.Offset(batch_idx)));
  }
  preprocessor_->Execute(net_inputs, model_, finfo);
}