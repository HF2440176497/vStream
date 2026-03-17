
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
  QueuingTicket ticket = output_res_->PickUpTicket(reserve_ticket);
  auto bidx = batch_idx_;

  std::shared_ptr<InferTask> task = std::make_shared<InferTask>([this, ticket, finfo, bidx]() -> int {
    QueuingTicket t = ticket;
    IOResValue value = this->output_res_->WaitResourceByTicket(&t);
    this->ProcessOneFrame(finfo, bidx, value);
    this->output_res_->DeallingDone();
    return 0;
  });
  batch_idx_ = (batch_idx_ + 1) % batchsize_;
  return task;
}





void IOBatchingStage::ProcessOneFrame(std::shared_ptr<FrameInfo> finfo, uint32_t bidx, IOResValue& value) {
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::cout << "IOBatchingStage, bidx: " << bidx 
      << "; ["<< finfo->batch_index << ", " << finfo->item_index << "] " << std::endl;
}
