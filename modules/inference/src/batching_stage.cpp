
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
  // 这样同一批次内的 output_res_ 返回对应相同 promise 的 shared_future，同一 batch 内的可以进行并发
  QueuingTicket ticket = output_res_->PickUpTicket(reserve_ticket);
  auto bidx = batch_idx_;

  std::shared_ptr<InferTask> task = std::make_shared<InferTask>([this, ticket, finfo, bidx]() -> int {
    QueuingTicket t = ticket;
    IOResValue value = this->output_res_->WaitResourceByTicket(&t);

#ifdef UNIT_TEST
    // std::this_thread::sleep_for(std::chrono::milliseconds(800));

    std::cout << "IOBatchingStage, bidx: " << bidx
        << "; [" << finfo.first->stream_id << ", " << finfo.first->timestamp << "] " << std::endl;
  
#endif

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


/**
 * @brief 单张图片的预处理
 * 
 * @param finfo 图片信息
 * @param batch_idx 批次索引 [0, batchsize_ - 1]
 * @param value cpu_input_res 对应的 IOResValue
 */
void CpuPreprocessingBatchingStage::ProcessOneFrame(std::shared_ptr<FrameInfo> finfo, uint32_t batch_idx,
                                                    const IOResValue& value) {
  std::vector<float*> cpu_outputs;  // cpu-preprocess output still in cpu

  // cpu_outputs.size == input tensor num
  // cpu_outputs 每个元素定位到 tensor 对应的 batch_idx 位置
  for (auto it : value.datas) {
    cpu_outputs.push_back(reinterpret_cast<float*>(it.Offset(batch_idx)));
  }
  preprocessor_->Execute(cpu_outputs, model_, finfo);
}