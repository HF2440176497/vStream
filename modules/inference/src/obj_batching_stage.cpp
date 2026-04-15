/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include "obj_batching_stage.hpp"
#include <memory>
#include <vector>
#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "infer_resource.hpp"
#include "infer_task.hpp"
#include "preproc.hpp"

namespace cnstream {

std::shared_ptr<InferTask> IOObjBatchingStage::Batching(std::shared_ptr<FrameInfo> finfo,
                                                        std::shared_ptr<InferObject> obj) {
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
  std::shared_ptr<InferTask> task = std::make_shared<InferTask>([this, ticket, finfo, obj, bidx]() -> int {
    QueuingTicket t = ticket;
    IOResValue value = this->output_res_->WaitResourceByTicket(&t);
    this->ProcessOneObject(finfo, obj, bidx, value);
    this->output_res_->DeallingDone();
    return 0;
  });
  task->task_msg = "infer task.";
  batch_idx_ = (batch_idx_ + 1) % batchsize_;
  return task;
}

CpuPreprocessingObjBatchingStage::CpuPreprocessingObjBatchingStage(ModelLoader* model,
                                                                   uint32_t batchsize,
                                                                   std::shared_ptr<ObjPreproc> preprocessor,
                                                                   std::shared_ptr<CpuInputResource> cpu_input_res)
    : IOObjBatchingStage(model, batchsize, cpu_input_res), preprocessor_(preprocessor) {}

CpuPreprocessingObjBatchingStage::~CpuPreprocessingObjBatchingStage() {}

void CpuPreprocessingObjBatchingStage::ProcessOneObject(std::shared_ptr<FrameInfo> finfo,
                                                        std::shared_ptr<InferObject> obj, uint32_t batch_idx,
                                                        const IOResValue& value) {
  // 前处理的输出
  std::vector<float*> cpu_outputs;
  for (auto it : value.datas) {
    cpu_outputs.push_back(reinterpret_cast<float*>(it.Offset(batch_idx)));
  }
  preprocessor_->Execute(cpu_outputs, model_, finfo, obj);
}

}  // namespace cnstream
