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


#include <sys/stat.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "infer_engine.hpp"
#include "infer_resource.hpp"
#include "infer_task.hpp"
#include "postproc.hpp"
#include "queuing_server.hpp"

#include "batching_done_stage.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {


std::vector<std::shared_ptr<InferTask>> H2DBatchingDoneStage::BatchingDone(const BatchingDoneInput& finfos) {
  std::vector<InferTaskSptr> tasks;
  InferTaskSptr task;

  QueuingTicket cpu_input_res_ticket = cpu_input_res_->PickUpNewTicket();
  QueuingTicket net_input_res_ticket = net_input_res_->PickUpNewTicket();

  task = std::make_shared<InferTask>([cpu_input_res_ticket, net_input_res_ticket, this, finfos]() -> int {
    QueuingTicket cir_ticket = cpu_input_res_ticket;
    QueuingTicket mir_ticket = net_input_res_ticket;

    // waiting for schedule
    IOResValue cpu_value = this->cpu_input_res_->WaitResourceByTicket(&cir_ticket);
    IOResValue net_value = this->net_input_res_->WaitResourceByTicket(&mir_ticket);

#ifdef UNIT_TEST
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(finfos.size() == batchsize_);

    for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
      LOGI(H2DBatchingDoneStage) << "bidx: " << bidx << "; [" << finfos[bidx].first->stream_id << "], ts: " << finfos[bidx].first->timestamp;
    }
#endif

    for (int i = 0; i < model_->InputNum(); i++) {
      void* src_cpu = cpu_value.ptrs[i].get();
      void* dst_net = net_value.ptrs[i].get();
      auto input_data_type = model_->InputDataType(i);
      size_t data_size = net_value.datas[i].shape.DataCount() * data_type_size(input_data_type);
      
      // cpu shape 与 net shape 应该一致
      // LOGD(CopyFromHost H2D) << " index: " << i << "; cpu shape: " << cpu_value.datas[i].shape << "; net shape:" << net_value.datas[i].shape << std::endl;
      // LOGD(CopyFromHost H2D) << " index: " << i << "; count:" << net_value.datas[i].shape.DataCount() << ", data_size: " << data_size << std::endl;

      memop_->CopyFromHost(dst_net, src_cpu, data_size);
    }
  
    this->cpu_input_res_->DeallingDone();
    this->net_input_res_->DeallingDone();
    return 0;
  });
  tasks.push_back(task);
  return tasks;
} 


InferBatchingDoneStage::InferBatchingDoneStage(ModelLoader* model,
                                               uint32_t batchsize,
                                               int device_id,
                                               std::shared_ptr<NetInputResource> net_input_res,
                                               std::shared_ptr<NetOutputResource> net_output_res)
    : BatchingDoneStage(model, batchsize, device_id),
      net_input_res_(net_input_res),
      net_output_res_(net_output_res) {
}

InferBatchingDoneStage::~InferBatchingDoneStage() {}

std::vector<std::shared_ptr<InferTask>> InferBatchingDoneStage::BatchingDone(const BatchingDoneInput& finfos) {
  std::vector<InferTaskSptr> tasks;
  InferTaskSptr task;

  QueuingTicket net_input_res_ticket = net_input_res_->PickUpNewTicket();
  QueuingTicket net_output_res_ticket = net_output_res_->PickUpNewTicket();
  task = std::make_shared<InferTask>([net_input_res_ticket, net_output_res_ticket, this, finfos]() -> int {
    QueuingTicket nir_ticket = net_input_res_ticket;
    QueuingTicket nor_ticket = net_output_res_ticket;
    IOResValue net_input_value = this->net_input_res_->WaitResourceByTicket(&nir_ticket);
    IOResValue net_output_value = this->net_output_res_->WaitResourceByTicket(&nor_ticket);

#ifdef UNIT_TEST
    // std::this_thread::sleep_for(std::chrono::milliseconds(800));
    assert(finfos.size() == batchsize_);

    for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
      LOGD(InferBatchingDoneStage) << "bidx: " << bidx << "; [" << finfos[bidx].first->stream_id << "], ts: " << finfos[bidx].first->timestamp;
    }
#endif

    if (profiler_) {  // module profiler
      for (auto it : finfos)
        profiler_->RecordProcessStart("RUN MODEL", std::make_pair(it.first->stream_id, it.first->timestamp));
    }
    // debug for net_input
    if (!dump_resized_image_dir_.empty()) {
      // dump_resized_image(net_input_value, dump_resized_image_dir_);
    }
    model_->RunSync(net_input_value.ptrs, net_output_value.ptrs);

    if (profiler_) {
      for (auto it : finfos)
        profiler_->RecordProcessEnd("RUN MODEL", std::make_pair(it.first->stream_id, it.first->timestamp));
    }

    this->net_input_res_->DeallingDone();
    this->net_output_res_->DeallingDone();
    return 0;
  });
  tasks.push_back(task);
  return tasks;
}


std::vector<std::shared_ptr<InferTask>> D2HBatchingDoneStage::BatchingDone(const BatchingDoneInput& finfos) {
  std::vector<InferTaskSptr> tasks;
  InferTaskSptr task;
  QueuingTicket net_output_res_ticket = net_output_res_->PickUpNewTicket();
  QueuingTicket cpu_output_res_ticket = cpu_output_res_->PickUpNewTicket();
  task = std::make_shared<InferTask>([net_output_res_ticket, cpu_output_res_ticket, this, finfos]() -> int {
    QueuingTicket mor_ticket = net_output_res_ticket;
    QueuingTicket cor_ticket = cpu_output_res_ticket;
    IOResValue net_output_value = this->net_output_res_->WaitResourceByTicket(&mor_ticket);
    IOResValue cpu_output_value = this->cpu_output_res_->WaitResourceByTicket(&cor_ticket);

#ifdef UNIT_TEST
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(finfos.size() == batchsize_);

    for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
      LOGI(D2HBatchingDoneStage) << "bidx: " << bidx << "; [" << finfos[bidx].first->stream_id << "], ts: " << finfos[bidx].first->timestamp;
    }
#endif

    for (int i = 0; i < model_->OutputNum(); i++) {
      void* src_net = net_output_value.ptrs[i].get();
      void* dst_cpu = cpu_output_value.ptrs[i].get();
      auto output_data_type = model_->OutputDataType(i);
      size_t data_size = net_output_value.datas[i].shape.DataCount() * data_type_size(output_data_type);
      memop_->CopyToHost(dst_cpu, src_net, data_size);
    }

    this->net_output_res_->DeallingDone();
    this->cpu_output_res_->DeallingDone();
    return 0;
  });
  tasks.push_back(task);
  return tasks;
}

/**
 * @brief 根据构造时的 res_ 成员选择后处理函数
 */
std::vector<std::shared_ptr<InferTask>> PostprocessingBatchingDoneStage::BatchingDone(const BatchingDoneInput& finfos) {
  if (cpu_output_res_ != nullptr) {
    return BatchingDone(finfos, cpu_output_res_);
  } else if (net_output_res_ != nullptr) {
    return BatchingDone(finfos, net_output_res_);
  } else {
    LOGE(STAGE) << "PostprocessingBatchingDoneStage: cpu_output and net_output are both null";
    assert(false);
  }
  return {};
}

/**
 * @brief 帧级并行后处理
 */
std::vector<std::shared_ptr<InferTask>> PostprocessingBatchingDoneStage::BatchingDone(
    const BatchingDoneInput& finfos, const std::shared_ptr<CpuOutputResource>& cpu_output_res) {
  std::vector<InferTaskSptr> tasks;
  // task size == batch_size
  for (int bidx = 0; bidx < static_cast<int>(finfos.size()); ++bidx) {
    auto finfo = finfos[bidx];
    QueuingTicket cpu_output_res_ticket;
    if (0 == bidx) {
      // 对于第一个元素
      cpu_output_res_ticket = cpu_output_res->PickUpNewTicket(true);
    } else {
      cpu_output_res_ticket = cpu_output_res->PickUpTicket(true);
    }

    InferTaskSptr task =
        std::make_shared<InferTask>([cpu_output_res_ticket, cpu_output_res, this, finfo, bidx]() -> int {

          QueuingTicket cor_ticket = cpu_output_res_ticket;
          IOResValue cpu_output_value = cpu_output_res->WaitResourceByTicket(&cor_ticket);
          std::vector<float*> net_outputs;

          // net_outputs 长度 == output tensor num
          for (size_t output_idx = 0; output_idx < cpu_output_value.datas.size(); ++output_idx) {
            // bidx 指明了在当前 batch 中的 index
            net_outputs.push_back(reinterpret_cast<float*>(cpu_output_value.datas[output_idx].Offset(bidx)));
          }
          if (!cnstream::IsStreamRemoved(finfo.first->stream_id)) {
            this->postprocessor_->Execute(net_outputs, this->model_, finfo.first);
          }
          cpu_output_res->DeallingDone();
          return 0;
        });  // task
    
#ifdef UNIT_TEST
    task->task_msg = "PostprocessingBatchingDoneStage, bidx: " + std::to_string(bidx);
#endif
    tasks.push_back(task);
  }  // end for bidx
  return tasks;
}


std::vector<std::shared_ptr<InferTask>> PostprocessingBatchingDoneStage::BatchingDone(
    const BatchingDoneInput& finfos, const std::shared_ptr<NetOutputResource>& net_output_res) {
  
  QueuingTicket net_output_res_ticket = net_output_res->PickUpNewTicket(false);

  std::vector<InferTaskSptr> tasks;
  InferTaskSptr task = std::make_shared<InferTask>([net_output_res_ticket, net_output_res, this, finfos]() -> int {
    QueuingTicket mor_ticket = net_output_res_ticket;
    IOResValue net_output_value = net_output_res->WaitResourceByTicket(&mor_ticket);
    std::vector<void*> net_outputs;
    for (size_t output_idx = 0; output_idx < net_output_value.datas.size(); ++output_idx) {
      net_outputs.push_back(net_output_value.datas[output_idx].ptr);
    }

    std::vector<std::shared_ptr<FrameInfo>> batched_finfos;
    for (const auto& it : finfos) batched_finfos.push_back(it.first);

    this->postprocessor_->Execute(net_outputs, this->model_, batched_finfos);
    net_output_res->DeallingDone();
    return 0;
  });
  tasks.push_back(task);
  return tasks;
}


std::vector<std::shared_ptr<InferTask>> ObjPostprocessingBatchingDoneStage::ObjBatchingDone(
    const BatchingDoneInput& finfos, const std::vector<std::shared_ptr<InferObject>>& objs) {
  if (cpu_output_res_ != nullptr) {
    return ObjBatchingDone(finfos, objs, cpu_output_res_);
  } else if (net_output_res_ != nullptr) {
    return ObjBatchingDone(finfos, objs, net_output_res_);
  } else {
    LOGE(STAGE) << "ObjPostprocessingBatchingDoneStage: cpu_output and net_output are both null";
    assert(false);
  }
  return {};
}

std::vector<std::shared_ptr<InferTask>> ObjPostprocessingBatchingDoneStage::ObjBatchingDone(
    const BatchingDoneInput& finfos, const std::vector<std::shared_ptr<InferObject>>& objs,
    const std::shared_ptr<CpuOutputResource>& cpu_output_res) {
  std::vector<InferTaskSptr> tasks;
  for (int bidx = 0; bidx < static_cast<int>(finfos.size()); ++bidx) {
    auto finfo = finfos[bidx];
    auto obj = objs[bidx];
    QueuingTicket cpu_output_res_ticket;
    if (0 == bidx) {
      cpu_output_res_ticket = cpu_output_res->PickUpNewTicket(true);
    } else {
      cpu_output_res_ticket = cpu_output_res->PickUpTicket(true);
    }
    InferTaskSptr task =
        std::make_shared<InferTask>([cpu_output_res_ticket, cpu_output_res, this, finfo, obj, bidx]() -> int {
          QueuingTicket cor_ticket = cpu_output_res_ticket;
          IOResValue cpu_output_value = cpu_output_res->WaitResourceByTicket(&cor_ticket);
          std::vector<float*> net_outputs;
          for (size_t output_idx = 0; output_idx < cpu_output_value.datas.size(); ++output_idx) {
            net_outputs.push_back(reinterpret_cast<float*>(cpu_output_value.datas[output_idx].Offset(bidx)));
          }
          if (!cnstream::IsStreamRemoved(finfo.first->stream_id)) {
            this->postprocessor_->Execute(net_outputs, this->model_, finfo.first, obj);
          }
          cpu_output_res->DeallingDone();
          return 0;
        });
    tasks.push_back(task);
  }
  return tasks;
}

std::vector<std::shared_ptr<InferTask>> ObjPostprocessingBatchingDoneStage::ObjBatchingDone(
    const BatchingDoneInput& finfos, const std::vector<std::shared_ptr<InferObject>>& objs,
    const std::shared_ptr<NetOutputResource>& net_output_res) {
  std::vector<InferTaskSptr> tasks;
  QueuingTicket net_output_res_ticket = net_output_res->PickUpNewTicket(false);
  InferTaskSptr task =
      std::make_shared<InferTask>([net_output_res_ticket, net_output_res, this, finfos, objs]() -> int {
        QueuingTicket mor_ticket = net_output_res_ticket;
        IOResValue net_output_value = net_output_res->WaitResourceByTicket(&mor_ticket);
        std::vector<void*> net_outputs;
        for (size_t output_idx = 0; output_idx < net_output_value.datas.size(); ++output_idx) {
          net_outputs.push_back(net_output_value.datas[output_idx].ptr);
        }

        std::vector<std::pair<std::shared_ptr<FrameInfo>, std::shared_ptr<InferObject>>> batched_objs;
        for (int bidx = 0; bidx < static_cast<int>(finfos.size()); ++bidx) {
          auto finfo = finfos[bidx];
          auto obj = objs[bidx];
          // finfo.first: std::shared_ptr<FrameInfo>
          // obj: std::shared_ptr<InferObject>
          batched_objs.push_back(std::make_pair(std::move(finfo.first), std::move(obj)));
        }

        this->postprocessor_->Execute(net_outputs, this->model_, batched_objs);
        net_output_res->DeallingDone();
        return 0;
      });
  tasks.push_back(task);

  return tasks;
}


}  // namespace cnstream