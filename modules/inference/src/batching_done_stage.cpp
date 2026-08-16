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

  // cpu_input: 链式票据，延续本批预处理共享票据的 run（读写同一份 cpu buffer）
  // net_input: 新 run，从空闲池取本批专属的 device buffer
  QueuingTicket cpu_input_res_ticket = cpu_input_res_->PickUpNewTicket(false, true);
  QueuingTicket net_input_res_ticket = net_input_res_->PickUpNewTicket();

  task = std::make_shared<InferTask>([cpu_input_res_ticket, net_input_res_ticket, this, finfos]() -> int {
    QueuingTicket cir_ticket = cpu_input_res_ticket;
    QueuingTicket mir_ticket = net_input_res_ticket;

    // waiting for schedule
    IOResValue cpu_value = this->cpu_input_res_->WaitResourceByTicket(&cir_ticket);
    IOResValue net_value = this->net_input_res_->WaitResourceByTicket(&mir_ticket);

#ifdef VSTREAM_UNIT_TEST
    assert(finfos.size() == batchsize_);
#endif

    for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
      LOGU(H2D) << "bidx: " << bidx << "; [" << finfos[bidx].first->stream_id << "], ts: " << finfos[bidx].first->timestamp;
    }

    // slot 执行流（异步模式）：后续 infer 在同一流上排队，天然保证读序；
    // 未启用异步时回退模型默认流
    void* infer_stream = net_value.stream ? net_value.stream : model_->GetStream();
    for (int i = 0; i < model_->InputNum(); i++) {
      void* src_cpu = cpu_value.ptrs[i].get();
      void* dst_net = net_value.ptrs[i].get();
      auto input_data_type = model_->InputDataType(i);
      size_t data_size = net_value.datas[i].shape.DataCount() * data_type_size(input_data_type);

      // cpu shape 与 net shape 应该一致
      // LOGU(H2D) << " index: " << i << "; cpu shape: " << cpu_value.datas[i].shape << "; net shape:" << net_value.datas[i].shape << std::endl;
      // LOGU(H2D) << " index: " << i << "; count:" << net_value.datas[i].shape.DataCount() << ", data_size: " << data_size << std::endl;

      memop_->CopyFromHostAsync(dst_net, src_cpu, data_size, infer_stream);
    }

    // cpu buffer 仍在被 GPU 拷贝读取，必须等拷贝完成才能释放 cpu_input 票据，
    // 否则下一批次预处理复用同一 slot 的 cpu buffer 会产生数据竞争。
    // net_input 票据随之在拷贝完成后才转交 Infer（链式）：推理提交虽晚一个拷贝时长，
    // 但其 RunAsync 与 H2D 排在同一 slot 流上，GPU 执行顺序不变；
    // 批间重叠由其他 slot 承担（本批拷贝期间，下一批的 H2D 已在其他 slot 流上执行）
    memop_->SyncStream(infer_stream);
    this->cpu_input_res_->DeallingDone(cir_ticket);
    this->net_input_res_->DeallingDone(mir_ticket);
    return 0;
  });
  tasks.push_back(task);
  return tasks;
}


InferBatchingDoneStage::InferBatchingDoneStage(ModelLoader* model,
                                               uint32_t batchsize,
                                               std::shared_ptr<IOResource> input_res,
                                               std::shared_ptr<IOResource> output_res)
    : BatchingDoneStage(model, batchsize, model ? model->GetDeviceId() : -1),
      input_res_(input_res),
      output_res_(output_res) {
}

InferBatchingDoneStage::~InferBatchingDoneStage() {}

std::vector<std::shared_ptr<InferTask>> InferBatchingDoneStage::BatchingDone(const BatchingDoneInput& finfos) {
  std::vector<InferTaskSptr> tasks;
  InferTaskSptr task;

  // input: 链式票据，延续 H2D（CUDA）或预处理共享票据（CPU）的 run，同一 slot buffer
  // output: 新 run，本批专属输出 buffer
  QueuingTicket input_res_ticket = input_res_->PickUpNewTicket(false, true);
  QueuingTicket output_res_ticket = output_res_->PickUpNewTicket();
  task = std::make_shared<InferTask>([input_res_ticket, output_res_ticket, this, finfos]() -> int {
    QueuingTicket ir_ticket = input_res_ticket;
    QueuingTicket or_ticket = output_res_ticket;
    IOResValue input_value = this->input_res_->WaitResourceByTicket(&ir_ticket);
    IOResValue output_value = this->output_res_->WaitResourceByTicket(&or_ticket);

#ifdef VSTREAM_UNIT_TEST
    assert(finfos.size() == batchsize_);
#endif

    for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
      LOGU(INFER) << "bidx: " << bidx << "; [" << finfos[bidx].first->stream_id << "], ts: " << finfos[bidx].first->timestamp;
    }

    if (profiler_) {
      for (const auto& finfo : finfos) {
        profiler_->RecordProcessStart(kMODEL_PROFILER_NAME, std::make_pair(finfo.first->stream_id, finfo.first->timestamp));
      }
    }
    if (!dump_resized_image_dir_.empty()) {
      // dump_resized_image(net_input_value, dump_resized_image_dir_);
    }

    // 推理提交到 input slot 执行流：与 H2D 同流串行，无需额外同步即保证读序；
    // 等待本批完成期间，其他 slot 上的批次可继续 H2D/推理（批间重叠）。
    // 未启用异步的平台（RKNN/CPU）RunAsync 回退 RunSync 并返回 nullptr
    void* slot_stream = input_value.stream ? input_value.stream : model_->GetStream();
    void* event = model_->RunAsync(input_value.ptrs, output_value.ptrs, slot_stream);
    if (event) {
      // 推理完成才释放票据：输入 buffer 不再被读取、输出 buffer 已写就绪
      model_->SyncEvent(event);
    }

    if (profiler_) {
      for (const auto& finfo : finfos) {
        profiler_->RecordProcessEnd(kMODEL_PROFILER_NAME, std::make_pair(finfo.first->stream_id, finfo.first->timestamp));
      }
    }

    this->input_res_->DeallingDone(ir_ticket);
    this->output_res_->DeallingDone(or_ticket);
    return 0;
  });
  tasks.push_back(task);
  return tasks;
}


std::vector<std::shared_ptr<InferTask>> D2HBatchingDoneStage::BatchingDone(const BatchingDoneInput& finfos) {
  std::vector<InferTaskSptr> tasks;
  InferTaskSptr task;
  // net_output: 链式票据，延续本批 Infer 的 run（同一 slot 输出 buffer）
  // cpu_output: 新 run，本批专属 cpu 输出 buffer
  QueuingTicket net_output_res_ticket = net_output_res_->PickUpNewTicket(false, true);
  QueuingTicket cpu_output_res_ticket = cpu_output_res_->PickUpNewTicket();
  task = std::make_shared<InferTask>([net_output_res_ticket, cpu_output_res_ticket, this, finfos]() -> int {
    QueuingTicket mor_ticket = net_output_res_ticket;
    QueuingTicket cor_ticket = cpu_output_res_ticket;
    IOResValue net_output_value = this->net_output_res_->WaitResourceByTicket(&mor_ticket);
    IOResValue cpu_output_value = this->cpu_output_res_->WaitResourceByTicket(&cor_ticket);

#ifdef VSTREAM_UNIT_TEST
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(finfos.size() == batchsize_);
#endif

    for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
      LOGU(D2H) << "bidx: " << bidx << "; [" << finfos[bidx].first->stream_id << "], ts: " << finfos[bidx].first->timestamp;
    }

    // Infer 阶段释放票据前已 SyncEvent，输出数据就绪；
    // 用 net_output slot 流拷贝，可与其他 slot 上的推理并行
    void* d2h_stream = net_output_value.stream ? net_output_value.stream : model_->GetStream();
    for (int i = 0; i < model_->OutputNum(); i++) {
      void* src_net = net_output_value.ptrs[i].get();
      void* dst_cpu = cpu_output_value.ptrs[i].get();
      auto output_data_type = model_->OutputDataType(i);
      size_t data_size = net_output_value.datas[i].shape.DataCount() * data_type_size(output_data_type);
      memop_->CopyToHostAsync(dst_cpu, src_net, data_size, d2h_stream);
    }
    memop_->SyncStream(d2h_stream);

    this->net_output_res_->DeallingDone(mor_ticket);
    this->cpu_output_res_->DeallingDone(cor_ticket);
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
      // 链式票据：延续本批 D2H 的 run，读 D2H 写入的同一份 cpu buffer；
      // 首帧取新共享票据，同批其余帧共享之
      cpu_output_res_ticket = cpu_output_res->PickUpNewTicket(true, true);
    } else {
      cpu_output_res_ticket = cpu_output_res->PickUpTicket(true);
    }

    InferTaskSptr task =
        std::make_shared<InferTask>([cpu_output_res_ticket, cpu_output_res, this, finfo, bidx]() -> int {

          QueuingTicket cor_ticket = cpu_output_res_ticket;
          IOResValue cpu_output_value = cpu_output_res->WaitResourceByTicket(&cor_ticket);
          std::vector<float*> cpu_outputs;

          // cpu_outputs 长度 == output tensor num
          for (size_t output_idx = 0; output_idx < cpu_output_value.datas.size(); ++output_idx) {
            // bidx 指明了在当前 batch 中的 index
            cpu_outputs.push_back(reinterpret_cast<float*>(cpu_output_value.datas[output_idx].Offset(bidx)));
          }
          if (!cnstream::IsStreamRemoved(finfo.first->stream_id)) {
            this->postprocessor_->Execute(cpu_outputs, this->model_, finfo.first);
          }
          cpu_output_res->DeallingDone(cor_ticket);
          return 0;
        });  // task

    tasks.push_back(task);
  }  // end for bidx
  return tasks;
}


std::vector<std::shared_ptr<InferTask>> PostprocessingBatchingDoneStage::BatchingDone(
    const BatchingDoneInput& finfos, const std::shared_ptr<NetOutputResource>& net_output_res) {

  // 链式票据：延续本批 Infer 的 run，直接读 device 输出 buffer
  QueuingTicket net_output_res_ticket = net_output_res->PickUpNewTicket(false, true);

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
    net_output_res->DeallingDone(mor_ticket);
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
      // 链式票据：延续本批 D2H 的 run，读 D2H 写入的同一份 cpu buffer
      cpu_output_res_ticket = cpu_output_res->PickUpNewTicket(true, true);
    } else {
      cpu_output_res_ticket = cpu_output_res->PickUpTicket(true);
    }
    InferTaskSptr task =
        std::make_shared<InferTask>([cpu_output_res_ticket, cpu_output_res, this, finfo, obj, bidx]() -> int {
          QueuingTicket cor_ticket = cpu_output_res_ticket;
          IOResValue cpu_output_value = cpu_output_res->WaitResourceByTicket(&cor_ticket);
          std::vector<float*> cpu_outputs;
          for (size_t output_idx = 0; output_idx < cpu_output_value.datas.size(); ++output_idx) {
            cpu_outputs.push_back(reinterpret_cast<float*>(cpu_output_value.datas[output_idx].Offset(bidx)));
          }
          if (!cnstream::IsStreamRemoved(finfo.first->stream_id)) {
            this->postprocessor_->Execute(cpu_outputs, this->model_, finfo.first, obj);
          }
          cpu_output_res->DeallingDone(cor_ticket);
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
  // 链式票据：延续本批 Infer 的 run，直接读 device 输出 buffer
  QueuingTicket net_output_res_ticket = net_output_res->PickUpNewTicket(false, true);
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
        net_output_res->DeallingDone(mor_ticket);
        return 0;
      });
  tasks.push_back(task);

  return tasks;
}


}  // namespace cnstream