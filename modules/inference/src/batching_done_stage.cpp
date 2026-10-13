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
#include "input_deriver.hpp"
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

    // 异常安全：正常返回或中途抛出都必须归还两张票据，否则队首票据泄漏导致流水线冻结
    DeallingDoneGuard ticket_guard{this->cpu_input_res_.get(), this->net_input_res_.get()};

    // waiting for schedule
    IOResValue cpu_value = this->cpu_input_res_->WaitResourceByTicket(&cir_ticket);
    IOResValue net_value = this->net_input_res_->WaitResourceByTicket(&mir_ticket);

#ifdef VSTREAM_UNIT_TEST
    assert(finfos.size() == batchsize_);
#endif

    for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
      LOGU(H2D) << "bidx: " << bidx << "; [" << finfos[bidx].first->stream_id << "], ts: " << finfos[bidx].first->timestamp;
    }

    // 形状契约随批流动：cpu 侧 -> device 侧
    if (cpu_value.run && net_value.run) {
      *net_value.run = *cpu_value.run;
    }

    for (int i = 0; i < model_->InputNum(); i++) {
      void* src_cpu = cpu_value.ptrs[i].get();
      void* dst_net = net_value.ptrs[i].get();
      auto input_data_type = model_->InputDataType(i);
      // 动态模式只拷贝 actual 大小；无形状契约时回落到缓冲区形状（与历史一致）
      const TensorShape* actual = nullptr;
      if (cpu_value.run != nullptr && i < static_cast<int>(cpu_value.run->inputs.size())) {
        actual = &cpu_value.run->inputs[i].actual;
      }
      size_t data_size = (actual != nullptr ? 
                          actual->DataCount(): 
                          net_value.datas[i].shape.DataCount()) * data_type_size(input_data_type);

      memop_->CopyFromHostAsync(dst_net, src_cpu, data_size, model_->GetStream());
    }

    // 异步拷贝可能尚未执行，必须等其完成后再释放 cpu_input 票据：
    // 否则下一批预处理将覆写仍在被 GPU 读取的源 buffer（输入撕裂）
    memop_->SyncStream(model_->GetStream());

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
      output_res_(output_res),
      exec_ctx_(model ? model->AcquireExecutionContext() : nullptr) {
}

InferBatchingDoneStage::~InferBatchingDoneStage() {
  if (model_) {
    // 交给后端 model_loader 释放 context
    model_->ReleaseExecutionContext(exec_ctx_);
  }
  exec_ctx_ = nullptr;
}

std::vector<std::shared_ptr<InferTask>> InferBatchingDoneStage::BatchingDone(const BatchingDoneInput& finfos) {
  std::vector<InferTaskSptr> tasks;
  InferTaskSptr task;

  QueuingTicket input_res_ticket = input_res_->PickUpNewTicket();
  QueuingTicket output_res_ticket = output_res_->PickUpNewTicket();
  task = std::make_shared<InferTask>([input_res_ticket, output_res_ticket, this, finfos]() -> int {
    QueuingTicket ir_ticket = input_res_ticket;
    QueuingTicket or_ticket = output_res_ticket;

    DeallingDoneGuard ticket_guard{this->input_res_.get(), this->output_res_.get()};

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
    // 输入侧与输出侧的契约各自传入：
    //   in_shapes  = 输入资源（前处理写入的 inputs[].actual）
    //   out_shapes = 输出资源约（后端回填 outputs[].actual，供 D2H / 后处理消费）
    model_->RunSync(exec_ctx_, input_value.ptrs, output_value.ptrs,
                    input_value.run.get(), output_value.run.get());

    if (profiler_) {
      for (const auto& finfo : finfos) {
        profiler_->RecordProcessEnd(kMODEL_PROFILER_NAME, std::make_pair(finfo.first->stream_id, finfo.first->timestamp));
      }
    }

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

    DeallingDoneGuard ticket_guard{this->net_output_res_.get(), this->cpu_output_res_.get()};

    IOResValue net_output_value = this->net_output_res_->WaitResourceByTicket(&mor_ticket);
    IOResValue cpu_output_value = this->cpu_output_res_->WaitResourceByTicket(&cor_ticket);

#ifdef VSTREAM_UNIT_TEST
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(finfos.size() == batchsize_);
#endif

    for (uint32_t bidx = 0; bidx < batchsize_; bidx++) {
      LOGU(D2H) << "bidx: " << bidx << "; [" << finfos[bidx].first->stream_id << "], ts: " << finfos[bidx].first->timestamp;
    }

    // 形状契约随批流动：device 侧 -> cpu 侧
    if (net_output_value.run && cpu_output_value.run) {
      *cpu_output_value.run = *net_output_value.run;
    }

    for (int i = 0; i < model_->OutputNum(); i++) {
      void* src_net = net_output_value.ptrs[i].get();
      void* dst_cpu = cpu_output_value.ptrs[i].get();
      auto output_data_type = model_->OutputDataType(i);
      // 动态模式只回拷 actual 大小（输出尺寸随输入变化）；否则回落到缓冲区形状
      const TensorShape* actual = nullptr;
      if (net_output_value.run != nullptr && i < static_cast<int>(net_output_value.run->outputs.size())) {
        actual = &net_output_value.run->outputs[i].actual;
      }
      size_t data_size = (actual != nullptr ? 
                          actual->DataCount(): 
                          net_output_value.datas[i].shape.DataCount()) * data_type_size(output_data_type);
      memop_->CopyToHostAsync(dst_cpu, src_net, data_size, model_->GetStream());
    }
    memop_->SyncStream(model_->GetStream());

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

          DeallingDoneGuard ticket_guard{cpu_output_res.get()};

          IOResValue cpu_output_value = cpu_output_res->WaitResourceByTicket(&cor_ticket);

          LOGU(POSTPROC) << "bidx: " << bidx << "; [" << finfo.first->stream_id << "], ts: "
                         << finfo.first->timestamp;

          std::vector<float*> cpu_outputs;

          // cpu_outputs 长度 == output tensor num
          for (size_t output_idx = 0; output_idx < cpu_output_value.datas.size(); ++output_idx) {
            // bidx 指明了在当前 batch 中的 index
            cpu_outputs.push_back(reinterpret_cast<float*>(cpu_output_value.datas[output_idx].Offset(bidx)));
          }
          if (!cnstream::IsStreamRemoved(finfo.first->stream_id)) {
            this->postprocessor_->Execute(cpu_outputs, this->model_, finfo.first,
                                          cpu_output_value.run.get());
            // 输入派生还原：把本模型产物从模块派生图坐标还原回基准图坐标
            if (this->input_deriver_) {
              this->input_deriver_->Restore(finfo.first, this->model_->get_name());
            }
          }
          return 0;
        });  // task

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

    DeallingDoneGuard ticket_guard{net_output_res.get()};

    IOResValue net_output_value = net_output_res->WaitResourceByTicket(&mor_ticket);

    LOGU(POSTPROC) << "batch postproc on device; frame num: " << finfos.size();

    std::vector<void*> net_outputs;
    for (size_t output_idx = 0; output_idx < net_output_value.datas.size(); ++output_idx) {
      net_outputs.push_back(net_output_value.datas[output_idx].ptr);
    }

    std::vector<std::shared_ptr<FrameInfo>> batched_finfos;
    for (const auto& it : finfos) batched_finfos.push_back(it.first);

    this->postprocessor_->Execute(net_outputs, this->model_, batched_finfos,
                                  net_output_value.run.get());
    // 输入派生还原：把本模型产物从模块派生图坐标还原回基准图坐标
    if (this->input_deriver_) {
      for (const auto& finfo : batched_finfos) {
        this->input_deriver_->Restore(finfo, this->model_->get_name());
      }
    }
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

          DeallingDoneGuard ticket_guard{cpu_output_res.get()};

          IOResValue cpu_output_value = cpu_output_res->WaitResourceByTicket(&cor_ticket);

          LOGU(POSTPROC) << "bidx: " << bidx << "; [" << finfo.first->stream_id << "], ts: "
                         << finfo.first->timestamp;

          std::vector<float*> cpu_outputs;
          for (size_t output_idx = 0; output_idx < cpu_output_value.datas.size(); ++output_idx) {
            cpu_outputs.push_back(reinterpret_cast<float*>(cpu_output_value.datas[output_idx].Offset(bidx)));
          }
          if (!cnstream::IsStreamRemoved(finfo.first->stream_id)) {
            this->postprocessor_->Execute(cpu_outputs, this->model_, finfo.first, obj,
                                          cpu_output_value.run.get());
          }
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

        DeallingDoneGuard ticket_guard{net_output_res.get()};

        IOResValue net_output_value = net_output_res->WaitResourceByTicket(&mor_ticket);

        LOGU(POSTPROC) << "batch obj postproc on device; obj num: " << objs.size();

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

        this->postprocessor_->Execute(net_outputs, this->model_, batched_objs,
                                      net_output_value.run.get());
        return 0;
      });
  tasks.push_back(task);

  return tasks;
}


}  // namespace cnstream