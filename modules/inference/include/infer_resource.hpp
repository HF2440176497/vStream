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

#ifndef MODULES_INFERENCE_INFER_RESOURCE_HPP_
#define MODULES_INFERENCE_INFER_RESOURCE_HPP_


#include <memory>
#include <vector>

#include "cnstream_frame_va.hpp"
#include "exception.hpp"
#include "model_loader.hpp"
#include "queuing_server.hpp"
#include "tensor.hpp"

namespace cnstream {

class ModelLoader;

// 资源基类：组合 QueuingServer 的 slot 排队语义与 ModelLoader 句柄
template <typename RetT>
class InferResource : public QueuingServer {
 public:
  InferResource(ModelLoader* model) : model_(model) {}
  virtual ~InferResource() {}
  virtual void Init() {}
  virtual void Destroy() {}

 protected:
  ModelLoader* model_ = nullptr;
};  // class InferResource

struct IOResValue {
  struct IOResData {
    void* ptr = nullptr;  // 每个 tensor 的内存数据指针
    TensorShape shape;
    size_t batch_offset = 0;  // 单个数据的偏移量
    uint32_t batchsize = 0;
    void* Offset(int batch_idx) const {
      return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) + batch_offset * batch_idx);
    }
  };
  // size == input/output tensor num
  std::vector<std::shared_ptr<void>> ptrs {};  // RAII 内存，在 Resource 析构时自动释放
  std::vector<IOResData> datas {};

  void* stream = nullptr;  // slot 绑定的执行流
                           // 异步流水线模式，WaitResourceByTicket 时由 ModelLoader 查询填充；
                           // 同流串行保证 H2D/infer/D2H 读写顺序；未启用异步时为 nullptr，各 stage 回退 GetStream()
};  // struct IOResValue

// InferResource 规定获取 value_ 的接口
// 派生的子类再实现 Allocate 的方法
CNSTREAM_REGISTER_EXCEPTION(IOResource);

// IOResource: 通过 Allocate 分配内存，通过 Deallocate 释放内存
// 子类定义封装 IOResValue 的获取和析构方法
// 池化：values_ 为 N 份 buffer（N = 资源池深度，默认 1），由 QueuingServer 的 slot 机制分配
class IOResource : public InferResource<IOResValue> {
 public:
  IOResource(ModelLoader* model);
  virtual ~IOResource();

  void Init() override;
  void Destroy() override;

  // 设置资源池深度（须在 Init 之前调用），默认 1 与原始单 buffer 语义等价
  void SetResPoolSize(uint32_t n) { res_pool_size_ = n < 1 ? 1 : n; }
  uint32_t GetResPoolSize() const { return res_pool_size_; }

  // 等待票据被服务后，返回该票据绑定 slot 对应的 buffer 及其执行流；
  // 同一 run 的各阶段取到同一 slot（chain 转交），不同 run 取到不同 slot
  IOResValue WaitResourceByTicket(QueuingTicket* pticket) {
    WaitByTicket(pticket);
    const int slot = pticket->Slot();
    IOResValue value = values_[slot];
    value.stream = model_ ? model_->GetSlotStream(slot) : nullptr;
    return value;
  }

 protected:
  virtual IOResValue Allocate(ModelLoader* model) = 0;
  virtual void Deallocate(ModelLoader* model, const IOResValue& value) = 0;

 protected:
  std::vector<IOResValue> values_;  // 池化 buffer，长度 == res_pool_size_
  uint32_t res_pool_size_ = 1;
  std::shared_ptr<MemOp> memop_ = nullptr;  // 平台相关的内存操作接口
};  // class IOResource

class CpuInputResource : public IOResource {
 public:
  CpuInputResource(ModelLoader* model);
  ~CpuInputResource();

 protected:
  IOResValue Allocate(ModelLoader* model) override;
  void Deallocate(ModelLoader* model, const IOResValue& value) override;
};  // class CpuInputResource

class CpuOutputResource : public IOResource {
 public:
  CpuOutputResource(ModelLoader* model);
  ~CpuOutputResource();

 protected:
  IOResValue Allocate(ModelLoader* model) override;
  void Deallocate(ModelLoader* model, const IOResValue& value) override;
};  // class CpuOutputResource

class NetInputResource : public IOResource {
 public:
  NetInputResource(ModelLoader* model);
  ~NetInputResource();

 protected:
  IOResValue Allocate(ModelLoader* model) override;
  void Deallocate(ModelLoader* model, const IOResValue& value) override;
};  // class NetInputResource

class NetOutputResource : public IOResource {
 public:
  NetOutputResource(ModelLoader* model);
  ~NetOutputResource();

 protected:
  IOResValue Allocate(ModelLoader* model) override;
  void Deallocate(ModelLoader* model, const IOResValue& value) override;
};  // class NetOutputResource


using CpuInputResourcePtr = std::shared_ptr<CpuInputResource>;
using CpuOutputResourcePtr = std::shared_ptr<CpuOutputResource>;
using NetInputResourcePtr = std::shared_ptr<NetInputResource>;
using NetOutputResourcePtr = std::shared_ptr<NetOutputResource>;

}  // namespace cnstream

#endif  // MODULES_INFERENCE_SRC_INFER_RESOURCE_HPP_
