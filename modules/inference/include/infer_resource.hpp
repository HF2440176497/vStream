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


#include "model_loader.hpp"

#include <memory>
#include <vector>

#include "cnstream_frame_va.hpp"
#include "exception.hpp"
#include "queuing_server.hpp"

namespace cnstream {

template <typename RetT>
class InferResource : public QueuingServer {
 public:
  InferResource(ModelLoader* model, uint32_t batchsize) : model_(model), batchsize_(batchsize) {}
  virtual ~InferResource() {}
  virtual void Init() {}
  virtual void Destroy() {}
  RetT WaitResourceByTicket(QueuingTicket* pticket) {
    WaitByTicket(pticket);
    return value_;
  }
  // InferResource 相当于帮助我们保管 value_，等待唤醒后才能够操作 value_

  RetT GetDataDirectly() const { return value_; }

 protected:
  ModelLoader* model_ = nullptr;
  const uint32_t batchsize_ = 0;
  RetT value_;
};  // class InferResource

// Note: ptrs 含有 RAII 内存，在 Resource 析构时自动释放
struct IOResValue {
  struct IOResData {
    void* ptr = nullptr;
    TensorShape shape;
    size_t batch_offset = 0;  // 每个 batch 数据的偏移量
    uint32_t batchsize = 0;
    void* Offset(int batch_idx) const {
      return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) + batch_offset * batch_idx);
    }
  };
  std::vector<std::shared_ptr<void>> ptrs;
  std::vector<IOResData> datas;
};  // struct IOResValue

// InferResource 规定获取 value_ 的接口
// 派生的子类再实现 Allocate 的方法
CNSTREAM_REGISTER_EXCEPTION(IOResource);

// IOResource: 通过 Allocate 分配内存，通过 Deallocate 释放内存
// 子类定义封装 IOResValue 
class IOResource : public InferResource<IOResValue> {
 public:
  IOResource(ModelLoader* model, uint32_t batchsize);
  virtual ~IOResource();

  void Init() override { value_ = Allocate(model_, batchsize_); }
  void Destroy() override { Deallocate(model_, batchsize_, value_); }

 protected:
  virtual IOResValue Allocate(ModelLoader* model, uint32_t batchsize) = 0;
  virtual void Deallocate(ModelLoader* model, uint32_t batchsize, const IOResValue& value) = 0;

 protected:
  std::shared_ptr<MemOp> memop_ = nullptr;  // 平台相关的内存操作接口
};  // class IOResource

class CpuInputResource : public IOResource {
 public:
  CpuInputResource(ModelLoader* model, uint32_t batchsize);
  ~CpuInputResource();

 protected:
  IOResValue Allocate(ModelLoader* model, uint32_t batchsize) override;
  void Deallocate(ModelLoader* model, uint32_t batchsize, const IOResValue& value) override;
};  // class CpuInputResource

class CpuOutputResource : public IOResource {
 public:
  CpuOutputResource(ModelLoader* model, uint32_t batchsize);
  ~CpuOutputResource();

 protected:
  IOResValue Allocate(ModelLoader* model, uint32_t batchsize) override;
  void Deallocate(ModelLoader* model, uint32_t batchsize, const IOResValue& value) override;
};  // class CpuOutputResource

class NetInputResource : public IOResource {
 public:
  NetInputResource(ModelLoader* model, uint32_t batchsize);
  ~NetInputResource();

 protected:
  IOResValue Allocate(ModelLoader* model, uint32_t batchsize) override;
  void Deallocate(ModelLoader* model, uint32_t batchsize, const IOResValue& value) override;
};  // class NetInputResource

class NetOutputResource : public IOResource {
 public:
  NetOutputResource(ModelLoader* model, uint32_t batchsize);
  ~NetOutputResource();

 protected:
  IOResValue Allocate(ModelLoader* model, uint32_t batchsize) override;
  void Deallocate(ModelLoader* model, uint32_t batchsize, const IOResValue& value) override;
};  // class NetOutputResource


using CpuInputResourcePtr = std::shared_ptr<CpuInputResource>;
using CpuOutputResourcePtr = std::shared_ptr<CpuOutputResource>;
using NetInputResourcePtr = std::shared_ptr<NetInputResource>;
using NetOutputResourcePtr = std::shared_ptr<NetOutputResource>;

}  // namespace cnstream

#endif  // MODULES_INFERENCE_SRC_INFER_RESOURCE_HPP_
