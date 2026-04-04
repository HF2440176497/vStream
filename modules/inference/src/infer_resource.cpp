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


#include <map>
#include <memory>
#include <string>
#include <utility>
#include <bitset>

#include "memop.hpp"
#include "memop_factory.hpp"

#include "inference.hpp"
#include "infer_resource.hpp"
#include "model_loader.hpp"


namespace cnstream {

// Note: virtual functions should not be called in constructor
IOResource::IOResource(ModelLoader* model)
    : InferResource<IOResValue>(model) {
}

IOResource::~IOResource() {}

CpuInputResource::CpuInputResource(ModelLoader* model)
    : IOResource(model) {
  memop_ = MemOpFactory::Instance().CreateMemOp(DevType::CPU, -1);
}

CpuInputResource::~CpuInputResource() {}


static IOResValue allocate_input_iovalue(ModelLoader* model, std::shared_ptr<MemOp> memop) {
  if (!memop || !model) {
    LOGE("InputResource Allocate input IOValue failed, memop or model is null");
    return IOResValue();
  }
  IOResValue value;
  int input_num = model->InputNum();
  value.datas.resize(input_num);
  value.ptrs.resize(input_num);

  for (int idx = 0; idx < input_num; ++idx) {
    auto shape = model->InputShape(idx);
    auto data_type = model->InputDataType(idx);  // 已确保为 FLOAT32
    size_t data_size = shape.DataCount() * data_type_size(data_type);
    size_t batch_offset = (shape.DataCount() / shape.N()) * data_type_size(data_type);

    value.ptrs[idx] = memop->Allocate(data_size);
    value.datas[idx].ptr = value.ptrs[idx].get();
    value.datas[idx].shape = shape;
    value.datas[idx].batch_offset = batch_offset;  // bytes
    value.datas[idx].batchsize = shape.N();
  }
  return value;
}

static IOResValue allocate_output_iovalue(ModelLoader* model, std::shared_ptr<MemOp> memop) {
  if (!memop || !model) {
    LOGE("OutputResource Allocate output IOValue failed, memop or model is null");
    return IOResValue();
  }
  int output_num = model->OutputNum();
  IOResValue value;
  value.datas.resize(output_num);
  value.ptrs.resize(output_num);

  for (int idx = 0; idx < output_num; ++idx) {
    auto shape = model->OutputShape(idx);
    auto data_type = model->OutputDataType(idx);  // 已确保为 FLOAT32
    size_t data_size = shape.DataCount() * data_type_size(data_type);
    size_t batch_offset = (shape.DataCount() / shape.N()) * data_type_size(data_type);

    value.ptrs[idx] = memop->Allocate(data_size);
    value.datas[idx].ptr = value.ptrs[idx].get();
    value.datas[idx].shape = shape;
    value.datas[idx].batch_offset = batch_offset;
    value.datas[idx].batchsize = shape.N();
  }
  return value;
}

IOResValue CpuInputResource::Allocate(ModelLoader* model) {
  return allocate_input_iovalue(model, memop_);
}

void CpuInputResource::Deallocate(ModelLoader* model, const IOResValue& value) {
}

CpuOutputResource::CpuOutputResource(ModelLoader* model)
    : IOResource(model) {
  memop_ = MemOpFactory::Instance().CreateMemOp(DevType::CPU, -1);
}

CpuOutputResource::~CpuOutputResource() {}

IOResValue CpuOutputResource::Allocate(ModelLoader* model) {
  return allocate_output_iovalue(model, memop_);
}

void CpuOutputResource::Deallocate(ModelLoader* model, const IOResValue& value) {
}

NetInputResource::NetInputResource(ModelLoader* model)
    : IOResource(model) {
  memop_ = MemOpFactory::Instance().CreateMemOp(model->GetDeviceType(), model->GetDeviceId());
}

NetInputResource::~NetInputResource() {}

IOResValue NetInputResource::Allocate(ModelLoader* model) {
  return allocate_input_iovalue(model, memop_);
}

void NetInputResource::Deallocate(ModelLoader* model, const IOResValue& value) {
}

NetOutputResource::NetOutputResource(ModelLoader* model)
    : IOResource(model) {
  memop_ = MemOpFactory::Instance().CreateMemOp(model->GetDeviceType(), model->GetDeviceId());
}

NetOutputResource::~NetOutputResource() {}

IOResValue NetOutputResource::Allocate(ModelLoader* model) {
  return allocate_output_iovalue(model, memop_);
}

void NetOutputResource::Deallocate(ModelLoader* model, const IOResValue& value) {
}

}  // namespace cnstream
