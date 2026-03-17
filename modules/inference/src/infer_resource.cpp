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


namespace cnstream {

// Note: virtual functions should not be called in constructor
IOResource::IOResource(std::shared_ptr<ModelLoader> model, uint32_t batchsize)
    : InferResource<IOResValue>(model, batchsize) {
}

IOResource::~IOResource() {}

CpuInputResource::CpuInputResource(std::shared_ptr<ModelLoader> model, uint32_t batchsize)
    : IOResource(model, batchsize) {
  memop_ = MemOpFactory::Instance().CreateMemOp(DevType::CPU, -1);
}

CpuInputResource::~CpuInputResource() {}


IOResValue CpuInputResource::Allocate(std::shared_ptr<ModelLoader> model, uint32_t batchsize) {
  int input_num = model->InputNum();

  IOResValue value;
  value.datas.resize(input_num);
  value.ptrs.resize(input_num);

  for (int input_idx = 0; input_idx < input_num; ++input_idx) {
    auto shape = model->InputShape(input_idx);
    auto data_type = model->InputDataType(input_idx);  // 已确保为 FLOAT32
    size_t data_size = shape.DataCount() * data_type_size(data_type);
    size_t batch_offset = (shape.DataCount() / shape.N()) * data_type_size(data_type);
    value.ptrs[input_idx] = memop_->Allocate(data_size);
    value.datas[input_idx].ptr = value.ptrs[input_idx].get();
    value.datas[input_idx].shape = shape;
    value.datas[input_idx].batch_offset = batch_offset;
    value.datas[input_idx].batchsize = batchsize;
  }
  return value;
}

void CpuInputResource::Deallocate(std::shared_ptr<ModelLoader> model, uint32_t batchsize,
                                  const IOResValue& value) {
}

CpuOutputResource::CpuOutputResource(std::shared_ptr<ModelLoader> model, uint32_t batchsize)
    : IOResource(model, batchsize) {}

CpuOutputResource::~CpuOutputResource() {}

IOResValue CpuOutputResource::Allocate(std::shared_ptr<ModelLoader> model, uint32_t batchsize) {
  int output_num = model->OutputNum();
  IOResValue value;
  value.datas.resize(output_num);
  value.ptrs.resize(output_num);

  for (int output_idx = 0; output_idx < output_num; ++output_idx) {
    auto shape = model->OutputShape(output_idx);
    auto data_type = model->OutputDataType(output_idx);  // 已确保为 FLOAT32
    size_t data_size = shape.DataCount() * data_type_size(data_type);
    size_t batch_offset = (shape.DataCount() / shape.N()) * data_type_size(data_type);
    value.ptrs[output_idx] = memop_->Allocate(data_size);
    value.datas[output_idx].ptr = value.ptrs[output_idx].get();
    value.datas[output_idx].shape = shape;
    value.datas[output_idx].batch_offset = batch_offset;
    value.datas[output_idx].batchsize = batchsize;
  }
  return value;
}

void CpuOutputResource::Deallocate(std::shared_ptr<ModelLoader> model, uint32_t batchsize,
                                   const IOResValue& value) {
}

NetInputResource::NetInputResource(std::shared_ptr<ModelLoader> model, uint32_t batchsize)
    : IOResource(model, batchsize) {
  memop_ = MemOpFactory::Instance().CreateMemOp(model->GetDeviceType(), model->GetDeviceId());
}

NetInputResource::~NetInputResource() {}

IOResValue NetInputResource::Allocate(std::shared_ptr<ModelLoader> model, uint32_t batchsize) {
  int input_num = model->InputNum();
  IOResValue value;
  value.datas.resize(input_num);
  value.ptrs.resize(input_num);

  for (int input_idx = 0; input_idx < input_num; ++input_idx) {
    auto shape = model->InputShape(input_idx);
    auto data_type = model->InputDataType(input_idx);  // 已确保为 FLOAT32
    size_t data_size = shape.DataCount() * data_type_size(data_type);
    size_t batch_offset = (shape.DataCount() / shape.N()) * data_type_size(data_type);
    value.ptrs[input_idx] = memop_->Allocate(data_size);
    value.datas[input_idx].ptr = value.ptrs[input_idx].get();
    value.datas[input_idx].shape = shape;
    value.datas[input_idx].batch_offset = batch_offset;
    value.datas[input_idx].batchsize = batchsize;
  }
  return value;
}

void NetInputResource::Deallocate(std::shared_ptr<ModelLoader> model, uint32_t batchsize,
                                  const IOResValue& value) {
}

NetOutputResource::NetOutputResource(std::shared_ptr<ModelLoader> model, uint32_t batchsize)
    : IOResource(model, batchsize) {
  memop_ = MemOpFactory::Instance().CreateMemOp(model->GetDeviceType(), model->GetDeviceId());
}

NetOutputResource::~NetOutputResource() {}

IOResValue NetOutputResource::Allocate(std::shared_ptr<ModelLoader> model, uint32_t batchsize) {
  int output_num = model->OutputNum();
  IOResValue value;
  value.datas.resize(output_num);
  value.ptrs.resize(output_num);

  for (int output_idx = 0; output_idx < output_num; ++output_idx) {
    auto shape = model->OutputShape(output_idx);
    auto data_type = model->OutputDataType(output_idx);  // 已确保为 FLOAT32
    size_t data_size = shape.DataCount() * data_type_size(data_type);
    size_t batch_offset = (shape.DataCount() / shape.N()) * data_type_size(data_type);
    value.ptrs[output_idx] = memop_->Allocate(data_size);
    value.datas[output_idx].ptr = value.ptrs[output_idx].get();
    value.datas[output_idx].shape = shape;
    value.datas[output_idx].batch_offset = batch_offset;
    value.datas[output_idx].batchsize = batchsize;
  }
  return value;
}

void NetOutputResource::Deallocate(std::shared_ptr<ModelLoader> model, uint32_t batchsize,
                                   const IOResValue& value) {
}

}  // namespace cnstream
