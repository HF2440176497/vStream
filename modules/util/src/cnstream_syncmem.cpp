/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "cnstream_common.hpp"
#include "cnstream_logging.hpp"

#include "cnstream_syncmem.hpp"

namespace cnstream {

CNSyncedMemory::CNSyncedMemory(size_t size) : size_(size) {
  own_dev_data_[DevType::CPU] = false;
}

CNSyncedMemory::~CNSyncedMemory() {
  if (0 == size_) return;
  if (cpu_ptr_ && own_dev_data_[DevType::CPU]) {
    CNStreamFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = nullptr;
}

void* CNSyncedMemory::Allocate() {
  return GetMutableCpuData();
}

void CNSyncedMemory::SetData(void* data) {
  SetCpuData(data);
}

void CNSyncedMemory::SetCpuData(void* data) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (0 == size_) return;
  LOGF_IF(FRAME, NULL == data) << "data is NULL.";
  if (cpu_ptr_ && own_dev_data_[DevType::CPU]) {  // free original cpu data
    CNStreamFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = SyncedHead::HEAD_AT_CPU;
  own_dev_data_[DevType::CPU] = false;
}

int CNSyncedMemory::GetDevId() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return device_id_;
}

const void* CNSyncedMemory::GetCpuData() {
  std::lock_guard<std::mutex> lock(mutex_);
  ToCpu();
  return const_cast<const void*>(cpu_ptr_);
}

void* CNSyncedMemory::GetMutableCpuData() {
  std::lock_guard<std::mutex> lock(mutex_);
  ToCpu();
  head_ = SyncedHead::HEAD_AT_CPU;  // 表示数据在 CPU 最新的，后续操作都需要同步
  return cpu_ptr_;
}

void CNSyncedMemory::ToCpu() {
  if (0 == size_) return;
  
  switch (head_) {
    case SyncedHead::UNINITIALIZED:
      if (cpu_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCpu ERROR, cpu_ptr_ should be NULL.";
        return;
      }
      CNStreamMallocHost(&cpu_ptr_, size_);
      memset(cpu_ptr_, 0, size_);
      head_ = SyncedHead::HEAD_AT_CPU;
      own_dev_data_[DevType::CPU] = true;
      break;
    case SyncedHead::HEAD_AT_CPU:
      if (!cpu_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCpu ERROR, cpu_ptr_ should not be NULL.";
        return;
      }
    case SyncedHead::SYNCED:
      if (!cpu_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCpu ERROR, cpu_ptr_ should not be NULL.";
        return;
      }
      break;
  }  // end switch
}

}  // namespace cnstream
