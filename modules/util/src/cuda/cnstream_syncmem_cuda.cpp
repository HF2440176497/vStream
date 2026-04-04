
#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_syncmem_cuda.hpp"

namespace cnstream {

CNSyncedMemoryCuda::CNSyncedMemoryCuda(size_t size) : CNSyncedMemory(size) {
  own_dev_data_[DevType::CPU] = false;
  own_dev_data_[DevType::CUDA] = false;
}

CNSyncedMemoryCuda::CNSyncedMemoryCuda(size_t size, int device_id) : CNSyncedMemory(size) {
  std::lock_guard<std::mutex> lock(mutex_);
  int device_count = 0;
  CHECK_CUDA_RUNTIME(cudaGetDeviceCount(&device_count));
  if (device_id < 0 || device_id >= device_count) {
    LOGF(FRAME) << "Invalid CUDA device id: " << device_id << ", available devices: " << device_count;
    device_id = 0;
  }
  device_id_ = device_id;
  own_dev_data_[DevType::CPU] = false;
  own_dev_data_[DevType::CUDA] = false;
}

CNSyncedMemoryCuda::~CNSyncedMemoryCuda() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (0 == size_) return;
  if (cuda_ptr_ && own_dev_data_[DevType::CUDA]) {
    // Set device before freeing memory
    if (device_id_ >= 0) {
      CHECK_CUDA_RUNTIME(cudaSetDevice(device_id_));
    }
    CHECK_CUDA_RUNTIME(cudaFree(cuda_ptr_));
  }
  cuda_ptr_ = nullptr;
  // cpu_ptr_ 交给 父类析构
}

void* CNSyncedMemoryCuda::Allocate() {
  return GetMutableCudaData();
}

void CNSyncedMemoryCuda::SetData(void* data) {
  SetCudaData(data);
}

void CNSyncedMemoryCuda::ToCpu() {
  if (0 == size_) return;
  CHECK_CUDA_RUNTIME(cudaSetDevice(device_id_));
  switch (head_) {
    case SyncedHead::UNINITIALIZED:
      if (cuda_ptr_ or cpu_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCpu ERROR, cuda_ptr_ and cpu_ptr_ should be NULL.";
        return;
      }
      CNStreamMallocHost(&cpu_ptr_, size_);
      memset(cpu_ptr_, 0, size_);
      head_ = SyncedHead::HEAD_AT_CPU;
      own_dev_data_[DevType::CPU] = true;
      break;
    case SyncedHead::HEAD_AT_CUDA:
      if (NULL == cuda_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCpu ERROR, cuda_ptr_ should not be NULL.";
        return;
      }
      if (NULL == cpu_ptr_) {
        CNStreamMallocHost(&cpu_ptr_, size_);
        memset(cpu_ptr_, 0, size_);
        own_dev_data_[DevType::CPU] = true;
      }
      // Sasha: 如果 cpu_ptr_ 已经之前指定过，上面判断也就不通过，所以也不应该改变 own_dev_data_ 对应标记
      CHECK_CUDA_RUNTIME(cudaMemcpy(cpu_ptr_, cuda_ptr_, size_, cudaMemcpyDeviceToHost));
      head_ = SyncedHead::SYNCED;
      break;
    case SyncedHead::HEAD_AT_CPU:
    case SyncedHead::SYNCED:
      break;
  }
}

void CNSyncedMemoryCuda::ToCuda() {
  if (0 == size_) return;
  // Set device if specified
  CHECK_CUDA_RUNTIME(cudaSetDevice(device_id_));
  switch (head_) {
    case SyncedHead::UNINITIALIZED:
      if (cuda_ptr_ or cpu_ptr_) {  // 不应该存在已分配的 CUDA 内存
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCuda ERROR, cuda_ptr_ and cpu_ptr_ should be NULL.";
        return;
      }
      CHECK_CUDA_RUNTIME(cudaMalloc(&cuda_ptr_, size_));
      head_ = SyncedHead::HEAD_AT_CUDA;
      own_dev_data_[DevType::CUDA] = true;
      break;
    case SyncedHead::HEAD_AT_CPU:
      if (NULL == cpu_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCuda ERROR, cpu_ptr_ should not be NULL.";
        return;
      }
      if (NULL == cuda_ptr_) {
        CHECK_CUDA_RUNTIME(cudaMalloc(&cuda_ptr_, size_));
        own_dev_data_[DevType::CUDA] = true;
      }
      CHECK_CUDA_RUNTIME(cudaMemcpy(cuda_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice));
      head_ = SyncedHead::SYNCED;
      break;
    case SyncedHead::HEAD_AT_CUDA:
    case SyncedHead::SYNCED:
      break;
  }
}

const void* CNSyncedMemoryCuda::GetCudaData() {
  std::lock_guard<std::mutex> lock(mutex_);
  ToCuda();
  return const_cast<const void*>(cuda_ptr_);
}

void CNSyncedMemoryCuda::SetCudaData(void* data) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (0 == size_) return;
  LOGF_IF(FRAME, nullptr == data) << "data is NULL.";
  if (own_dev_data_[DevType::CUDA]) {
    if (!cuda_ptr_) {
      LOGE(FRAME) << "CNSyncedMemoryCuda::SetCudaData ERROR, cuda_ptr_ should not be NULL.";
      return;
    }
    CHECK_CUDA_RUNTIME(cudaSetDevice(device_id_));
    CHECK_CUDA_RUNTIME(cudaFree(cuda_ptr_));
  }
  cuda_ptr_ = data;
  head_ = SyncedHead::HEAD_AT_CUDA;
  own_dev_data_[DevType::CUDA] = false;
}

void* CNSyncedMemoryCuda::GetMutableCudaData() {
  std::lock_guard<std::mutex> lock(mutex_);
  ToCuda();
  head_ = SyncedHead::HEAD_AT_CUDA;
  return cuda_ptr_;
}

const void* CNSyncedMemoryCuda::GetCpuData() {
  std::lock_guard<std::mutex> lock(mutex_);
  ToCpu();
  return const_cast<const void*>(cpu_ptr_);
}

void* CNSyncedMemoryCuda::GetMutableCpuData() {
  std::lock_guard<std::mutex> lock(mutex_);
  ToCpu();
  head_ = SyncedHead::HEAD_AT_CPU;
  return cpu_ptr_;
}

}