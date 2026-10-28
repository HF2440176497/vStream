

#ifndef CNSTREAM_ALLOCATOR_CUDA_HPP_
#define CNSTREAM_ALLOCATOR_CUDA_HPP_

#include <atomic>
#include <memory>
#include <new>

#include "cnstream_allocator.hpp"
#include "cnstream_logging.hpp"
#include "cuda/cuda_check.hpp"

namespace cnstream {

/*!
 * @class CudaDeviceGuard
 *
 * @brief CudaDeviceGuard is a class for setting current thread's CUDA device handler.
 */
class CudaDeviceGuard : private NonCopyable {
 public:
  /*!
   * @brief Sets the CUDA device handler with the given device ordinal.
   *
   * @param[in] device_id The CUDA device ordinal to retrieve.
   *
   * @return No return value.
   */
  explicit CudaDeviceGuard(int device_id): device_id_(device_id) {
    if (device_id < 0) {
      LOGW(CORE) << "CudaDeviceGuard: Invalid device ID " << device_id << ". Do nothing.";
      ok_ = true;  // 保持旧行为：无有效 CUDA 设备时不做任何 CUDA 操作
      return;
    }
    cudaError_t err = cudaGetDevice(&prev_device_id_);
    if (err != cudaSuccess) {
      LOGE(CORE) << "CudaDeviceGuard: Failed to get current device: " << cudaGetErrorString(err);
      prev_device_id_ = -1;
    }
    err = cudaSetDevice(device_id_);
    ok_ = (err == cudaSuccess);
    if (!ok_) {
      LOGE(CORE) << "CudaDeviceGuard: cudaSetDevice(" << device_id_ << ") failed: "
                 << cudaGetErrorName(err) << " (" << cudaGetErrorString(err)
                 << "). Caller must skip CUDA work.";
    }
  }

  // 调用方必须检查：false 表示当前线程没有切到目标设备，不能再执行
  // cudaMalloc/cudaFree/cudaMemcpy 等操作，否则会在错误 context 上继续并污染清理路径。
  bool ok() const { return ok_; }
  /*!
   * @brief Destructs an object.
   *
   * @return No return value.
   */
  ~CudaDeviceGuard() {
    if (ok_ && prev_device_id_ >= 0 && prev_device_id_ != device_id_) {
      cudaError_t err = cudaSetDevice(prev_device_id_);
      if (err != cudaSuccess) {
        LOGE(CORE) << "CudaDeviceGuard: restore device " << prev_device_id_
                   << " failed: " << cudaGetErrorName(err);
      }
    }
  }
 private:
  int device_id_ = 0;
  int prev_device_id_ = -1;  // Store previous device ID for restoration
  bool ok_ = false;
};  // end CudaDeviceGuard


class CudaAllocator : public MemoryAllocator {
 public:
  explicit CudaAllocator(int device_id = 0) : MemoryAllocator(device_id) {}
  ~CudaAllocator() = default;

  void *alloc(size_t size, int timeout_ms = 0) override {
    size_t alloc_size = (size + 4095) & (~0xFFF);  // Align 4096
    size_ = alloc_size;

    std::lock_guard<std::mutex> lk(mutex_);
    CudaDeviceGuard guard(device_id_);
    if (!guard.ok()) {
      LOGE(CORE) << "CudaAllocator: skip alloc, cannot set CUDA device " << device_id_;
      return nullptr;
    }

    void *cuda_ptr;
    cudaError_t err = cudaMalloc(&cuda_ptr, alloc_size);
    if (err != cudaSuccess) {
        LOGE(CORE) << "CudaAllocator: Failed to allocate memory: " << cudaGetErrorString(err);
        return nullptr;
    }
    return cuda_ptr;
  }
  void free(void *p) override {
    std::lock_guard<std::mutex> lk(mutex_);
    CudaDeviceGuard guard(device_id_);
    if (!guard.ok()) {
      LOGE(CORE) << "CudaAllocator: skip cudaFree, cannot set CUDA device " << device_id_;
      return;
    }
    CHECK_CUDA_RUNTIME(cudaFree(p));
  }
};

/**
 * CUDA 平台全局函数
 */
inline std::shared_ptr<void> cnCudaMemAlloc(size_t size, int device_id) {
  std::shared_ptr<MemoryAllocator> allocator = std::make_shared<CudaAllocator>(device_id);
  return cnMemAlloc(size, allocator);  // 调用 CudaAllocator::alloc 方法
}

}  // namespace cnstream

#endif  // CNSTREAM_ALLOCATOR_CUDA_HPP_