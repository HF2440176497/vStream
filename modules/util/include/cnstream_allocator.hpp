/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
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
#ifndef CNSTREAM_ALLOCATOR_HPP_
#define CNSTREAM_ALLOCATOR_HPP_

#include <atomic>
#include <memory>
#include <new>

#include "cnstream_common.hpp"


/**
 *  @file cnstream_allocator.hpp
 *
 *  Allocator 是利用 RAII 机制实现的内存分配器，确保在对象生命周期结束时自动释放内存。
 */
namespace cnstream {


class MemoryAllocator : private NonCopyable {
 public:
  explicit MemoryAllocator() {};
  explicit MemoryAllocator(int device_id) : device_id_(device_id) {}
  virtual ~MemoryAllocator() = default;
  virtual void *alloc(size_t size, int timeout_ms = 0) = 0;
  virtual void free(void *p) = 0;
  int device_id() const { return device_id_; }
  void set_device_id(int device_id) { device_id_ = device_id; }
#ifdef VSTREAM_UNIT_TEST
 public:
#else
 protected:
#endif
  int device_id_ = -1;
  std::mutex mutex_;
  size_t size_ = 0;
};


/**
 * 删除器（1）使用 shared_ptr 确保使用 cnMemAlloc 时 allocator 还未销毁（2）满足可调用对象
 */
class CnAllocDeleter final {
 public:
  explicit CnAllocDeleter(std::shared_ptr<MemoryAllocator> allocator) : allocator_(allocator) {}
  void operator()(void *ptr) { allocator_->free(ptr); }

 private:
  std::shared_ptr<MemoryAllocator> allocator_;
};

/**
 * 内存分配器中间函数，根据指定的 allocator 分配内存
 */
inline std::shared_ptr<void> cnMemAlloc(size_t size, std::shared_ptr<MemoryAllocator> allocator) {
  if (allocator) {
    std::shared_ptr<void> ds(allocator->alloc(size), CnAllocDeleter(allocator));
    return ds;
  }
  return nullptr;
}


class CpuAllocator : public MemoryAllocator {
 public:
  explicit CpuAllocator() : MemoryAllocator() {}
  ~CpuAllocator() = default;

  void *alloc(size_t size, int timeout_ms = 0) override {
    size_t alloc_size = (size + 4095) & (~0xFFF);  // Align 4096
    size_ = alloc_size;
    return static_cast<void *>(new (std::nothrow) uint8_t[alloc_size]);
  }
  void free(void *p) override {
    uint8_t *ptr = static_cast<uint8_t *>(p);
    delete[] ptr;
  }
};

/**
 * 返回后 allocator 仍存在，不会立刻调用 free 释放
 */
inline std::shared_ptr<void> cnCpuMemAlloc(size_t size) {
  std::shared_ptr<MemoryAllocator> allocator = std::make_shared<CpuAllocator>();
  return cnMemAlloc(size, allocator);
}

}  // namespace cnstream

#endif  // CNSTREAM_ALLOCATOR_HPP_
