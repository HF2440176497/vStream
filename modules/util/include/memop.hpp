
#ifndef MEMOP_HPP_
#define MEMOP_HPP_

#include <memory>
#include <map>
#include <mutex>

#include "cnstream_syncmem.hpp"
#include "cnstream_allocator.hpp"

namespace cnstream {

struct Buffer {
  std::shared_ptr<void> data = nullptr;
  size_t size = 0;
  int device_id = -1;
  Buffer() = default;
  Buffer(std::shared_ptr<void> d, size_t s, int dev) 
      : data(std::move(d)), size(s), device_id(dev) {}
  ~Buffer() {
    if (data) {
      data.reset();
    }
  }
};

/**
 * 用于包含多平台分配的内存
 */
class MemoryBufferCollection {
 public:
  MemoryBufferCollection() = default;
  ~MemoryBufferCollection() {
    ClearAll();
  }
  Buffer& GetBuffer(DevType type, size_t size, int device_id);
  bool Has(DevType type);
  Buffer* Get(DevType type);
  void Clear(DevType type);
  void ClearAll();
  size_t GetDeviceCount();

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  std::map<DevType, Buffer> buffers_ {};
  std::mutex mutex_;
};

/**
 * @brief 向上取为 64 KB 的整数倍
 */
inline size_t RoundUpSize(size_t bytes) {
  const size_t alignment = 64 * 1024;
  return (bytes + alignment - 1) / alignment * alignment;
}

struct DecodeFrame;

/**
 * @brief 内存操作算子
 */
class MemOp {
 public:
  MemOp();
  virtual ~MemOp();
  
  virtual std::shared_ptr<void> Allocate(size_t bytes);  // 分配 RAII 内存
  virtual void Copy(void* dst, const void* src, size_t size);
  virtual void CopyFromHost(void* dst, const void* src, size_t size);
  virtual void CopyToHost(void* dst, const void* src, size_t size);

  virtual int GetDeviceId() const;
  virtual std::unique_ptr<CNSyncedMemory> CreateSyncedMemory(size_t size);
  virtual int ConvertImageFormat(CNSyncedMemory* dst_mem, DataFormat dst_fmt, const DecodeFrame* src_frame);

 protected:
  size_t size_ = 0;  // only used in Allocate RAII mem
};

}  // namespace cnstream

#endif