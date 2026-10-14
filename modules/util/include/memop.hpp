
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

  virtual void CopyFromHostAsync(void* dst, const void* src, size_t size, void* stream = nullptr);
  virtual void CopyToHostAsync(void* dst, const void* src, size_t size, void* stream = nullptr);
  virtual void SyncStream(void* stream = nullptr);

  /**
   * @brief event 同步原语（平台可选能力，用于替代整流同步 SyncStream）
   *
   * 语义差别（以同一条 stream 为例）：
   *   SyncStream(stream)  : 等待「调用时刻该流上所有已提交的工作」
   *   Record+WaitEvent    : 等待「RecordEvent 之前已提交的工作」
   *
   * 生命周期由调用方负责：CreateEvent() 创建、DestroyEvent() 销毁。
   * 同一个 event 可以反复 Record（覆盖式记录）。
   */
  virtual void* CreateEvent();
  /**
   * @return true 成功；false 失败
   * @note 必须检查返回值
   */
  virtual bool RecordEvent(void* event, void* stream);
  virtual bool WaitEvent(void* event);
  virtual void DestroyEvent(void* event);

  virtual int GetDeviceId() const;
  virtual std::unique_ptr<CNSyncedMemory> CreateSyncedMemory(size_t size);
  virtual int ConvertImageFormat(CNSyncedMemory* dst_mem, DataFormat dst_fmt, const DecodeFrame* src_frame,
                                  void* stream = nullptr);

 protected:
  size_t size_ = 0;  // only used in Allocate RAII mem
};

}  // namespace cnstream

#endif