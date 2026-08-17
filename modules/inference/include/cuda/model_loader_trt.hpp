

#ifndef MODULES_INFERENCE_SRC_MODEL_LOADER_TRT_HPP_
#define MODULES_INFERENCE_SRC_MODEL_LOADER_TRT_HPP_


#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda_fp16.h>

#include <NvInfer.h>

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <deque>

#include "model_loader.hpp"

namespace cnstream {

struct TrtDeleter {
  template<typename T>
  void operator()(T* ptr) const {
    if (ptr) {
      delete ptr;
    }
  }
};

/**
 * @brief 异步执行 slot：独立的执行上下文 + 执行流 + 完成事件
 * 同一 slot 上的任务在流上天然串行；不同 slot 之间可并发执行
 */
struct TrtAsyncSlot {
  nvinfer1::IExecutionContext* context = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  std::mutex mtx;  // 保护该 slot 上下文的地址设置 + enqueue（多引擎共享 loader 时串行化）
};

/**
 * @brief 模型加载器, 进行实际的模型加载和解析
 */
class ModelLoaderTrt : public ModelLoader {

 public:
  class Logger : public nvinfer1::ILogger {
   public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
  };
  ModelLoaderTrt(int device_id);
  ~ModelLoaderTrt();

  bool Init(const std::string& engine_path, const InferParams& params) override;

  bool IsValid() override { return engine_ != nullptr; }

  bool RunSync(std::vector<std::shared_ptr<void>> inputs, std::vector<std::shared_ptr<void>> outputs) override;

  bool EnableAsync(int slot_num) override;
  void* GetSlotStream(int slot) const override;
  void* RunAsync(const std::vector<std::shared_ptr<void>>& inputs,
                 const std::vector<std::shared_ptr<void>>& outputs,
                 void* stream) override;
  void SyncEvent(void* event) override;

  void* GetStream() const override { return static_cast<void*>(stream_); }

  nvinfer1::IExecutionContext* CreateExecutionContext();

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif

  bool LoadEngine(const std::string& engine_path);
  bool ParseBindings();

  bool ApplyInputShapes(nvinfer1::IExecutionContext* context);
  TrtAsyncSlot* FindSlotByStream(void* stream);
  void DestroyAsyncSlots();          // 需在 async_mtx_ 保护下调用
  void DestroyAsyncSlotsLocked();    // 需在 async_mtx_ 保护下调用

  ModelLoaderTrt::Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_ = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_ = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::mutex mutex_;

  std::deque<TrtAsyncSlot> async_slots_;  // 异步执行 slot 池（EnableAsync 创建）
  std::mutex async_mtx_;                  // 保护 async_slots_

};  // end of ModelLoaderTrt

}  // end of inference

#endif  // MODULES_INFERENCE_SRC_MODEL_LOADER_TRT_HPP_