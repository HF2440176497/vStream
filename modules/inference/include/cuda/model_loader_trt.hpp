

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

  nvinfer1::IExecutionContext* CreateExecutionContext();

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif

  bool LoadEngine(const std::string& engine_path);
  bool ParseBindings();

  ModelLoaderTrt::Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_ = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_ = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::mutex mutex_;

};  // end of ModelLoaderTrt

}  // end of inference

#endif  // MODULES_INFERENCE_SRC_MODEL_LOADER_TRT_HPP_