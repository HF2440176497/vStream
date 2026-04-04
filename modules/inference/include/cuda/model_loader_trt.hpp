

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

#include "model_loader.hpp"

/**
 * @brief 模型加载器, 进行实际的模型加载和解析
 */
class ModelLoaderTrt : public ModelLoader {

  class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
  };

 public:
  ModelLoaderTrt(int device_id = 0);
  ~ModelLoaderTrt();

  bool Init(const std::string& engine_path) override;

  bool IsValid() override { return engine_ != nullptr; }

  bool RunSync(std::vector<std::shared_ptr<void>> inputs, std::vector<std::shared_ptr<void>> outputs) override;

  nvinfer1::IExecutionContext* CreateExecutionContext();

#ifdef UNIT_TEST
 public:
#else
 private:
#endif

  bool LoadEngine(const std::string& engine_path);
  bool ParseBindings();

  ModelLoaderTrt::Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;
  cudaStream_t stream_ = nullptr;

};  // end of ModelLoaderTrt

#endif  // MODULES_INFERENCE_SRC_MODEL_LOADER_TRT_HPP_