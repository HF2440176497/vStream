
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

#include "model_loader_base.hpp"

/**
 * @brief 模型加载器, 进行实际的模型加载和解析
 */
class TrtModelLoader : public ModelLoader {

  class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
  };

 public:
  TrtModelLoader(int device_id = 0);
  ~TrtModelLoader();

  bool Init(const std::string& engine_path) override;

  bool IsValid() override { return engine_ != nullptr; }

  nvinfer1::IExecutionContext* CreateExecutionContext();

  // return dims of index
  size_t GetInputDataBatchAlignSize(uint32_t index) const;
  size_t GetOutputDataBatchAlignSize(uint32_t index) const;

#ifdef UNIT_TEST
 public:
#else
 private:
#endif

  bool LoadEngine(const std::string& engine_path);
  bool ParseBindings();

  TrtModelLoader::Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;

};  // end of TrtModelLoader
