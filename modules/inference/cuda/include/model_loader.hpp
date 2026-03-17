
// ==================== TensorRT Includes =================
#include <NvInfer.h>

// ==================== CUDA Includes =====================
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda_fp16.h>

// ==================== Common Includes ===================
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
  TrtModelLoader(const std::string& engine_path, int device_id = 0);
  ~TrtModelLoader();

  bool IsValid() override { return engine_ != nullptr; }

  nvinfer1::ICudaEngine* GetEngine() const { 
    if (!engine_) {
      return nullptr;
    }
    return engine_.get(); 
  }
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
  void ParseBindings();

  TrtModelLoader::Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;

  std::map<std::string, int> bind_name_index_map_ {};  // bind_name - index
  std::string input_name_;  // frist input name
  std::string output_name_;  // frist output name
  int input_ordered_index_ = 0;
  int output_ordered_index_ = 0;

};  // end of TrtModelLoader
