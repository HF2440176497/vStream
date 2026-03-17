

#include "builder.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <sstream>

using namespace nvinfer1;

class Logger : public ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "[TRT][FATAL] " << msg << std::endl;
        break;
      case Severity::kERROR:
        std::cerr << "[TRT][ERROR] " << msg << std::endl;
        break;
      case Severity::kWARNING:
        std::cerr << "[TRT][WARN] " << msg << std::endl;
        break;
      case Severity::kINFO:
        std::cerr << "[TRT][INFO] " << msg << std::endl;
        break;
      default:
        std::cerr << "[TRT] " << msg << std::endl;
    }
  }
};
static Logger gLogger;

namespace TRT {

static std::string join_dims(const std::vector<int>& dims) {
  if (dims.empty()) return "()";
  std::string result = "(";
  for (size_t i = 0; i < dims.size(); ++i) {
    result += std::to_string(dims[i]);
    if (i < dims.size() - 1) result += ", ";
  }
  result += ")";
  return result;
}

template <typename _T>
static void destroy_trt_pointer(_T* ptr) {
  if (ptr) delete ptr;
}

const char* mode_string(Mode type) {
  switch (type) {
    case Mode::FP32:
      return "FP32";
    case Mode::FP16:
      return "FP16";
    case Mode::INT8:
      return "INT8";
    default:
      return "Unknown";
  }
}

// ==================== INT8 校准器实现 ====================

class Int8EntropyCalibrator : public IInt8EntropyCalibrator2 {
 public:
  Int8EntropyCalibrator(const std::vector<std::vector<uint8_t>>& calibration_data, const std::vector<int>& input_dims,
                        const std::string& cache_file = "")
      : calibration_data_(calibration_data), input_dims_(input_dims), cache_file_(cache_file), current_batch_(0) {
    // 计算 batch size 和单样本大小
    batch_size_ = input_dims_[0];
    size_t single_size =
        std::accumulate(input_dims_.begin() + 1, input_dims_.end(), sizeof(float), std::multiplies<size_t>());
    batch_bytes_ = batch_size_ * single_size;

    // 分配 GPU 内存
    cudaMalloc(&device_input_, batch_bytes_);

    // 尝试加载已有缓存
    if (!cache_file_.empty() && file_exists(cache_file_)) {
      load_cache();
    }
  }

  ~Int8EntropyCalibrator() {
    if (device_input_) cudaFree(device_input_);
  }

  int getBatchSize() const noexcept override { return batch_size_; }

  bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
    if (current_batch_ >= calibration_data_.size()) return false;

    // 将当前 batch 数据拷贝到 GPU
    cudaMemcpy(device_input_, calibration_data_[current_batch_].data(), batch_bytes_, cudaMemcpyHostToDevice);

    // 假设第一个输入是数据输入（根据实际模型调整）
    bindings[0] = device_input_;
    current_batch_++;
    return true;
  }

  const void* readCalibrationCache(size_t& length) noexcept override {
    if (cache_.empty()) return nullptr;
    length = cache_.size();
    return cache_.data();
  }

  void writeCalibrationCache(const void* cache, size_t length) noexcept override {
    cache_.assign(static_cast<const uint8_t*>(cache), static_cast<const uint8_t*>(cache) + length);
    if (!cache_file_.empty()) {
      save_cache();
    }
  }

 private:
  bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
  }

  void load_cache() {
    std::ifstream file(cache_file_, std::ios::binary);
    if (file) {
      cache_.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
      std::cout << "Loaded calibration cache from " << cache_file_.c_str() << std::endl;
    }
  }

  void save_cache() {
    std::ofstream file(cache_file_, std::ios::binary);
    if (file) {
      file.write(reinterpret_cast<const char*>(cache_.data()), cache_.size());
      std::cout << "Saved calibration cache to " << cache_file_.c_str() << std::endl;
    }
  }

  std::vector<std::vector<uint8_t>> calibration_data_;
  std::vector<int>                  input_dims_;
  std::string                       cache_file_;
  size_t                            current_batch_;
  int                               batch_size_;
  size_t                            batch_bytes_;
  void*                             device_input_ = nullptr;
  std::vector<uint8_t>              cache_;
};

// ==================== 模型源和输出实现 ====================

ModelSource::ModelSource(const char* onnxmodel) : type_(ModelSourceType::ONNX), onnxmodel_(onnxmodel) {}

ModelSource::ModelSource(const std::string& onnxmodel) : type_(ModelSourceType::ONNX), onnxmodel_(onnxmodel) {}

ModelSource::ModelSource(const void* data, size_t size)
    : type_(ModelSourceType::ONNXDATA), onnx_data_(data), onnx_data_size_(size), onnxmodel_("(memory)") {}

const void*     ModelSource::onnx_data() const { return onnx_data_; }
size_t          ModelSource::onnx_data_size() const { return onnx_data_size_; }
std::string     ModelSource::onnxmodel() const { return onnxmodel_; }
ModelSourceType ModelSource::type() const { return type_; }

std::string ModelSource::descript() const {
  if (type_ == ModelSourceType::ONNX) {
    return "ONNX Model '" + onnxmodel_ + "'";
  } else if (type_ == ModelSourceType::ONNXDATA) {
    std::ostringstream oss;
    oss << "ONNX Data [" << onnx_data_ << ", " << onnx_data_size_ << " bytes]";
    return oss.str();
  }
  return "Unknown source type";
}

CompileOutput::CompileOutput(CompileOutputType type) : type_(type) {}
CompileOutput::CompileOutput(const std::string& file) : type_(CompileOutputType::File), file_(file) {}
CompileOutput::CompileOutput(const char* file) : type_(CompileOutputType::File), file_(file) {}

void CompileOutput::set_data(const std::vector<uint8_t>& data) { data_ = data; }
void CompileOutput::set_data(std::vector<uint8_t>&& data) { data_ = std::move(data); }


/**
 * 将 ONNX 量化后的模型编译为 TensorRT 引擎
 * @param mode 编译模式 (FP32, FP16, INT8)
 * @param source 模型源 (ONNX 文件路径或内存数据)
 * @param saveto 输出配置 (文件路径或内存指针)
 * @param config 编译配置
 * @return 是否编译成功
 */
bool compile(
    Mode mode,
    const ModelSource& source, 
    const CompileOutput& saveto,
    const CompileConfig& config) {
  
  std::cout << "Compiling " << mode_string(mode) << " | " << source.descript().c_str() << std::endl;

  // 1. 创建 Builder
  std::shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroy_trt_pointer<IBuilder>);
  if (!builder) {
    std::cerr << "Failed to create TensorRT builder" << std::endl;
    return false;
  }

  // 2. 创建 Network (显式 batch 模式支持动态 shape)
  // deprecated
  // const uint32_t explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  if (!config.strict_qdq) {
    std::cerr << "Warning: Strict QDQ mode is disabled. Not supported." << std::endl;
    return false;
  } 
  const uint32_t strongly_typed_ = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  std::shared_ptr<INetworkDefinition> network(builder->createNetworkV2(strongly_typed_),
                                              destroy_trt_pointer<INetworkDefinition>);

  if (!network) {
    std::cerr << "Failed to create network" << std::endl;
    return false;
  }

  // 3. 创建 Parser 并解析 ONNX
  std::shared_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger),
                                                destroy_trt_pointer<nvonnxparser::IParser>);
  if (!parser) {
    std::cerr << "Failed to create ONNX parser" << std::endl;
    return false;
  }

  bool parsed = false;
  if (source.type() == ModelSourceType::ONNX) {
    parsed = parser->parseFromFile(source.onnxmodel().c_str(), static_cast<int>(ILogger::Severity::kWARNING));
  } else {
    parsed = parser->parse(source.onnx_data(), source.onnx_data_size());
  }

  if (!parsed) {
    int num_errors = parser->getNbErrors();
    for (int i = 0; i < num_errors; ++i) {
      auto* error = parser->getError(i);
      std::cerr << "ONNX Parse Error [" << i << "]: " << error->desc() << " (code: " << static_cast<int>(error->code()) << ")" << std::endl;
    }
    return false;
  }

  // 4. 打印模型信息
  std::cout << "========== Model Information ==========" << std::endl;

  int num_inputs = network->getNbInputs();
  int num_outputs = network->getNbOutputs();

  std::cout << "Inputs: " << num_inputs << ", Outputs: " << num_outputs << std::endl;

  std::vector<nvinfer1::Dims> input_dims_list;
  for (int i = 0; i < num_inputs; ++i) {
    auto* tensor = network->getInput(i);
    auto  dims = tensor->getDimensions();
    input_dims_list.push_back(dims);

    std::string dims_str;
    for (int j = 0; j < dims.nbDims; ++j) {
      dims_str += std::to_string(dims.d[j]);
      if (j < dims.nbDims - 1) dims_str += "x";
    }
    std::cout << "  Input[" << i << "] '" << tensor->getName() << "': " << dims_str.c_str() << " [dtype=" << static_cast<int>(tensor->getType()) << "]" << std::endl;
  }

  for (int i = 0; i < num_outputs; ++i) {
    auto* tensor = network->getOutput(i);
    auto  dims = tensor->getDimensions();
    std::string dims_str;
    for (int j = 0; j < dims.nbDims; ++j) {
      dims_str += std::to_string(dims.d[j]);
      if (j < dims.nbDims - 1) dims_str += "x";
    }
    std::cout << "  Output[" << i << "] '" << tensor->getName() << "': " << dims_str.c_str() << " [dtype=" << static_cast<int>(tensor->getType()) << "]" << std::endl;
  }
  std::cout << "=======================================" << std::endl;

  // 5. 创建 Builder Config
  std::shared_ptr<IBuilderConfig> builder_config(builder->createBuilderConfig(),
                                                 destroy_trt_pointer<IBuilderConfig>);
  if (!builder_config) {
    std::cerr << "Failed to create builder config" << std::endl;
    return false;
  }

  // 6. 设置工作空间内存池
  size_t workspace_size = config.max_workspace_size > 0 ? config.max_workspace_size : (2ULL << 30);  // 默认 2GB
  builder_config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspace_size);
  std::cout << "Workspace limit: " << workspace_size / 1024.0 / 1024.0 << " MB" << std::endl;

  // 7. 设置精度模式

  // BuilderFlag: FP16 INT8 deprecated, use strongly-typed mode instead
  // 
  // if (mode == Mode::FP16) {
  //   builder_config->setFlag(BuilderFlag::kFP16);
  //   std::cout << "Enabled FP16 mode" << std::endl;
  // } else if (mode == Mode::INT8) {
  //   // 关键：INT8 模式设置
  //   builder_config->setFlag(BuilderFlag::kINT8);
  //   std::cout << "Enabled INT8 mode" << std::endl;
  // }

  // GPU 回退（如果某些层不支持目标精度，回退到 FP32）
  builder_config->setFlag(BuilderFlag::kGPU_FALLBACK);

  // 8. 处理动态 Shape 和 Optimization Profile
  if (config.dynamic_batch) {
    auto* profile = builder->createOptimizationProfile();
    if (!profile) {
      std::cerr << "Failed to create optimization profile" << std::endl;
      return false;
    }

    for (int i = 0; i < num_inputs; ++i) {
      auto*       input = network->getInput(i);
      auto        dims = input->getDimensions();
      const char* name = input->getName();

      nvinfer1::Dims min_dims = dims;
      nvinfer1::Dims opt_dims = dims;
      nvinfer1::Dims max_dims = dims;

      if (config.dynamic_batch) {
        min_dims.d[0] = 1;
        opt_dims.d[0] = std::max(1, config.opt_batch_size);
        max_dims.d[0] = config.max_batch_size;

        std::cout << "Dynamic batch for '" << name << "': min=" << min_dims.d[0] << ", opt=" << opt_dims.d[0] << ", max=" << max_dims.d[0] << std::endl;
      }

      if (!config.profile_shapes.empty() && config.profile_shapes.count(name) > 0) {
        const auto& shape_cfg = config.profile_shapes.at(name);
        min_dims = shape_cfg.min;
        opt_dims = shape_cfg.opt;
        max_dims = shape_cfg.max;
      }

      profile->setDimensions(name, OptProfileSelector::kMIN, min_dims);
      profile->setDimensions(name, OptProfileSelector::kOPT, opt_dims);
      profile->setDimensions(name, OptProfileSelector::kMAX, max_dims);
    }

    if (!builder_config->addOptimizationProfile(profile)) {
      std::cerr << "Failed to add optimization profile" << std::endl;
      return false;
    }
  }

  // 10. 构建引擎
  std::cout << "Building TensorRT engine (this may take a while)..." << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();

  std::shared_ptr<ICudaEngine> engine(builder->buildEngineWithConfig(*network, *builder_config),
                                      destroy_trt_pointer<ICudaEngine>);
  if (!engine) {
    std::cerr << "Engine build failed! Check the logs above for details." << std::endl;
    return false;
  }
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
  std::cout << "Engine built successfully in " << duration << " ms" << std::endl;

  // 11. 序列化并保存
  std::shared_ptr<IHostMemory> serialized(engine->serialize(), destroy_trt_pointer<IHostMemory>);
  if (!serialized || serialized->size() == 0) {
    std::cerr << "Engine serialization failed" << std::endl;
    return false;
  }
  std::cout << "Serialized engine size: " << serialized->size() / 1024.0 / 1024.0 << " MB" << std::endl;

  if (saveto.type_ == CompileOutputType::File) {
    std::ofstream file(saveto.file_, std::ios::binary);
    if (!file) {
      std::cerr << "Failed to open output file: " << saveto.file_.c_str() << std::endl;
      return false;
    }
    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    std::cout << "Engine saved to: " << saveto.file_.c_str() << std::endl;
  } else {
    const_cast<CompileOutput&>(saveto).set_data(
        std::vector<uint8_t>(static_cast<const uint8_t*>(serialized->data()),
                             static_cast<const uint8_t*>(serialized->data()) + serialized->size()));
  }
  return true;
}

}  // namespace TRT

int main(int argc, char* argv[]) {

  TRT::CompileConfig config;
  config.dynamic_batch = false;
  config.max_batch_size = 16;
  config.opt_batch_size = 4;
  config.strict_qdq = true;  // 关键：启用强类型模式

  std::string onnx_path = "onnx_model/yolov8s_tracing_static_b4_quant_fix.onnx";
  std::string out_engine_path = "onnx_model/yolov8s_tracing_static_b4_quant.engine";
  TRT::compile(TRT::Mode::INT8, TRT::ModelSource(onnx_path), TRT::CompileOutput(out_engine_path), config);

  return 0;
}