

#include "trans_model.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <numeric>
#include <sstream>

using namespace nvinfer1;

#define CHECK_CUDA_RUNTIME(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

inline bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
  if (code != cudaSuccess) {
    const char* err_name = cudaGetErrorName(code);
    const char* err_message = cudaGetErrorString(code);
    printf("check_cuda_runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", 
		file, line, op, err_name, err_message);
    return false;
  }
  return true;
}

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
 * @param mode 编译模式 (FP32, FP16, INT8)
 * @param source 模型源 (ONNX 文件路径或内存数据)
 * @param saveto 输出配置 (文件路径或内存指针)
 * @param config 编译配置
 * @return 是否编译成功
 */
bool compile(const ModelSource& source, const CompileOutput& saveto,
             const CompileConfig& config) {

  std::shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroy_trt_pointer<IBuilder>);
  if (!builder) {
    std::cerr << "Failed to create TensorRT builder" << std::endl;
    return false;
  }

  uint32_t network_flags = 0;
  if (config.strict_qdq) {
    network_flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    std::cout << "QDQ strict mode: kSTRONGLY_TYPED enabled" << std::endl;
  }
  std::shared_ptr<INetworkDefinition> network(builder->createNetworkV2(network_flags),
                                              destroy_trt_pointer<INetworkDefinition>);
  if (!network) {
    std::cerr << "Failed to create network" << std::endl;
    return false;
  }

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

  std::cout << "========== Model Information ==========" << std::endl;

  int num_inputs = network->getNbInputs();
  int num_outputs = network->getNbOutputs();
  std::cout << "Inputs: " << num_inputs << ", Outputs: " << num_outputs << std::endl;

  bool has_dynamic_shape = false;
  std::vector<nvinfer1::Dims> input_dims_list;
  for (int i = 0; i < num_inputs; ++i) {
    auto* tensor = network->getInput(i);
    auto  dims = tensor->getDimensions();
    input_dims_list.push_back(dims);

    std::string dims_str;
    for (int j = 0; j < dims.nbDims; ++j) {
      dims_str += std::to_string(dims.d[j]);
      if (j < dims.nbDims - 1) dims_str += "x";
      if (dims.d[j] == -1) has_dynamic_shape = true;
    }
    std::cout << "  Input[" << i << "] '" << tensor->getName() << "': " << dims_str.c_str()
              << " [dtype=" << static_cast<int>(tensor->getType()) << "]" << std::endl;
  }

  for (int i = 0; i < num_outputs; ++i) {
    auto* tensor = network->getOutput(i);
    auto  dims = tensor->getDimensions();
    std::string dims_str;
    for (int j = 0; j < dims.nbDims; ++j) {
      dims_str += std::to_string(dims.d[j]);
      if (j < dims.nbDims - 1) dims_str += "x";
    }
    std::cout << "  Output[" << i << "] '" << tensor->getName() << "': " << dims_str.c_str()
              << " [dtype=" << static_cast<int>(tensor->getType()) << "]" << std::endl;
  }
  std::cout << "Dynamic shape: " << (has_dynamic_shape ? "YES" : "NO (static)") << std::endl;
  std::cout << "=======================================" << std::endl;

  std::shared_ptr<IBuilderConfig> builder_config(builder->createBuilderConfig(),
                                                 destroy_trt_pointer<IBuilderConfig>);
  if (!builder_config) {
    std::cerr << "Failed to create builder config" << std::endl;
    return false;
  }

  size_t workspace_size = config.max_workspace_size > 0 ? config.max_workspace_size : (2ULL << 30);
  builder_config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspace_size);
  std::cout << "Workspace limit: " << workspace_size / 1024.0 / 1024.0 << " MB" << std::endl;

  if (has_dynamic_shape) {
    if (!config.dynamic_batch) {
      std::cerr << "ERROR: Model has dynamic dimensions but dynamic_batch is disabled. "
                << "Please set dynamic_batch=true in CompileConfig." << std::endl;
      return false;
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
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

      if (config.profile_shapes.count(name) > 0) {
        const auto& shape_cfg = config.profile_shapes.at(name);
        min_dims = shape_cfg.min;
        opt_dims = shape_cfg.opt;
        max_dims = shape_cfg.max;
      } else {
        for (int j = 0; j < dims.nbDims; ++j) {
          if (dims.d[j] == -1) {
            min_dims.d[j] = config.min_batch_size;
            opt_dims.d[j] = config.opt_batch_size;
            max_dims.d[j] = config.max_batch_size;
          }
        }
      }

      std::string min_str, opt_str, max_str;
      for (int j = 0; j < min_dims.nbDims; ++j) {
        min_str += std::to_string(min_dims.d[j]) + (j < min_dims.nbDims - 1 ? "x" : "");
        opt_str += std::to_string(opt_dims.d[j]) + (j < opt_dims.nbDims - 1 ? "x" : "");
        max_str += std::to_string(max_dims.d[j]) + (j < max_dims.nbDims - 1 ? "x" : "");
      }
      std::cout << "  Profile '" << name << "': min=" << min_str << " opt=" << opt_str << " max=" << max_str << std::endl;

      profile->setDimensions(name, OptProfileSelector::kMIN, min_dims);
      profile->setDimensions(name, OptProfileSelector::kOPT, opt_dims);
      profile->setDimensions(name, OptProfileSelector::kMAX, max_dims);
    }

    if (!builder_config->addOptimizationProfile(profile)) {
      std::cerr << "Failed to add optimization profile" << std::endl;
      return false;
    }

  }  // end if (has_dynamic_shape) 


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
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <onnx_path> <out_engine_path>" << std::endl;
    std::cerr << "  Example: " << argv[0] << " model.onnx model.engine" << std::endl;
    return 1;
  }

  std::string onnx_path = argv[1];
  std::string out_engine_path = argv[2];

  TRT::CompileConfig config;
  config.dynamic_batch = true;
  config.max_batch_size = 8;
  config.opt_batch_size = 4;
  config.min_batch_size = 1;

  config.strict_qdq = true;
  TRT::compile(TRT::ModelSource(onnx_path), TRT::CompileOutput(out_engine_path), config);

  return 0;
}