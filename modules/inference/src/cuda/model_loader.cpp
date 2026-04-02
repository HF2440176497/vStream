
#include "tensor.hpp"
#include "model_loader.hpp"


static bool RegisterModelLoader() {
  auto& factory = ModelLoaderFactory::Instance();
  bool result = true;
  result &= factory.RegisterModelLoaderCreator(DevType::CUDA, 
    [](int dev_id) {
      return std::make_unique<TrtModelLoader>(dev_id);
    });
  return result;
}

static bool model_loader_registered = RegisterModelLoader();

static std::vector<int> dims_to_vector(const nvinfer1::Dims& dims) {
  std::vector<int> shape(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    shape[i] = dims.d[i];
  }
  return shape;
}


static DataType trt_dtype_to_tensor_dtype(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return DataType::FLOAT32;
    case nvinfer1::DataType::kHALF:
      return DataType::FLOAT16;
    case nvinfer1::DataType::kUINT8:
      return DataType::UINT8;
    case nvinfer1::DataType::kINT8:
      return DataType::INT8;
    case nvinfer1::DataType::kINT32:
      return DataType::INT32;
    default:
      return DataType::UNKNOWN;
  }
}


static TensorFormat trt_format_to_tensor_format(nvinfer1::TensorFormat format) {
  switch (format) {
    case nvinfer1::TensorFormat::kLINEAR:
      return TensorFormat::LINEAR;
    case nvinfer1::TensorFormat::kCHW2:
      return TensorFormat::CHW2;
    case nvinfer1::TensorFormat::kCHW4:
      return TensorFormat::CHW4;
    case nvinfer1::TensorFormat::kCHW32:
      return TensorFormat::CHW32;
    case nvinfer1::TensorFormat::kHWC8:
      return TensorFormat::HWC8;
    default:
      return TensorFormat::INVALID;
  }
}

void TrtModelLoader::Logger::log(nvinfer1::Severity severity, const char* msg) noexcept {
  switch (severity) {
    case nvinfer1::Severity::kINTERNAL_ERROR:
      std::cerr << "[TRT][INTERNAL_ERROR] " << msg << std::endl;
      break;
    case nvinfer1::Severity::kERROR:
      std::cerr << "[TRT][ERROR] " << msg << std::endl;
      break;
    case nvinfer1::Severity::kWARNING:
      std::cout << "[TRT][WARNING] " << msg << std::endl;
      break;
    case nvinfer1::Severity::kINFO:
      std::cout << "[TRT][INFO] " << msg << std::endl;
      break;
    case nvinfer1::Severity::kVERBOSE:
      break;
    default:
      break;
  }
}


TrtModelLoader::TrtModelLoader(int dev_id) : ModelLoader(DevType::CUDA, dev_id) {
  cudaSetDevice(dev_id_);
}

bool TrtModelLoader::Init(const std::string& engine_path) {
  if (engine_path.empty()) {
    LOGF(MODEL) << "Empty engine path";
    return false;
  }
  return LoadEngine(engine_path);
}

TrtModelLoader::~TrtModelLoader() {
  // "destroy" has deprecated in TensorRT 8.0
  if (context_) {
    context_.reset();
  }
  if (engine_) {
    engine_.reset();
  }
  if (runtime_) {
    runtime_.reset();
  }
}

bool TrtModelLoader::LoadEngine(const std::string& engine_path) {
  auto model_data = utils::load_model(engine_path);
  if (model_data.empty()) {
    LOGF(MODEL) << "Failed to load model file: " << engine_path;
    return false;
  }
  runtime_ = std::unique_ptr<IRuntime, decltype(trt_deleter)>(createInferRuntime(logger_), trt_deleter);
  if (runtime_ == nullptr) {
    LOGF(MODEL) << "Failed to create TensorRT runtime";
    return false;
  }

  // before release runtime_, all ICudaEngine instance should be destroyed
  engine_ = std::unique_ptr<ICudaEngine, decltype(trt_deleter)>(runtime_->deserializeCudaEngine(model_data.data(), model_data.size()),
                                                                trt_deleter);
  if (engine_ == nullptr) {
    LOGF(MODEL) << "Failed to deserialize TensorRT engine";
    return false;
  }

  context_ = std::unique_ptr<IExecutionContext, decltype(trt_deleter)>(engine_->createExecutionContext(), trt_deleter);
  if (context_ == nullptr) {
    LOGF(MODEL) << "Failed to create TensorRT execution context";
    return false;
  }
  if (!ParseBindings()) {
    return false;
  }
  engine_path_ = engine_path;
  return true;
}

bool TrtModelLoader::ParseBindings() {
  input_shapes_.clear();
  output_shapes_.clear();
  input_data_types_.clear();
  output_data_types_.clear();
  input_names_.clear();
  output_names_.clear();
  bind_name_index_map_.clear();

  auto bind_num = engine_->getNbIOTensors();

  if (bind_num <= 2) {
    LOGE(MODEL) << "Model with tensor num: " << bind_num << " is not supported";
    return false;
  }

  for (int i = 0; i < bind_num; ++i) {
    auto const bind_name = engine_->getIOTensorName(i);
    nvinfer1::DataType dtype = engine_->getTensorDataType(bind_name);
    DataType data_type = trt_dtype_to_tensor_dtype(dtype);
    if (data_type != DataType::FLOAT32) {
      LOGE(MODEL) << "Unsupported data type: " << dtype << " for tensor: " << bind_name;
      continue;
    }
    nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(bind_name);
    if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
      input_names_.push_back(bind_name);
      input_data_types_.push_back(data_type);
    } else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_names_.push_back(bind_name);
      output_data_types_.push_back(data_type);
    } else {
      LOGW("WARNING: Unsupport IO mode: %d for tensor: %s", io_mode, bind_name);
      continue;
    }
    auto trt_format = engine_->getTensorFormat(bind_name);
    auto format = trt_format_to_tensor_format(trt_format);
    if (format != TensorFormat::LINEAR) {
      LOGE(MODEL) << "Unsupported format: " << trt_format << " for tensor: " << bind_name;
      continue;
    }
    bind_name_index_map_[bind_name] = i;
  }

  // TODO: 目前只支持单个输入输出
  if (input_names_.size() > 1) {
      LOGW("WARNING: Model should has one input %d", input_names_.size());
      input_name_ = input_names_[input_ordered_index_];
      int index_ = bind_name_index_map_[input_name_];
      if (index_ != input_ordered_index_) {
        LOGF(MODEL) << "input_index_ not match bind_name: " << input_name_ << " actual_index: " << index_ << std::endl;
        return false;
      }
  }

  if (output_names_.size() > 1) {
      LOGW("WARNING: Model should has one output %d", output_names_.size());
      output_name_ = output_names_[output_ordered_index_];
      int index_ = bind_name_index_map_[output_name_];
      if (index_ != output_ordered_index_) {
        LOGF(MODEL) << "output_index_ not match bind_name: " << output_name_ << " actual_index: " << index_ << std::endl;
        return false;
      }
  }

  int input_num = 0;
  for (auto& input_name : input_names_) {
    nvinfer1::Dims opt_dims;
    auto dims = engine_->getTensorShape(input_name.c_str());
    LOGI(MODEL) << "input_name [" << input_num++ << "]: " << input_name << "; dims: " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << "x" << dims.d[3];
    
    // for dynamic input, get opt shape
    if (dims.d[0] == -1) {
      auto opt_profile_index = context_->getOptimizationProfile();
      opt_dims = engine_->getProfileShape(input_name.c_str(), 
                                          opt_profile_index,
                                          nvinfer1::OptProfileSelector::kOPT);
      context_->setInputTensorShape(input_name.c_str(), opt_dims);
    } else {  // static shape
      opt_dims = dims;
    }
    TensorShape input_shape(dims_to_vector(opt_dims));
    input_shapes_.push_back(input_shape);  // 对应 input_names_ 顺序
  }  // end of input_names_

  int output_num = 0;
  for (auto& output_name : output_names_) {
    auto dims = engine_->getTensorShape(output_name.c_str());
    if (dims.d[0] == -1) {
      LOGF(MODEL) << "Model with dynamic output, index: " << output_num << " name: " << output_name;
      return false;
    }
    LOGI(MODEL) << "output_name [" << output_num++ << "]: " << output_name << "; dims: " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << "x" << dims.d[3];
    TensorShape output_shape(dims_to_vector(dims));
    output_shapes_.push_back(output_shape);  // 对应 output_names_ 顺序
  }  // end of output_names_
  return true;
}  // end of ParseBindings

/**
 * @brief 运行模型推理
 * @note inputs outputs size == tensor num
 * 在解析模型阶段，需要确保 tensor shape 已设置
 */
bool TrtModelLoader::RunSync(std::vector<std::shared_ptr<void>> inputs, std::vector<std::shared_ptr<void>> outputs) {
  for (int i = 0; i < inputs.size(); ++i) {
    context_->setInputTensorAddress(input_names_[i].c_str(), inputs[i].get());
  }
  for (int i = 0; i < outputs.size(); ++i) {
    context_->setOutputTensorAddress(output_names_[i].c_str(), outputs[i].get());
  }
  bool execute_result = context_->enqueueV3(stream_);
  if (!execute_result) {
    auto code = cudaGetLastError();
    LOGF(MODEL) << "execute fail, code: " << code << ", message: " << cudaGetErrorName(code) << ", " << cudaGetErrorString(code);
    return false;
  }
  CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  return true;
}

nvinfer1::IExecutionContext* TrtModelLoader::CreateExecutionContext() {
  if (!engine_) return nullptr;
  return engine_->createExecutionContext();
}
