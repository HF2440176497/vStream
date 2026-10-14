
#include "tensor.hpp"
#include "cuda/model_loader_trt.hpp"

#include "common.hpp"
#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_allocator_cuda.hpp"
#include "cuda/cnstream_cuda_env.hpp"
#include "cuda/cnstream_cuda_init.hpp"

namespace cnstream {

static bool RegisterModelLoader() {
  auto& factory = ModelLoaderFactory::Instance();
  bool result = true;
  result &= factory.RegisterModelLoaderCreator(DevType::CUDA, 
    [](int device_id) {
      return std::make_unique<ModelLoaderTrt>(device_id);
    });
  return result;
}

static bool model_loader_registered = RegisterModelLoader();

static std::vector<int> dims_to_vector(const nvinfer1::Dims& dims) {
  std::vector<int> shape;
  // 防御: 非法 Dims（如 getProfileShape 对非输入张量返回的 Dims{-1, {}}）返回空 vector，
  // 避免 std::vector<int>(-1) 异常
  if (dims.nbDims <= 0) {
    return shape;
  }
  shape.resize(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    shape[i] = dims.d[i];
  }
  return shape;
}

// 是否含未解析的动态维度(-1)
static bool has_dynamic_dims(const nvinfer1::Dims& dims) {
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] == -1) return true;
  }
  return false;
}

// 维度是否已全部解析且为正
static bool is_static_shape(const nvinfer1::Dims& dims) {
  if (dims.nbDims <= 0) return false;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) return false;
  }
  return true;
}

// TensorShape -> nvinfer1::Dims；要求各维均已解析且为正
static bool tensor_shape_to_dims(const TensorShape& s, nvinfer1::Dims* out) {
  if (!out) return false;
  int ndims = s.ndims();
  if (ndims <= 0 || ndims > nvinfer1::Dims::MAX_DIMS) return false;
  out->nbDims = ndims;
  for (int i = 0; i < ndims; ++i) {
    int d = s.shape(i);
    if (d <= 0) return false;
    out->d[i] = d;
  }
  return true;
}

static std::string dims_to_str(const nvinfer1::Dims& dims) {
  if (dims.nbDims <= 0) return "(invalid)";
  std::string result = "(";
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      result += "?";
    } else {
      result += std::to_string(dims.d[i]);
    }
    if (i < dims.nbDims - 1) result += ", ";
  }
  result += ")";
  return result;
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
      return DataType::INVALID;
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

static std::string trt_dtype_to_str(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return "FLOAT32";
    case nvinfer1::DataType::kHALF:
      return "FLOAT16";
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    default:
      return "INVALID";
  }
}

static std::string trt_format_to_str(nvinfer1::TensorFormat format) {
  switch (format) {
    case nvinfer1::TensorFormat::kLINEAR:
      return "LINEAR";
    case nvinfer1::TensorFormat::kCHW2:
      return "CHW2";
    case nvinfer1::TensorFormat::kCHW4:
      return "CHW4";
    case nvinfer1::TensorFormat::kCHW32:
      return "CHW32";
    case nvinfer1::TensorFormat::kHWC8:
      return "HWC8";
    default:
      return "INVALID";
  }
}

static std::string trt_io_mode_to_str(nvinfer1::TensorIOMode io_mode) {
  switch (io_mode) {
    case nvinfer1::TensorIOMode::kINPUT:
      return "INPUT";
    case nvinfer1::TensorIOMode::kOUTPUT:
      return "OUTPUT";
    default:
      return "INVALID";
  }
}


void ModelLoaderTrt::Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
  switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
      std::cerr << "[TRT][INTERNAL_ERROR] " << msg << std::endl;
      break;
    case nvinfer1::ILogger::Severity::kERROR:
      std::cerr << "[TRT][ERROR] " << msg << std::endl;
      break;
    case nvinfer1::ILogger::Severity::kWARNING:
      std::cout << "[TRT][WARNING] " << msg << std::endl;
      break;
    case nvinfer1::ILogger::Severity::kINFO:
      std::cout << "[TRT][INFO] " << msg << std::endl;
      break;
    case nvinfer1::ILogger::Severity::kVERBOSE:
      std::cout << "[TRT][VERBOSE] " << msg << std::endl;
      break;
    default:
      break;
  }
}


ModelLoaderTrt::ModelLoaderTrt(int device_id) : ModelLoader(device_id) {
  device_type_ = DevType::CUDA;
  CudaInitResult device_ret = RunCudaInit(
      "set_device",
      [this]() -> int { return static_cast<int>(cudaSetDevice(device_id_)); },
      ClassifyCudaRuntimeError);
  if (!device_ret.ok) {
    LOGE(MODEL) << "cudaSetDevice failed on device " << device_id_ << "["
                << device_ret.attempt << " attempt(s), " << device_ret.elapsed_ms
                << " ms]: " << CudaInitErrorName(device_ret.last_error);
  }
  CudaInitResult stream_ret = RunCudaInit(
      "stream_create",
      [this]() -> int {
        return static_cast<int>(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_)));
      },
      ClassifyCudaRuntimeError);
  if (!stream_ret.ok) {
    LOGE(MODEL) << "cudaStreamCreate failed on device " << device_id_ << " after "
                << stream_ret.attempt << " attempt(s), " << stream_ret.elapsed_ms << " ms: "
                << CudaInitErrorName(stream_ret.last_error);
  }
}

bool ModelLoaderTrt::Init(const std::string& engine_path, const InferParams& params) {
  if (engine_path.empty()) {
    LOGE(MODEL) << "Empty engine path";
    return false;
  }
  if (stream_ == nullptr) {
    LOGE(MODEL) << "stream is unavailable (cudaStreamCreate failed),"
                << " skip load: " << engine_path;
    return false;
  }
  // Set input ordered index
  SetInputOrderedIndex(params.input_ordered_index);
  shape_mode_ = params.shape_mode;
  return LoadEngine(engine_path);
}

ModelLoaderTrt::~ModelLoaderTrt() {
  CudaDeviceGuard guard(device_id_);
  if (stream_) {
    CHECK_CUDA_RUNTIME(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }
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

bool ModelLoaderTrt::LoadEngine(const std::string& engine_path) {
  CudaDeviceGuard guard(device_id_);
  if (!ProbeCudaDevice(device_id_)) {
    LOGE(MODEL) << "CUDA environment probe failed on device " << device_id_
                << ", abort loading TensorRT engine: " << engine_path;
    return false;
  }
  auto model_data = utils::load_model(engine_path);
  if (model_data.empty()) {
    LOGE(MODEL) << "Failed to load model file: " << engine_path;
    return false;
  }
  nvinfer1::IRuntime* runtime = nullptr;
  CudaInitResult runtime_ret = RunCudaInit(
      "runtime",
      [this, &runtime]() -> int {
        runtime = nvinfer1::createInferRuntime(logger_);
        return runtime != nullptr ? 0 : kCudaInitNoCode;
      },
      nullptr);
  runtime_ = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>(runtime);
  if (runtime_ == nullptr) {
    // 模型加载失败只上报、返回 false
    LOGE(MODEL) << "Failed to create TensorRT runtime [" << runtime_ret.attempt
                << " attempt(s), " << runtime_ret.elapsed_ms << " ms]";
    return false;
  }

  nvinfer1::ICudaEngine* engine = nullptr;
  CudaInitResult engine_ret = RunCudaInit(
      "engine",
      [this, &engine, &model_data]() -> int {
        engine = runtime_->deserializeCudaEngine(model_data.data(), model_data.size());
        return engine != nullptr ? 0 : kCudaInitNoCode;
      },
      nullptr);
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>(engine);
  if (engine_ == nullptr) {
    LOGE(MODEL) << "Failed to deserialize TensorRT engine [" << engine_ret.attempt
                << " attempt(s), " << engine_ret.elapsed_ms << " ms]";
    return false;
  }

  context_ = CreateExecutionContext();
  if (context_ == nullptr) {
    LOGE(MODEL) << "Failed to create TensorRT execution context";
    return false;
  }
  if (!ParseBindings()) {
    return false;
  }
  // 初始化阶段专用 context 到此结束；运行期各流水线独占持有自己的执行上下文
  context_.reset();
  name_ = utils::get_filename_without_ext(engine_path);
  engine_path_ = engine_path;
  // 形状能力（min/max/dynamic）在此定型并缓存
  FillRunShapeCaps(&caps_);
  caps_init_ = true;
  return true;
}

bool ModelLoaderTrt::ParseBindings() {
  input_shapes_.clear();
  output_shapes_.clear();
  input_shapes_min_.clear();
  input_shapes_max_.clear();
  output_shapes_max_.clear();
  input_data_types_.clear();
  output_data_types_.clear();
  input_names_.clear();
  output_names_.clear();
  bind_name_index_map_.clear();

  auto bind_num = engine_->getNbIOTensors();

  if (bind_num < 2) {
    LOGE(MODEL) << "Model with tensor num: " << bind_num << " is not supported";
    return false;
  }

  for (int i = 0; i < bind_num; ++i) {
    auto const bind_name = engine_->getIOTensorName(i);
    nvinfer1::DataType dtype = engine_->getTensorDataType(bind_name);
    DataType data_type = trt_dtype_to_tensor_dtype(dtype);
    if (data_type != DataType::FLOAT32) {
      LOGE(MODEL) << "Unsupported data type: " << trt_dtype_to_str(dtype) << " for tensor: " << bind_name;
      return false;
    }
    auto trt_format = engine_->getTensorFormat(bind_name);
    auto format = trt_format_to_tensor_format(trt_format);
    if (format != TensorFormat::LINEAR) {
      LOGE(MODEL) << "Unsupported format: " << trt_format_to_str(trt_format) << " for tensor: " << bind_name;
      return false;
    }
    nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(bind_name);
    if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
      input_names_.push_back(bind_name);
      input_data_types_.push_back(data_type);
    } else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_names_.push_back(bind_name);
      output_data_types_.push_back(data_type);
    } else {
      LOGW(MODEL) << "Unsupported IO mode: " << trt_io_mode_to_str(io_mode) << " for tensor: " << bind_name;
      continue;
    }
    bind_name_index_map_[bind_name] = i;
  }

  if (input_names_.size() > 1) {
      LOGW(MODEL) << "Model with " << input_names_.size() << " inputs, choose input index: " << input_ordered_index_;
  }
  if (input_ordered_index_ < 0 || input_ordered_index_ >= static_cast<int>(input_names_.size())) {
    LOGE(MODEL) << "input_ordered_index out of range: " << input_ordered_index_
                << ", input count: " << input_names_.size();
    return false;
  }
  input_name_ = input_names_[input_ordered_index_];

  int input_num = 0;
  bool has_dynamic_input = false;
  for (auto& input_name : input_names_) {
    nvinfer1::Dims dims = engine_->getTensorShape(input_name.c_str());

    if (has_dynamic_dims(dims)) {
      has_dynamic_input = true;
      auto opt_profile_index = context_->getOptimizationProfile();
      nvinfer1::Dims min_dims = engine_->getProfileShape(input_name.c_str(),
                                                         opt_profile_index,
                                                         nvinfer1::OptProfileSelector::kMIN);
      nvinfer1::Dims opt_dims = engine_->getProfileShape(input_name.c_str(),
                                                         opt_profile_index,
                                                         nvinfer1::OptProfileSelector::kOPT);
      nvinfer1::Dims max_dims = engine_->getProfileShape(input_name.c_str(),
                                                         opt_profile_index,
                                                         nvinfer1::OptProfileSelector::kMAX);
      if (!is_static_shape(opt_dims)) {
        LOGE(MODEL) << "Input [" << input_name << "] dims unresolved: " << dims_to_str(opt_dims)
                    << " (engine dims: " << dims_to_str(dims) << ", profile index: " << opt_profile_index << ")";
        return false;
      }
      // kMIN / kMAX 必须同样可解析
      if (!is_static_shape(min_dims) || !is_static_shape(max_dims)) {
        if (shape_mode_ == ShapeMode::DYNAMIC) {
          LOGE(MODEL) << "Input [" << input_name << "] profile min/max unresolved: min "
                      << dims_to_str(min_dims) << ", max " << dims_to_str(max_dims)
                      << "; required by shape_mode=dynamic";
          return false;
        }
        LOGW(MODEL) << "Input [" << input_name << "] profile min/max unresolved: min "
                    << dims_to_str(min_dims) << ", max " << dims_to_str(max_dims)
                    << "; fall back to kOPT for both";
        min_dims = opt_dims;
        max_dims = opt_dims;
      }
      // 运行期动态形状要求 batch(N) 恒为 1：
      // IOResData::batch_offset N>1 时每批形状一变就会错位。这里在加载期直接拒绝
      if (shape_mode_ == ShapeMode::DYNAMIC) {
        if (min_dims.nbDims <= 0 || opt_dims.nbDims <= 0 || max_dims.nbDims <= 0 ||
            min_dims.d[0] != 1 || opt_dims.d[0] != 1 || max_dims.d[0] != 1) {
          LOGE(MODEL) << "DYNAMIC requires batch(N)==1 in the optimization profile, "
                      << "but input '" << input_name << "' has min/opt/max N = "
                      << (min_dims.nbDims > 0 ? min_dims.d[0] : -1) << "/"
                      << (opt_dims.nbDims > 0 ? opt_dims.d[0] : -1) << "/"
                      << (max_dims.nbDims > 0 ? max_dims.d[0] : -1);
          return false;
        }
      }
      if (!context_->setInputShape(input_name.c_str(), opt_dims)) {
        LOGE(MODEL) << "setInputShape failed for input: [" << input_name
                    << "] dims: " << dims_to_str(opt_dims);
        return false;
      }
      dims = opt_dims;
      input_shapes_min_.push_back(TensorShape(dims_to_vector(min_dims)));
      input_shapes_max_.push_back(TensorShape(dims_to_vector(max_dims)));
    } else {
      if (!is_static_shape(dims)) {
        LOGE(MODEL) << "Input [" << input_name << "] has invalid dims: " << dims_to_str(dims);
        return false;
      }
      // 全静态输入：下限/上限就是定形形状，Dynamic 模式下 clamp 会自然退化为常量
      input_shapes_min_.push_back(TensorShape(dims_to_vector(dims)));
      input_shapes_max_.push_back(TensorShape(dims_to_vector(dims)));
    }

    LOGI(MODEL) << "Input [" << input_num++ << "]: " << input_name << "; dims: " << dims_to_str(dims)
                << "; min: " << input_shapes_min_.back().ToString()
                << " max: " << input_shapes_max_.back().ToString();
    input_shapes_.push_back(TensorShape(dims_to_vector(dims)));  // 对应 input_names_ 顺序
  }  // end of input_names_

  if (shape_mode_ == ShapeMode::DYNAMIC && !has_dynamic_input) {
    LOGE(MODEL) << "DYNAMIC but the engine has no dynamic input dims; "
                   "back to static";
    shape_mode_ = ShapeMode::STATIC;
  }

  int output_num = 0;
  for (auto& output_name : output_names_) {
    nvinfer1::Dims dims = engine_->getTensorShape(output_name.c_str());

    if (has_dynamic_dims(dims)) {
      // 形状依赖输入的输出在运行期解析：
      // 此时输入 shape 已按 opt profile 设置，可拿到具体维度；
      // 如果数据相关的输出仍为 -1，应当报错
      dims = context_->getTensorShape(output_name.c_str());
    }
    if (!is_static_shape(dims)) {
      LOGE(MODEL) << "Output [" << output_name << "] dims unresolved: " << dims_to_str(dims)
                  << ". Requires static output shapes";
      return false;
    }

    LOGI(MODEL) << "Output [" << output_num++ << "]: " << output_name << "; dims: " << dims_to_str(dims);
    output_shapes_.push_back(TensorShape(dims_to_vector(dims)));  // 对应 output_names_ 顺序
  }  // end of output_names_

  if (shape_mode_ != ShapeMode::DYNAMIC) {
    // 静态模式：上限即定形形状，缓冲区尺寸与历史一致
    output_shapes_max_ = output_shapes_;
    return true;
  }

  // 动态模式：输出尺寸依赖输入宽度（如 T = W/8），只有把输入设成 MAX
  // 之后才能拿到「最大输出尺寸」。缓冲区必须按它分配，否则会越界写。
  for (size_t i = 0; i < input_names_.size(); ++i) {
    nvinfer1::Dims dims = engine_->getTensorShape(input_names_[i].c_str());
    if (!has_dynamic_dims(dims)) continue;
    nvinfer1::Dims max_dims = engine_->getProfileShape(input_names_[i].c_str(),
                                                       context_->getOptimizationProfile(),
                                                       nvinfer1::OptProfileSelector::kMAX);
    if (!is_static_shape(max_dims)) {
      LOGE(MODEL) << "Input [" << input_names_[i] << "] kMAX dims unresolved: "
                  << dims_to_str(max_dims);
      return false;
    }
    if (!context_->setInputShape(input_names_[i].c_str(), max_dims)) {
      LOGE(MODEL) << "setInputShape(kMAX) failed for input: " << input_names_[i]
                  << ", dims: " << dims_to_str(max_dims);
      return false;
    }
  }

  for (auto& output_name : output_names_) {
    nvinfer1::Dims dims = context_->getTensorShape(output_name.c_str());
    if (!is_static_shape(dims)) {
      LOGE(MODEL) << "Output [" << output_name << "] dims unresolved at kMAX profile: "
                  << dims_to_str(dims);
      return false;
    }
    output_shapes_max_.push_back(TensorShape(dims_to_vector(dims)));
    LOGI(MODEL) << "Output at kMAX: " << output_name << "; dims: " << dims_to_str(dims);
  }

  // 还原到 OPT，避免把探测状态留给后续逻辑（context_ 随后会被释放）
  for (size_t i = 0; i < input_names_.size(); ++i) {
    nvinfer1::Dims dims = engine_->getTensorShape(input_names_[i].c_str());
    if (!has_dynamic_dims(dims)) continue;
    nvinfer1::Dims opt_dims = engine_->getProfileShape(input_names_[i].c_str(),
                                                       context_->getOptimizationProfile(),
                                                       nvinfer1::OptProfileSelector::kOPT);
    if (!is_static_shape(opt_dims) ||
        !context_->setInputShape(input_names_[i].c_str(), opt_dims)) {
      LOGE(MODEL) << "restore input [" << input_names_[i] << "] to kOPT failed";
    }
  }
  return true;
}  // end of ParseBindings

/**
 * @brief 运行模型推理
 * @note inputs outputs size == tensor num
 * 在解析模型阶段，需要确保 tensor shape 已设置
 * @note 无锁：exec_ctx 由调用方独占（见 AcquireExecutionContext），同一流水线的
 *       推理任务已被票据 FIFO 串行化；多流水线共享同一 cudaStream 时，
 *       GPU 执行序 = 入队序，cudaStreamSynchronize 本身线程安全，等待至多
 *       覆盖他人先入队的工作，不影响正确性
 */
bool ModelLoaderTrt::RunSync(void* exec_ctx,
                             std::vector<std::shared_ptr<void>> inputs,
                             std::vector<std::shared_ptr<void>> outputs) {
  CudaDeviceGuard guard(device_id_);
  nvinfer1::IExecutionContext* context = static_cast<nvinfer1::IExecutionContext*>(exec_ctx);
  if (context == nullptr) {
    LOGE(MODEL) << "execution context is null, acquire it via AcquireExecutionContext() first";
    return false;
  }
  if (inputs.size() != input_names_.size() || outputs.size() != output_names_.size()) {
    LOGE(MODEL) << "Tensor count mismatch: inputs " << inputs.size() << " vs " << input_names_.size()
                << ", outputs " << outputs.size() << " vs " << output_names_.size();
    return false;
  }
  for (int i = 0; i < inputs.size(); ++i) {
    context->setInputTensorAddress(input_names_[i].c_str(), inputs[i].get());
  }
  for (int i = 0; i < outputs.size(); ++i) {
    context->setOutputTensorAddress(output_names_[i].c_str(), outputs[i].get());
  }
  bool execute_result = context->enqueueV3(stream_);
  if (!execute_result) {
    auto code = cudaGetLastError();
    // 执行失败是运行期可恢复错误（显存瞬时不足/形状越界等），不能 LOG(FATAL) 直接 abort，
    // 交由调用方按返回值处置
    LOGE(MODEL) << "execute fail, code: " << code << ", message: " << cudaGetErrorName(code) << ", " << cudaGetErrorString(code);
    return false;
  }
  return CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream_));
}

bool ModelLoaderTrt::RunSync(void* exec_ctx,
                             std::vector<std::shared_ptr<void>> inputs,
                             std::vector<std::shared_ptr<void>> outputs,
                             RunShapes* in_shapes,
                             RunShapes* out_shapes) {
  CudaDeviceGuard guard(device_id_);
  nvinfer1::IExecutionContext* context = static_cast<nvinfer1::IExecutionContext*>(exec_ctx);
  if (context == nullptr) {
    LOGE(MODEL) << "execution context is null, acquire it via AcquireExecutionContext() first";
    return false;
  }
  if (inputs.size() != input_names_.size() || outputs.size() != output_names_.size()) {
    LOGE(MODEL) << "Tensor count mismatch: inputs " << inputs.size() << " vs " << input_names_.size()
                << ", outputs " << outputs.size() << " vs " << output_names_.size();
    return false;
  }

  // 动态维度需要两侧契约都在：从输入侧读 actual 设形状，把实测输出写进输出侧
  const bool dynamic = (in_shapes != nullptr) && in_shapes->dynamic && (out_shapes != nullptr);
  if (dynamic && (in_shapes->inputs.size() != inputs.size() ||
                  out_shapes->outputs.size() != outputs.size())) {
    LOGE(MODEL) << "RunShapes size mismatch: shape inputs " << in_shapes->inputs.size() << " vs " << inputs.size()
                << ", shape outputs " << out_shapes->outputs.size() << " vs " << outputs.size();
    return false;
  }

  // 1) 先设形状：setInputShape 须在绑地址之前
  if (dynamic) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      // 越界收敛到 [min, max]，避免 TRT 直接失败；前处理算出的宽度异常时这里兜底
      const TensorShape& want = in_shapes->inputs[i].actual;
      const TensorShape& lo = in_shapes->inputs[i].min;
      const TensorShape& hi = in_shapes->inputs[i].max;
      if (lo.ndims() != want.ndims() || hi.ndims() != want.ndims()) {
        LOGE(MODEL) << "run input shape rank mismatch for '" << input_names_[i] << "': "
                    << want.ToString() << ", min " << lo.ToString() << ", max " << hi.ToString();
        return false;
      }
      std::vector<int> clamped(want.ndims());
      for (int d = 0; d < want.ndims(); ++d) {
        int v = want.shape(d);
        if (v < lo.shape(d)) {
          LOGE(MODEL) << "Input [" << i << "] dim " << d << " = " << v
                      << " below min " << lo.shape(d) << ", clamped";
          v = lo.shape(d);
        } else if (v > hi.shape(d)) {
          LOGE(MODEL) << "Input [" << i << "] dim " << d << " = " << v
                      << " above max " << hi.shape(d) << ", clamped";
          v = hi.shape(d);
        }
        clamped[d] = v;
      }

      nvinfer1::Dims dims;
      if (!tensor_shape_to_dims(TensorShape(clamped), &dims)) {
        LOGE(MODEL) << "Invalid run input shape for '" << input_names_[i] << "': " << want.ToString();
        return false;
      }
      if (!context->setInputShape(input_names_[i].c_str(), dims)) {
        LOGE(MODEL) << "setInputShape failed for input tensor: " << input_names_[i]
                    << ", dims: " << dims_to_str(dims);
        return false;  // 不继续 enqueue，否则会跑在错误形状上
      }
      // 动态维度时，打印实际的输入形状
      LOGI(MODEL) << "setInputShape for input: " << input_names_[i]
                  << ", dims: " << dims_to_str(dims);
    }  // end for inputs
  }

  // 2) 再绑地址
  for (size_t i = 0; i < inputs.size(); ++i) {
    context->setInputTensorAddress(input_names_[i].c_str(), inputs[i].get());
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    context->setOutputTensorAddress(output_names_[i].c_str(), outputs[i].get());
  }

  // 3) 执行
  bool execute_result = context->enqueueV3(stream_);
  if (!execute_result) {
    auto code = cudaGetLastError();
    LOGE(MODEL) << "execute fail, code: " << code << ", message: " << cudaGetErrorName(code) << ", " << cudaGetErrorString(code);
    return false;
  }

  // 4) 把实测输出尺寸回填到输出侧契约（下游 D2H 拷贝量、后处理依赖此处）
  if (dynamic) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      out_shapes->outputs[i].actual =
          TensorShape(dims_to_vector(context->getTensorShape(output_names_[i].c_str())));
      LOGI(MODEL) << "Output [" << i << "] actual: " << out_shapes->outputs[i].actual.ToString();
    }
  }
  return CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream_));
}

void ModelLoaderTrt::FillRunShapeCaps(RunShapes* rs) {
  if (!rs) return;
  rs->dynamic = (shape_mode_ == ShapeMode::DYNAMIC);
  rs->inputs.clear();
  rs->outputs.clear();
  for (uint32_t i = 0; i < InputNum(); ++i) {
    RunShape s;
    s.min = (i < input_shapes_min_.size()) ? input_shapes_min_[i] : InputShape(i);
    s.max = (i < input_shapes_max_.size()) ? input_shapes_max_[i] : InputShape(i);
    s.actual = s.max;  // 不变式：actual 必须初始化，否则未接入的前后处理会拿到空形状
    rs->inputs.push_back(s);
  }
  for (uint32_t i = 0; i < OutputNum(); ++i) {
    RunShape s;
    s.min = OutputShape(i);  // 输出下界不参与决策，给定形值
    s.max = (i < output_shapes_max_.size()) ? output_shapes_max_[i] : OutputShape(i);
    s.actual = s.max;
    rs->outputs.push_back(s);
  }
}

TensorShape ModelLoaderTrt::InputShapeMax(uint32_t index) const {
  return (index < input_shapes_max_.size()) ? input_shapes_max_[index] : InputShape(index);
}

TensorShape ModelLoaderTrt::OutputShapeMax(uint32_t index) const {
  return (index < output_shapes_max_.size()) ? output_shapes_max_[index] : OutputShape(index);
}

void* ModelLoaderTrt::AcquireExecutionContext() {
  std::lock_guard<std::mutex> lock(exec_ctx_mtx_);
  CudaDeviceGuard guard(device_id_);
  return CreateExecutionContext().release();
}

void ModelLoaderTrt::ReleaseExecutionContext(void* exec_ctx) {
  if (exec_ctx == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(exec_ctx_mtx_);
  CudaDeviceGuard guard(device_id_);
  TrtDeleter{}(static_cast<nvinfer1::IExecutionContext*>(exec_ctx));
}

std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> ModelLoaderTrt::CreateExecutionContext() {
  if (!engine_) return nullptr;
  nvinfer1::IExecutionContext* raw_ctx = nullptr;
  // 执行上下文会申请设备侧显存（activation）
  // 闸门串行 + 有界重试；TensorRT 只返回 nullptr，按瞬时争用处理
  CudaInitResult create_ret = RunCudaInit(
      "exec_context",
      [this, &raw_ctx]() -> int {
        raw_ctx = engine_->createExecutionContext();
        return raw_ctx != nullptr ? 0 : kCudaInitNoCode;
      },
      nullptr);
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context(raw_ctx);
  if (context == nullptr) {
    LOGE(MODEL) << "Failed to create TensorRT execution context [" << create_ret.attempt
                << " attempt(s), " << create_ret.elapsed_ms << " ms]";
    return nullptr;
  }
  // 动态 shape 模型：每个新建 context 都要按 opt profile 重新 setInputShape，
  // 与 ParseBindings 中的处理保持一致
  for (size_t i = 0; i < input_names_.size(); ++i) {
    const auto& name = input_names_[i];
    nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
    if (!has_dynamic_dims(dims)) continue;
    nvinfer1::Dims opt_dims = engine_->getProfileShape(name.c_str(),
                                                       context->getOptimizationProfile(),
                                                       nvinfer1::OptProfileSelector::kOPT);
    if (!is_static_shape(opt_dims)) {
      LOGE(MODEL) << "Input [" << name << "] dims unresolved: " << dims_to_str(opt_dims);
      return nullptr;
    }
    if (!context->setInputShape(name.c_str(), opt_dims)) {
      LOGE(MODEL) << "setInputShape failed for tensor: " << name << ", dims: " << dims_to_str(opt_dims);
      return nullptr;
    }
  }
  return context;
}

}  // namespace cnstream
