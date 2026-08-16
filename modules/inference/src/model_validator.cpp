
#include "model_validator.hpp"
#include "memop_factory.hpp"

#include <algorithm>
#include <numeric>
#include <cstring>

namespace cnstream {

ModelValidator::ModelValidator(const std::string& model_path,
                               const std::string& device_type,
                               int device_id,
                               int input_ordered_index)
    : model_path_(model_path),
      device_id_(device_id),
      input_ordered_index_(static_cast<uint32_t>(input_ordered_index)) {
  auto it = device_type_map.find(device_type);
  if (it != device_type_map.end()) {
    device_type_ = it->second;
  } else {
    LOGE(MODEL_VALIDATOR) << "Unknown device_type: " << device_type << ", fallback to CPU";
    device_type_ = DevType::CPU;
    device_id_ = -1;
  }
}

ModelValidator::~ModelValidator() {
  model_loader_.reset();
  memop_.reset();
  dev_input_bufs_.clear();
  dev_output_bufs_.clear();
  cpu_input_bufs_.clear();
  cpu_output_bufs_.clear();
}


bool ModelValidator::Load() {
  if (model_path_.empty()) {
    LOGE(MODEL_VALIDATOR) << "model_path is empty";
    return false;
  }

  auto& factory = ModelLoaderFactory::Instance();
  model_loader_ = factory.CreateModelLoader(device_type_, device_id_);
  if (!model_loader_) {
    LOGE(MODEL_VALIDATOR) << "CreateModelLoader failed, device: " << DevType2Str(device_type_)
                          << ", id: " << device_id_;
    return false;
  }

  InferParams params;
  params.device_type = device_type_;
  params.device_id = static_cast<uint32_t>(device_id_);
  params.input_ordered_index = input_ordered_index_;

  if (!model_loader_->Init(model_path_, params)) {
    LOGE(MODEL_VALIDATOR) << "ModelLoader::Init failed, path: " << model_path_;
    return false;
  }

  if (!model_loader_->IsValid()) {
    LOGE(MODEL_VALIDATOR) << "Model not valid after init: " << model_path_;
    return false;
  }

  // Create MemOp for device-aware memory operations
  memop_ = MemOpFactory::Instance().CreateMemOp(device_type_, device_id_);
  if (!memop_) {
    LOGE(MODEL_VALIDATOR) << "CreateMemOp failed, device: " << DevType2Str(device_type_);
    return false;
  }

  AllocateBuffers();

  LOGI(MODEL_VALIDATOR) << "Model loaded: " << model_path_
                        << " | device: " << DevType2Str(device_type_)
                        << " | batch: " << model_loader_->get_batch_size()
                        << " | input: " << model_loader_->get_width()
                        << "x" << model_loader_->get_height()
                        << "x" << model_loader_->get_channel()
                        << " | inputs: " << model_loader_->InputNum()
                        << " | outputs: " << model_loader_->OutputNum();
  return true;
}

bool ModelValidator::IsLoaded() const {
  return model_loader_ && model_loader_->IsValid();
}


ModelInfo ModelValidator::GetModelInfo() const {
  ModelInfo info;
  info.model_path = model_path_;
  info.device_type = DevType2Str(device_type_);
  info.device_id = device_id_;

  if (!IsLoaded()) return info;

  info.batch_size = static_cast<int>(model_loader_->get_batch_size());
  info.channel = static_cast<int>(model_loader_->get_channel());
  info.height = static_cast<int>(model_loader_->get_height());
  info.width = static_cast<int>(model_loader_->get_width());

  for (uint32_t i = 0; i < model_loader_->InputNum(); ++i) {
    ModelTensorInfo t;
    t.name = model_loader_->InputName(i);
    auto shape = model_loader_->InputShape(i);
    t.shape = {shape.N(), shape.C(), shape.H(), shape.W()};
    t.dtype = DataTypeToString(model_loader_->InputDataType(i));
    info.inputs.push_back(std::move(t));
  }

  for (uint32_t i = 0; i < model_loader_->OutputNum(); ++i) {
    ModelTensorInfo t;
    t.name = model_loader_->OutputName(i);
    auto shape = model_loader_->OutputShape(i);
    t.shape = {shape.N(), shape.C(), shape.H(), shape.W()};
    t.dtype = DataTypeToString(model_loader_->OutputDataType(i));
    info.outputs.push_back(std::move(t));
  }

  return info;
}


/**
 * @brief 通过 memop 分配 dev_input_bufs_ 和 dev_output_bufs_ 设备相关内存
 */
void ModelValidator::AllocateBuffers() {
  uint32_t in_num = model_loader_->InputNum();
  uint32_t out_num = model_loader_->OutputNum();

  cpu_input_bufs_.assign(in_num, {});
  dev_input_bufs_.assign(in_num, {});
  cpu_output_bufs_.assign(out_num, {});
  dev_output_bufs_.assign(out_num, {});

  for (uint32_t i = 0; i < in_num; ++i) {
    auto shape = model_loader_->InputShape(i);
    auto data_type = model_loader_->InputDataType(i);
    size_t elem_count = static_cast<size_t>(shape.DataCount());
    size_t dsize = static_cast<size_t>(data_type_size(data_type));
    size_t batch_size = elem_count / shape.N();
    size_t byte_size = elem_count * dsize;

    if (batch_size != 1) {
      LOGE(MODEL_VALIDATOR) << "Batch size must be 1 for image validation";
      return;
    }
    
    LOGE(MODEL_VALIDATOR) << "Allocate input[" << i << "] shape=" << shape
                          << ", dtype=" << DataTypeToString(data_type);
    cpu_input_bufs_[i].assign(elem_count, 0.0f);  // 分配缓冲区
    dev_input_bufs_[i] = memop_->Allocate(byte_size);
    if (!dev_input_bufs_[i]) {
      LOGE(MODEL_VALIDATOR) << "Allocate input[" << i << "] device buffer failed, size=" << byte_size;
    }
  }

  for (uint32_t i = 0; i < out_num; ++i) {

    auto shape = model_loader_->OutputShape(i);
    auto data_type = model_loader_->OutputDataType(i);
    size_t elem_count = static_cast<size_t>(shape.DataCount());
    size_t dsize = static_cast<size_t>(data_type_size(data_type));
    size_t batch_size = elem_count / shape.N();
    size_t byte_size = elem_count * dsize;

    if (batch_size != 1) {
      LOGE(MODEL_VALIDATOR) << "Batch size must be 1 for image validation";
      return;
    }

    LOGE(MODEL_VALIDATOR) << "Allocate output[" << i << "] shape=" << shape
                          << ", dtype=" << DataTypeToString(data_type);
    cpu_output_bufs_[i].assign(elem_count, 0.0f);
    dev_output_bufs_[i] = memop_->Allocate(byte_size);
    if (!dev_output_bufs_[i]) {
      LOGE(MODEL_VALIDATOR) << "Allocate output[" << i << "] device buffer failed, size=" << byte_size;
    }
  }
}

/**
 * @brief 运行推理 裸接口 使用输入参数指定的缓冲区
 */
std::vector<std::vector<float>>
ModelValidator::Infer(const std::vector<std::vector<float>>& inputs) {
  std::vector<std::vector<float>> results;

  if (!IsLoaded()) {
    LOGE(MODEL_VALIDATOR) << "Model not loaded, call Load() first";
    return results;
  }

  if (inputs.size() != model_loader_->InputNum()) {
    LOGE(MODEL_VALIDATOR) << "Input count mismatch: got " << inputs.size()
                          << ", expected " << model_loader_->InputNum();
    return results;
  }

  // Copy user data into CPU buffers, then H2D
  for (uint32_t i = 0; i < inputs.size(); ++i) {
    size_t expected = static_cast<size_t>(model_loader_->InputShape(i).DataCount());
    if (inputs[i].size() != expected) {
      LOGE(MODEL_VALIDATOR) << "Input[" << i << "] size mismatch: got " << inputs[i].size()
                            << ", expected " << expected;
      return results;
    }
    size_t dsize = static_cast<size_t>(data_type_size(model_loader_->InputDataType(i)));
    size_t byte_size = expected * dsize;
    memop_->CopyFromHost(dev_input_bufs_[i].get(), inputs[i].data(), byte_size);
  }

  // Run inference
  if (!model_loader_->RunSync(dev_input_bufs_, dev_output_bufs_)) {
    LOGE(MODEL_VALIDATOR) << "RunSync failed";
    return results;
  }

  // D2H and return
  results.resize(model_loader_->OutputNum());
  for (uint32_t i = 0; i < model_loader_->OutputNum(); ++i) {
    size_t elem_count = static_cast<size_t>(model_loader_->OutputShape(i).DataCount());
    size_t dsize = static_cast<size_t>(data_type_size(model_loader_->OutputDataType(i)));
    size_t byte_size = elem_count * dsize;
    results[i].resize(elem_count);
    memop_->CopyToHost(results[i].data(), dev_output_bufs_[i].get(), byte_size);
  }

  return results;
}


E2EResult ModelValidator::RunE2E(
    const cv::Mat& image,
    const std::string& preproc_name,
    const std::string& postproc_name,
    const std::map<std::string, std::string>& preproc_params,
    const std::map<std::string, std::string>& postproc_params) {

  E2EResult result;

  if (!IsLoaded()) {
    result.error = "Model not loaded, call Load() first";
    return result;
  }

  if (image.empty()) {
    result.error = "Input image is empty";
    return result;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  FrameInfoPtr frame_info = CreateFrameInfo(image);
  if (!frame_info) {
    result.error = "CreateFrameInfo failed";
    return result;
  }

  std::shared_ptr<Preproc> preproc;
  if (!preproc_name.empty()) {
    preproc = std::shared_ptr<Preproc>(Preproc::Create(preproc_name));
    if (!preproc) {
      result.error = "Preproc::Create failed: " + preproc_name;
      return result;
    }
    if (!preproc->Init(preproc_params)) {
      result.error = "Preproc::Init failed: " + preproc_name;
      return result;
    }
    if (!RunPreproc(preproc.get(), frame_info)) {
      result.error = "RunPreproc failed";
      return result;
    }
  } else {
    // No preproc — need user to provide raw float input via Infer() instead
    result.error = "preproc_name is empty, use Infer() for raw tensor input";
    return result;
  }

  if (!RunInference()) {
    result.error = "RunInference failed";
    return result;
  }

  if (!postproc_name.empty()) {
    std::shared_ptr<Postproc> postproc(Postproc::Create(postproc_name));
    if (!postproc) {
      result.error = "Postproc::Create failed: " + postproc_name;
      return result;
    }
    if (!postproc->Init(postproc_params)) {
      result.error = "Postproc::Init failed: " + postproc_name;
      return result;
    }
    if (!RunPostproc(postproc.get(), frame_info)) {
      result.error = "RunPostproc failed";
      return result;
    }
    result.detections = ExtractDetections(frame_info);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  result.latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  LOGI(MODEL_VALIDATOR) << "E2E done: " << result.detections.size()
                        << " detections, " << result.latency_ms << " ms";
  return result;
}


std::vector<BenchmarkResult> ModelValidator::Benchmark(
    const cv::Mat& image,
    const std::string& preproc_name,
    const std::string& postproc_name,
    const std::map<std::string, std::string>& preproc_params,
    const std::map<std::string, std::string>& postproc_params,
    int warmup_runs,
    int test_runs,
    const std::vector<int>& batch_sizes) {

  std::vector<BenchmarkResult> results;

  if (!IsLoaded()) {
    LOGE(MODEL_VALIDATOR) << "Model not loaded";
    return results;
  }

  // Currently only batch_size=1 is supported (single image validation)
  for (int bs : batch_sizes) {
    if (bs != 1) {
      LOGW(MODEL_VALIDATOR) << "batch_size=" << bs << " not supported yet, skipping";
      continue;
    }

    BenchmarkResult r;
    r.batch_size = bs;

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
      E2EResult tmp = RunE2E(image, preproc_name, postproc_name,
                             preproc_params, postproc_params);
      if (!tmp.error.empty()) {
        LOGW(MODEL_VALIDATOR) << "Warmup failed: " << tmp.error;
      }
    }

    // Measure
    std::vector<double> latencies;
    latencies.reserve(test_runs);
    for (int i = 0; i < test_runs; ++i) {
      E2EResult tmp = RunE2E(image, preproc_name, postproc_name,
                             preproc_params, postproc_params);
      if (!tmp.error.empty()) {
        r.error_count++;
        continue;
      }
      latencies.push_back(tmp.latency_ms);
    }

    if (!latencies.empty()) {
      std::sort(latencies.begin(), latencies.end());
      r.min_ms = latencies.front();
      r.max_ms = latencies.back();
      r.avg_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
      // P99: index = ceil(0.99 * n) - 1
      int p99_idx = static_cast<int>(std::ceil(0.99 * latencies.size())) - 1;
      p99_idx = std::max(0, std::min(p99_idx, static_cast<int>(latencies.size()) - 1));
      r.p99_ms = latencies[p99_idx];
      r.fps = r.avg_ms > 0 ? 1000.0 / r.avg_ms : 0.0;
    }

    LOGI(MODEL_VALIDATOR) << "Benchmark bs=" << bs
                          << " | avg=" << r.avg_ms << "ms"
                          << " | p99=" << r.p99_ms << "ms"
                          << " | fps=" << r.fps
                          << " | errors=" << r.error_count;

    results.push_back(std::move(r));
  }

  return results;
}


FrameInfoPtr ModelValidator::CreateFrameInfo(const cv::Mat& image,
                                             const std::string& stream_id) {
  FrameInfoPtr frame_info = FrameInfo::Create(stream_id);
  if (!frame_info) return nullptr;

  // DataFrame — postproc reads GetWidth()/GetHeight() from it
  DataFramePtr frame = std::make_shared<DataFrame>();
  frame->SetImage(image);
  frame_info->collection.Add<DataFramePtr>(kDataFrameTag, frame);

  // ModelInputImage — preproc reads image via GetModelInputImage()
  ModelInputImagePtr model_input = std::make_shared<ModelInputImage>();
  model_input->image = image.clone();
  model_input->cur_width = image.cols;
  model_input->cur_height = image.rows;
  frame_info->collection.Add<ModelInputImagePtr>(kModelInputImageTag, model_input);

  // InferObjs — postproc writes detection results here
  InferObjsPtr objs = std::make_shared<InferObjs>();
  frame_info->collection.Add<InferObjsPtr>(kInferObjsTag, objs);

  // InferData — some postproc implementations may read it
  InferDataPtr infer_data = std::make_shared<InferData>();
  frame_info->collection.Add<InferDataPtr>(kInferDataTag, infer_data);

  return frame_info;
}

/**
 * @brief 运行预处理
 * @details 从单帧图像中得到处理结果, 内存模型假定 batch_size = 1
 */
bool ModelValidator::RunPreproc(Preproc* preproc, const FrameInfoPtr& frame_info) {
  // Build float* array pointing to cpu_input_bufs_
  std::vector<float*> input_ptrs;
  input_ptrs.reserve(cpu_input_bufs_.size());
  for (auto& buf : cpu_input_bufs_) {
    input_ptrs.push_back(buf.data());
  }

  int ret = preproc->Execute(input_ptrs, model_loader_.get(), frame_info);
  if (ret != 0) {
    LOGE(MODEL_VALIDATOR) << "Preproc::Execute returned " << ret;
    return false;
  }
  return true;
}

bool ModelValidator::RunInference() {
  // H2D: copy cpu_input_bufs_ -> dev_input_bufs_
  for (size_t i = 0; i < cpu_input_bufs_.size(); ++i) {
    size_t dsize = static_cast<size_t>(data_type_size(model_loader_->InputDataType(i)));
    size_t byte_size = cpu_input_bufs_[i].size() * dsize;
    memop_->CopyFromHost(dev_input_bufs_[i].get(), cpu_input_bufs_[i].data(), byte_size);
  }

  // RunSync
  if (!model_loader_->RunSync(dev_input_bufs_, dev_output_bufs_)) {
    LOGE(MODEL_VALIDATOR) << "RunSync failed";
    return false;
  }

  // D2H: copy dev_output_bufs_ -> cpu_output_bufs_
  for (size_t i = 0; i < cpu_output_bufs_.size(); ++i) {
    size_t dsize = static_cast<size_t>(data_type_size(model_loader_->OutputDataType(i)));
    size_t byte_size = cpu_output_bufs_[i].size() * dsize;
    memop_->CopyToHost(cpu_output_bufs_[i].data(), dev_output_bufs_[i].get(), byte_size);
  }

  return true;
}

/**
 * @brief CPU 后处理
 */
bool ModelValidator::RunPostproc(Postproc* postproc, const FrameInfoPtr& frame_info) {
  // Build float* array pointing to cpu_output_bufs_
  std::vector<float*> output_ptrs;
  output_ptrs.reserve(cpu_output_bufs_.size());
  for (auto& buf : cpu_output_bufs_) {
    output_ptrs.push_back(buf.data());
  }

  int ret = postproc->Execute(output_ptrs, model_loader_.get(), frame_info);
  if (ret != 0) {
    LOGE(MODEL_VALIDATOR) << "Postproc::Execute returned " << ret;
    return false;
  }
  return true;
}

std::vector<ValidatorDetection>
ModelValidator::ExtractDetections(const FrameInfoPtr& frame_info) {
  std::vector<ValidatorDetection> detections;

  if (!frame_info->collection.HasValue(kInferObjsTag)) {
    return detections;
  }

  InferObjsPtr objs = frame_info->collection.Get<InferObjsPtr>(kInferObjsTag);
  if (!objs) return detections;

  std::lock_guard<std::mutex> lock(objs->mutex_);
  for (const auto& obj : objs->objs_) {
    if (!obj) continue;
    ValidatorDetection det;
    det.class_id = obj->id;
    det.score = obj->score;
    det.x = obj->bbox.x;
    det.y = obj->bbox.y;
    det.w = obj->bbox.w;
    det.h = obj->bbox.h;
    // Try to get class name from InferObjectInfo classes
    if (!obj->classes.empty()) {
      det.class_name = obj->classes[0].name;
    }
    detections.push_back(std::move(det));
  }

  return detections;
}


std::string ModelValidator::DataTypeToString(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32:  return "float32";
    case DataType::FLOAT16:  return "float16";
    case DataType::INT8:     return "int8";
    case DataType::UINT8:    return "uint8";
    case DataType::INT16:    return "int16";
    case DataType::INT32:    return "int32";
    default:                 return "unknown";
  }
}

}  // namespace cnstream
