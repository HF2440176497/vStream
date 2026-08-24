
#include "rockchip/model_loader_rknn.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include "cnstream_logging.hpp"
#include "common.hpp"
#include "tensor.hpp"

namespace cnstream {

namespace {

static DataType RknnDtypeToTensorDtype(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
      return DataType::FLOAT32;
    case RKNN_TENSOR_FLOAT16:
      return DataType::FLOAT16;
    case RKNN_TENSOR_INT8:
      return DataType::INT8;
    case RKNN_TENSOR_UINT8:
      return DataType::UINT8;
    case RKNN_TENSOR_INT16:
      return DataType::INT16;
    case RKNN_TENSOR_INT32:
      return DataType::INT32;
    default:
      return DataType::INVALID;
  }
}

static TensorFormat RknnFormatToTensorFormat(rknn_tensor_format fmt) {
  switch (fmt) {
    case RKNN_TENSOR_NCHW:
      return TensorFormat::LINEAR;
    case RKNN_TENSOR_NHWC:
      // NOTE: NHWC 在 TensorShape 内部统一归一化为 NCHW 存储，对外语义等同 LINEAR
      return TensorFormat::LINEAR;
    default:
      return TensorFormat::INVALID;
  }
}

/**
 * @brief 将 RKNN tensor dims 转换为 NCHW 顺序的 TensorShape。
 *
 * RKNN 模型的 dims 排列取决于 attr.fmt：
 *   - RKNN_TENSOR_NCHW: dims = [N, C, H, W] — 直接使用
 *   - RKNN_TENSOR_NHWC: dims = [N, H, W, C] — 需重排为 [N, C, H, W]
 *
 * 框架固定按 NCHW 解释 (N=shape_[0], C=shape_[1], H=shape_[2], W=shape_[3])，
 * 因此 NHWC 的 dims 必须在此处做转置
 *
 * 对于 n_dims < 4 的情况（如一维输出），补 1 对齐到 4 维 NCHW。
 */
static TensorShape RknnDimsToTensorShape(const rknn_tensor_attr& attr) {
  if (attr.n_dims >= 4) {
    if (attr.fmt == RKNN_TENSOR_NHWC) {
      // NHWC -> NCHW
      std::vector<int> nchw = {attr.dims[0], attr.dims[3], attr.dims[1], attr.dims[2]};
      return TensorShape(nchw);
    } else {
      // NCHW 或其他格式，按原始顺序
      std::vector<int> dims(attr.dims, attr.dims + attr.n_dims);
      return TensorShape(dims);
    }
  } else {
    std::vector<int> dims(4, 1);
    for (int i = 0; i < attr.n_dims; ++i) {
      dims[4 - attr.n_dims + i] = attr.dims[i];
    }
    return TensorShape(dims);
  }
}

static std::string RknnDtypeToStr(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
      return "FLOAT32";
    case RKNN_TENSOR_FLOAT16:
      return "FLOAT16";
    case RKNN_TENSOR_INT8:
      return "INT8";
    case RKNN_TENSOR_UINT8:
      return "UINT8";
    case RKNN_TENSOR_INT16:
      return "INT16";
    case RKNN_TENSOR_INT32:
      return "INT32";
    default:
      return "UNKNOWN";
  }
}

static std::string RknnFormatToStr(rknn_tensor_format fmt) {
  switch (fmt) {
    case RKNN_TENSOR_NCHW:
      return "NCHW";
    case RKNN_TENSOR_NHWC:
      return "NHWC";
    case RKNN_TENSOR_NC1HWC2:
      return "NC1HWC2";
    case RKNN_TENSOR_UNDEFINED:
      return "UNDEFINED";
    default:
      return "UNKNOWN";
  }
}

static std::vector<uint8_t> LoadModelFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    LOGE(MODEL) << "Failed to open rknn model file: " << path;
    return {};
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    LOGE(MODEL) << "Failed to read rknn model file: " << path;
    return {};
  }
  return buffer;
}

}  // namespace

ModelLoaderRknn::ModelLoaderRknn(int device_id) : ModelLoader(device_id) {
  device_type_ = DevType::ROCKCHIP;
  std::memset(&io_num_, 0, sizeof(io_num_));
}

ModelLoaderRknn::~ModelLoaderRknn() {
  if (rknn_ctx_ != 0) {
    rknn_destroy(rknn_ctx_);
    rknn_ctx_ = 0;
  }
}

bool ModelLoaderRknn::Init(const std::string& engine_path, const InferParams& params) {
  if (engine_path.empty()) {
    LOGF(MODEL) << "Empty rknn model path";
    return false;
  }

  SetInputOrderedIndex(params.input_ordered_index);

  // 从 custom_preproc_params 读取 want_float 等配置，暂不改动 InferParams

  // want_float=true：RKNN 将所有输出统一转为 float32 返回
  // want_float=false：输出保持原始类型（如 INT8），由 postproc 手动反量化。
  auto it_wf = params.custom_preproc_params.find(key_want_float);
  if (it_wf != params.custom_preproc_params.end()) {
    const std::string& v = it_wf->second;
    want_float_ = !(v == "0" || v == "false" || v == "False" || v == "FALSE");
  }

  // typedef enum _rknn_core_mask {
  //     RKNN_NPU_CORE_AUTO = 0,                                       /* default, run on NPU core randomly. */
  //     RKNN_NPU_CORE_0 = 1,                                          /* run on NPU core 0. */
  //     RKNN_NPU_CORE_1 = 2,                                          /* run on NPU core 1. */
  //     RKNN_NPU_CORE_2 = 4,                                          /* run on NPU core 2. */
  //     RKNN_NPU_CORE_0_1 = RKNN_NPU_CORE_0 | RKNN_NPU_CORE_1,        /* run on NPU core 0 and core 1. */
  //     RKNN_NPU_CORE_0_1_2 = RKNN_NPU_CORE_0_1 | RKNN_NPU_CORE_2,    /* run on NPU core 0 and core 1 and core 2. */
  //     RKNN_NPU_CORE_UNDEFINED,
  // } rknn_core_mask;
  auto it = params.custom_preproc_params.find(key_core_mask);
  if (it != params.custom_preproc_params.end()) {
    try {
      core_mask_ = std::stoi(it->second);
    } catch (const std::exception&) {
      LOGW(MODEL) << "Invalid " << key_core_mask << " value: " << it->second << ", using default";
      core_mask_ = 0;  // RKNN_NPU_CORE_AUTO
    }
  }

  if (!LoadEngine(engine_path)) {
    return false;
  }

  if (!SetCoreMask(core_mask_)) {
    LOGW(MODEL) << "SetCoreMask failed [core_mask=" << core_mask_
                << "] continuing with default core";
  }

  if (!QueryTensorInfo()) {
    return false;
  }

  name_ = utils::get_filename_without_ext(engine_path);
  engine_path_ = engine_path;
  return true;
}

bool ModelLoaderRknn::SetCoreMask(int core_mask) {
  if (rknn_ctx_ == 0) {
    LOGE(MODEL) << "RKNN context not initialized, cannot set core mask";
    return false;
  }
  int ret = rknn_set_core_mask(rknn_ctx_, static_cast<rknn_core_mask>(core_mask));
  if (ret != RKNN_SUCC) {
    LOGE(MODEL) << "rknn_set_core_mask fail, core_mask=" << core_mask << ", ret=" << ret;
    return false;
  }
  LOGI(MODEL) << "RKNN core mask set to " << core_mask;
  return true;
}

bool ModelLoaderRknn::LoadEngine(const std::string& engine_path) {
  auto model_data = LoadModelFile(engine_path);
  if (model_data.empty()) {
    return false;
  }

  int ret = rknn_init(&rknn_ctx_, model_data.data(), model_data.size(), 0, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE(MODEL) << "rknn_init fail, ret=" << ret;
    rknn_ctx_ = 0;
    return false;
  }

  return true;
}

bool ModelLoaderRknn::QueryTensorInfo() {
  int ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC) {
    LOGE(MODEL) << "rknn_query RKNN_QUERY_IN_OUT_NUM fail, ret=" << ret;
    return false;
  }

  LOGI(MODEL) << "RKNN model input num: " << io_num_.n_input
              << ", output num: " << io_num_.n_output;

  if (io_num_.n_input < 1 || io_num_.n_output < 1) {
    LOGE(MODEL) << "RKNN model has no input or output";
    return false;
  }

  return ParseInputOutputAttr();
}

bool ModelLoaderRknn::ParseInputOutputAttr() {
  input_shapes_.clear();
  output_shapes_.clear();
  input_data_types_.clear();
  output_data_types_.clear();
  input_names_.clear();
  output_names_.clear();
  bind_name_index_map_.clear();
  input_attrs_.clear();
  output_attrs_.clear();

  // 输入属性
  input_attrs_.resize(io_num_.n_input);
  for (int i = 0; i < io_num_.n_input; ++i) {
    input_attrs_[i].index = i;
    int ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      LOGE(MODEL) << "rknn_query RKNN_QUERY_INPUT_ATTR fail, index=" << i << ", ret=" << ret;
      return false;
    }

    const auto& attr = input_attrs_[i];
    LOGI(MODEL) << "RKNN input [" << i << "] name=" << attr.name
                << ", dims=[" << attr.dims[0] << "," << attr.dims[1] << "," << attr.dims[2] << "," << attr.dims[3]
                << "], type=" << RknnDtypeToStr(attr.type)
                << ", fmt=" << RknnFormatToStr(attr.fmt);

    DataType data_type = RknnDtypeToTensorDtype(attr.type);
    if (data_type == DataType::INVALID) {
      LOGE(MODEL) << "Unsupported RKNN input data type: " << attr.type;
      return false;
    }

    TensorFormat format = RknnFormatToTensorFormat(attr.fmt);
    if (format == TensorFormat::INVALID) {
      LOGE(MODEL) << "Unsupported RKNN input format: " << attr.fmt;
      return false;
    }

    input_names_.push_back(attr.name);
    input_data_types_.push_back(data_type);
    input_shapes_.push_back(RknnDimsToTensorShape(attr));
    bind_name_index_map_[attr.name] = i;
  }

  // 输出属性
  output_attrs_.resize(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; ++i) {
    output_attrs_[i].index = i;
    int ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      LOGE(MODEL) << "rknn_query RKNN_QUERY_OUTPUT_ATTR fail, index=" << i << ", ret=" << ret;
      return false;
    }

    const auto& attr = output_attrs_[i];
    LOGI(MODEL) << "RKNN output [" << i << "] name=" << attr.name
                << ", dims=[" << attr.dims[0] << "," << attr.dims[1] << "," << attr.dims[2] << "," << attr.dims[3]
                << "], type=" << RknnDtypeToStr(attr.type)
                << ", fmt=" << RknnFormatToStr(attr.fmt)
                << ", qnt=" << attr.qnt_type;

    DataType data_type = RknnDtypeToTensorDtype(attr.type);
    if (data_type == DataType::INVALID) {
      LOGE(MODEL) << "Unsupported RKNN output data type: " << attr.type;
      return false;
    }

    output_names_.push_back(attr.name);
    // output_data_types_ 必须与 want_float_ 保持一致：
    //   want_float_=true  → FLOAT32（RKNN 内部做反量化）
    //   want_float_=false → 原始类型（TODO: 需要调度框架支持）
    if (want_float_) {
      output_data_types_.push_back(DataType::FLOAT32);
      if (data_type != DataType::FLOAT32) {
        LOGI(MODEL) << "RKNN output [" << i << "] type " << RknnDtypeToStr(attr.type)
                    << " will be delivered as FLOAT32 (want_float=true)";
      }
    } else {
      output_data_types_.push_back(data_type);
    }
    output_shapes_.push_back(RknnDimsToTensorShape(attr));
    // NOTE: For RKNN，bind_name_index_map_ 是手动偏移后的 binding_index, 无实际作用
    bind_name_index_map_[attr.name] = io_num_.n_input + i;
  }

  // 判断模型是否量化：以第一个输出为准（仅作信息记录，供日志和 postproc 参考）
  if (!output_attrs_.empty()) {
    is_quant_ = (output_attrs_[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                 output_attrs_[0].type == RKNN_TENSOR_INT8);
  }
  if (is_quant_) {
    LOGW(MODEL) << "RKNN model is quantized"
              << ", want_float=" << (want_float_ ? "true" : "false");
  }

  if (input_names_.size() > 1) {
    LOGW(MODEL) << "RKNN model has " << input_names_.size()
                << " inputs, choose input index: " << input_ordered_index_;
  }
  if (input_ordered_index_ < 0 || input_ordered_index_ >= static_cast<int>(input_names_.size())) {
    LOGF(MODEL) << "input_ordered_index_ out of range: " << input_ordered_index_
                << ", input count: " << input_names_.size();
    return false;
  }
  input_name_ = input_names_[input_ordered_index_];

  return true;
}

bool ModelLoaderRknn::RunSync(void* exec_ctx,
                              std::vector<std::shared_ptr<void>> inputs,
                              std::vector<std::shared_ptr<void>> outputs) {
  (void)exec_ctx;  // RKNN 单 context，由基类默认实现返回 nullptr
  std::lock_guard<std::mutex> lock(mutex_);

  if (rknn_ctx_ == 0) {
    LOGE(MODEL) << "RKNN context is not initialized";
    return false;
  }

  if (inputs.size() != static_cast<size_t>(io_num_.n_input) ||
      outputs.size() != static_cast<size_t>(io_num_.n_output)) {
    LOGE(MODEL) << "RKNN tensor count mismatch: inputs " << inputs.size() << " vs " << io_num_.n_input
                << ", outputs " << outputs.size() << " vs " << io_num_.n_output;
    return false;
  }

  // 1. 设置输入
  // 仿照 demo, 设置为局部变量
  std::vector<rknn_input> rknn_inputs(io_num_.n_input);
  for (int i = 0; i < io_num_.n_input; ++i) {
    rknn_inputs[i].index = i;
    rknn_inputs[i].type = input_attrs_[i].type;
    rknn_inputs[i].fmt = input_attrs_[i].fmt;
    rknn_inputs[i].size = input_attrs_[i].size;
    rknn_inputs[i].buf = inputs[i].get();
  }

  int ret = rknn_inputs_set(rknn_ctx_, io_num_.n_input, rknn_inputs.data());
  if (ret != RKNN_SUCC) {
    LOGE(MODEL) << "rknn_inputs_set fail, ret=" << ret;
    return false;
  }

  // 2. 推理
  ret = rknn_run(rknn_ctx_, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE(MODEL) << "rknn_run fail, ret=" << ret;
    return false;
  }

  // 3. 获取输出
  std::vector<rknn_output> rknn_outputs(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; ++i) {
    rknn_outputs[i].index = i;
    rknn_outputs[i].want_float = want_float_;
  }

  ret = rknn_outputs_get(rknn_ctx_, io_num_.n_output, rknn_outputs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    LOGE(MODEL) << "rknn_outputs_get fail, ret=" << ret;
    return false;
  }

  for (int i = 0; i < io_num_.n_output; ++i) {
    // 使用 rknn_outputs[i].size（RKNN 实际返回的数据字节数）而非 output_attrs_[i].size（原始类型大小）。
    // 当 want_float=true 时，RKNN 将 INT8/UINT8 输出反量化为 float32，
    // 实际数据大小 = n_elems * sizeof(float) 可能与 buf_size 不一致。
    size_t actual_size = rknn_outputs[i].size;
    size_t buf_size = GetOutputDataBatchAlignSize(i);
    if (actual_size > buf_size) {
      LOGE(MODEL) << "[" << i << "] output size " << actual_size
                 << " exceeds buffer size " << buf_size
                 << ", truncating (possible data loss)";
      actual_size = buf_size;
    }
    if (rknn_outputs[i].buf != nullptr && outputs[i] != nullptr && actual_size > 0) {
      std::memcpy(outputs[i].get(), rknn_outputs[i].buf, actual_size);
    }
  }

  rknn_outputs_release(rknn_ctx_, io_num_.n_output, rknn_outputs.data());
  return true;
}

static bool RegisterModelLoader() {
  auto& factory = ModelLoaderFactory::Instance();
  bool result = true;
  result &= factory.RegisterModelLoaderCreator(
      DevType::ROCKCHIP, [](int device_id) { return std::make_unique<ModelLoaderRknn>(device_id); });
  return result;
}

static bool model_loader_registered = RegisterModelLoader();

}  // namespace cnstream
