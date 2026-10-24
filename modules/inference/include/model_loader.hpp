
#ifndef MODULES_INFERENCE_MODEL_LOADER_HPP_
#define MODULES_INFERENCE_MODEL_LOADER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "data_source_param.hpp"
#include "tensor.hpp"
#include "infer_params.hpp"
#include "run_shape.hpp"

namespace cnstream {

class ModelLoader {
 public:
  ModelLoader(int device_id = -1): device_id_(device_id) {
    device_type_ = DevType::CPU;
  }
  virtual ~ModelLoader() = default;
  virtual bool IsValid() = 0;
  virtual bool Init(const std::string& engine_path, const InferParams& params) = 0;
  /**
   * @brief 加载模型文件并初始化推理上下文。
   * 由 Init() 内部调用，各平台实现（TensorRT/RKNN/CPU）负责各自的加载流程。
   */
  virtual bool LoadEngine(const std::string& engine_path) = 0;
  virtual void SetInputOrderedIndex(int input_index) {
    input_ordered_index_ = input_index;
  }
  const std::string& get_name() { return name_; };

  public:
  int GetDeviceId() const { return device_id_; }
  DevType GetDeviceType() const { return device_type_; }
  virtual void* GetStream() const { return nullptr; }
  // 该后端是否必须有独占的 execution context 才可推理。
  virtual bool RequiresExecutionContext() const { return false; }

  /** 单输入模型，获取对应的信息 */
  uint32_t get_batch_size() const { return input_shapes_[input_ordered_index_].N(); }
  uint32_t get_channel() const { return input_shapes_[input_ordered_index_].C(); }
  uint32_t get_height() const { return input_shapes_[input_ordered_index_].H(); }
  uint32_t get_width() const { return input_shapes_[input_ordered_index_].W(); }

  uint32_t InputNum() const { return static_cast<uint32_t>(input_shapes_.size()); }
  uint32_t OutputNum() const { return static_cast<uint32_t>(output_shapes_.size()); }

  TensorShape InputShape(uint32_t index) const {
    if (index < input_shapes_.size()) {
      return input_shapes_[index];
    }
    return TensorShape();
  }

  TensorShape OutputShape(uint32_t index) const {
    if (index < output_shapes_.size()) {
      return output_shapes_[index];
    }
    return TensorShape();
  }

  DataType InputDataType(uint32_t index) const {
    if (index < input_data_types_.size()) {
      return input_data_types_[index];
    }
    return DataType::INVALID;
  }

  DataType OutputDataType(uint32_t index) const {
    if (index < output_data_types_.size()) {
      return output_data_types_[index];
    }
    return DataType::INVALID;
  }

  std::string InputName(uint32_t index) const {
    if (index < input_names_.size()) {
      return input_names_[index];
    }
    return "";
  }

  std::string OutputName(uint32_t index) const {
    if (index < output_names_.size()) {
      return output_names_[index];
    }
    return "";
  }
  int get_input_ordered_index() const { return input_ordered_index_; }

  size_t GetInputDataBatchAlignSize(uint32_t index) const {
    return InputShape(index).DataCount() * data_type_size(input_data_types_[index]);
  }

  size_t GetOutputDataBatchAlignSize(uint32_t index) const {
    return OutputShape(index).DataCount() * data_type_size(output_data_types_[index]);
  }

  /**
   * 获取一个调用方独占的执行上下文句柄（如 TensorRT 的 IExecutionContext）。
   * 支持多执行上下文的后端返回有效句柄；句柄同一时刻只允许一个线程使用，
   * 用完后通过 ReleaseExecutionContext() 归还。
   * 不支持的后端（RKNN/CPU）返回 nullptr，RunSync 将忽略句柄。
   */
  virtual void* AcquireExecutionContext() { return nullptr; }
  virtual void ReleaseExecutionContext(void* exec_ctx) { (void)exec_ctx; }

  virtual bool RunSync(void* exec_ctx,
                       std::vector<std::shared_ptr<void>> inputs,
                       std::vector<std::shared_ptr<void>> outputs) = 0;

  /**
   * @brief 运行期动态维度调用接口
   *
   * 输入侧与输出侧的形状契约分别传入
   *
   * @param[in]  in_shapes  输入侧契约，来自输入资源。后端只读 inputs[].actual，
   *                        即前处理在写数据时回填的本次尺寸。
   * @param[out] out_shapes 输出侧契约，来自输出资源。后端把本次真实输出尺寸
   *                        写进 outputs[].actual，供 D2H / 后处理消费。
   */
  virtual bool RunSync(void* exec_ctx,
                       std::vector<std::shared_ptr<void>> inputs,
                       std::vector<std::shared_ptr<void>> outputs,
                       RunShapes* in_shapes,
                       RunShapes* out_shapes) {
    (void)in_shapes;
    (void)out_shapes;
    return RunSync(exec_ctx, inputs, outputs);
  }

  /**
   * @brief 填充形状能力（静态部分）到 rs：min / max / actual
   *
   * 基类默认实现按当前的 InputShape/OutputShape 填充
   * 支持动态形状的后端覆写为 profile 的 kMIN/kMAX
   */
  virtual void FillRunShapeCaps(RunShapes* rs) {
    if (!rs) return;
    rs->dynamic = false;
    rs->inputs.clear();
    rs->outputs.clear();
    for (uint32_t i = 0; i < InputNum(); ++i) {
      RunShape s;
      s.min = s.max = s.actual = InputShape(i);
      rs->inputs.push_back(s);
    }
    for (uint32_t i = 0; i < OutputNum(); ++i) {
      RunShape s;
      s.min = s.max = s.actual = OutputShape(i);
      rs->outputs.push_back(s);
    }
  }

  /**
   * @brief 加载期缓存的形状能力
   */
  const RunShapes& GetRunShapeCaps() {
    if (!caps_init_) {
      FillRunShapeCaps(&caps_);
      caps_init_ = true;
    }
    return caps_;
  }

  /**
   * @brief 缓冲区尺寸所用的形状上限
   */
  virtual TensorShape InputShapeMax(uint32_t index) const { return InputShape(index); }
  virtual TensorShape OutputShapeMax(uint32_t index) const { return OutputShape(index); }

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 protected:
#endif
  std::string name_;  // model name
  std::string engine_path_;
  DevType     device_type_ = DevType::INVALID;
  int         device_id_ = -1;

  std::vector<TensorShape>   input_shapes_;
  std::vector<TensorShape>   output_shapes_;
  // 缓冲区尺寸上限（Dynamic 时来自 profile kMAX）；默认与上面相同，由后端按需填
  std::vector<TensorShape>   input_shapes_max_;
  std::vector<TensorShape>   output_shapes_max_;
  // 形状下界（Dynamic 时来自 profile kMIN）；默认与上面相同，前处理据此 clamp
  std::vector<TensorShape>   input_shapes_min_;
  std::vector<DataType>      input_data_types_;
  std::vector<DataType>      output_data_types_;
  std::vector<std::string>   input_names_;
  std::vector<std::string>   output_names_;
  
  std::map<std::string, int> bind_name_index_map_{};  // bind_name - index
  std::string                input_name_;             // one input name
  std::string                output_name_;            // one output name

  // note: 图像模型需要确定 input_tensor_index 才能确定 batch_size
  uint32_t                   input_ordered_index_;

  // 输入形状处理模式
  ShapeMode                  shape_mode_ = ShapeMode::STATIC;
  // 加载期缓存的形状能力，供 IOResource::Allocate() 拷贝
  RunShapes                  caps_;
  bool                       caps_init_ = false;
};


/**
 * @brief 仿照 MemOpFactory, 用于创建不同设备类型的 ModelLoader 实例
 */
class ModelLoaderFactory {
 public:

  static ModelLoaderFactory& Instance();

  bool RegisterModelLoaderCreator(DevType device_type,
                           std::function<std::unique_ptr<ModelLoader>(int device_id)> creator);

  std::unique_ptr<ModelLoader> CreateModelLoader(DevType device_type, int device_id);

 private:
  ModelLoaderFactory();
  ~ModelLoaderFactory();
  ModelLoaderFactory(const ModelLoaderFactory&) = delete;
  ModelLoaderFactory& operator=(const ModelLoaderFactory&) = delete;

 public:
  void PrintRegisteredCreators() {
    LOGI(MODEL_LOADER_FACTORY) << "PrintRegisteredCreators size: " << creators_.size();
    for (const auto& pair : creators_) {
      LOGI(MODEL_LOADER_FACTORY) << DevType2Str(pair.first) 
      << " -> Creator Func Address: " << &pair.second;
    }
  }

 private:
  struct DevTypeHash {
    template <typename T>
    std::size_t operator()(const T& device_type) const {
      return static_cast<std::size_t>(device_type);
    }
  };

  std::unordered_map<DevType, std::function<std::unique_ptr<ModelLoader>(int device_id)>, DevTypeHash> creators_ {};
  std::mutex mutex_;
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_MODEL_LOADER_HPP_
