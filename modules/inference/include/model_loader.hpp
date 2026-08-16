
#ifndef MODULES_INFERENCE_MODEL_LOADER_HPP_
#define MODULES_INFERENCE_MODEL_LOADER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "data_source_param.hpp"
#include "tensor.hpp"
#include "infer_params.hpp"

namespace cnstream {

class ModelLoader {
 public:
  ModelLoader(int device_id = -1): device_id_(device_id) {
    device_type_ = DevType::CPU;
  }
  virtual ~ModelLoader() = default;
  virtual bool IsValid() = 0;
  virtual bool Init(const std::string& engine_path, const InferParams& params) = 0;
  virtual void SetInputOrderedIndex(int input_index) {
    input_ordered_index_ = input_index;
  }
  const std::string& get_name() { return name_; };

  public:
  int GetDeviceId() const { return device_id_; }
  DevType GetDeviceType() const { return device_type_; }
  virtual void* GetStream() const { return nullptr; }

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

  virtual bool RunSync(std::vector<std::shared_ptr<void>> inputs, std::vector<std::shared_ptr<void>> outputs) = 0;

  /**
   * @brief 启用异步推理流水线（平台可选实现）
   * @param slot_num 执行 slot 数量（与推理资源池深度一致）
   * @return false 表示平台不支持（如 RKNN），调用方应维持 RunSync 串行链路
   * @note 支持的平台会为每个 slot 创建独立的执行上下文与执行流；
   *       同一 loader 被多个 InferEngine 共享时须可重入（已启用且 slot 数足够则直接返回 true）
   */
  virtual bool EnableAsyncInfer(int slot_num) { (void)slot_num; return false; }

  /**
   * @brief 查询 slot 绑定的执行流（EnableAsyncInfer 成功后有效）
   * @return 未启用异步或 slot 越界时返回 nullptr，调用方应回退 GetStream()
   */
  virtual void* GetSlotStream(int slot) const { (void)slot; return nullptr; }

  /**
   * @brief 异步推理：将推理任务提交到 stream 并立即返回
   * @param stream GetSlotStream 返回的 slot 执行流，为 nullptr 时回退同步语义
   * @return 完成事件句柄，供后续阶段同步；返回 nullptr 表示已同步完成（回退路径）
   * @note 基类默认实现为同步 RunSync
   */
  virtual void* RunAsync(const std::vector<std::shared_ptr<void>>& inputs,
                         const std::vector<std::shared_ptr<void>>& outputs,
                         void* stream) {
    (void)stream;
    RunSync(inputs, outputs);
    return nullptr;
  }

  /**
   * @brief 等待 RunAsync 返回的事件完成（阻塞至推理结束）
   * @note 仅在 RunAsync 返回非空事件后调用；基类默认空实现
   */
  virtual void SyncEvent(void* event) { (void)event; }

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
  std::vector<DataType>      input_data_types_;
  std::vector<DataType>      output_data_types_;
  std::vector<std::string>   input_names_;
  std::vector<std::string>   output_names_;
  
  std::map<std::string, int> bind_name_index_map_{};  // bind_name - index
  std::string                input_name_;             // one input name
  std::string                output_name_;            // one output name

  // note: 图像模型需要确定 input_tensor_index 才能确定 batch_size
  uint32_t                   input_ordered_index_;
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
      LOGI(MODEL_LOADER_FACTORY) << DevType2Str(pair.first) << " -> Creator Func Address: " << &pair.second;
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
