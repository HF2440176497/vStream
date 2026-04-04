
#ifndef MODULES_INFERENCE_MODEL_LOADER_HPP_
#define MODULES_INFERENCE_MODEL_LOADER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "data_source_param.hpp"
#include "tensor.hpp"

namespace cnstream {

class ModelLoader {
 public:
  ModelLoader(DevType device_type, int device_id = 0):
    device_type_(device_type), device_id_(device_id) {};
  virtual ~ModelLoader() = default;
  virtual bool IsValid() = 0;
  virtual bool Init(const std::string& engine_path) = 0;

 public:
  int GetDeviceId() const { return device_id_; }
  DevType GetDeviceType() const { return device_type_; }

  /** 单输入模型，获取对应的信息 */
  uint32_t get_batch_size() const { return input_shapes_[input_ordered_index_].N(); }
  uint32_t get_channel_size() const { return input_shapes_[input_ordered_index_].C(); }
  uint32_t get_height_size() const { return input_shapes_[input_ordered_index_].H(); }
  uint32_t get_width_size() const { return input_shapes_[input_ordered_index_].W(); }

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
  int get_output_ordered_index() const { return output_ordered_index_; }

  size_t GetInputDataBatchAlignSize(uint32_t index) const {
    return InputShape(index).DataCount() * data_type_size(input_data_types_[index]);
  }

  size_t GetOutputDataBatchAlignSize(uint32_t index) const {
    return OutputShape(index).DataCount() * data_type_size(output_data_types_[index]);
  }

  virtual bool RunSync(std::vector<std::shared_ptr<void>> inputs, std::vector<std::shared_ptr<void>> outputs) = 0;

#ifdef UNIT_TEST
 public:
#else
 protected:
#endif
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
  std::string                input_name_;             // frist input name
  std::string                output_name_;            // frist output name
  int                        input_ordered_index_ = 0;
  int                        output_ordered_index_ = 0;
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
      LOGI(MODEL_LOADER_FACTORY) << "DevType: " << DevType2Str(pair.first) << " -> Creator Func Address: " << &pair.second;
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
