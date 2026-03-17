
#ifndef MODULES_INFERENCE_MODEL_LOADER_HPP_
#define MODULES_INFERENCE_MODEL_LOADER_HPP_

#include <memory>
#include <string>
#include <vector>

namespace cnstream {

class ModelLoader {
 public:
  ModelLoader(const std::string& engine_path, DevType dev_type, int device_id = 0):
    engine_path_(engine_path), device_type_(dev_type), device_id_(device_id) {};
  virtual ~ModelLoader() = default;
  bool IsValid() = 0;

 public:
  int GetDeviceId() const { return device_id_; }
  DeviceType GetDeviceType() const { return device_type_; }

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

 protected:
  std::string engine_path_;
  DevType     device_type_ = DevType::INVALID;
  int         device_id_ = -1;
  // 保存模型映射信息
  std::vector<TensorShape>   input_shapes_;
  std::vector<TensorShape>   output_shapes_;
  std::vector<DataType>      input_data_types_;
  std::vector<DataType>      output_data_types_;
  std::vector<std::string>   input_names_;
  std::vector<std::string>   output_names_;
};


/**
 * @brief 仿照 MemOpFactory, 用于创建不同设备类型的 ModelLoader 实例
 */
class ModelLoaderFactory {
 public:

  static ModelLoaderFactory& Instance();

  bool RegisterModelLoaderCreator(DevType dev_type,
                           std::function<std::unique_ptr<ModelLoader>(int dev_id)> creator);

  std::unique_ptr<ModelLoader> CreateModelLoader(DevType dev_type, int dev_id);

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
    std::size_t operator()(const T& dev_type) const {
      return static_cast<std::size_t>(dev_type);
    }
  };

  std::unordered_map<DevType, std::function<std::unique_ptr<ModelLoader>(int dev_id)>, DevTypeHash> creators_ {};
  std::mutex mutex_;
};

using ModelLoaderPtr = std::shared_ptr<ModelLoader>;

}  // namespace cnstream

#endif  // MODULES_INFERENCE_CUDA_MODEL_LOADER_HPP_
