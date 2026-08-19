

#ifndef MODULES_INFERENCE_ROCKCHIP_MODEL_LOADER_RKNN_HPP_
#define MODULES_INFERENCE_ROCKCHIP_MODEL_LOADER_RKNN_HPP_

#include <rknn_api.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "infer_params.hpp"
#include "model_loader.hpp"

namespace cnstream {

inline constexpr std::string key_want_float = "want_float";
inline constexpr std::string key_core_mask = "core_mask";


class ModelLoaderRknn : public ModelLoader {
 public:
  explicit ModelLoaderRknn(int device_id);
  ~ModelLoaderRknn() override;

  bool Init(const std::string& engine_path, const InferParams& params) override;
  bool IsValid() override { return rknn_ctx_ != 0; }
  bool RunSync(void* exec_ctx,
               std::vector<std::shared_ptr<void>> inputs,
               std::vector<std::shared_ptr<void>> outputs) override;

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  bool LoadModel(const std::string& engine_path);
  bool QueryTensorInfo();
  bool ParseInputOutputAttr();
  bool SetCoreMask(int core_mask);

  rknn_context rknn_ctx_ = 0;
  rknn_input_output_num io_num_;
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  bool is_quant_ = false;        // 模型是否量化（仅作信息记录）
  bool want_float_ = true;       // RKNN 输出是否转为 float32
  int core_mask_ = 0;            // RKNN NPU core mask

  std::mutex mutex_;
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_ROCKCHIP_MODEL_LOADER_RKNN_HPP_
