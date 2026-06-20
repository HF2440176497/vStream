
#ifndef MODULES_INFERENCE_PIPELINE_STRATEGY_HPP_
#define MODULES_INFERENCE_PIPELINE_STRATEGY_HPP_

#include <memory>
#include <vector>

#include "batching_done_stage.hpp"
#include "batching_stage.hpp"
#include "data_source_param.hpp"
#include "infer_options.hpp"
#include "infer_resource.hpp"
#include "obj_batching_stage.hpp"

namespace cnstream {

class ModelLoader;

/**
 * @brief 流水线组装结果。
 *        PipelineStrategy 根据设备类型创建对应的资源与阶段
 */
struct PipelineConfig {
  std::shared_ptr<BatchingStage> batching_stage = nullptr;
  std::shared_ptr<ObjBatchingStage> obj_batching_stage = nullptr;
  std::vector<std::shared_ptr<BatchingDoneStage>> batching_done_stages;
  std::shared_ptr<ObjPostprocessingBatchingDoneStage> obj_postproc_stage = nullptr;

  // 通用资源句柄
  std::shared_ptr<IOResource> input_res = nullptr;
  std::shared_ptr<IOResource> output_res = nullptr;

  // 具体资源指针，保留用于兼容与调试
  std::shared_ptr<CpuInputResource> cpu_input_res = nullptr;
  std::shared_ptr<CpuOutputResource> cpu_output_res = nullptr;
  std::shared_ptr<NetInputResource> net_input_res = nullptr;
  std::shared_ptr<NetOutputResource> net_output_res = nullptr;
};

class PipelineStrategy {
 public:
  virtual ~PipelineStrategy() = default;
  virtual PipelineConfig Build(ModelLoader* model, const InferOptions& options) = 0;
  static std::unique_ptr<PipelineStrategy> Create(DevType device_type);
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_PIPELINE_STRATEGY_HPP_
