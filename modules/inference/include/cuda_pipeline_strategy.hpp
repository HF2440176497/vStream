
#ifndef MODULES_INFERENCE_CUDA_PIPELINE_STRATEGY_HPP_
#define MODULES_INFERENCE_CUDA_PIPELINE_STRATEGY_HPP_

#include "pipeline_strategy.hpp"

namespace cnstream {

/**
 * @brief CUDA / TensorRT 平台流水线组装策略。
 *
 * 完整流水线：Preproc(CPU) -> H2D -> Infer(GPU) -> D2H -> Postproc(CPU)。
 */
class CudaPipelineStrategy : public PipelineStrategy {
 public:
  PipelineConfig Build(ModelLoader* model, const InferOptions& options) override;
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_CUDA_PIPELINE_STRATEGY_HPP_
