
#ifndef MODULES_INFERENCE_CPU_PIPELINE_STRATEGY_HPP_
#define MODULES_INFERENCE_CPU_PIPELINE_STRATEGY_HPP_

#include "pipeline_strategy.hpp"

namespace cnstream {

/**
 * @brief CPU / host-visible 推理平台流水线组装策略。
 *
 * 推理输入输出均直接位于 host 内存，无需 H2D/D2H 拷贝。
 * 流水线简化为：Preproc -> Infer -> Postproc。
 */
class CpuPipelineStrategy : public PipelineStrategy {
 public:
  PipelineConfig Build(ModelLoader* model, const InferOptions& options) override;
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_CPU_PIPELINE_STRATEGY_HPP_
