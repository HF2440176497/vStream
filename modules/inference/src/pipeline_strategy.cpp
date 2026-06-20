
#include "pipeline_strategy.hpp"

#include <memory>

#include "cnstream_logging.hpp"
#include "cpu_pipeline_strategy.hpp"
#include "cuda_pipeline_strategy.hpp"

namespace cnstream {

std::unique_ptr<PipelineStrategy> PipelineStrategy::Create(DevType device_type) {
  switch (device_type) {
    case DevType::CUDA:
      return std::make_unique<CudaPipelineStrategy>();
    case DevType::CPU:
      return std::make_unique<CpuPipelineStrategy>();
    case DevType::ROCKCHIP:
      return std::make_unique<CpuPipelineStrategy>();
    default:
      LOGW(INFER) << "PipelineStrategy: unsupported device type "
                  << DevType2Str(device_type) << ", fallback to CPU strategy.";
      return std::make_unique<CpuPipelineStrategy>();
  }
}

}  // namespace cnstream
