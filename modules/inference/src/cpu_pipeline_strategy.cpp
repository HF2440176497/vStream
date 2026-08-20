

#include "cpu_pipeline_strategy.hpp"

#include <memory>

#include "batching_done_stage.hpp"
#include "batching_stage.hpp"
#include "cnstream_logging.hpp"
#include "infer_options.hpp"
#include "infer_resource.hpp"
#include "model_loader.hpp"
#include "obj_batching_stage.hpp"
#include "postproc.hpp"
#include "preproc.hpp"

namespace cnstream {

PipelineConfig CpuPipelineStrategy::Build(ModelLoader* model, const InferOptions& options) {
  PipelineConfig config;

  const uint32_t batchsize = model->get_batch_size();
  const bool batching_by_obj = options.batching_by_obj();

  if (options.postproc_on_device()) {
    LOGE(INFER) << "postproc_on_device is true, but not allowed for CPU inference";
    return config;
  }

  config.cpu_input_res = std::make_shared<CpuInputResource>(model);
  config.cpu_output_res = std::make_shared<CpuOutputResource>(model);

  // 用于票据等待看门狗日志：定位哪个模块的哪类缓冲发生阻塞
  const std::string res_prefix = options.module_name() + "/";
  config.cpu_input_res->SetName(res_prefix + "cpu_input");
  config.cpu_output_res->SetName(res_prefix + "cpu_output");

  config.cpu_input_res->Init();
  config.cpu_output_res->Init();

  config.input_res = config.cpu_input_res;
  config.output_res = config.cpu_output_res;
  config.net_input_res = nullptr;
  config.net_output_res = nullptr;

  // 预处理阶段
  if (batching_by_obj) {
    config.obj_batching_stage =
        std::make_shared<CpuPreprocessingObjBatchingStage>(model, batchsize, options.obj_preprocessor(),
                                                           config.cpu_input_res);
  } else {
    config.batching_stage =
        std::make_shared<CpuPreprocessingBatchingStage>(model, batchsize, options.preprocessor(),
                                                        config.cpu_input_res);
  }

  // 推理阶段：直接读写 CPU 缓冲，无需 H2D/D2H
  auto infer_stage = std::make_shared<InferBatchingDoneStage>(model, batchsize,
                                                              config.input_res, config.output_res);
  infer_stage->SetProfiler(options.profiler());
  infer_stage->SetDumpResizedImageDir(options.dump_resized_image_dir());
  infer_stage->SetSavingInputData(options.saving_infer_input(), options.module_name());
  config.batching_done_stages.push_back(infer_stage);

  // 后处理阶段
  if (batching_by_obj) {
    config.obj_postproc_stage =
        std::make_shared<ObjPostprocessingBatchingDoneStage>(model, batchsize,
                                                             options.obj_postprocessor(), config.cpu_output_res);
  } else {
    auto postproc_stage =
        std::make_shared<PostprocessingBatchingDoneStage>(model, batchsize,
                                                          options.postprocessor(), config.cpu_output_res);
    postproc_stage->SetProfiler(options.profiler());
    postproc_stage->SetDumpResizedImageDir(options.dump_resized_image_dir());
    config.batching_done_stages.push_back(postproc_stage);
  }

  return config;
}

}  // namespace cnstream
