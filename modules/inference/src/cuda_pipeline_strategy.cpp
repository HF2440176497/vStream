
#include "cuda_pipeline_strategy.hpp"

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

PipelineConfig CudaPipelineStrategy::Build(ModelLoader* model, const InferOptions& options) {
  PipelineConfig config;

  const uint32_t batchsize = model->get_batch_size();
  const bool batching_by_obj = options.batching_by_obj();
  const bool postproc_on_device = options.postproc_on_device();

  if (batching_by_obj && postproc_on_device) {
    LOGE(INFER) << "postproc_on_device is true, but not allowed for obj processing";
    return config;
  }

  config.cpu_input_res = std::make_shared<CpuInputResource>(model);
  config.cpu_output_res = std::make_shared<CpuOutputResource>(model);
  config.net_input_res = std::make_shared<NetInputResource>(model);
  config.net_output_res = std::make_shared<NetOutputResource>(model);

  config.cpu_input_res->Init();
  config.cpu_output_res->Init();
  config.net_input_res->Init();
  config.net_output_res->Init();

  config.input_res = config.cpu_input_res;
  config.output_res = config.cpu_output_res;
  if (postproc_on_device) {
    config.output_res = config.net_output_res;
  }
  // 预处理阶段：结果写入 CPU 输入缓冲
  if (batching_by_obj) {
    config.obj_batching_stage =
        std::make_shared<CpuPreprocessingObjBatchingStage>(model, batchsize, options.obj_preprocessor(),
                                                           config.cpu_input_res);
  } else {
    config.batching_stage =
        std::make_shared<CpuPreprocessingBatchingStage>(model, batchsize, options.preprocessor(),
                                                        config.cpu_input_res);
  }

  auto h2d_stage = std::make_shared<H2DBatchingDoneStage>(model, batchsize,
                                                          config.cpu_input_res, config.net_input_res);
  h2d_stage->SetProfiler(options.profiler());
  config.batching_done_stages.push_back(h2d_stage);

  auto infer_stage = std::make_shared<InferBatchingDoneStage>(model, batchsize,
                                                              config.net_input_res, config.net_output_res);
  infer_stage->SetProfiler(options.profiler());
  infer_stage->SetDumpResizedImageDir(options.dump_resized_image_dir());
  infer_stage->SetSavingInputData(options.saving_infer_input(), options.module_name());
  config.batching_done_stages.push_back(infer_stage);

  // D2H：（若后处理在 CPU）
  if (!postproc_on_device) {
    auto d2h_stage = std::make_shared<D2HBatchingDoneStage>(model, batchsize,
                                                            config.net_output_res, config.cpu_output_res);
    d2h_stage->SetProfiler(options.profiler());
    config.batching_done_stages.push_back(d2h_stage);
  }

  if (batching_by_obj) {
    config.obj_postproc_stage =
        std::make_shared<ObjPostprocessingBatchingDoneStage>(model, batchsize,
                                                             options.obj_postprocessor(), config.cpu_output_res);
  } else {
    if (postproc_on_device) {
      auto postproc_stage =
          std::make_shared<PostprocessingBatchingDoneStage>(model, batchsize,
                                                            options.postprocessor(), config.net_output_res);
      postproc_stage->SetProfiler(options.profiler());
      postproc_stage->SetDumpResizedImageDir(options.dump_resized_image_dir());
      config.batching_done_stages.push_back(postproc_stage);
    } else {
      auto postproc_stage =
          std::make_shared<PostprocessingBatchingDoneStage>(model, batchsize,
                                                            options.postprocessor(), config.cpu_output_res);
      postproc_stage->SetProfiler(options.profiler());
      postproc_stage->SetDumpResizedImageDir(options.dump_resized_image_dir());
      config.batching_done_stages.push_back(postproc_stage);
    }
  }

  return config;
}

}  // namespace cnstream
