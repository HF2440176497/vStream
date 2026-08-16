
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

namespace {
// 异步流水线 slot 深度
constexpr int kAsyncSlotDepth = 3;
}  // namespace

PipelineConfig CudaPipelineStrategy::Build(ModelLoader* model, const InferOptions& options) {
  PipelineConfig config;

  const uint32_t batchsize = model->get_batch_size();
  const bool batching_by_obj = options.batching_by_obj();
  const bool postproc_on_device = options.postproc_on_device();

  if (batching_by_obj && postproc_on_device) {
    LOGE(INFER) << "postproc_on_device is true, but not allowed for obj processing";
    return config;
  }

  // 平台支持时启用异步流水线：为每个 slot 创建独立执行上下文与执行流；
  // 未启用时池深为 1，票据语义退化为单 buffer 串行，行为与改造前一致。
  // 拓扑统一为 H2D -> Infer -> D2H -> Post
  
  // const bool async_infer = false;
  const bool async_infer = model->EnableAsyncInfer(kAsyncSlotDepth);
  const uint32_t res_pool_size = async_infer ? kAsyncSlotDepth : 1;

  config.cpu_input_res = std::make_shared<CpuInputResource>(model);
  config.cpu_output_res = std::make_shared<CpuOutputResource>(model);
  config.net_input_res = std::make_shared<NetInputResource>(model);
  config.net_output_res = std::make_shared<NetOutputResource>(model);

  config.cpu_input_res->SetResPoolSize(res_pool_size);
  config.net_input_res->SetResPoolSize(res_pool_size);
  config.net_output_res->SetResPoolSize(res_pool_size);
  config.cpu_output_res->SetResPoolSize(res_pool_size);

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

  // H2D：cpu_input 链式接入预处理 run，net_input 开启新 run
  auto h2d_stage = std::make_shared<H2DBatchingDoneStage>(model, batchsize,
                                                          config.cpu_input_res, config.net_input_res);
  h2d_stage->SetProfiler(options.profiler());
  config.batching_done_stages.push_back(h2d_stage);

  // Infer：input 链式接入 H2D run，output 开启新 run；
  // 内部 RunAsync 提交至 slot 流 + SyncEvent 等待，平台未启用时回退 RunSync
  auto infer_stage = std::make_shared<InferBatchingDoneStage>(model, batchsize,
                                                              config.net_input_res, config.net_output_res);
  infer_stage->SetProfiler(options.profiler());
  infer_stage->SetDumpResizedImageDir(options.dump_resized_image_dir());
  infer_stage->SetSavingInputData(options.saving_infer_input(), options.module_name());
  config.batching_done_stages.push_back(infer_stage);

  // D2H：net_output 链式接入 Infer run，cpu_output 开启新 run（后处理在 CPU 时）
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
