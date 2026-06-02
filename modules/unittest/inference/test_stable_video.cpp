
/**
 * 测试稳定性:
 * Infer 模块并行度 = 2, 检验在 stream 间的是否存在干扰
 * 检查高帧率情况下的队列, 检查性能统计, 队列长度情况
 * 检查频繁启停, 内存使用情况
 */

#include "base.hpp"
#include "data_source_param.hpp"
#include "cnstream_frame_va.hpp"

#include "data_source.hpp"
#include "data_handler_pull.hpp"
#include "cnstream_pipeline.hpp"

#include "data_sink.hpp"

#include "common.hpp"
#include "inference.hpp"
#include "infer_params.hpp"
#include "cuda/inspect_mem.hpp"

#include <csignal>
#include <opencv2/opencv.hpp>


namespace cnstream {

static const std::string             source_module_name = "source";
static const std::string             sink_module_name = "sink";
static const std::string             inference_module_name = "inference";

static const std::string             stream_id_1_ = "channel-1";
static const std::string             stream_id_2_ = "channel-2";
static const std::string             stream_id_3_ = "channel-3";
static const std::string             stream_id_4_ = "channel-4";
static std::vector<std::string>      stream_ids_image_push_ = {stream_id_1_};
static std::vector<std::string>      stream_ids_pull_push_ = {stream_id_2_};
static std::vector<std::string>      stream_ids_image_queue_ = {stream_id_3_};
static std::vector<std::string>      stream_ids_send_queue_ = {stream_id_4_};

static std::string test_pipeline_json = "pipeline_stable_video.json";

class StableVideo : public testing::Test {
 protected:
  virtual void SetUp() {
    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipelineByJSONFile(test_pipeline_json));
  }

  virtual void TearDown() {
    if (pipeline_) {
      pipeline_->Stop();
    }
    pull_handler_.reset();
  }

 protected:
  std::shared_ptr<PullHandler> pull_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
};


TEST_F(StableVideo, Run) {
  bool force_exit = false;
  EXPECT_TRUE(pipeline_->Start());

  DataSource *source = dynamic_cast<DataSource*>(pipeline_->GetModule(source_module_name));
  EXPECT_NE(source, nullptr);

  for (auto stream_id : stream_ids_pull_push_) {
    auto source_handler = PullHandler::Create(source, stream_id);
    auto handler = std::dynamic_pointer_cast<PullHandler>(source_handler);
    EXPECT_NE(handler, nullptr);
    EXPECT_EQ(source->AddSource(handler), 0);
    EXPECT_TRUE(handler->impl_->IsRunning());
  }

  DataSink *sink = dynamic_cast<DataSink*>(pipeline_->GetModule(sink_module_name));
  EXPECT_NE(sink, nullptr);

  for (auto stream_id : stream_ids_pull_push_) {
    auto sink_handler = PushHandler::Create(sink, stream_id);
    auto push_handler = std::dynamic_pointer_cast<PushHandler>(sink_handler);
    EXPECT_NE(push_handler, nullptr);
    EXPECT_EQ(sink->AddSink(push_handler), 0);
  }

  auto inference_module = pipeline_->GetModule(inference_module_name);
  EXPECT_NE(inference_module, nullptr);

  std::this_thread::sleep_for(std::chrono::seconds(30));

  auto profiler = inference_module->GetProfiler();
  if (profiler) {
    auto infer_profile = profiler->GetProcessProfile(kINFERENCE_PROFILER_NAME);
    auto module_profile = profiler->GetProcessProfile(kPROCESS_PROFILER_NAME);
    LOGI(T_STABLE) << "Inference Profile: " << infer_profile;
    LOGI(T_STABLE) << "Module Profile: " << module_profile;
  }

  std::this_thread::sleep_for(std::chrono::seconds(180));
  if (!force_exit) {
    pipeline_->Stop();
  } else {
    system("pause");
  }
}

}  // namespace cnstream
