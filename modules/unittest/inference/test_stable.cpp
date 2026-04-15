
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
#include "data_handler_image.hpp"
#include "cnstream_pipeline.hpp"

#include "cuda/inspect_mem.hpp"

#include "common.hpp"
#include "inference.hpp"

#include <opencv2/opencv.hpp>


namespace cnstream {

static std::string test_pipeline_json = "pipeline_stable.json";

class StableTest : public testing::Test {
 protected:
  virtual void SetUp() {
    std::string json_content = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_content.empty()) << "Read json file failed";
    cnstream::CNGraphConfig graph_config;
    graph_config.config_root_dir = "./";
    graph_config.ParseByJSONStr(json_content);
    graph_config_ = graph_config;

    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipeline(graph_config_));
  }

  virtual void TearDown() {  // 当前用例结束
    LOGI(StableTest) << "TearDown";
    if (pipeline_) {
      pipeline_->Stop();
    }
    image_handler_.reset();
  }

 protected:
  const std::string             stream_id_1_ = "channel-1";
  const std::string             stream_id_2_ = "channel-2";
  const std::string             stream_id_3_ = "channel-3";
  std::vector<std::string>      stream_ids_ = {stream_id_1_, stream_id_2_, stream_id_3_};

  std::shared_ptr<ImageHandler> image_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
  cnstream::CNGraphConfig       graph_config_;
};


TEST_F(StableTest, MultiStream) {

  GPUInspect inspect(0);
  bool force_exit = false;

  DataSource *source = dynamic_cast<DataSource*>(pipeline_->GetModule("decoder"));
  EXPECT_NE(source, nullptr);

  for (auto stream_id : stream_ids_) {
    std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_id);
    auto handler = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
    EXPECT_NE(handler, nullptr);
    EXPECT_EQ(source->AddSource(handler), 0);
    EXPECT_TRUE(handler->impl_->running_);
  }

  auto inference_module = pipeline_->GetModule("Inference");
  EXPECT_NE(inference_module, nullptr);

  EXPECT_TRUE(pipeline_->Start());
  std::this_thread::sleep_for(std::chrono::seconds(10));

  auto profiler = inference_module->GetProfiler();
  if (profiler) {
    ModuleProfile profile = profiler->GetProfile();
    LOGI(StableTest) << "Inference profile: " << profile;
  }

  std::this_thread::sleep_for(std::chrono::seconds(10));
  LOGI(StableTest) << "Inspect brief info: " << inspect.GetBriefInfo();;

  std::this_thread::sleep_for(std::chrono::seconds(10));
  if (!force_exit) {
    pipeline_->Stop();
  } else {
    system("pause");
  }

}


}  // namespace cnstream