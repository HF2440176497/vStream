

#include "base.hpp"
#include "memop.hpp"
#include "memop_factory.hpp"
#include "data_source_param.hpp"
#include "cnstream_frame_va.hpp"
#include "data_source.hpp"
#include "data_handler_image.hpp"
#include "cnstream_pipeline.hpp"

#include "reflex_object.h"
#include "common.hpp"
#include "tensor.hpp"
#include "infer_params.hpp"
#include "infer_resource.hpp"
#include "model_loader.hpp"
#include "inference.hpp"

#include <opencv2/opencv.hpp>


namespace cnstream {

static std::string test_pipeline_json = "pipeline_inference.json";

class InferenceTest : public testing::Test {
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
    LOGI(InferenceTest) << "TearDown";
    if (pipeline_) {
      pipeline_->Stop();
    }
    image_handler_.reset();
  }

 protected:
  const std::string             stream_id_ = "channel-1";
  std::shared_ptr<ImageHandler> image_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
  cnstream::CNGraphConfig       graph_config_;
};

/**
 * 运行YOLO推理管道
 */
TEST_F(InferenceTest, RunYOLO) {

  // 首先验证前后处理的注册
  std::map<std::string, ClassInfo<ReflexObject>>& obj_map = CheckGlobalObjMap();
  for (auto it = obj_map.begin(); it != obj_map.end(); it++) {
    std::string name = it->first;
    LOGI(RUN_YOLO) << "REFLEX: obj_map name = " << name << std::endl;
  }
  
  Module* module_in_pipeline = pipeline_->GetModule("decoder");
  ASSERT_NE(module_in_pipeline, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);
  ASSERT_NE(source, nullptr);

  std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_id_);
  image_handler_ = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
  ASSERT_NE(image_handler_, nullptr);

  ASSERT_TRUE(pipeline_->Start());
  ASSERT_FALSE(IsStreamRemoved(stream_id_));  // 此处不应当被移除

  ASSERT_EQ(source->AddSource(image_handler_), 0);
  ASSERT_TRUE(image_handler_->impl_->running_);

  std::this_thread::sleep_for(std::chrono::milliseconds(2000));  // running for a while
  LOGI(InferenceTest) << "Handler stream idx: " << image_handler_->GetStreamIndex();
  EXPECT_NE(image_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // 等同 data->GetStreamIndex
  EXPECT_TRUE(pipeline_->IsRunning());
  
  PrintStreamEos();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  LOGI(InferenceTest) << "Wait for EOS message to be processed";
  LOGI(InferenceTest) << "CheckStreamEosReached(stream_id_) = " << std::boolalpha << CheckStreamEosReached(stream_id_, true);
  LOGI(InferenceTest) << "Wait for EOS message complete";
  
  // 直接调用 pipeline->stop 可以实现 source handler 的 stop
  pipeline_->Stop();

}


}  // namespace cnstream