

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

static const std::string             key_source_module_name = "source";
static const std::string             key_sink_module_name = "sink";
static const std::string             key_inference_module_name = "inference";

static const std::string             stream_id_1_ = "channel-1";
static const std::string             stream_id_2_ = "channel-2";
static const std::string             stream_id_3_ = "channel-3";
static const std::string             stream_id_4_ = "channel-4";
static std::vector<std::string>      stream_ids_image_push_ = {stream_id_1_};
static std::vector<std::string>      stream_ids_pull_push_ = {stream_id_2_};
static std::vector<std::string>      stream_ids_image_queue_ = {stream_id_3_};
static std::vector<std::string>      stream_ids_send_queue_ = {stream_id_4_};

static std::string test_pipeline_json = "pipeline_inference.json";

class InferenceTest : public testing::Test {
 protected:
  virtual void SetUp() {
    std::string json_content = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_content.empty()) << "Read json file failed";

    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipelineByJSONFile(test_pipeline_json));
  }

  virtual void TearDown() {
    if (pipeline_) {
      pipeline_->Stop();
    }
    image_handler_.reset();
  }

 protected:
  std::shared_ptr<ImageHandler> image_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
};

/**
 * 运行YOLO推理管道
 */
TEST_F(InferenceTest, RunYOLO) {

  // 首先验证前后处理的注册
  std::map<std::string, ClassInfo<ReflexObject>>& obj_map = check_reflex_map();
  for (auto it = obj_map.begin(); it != obj_map.end(); it++) {
    std::string name = it->first;
    LOGI(T_INFERENCE) << "REFLEX: obj_map name = " << name << std::endl;
  }
  
  ASSERT_TRUE(pipeline_->Start());

  Module* module_in_pipeline = pipeline_->GetModule(key_source_module_name);
  ASSERT_NE(module_in_pipeline, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);
  ASSERT_NE(source, nullptr);
  
  for (auto stream_id : stream_ids_image_push_) {
    auto source_handler = ImageHandler::Create(source, stream_id);
    image_handler_ = std::dynamic_pointer_cast<ImageHandler>(source_handler);
    ASSERT_NE(image_handler_, nullptr);
    ASSERT_FALSE(IsStreamRemoved(stream_id));
    EXPECT_EQ(source->AddSource(image_handler_), 0);
  }
  
  auto stream_id = stream_ids_image_push_[0];

  std::this_thread::sleep_for(std::chrono::seconds(2));
  LOGI(T_INFERENCE) << "Handler stream idx: " << image_handler_->GetStreamIndex();
  EXPECT_NE(image_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // 等同 data->GetStreamIndex
  EXPECT_TRUE(pipeline_->IsRunning());

  LOGI(T_INFERENCE) << "Wait for EOS message to be processed";
  LOGI(T_INFERENCE) << "CheckStreamEosReached = " << std::boolalpha << CheckStreamEosReached(stream_id, true);
  LOGI(T_INFERENCE) << "Wait for EOS message complete";
  
  // 直接调用 pipeline->stop 可以实现 source handler 的 stop
  pipeline_->Stop();

}


}  // namespace cnstream