

#include "base.hpp"

#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

#include "data_source.hpp"
#include "data_handler_image.hpp"
#include "data_handler_send.hpp"

#include "infer_params.hpp"
#include "data_sink.hpp"

#include "common.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <typeinfo>

namespace cnstream {

static const std::string             source_module_name = "source";
static const std::string             inference_module_name = "inference";
static const std::string             sink_module_name = "sink";

static const std::string             stream_id_1_ = "channel-1";
static const std::string             stream_id_2_ = "channel-2";
static const std::string             stream_id_3_ = "channel-3";
static const std::string             stream_id_4_ = "channel-4";
static std::vector<std::string>      stream_ids_image_push_ = {stream_id_1_};
static std::vector<std::string>      stream_ids_pull_push_ = {stream_id_2_};
static std::vector<std::string>      stream_ids_image_queue_ = {stream_id_3_};
static std::vector<std::string>      stream_ids_send_queue_ = {stream_id_4_};

static std::string test_pipeline_send_json = "pipeline_source_send.json";
// static std::string test_pipeline_send_ocr_json = "OCR/pipeline_source_send_ocr.json";

static std::string test_image_path = "OCR/image.jpg";
static std::string test_image_folder = "OCR/images";
// static std::string test_image_path = "image.png";

class SourceSend : public testing::Test {

 protected:
  virtual void SetUp() {
    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipelineByJSONFile(test_pipeline_send_json));
  }

 protected:
  std::shared_ptr<SendHandler>  send_handler_ = nullptr;
  std::shared_ptr<QueueHandler> queue_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;

 protected:
   int send_count_ = 0;

};  // SourceSend


/*
 * @brief 启动线程读取图片，不断发送给 SendHandler
 */
TEST_F(SourceSend, Run) {
  EXPECT_TRUE(pipeline_->Start());

  Module* source_module = pipeline_->GetModule(source_module_name);
  EXPECT_NE(source_module, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(source_module);
  EXPECT_NE(source, nullptr);

  for (auto stream_id : stream_ids_send_queue_) {
    auto source_handler = SendHandler::Create(source, stream_id);
    EXPECT_NE(source_handler, nullptr);
    send_handler_ = std::dynamic_pointer_cast<SendHandler>(source_handler);
    EXPECT_NE(send_handler_, nullptr);
    EXPECT_EQ(source->AddSource(send_handler_), 0);
  }
  
  Module* sink_module = pipeline_->GetModule(sink_module_name);
  EXPECT_NE(sink_module, nullptr);

  DataSink *sink = dynamic_cast<DataSink*>(sink_module);
  EXPECT_NE(sink, nullptr);

  for (auto stream_id : stream_ids_send_queue_) {
    auto sink_handler = QueueHandler::Create(sink, stream_id);
    EXPECT_NE(sink_handler, nullptr);
    queue_handler_ = std::dynamic_pointer_cast<QueueHandler>(sink_handler);
    EXPECT_NE(queue_handler_, nullptr);
    EXPECT_EQ(sink->AddSink(queue_handler_), 0);
  }
  
  utils::ImageFolderReader image_loader(test_image_folder);
  std::atomic<bool> running{true};

  std::thread send_thread([&]() {
    while (running.load()) {
      uint64_t pts = get_timestamp_ms();

      cv::Mat image = image_loader.read();
      if (image.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        continue;
      }

      // frame_id_s start from 0
      send_handler_->Send(pts, std::to_string(send_count_), image);
      send_count_++;
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  std::thread receive_thread([&]() {
    int count = 0;
    while (running.load()) {
      s_output_data data = queue_handler_->GetData();
      if (data.result != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        continue;
      }
      count++;
      if (count % 2 == 0) {
        // LOGI(T_SEND) << "Receive: " << data;
        LOGI(T_SEND) << "Received: " << count << "; Send: " << send_count_ << "; frames; id_s: " << data.frame_id_s;
      }
    }
  });

  auto inference_module = pipeline_->GetModule(inference_module_name);
  ASSERT_NE(inference_module, nullptr);
  std::this_thread::sleep_for(std::chrono::seconds(10));

  auto profiler = inference_module->GetProfiler();
  if (profiler) {
    auto infer_profile = profiler->GetProcessProfile(kINFERENCE_PROFILER_NAME);
    auto model_profile = profiler->GetProcessProfile(kMODEL_PROFILER_NAME);
    auto module_profile = profiler->GetProcessProfile(kPROCESS_PROFILER_NAME);
    LOGI(T_STABLE) << "Inference Profile: " << infer_profile;
    LOGI(T_STABLE) << "Model Profile: " << model_profile;
    LOGI(T_STABLE) << "Module Profile: " << module_profile;
  }

  std::this_thread::sleep_for(std::chrono::seconds(20));
  pipeline_->Stop();

  running.store(false);
  send_thread.join();
  receive_thread.join();
}

}  // namespace cnstream
