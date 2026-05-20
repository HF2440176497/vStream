

#include "base.hpp"

#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

#include "data_source.hpp"
#include "data_handler_image.hpp"
#include "data_handler_send.hpp"

#include "data_sink.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <typeinfo>

namespace cnstream {

static const std::string             stream_id_1_ = "channel-1";
static const std::string             stream_id_2_ = "channel-2";
static const std::string             stream_id_3_ = "channel-3";
static const std::string             stream_id_4_ = "channel-4";
static std::vector<std::string>      stream_ids_image_push_ = {stream_id_1_};
static std::vector<std::string>      stream_ids_pull_push_ = {stream_id_2_};
static std::vector<std::string>      stream_ids_image_queue_ = {stream_id_3_};
static std::vector<std::string>      stream_ids_send_queue_ = {stream_id_4_};

static std::string test_pipeline_send_json = "pipeline_source_send.json";
static std::string test_image_path = "image.png";


class SourceSendTest : public testing::Test {

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
   cv::Mat   image_;

};  // SourceSendTest


/*
 * @brief 启动线程读取图片，不断发送给 SendHandler
 */
TEST_F(SourceSendTest, TestSend) {
  EXPECT_TRUE(pipeline_->Start());

  Module* source_module = pipeline_->GetModule("decoder");
  EXPECT_NE(source_module, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(source_module);
  EXPECT_NE(source, nullptr);

  for (auto stream_id : stream_ids_send_queue_) {
    std::shared_ptr<SourceHandler> source_handler_ptr = SendHandler::Create(source, stream_id);
    EXPECT_NE(source_handler_ptr, nullptr);
    send_handler_ = std::dynamic_pointer_cast<SendHandler>(source_handler_ptr);
    EXPECT_NE(send_handler_, nullptr);
    EXPECT_EQ(source->AddSource(send_handler_), 0);
  }
  
  Module* sink_module = pipeline_->GetModule("sink");
  EXPECT_NE(sink_module, nullptr);

  DataSink *sink = dynamic_cast<DataSink*>(sink_module);
  EXPECT_NE(sink, nullptr);

  for (auto stream_id : stream_ids_send_queue_) {
    std::shared_ptr<SinkHandler> sink_handler_ptr = QueueHandler::Create(sink, stream_id);
    EXPECT_NE(sink_handler_ptr, nullptr);
    queue_handler_ = std::dynamic_pointer_cast<QueueHandler>(sink_handler_ptr);
    EXPECT_NE(queue_handler_, nullptr);
    EXPECT_EQ(sink->AddSink(queue_handler_), 0);
  }
  
  image_ = cv::imread(test_image_path, cv::IMREAD_COLOR);
  ASSERT_FALSE(image_.empty()) << "Failed to load " << test_image_path;

  std::atomic<bool> running{true};

  std::thread send_thread([&]() {
    while (running.load()) {
      uint64_t pts = get_timestamp_ms();

      // frame_id_s start from 0
      send_handler_->Send(pts, std::to_string(send_count_), image_);
      send_count_++;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });

  std::thread receive_thread([&]() {
    int count = 0;
    while (running.load()) {
      s_output_data data = queue_handler_->GetData();
      if (data.result != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      count++;
      if (count % 20 == 0) {
        LOGI(T_SEND) << "Receive: " << count << " frames; id_s: " << data.frame_id_s;
      }
    }
  });

  std::this_thread::sleep_for(std::chrono::seconds(10));
  pipeline_->Stop();

  running.store(false);
  send_thread.join();
  receive_thread.join();
}

}  // namespace cnstream
