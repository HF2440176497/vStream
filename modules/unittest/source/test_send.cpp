

#include "base.hpp"

#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

#include "data_source.hpp"
#include "data_handler_image.hpp"
#include "data_handler_send.hpp"

#include "decode_queue.hpp"  // osd module

#include <atomic>
#include <chrono>
#include <thread>
#include <typeinfo>

static std::string test_pipeline_send_json = "pipeline_source_send.json";
static std::string test_image_path = "test_image.png";


class SourceSendTest : public testing::Test {

 protected:
  virtual void SetUp() {
    std::string json_content = readFile(test_pipeline_send_json.c_str());
    EXPECT_FALSE(json_content.empty()) << "Read json file failed";
    cnstream::CNGraphConfig graph_config;
    graph_config.ParseByJSONStr(json_content);
    graph_config_ = graph_config;

    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipeline(graph_config_));
  }

 protected:
  const std::string             stream_id_ = "channel-1";
  std::shared_ptr<SendHandler>  send_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
  cnstream::CNGraphConfig       graph_config_;

 protected:
   int send_count_ = 0;
   cv::Mat   image_;

};  // SourceSendTest


/*
 * @brief 启动线程读取图片，不断发送给 SendHandler
 */
TEST_F(SourceSendTest, TestSend) {

  Module* source_module = pipeline_->GetModule("decoder");
  EXPECT_NE(source_module, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(source_module);
  EXPECT_NE(source, nullptr);

  std::shared_ptr<SourceHandler> source_handler_ptr = SendHandler::Create(source, stream_id_);
  send_handler_ = std::dynamic_pointer_cast<SendHandler>(source_handler_ptr);
  EXPECT_NE(send_handler_, nullptr);

  EXPECT_EQ(source->AddSource(send_handler_), 0);
  EXPECT_TRUE(pipeline_->Start());

  Module* consume_module = pipeline_->GetModule("osd");  // name in pipeline json
  EXPECT_NE(consume_module, nullptr);

  DecodeQueue* decode_queue = dynamic_cast<DecodeQueue*>(consume_module);
  EXPECT_NE(decode_queue, nullptr);

  image_ = cv::imread(test_image_path, cv::IMREAD_COLOR);
  ASSERT_FALSE(image_.empty()) << "Failed to read test_image.png";

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
      s_output_data data = decode_queue->GetData();
      if (data.result != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      count++;
      if (count % 20 == 0) {
        LOGI(OSD) << "Receive: " << count << " frames; id_s: " << data.frame_id_s;
      }
    }
  });

  std::this_thread::sleep_for(std::chrono::seconds(5));
  
  pipeline_->Stop();

  running.store(false);
  send_thread.join();
  receive_thread.join();
}
