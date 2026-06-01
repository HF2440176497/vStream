
#include "base.hpp"

#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

#include "data_source.hpp"
#include "data_handler_image.hpp"
#include "data_handler_pull.hpp"

#include "data_sink.hpp"

#include <chrono>
#include <typeinfo>

#include <opencv2/opencv.hpp>

#ifdef VSTREAM_USE_CUDA
#include "cuda/inspect_mem.hpp"
#endif

namespace cnstream {

static const std::string             source_module_name = "source";
static const std::string             sink_module_name = "sink";

static const std::string             stream_id_1_ = "channel-1";
static const std::string             stream_id_2_ = "channel-2";
static const std::string             stream_id_3_ = "channel-3";
static const std::string             stream_id_4_ = "channel-4";
static std::vector<std::string>      stream_ids_image_push_ = {stream_id_1_};
static std::vector<std::string>      stream_ids_pull_push_ = {stream_id_2_};
static std::vector<std::string>      stream_ids_image_queue_ = {stream_id_3_};
static std::vector<std::string>      stream_ids_send_queue_ = {stream_id_4_};

static std::string test_pipeline_json = "pipeline_source_base.json";
static std::string test_pipeline_video_json = "pipeline_source_video.json";

static std::string process_module_name = "count_one";

static bool has_save_frame_mat = false;
static std::string save_file = "save/test_source_save.jpg";


class EosObserver : public StreamMsgObserver {
 public:
  void Update(const StreamMsg &msg) override {}
};

class SourceBase : public testing::Test {
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
    image_handler_.reset();
  }

 protected:
  std::shared_ptr<ImageHandler> image_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
};

/**
 * @brief 测试进行硬解码
 */
class SourceVideo : public testing::Test {
 protected:
  virtual void SetUp() {
    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipelineByJSONFile(test_pipeline_video_json));
  }

  virtual void TearDown() {
    if (pipeline_) {
      pipeline_->Stop();
    }
    pull_handler_.reset();
  }

 protected:
  std::shared_ptr<PullHandler> pull_handler_ = nullptr;
  std::shared_ptr<PushHandler>  push_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
};

TEST_F(SourceBase, Init) {

  std::string json_content = readFile(test_pipeline_video_json.c_str());
  EXPECT_FALSE(json_content.empty()) << "Read json file failed";
  cnstream::CNGraphConfig graph_config;
  graph_config.ParseByJSONStr(json_content);

  std::unique_ptr<CNGraph<NodeContext>> graph = std::make_unique<CNGraph<NodeContext>>();
  EXPECT_NE(nullptr, graph.get());
  EXPECT_TRUE(graph->Init(graph_config));

  // 检查 Module 相关 mask 标志位
  // PS: 必须要在 Build 完成的 Pipeline 中看到
  LOGI(T_SOURCE) << "---------- Module Mask: " << std::endl;
  for (auto node_iter = pipeline_->graph_->DFSBegin(); node_iter != pipeline_->graph_->DFSEnd(); ++node_iter) {
    if (!node_iter->data.parent_nodes_mask) {  // head node
      LOGI(T_SOURCE) << "--- head node name: " << node_iter->data.module->GetName() << std::endl;
    } else {  // not head node
      LOGI(T_SOURCE) << "--- not head node name: " << node_iter->data.module->GetName() << std::endl;
    }
    LOGI(T_SOURCE) << "node name: " << node_iter->data.module->GetName() << std::endl;
    LOGI(T_SOURCE) << "module id: " << node_iter->data.module->GetId() << std::endl;
    LOGI(T_SOURCE) << "route_mask: " << node_iter->data.route_mask << std::endl;
    LOGI(T_SOURCE) << "parent_nodes_mask: " << node_iter->data.parent_nodes_mask << std::endl;
  }
  // DataSource 标记 route_mask 非 0, parent_nodes_mask 为 0
  // 其余 Module 标记 route_mask 为 0 (因为是头节点) parent_nodes_mask 非 0 

  // 发现：DataSource 的 route_mask 也包含了自身 Module 的标记

  std::vector<std::string> registed_modules = ModuleFactory::Instance()->GetRegisted();
  LOGI(T_SOURCE) << "-------- SourceBase module name: " << std::endl;
  for (auto& module_name : registed_modules) {
    LOGI(T_SOURCE) << "module name: " << module_name << std::endl;
  }
  EXPECT_TRUE(std::find(registed_modules.begin(), registed_modules.end(), "cnstream::DataSource") != registed_modules.end());
}

/**
 * 读取图片
 */
TEST_F(SourceBase, Run) {
  // 提取 pipeline 中的 DataSource 模块
  EXPECT_NE(pipeline_, nullptr);
  Module* module_in_pipeline = pipeline_->GetModule(source_module_name);
  EXPECT_NE(module_in_pipeline, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);
  EXPECT_NE(source, nullptr);

  EXPECT_TRUE(pipeline_->Start());

  if (!stream_ids_image_push_.empty()) {
    std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_ids_image_push_[0]);
    image_handler_ = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
    EXPECT_NE(image_handler_, nullptr);
    EXPECT_EQ(source->AddSource(image_handler_), 0);
  }
  auto stream_id = stream_ids_image_push_[0];

  EXPECT_TRUE(image_handler_->impl_->IsRunning());
  LOGI(T_SOURCE) << "Handler image_path = " << image_handler_->impl_->image_path_ << std::endl;
  LOGI(T_SOURCE) << "Handler frame_rate_ = " << image_handler_->impl_->frame_rate_ << std::endl;

  std::this_thread::sleep_for(std::chrono::seconds(2));
  LOGI(T_SOURCE) << "Handler stream idx: " << image_handler_->GetStreamIndex();
  EXPECT_NE(image_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // == data->stream_idx_
  EXPECT_TRUE(pipeline_->IsRunning());
  
  image_handler_->Stop();
  image_handler_->Close();
  
  LOGI(T_SOURCE) << "Wait for EOS message to be processed";
  LOGI(T_SOURCE) << "CheckStreamEosReached = " << std::boolalpha << CheckStreamEosReached(stream_id, true);
  LOGI(T_SOURCE) << "Wait for EOS message complete";
  
  pipeline_->Stop();
}

/**
 * 测试多个流，每个流处理各自的图像
 */
TEST_F(SourceBase, MutilStream) {
  Module* module_in_pipeline = pipeline_->GetModule(source_module_name);
  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);

  std::unordered_map<std::string, std::shared_ptr<ImageHandler>> handlers;

  EXPECT_TRUE(pipeline_->Start());

  for (auto stream_id : stream_ids_image_push_) {
    std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_id);
    auto handler = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
    EXPECT_NE(handler, nullptr);
    handlers[stream_id] = handler;
    EXPECT_EQ(source->AddSource(handlers[stream_id]), 0);
    EXPECT_TRUE(handlers[stream_id]->impl_->IsRunning());
  }

  for (auto stream_id : stream_ids_image_queue_) {
    std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_id);
    auto handler = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
    EXPECT_NE(handler, nullptr);
    handlers[stream_id] = handler;
    EXPECT_EQ(source->AddSource(handlers[stream_id]), 0);
    EXPECT_TRUE(handlers[stream_id]->impl_->IsRunning());
  }
  
  Module* module_process = pipeline_->GetModule(process_module_name);

  ASSERT_NE(module_process, nullptr);
  ASSERT_NE(module_process->GetConnector(), nullptr);
  int conveyor_count = module_process->GetConnector()->conveyor_count_;
  LOGI(T_SOURCE) << process_module_name << "connector conveyor count: " << conveyor_count << std::endl;
  
  // note: threads 含有 TaskLoop, 等于各个 Module 的 parallelism 的累加 
  LOGI(T_SOURCE) << "pipeline_->threads_.size() = " << pipeline_->threads_.size() << std::endl;

  // 运行开始，我们查看 Pipeline 内部：
  // （1）每个流的索引是否正确
  // （2）数据传输过程中的详细信息
  for (auto stream_id : stream_ids_image_push_) {
    LOGI(T_SOURCE) << "stream_id = " << stream_id << "; " << "stream_index = " << handlers[stream_id]->GetStreamIndex() << std::endl;
    EXPECT_EQ(handlers[stream_id]->GetStreamId(), stream_id);
    EXPECT_EQ(handlers[stream_id]->GetStreamIndex(), pipeline_->idxManager_->stream_idx_map[stream_id]);
    EXPECT_EQ(handlers[stream_id]->GetStreamIndex(), pipeline_->GetStreamIndex(stream_id));
  }

  for (auto stream_id : stream_ids_image_queue_) {
    LOGI(T_SOURCE) << "stream_id = " << stream_id << "; " << "stream_index = " << handlers[stream_id]->GetStreamIndex() << std::endl;
    EXPECT_EQ(handlers[stream_id]->GetStreamId(), stream_id);
    EXPECT_EQ(handlers[stream_id]->GetStreamIndex(), pipeline_->idxManager_->stream_idx_map[stream_id]);
    EXPECT_EQ(handlers[stream_id]->GetStreamIndex(), pipeline_->GetStreamIndex(stream_id));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  pipeline_->Stop();

}

/**
 * 单独使用一个 pipeline 测试 pull_handler
 */
TEST_F(SourceVideo, Run) {
  EXPECT_NE(pipeline_, nullptr);
  EXPECT_TRUE(pipeline_->Start());

  Module* module_in_pipeline = pipeline_->GetModule(source_module_name);
  EXPECT_NE(module_in_pipeline, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);
  EXPECT_NE(source, nullptr);

  for (auto stream_id : stream_ids_pull_push_) {
    std::shared_ptr<SourceHandler> source_handler_ptr = PullHandler::Create(source, stream_id);
    pull_handler_ = std::dynamic_pointer_cast<PullHandler>(source_handler_ptr);
    EXPECT_NE(pull_handler_, nullptr);
    EXPECT_EQ(source->AddSource(pull_handler_), 0);
  }
  Module* sink_module = pipeline_->GetModule(sink_module_name);
  EXPECT_NE(sink_module, nullptr);

  DataSink *sink = dynamic_cast<DataSink*>(sink_module);
  EXPECT_NE(sink, nullptr);

  for (auto stream_id : stream_ids_pull_push_) {
    std::shared_ptr<SinkHandler> sink_handler = PushHandler::Create(sink, stream_id);
    push_handler_ = std::dynamic_pointer_cast<PushHandler>(sink_handler);
    EXPECT_NE(push_handler_, nullptr);
    EXPECT_EQ(sink->AddSink(push_handler_), 0);
  }
  
  auto stream_id = stream_ids_pull_push_[0];
  
  LOGI(T_SOURCE) << "Handler stream_url_ = " << pull_handler_->impl_->stream_url_ << std::endl;
  LOGI(T_SOURCE) << "Handler frame_rate_ = " << pull_handler_->impl_->frame_rate_ << std::endl;
  
  if (pull_handler_->impl_->IsRunning()) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  LOGI(T_SOURCE) << "Handler stream idx: " << pull_handler_->GetStreamIndex();
  EXPECT_NE(pull_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // == data->stream_idx_
  
  pull_handler_->Stop();
  pull_handler_->Close();
  
  PrintStreamEos();
  std::this_thread::sleep_for(std::chrono::seconds(2));

  LOGI(T_SOURCE) << "Wait for EOS message to be processed";
  LOGI(T_SOURCE) << "CheckStreamEosReached = " << std::boolalpha << CheckStreamEosReached(stream_id, true);
  LOGI(T_SOURCE) << "Wait for EOS message complete";
  
  pipeline_->Stop();
}

}  // namespace cnstream