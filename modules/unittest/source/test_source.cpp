
#include "base.hpp"

#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

#include "data_source.hpp"
#include "data_handler_image.hpp"
#include "data_handler_video.hpp"

#include "data_sink.hpp"

#include <chrono>
#include <typeinfo>

#include <opencv2/opencv.hpp>

#ifdef VSTREAM_USE_CUDA
#include "cuda/inspect_mem.hpp"
#endif

namespace cnstream {

static std::string test_pipeline_json = "pipeline_source_base.json";
static std::string test_pipeline_video_json = "pipeline_source_video.json";

std::string process_module_name = "count_one";

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
  const std::string             stream_id_ = "channel-1";
  std::shared_ptr<ImageHandler> image_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
};

/**
 * @brief 测试 VideoSourceHandler 进行硬解码
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
    video_handler_.reset();
  }

 protected:
  const std::string stream_id_ = "channel-1";
  std::shared_ptr<VideoHandler> video_handler_ = nullptr;
  std::shared_ptr<PushHandler> push_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
};

TEST_F(SourceBase, PipelineInit) {

  std::string json_content = readFile(test_pipeline_video_json.c_str());
  EXPECT_FALSE(json_content.empty()) << "Read json file failed";
  cnstream::CNGraphConfig graph_config;
  graph_config.ParseByJSONStr(json_content);

  std::unique_ptr<CNGraph<NodeContext>> graph = std::make_unique<CNGraph<NodeContext>>();
  EXPECT_NE(nullptr, graph.get());
  EXPECT_TRUE(graph->Init(graph_config));

  // 检查 Module 相关 mask 标志位
  // PS: 必须要在 Build 完成的 Pipeline 中看到
  std::cout << "---------- Module Mask: " << std::endl;
  for (auto node_iter = pipeline_->graph_->DFSBegin(); node_iter != pipeline_->graph_->DFSEnd(); ++node_iter) {
    if (!node_iter->data.parent_nodes_mask) {  // head node
      std::cout << "--- head node name: " << node_iter->data.module->GetName() << std::endl;
    } else {  // not head node
      std::cout << "--- not head node name: " << node_iter->data.module->GetName() << std::endl;
    }
    std::cout << "node name: " << node_iter->data.module->GetName() << std::endl;
    std::cout << "module id: " << node_iter->data.module->GetId() << std::endl;
    std::cout << "route_mask: " << node_iter->data.route_mask << std::endl;
    std::cout << "parent_nodes_mask: " << node_iter->data.parent_nodes_mask << std::endl;
  }
  // DataSource 标记 route_mask 非 0, parent_nodes_mask 为 0
  // 其余 Module 标记 route_mask 为 0 (因为是头节点) parent_nodes_mask 非 0 

  // 发现：DataSource 的 route_mask 也包含了自身 Module 的标记

  std::vector<std::string> registed_modules = ModuleFactory::Instance()->GetRegisted();
  std::cout << "-------- SourceBase module name: " << std::endl;
  for (auto& module_name : registed_modules) {
    std::cout << "module name: " << module_name << std::endl;
  }
  EXPECT_TRUE(std::find(registed_modules.begin(), registed_modules.end(), "cnstream::DataSource") != registed_modules.end());
}

/**
 * 读取图片
 */
TEST_F(SourceBase, RUN) {
  // 提取 pipeline 中的 DataSource 模块
  EXPECT_NE(pipeline_, nullptr);
  Module* module_in_pipeline = pipeline_->GetModule("decoder");
  EXPECT_NE(module_in_pipeline, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);
  EXPECT_NE(source, nullptr);

  std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_id_);
  image_handler_ = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
  EXPECT_NE(image_handler_, nullptr);

  EXPECT_TRUE(pipeline_->Start());
  EXPECT_EQ(source->AddSource(image_handler_), 0);

  EXPECT_TRUE(image_handler_->impl_->IsRunning());
  std::cout << "image_handler_->impl_->image_path = " << image_handler_->impl_->image_path_ << std::endl;
  std::cout << "image_handler_->impl_->frame_rate_ = " << image_handler_->impl_->frame_rate_ << std::endl;

  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  LOGI(T_SOURCE) << "Handler stream idx: " << image_handler_->GetStreamIndex();
  EXPECT_NE(image_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // == data->stream_idx_
  EXPECT_TRUE(pipeline_->IsRunning());
  
  image_handler_->Stop();
  image_handler_->Close();
  
  PrintStreamEos();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  
  LOGI(T_SOURCE) << "Wait for EOS message to be processed";
  LOGI(T_SOURCE) << "CheckStreamEosReached(stream_id_) = " << std::boolalpha << CheckStreamEosReached(stream_id_, true);
  LOGI(T_SOURCE) << "Wait for EOS message complete";
  
  pipeline_->Stop();
}

/**
 * 测试多个流，每个流处理各自的图像
 */
TEST_F(SourceBase, MutilStream) {
  Module* module_in_pipeline = pipeline_->GetModule("decoder");
  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);

  std::vector<std::string> stream_ids = {"channel-1", "channel-2"};
  std::unordered_map<std::string, std::shared_ptr<ImageHandler>> handlers;

  for (auto stream_id : stream_ids) {
    std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_id);
    auto handler = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
    EXPECT_NE(handler, nullptr);
    handlers[stream_id] = handler;
  }

  EXPECT_TRUE(pipeline_->Start());
  for (auto stream_id : stream_ids) {
    EXPECT_EQ(source->AddSource(handlers[stream_id]), 0);
    EXPECT_TRUE(handlers[stream_id]->impl_->IsRunning());
  }
  
  Module* module_process = pipeline_->GetModule(process_module_name);

  ASSERT_NE(module_process, nullptr);
  ASSERT_NE(module_process->GetConnector(), nullptr);
  int conveyor_count = module_process->GetConnector()->conveyor_count_;
  std::cout << "Process Module connector conveyor count: " << conveyor_count << std::endl;
  
  // note: threads 含有 TaskLoop, 等于各个 Module 的 parallelism 的累加 
  std::cout << "pipeline_->threads_.size() = " << pipeline_->threads_.size() << std::endl;

  // 运行开始，我们查看 Pipeline 内部：
  // （1）每个流的索引是否正确
  // （2）数据传输过程中的详细信息
  for (auto stream_id : stream_ids) {
    std::cout << "stream_id = " << stream_id << "; " << "stream_index = " << handlers[stream_id]->GetStreamIndex() << std::endl;
    EXPECT_EQ(handlers[stream_id]->GetStreamId(), stream_id);
    EXPECT_EQ(handlers[stream_id]->GetStreamIndex(), pipeline_->idxManager_->stream_idx_map[stream_id]);
    EXPECT_EQ(handlers[stream_id]->GetStreamIndex(), pipeline_->GetStreamIndex(stream_id));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  pipeline_->Stop();

}

/**
 * 单独使用一个 pipeline 测试 video_handler
 */
TEST_F(SourceVideo, RUN) {
  EXPECT_NE(pipeline_, nullptr);
  Module* module_in_pipeline = pipeline_->GetModule("decoder");
  EXPECT_NE(module_in_pipeline, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);
  EXPECT_NE(source, nullptr);

  std::shared_ptr<SourceHandler> source_handler_ptr = VideoHandler::Create(source, stream_id_);
  video_handler_ = std::dynamic_pointer_cast<VideoHandler>(source_handler_ptr);
  EXPECT_NE(video_handler_, nullptr);

  Module* sink_module = pipeline_->GetModule("sink");
  EXPECT_NE(sink_module, nullptr);

  DataSink *sink = dynamic_cast<DataSink*>(sink_module);
  EXPECT_NE(sink, nullptr);

  std::shared_ptr<SinkHandler> sink_handler = PushHandler::Create(sink, stream_id_);
  push_handler_ = std::dynamic_pointer_cast<PushHandler>(sink_handler);
  EXPECT_NE(push_handler_, nullptr);

  EXPECT_TRUE(pipeline_->Start());
  EXPECT_EQ(source->AddSource(video_handler_), 0);
  EXPECT_EQ(sink->AddSink(push_handler_), 0);

  std::cout << "video_handler_->impl_->stream_url_ = " << video_handler_->impl_->stream_url_ << std::endl;
  std::cout << "video_handler_->impl_->frame_rate_ = " << video_handler_->impl_->frame_rate_ << std::endl;

  if (video_handler_->impl_->IsRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }

  auto module_profiler = pipeline_->GetModuleProfiler(process_module_name);
  if (module_profiler) {
    auto module_profile = module_profiler->GetProfile();
    std::cout << "Process Module profile: " << ModuleProfileToString(module_profile) << std::endl;
  }

  if (video_handler_->impl_->IsRunning()) {
    std::this_thread::sleep_for(std::chrono::seconds(100));
  }
  LOGI(T_SOURCE) << "Handler stream idx: " << video_handler_->GetStreamIndex();
  EXPECT_NE(video_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // == data->stream_idx_
  EXPECT_TRUE(pipeline_->IsRunning());
  
  video_handler_->Stop();
  video_handler_->Close();
  
  PrintStreamEos();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  LOGI(T_SOURCE) << "Wait for EOS message to be processed";
  LOGI(T_SOURCE) << "CheckStreamEosReached(stream_id_) = " << std::boolalpha << CheckStreamEosReached(stream_id_, true);
  LOGI(T_SOURCE) << "Wait for EOS message complete";
  
  pipeline_->Stop();
}

}  // namespace cnstream