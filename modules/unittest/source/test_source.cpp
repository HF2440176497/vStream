
#include "base.hpp"

#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

#include "data_source.hpp"
#include "data_handler_image.hpp"
#include "data_handler_video.hpp"

#include <chrono>
#include <typeinfo>

#include <opencv2/opencv.hpp>

#ifdef VSTREAM_USE_CUDA
#include "cuda/inspect_mem.hpp"
#endif

namespace cnstream {

static std::string test_pipeline_json = "pipeline_source_base.json";
static std::string test_pipeline_video_json = "pipeline_source_video.json";

// 定义在 base.hpp 中
std::string process_module_name = "ProcessOne";


static bool has_save_frame_mat = false;
static std::string save_file = "save_image/test_source_save.jpg";

// 在测试实例中，定义出这个 virtual module
class InferenceProcess: public Module, public ModuleCreator<InferenceProcess> {
  public:
    InferenceProcess(const std::string &name) : Module(name) {}
    ~InferenceProcess() {}
    bool Open(ModuleParamSet params) override {
      return true;
    }
    uint64_t frame_count_ = 0;

    void Close() override {
      LOGI(InferenceProcess) << "Close";
    }

    void OnEos(const std::string& stream_id) override {
      LOGI(InferenceProcess) << "OnEos: " << stream_id;
    }

    int Process(std::shared_ptr<FrameInfo> frame_info) override {
      frame_count_++;
      LOGI(InferenceProcess) << "---------- Process frame " << frame_info->stream_id << "; with time: " << frame_info->timestamp;
      DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
      if (!frame) {
        LOGE(InferenceProcess) << "frame is empty";
        return -1;
      }
      std::cout << "--- frame datafmt: " << DataFormat2Str(frame->GetFmt()) << std::endl;
      std::cout << "--- frame devtype: " << DevType2Str(frame->GetCtx().device_type) << std::endl;
      std::cout << "--- frame devid: " << frame->GetCtx().device_id << std::endl;
      
      std::cout << "--- frame image height: " << frame->GetHeight() << std::endl;  
      std::cout << "--- frame image width: " << frame->GetWidth() << "; stride: " << frame->GetStride(0) << std::endl;

      // 打印 SyncMem 状态
      for (int i = 0; i < frame->GetPlanes(); ++i) {
        std::string mem_status_info = frame->data_[i]->StatusToString();
        std::cout << "--- frame plane " << i << " mem status: " << mem_status_info << std::endl;
      }

      // 尝试通过 SyncMem 落地图片
      if (frame->HasImage()) {
        LOGW(InferenceProcess) << "before get image, frame_mat_ should be empty";
      }
      cv::Mat frame_mat = frame->GetImage();
      if (frame_mat.empty()) {
        LOGE(InferenceProcess) << "frame_mat_ is empty";
        return -1;
      }

      if (frame_count_ % 100 == 0) {
        GPUInspect inspect(0);
        std::cout << inspect.GetBriefInfo() << std::endl;
      }
      if (!has_save_frame_mat) {
        cv::imwrite(save_file, frame_mat);
        has_save_frame_mat = true;
      }
      return 0;
    }
};

REGISTER_MODULE(InferenceProcess);

class EosObserver : public StreamMsgObserver {
 public:
  void Update(const StreamMsg &msg) override {}
};

class SourceModuleTest : public testing::Test {
 protected:
  virtual void SetUp() {
    std::string json_content = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_content.empty()) << "Read json file failed";
    cnstream::CNGraphConfig graph_config;
    graph_config.ParseByJSONStr(json_content);
    graph_config_ = graph_config;

    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipeline(graph_config_));
  }

  virtual void TearDown() {  // 当前用例结束
    LOGI(SourceModuleTest) << "TearDown";
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
 * @brief 测试 VideoSourceHandler 进行硬解码
 */
class VideoSourceTest : public testing::Test {
 protected:
  virtual void SetUp() {
    std::string json_content = readFile(test_pipeline_video_json.c_str());
    EXPECT_FALSE(json_content.empty()) << "Read json file failed";
    cnstream::CNGraphConfig graph_config;
    graph_config.ParseByJSONStr(json_content);
    graph_config_ = graph_config;

    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipeline(graph_config_));
  }

  virtual void TearDown() {
    LOGI(VideoSourceTest) << "TearDown";
    if (pipeline_) {
      pipeline_->Stop();
    }
    video_handler_.reset();
  }

 protected:
  const std::string stream_id_ = "channel-1";
  std::shared_ptr<VideoHandler> video_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
  cnstream::CNGraphConfig       graph_config_;
};

TEST(Source, BasicOutput) {
  LOGI(MODULESUNITEST) << "LOGI test message";
  LOGW(MODULESUNITEST) << "LOGW test message";
  LOGE(MODULESUNITEST) << "LOGE test message";
}

TEST_F(SourceModuleTest, PipelineInit) {

  // 先测试配置加载
  std::unique_ptr<CNGraph<NodeContext>> graph = std::make_unique<CNGraph<NodeContext>>();
  EXPECT_NE(nullptr, graph.get());
  EXPECT_TRUE(graph->Init(graph_config_));

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
  std::cout << "-------- SourceModuleTest module name: " << std::endl;
  // EXPECT_EQ(registed_modules.size(), expected_nodes.size());
  for (auto& module_name : registed_modules) {
    std::cout << "module name: " << module_name << std::endl;
  }
  EXPECT_TRUE(std::find(registed_modules.begin(), registed_modules.end(), "cnstream::DataSource") != registed_modules.end());
}  // PipelineInit

/**
 * 读取图片
 */
TEST_F(SourceModuleTest, Loop) {
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
  EXPECT_FALSE(IsStreamRemoved(stream_id_));  // 此处不应当被移除

  EXPECT_EQ(source->AddSource(image_handler_), 0);
  EXPECT_TRUE(image_handler_->impl_->running_);

  // AddSource 之后，handler handler 理应可以获取到配置参数
  std::cout << "image_handler_->impl_->image_path = " << image_handler_->impl_->image_path_ << std::endl;
  std::cout << "image_handler_->impl_->frame_rate_ = " << image_handler_->impl_->frame_rate_ << std::endl;

  std::this_thread::sleep_for(std::chrono::milliseconds(2000));  // running for a while
  LOGI(SourceModuleTest) << "Handler stream idx: " << image_handler_->GetStreamIndex();
  EXPECT_NE(image_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // 等同 data->GetStreamIndex
  EXPECT_TRUE(pipeline_->IsRunning());
  
  image_handler_->Stop();
  image_handler_->Close();
  
  PrintStreamEos();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  LOGI(SourceModuleTest) << "Wait for EOS message to be processed";
  LOGI(SourceModuleTest) << "CheckStreamEosReached(stream_id_) = " << std::boolalpha << CheckStreamEosReached(stream_id_, true);
  LOGI(SourceModuleTest) << "Wait for EOS message complete";
  
  pipeline_->Stop();
}

/**
 * 测试多个流，每个流处理各自的图像
 */
TEST_F(SourceModuleTest, MutilStream) {
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
    EXPECT_TRUE(handlers[stream_id]->impl_->running_);
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
TEST_F(VideoSourceTest, DISABLED_Loop) {
// TEST_F(VideoSourceTest, Loop) {
  EXPECT_NE(pipeline_, nullptr);
  Module* module_in_pipeline = pipeline_->GetModule("decoder");
  EXPECT_NE(module_in_pipeline, nullptr);

  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);
  EXPECT_NE(source, nullptr);

  std::shared_ptr<SourceHandler> source_handler_ptr = VideoHandler::Create(source, stream_id_);
  video_handler_ = std::dynamic_pointer_cast<VideoHandler>(source_handler_ptr);
  EXPECT_NE(video_handler_, nullptr);

  EXPECT_TRUE(pipeline_->Start());
  EXPECT_FALSE(IsStreamRemoved(stream_id_));  // 此处不应当被移除

  EXPECT_EQ(source->AddSource(video_handler_), 0);
  EXPECT_TRUE(video_handler_->impl_->running_);

  // AddSource 之后，handler handler 理应可以获取到配置参数
  std::cout << "video_handler_->impl_->stream_url_ = " << video_handler_->impl_->stream_url_ << std::endl;
  std::cout << "video_handler_->impl_->frame_rate_ = " << video_handler_->impl_->frame_rate_ << std::endl;

  std::this_thread::sleep_for(std::chrono::milliseconds(3000));  // running for a while

  // 需要判断是否有 nullptr (可能是未开启 profile)
  // source module 不会有 kINPUT_PROFILER_NAME kPROCESS_PROFILER_NAME
  auto module_profiler = pipeline_->GetModuleProfiler(process_module_name);
  if (module_profiler) {
    auto module_profile = module_profiler->GetProfile();
    std::cout << "Process Module profile: " << ModuleProfileToString(module_profile) << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds(100));  // running for a while

  LOGI(VideoSourceTest) << "Handler stream idx: " << video_handler_->GetStreamIndex();
  EXPECT_NE(video_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // 等同 data->GetStreamIndex
  EXPECT_TRUE(pipeline_->IsRunning());
  
  video_handler_->Stop();
  video_handler_->Close();
  
  PrintStreamEos();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  LOGI(VideoSourceTest) << "Wait for EOS message to be processed";
  LOGI(VideoSourceTest) << "CheckStreamEosReached(stream_id_) = " << std::boolalpha << CheckStreamEosReached(stream_id_, true);
  LOGI(VideoSourceTest) << "Wait for EOS message complete";
  
  pipeline_->Stop();
}

}  // namespace cnstream