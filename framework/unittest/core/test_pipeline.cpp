

#include "base.hpp"
#include "cnstream_pipeline.hpp"


namespace cnstream {


static std::string test_pipeline_json = "pipeline_config.json";

// 在测试实例中，定义出这个 virtual module
class InferenceProcess: public Module, public ModuleCreator<InferenceProcess> {
  public:
    InferenceProcess(const std::string &name) : Module(name) {}
    ~InferenceProcess() {}
    bool Open(ModuleParamSet params) override {
      return true;
    }
    void Close() override {
      LOGI(InferenceProcess) << "Close";
    }
    int Process(std::shared_ptr<FrameInfo> frame) override {
      LOGI(InferenceProcess) << "Process frame " << frame->stream_id << "; with time: " << frame->timestamp;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return 0;
    }
};

REGISTER_MODULE(InferenceProcess);


class PipelineConfigLoad : public testing::Test {
  protected:
    virtual void SetUp() {
      std::string json_content = readFile(test_pipeline_json.c_str());
      EXPECT_FALSE(json_content.empty()) << "Read json file failed";
      cnstream::CNGraphConfig graph_config;
      graph_config.ParseByJSONStr(json_content);
      graph_config_ = graph_config;
    }
    virtual void TearDown() {  // 当前用例结束
      LOGI(TestConfigLoad) << "TearDown";
    }
  protected:
    cnstream::CNGraphConfig graph_config_;
};


}  // namespace cnstream