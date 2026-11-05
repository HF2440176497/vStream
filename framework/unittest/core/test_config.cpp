
/**
* 测试 Json 文件读取，配置解析
*/

#include "base.hpp"
#include "cnstream_config.hpp"

static std::string test_pipeline_json = "pipeline_inference.json";
static std::string inference_name = "inference";

static std::string source_name = "source";
static std::string sink_name = "sink";


TEST(JSON, ReadFile) {
    std::string json_str = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_str.empty()) << "Read json file failed";
    nlohmann::json doc = nlohmann::json::parse(json_str);

    EXPECT_TRUE(doc.contains(kProfilerConfigName)) << "Json file has no profiler_config field";
    EXPECT_TRUE(doc.contains(source_name)) << "Json file has no source field";
    EXPECT_FALSE(doc.contains(sink_name));
}

/**
* 测试 Json 文件读取后，读取某字段再得到字符串
* @detail 测试 Inference 字段
*/
TEST(JSON, ReadFile2Str) {
    std::string json_str = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_str.empty()) << "Read json file failed";
    nlohmann::json doc = nlohmann::json::parse(json_str);

    EXPECT_TRUE(doc.contains(inference_name)) << "Json file has no inference field";
    const nlohmann::json& inference = doc[inference_name];
    EXPECT_TRUE(inference.is_object()) << "Inference field is not object";
    
    std::string inference_str = inference.dump();
    EXPECT_TRUE(!inference_str.empty()) << "Inference field is empty";
    LOGI(COREUNITEST) << "Inference field: " << inference_str << std::endl;
}

/**
 * @brief 创建一个临时文件，测试 CNConfigBase 基类
 */
TEST(CoreConfig, ParseByJSONFile) {
    struct TestConfig : public cnstream::CNConfigBase {
        bool ParseByJSONStr(const std::string& jstr) override {return true;}
    };
    TestConfig test_config;
    auto config_file = CreateTempFile("pipeline_temp");  // fd-filename
    EXPECT_TRUE(test_config.ParseByJSONStr(config_file.second));
    EXPECT_TRUE(test_config.config_root_dir.empty());
    unlink(config_file.second.c_str());
    close(config_file.first);  // close fd
}

/**
 * @brief 测试 CNModuleConfig 解析，将字段 Inference 解析为 CNModuleConfig
 */
TEST(CoreConfig, ModuleConfig) {
    std::string json_str = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_str.empty()) << "Read json file failed";
    nlohmann::json doc = nlohmann::json::parse(json_str);

    EXPECT_TRUE(doc.contains(inference_name)) << "Json file has no [" << inference_name << "] field";
    const nlohmann::json& inference = doc[inference_name];
    EXPECT_TRUE(inference.is_object()) << "[" << inference_name << "] field is not object";
    
    std::string inference_str = inference.dump();

    // CMoudleConfig
    cnstream::CNModuleConfig inference_config;
    EXPECT_TRUE(inference_config.ParseByJSONStr(inference_str));
    EXPECT_TRUE(inference_config.name.empty());

    // className 是自定义指定的， 后续再通过 ModuleFactory 创建模块
    EXPECT_EQ(inference_config.className, "cnstream::Inference");
    EXPECT_EQ(inference_config.next.size(), 1);  // next_modules 

    LOGI(COREUNITEST) << "Inference next modules: please check with file [" << test_pipeline_json << "]";
    for (const auto& elem : inference_config.next) {
        LOGI(COREUNITEST) << "module: " << elem << " ";
    }
    EXPECT_EQ(inference_config.config_root_dir, inference_config.parameters[CNS_JSON_DIR_PARAM_NAME]);
}

/**
 * @brief Test CNGraphConfig 
 * @detail 这是最大一级的解析层级
 */
TEST(CoreConfig, CNGraphConfig) {
    std::string json_content = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_content.empty()) << "Read json file failed";

    cnstream::CNGraphConfig graph_config;
    EXPECT_TRUE(graph_config.ParseByJSONStr(json_content));

    // 检查 profiler_config 模块
    EXPECT_TRUE(graph_config.profiler_config.enable_profile);
}

/**
 * @brief 校验 ProfilerConfig 的默认值与新增 key 的解析/校验
 */
TEST(CoreConfig, ProfilerConfigParsing) {
    cnstream::ProfilerConfig config;
    // 默认开启性能分析
    EXPECT_TRUE(config.enable_profile);
    EXPECT_EQ(config.report_interval_ms, 5000);
    EXPECT_EQ(config.window_ms, 5000);
    EXPECT_EQ(config.bucket_ms, 1000);

    const std::string jstr =
        R"({"enable_profile":false,"report_interval_ms":1000,"window_ms":2000,"bucket_ms":500})";
    EXPECT_TRUE(config.ParseByJSONStr(jstr));
    EXPECT_FALSE(config.enable_profile);
    EXPECT_EQ(config.report_interval_ms, 1000);
    EXPECT_EQ(config.window_ms, 2000);
    EXPECT_EQ(config.bucket_ms, 500);

    // bucket_ms 大于 window_ms 时收敛到 window_ms
    cnstream::ProfilerConfig clamped;
    EXPECT_TRUE(clamped.ParseByJSONStr(R"({"window_ms":100,"bucket_ms":1000})"));
    EXPECT_EQ(clamped.bucket_ms, 100);

    // 非法取值应被拒绝
    cnstream::ProfilerConfig bad_window;
    EXPECT_FALSE(bad_window.ParseByJSONStr(R"({"window_ms":0})"));

    cnstream::ProfilerConfig bad_bucket;
    EXPECT_FALSE(bad_bucket.ParseByJSONStr(R"({"bucket_ms":-1})"));

    cnstream::ProfilerConfig bad_interval;
    EXPECT_FALSE(bad_interval.ParseByJSONStr(R"({"report_interval_ms":1.5})"));
}
