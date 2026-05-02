
/**
* 测试 Json 文件读取，配置解析
*/

#include "base.hpp"
#include "cnstream_config.hpp"

static std::string test_pipeline_json = "pipeline_inference.json";
static std::string inference_name = "Inference";

TEST(JSON, ReadFile) {
    std::string json_str = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_str.empty()) << "Read json file failed";
    nlohmann::json doc = nlohmann::json::parse(json_str);

    EXPECT_TRUE(doc.contains("profiler_config")) << "Json file has no profiler_config field";
    EXPECT_TRUE(doc.contains("decoder")) << "Json file has no decoder field";
    EXPECT_FALSE(doc.contains("sort_h"));
    EXPECT_FALSE(doc.contains("sink"));
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
