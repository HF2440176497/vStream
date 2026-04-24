
// 使用 nlohmann-json 实现解析 Json 文件

#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

static std::string json_file = "pipeline_inference.json";
static std::string module_name = "Inference";

bool parse_from_file(std::string json_file, json& data) {
  std::ifstream file(json_file);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << json_file << std::endl;
    return false;
  }
  data = json::parse(file);

  if (data.contains(module_name)) {
    std::cout << module_name << ": " << data[module_name] << std::endl;
    std::vector<std::string> next_modules = data[module_name]["next_modules"];
    std::cout << "------ next_modules size: " << next_modules.size() << std::endl;
    for (auto& module : next_modules) {
      std::cout << "   next_module: " << module << std::endl;
    }
    if (data[module_name].contains("custom_params")) {
      std::cout << "------ custom_params: " << data[module_name]["custom_params"] << std::endl;
      std::cout << "   model_path default value: " << data[module_name]["custom_params"].value("model_path", "default_model_path") << std::endl;
      std::cout << "   model_path null: " << data[module_name]["custom_params"]["model_path"].is_null() << std::endl;
    }
  }

  // 使用 find 查找 model_path
  if (data[module_name].contains("custom_params")) {
    std::cout << "------ module custom_params: " << data[module_name]["custom_params"] << std::endl;
    auto module_custom_params = data[module_name]["custom_params"];
    if (module_custom_params.find("model_path") != module_custom_params.end()) {
      std::cout << "   module model_path found: " << module_custom_params["model_path"] << std::endl;
    } else {
      std::cout << "   module model_path not found" << std::endl;
    }
    if (module_custom_params.is_object()) {
      std::cout << "   module custom_params key nums: " << module_custom_params.size() << std::endl;
    }
  }

  // 尝试将 custom_params 先转换为 string，再解析为 json 对象
  std::cout << "--------------------- " << std::endl;
  std::string custom_params_str;
  if (data[module_name].is_object()) {
    custom_params_str = data[module_name]["custom_params"].dump();
    std::cout << "   custom_params str: " << custom_params_str << std::endl;
  } else {
    return false;
  }
  json custom_params_json = json::parse(custom_params_str);
  std::cout << "   custom_params json: " << custom_params_json << std::endl;
  if (custom_params_json.contains("model_path")) {
    std::cout << "   custom_params model_path found: " << custom_params_json["model_path"] << std::endl;
  } else {
    std::cout << "   custom_params model_path not found" << std::endl;
  }
  
  return true;
}

int main(int argc, char* argv[]) {
  json j;
  if (!parse_from_file(json_file, j)) {
    return 1;
  }
  return 0;
}