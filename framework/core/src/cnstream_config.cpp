/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cnstream_logging.hpp"
#include "cnstream_config.hpp"

namespace cnstream {

static inline
bool IsProfilerItem(const std::string& item_name) {
  return kProfilerConfigName == item_name;
}

/**
 * @brief 获取路径的目录部分
 * @param path 文件路径
 * @example "save_image/test.jpg" -> "save_image/"
 */
static inline
std::string GetPathDir(const std::string& path) {
  auto slash_pos = path.rfind("/");
  return slash_pos == std::string::npos ? "" : path.substr(0, slash_pos) + "/";
}

/**
 * @brief 从 JSON 文件中解析配置
 * 调用接口 ParseByJSONStr (pure virtual func interface)
 */
bool CNConfigBase::ParseByJSONFile(const std::string& jfile) {
  std::ifstream ifs(jfile);
  if (!ifs.is_open()) {
    LOGE(CORE) << "Config file open failed :" << jfile;
    return false;
  }
  std::string jstr((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  ifs.close();
  config_root_dir = GetPathDir(jfile);
  if (!ParseByJSONStr(jstr)) {
    return false;
  }
  return true;
}

bool ProfilerConfig::ParseByJSONStr(const std::string& jstr) {
  nlohmann::ordered_json doc = nlohmann::ordered_json::parse(jstr);
  if (!doc.is_object()) {
    LOGE(CORE) << "Profiler configuration must be object type.";
    return false;
  }
  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& key = it.key();
    const nlohmann::ordered_json& value = it.value();
    if (key == key_enable_profile) {
      if (value.is_boolean()) {
        this->enable_profile = value.get<bool>();  // explicit conversion
      } else {
        LOGE(CORE) << "Profiler enable_profile must be boolean type.";
        return false;
      }
    } else {
        LOGE(CORE) << "unknown parameter named [" << key << "] for profiler_config.";
        return false;
    }
  }
  return true;
}

/**
 * @detail 注意：没有设置 name (std::string) 成员
 */
bool CNModuleConfig::ParseByJSONStr(const std::string& jstr) {
  nlohmann::ordered_json doc = nlohmann::ordered_json::parse(jstr);
  if (!doc.is_object()) {
    LOGE(CORE) << "Module configuration must be object type.";
    return false;
  }
  // className
  if (!doc.contains(key_class_name)) {
    LOGE(CORE) << "Module has to have a class_name.";
    return false;
  } else {
    if (!doc[key_class_name].is_string()) {
      LOGE(CORE) << "class_name must be string type.";
      return false;
    }
    this->className = doc[key_class_name].get<std::string>();
  }
  // parallelism
  if (!doc.contains(key_parallelism)) {
    this->parallelism = 1;
  } else {
    if (!doc[key_parallelism].is_number_unsigned()) {
      LOGE(CORE) << "parallelism must be uint type.";
      return false;
    }
    this->parallelism = doc[key_parallelism].get<int>();
  }
  // maxInputQueueSize
  if (!doc.contains(key_max_input_queue_size)) {
    this->maxInputQueueSize = 20;
  } else {
    if (!doc[key_max_input_queue_size].is_number_unsigned()) {
      LOGE(CORE) << "max_input_queue_size must be uint type.";
      return false;
    }
    this->maxInputQueueSize = doc[key_max_input_queue_size].get<int>();
  }
  // next
  if (doc.contains(key_next_modules)) {
    if (!doc[key_next_modules].is_array()) {
      LOGE(CORE) << "next_modules must be array type.";
      return false;
    }
    auto values = doc[key_next_modules].get<std::vector<std::string>>();
    for (auto& module : values) {
      this->next.insert(module);
    }
  } else {
    this->next = {};
  }
  // custom parameters: value 都强转为 string 类型
  if (doc.contains(key_custom_params)) {
    if (!doc[key_custom_params].is_object()) {
      LOGE(CORE) << "custom_params must be object type.";
      return false;
    }
    this->parameters.clear();
    for (auto it = doc[key_custom_params].begin(); it != doc[key_custom_params].end(); ++it) {
      const std::string& key = it.key();
      const nlohmann::ordered_json& value = it.value();
      std::string str_value;
      if (value.is_string()) {
        str_value = value.get<std::string>();
      } else {
        str_value = value.dump();
      }
      this->parameters.insert(std::make_pair(key, str_value));
    }
    if (this->parameters.end() != this->parameters.find(CNS_JSON_DIR_PARAM_NAME)) {
      config_root_dir = this->parameters[CNS_JSON_DIR_PARAM_NAME];
      this->parameters.insert(std::make_pair(CNS_JSON_DIR_PARAM_NAME, config_root_dir));
    } else {
      // this->parameters.insert(std::make_pair(CNS_JSON_DIR_PARAM_NAME, "./"));
      LOGD(CORE) << "Parameter [" << CNS_JSON_DIR_PARAM_NAME << "]  not set.";
    }
  } else {
    this->parameters = {};
  }
  return true;
}

bool CNGraphConfig::ParseByJSONStr(const std::string& json_str) {
  nlohmann::ordered_json doc = nlohmann::ordered_json::parse(json_str);
  if (!doc.is_object()) {
    LOGE(CORE) << "Graph configuration must be object type.";
    return false;
  }
  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& item_name = it.key();
    const nlohmann::ordered_json& item_value = it.value();
    std::string item_value_str = item_value.dump();
    if (IsProfilerItem(item_name)) {
      if (!profiler_config.ParseByJSONStr(item_value_str)) {
        LOGE(CORE) << "Parse profiler config failed.";
        return false;
      }
    } else {  // Module 
      CNModuleConfig mconf;
      mconf.config_root_dir = config_root_dir;
      mconf.name = item_name;
      if (!mconf.ParseByJSONStr(item_value_str)) {
        LOGE(CORE) << "Parse module config failed. Module name : [" << mconf.name << "]";
        return false;
      }
      module_configs.push_back(std::move(mconf));
    }
  }  // end for (json items)
  return true;
}

/**
 * @brief 如果 path 是绝对路径，直接返回；否则，返回相对于 json_file_dir 参数目录的路径
 * @param path 
 * @param param_set 
 * @note CheckPath 调用
 */
std::string GetPathRelativeToTheJSONFile(const std::string& path, const ModuleParamSet& param_set) {
  std::string config_dir = "./";
  // pipeline json dir
  if (param_set.find(CNS_JSON_DIR_PARAM_NAME) != param_set.end()) {
    config_dir = param_set.find(CNS_JSON_DIR_PARAM_NAME)->second;
  }

  std::string ret = "";
  if (path.size() > 0 && '/' == path[0]) {
    /*absolute path*/
    ret = path;
  } else {
    ret = config_dir + path;
  }
  return ret;
}

}  // namespace cnstream
