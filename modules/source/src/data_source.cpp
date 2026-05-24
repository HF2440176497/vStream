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


#include <algorithm>
#include <atomic>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

#include "data_source.hpp"

namespace cnstream {

DataSource::DataSource(const std::string &name) : SourceModule(name) {
  param_register_.SetModuleDesc(
      "DataSource is a module for handling input data (videos or images)."
      " Feed data to codec and send decoded data to the next module if there is one.");
  param_register_.Register(key_source_config_file, "data source config file");
}

DataSource::~DataSource() {}

static int GetDeviceId(ModuleParamSet paramSet) {
  if (paramSet.find(key_device_id) == paramSet.end()) {
    return -1;
  }
  std::stringstream ss;
  int device_id;
  ss << paramSet.at(key_device_id);
  ss >> device_id;
  /*check device_id valid or not,FIXME*/
  return device_id;
}

/**
 * @brief copy and parse paramset to param_
 */
bool DataSource::Open(ModuleParamSet paramSet) {
  if(!CheckParamSet(paramSet)) {
    LOGE(SOURCE) << "CheckParamSet failed";
    return false;
  }
  param_.param_set_ = paramSet;  // note: use param_set_ instead
  param_set_ = paramSet;  // of SourceModule, for handlers
  if (paramSet.find(key_source_config_file) != paramSet.end()) {
    std::string config_file = paramSet.at(key_source_config_file);

    // paramSet from custom params, can include "config_file_path"
    std::string config_path = GetPathRelativeToTheJSONFile(config_file, paramSet);
    if (!LoadStreamConf(config_path)) {
      LOGE(SOURCE) << "LoadStreamConf failed: " << config_path;
      return false;
    }
    LOGI(SOURCE) << "Loaded " << stream_configs_.size() << " stream configs from " << config_path;
  }
  return true;
}

/**
 * Pipeline::Stop() 调用 Module->Close()
 * @todo 可尝试同步方式等待各模块接收 EOS
 */
void DataSource::Close() { RemoveSources(true); }

/**
 * 在 Open 中，使用 paramSet 首先进行检查
 */
bool DataSource::CheckParamSet(const ModuleParamSet &paramSet) const {
  bool ret = true;
  for (auto &it : paramSet) {
    if (!param_register_.IsRegisted(it.first)) {
      LOGW(SOURCE) << "unknown param: " << it.first;
    }
  }
  return ret;
}

int DataSource::Process(std::shared_ptr<FrameInfo> data) {
  LOGI(SOURCE) << "Process receive frame_id: " << data->stream_id;
  return 0;
}

DataSourceParam DataSource::GetSourceParam() const { 
  return param_; 
}

/**
 * 加载 config_file 文件的参数到 stream_configs_
 */
bool DataSource::LoadStreamConf(const std::string& config_file) {
  std::ifstream ifs(config_file);
  if (!ifs.is_open()) {
    LOGE(SOURCE) << "LoadStreamConf: cannot open " << config_file;
    return false;
  }
  try {
    nlohmann::json doc = nlohmann::json::parse(ifs);
    if (!doc.is_object()) {
      LOGE(SOURCE) << "LoadStreamConf: root must be an object";
      return false;
    }
    stream_configs_.clear();
    for (auto it = doc.begin(); it != doc.end(); ++it) {
      const std::string& stream_id = it.key();
      const nlohmann::json& stream_value = it.value();

      // "stream_id" : {
      //   "param1" : "value1",
      //   "param2" : "value2",
      // }
      if (!stream_value.is_object()) {
        LOGW(SOURCE) << "LoadStreamConf: stream [" << stream_id << "] value is not an object, skip";
        continue;
      }
      ModuleParamSet params;
      for (auto pit = stream_value.begin(); pit != stream_value.end(); ++pit) {
        std::string val;
        if (pit.value().is_string()) {
          val = pit.value().get<std::string>();
        } else {
          val = pit.value().dump();
        }
        params[pit.key()] = val;
      }
      stream_configs_[stream_id] = std::move(params);
      LOGI(SOURCE) << "Loaded config for stream [" << stream_id << "]";
    }
  } catch (const nlohmann::json::exception& e) {
    LOGE(SOURCE) << "LoadStreamConf: JSON parse error: " << e.what();
    return false;
  }
  // 检查 config_file 中设置的参数是否合法
  std::string err_msg;
  ParametersChecker checker;
  for (auto &it : stream_configs_) {
    const std::string& stream_id = it.first;
    const ModuleParamSet& paramSet = it.second;
    if (!checker.IsNum({key_device_id}, paramSet, err_msg, true)) {
      LOGE(SOURCE) << stream_id << " [device_id] check failed: " << err_msg;
      return false;
    }
    int device_id = GetDeviceId(paramSet);
    // output_type
    if (paramSet.find(key_output_type) != paramSet.end()) {
      std::string out_type = paramSet.at(key_output_type);
      if (param_output_map.find(out_type) == param_output_map.end()) {
        LOGE(SOURCE) << stream_id << " [output_type] " << out_type << " check failed";
        return false;
      }
      auto output_type = param_output_map.at(out_type);
      if (output_type != OutputType::OUTPUT_CPU) {
        if (device_id < 0) {
          LOGE(SOURCE) << stream_id << " [output_type] " << out_type << " : device_id must be set";
          return false;
        }
      }
    }
    if (!checker.IsNum({key_interval}, paramSet, err_msg, false)) {
      LOGE(SOURCE) << stream_id << " [interval] check failed: " << err_msg;
      return false;
    }

    bool has_file = (paramSet.find(key_file_path) != paramSet.end());
    bool has_url = (paramSet.find(key_input_url) != paramSet.end());

    if (has_file) {
      if (paramSet.at(key_file_path).empty()) {
        LOGE(SOURCE) << stream_id << " [file_path] must not be empty";
        return false;
      }
      if (paramSet.find(key_frame_rate) == paramSet.end()) {
        LOGE(SOURCE) << stream_id << " [frame_rate] is required for image stream";
        return false;
      }
      if (!checker.IsNum({key_frame_rate}, paramSet, err_msg, false)) {
        LOGE(SOURCE) << stream_id << " [frame_rate] " << err_msg;
        return false;
      }
    }

    if (has_url) {
      if (paramSet.at(key_input_url).empty()) {
        LOGE(SOURCE) << stream_id << " [url] must not be empty";
        return false;
      }
    }
  }
  return true;
}

ModuleParamSet DataSource::GetStreamParams(const std::string& stream_id) const {
  auto it = stream_configs_.find(stream_id);
  if (it != stream_configs_.end()) {
    return it->second;
  }
  LOGW(SOURCE) << "Stream [" << stream_id << "] not found in config file";
  return {};
}

}  // namespace cnstream
