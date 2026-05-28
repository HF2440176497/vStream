
#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include "data_sink.hpp"
#include "nlohmann/json.hpp"

namespace cnstream {

/**
 * @brief 同样仿照 DataSource 这里注册基本的参数
 * CheckParamSet 只是校验这些基本参数
 */
DataSink::DataSink(const std::string &name) : SinkModule(name) {
  param_register_.SetModuleDesc(
      "DataSink is a module for handling output data (videos or images)."
      " Receive processed data from upstream and dispatch to various sink handlers.");
  param_register_.Register(key_sink_config_file, "data sink config file");
}

DataSink::~DataSink() {}

bool DataSink::Open(ModuleParamSet paramSet) {
  if (!CheckParamSet(paramSet)) {
    LOGE(SINK) << "CheckParamSet failed";
    return false;
  }
  param_set_ = paramSet;

  if (paramSet.find(key_sink_config_file) != paramSet.end()) {
    std::string config_file = paramSet.at(key_sink_config_file);
    std::string config_path = GetPathRelativeToTheJSONFile(config_file, paramSet);
    if (!LoadStreamConf(config_path)) {
      LOGE(SINK) << "LoadStreamConf failed: " << config_path;
      return false;
    }
    LOGI(SINK) << "Loaded " << stream_configs_.size() << " stream configs from " << config_path;
  }

  return true;
}

void DataSink::Close() { RemoveSinks(true); }

bool DataSink::CheckParamSet(const ModuleParamSet &paramSet) const {
  for (auto &it : paramSet) {
    if (!param_register_.IsRegisted(it.first)) {
      LOGW(SINK) << "unknown param: " << it.first;
    }
  }
  return true;
}

bool DataSink::LoadStreamConf(const std::string& config_file_path) {
  std::ifstream ifs(config_file_path);
  if (!ifs.is_open()) {
    LOGE(SINK) << "LoadStreamConf: cannot open " << config_file_path;
    return false;
  }
  try {
    nlohmann::json doc = nlohmann::json::parse(ifs);
    if (!doc.is_object()) {
      LOGE(SINK) << "LoadStreamConf: root must be an object";
      return false;
    }
    stream_configs_.clear();
    for (auto it = doc.begin(); it != doc.end(); ++it) {
      const std::string& stream_id = it.key();
      const nlohmann::json& stream_value = it.value();
      if (!stream_value.is_object()) {
        LOGW(SINK) << "LoadStreamConf: stream [" << stream_id << "] value is not an object, skip";
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
      LOGI(SINK) << "LoadStreamConf: loaded config for stream [" << stream_id << "]";
    }
  } catch (const nlohmann::json::exception& e) {
    LOGE(SINK) << "LoadStreamConf: JSON parse error: " << e.what();
    return false;
  }

  std::string err_msg;
  ParametersChecker checker;
  for (auto &it : stream_configs_) {
    const std::string& stream_id = it.first;
    const ModuleParamSet& paramSet = it.second;

    bool is_push_stream = (paramSet.find(key_output_url) != paramSet.end());
    bool is_queue_stream = (paramSet.find(key_queue_size) != paramSet.end());

    if (is_push_stream) {
      if (paramSet.at(key_output_url).empty()) {
        LOGE(SINK) << "[" << stream_id << "]: [url] is required for push stream and must be non-empty";
        return false;
      }
      if (paramSet.find(key_output_fps) == paramSet.end()) {
        LOGE(SINK) << "[" << stream_id << "]: [fps] is required for push stream";
        return false;
      }
      if (paramSet.find(key_output_width) == paramSet.end()) {
        LOGE(SINK) << "[" << stream_id << "]: [width] is required for push stream";
        return false;
      }
      if (paramSet.find(key_output_height) == paramSet.end()) {
        LOGE(SINK) << "[" << stream_id << "]: [height] is required for push stream";
        return false;
      }
      // checker for push handler
      if (!checker.IsNum({key_output_fps}, paramSet, err_msg, false)) {
        LOGE(SINK) << "[" << stream_id << "]: [fps] " << err_msg;
        return false;
      }
      if (!checker.IsNum({key_output_width}, paramSet, err_msg, false)) {
        LOGE(SINK) << "[" << stream_id << "]: [width] " << err_msg;
        return false;
      }
      if (!checker.IsNum({key_output_height}, paramSet, err_msg, false)) {
        LOGE(SINK) << "[" << stream_id << "]: [height] " << err_msg;
        return false;
      }
    }

    if (is_queue_stream) {
      if (!checker.IsNum({key_queue_size}, paramSet, err_msg, false)) {
        LOGE(SINK) << "[" << stream_id << "]: [queue_size] " << err_msg;
        return false;
      }
      if (paramSet.find(key_queue_size) != paramSet.end()) {
        int qs = std::stoi(paramSet.at(key_queue_size));
        if (qs <= 0) {
          LOGE(SINK) << "[" << stream_id << "]: [queue_size] must be positive, got " << qs;
          return false;
        }
      }
    }
    
    if (!checker.IsNum({key_output_device_id}, paramSet, err_msg, false)) {
      LOGE(SINK) << "[" << stream_id << "]: [device_id] " << err_msg;
      return false;
    }
    if (!checker.IsNum({key_output_bitrate}, paramSet, err_msg, false)) {
      LOGE(SINK) << "[" << stream_id << "]: [bitrate] " << err_msg;
      return false;
    }
  }

  return true;
}

ModuleParamSet DataSink::GetStreamParams(const std::string& stream_id) const {
  auto it = stream_configs_.find(stream_id);
  if (it != stream_configs_.end()) {
    return it->second;
  }
  return {};
}

}  // namespace cnstream
