
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
  param_register_.Register(key_config_file, "data sink config file");
}

DataSink::~DataSink() {}

bool DataSink::Open(ModuleParamSet paramSet) {
  if (!CheckParamSet(paramSet)) {
    LOGE(SINK) << "CheckParamSet failed";
    return false;
  }
  param_set_ = paramSet;

  if (paramSet.find(key_config_file) != paramSet.end()) {
    std::string config_file = paramSet.at(key_config_file);
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
  ParametersChecker checker;
  for (auto &it : paramSet) {
    if (!param_register_.IsRegisted(it.first)) {
      LOGW(SINK) << "unknown param: " << it.first << "; maybe for handler usage";
    }
  }
  // PushHandler: output_format
  if (paramSet.find(key_output_format) != paramSet.end()) {
    if (key_supported_formats.find(paramSet.at(key_output_format)) == key_supported_formats.end()) {
      LOGE(SINK) << "output_format " << paramSet.at(key_output_format) << " is not supported";
      return false;
    }
  }
  std::string err_msg;
  if (paramSet.find(key_output_fps) != paramSet.end()) {
    if (!checker.IsNum({key_output_fps}, paramSet, err_msg, false)) {
      LOGE(SINK) << "output_fps check failed: " << err_msg;
      return false;
    }
  }
  if (paramSet.find(key_output_height) != paramSet.end()) {
    if (!checker.IsNum({key_output_height}, paramSet, err_msg, false)) {
      LOGE(SINK) << "output_height check failed: " << err_msg;
      return false;
    }
  }
  if (paramSet.find(key_output_width) != paramSet.end()) {
    if (!checker.IsNum({key_output_width}, paramSet, err_msg, false)) {
      LOGE(SINK) << "output_width check failed: " << err_msg;
      return false;
    }
  }
  if (paramSet.find(key_output_bitrate) != paramSet.end()) {
    if (!checker.IsNum({key_output_bitrate}, paramSet, err_msg, false)) {
      LOGE(SINK) << "output_bitrate check failed: " << err_msg;
      return false;
    }
  }
  if (paramSet.find(key_output_device_id) != paramSet.end()) {
    if (!checker.IsNum({key_output_device_id}, paramSet, err_msg, true)) {
      LOGE(SINK) << "output_device_id check failed: " << err_msg;
      return false;
    }
  }
  // QueueHandler: queue_size
  if (!checker.IsNum({key_queue_size}, paramSet, err_msg, false)) {
    LOGE(SINK) << "queue_size check failed: " << err_msg;
    return false;
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
