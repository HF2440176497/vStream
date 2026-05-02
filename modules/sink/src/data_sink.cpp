
#include <algorithm>
#include <map>
#include <memory>
#include <string>

#include "data_sink.hpp"

namespace cnstream {

/**
 * @brief 同样仿照 DataSource 这里注册基本的参数
 * CheckParamSet 只是校验这些基本参数
 */
DataSink::DataSink(const std::string &name) : SinkModule(name) {
  param_register_.SetModuleDesc(
      "DataSink is a module for handling output data (videos or images)."
      " Receive processed data from upstream and dispatch to various sink handlers.");
  param_register_.Register(key_output_url, "The target URL or path for output.");
  param_register_.Register(key_output_format, "Output format, e.g., mp4, flv, jpg.");
  param_register_.Register(key_queue_size, "Queue size for sink handlers.");
}

DataSink::~DataSink() {}

bool DataSink::Open(ModuleParamSet paramSet) {
  if (!CheckParamSet(paramSet)) {
    LOGE(SINK) << "CheckParamSet failed";
    return false;
  }
  param_set_ = paramSet;
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
  // QueueHandler: queue_size
  std::string err_msg;
  if (!checker.IsNum({key_queue_size}, paramSet, err_msg, true)) {
    LOGE(SINK) << "queue_size check failed: " << err_msg;
    return false;
  }
  return true;
}

}  // namespace cnstream
