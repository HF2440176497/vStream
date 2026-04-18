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
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "data_source.hpp"

namespace cnstream {

DataSource::DataSource(const std::string &name) : SourceModule(name) {
  param_register_.SetModuleDesc(
      "DataSource is a module for handling input data (videos or images)."
      " Feed data to codec and send decoded data to the next module if there is one.");
  param_register_.Register(key_output_type,
                           "Where the outputs will be stored. It could be cpu or mlu,"
                           "It is used when decoder_type is cpu.");
  param_register_.Register(key_device_id, "Which device will be used. If there is only one device, it might be 0.");
  param_register_.Register(key_interval,
                           "How many frames will be discarded between two frames"
                           " which will be sent to next modules.");
  param_register_.Register(key_decoder_type, "Which the input data will be decoded by. It could be cpu or mlu.");
  param_register_.Register(key_only_key_frame, "Only decode key frames and other frames are discarded. Default is false");
}

DataSource::~DataSource() {}

static int GetDeviceId(ModuleParamSet paramSet) {
  if (paramSet.find(key_device_id) == paramSet.end()) {
    return -1;
  }
  std::stringstream ss;
  int device_id;
  ss << paramSet[key_device_id];
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
  if (paramSet.find(key_output_type) != paramSet.end()) {
    std::string out_type = paramSet[key_output_type];
    param_.output_type_ = param_output_map_.at(out_type);
  }
  if (paramSet.find(key_decoder_type) != paramSet.end()) {
    std::string dec_type = paramSet[key_decoder_type];
    param_.decoder_type_ = param_decoder_map_.at(dec_type);
  }
  if (paramSet.find(key_interval) != paramSet.end()) {
    std::stringstream ss;
    int interval;
    ss << paramSet[key_interval];
    ss >> interval;
    if (interval <= 0) {
      LOGE(SOURCE) << "interval : invalid";
      return false;
    }
    param_.interval_ = interval;
  }
  param_.device_id_ = GetDeviceId(paramSet);
  if (paramSet.find(key_only_key_frame) != paramSet.end()) {
    param_.only_key_frame_ = (paramSet[key_only_key_frame] == "true");
  }
  param_.param_set_ = paramSet;
  param_set_ = paramSet;  // of SourceModule, for handlers
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
  ParametersChecker checker;
  for (auto &it : paramSet) {
    if (!param_register_.IsRegisted(it.first)) {
      LOGW(SOURCE) << "[DataSource] Unknown param: " << it.first << "; Maybe for handler usage";
    }
  }
  std::string err_msg;
  if (!checker.IsNum({key_device_id}, paramSet, err_msg, true)) {
    LOGE(SOURCE) << "[DataSource] " << err_msg;
    return false;
  }
  int device_id = GetDeviceId(paramSet);
  // 1. output_type
  if (paramSet.find(key_output_type) != paramSet.end()) {
    std::string out_type = paramSet.at(key_output_type);
    if (param_output_map_.find(out_type) == param_output_map_.end()) {
      LOGE(SOURCE) << "[DataSource] [output_type] " << out_type << " not supported";
      return false;
    }
    auto output_type = param_output_map_.at(out_type);
    if (output_type != OutputType::OUTPUT_CPU) {
      if (device_id < 0) {
        LOGE(SOURCE) << "[DataSource] [output_type] " << out_type << " : device_id must be set";
        return false;
      }
    }
  }
  if (!checker.IsNum({key_interval}, paramSet, err_msg, false)) {
    LOGE(SOURCE) << "[DataSource] " << err_msg;
    return false;
  }
  // 2. decoder_type
  if (paramSet.find(key_decoder_type) != paramSet.end()) {
    std::string dec_type = paramSet.at(key_decoder_type);
    if (param_decoder_map_.find(dec_type) == param_decoder_map_.end()) {
      LOGE(SOURCE) << "[DataSource] [decoder_type] " << dec_type << " not supported";
      return false;
    }
    auto decoder_type = param_decoder_map_.at(dec_type);
    if (decoder_type != DecoderType::DECODER_CPU) {
      if (device_id < 0) {
        LOGE(SOURCE) << "[DataSource] [decoder_type] " << dec_type << " : device_id must be set";
        return false;
      }
    }
  }
  return ret;
}

int DataSource::Process(std::shared_ptr<FrameInfo> data) {
  LOGI(SOURCE) << "[DataSource] Process receive frame_id: " << data->stream_id;
  return 0;
}

DataSourceParam DataSource::GetSourceParam() const { 
  return param_; 
}


}  // namespace cnstream
