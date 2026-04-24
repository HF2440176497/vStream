/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
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

#include "infer_params.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define ASSERT(value)                           \
  {                                             \
    bool __attribute__((unused)) ret = (value); \
    assert(ret);                                \
  }

namespace cnstream {

static bool STR2BOOL(const std::string &value, bool *ret) {
  if (!ret) return false;
  static const std::set<std::string> true_value_list = {"1", "true", "True", "TRUE"};
  static const std::set<std::string> false_value_list = {"0", "false", "False", "FALSE"};

  if (true_value_list.find(value) != true_value_list.end()) {
    *ret = true;
    return true;
  }
  if (false_value_list.find(value) != false_value_list.end()) {
    *ret = false;
    return true;
  }
  return false;
}

static bool STR2U32(const std::string &value, uint32_t *ret) {
  if (!ret) return false;
  unsigned long t = 0;  // NOLINT
  try {
    t = stoul(value);
    if (t > std::numeric_limits<uint32_t>::max()) return false;
    *ret = t;
  } catch (std::exception &e) {
    LOGE(INFERENCER) << "STR2U32 failed. value is " << value;
    return false;
  }
  return true;
}

static bool STR2FLOAT(const std::string &value, float *ret) {
  if (!ret) return false;
  try {
    *ret = stof(value);
  } catch (std::exception &e) {
    return false;
  }
  return true;
}

/**
 * invoked in Inference::Inference
 */
void InferParamManager::RegisterAll(ParamRegister *pregister) {
  InferParamDesc param;
  param.name = "model_path";
  param.desc_str = "Required. The path of the offline model.";
  param.default_value = "";
  param.type = "string";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    param_set->model_path = value;
    return true;
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "input_ordered_index";
  param.desc_str = "Optional. The index of the input tensor in the model. Default is 0.";
  param.default_value = "0";
  param.type = "uint32";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    return STR2U32(value, &param_set->input_ordered_index);
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "postproc_name";
  param.desc_str =
      "Required. The class name for postprocess. The class specified by this name "
      "must inherited from class cnstream::Postproc when object_infer set to false, "
      "otherwise the class specified by this name must inherit from class "
      "cnstream::ObjPostproc.";
  param.default_value = "";
  param.type = "string";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    param_set->postproc_name = value;
    return true;
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "preproc_name";
  param.desc_str =
      "Optional. The class name for custom preprocessing. The class specified by this"
      " name must inherited from class cnstream::Preproc when object_infer is false, "
      "otherwise the class specified by this name must inherit from class cnstream::ObjPreproc. ";
  param.default_value = "";
  param.type = "string";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    param_set->preproc_name = value;
    return true;
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "device_type";
  param.desc_str = "Optional. The inference device type. CPU/CUDA/...";
  param.default_value = "cpu";
  param.type = "string";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    if (device_type_map.find(value) == device_type_map.end()) {
      return false;
    }
    param_set->device_type = device_type_map[value];
    return true;
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "device_id";
  param.desc_str = "Optional. MLU device ordinal number.";
  param.default_value = "0";
  param.type = "uint32";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    return STR2U32(value, &param_set->device_id);
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "batching_timeout";
  param.desc_str = "Optional. The batching timeout. unit[ms].";
  param.default_value = "3000";
  param.type = "uint32";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    return STR2U32(value, &param_set->batching_timeout);
  };
  ASSERT(RegisterParam(pregister, param));

  // param.name = "threshold";
  // param.desc_str = "Optional. The threshold pass to postprocessing function.";
  // param.default_value = "0";
  // param.type = "float";
  // param.parser = [](const std::string &value, InferParams *param_set) -> bool {
  //   return STR2FLOAT(value, &param_set->threshold);
  // };
  // ASSERT(RegisterParam(pregister, param));

  param.name = "infer_interval";
  param.desc_str = "Optional. Inferencing one frame every [infer_interval] frames.";
  param.default_value = "1";
  param.type = "uint32";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    return STR2U32(value, &param_set->infer_interval);
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "trans_data_size";
  param.desc_str = "Optional. The size of the trans_data_helper queue.";
  param.default_value = "20";
  param.type = "uint32";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    return STR2U32(value, &param_set->trans_data_size);
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "postproc_on_device";
  param.desc_str = "Optional. Whether to postprocess on device. Default is false.";
  param.default_value = "false";
  param.type = "bool";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    return STR2BOOL(value, &param_set->postproc_on_device);
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "object_infer";
  param.desc_str =
      "Optional. if object_infer is set to true, the detection target is used as the input to"
      " inferencing. if it is set to false, the video frame is used as the input to inferencing."
      " 1/true/TRUE/True/0/false/FALSE/False these values are accepted.";
  param.default_value = "false";
  param.type = "bool";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    return STR2BOOL(value, &param_set->object_infer);
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "obj_filter_name";
  param.desc_str =
      "Optional. The class name for object filter. See cnstream::ObjFilter. "
      "This parameter is valid when this parameter is true. "
      "No object will be filtered when this parameter not set.";
  param.default_value = "";
  param.type = "string";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    param_set->obj_filter_name = value;
    return true;
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "dump_resized_image_dir";
  param.desc_str = "Optional. Where to dump the resized image.";
  param.default_value = "";
  param.type = "string";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    param_set->dump_resized_image_dir = value;
    return true;
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "saving_infer_input";
  param.desc_str = "Optional. Save the data close to inferencing ";
  param.default_value = "false";
  param.type = "bool";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    return STR2BOOL(value, &param_set->saving_infer_input);
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "custom_preproc_params";
  param.desc_str =
      "Optional. Custom preprocessing parameters. After the inferencer module creates an instance of "
      "the preprocessing class specified by preproc_name or obj_preproc_name, the Init function of the specified "
      "preprocessing class will be called, and these parameters will be passed to Init. See Preproc::Init "
      "and ObjPreproc::Init for detail.";
  param.default_value = "";
  param.type = "json string";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    if (value.empty()) {
      param_set->custom_preproc_params.clear();
      return true;
    }
    
    auto doc = nlohmann::ordered_json::parse(value);
    if (!doc.is_object()) {
      LOGE(CORE) << "Custom preprocessing parameters configuration must be object type.";
      return false;
    }
    param_set->custom_preproc_params.clear();

    std::string value_str {};
    for (auto& [key, val] : doc.items()) {
      if (!val.is_string()) {
        value_str = val.dump();
      } else {
        value_str = val.get<std::string>();
      }
      param_set->custom_preproc_params[key] = value_str;
    }
    return true;
  };
  ASSERT(RegisterParam(pregister, param));

  param.name = "custom_postproc_params";
  param.desc_str =
      "Optional. Custom postprocessing parameters. After the inferencer module creates an instance of "
      "the postprocessing class specified by postproc_name or obj_postproc_name, the Init function of the specified "
      "postprocessing class will be called, and these parameters will be passed to Init. See Postproc::Init "
      "and ObjPostproc::Initfor detail.";
  param.default_value = "";
  param.type = "json string";
  param.parser = [](const std::string &value, InferParams *param_set) -> bool {
    if (value.empty()) {
      param_set->custom_postproc_params.clear();
      return true;
    }
    auto doc = nlohmann::ordered_json::parse(value);
    if (!doc.is_object()) {
      LOGE(CORE) << "Custom postprocessing parameters configuration must be object type.";
      return false;
    }
    param_set->custom_postproc_params.clear();

    std::string value_str {};
    for (auto& [key, val] : doc.items()) {
      if (!val.is_string()) {
        value_str = val.dump();
      } else {
        value_str = val.get<std::string>();
      }
      param_set->custom_postproc_params[key] = value_str;
    }
    return true;
  };
  ASSERT(RegisterParam(pregister, param));
}

/**
 * param_desc.name: 参数名
 */
bool InferParamManager::RegisterParam(ParamRegister *pregister, const InferParamDesc &param_desc) {
  if (!pregister) return false;
  if (!param_desc.IsLegal()) return false;
  auto insert_ret = param_descs_.insert(param_desc);
  if (!insert_ret.second) return false;
  std::string desc = param_desc.desc_str + " --- " + "type : [" + param_desc.type + "] --- " + "default value : [" +
                     param_desc.default_value + "]";
  pregister->Register(param_desc.name, desc);  // pair: {key: desc}
  return true;
}

/**
 * @brief 解析来自 raw_params 的模型参数到 pout
 * invoked in Inference::Open
 */
bool InferParamManager::ParseBy(const ModuleParamSet &raw_params, InferParams *pout) {
  if (!pout) return false;
  ModuleParamSet raws = raw_params;
  for (const InferParamDesc &desc : param_descs_) {
    std::string value = desc.default_value;
    auto it = raws.find(desc.name);
    if (it != raws.end()) {
      value = it->second;
      raws.erase(it);
    }
    // 在 raw_params 找到就采用设置值，否则采用默认值
    // 调用回调函数进行校验
    if (!desc.parser(value, pout)) {
      LOGE(INFERENCER) << "Parse parameter [" << desc.name << "] failed. value is [" << value << "]";
      return false;
    }
  }
  // 应当剩下 CNS_JSON_DIR_PARAM_NAME 参数
  for (const auto &it : raws) {
    if (it.first != CNS_JSON_DIR_PARAM_NAME) {
      LOGE(INFERENCER) << "Parameter named [" << it.first << "] did not registered.";
      return false;
    }
  }
  return true;
}

}  // namespace cnstream
