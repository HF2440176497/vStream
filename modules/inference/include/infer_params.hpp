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

#ifndef MODULES_INFERENCE_INFER_PARAMS_HPP_
#define MODULES_INFERENCE_INFER_PARAMS_HPP_

#include <functional>
#include <set>
#include <string>
#include <map>

#include "cnstream_config.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {

struct InferParams {
  DevType device_type = DevType::CPU;
  uint32_t device_id = -1;
  bool object_infer = false;
  float threshold = 0.0f;
  uint32_t infer_interval = 1;
  uint32_t batching_timeout = 3000;  // ms
  uint32_t trans_data_size = 20;  // queue size
  std::string model_path;
  uint32_t input_ordered_index = 0;  // 单输入单输出模型的张量索引，默认0
  uint32_t output_ordered_index = 0;
  std::string preproc_name;
  std::string postproc_name;
  std::string obj_filter_name;
  std::string dump_resized_image_dir = "";  // debug option, dump images(offline-model's input) before infer.
  bool saving_infer_input = false;
  std::map<std::string, std::string> custom_preproc_params;
  std::map<std::string, std::string> custom_postproc_params;
};  // struct InferParams

struct InferParamDesc {
  std::string name;
  std::string desc_str;
  std::string default_value;  // parser 负责解析为对应的类型
  std::string type;  // eg. bool
  std::function<bool(const std::string &value, InferParams *param_set)> parser = NULL;
  bool IsLegal() const {
    return name != "" && type != "" && parser;
  }
};  // struct InferParamDesc

struct InferParamDescLessCompare {
  bool operator() (const InferParamDesc &p1, const InferParamDesc &p2) const {
    return p1.name < p2.name;
  }
};  // struct InferParamDescLessCompare

class InferParamManager {
 public:
  void RegisterAll(ParamRegister *pregister);
  bool ParseBy(const ModuleParamSet &raw_params, InferParams *pout);

 private:
  bool RegisterParam(ParamRegister *pregister, const InferParamDesc &param_desc);
  std::set<InferParamDesc, InferParamDescLessCompare> param_descs_;
};  // struct InferParams

}  // namespace cnstream

#endif  // MODULES_INFERENCE_INFER_PARAMS_HPP_

