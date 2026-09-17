/*************************************************************************
 * Copyright (C) [2026] by vStream. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *************************************************************************/

#ifndef MODULES_UTIL_INPUT_DERIVER_HPP_
#define MODULES_UTIL_INPUT_DERIVER_HPP_

/**
 *  \file input_deriver.hpp
 *
 *  This file contains a declaration of class InputDeriver.
 *
 *  InputDeriver 是"业务级模块输入定制"的可插拔组件（反射注册，同 Preproc/ObjFilter 模式）：
 *  - 推理前（前处理执行前）：
 *    从基准图派生模块专属输入图（如旋转），写入 FrameInfo collection 的模块级 kModelInputImageTag 模块级控制面
 *    preproc 通过 GetModelInputImage(package, model_name) 无感知读取；
 *  - 推理后（后处理执行后）：把本模型产生的 obj 坐标从模块派生图坐标系
 *    还原到基准图坐标系，对下游模块透明。
 *
 *  NOTE：仅支持 object_infer=false 的帧级推理模块。
 */

#include <map>
#include <memory>
#include <string>

#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {

/**
 * @brief The base class of input image deriver.
 */
class InputDeriver : virtual public ReflexObjectEx<InputDeriver> {
 public:
  virtual ~InputDeriver() {}

  /**
   * @brief Creates relative input deriver.
   *
   * @param deriver_name The input deriver class name.
   *
   * @return Returns the pointer to input deriver object.
   */
  static InputDeriver* Create(const std::string& deriver_name) {
    return ReflexObjectEx<InputDeriver>::CreateObject(deriver_name);
  }

  /**
   * @brief Initializes deriver parameters.
   *
   * @param[in] params The custom_input_derive_params.
   *
   * @return Returns true for success, otherwise returns false.
   */
  virtual bool Init(const std::map<std::string, std::string>& params) {
    params_ = params;
    return true;
  }

  /**
   * @brief 推理前调用：由基准图派生模块专属输入图并写入模块级 tag。
   *
   * 基准图为帧级派生图（若存在）或原图。实现必须生成独立的派生图，并记录坐标还原所需元信息。
   *
   * @param[in] finfo 帧信息。
   * @param[in] model_name 本模块的模型名（模块级 tag 的维度）。
   *
   * @return Returns 0 if successful, otherwise returns -1.
   */
  virtual int Derive(const FrameInfoPtr& finfo, const std::string& model_name) = 0;

  /**
   * @brief 推理后调用：把本模型产物（obj 坐标等）从模块派生图还原到基准图，使帧信息对下游模块透明。
   *
   * 仅处理 obj->model_name == model_name 的对象。必须幂等：同一帧因批次 pad
   * 可能触发多次后处理，实现需保证还原只被执行一次。
   *
   * @param[in] finfo 帧信息。
   * @param[in] model_name 本模块的模型名。
   *
   * @return Returns 0 if successful, otherwise returns -1.
   */
  virtual int Restore(const FrameInfoPtr& finfo, const std::string& model_name) = 0;

 protected:
  std::map<std::string, std::string> params_;
};  // class InputDeriver

using InputDeriverPtr = std::shared_ptr<InputDeriver>;

}  // namespace cnstream

#endif  // MODULES_UTIL_INPUT_DERIVER_HPP_
