/*************************************************************************
 * Copyright (C) [2026] by vStream. All rights reserved
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

#ifndef MODULES_UTIL_FRAME_FILTER_HPP_
#define MODULES_UTIL_FRAME_FILTER_HPP_

/**
 *  \file frame_filter.hpp
 *
 *  This file contains a declaration of class FrameFilter.
 *  FrameFilter 与 ObjFilter 对齐：ObjFilter 用于对象级过滤（批内逐 obj 筛选），
 *  FrameFilter 用于帧级过滤（整帧门控，帧级不满足时该帧不进入推理批处理）。
 */

#include <memory>
#include <string>

#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {

/**
 * @brief The base class of frame filter.
 * @note Filter 会被多个 TaskLoop 线程并发调用，实现必须是无状态的（只读配置、只读 frame）。
 */
class FrameFilter : virtual public ReflexObjectEx<FrameFilter> {
 public:
  /**
   * @brief Does nothing.
   */
  virtual ~FrameFilter() {}
  /**
   * @brief Creates relative frame filter.
   *
   * @param filter_name The frame filter class name.
   *
   * @return None
   */
  static FrameFilter* Create(const std::string& filter_name) {
    return ReflexObjectEx<FrameFilter>::CreateObject(filter_name);
  }

  virtual bool Init(const std::map<std::string, std::string> &params) {
    return true;
  }

  /**
   * @brief Filters the frame.
   *
   * @param finfo: The smart pointer of struct to store origin frame data.
   *
   * @return Returns true if this frame should be inferred, otherwise returns false
   *         (false 时该帧跳过推理).
   */
  virtual bool Filter(const FrameInfoPtr& finfo) = 0;
};  // class FrameFilter

using FrameFilterPtr = std::shared_ptr<FrameFilter>;

}  // namespace cnstream

#endif  // ifndef MODULES_UTIL_FRAME_FILTER_HPP_
