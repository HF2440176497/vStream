
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

#include <iostream>
#include <memory>
#include <string>
#include <map>

#include "cnstream_frame.hpp"
#include "cnstream_module.hpp"

namespace cnstream {


std::shared_ptr<FrameInfo> FrameInfo::Create(const std::string& stream_id, bool eos) {
  if (stream_id == "") {
    LOGE(CORE) << "FrameInfo::Create() stream_id is empty string.";
    return nullptr;
  }
  std::shared_ptr<FrameInfo> ptr(new (std::nothrow) FrameInfo());
  if (!ptr) {
    LOGE(CORE) << "FrameInfo::Create() new FrameInfo failed.";
    return nullptr;
  }
  ptr->stream_id = stream_id;
  if (eos) {
	  ptr->flags |= static_cast<size_t>(DataFrameFlag::FRAME_FLAG_EOS);
    std::lock_guard<std::mutex> guard(s_eos_lock_);
    s_stream_eos_map_[stream_id] = false;
  }
  return ptr;
}

FrameInfo::~FrameInfo() {
  if (this->IsEos()) {
    std::lock_guard<std::mutex> guard(s_eos_lock_);
    s_stream_eos_map_[stream_id] = true;
  }
}


void FrameInfo::SetModulesMask(uint64_t mask) {
  std::lock_guard<std::mutex> lk(mask_lock_);
  modules_mask_ = mask;
}

uint64_t FrameInfo::GetModulesMask() {
  std::lock_guard<std::mutex> lk(mask_lock_);
  return modules_mask_;
}

uint64_t FrameInfo::MarkPassed(Module* module) {
  std::lock_guard<std::mutex> lk(mask_lock_);
  modules_mask_ |= (uint64_t)1 << module->GetId();
  return modules_mask_;
}

bool FrameInfo::MarkPassedOnce(Module* module, uint64_t* new_mask) {
  std::lock_guard<std::mutex> lk(mask_lock_);
  const uint64_t bit = (uint64_t)1 << module->GetId();
  if (modules_mask_ & bit) return false;
  modules_mask_ |= bit;
  *new_mask = modules_mask_;
  return true;
}

void FrameInfo::MarkSkipModule(Module* module) {
  std::lock_guard<std::mutex> lk(mask_lock_);
  skip_mask_ |= (uint64_t)1 << module->GetId();
}

uint64_t FrameInfo::MarkInferSkipped(uint64_t bit) {
  if (bit == 0) {
    std::lock_guard<std::mutex> lk(mask_lock_);
    return collection.HasValue(kSkipFrameTag) ? collection.Get<uint64_t>(kSkipFrameTag) : 0;
  }
  std::lock_guard<std::mutex> lk(mask_lock_);
  uint64_t mask = collection.HasValue(kSkipFrameTag) ? collection.Get<uint64_t>(kSkipFrameTag) : 0;
  mask |= bit;
  collection.Set(kSkipFrameTag, mask);
  return mask;
}

bool FrameInfo::IsModuleSkipped(Module* module) {
  std::lock_guard<std::mutex> lk(mask_lock_);
  return (skip_mask_ & ((uint64_t)1 << module->GetId())) != 0;
}

}  // namespace cnstream
