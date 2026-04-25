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


#include <memory>
#include <string>
#include <thread>
#include <map>

#include "cnstream_module.hpp"
#include "cnstream_pipeline.hpp"

namespace cnstream {

Module::~Module() {
  RwLockReadGuard guard(container_lock_);
  if (container_) {
    container_->ReturnModuleIdx(id_);
  }
}

void Module::SetContainer(Pipeline* container) {
  if (container) {
    {
      RwLockWriteGuard guard(container_lock_);
      container_ = container;
    }
    GetId();
  } else {
    RwLockWriteGuard guard(container_lock_);
    container_ = nullptr;
    id_ = INVALID_MODULE_ID;
  }
}

size_t Module::GetId() {
  if (id_ == INVALID_MODULE_ID) {
    RwLockReadGuard guard(container_lock_);
    if (container_)
      id_ = container_->GetModuleIdx();
  }
  return id_;
}

bool Module::PostEvent(EventType type, const std::string& msg) {
  Event event;
  event.type = type;
  event.message = msg;
  event.module_name = name_;

  return PostEvent(event);
}

/**
 * 通过 Pipeline 的 EventBus 成员 PostEvent
 */
bool Module::PostEvent(Event e) {
  RwLockReadGuard guard(container_lock_);
  if (container_) {
    return container_->GetEventBus()->PostEvent(e);
  } else {
    LOGW(CORE) << "[" << GetName() << "] module's container is not set";
    return false;
  }
}

/**
 * @return true 传输成功
 * @return false 传输失败
 */
bool Module::DoTransmitData(const std::shared_ptr<FrameInfo> data) {
  RwLockReadGuard guard(container_lock_);
  if (container_) {
    if (container_->IsRunning()) {
      return container_->ProvideData(this, data);
    }
    return false;
  } else {
    LOGE(CORE) << "[" << GetName() << "] module's container is not set";
    return false;
  }
  return false;
}

/**
 * 仅在 Pipeline::TaskLoop 中调用
 */
int Module::DoProcess(std::shared_ptr<FrameInfo> data) {
  bool removed = IsStreamRemoved(data->stream_id);
  if (!HasTransmit()) {
    if (!data->IsEos()) {
      if (!removed) {  // 并且不能是正在移除的 stream_id
        int ret = Process(data);
        if (ret != 0) {
          return ret;
        }
      }
    } else {
      this->OnEos(data->stream_id);  // 首先调用 Module 的 OnEos() 逻辑
    }
    if (DoTransmitData(data)) { return 0; } else { return -1; }
  } else {
    if (removed) {
      data->flags |= static_cast<size_t>(DataFrameFlag::FRAME_FLAG_REMOVED);
      return 0;  // TODO: 个人认为不再需要输入模块处理
    }
    return Process(data);
  }
  return -1;
}

/**
 * @brief 调用 DoTransmitData（借助 Pipeline 向下游 module 队列传输数据）
 */
bool Module::TransmitData(std::shared_ptr<FrameInfo> data) {
  if (!HasTransmit()) {
    return false;
  }
  if (DoTransmitData(data)) {
    return true;
  }
  return false;
}

ModuleProfiler* Module::GetProfiler() {
  RwLockReadGuard guard(container_lock_);
  if (container_ && container_->IsProfilingEnabled())
    return container_->GetModuleProfiler(GetName());
  return nullptr;
}

ModuleFactory* ModuleFactory::factory_ = nullptr;


class TestModuleOne : public Module, public ModuleCreator<TestModuleOne> {
  public:
   explicit TestModuleOne(const std::string& name = "ModuleOne") : Module(name) {}
   bool Open(ModuleParamSet params) override {return true;}
   void Close() override {}
   int Process(std::shared_ptr<FrameInfo> frame_info) override {return 0;}
 };


}  // namespace cnstream
