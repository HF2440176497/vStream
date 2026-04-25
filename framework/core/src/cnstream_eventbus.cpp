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

#include "cnstream_eventbus.hpp"

#include <list>
#include <memory>
#include <thread>
#include <utility>

namespace cnstream {

EventBus::~EventBus() {
  Stop();
}

bool EventBus::IsRunning() {
  return running_.load();
}

bool EventBus::Start() {
  running_.store(true);
  event_thread_ = std::thread(&EventBus::EventLoop, this);
  return true;
}

void EventBus::Stop() {
  if (IsRunning()) {
    running_.store(false);
    if (event_thread_.joinable()) {
      event_thread_.join();
    }
  }
}

// @return The number of bus watchers that has been added to this event bus.
uint32_t EventBus::AddBusWatch(BusWatcher func) {
  RwLockWriteGuard lk(watcher_rwlock_);
  bus_watchers_.push_front(func);
  return bus_watchers_.size();
}

void EventBus::ClearAllWatchers() {
  RwLockWriteGuard lk(watcher_rwlock_);
  bus_watchers_.clear();
}

const std::list<BusWatcher> &EventBus::GetBusWatchers() const {
  RwLockReadGuard lk(watcher_rwlock_);
  return bus_watchers_;
}

bool EventBus::PostEvent(Event event) {
  if (!IsRunning()) {
    LOGW(CORE) << "Post event failed, pipeline not running";
    return false;
  }
  // LOGI(CORE) << "Receieve event from [" << event.module->GetName() << "] :" << event.message;
  queue_.Push(event);
#ifdef VSTREAM_UNIT_TEST
  if (unit_test) {
    test_eventq_.Push(event);
    unit_test = false;  // just test only once
  }
#endif
  return true;
}

/**
 * @brief 尝试获取事件
 * @details 如果运行终止，则返回 EVENT_STOP 
 */
Event EventBus::PollEvent() {
  Event event;
  event.type = EventType::EVENT_INVALID;
  while (running_.load()) {
    if (queue_.WaitAndTryPop(event, std::chrono::milliseconds(100))) {
      break;
    }
  }
  if (!running_.load()) event.type = EventType::EVENT_STOP;
  return event;
}

/**
 * @brief 
 */
void EventBus::EventLoop() {
  const std::list<BusWatcher> &kWatchers = GetBusWatchers();
  EventHandleFlag flag = EventHandleFlag::EVENT_HANDLE_NULL;

  // start loop
  while (IsRunning()) {
    Event event = PollEvent();
    if (event.type == EventType::EVENT_INVALID) {
      LOGI(CORE) << "[EventLoop] event type is invalid";
      break;
    } else if (event.type == EventType::EVENT_STOP) {
      LOGI(CORE) << "[EventLoop] Get stop event";
      break;
    }
    // std::unique_lock<std::mutex> lk(watcher_mtx_);
    RwLockReadGuard lk(watcher_rwlock_);  // 获取读锁，不修改观察者列表
    for (auto &watcher : kWatchers) {
      flag = watcher(event);  // watcher handle result
      if (flag == EventHandleFlag::EVENT_HANDLE_INTERCEPTION || flag == EventHandleFlag::EVENT_HANDLE_STOP) {
        break;
      }
    }
    if (flag == EventHandleFlag::EVENT_HANDLE_STOP) {
      break;
    }
  }
  LOGI(CORE) << "Event bus exit.";
}

#ifdef VSTREAM_UNIT_TEST
Event EventBus::PollEventToTest() {
  Event event;
  event.type = EventType::EVENT_INVALID;
  while (running_.load()) {
    if (test_eventq_.WaitAndTryPop(event, std::chrono::milliseconds(100))) {
      break;
    }
  }
  if (!running_.load()) event.type = EventType::EVENT_STOP;
  return event;
}
#endif

}  // namespace cnstream
