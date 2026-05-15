/************************************************************************* * Copyright (C) [2020] by Cambricon, Inc. All rights reserved * * Licensed under the Apache License, Version 2.0 (the "License"); * you may not use this file except in compliance with the License. * You may obtain a copy of the License at * * http://www.apache.org/licenses/LICENSE-2.0 * * The above copyright notice and this permission notice shall be included in * all copies or substantial portions of the Software. * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN * THE SOFTWARE. *************************************************************************/

#include <limits>

#include "profiler/process_profiler.hpp"

namespace cnstream {

ProcessProfiler::ProcessProfiler(const ProfilerConfig& config, const std::string& process_name)
    : config_(config), process_name_(process_name) {
}

void ProcessProfiler::RecordStart(const RecordKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!start_time_initialized_) {
    start_time_ = std::chrono::steady_clock::now();
    start_time_initialized_ = true;
  }
  start_times_[key] = std::chrono::steady_clock::now();
}

void ProcessProfiler::RecordEnd(const RecordKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = start_times_.find(key);
  if (it == start_times_.end()) {
    // LOGW(CORE) << process_name_ << ": No start record found: " << key.first;
    return;
  }
  Time start_time = it->second;
  Time end_time = std::chrono::steady_clock::now();
  Duration latency = std::chrono::duration_cast<Duration>(end_time - start_time);
  
  completed_++;
  double latency_ms = latency.count();
  total_latency_ += latency_ms;
  
  if (latency_ms > max_latency_) {
    max_latency_ = latency_ms;
  }
  if (latency_ms < min_latency_) {
    min_latency_ = latency_ms;
  }
  // 移除已完成的记录
  start_times_.erase(it);
}

/**
 * @brief 记录丢弃的帧数
 * 当存在 start_record 时丢弃 start，认为有关记录失效
 */
void ProcessProfiler::RecordDropped(const RecordKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = start_times_.find(key);
  if (it != start_times_.end()) {
    start_times_.erase(it);
  }
  dropped_++;
}

ProcessProfile ProcessProfiler::GetProfile() {
  std::lock_guard<std::mutex> lock(mutex_);
  ProcessProfile profile;
  profile.process_name = process_name_;
  profile.completed = completed_;
  profile.dropped = dropped_;
  profile.counter = completed_ + dropped_;  // total frames
  if (completed_ > 0) {
    profile.avg_latency = total_latency_ / completed_;
  } else {
    profile.avg_latency = 0.0;
  }
  profile.max_latency = max_latency_;
  profile.min_latency = (min_latency_ == std::numeric_limits<double>::max()) ? 0.0 : min_latency_;
  
  Time now = std::chrono::steady_clock::now();
  if (start_time_initialized_) {
    Duration elapsed = std::chrono::duration_cast<Duration>(now - start_time_);
    double elapsed_ms = elapsed.count();
    if (elapsed_ms > 0) {
      profile.fps = completed_ / (elapsed_ms / 1000.0);
    } else {
      profile.fps = 0.0;
    }
  } else {
    profile.fps = 0.0;
  }
  return profile;
}

void ProcessProfiler::OnStreamEos(const std::string& stream_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = start_times_.begin(); it != start_times_.end();) {
    if (it->first.first == stream_name) {
      // dropped_++;  // Sasha: Eos 帧不计数为丢弃
      it = start_times_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace cnstream