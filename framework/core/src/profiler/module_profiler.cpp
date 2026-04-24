/************************************************************************* * Copyright (C) [2020] by Cambricon, Inc. All rights reserved * * Licensed under the Apache License, Version 2.0 (the "License"); * you may not use this file except in compliance with the License. * You may obtain a copy of the License at * * http://www.apache.org/licenses/LICENSE-2.0 * * The above copyright notice and this permission notice shall be included in * all copies or substantial portions of the Software. * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN * THE SOFTWARE. *************************************************************************/

#include "profiler/module_profiler.hpp"

namespace cnstream {

ModuleProfiler::ModuleProfiler(const ProfilerConfig& config, const std::string& module_name)
    : config_(config), module_name_(module_name) {
}

void ModuleProfiler::RegisterProcess(const std::string& process_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (process_profilers_.find(process_name) != process_profilers_.end()) {
    LOGE(CORE) << "Process " << process_name << " has been registered.";
    return;
  }
  process_profilers_.emplace(process_name, std::make_unique<ProcessProfiler>(config_, process_name));
}

void ModuleProfiler::RecordProcessStart(const std::string& process_name, const RecordKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = process_profilers_.find(process_name);
  if (it != process_profilers_.end()) {
    it->second->RecordStart(key);
  }
}

void ModuleProfiler::RecordProcessEnd(const std::string& process_name, const RecordKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = process_profilers_.find(process_name);
  if (it != process_profilers_.end()) {
    it->second->RecordEnd(key);
  }
}

void ModuleProfiler::RecordProcessDropped(const std::string& process_name, const RecordKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = process_profilers_.find(process_name);
  if (it != process_profilers_.end()) {
    it->second->RecordDropped(key);
  }
}

ProcessProfile ModuleProfiler::GetProcessProfile(const std::string& process_name) {
  if (process_profilers_.find(process_name) == process_profilers_.end()) {
    LOGE(CORE) << "Process " << process_name << " is not registered.";
    return ProcessProfile();
  }
  return process_profilers_[process_name]->GetProfile();  // process lock inside
}

ModuleProfile ModuleProfiler::GetProfile() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  ModuleProfile profile;
  profile.module_name = module_name_;
  
  // 汇总所有 Process 的性能数据
  uint64_t total_completed = 0;
  uint64_t total_dropped = 0;
  
  for (auto& it : process_profilers_) {
    ProcessProfile process_profile = it.second->GetProfile();
    profile.process_profiles.emplace_back(process_profile);
    total_completed += process_profile.completed;
    total_dropped += process_profile.dropped;
  }
  profile.completed = total_completed;
  profile.dropped = total_dropped;
  profile.counter = total_completed + total_dropped;
  return profile;
}

void ModuleProfiler::OnStreamEos(const std::string& stream_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& it : process_profilers_) {
    it.second->OnStreamEos(stream_name);
  }
}

}  // namespace cnstream
