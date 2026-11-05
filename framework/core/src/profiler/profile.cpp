/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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

#include "profiler/profile.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>

namespace cnstream {

std::string ProcessProfileToString(const ProcessProfile& profile, const std::string& indent = "  ") {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << indent << "Process Name: " << profile.process_name << "\n";
  oss << indent << "Window: " << profile.window_ms << " ms\n";
  oss << indent << "Frames: " << profile.counter << " (Completed: " << profile.completed 
      << ", Dropped: " << profile.dropped << ")\n";
  oss << indent << "Latency: Avg " << profile.avg_latency << " ms, Max " << profile.max_latency 
      << " ms, Min " << profile.min_latency << " ms\n";
  oss << indent << "Throughput: " << profile.fps << " FPS\n";
  if (profile.inflight > 0) {
    oss << indent << "Inflight: " << profile.inflight
        << " (Oldest: " << profile.oldest_inflight_ms << " ms)\n";
  }
  if (profile.purged > 0) {
    oss << indent << "Purged Records: " << profile.purged << "\n";
  }
  return oss.str();
}

std::string ModuleProfileToString(const ModuleProfile& profile) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "Module Name: " << profile.module_name << "\n";
  oss << "Window Frames: " << profile.counter << " (Completed: " << profile.completed 
      << ", Dropped: " << profile.dropped << ")\n\n";
  
  if (!profile.process_profiles.empty()) {
    oss << "Process Profiles:\n";
    for (const auto& process_profile : profile.process_profiles) {
      oss << ProcessProfileToString(process_profile) << "\n";
    }
  }
  return oss.str();
}

std::ostream& operator<<(std::ostream& os, const ProcessProfile& profile) {
  os << ProcessProfileToString(profile, "");
  return os;
}

std::ostream& operator<<(std::ostream& os, const ModuleProfile& profile) {
  os << ModuleProfileToString(profile);
  return os;
}

}  // namespace cnstream
