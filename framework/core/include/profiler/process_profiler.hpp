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

#ifndef CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROCESS_PROFILER_HPP_
#define CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROCESS_PROFILER_HPP_

#include <string>
#include <map>
#include <mutex>
#include <chrono>

#include "cnstream_logging.hpp"
#include "cnstream_config.hpp"
#include "profiler/profile.hpp"

namespace cnstream {

using Time = std::chrono::steady_clock::time_point;
using Duration = std::chrono::duration<double, std::milli>;
using RecordKey = std::pair<std::string, uint64_t>; // (stream_name, timestamp)

/*!
 * @class ProcessProfiler
 *
 * @brief ProcessProfiler is the profiler for a process. A process can be a function call or a piece of code.
 *
 * @note This class is thread safe. 
 */
class ProcessProfiler: private NonCopyable {
 public:
  /*!
   * @brief Constructs a ProcessProfiler object.
   *
   * @param[in] config The configuration of the profiler.
   * @param[in] process_name The name of the process.
   *
   * @return No return value. 
   */
  explicit ProcessProfiler(const ProfilerConfig& config, const std::string& process_name);
  
  /*!
   * @brief Destructs a ProcessProfiler object.
   *
   * @return No return value. 
   */
  ~ProcessProfiler() = default;

  /*!
   * @brief Records the start of the process.
   *
   * @param[in] key The unique identifier of a FrameInfo instance.
   *
   * @return No return value. 
   */
  void RecordStart(const RecordKey& key);

  /*!
   * @brief Records the end of the process.
   *
   * @param[in] key The unique identifier of a FrameInfo instance.
   *
   * @return No return value. 
   */
  void RecordEnd(const RecordKey& key);

  /*!
   * @brief Records a dropped frame.
   *
   * @param[in] key The unique identifier of a FrameInfo instance. 
   */
  void RecordDropped(const RecordKey& key);

  /*!
   * @brief Gets the name of the process.
   *
   * @return The name of the process. 
   */
  std::string GetName() const;

  /*!
   * @brief Gets profiling results of the process during the execution of the program.
   *
   * @return Returns the profiling results. 
   */
  ProcessProfile GetProfile();

  /*!
   * @brief Clears profiling data of the stream named by ``stream_name``, as the end of the stream is reached.
   *
   * @param[in] stream_name The name of the stream, usually the ``FrameInfo::stream_id``.
   */
  void OnStreamEos(const std::string& stream_name);

 private:
  ProfilerConfig config_;
  std::mutex     mutex_;
  std::string    process_name_;

  uint64_t                  completed_ = 0;                                    /*!< The number of completed frames. */
  uint64_t                  dropped_ = 0;                                      /*!< The number of dropped frames. */
  double                    total_latency_ = 0.0;                              /*!< Total latency (ms). */
  double                    max_latency_ = 0.0;                                /*!< Maximum latency (ms). */
  double                    min_latency_ = std::numeric_limits<double>::max(); /*!< Minimum latency (ms). */
  Time                      start_time_;                                       /*!< Start time. */
  bool                      start_time_initialized_ = false;                   /*!< Whether start time has been initialized. */
  std::map<RecordKey, Time> start_times_; /*!< Records the start time of each key. */
};

inline std::string ProcessProfiler::GetName() const {
  return process_name_;
}

}  // namespace cnstream

#endif  // CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROCESS_PROFILER_HPP_