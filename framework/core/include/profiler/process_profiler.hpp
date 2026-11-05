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

#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

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
   * @brief Gets profiling results of the process during the last profiling window.
   *
   * 统计口径为最近 ``window_ms`` 的滑动窗口，而非进程启动以来的累计值。
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
  /**
   * @brief 滑动窗口内的一个统计桶。
   *
   * index 为该桶对应的绝对桶序号（自 start_time_ 起按 bucket_ms 递增），
   * 槽位被复用时通过 index 判断数据是否仍属于当前窗口，因此无需显式清理中间桶。
   */
  struct Bucket {
    int64_t  index = -1;
    uint64_t completed = 0;
    int64_t  dropped = 0;
    double   latency_sum = 0.0;
    double   latency_max = 0.0;
    double   latency_min = std::numeric_limits<double>::max();
  };

  /*! @brief 当前时间对应的绝对桶序号，未初始化时返回 -1 */
  int64_t CurrentBucketIndex(Time now) const;
  /*! @brief 取得当前时间对应的可写桶，槽位过期时重置 */
  Bucket& CurrentBucket(Time now);
  /*! @brief 清理超过窗口上限仍未结束的残留记录，最多每 1s 执行一次（调用方需持锁） */
  void PurgeStaleRecords(Time now);

  ProfilerConfig config_;
  std::mutex     mutex_;
  std::string    process_name_;

  int64_t                 window_ms_ = 5000;  /*!< 统计窗口长度 */
  int64_t                 bucket_ms_ = 1000;  /*!< 分桶粒度 */
  std::vector<Bucket>     buckets_;           /*!< 环形桶数组 */
  size_t                  bucket_count_ = 0;

  Time                      start_time_;                       /*!< 首个记录的时间基准 */
  bool                      start_time_inited_ = false;
  Time                      last_purge_time_;                  /*!< 上次清理残留记录的时间 */
  bool                      last_purge_inited_ = false;

  std::map<RecordKey, Time> start_times_; /*!< 未结束记录的起始时间 */
  uint64_t                  purged_ = 0;  /*!< 生命周期内清理的残留记录数 */
};

inline std::string ProcessProfiler::GetName() const {
  return process_name_;
}

}  // namespace cnstream

#endif  // CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROCESS_PROFILER_HPP_