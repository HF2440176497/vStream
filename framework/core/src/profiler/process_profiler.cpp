/************************************************************************* * Copyright (C) [2020] by Cambricon, Inc. All rights reserved * * Licensed under the Apache License, Version 2.0 (the "License"); * you may not use this file except in compliance with the License. * You may obtain a copy of the License at * * http://www.apache.org/licenses/LICENSE-2.0 * * Unless required by applicable law or agreed to in writing, software * distributed under the License is distributed on an "AS IS" BASIS, * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. * See the License for the specific language governing permissions and * limitations under the License. *************************************************************************/

#include <limits>

#include "profiler/process_profiler.hpp"

namespace cnstream {

namespace {
constexpr int64_t kDefaultWindowMs = 5000;
constexpr int64_t kDefaultBucketMs = 1000;
// 残留记录清理：最多每 1s 触发一次扫描，超过 3 个窗口仍未结束的记录视为残留
constexpr int64_t kPurgeIntervalMs = 1000;
constexpr int64_t kPurgeWindowMultiplier = 3;
}  // namespace

ProcessProfiler::ProcessProfiler(const ProfilerConfig& config, const std::string& process_name)
    : config_(config), process_name_(process_name) {
  bucket_ms_ = config.bucket_ms > 0 ? config.bucket_ms : kDefaultBucketMs;
  window_ms_ = config.window_ms > 0 ? config.window_ms : kDefaultWindowMs;
  if (window_ms_ < bucket_ms_) {
    window_ms_ = bucket_ms_;
  }
  bucket_count_ = static_cast<size_t>((window_ms_ + bucket_ms_ - 1) / bucket_ms_);
  if (bucket_count_ == 0) {
    bucket_count_ = 1;
  }
  buckets_.assign(bucket_count_, Bucket());
}

int64_t ProcessProfiler::CurrentBucketIndex(Time now) const {
  if (!start_time_inited_) {
    return -1;
  }
  double elapsed_ms =
      std::chrono::duration_cast<Duration>(now - start_time_).count();
  return static_cast<int64_t>(elapsed_ms / static_cast<double>(bucket_ms_));
}

/**
 * 此处返回当前桶的引用，用于更新
 */
ProcessProfiler::Bucket& ProcessProfiler::CurrentBucket(Time now) {
  if (!start_time_inited_) {
    start_time_ = now;
    start_time_inited_ = true;
  }
  int64_t idx = CurrentBucketIndex(now);
  Bucket& bucket =
      buckets_[static_cast<size_t>(idx % static_cast<int64_t>(bucket_count_))];
  if (bucket.index != idx) {
    // 槽位被复用：旧数据已滑出窗口，直接重置
    bucket = Bucket();
    bucket.index = idx;
  }
  return bucket;
}

void ProcessProfiler::PurgeStaleRecords(Time now) {
  if (!start_time_inited_) {
    return;
  }
  if (last_purge_inited_ &&
      last_purge_time_ + std::chrono::milliseconds(kPurgeIntervalMs) > now) {
    return;
  }
  last_purge_time_ = now;
  last_purge_inited_ = true;

  const double purge_after_ms =
      static_cast<double>(window_ms_ * kPurgeWindowMultiplier);
  for (auto it = start_times_.begin(); it != start_times_.end();) {
    double age_ms =
        std::chrono::duration_cast<Duration>(now - it->second).count();
    if (age_ms > purge_after_ms) {
      it = start_times_.erase(it);
      ++purged_;
    } else {
      ++it;
    }
  }
}

void ProcessProfiler::RecordStart(const RecordKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  Time now = std::chrono::steady_clock::now();
  if (!start_time_inited_) {
    start_time_ = now;
    start_time_inited_ = true;
  }
  start_times_[key] = now;
  PurgeStaleRecords(now);
}

void ProcessProfiler::RecordEnd(const RecordKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = start_times_.find(key);
  if (it == start_times_.end()) {
    // LOGW(CORE) << process_name_ << ": No start record found: " << key.first;
    return;
  }
  Time now = std::chrono::steady_clock::now();
  double latency_ms =
      std::chrono::duration_cast<Duration>(now - it->second).count();
  start_times_.erase(it);

  // 延迟按“结束时刻”归入所在桶
  Bucket& bucket = CurrentBucket(now);
  ++bucket.completed;
  bucket.latency_sum += latency_ms;
  if (latency_ms > bucket.latency_max) {
    bucket.latency_max = latency_ms;
  }
  if (latency_ms < bucket.latency_min) {
    bucket.latency_min = latency_ms;
  }
  PurgeStaleRecords(now);
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
  Time now = std::chrono::steady_clock::now();
  Bucket& bucket = CurrentBucket(now);
  ++bucket.dropped;
  PurgeStaleRecords(now);
}

ProcessProfile ProcessProfiler::GetProfile() {
  std::lock_guard<std::mutex> lock(mutex_);

  ProcessProfile profile;
  profile.process_name = process_name_;
  profile.window_ms = static_cast<uint64_t>(window_ms_);

  Time now = std::chrono::steady_clock::now();
  double latency_sum = 0.0;
  double latency_max = 0.0;
  double latency_min = std::numeric_limits<double>::max();

  if (start_time_inited_) {
    int64_t cur_idx = CurrentBucketIndex(now);

    // CurrentBucketIndex 可知 index 按照经过的时间进行递增
    // index 最多向前找 bucket_count 的记录
    const int64_t oldest_valid = cur_idx - static_cast<int64_t>(bucket_count_);
    for (const auto& bucket : buckets_) {
      // 仅合并仍落在窗口内的桶：index ∈ (cur - K, cur]
      if (bucket.index <= oldest_valid || bucket.index > cur_idx) {
        continue;
      }
      profile.completed += bucket.completed;
      profile.dropped += bucket.dropped;
      latency_sum += bucket.latency_sum;
      if (bucket.completed > 0) {
        if (bucket.latency_max > latency_max) {
          latency_max = bucket.latency_max;
        }
        if (bucket.latency_min < latency_min) {
          latency_min = bucket.latency_min;
        }
      }
    }
  }

  profile.counter = profile.completed + profile.dropped;
  profile.avg_latency =
      profile.completed > 0 ? latency_sum / profile.completed : 0.0;
  profile.max_latency = latency_max;
  profile.min_latency =
      (latency_min == std::numeric_limits<double>::max()) ? 0.0 : latency_min;
  profile.fps =
      window_ms_ > 0 ? profile.completed * 1000.0 / window_ms_ : 0.0;

  // 当前仍在途的记录：数量与最老一条的已等待时长，用于暴露“正在阻塞”
  profile.inflight = start_times_.size();
  double oldest_inflight_ms = 0.0;
  for (const auto& record : start_times_) {
    double age_ms =
        std::chrono::duration_cast<Duration>(now - record.second).count();
    if (age_ms > oldest_inflight_ms) {
      oldest_inflight_ms = age_ms;
    }
  }
  profile.oldest_inflight_ms = oldest_inflight_ms;
  profile.purged = purged_;
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
