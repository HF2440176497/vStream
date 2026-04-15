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

#ifndef CNSTREAM_THREADSAFE_QUEUE_HPP_
#define CNSTREAM_THREADSAFE_QUEUE_HPP_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>
#include <functional>
#include <stdexcept>

namespace cnstream {

template <typename T>
class ThreadSafeQueue {
 public:
  explicit ThreadSafeQueue(uint32_t max_size = 0) : max_size_(max_size) {}
  ThreadSafeQueue(const ThreadSafeQueue& other) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue& other) = delete;

  bool TryPop(T& value);

  void WaitAndPop(T& value);

  bool WaitAndTryPop(T& value, const std::chrono::microseconds rel_time);

  bool Push(const T& new_value);

  bool Empty() {
    std::lock_guard<std::mutex> lk(data_m_);
    return q_.empty();
  }

  uint32_t Size() {
    std::lock_guard<std::mutex> lk(data_m_);
    return q_.size();
  }

 private:
  std::mutex data_m_;
  std::queue<T> q_;
  std::condition_variable notempty_cond_;
  std::condition_variable notfull_cond_;
  uint32_t max_size_ = 0;
};

template <typename T>
bool ThreadSafeQueue<T>::TryPop(T& value) {
  std::lock_guard<std::mutex> lk(data_m_);
  if (q_.empty()) {
    return false;
  } else {
    value = q_.front();
    q_.pop();
    return true;
  }
}

template <typename T>
void ThreadSafeQueue<T>::WaitAndPop(T& value) {
  std::unique_lock<std::mutex> lk(data_m_);
  notempty_cond_.wait(lk, [&] { return !q_.empty(); });
  value = q_.front();
  q_.pop();
}

template <typename T>
bool ThreadSafeQueue<T>::WaitAndTryPop(T& value, const std::chrono::microseconds rel_time) {
  std::unique_lock<std::mutex> lk(data_m_);
  if (notempty_cond_.wait_for(lk, rel_time, [&] { return !q_.empty(); })) {
    value = q_.front();
    q_.pop();
    return true;
  } else {
    return false;
  }
}

template <typename T>
bool ThreadSafeQueue<T>::Push(const T& new_value) {
  std::unique_lock<std::mutex> lk(data_m_);
  if (max_size_ > 0 && q_.size() >= max_size_) {
    return false;
  }
  q_.push(new_value);
  lk.unlock();
  notempty_cond_.notify_one();
  return true;
}

/**
 * @brief 线程安全的任务队列 带有最大长度与警告长度
 * @tparam Job 任务类型
 * @details 
 */
template <typename Job>
class ThreadSafeJobQueue {
 public:
  using SizeCallback = std::function<void(size_t)>;
  using ClearCallback = std::function<void(Job&)>;

  explicit ThreadSafeJobQueue(size_t max_size = 0, size_t warn_size = 0, SizeCallback warn_callback = nullptr,
                    ClearCallback clear_callback = nullptr)
      : max_size_(max_size),
        warn_size_(warn_size),
        warn_callback_(warn_callback),
        clear_callback_(clear_callback),
        run_(true) {
    if (max_size_ < 0 || warn_size_ < 0) {
      throw std::invalid_argument("size cannot be minus");
    }
    if (warn_size_ > max_size_ && max_size_ > 0) {
      throw std::invalid_argument("warn_size cannot exceed max_size");
    }
  }
  ThreadSafeJobQueue(const ThreadSafeJobQueue&) = delete;
  ThreadSafeJobQueue& operator=(const ThreadSafeJobQueue&) = delete;
  ThreadSafeJobQueue(ThreadSafeJobQueue&&) = default;
  ThreadSafeJobQueue& operator=(ThreadSafeJobQueue&&) = default;

 public:
  /**
   * @brief 获取任务队列中的任务 阻塞等待
   * @param fetch_jobs 输出参数 任务队列
   * @param max_size 最大获取任务数量
   * @return true 成功获取任务
   * @return false 队列已关闭
   */
  bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, size_t max_size) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !run_ || !queue_.empty(); });

    if (!run_) return false;

    fetch_jobs.clear();
    fetch_jobs.reserve(std::min(max_size, queue_.size()));

    while (!queue_.empty() && fetch_jobs.size() < max_size) {
      fetch_jobs.emplace_back(std::move(queue_.front()));
      queue_.pop();
    }
    if (max_size_ > 0) {  // 如果设置有最大空间限制
      lock.unlock();
      cond_space_.notify_one();
    } else {
      lock.unlock();
    }
    return true;
  }

  /**
   * @brief 添加单个任务 阻塞等待
   * @param job 任务对象
   */
  void emplace(Job&& job) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (max_size_ > 0) {
      cond_space_.wait(lock, [this] { return !run_ || queue_.size() < max_size_; });
      if (!run_) return;
    }
    queue_.push(std::move(job));
    check_queue_length();
    lock.unlock();
    cond_.notify_one();
  }

  void emplace(const Job& job) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (max_size_ > 0) {
      cond_space_.wait(lock, [this] { return !run_ || queue_.size() < max_size_; });
      if (!run_) return;
    }
    queue_.push(job);
    check_queue_length();
    lock.unlock();
    cond_.notify_one();
  }

  template <typename InputIt>
  void batch_emplace(InputIt begin, InputIt end) {
    if (begin == end) return;

    std::unique_lock<std::mutex> lock(mutex_);
    const size_t                 add_count = std::distance(begin, end);

    // 等待队列有足够空间
    if (max_size_ > 0) {
      cond_space_.wait(lock, [this, add_count] { return !run_ || (queue_.size() + add_count <= max_size_); });
      if (!run_) return;
    }
    for (auto it = begin; it != end; ++it) {
      queue_.push(std::move(*it));
    }
    check_queue_length();
    lock.unlock();
    cond_.notify_one();  // 通知消费者
  }

  /**
   * @brief 清空队列并处理未完成的任务
   */
  void clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
      auto& item = queue_.front();
      if (clear_callback_) {
        clear_callback_(item);
      }
      queue_.pop();
    }
    if (max_size_ > 0) {
      lock.unlock();
      cond_space_.notify_all();  // 释放所有等待的生产者
    }
  }

  /**
   * @brief 停止队列操作（唤醒所有等待线程）
   */
  void stop() {
    clear();  // 可以首先清空队列
    {
      std::unique_lock<std::mutex> lock(mutex_);
      run_ = false;
    }
    cond_.notify_all();
    cond_space_.notify_all();
  }

  /**
   * @brief 设置清除回调函数
   */
  void set_clear_callback(ClearCallback callback) {
    std::unique_lock<std::mutex> lock(mutex_);
    clear_callback_ = callback;
  }

  /**
   * @brief 设置警告回调函数
   */
  void set_warn_callback(SizeCallback callback) {
    std::unique_lock<std::mutex> lock(mutex_);
    warn_callback_ = callback;
  }

  size_t qsize() {
    std::unique_lock<std::mutex> lock(mutex_);
    size_t len = queue_.size();
    lock.unlock();
    return len;
  }

 private:
  /**
   * @brief 每次调用时检查队列长度，
   *        如果超过警告阈值且能允许触发告警，则调用警告回调。
   *        如果发现回落到阈值以下，允许再次触发警告。
   */
  void check_queue_length() {
    if (warn_callback_ && warn_size_ > 0 && queue_.size() >= warn_size_ && !warn_triggered_) {
      warn_triggered_ = true;
      warn_callback_(queue_.size());
    } else if (queue_.size() < warn_size_) {  // 回落到阈值以下，允许再次触发
      warn_triggered_ = false;
    }
  }

 private:
  mutable std::mutex      mutex_;
  std::condition_variable cond_;        // 通知消费者
  std::condition_variable cond_space_;  // 通知生产者
  std::queue<Job>         queue_;
  bool                    run_;
  size_t                  max_size_;   // 0 = unlimited
  size_t                  warn_size_;  // 0 = no warning
  bool                    warn_triggered_ = false;
  SizeCallback            warn_callback_;   // 长度预警回调
  ClearCallback           clear_callback_;  // 清除任务回调
};

}  // namespace cnstream

#endif  // CNSTREAM_THREADSAFE_QUEUE_HPP_
