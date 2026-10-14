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

#ifndef MODULES_INFERENCE_SRC_QUEUING_SERVER_HPP_
#define MODULES_INFERENCE_SRC_QUEUING_SERVER_HPP_

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <initializer_list>
#include <mutex>
#include <queue>
#include <string>
#include <vector>


struct QueuingTicketRoot {
  std::promise<void> root;
  uint32_t reserved_time = 0;
  // 看门狗诊断用：push 时分配的全局唯一票据 id，用于比对“我是谁/队首是谁”
  uint64_t id = 0;
  // 是否已被 Call() 唤醒过。promise::set_value 只能调一次，重复调用会抛
  // future_error；强制自救路径可能重复唤醒同一张票，这里做幂等保护
  bool called = false;
};

/**
 * @brief 携带 id 的票据句柄
 * @note WaitByTicket 看门狗凭 id 区分两类冻结：
 *       - 不是队首 → 队首票据泄漏（持有它的任务未归还）
 *       - 是队首但未被唤醒 → 取票/唤醒逻辑异常
 */
struct QueuingTicket {
  std::shared_future<void> fut;
  uint64_t id = 0;
};

class QueuingServerTest;
class QueuingServer {
 public:
  friend class QueuingServerTest;
  QueuingTicket PickUpTicket(bool reserve = false);
  QueuingTicket PickUpNewTicket(bool reserve = false);
  void DeallingDone();
  void WaitByTicket(QueuingTicket* pticket);

  // 日志标识：定位哪个资源（模块/缓冲）发生等待阻塞
  void SetName(const std::string& name) { name_ = name; }

  using FreezeHandler = std::function<void(const std::string&)>;

  /**
   * @brief 冻结看门狗：等待超过后主动上报，并可选强制自救
   * @param timeout_ms    超过该时长判定为冻结；<=0 表示永不自救（只保留告警日志）
   * @param force_release 是否强制弹出队首票据来解除阻塞
   * @param handler       冻结上报回调
   *
   * @note 原理：票据被置位只有两种可能（成为队首 / 队首被弹出后 Call）。
   *       若长时间既不前进也无进展，说明持有队首票据的任务已经不可能归还
   *       因此默认策略是上报一次后强制弹出队首票据，让流水线继续运行。
   *       若宁可卡死也不能有并发访问，把 force_release 设为 false。
   */
  void SetFreezeWatch(uint32_t timeout_ms, bool force_release, FreezeHandler handler);

 private:
  void Call();
  /**
   * @brief 强制解除队首票据，用于冻结自救
   * @return true 已采取措施（要么补唤醒自己，要么弹出队首）
   */
  bool ForceReleaseHead(uint64_t waiter_id);
  std::queue<QueuingTicketRoot> tickets_q_;
  QueuingTicket reserved_ticket_;
  bool reserved_ = false;
  std::string name_ = "unnamed";
  std::mutex mtx_;

  // 冻结看门狗配置
  uint32_t freeze_timeout_ms_ = 0;     // 0 = 未启用（只告警）
  bool force_release_ = false;
  FreezeHandler freeze_handler_ = nullptr;
};

/**
 * @brief RAII 票据归还守卫
 * @note 单 buffer 票据队列按 FIFO 唤醒：队首票据若因任务抛出异常而未归还
 *       （DeallingDone 未执行），其后所有任务的 WaitByTicket 将永久阻塞，
 *       整条流水线冻结。任务 lambda 必须通过本守卫保证异常路径也归还票据。
 */
class DeallingDoneGuard {
 public:
  explicit DeallingDoneGuard(std::initializer_list<QueuingServer*> servers) : servers_(servers) {}
  ~DeallingDoneGuard() {
    for (auto* server : servers_) {
      if (server != nullptr) server->DeallingDone();
    }
  }
  DeallingDoneGuard(const DeallingDoneGuard&) = delete;
  DeallingDoneGuard& operator=(const DeallingDoneGuard&) = delete;

 private:
  std::vector<QueuingServer*> servers_;
};


#endif  // MODULES_INFERENCE_SRC_QUEUING_SERVER_HPP_
