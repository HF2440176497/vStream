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
};

/**
 * @brief 携带 id 的票据句柄
 * @note WaitByTicket 看门狗凭 id 区分两类冻结：
 *       - 我不是队首 → 队首票据泄漏（持有它的任务未归还）
 *       - 我就是队首但未被唤醒 → 取票/唤醒逻辑异常
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

 private:
  void Call();
  std::queue<QueuingTicketRoot> tickets_q_;
  QueuingTicket reserved_ticket_;
  bool reserved_ = false;
  std::string name_ = "unnamed";
  std::mutex mtx_;
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
