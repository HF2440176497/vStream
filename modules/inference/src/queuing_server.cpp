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

#include <atomic>
#include <chrono>
#include <iostream>

#include "cnstream_logging.hpp"
#include "queuing_server.hpp"

namespace {
// 票据全局唯一 id：跨资源唯一，配合 QueuingServer::name_ 可精确定位冻结点
std::atomic<uint64_t> g_next_ticket_id{1};
}  // namespace

/**
 * @brief 从队列中取出一个 ticket
 * @param reserve 是否保留当前 ticket, 保留后, 后续 PickUpTicket 会返回该 ticket
 */
QueuingTicket QueuingServer::PickUpTicket(bool reserve) {
  std::lock_guard<std::mutex> lk(mtx_);
  QueuingTicket ticket;
  if (reserved_) {
    // last ticket reserved, return it.
    ticket = reserved_ticket_;
  } else {
    // create new ticket.
    tickets_q_.push(QueuingTicketRoot());
    QueuingTicketRoot& root = tickets_q_.back();
    root.id = g_next_ticket_id++;
    ticket.fut = root.root.get_future().share();
    ticket.id = root.id;
    if (tickets_q_.size() == 1) {  // only one ticket, call at once
      Call();
    }
  }
  if (reserve) {
    // reserve current ticket for next pick up.
    reserved_ticket_ = ticket;
    tickets_q_.back().reserved_time++;
    reserved_ = true;
  } else {
    // do not reserve the current ticket
    reserved_ = false;
    // tickets_q_.back().reserved_time = 0;
  }
  return ticket;
}


QueuingTicket QueuingServer::PickUpNewTicket(bool reserve) {
  std::lock_guard<std::mutex> lk(mtx_);
  QueuingTicket ticket;
  if (reserved_) {
    // last ticket reserved, clean it.
    if (0 == tickets_q_.back().reserved_time) {
      if (static_cast<int>(tickets_q_.size()) != 1) {
          std::cout << "Internel error" << std::endl;
      }
      tickets_q_.pop();
    } else {
      tickets_q_.back().reserved_time--;
    }
    reserved_ = false;  // 清空上次状态
  }
  // create new ticket.
  tickets_q_.push(QueuingTicketRoot());
  QueuingTicketRoot& root = tickets_q_.back();
  root.id = g_next_ticket_id++;
  ticket.fut = root.root.get_future().share();
  ticket.id = root.id;
  if (tickets_q_.size() == 1) {
    // only one ticket, call at once
    Call();
  }
  if (reserve) {
    // reserve current ticket for next pick up.
    reserved_ticket_ = ticket;
    tickets_q_.back().reserved_time++;
    reserved_ = true;
  }
  return ticket;
}

/**
 * @brief 减少队首元素的保留计数
 * 如果队首元素的保留计数减为0，则从队列中移除该元素
 */
void QueuingServer::DeallingDone() {
  std::lock_guard<std::mutex> lk(mtx_);
  if (!tickets_q_.empty()) {
    if (0 == tickets_q_.front().reserved_time) {
      tickets_q_.pop();
      Call();
    } else {
      tickets_q_.front().reserved_time--;
    }
  }
}

void QueuingServer::SetFreezeWatch(uint32_t timeout_ms, bool force_release, FreezeHandler handler) {
  std::lock_guard<std::mutex> lk(mtx_);
  freeze_timeout_ms_ = timeout_ms;
  force_release_ = force_release;
  freeze_handler_ = std::move(handler);
}

bool QueuingServer::ForceReleaseHead(uint64_t waiter_id) {
  std::lock_guard<std::mutex> lk(mtx_);  // 与 Call() 的调用约定一致：持锁调用
  if (tickets_q_.empty()) {
    return false;
  }
  if (tickets_q_.front().id == waiter_id) {
    // 自己是队首却没被唤醒：取号/唤醒路径异常，补一次唤醒
    LOGW(QUEUING) << "[" << name_ << "] self-heal: re-call head ticket #" << waiter_id;
    Call();
    return true;
  }
  // 队首票据疑似泄漏（持有者已不可能归还）：强制弹出并唤醒下一个
  const uint64_t victim = tickets_q_.front().id;
  tickets_q_.pop();
  Call();
  LOGW(QUEUING) << "[" << name_ << "] self-heal: force released leaked head ticket #"
                << victim << " for waiter #" << waiter_id;
  return true;
}

void QueuingServer::WaitByTicket(QueuingTicket* pticket) {
  if (pticket == nullptr) return;
  // 基于以下核心语义：
  // 一张票据的 future 被置位，当且仅当它成为队首的那一刻
  // 要么 push 时队列只有自己（PickUpTicket/PickUpNewTicket 里的 if (size == 1) Call()）
  // 或者前驱票据 pop 后 Call() 唤醒

  // 首次等待 1s, 后续每 10s 会重复检查
  const auto kFirstWarn = std::chrono::milliseconds(1000);
  const auto kRepeatWarn = std::chrono::milliseconds(10000);
  uint32_t freeze_timeout_ms = 0;
  bool force_release = false;
  FreezeHandler freeze_handler = nullptr;
  {
    // 快照一次配置：等待期间不再持锁访问成员，避免与 SetFreezeWatch 竞争
    std::lock_guard<std::mutex> lk(mtx_);
    freeze_timeout_ms = freeze_timeout_ms_;
    force_release = force_release_;
    freeze_handler = freeze_handler_;
  }
  auto status = pticket->fut.wait_for(kFirstWarn);
  auto waited = kFirstWarn;
  // 自救去重：同一个队首只强制释放一次，避免把票据队列一路弹空
  uint64_t released_head_id = 0;
  bool reported = false;
  while (status != std::future_status::ready) {
    size_t pending = 0;
    uint64_t head_id = 0;
    uint32_t head_reserved = 0;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      pending = tickets_q_.size();
      if (!tickets_q_.empty()) {
        head_id = tickets_q_.front().id;
        head_reserved = tickets_q_.front().reserved_time;
      }
    }
    // 队列已空，但是没有复位 future
    if (pending == 0) {
      LOGW(QUEUING) << "[" << name_ << "] #" << pticket->id << " waited="
                    << waited.count() << "ms, queue is EMPTY: "
                    << "ticket was popped but its future was never set.";
    } else if (head_id == pticket->id) {
      LOGW(QUEUING) << "[" << name_ << "] #" << pticket->id << " waited="
                    << waited.count() << "ms, this ticket is the head but never called: "
                    << "anomaly in PickUp/Call path [Head Reserved: " << head_reserved << "].";
    } else {
      LOGW(QUEUING) << "[" << name_ << "] #" << pticket->id << " waited="
                    << waited.count() << "ms, blocked behind head #" << head_id
                    << " [Head Reserved: " << head_reserved << ", Pend: " << pending
                    << "]. Head ticket #" << head_id
                    << " not released: the task holding it exited without DeallingDone.";
    }

    // 冻结自救：等待超过阈值后先上报一次，再强制弹出队首票据解除阻塞。
    // 未配置（freeze_timeout_ms == 0 / force_release == false）时退化为「只告警、继续等」。
    if (force_release && freeze_timeout_ms > 0 &&
        waited >= std::chrono::milliseconds(freeze_timeout_ms) && head_id != released_head_id) {
      if (!reported) {
        reported = true;
        std::string diag = "[" + name_ + "] pipeline frozen: #" +
                           std::to_string(pticket->id) + " waited " +
                           std::to_string(waited.count()) + "ms [Head: " +
                           std::to_string(head_id) + ", Pend: " + std::to_string(pending) +
                           "], force releasing head ticket";
        LOGE(QUEUING) << diag;
        if (freeze_handler) freeze_handler(diag);
      }
      released_head_id = head_id;
      ForceReleaseHead(pticket->id);
    }

    status = pticket->fut.wait_for(kRepeatWarn);
    waited += kRepeatWarn;
  }
}


/**
 * @brief 设置队首 ticket 已处理 相当于唤醒
 * provider 完成生产
 */
void QueuingServer::Call() {
  if (!tickets_q_.empty()) {
    QueuingTicketRoot& ticket_root = tickets_q_.front();
    if (ticket_root.called) {
      // 幂等保护：
      // 自救路径可能对同一张票重复 Call
      return;
    }
    ticket_root.called = true;
    ticket_root.root.set_value();
  }
}

