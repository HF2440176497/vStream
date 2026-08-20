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

void QueuingServer::WaitByTicket(QueuingTicket* pticket) {
  if (pticket == nullptr) return;
  // 看门狗：正常情况下票据等待仅为批次间串行耗时（毫秒级）；
  // 若队首票据未归还（任务异常退出或阻塞），后续任务将在此永久等待。
  // 凭票据 id 区分冻结类型：挡在泄漏票之后 / 我即队首但未被唤醒 / 队列空异常
  const auto kFirstWarn = std::chrono::milliseconds(1000);
  const auto kRepeatWarn = std::chrono::milliseconds(10000);
  auto status = pticket->fut.wait_for(kFirstWarn);
  auto waited = kFirstWarn;
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
    if (pending == 0) {
      LOGW(QUEUING) << "[" << name_ << "] ticket #" << pticket->id << " wait slow: waited="
                    << waited.count() << "ms, queue is EMPTY. Wake-logic anomaly: "
                    << "this ticket was popped but its future was never set.";
    } else if (head_id == pticket->id) {
      LOGW(QUEUING) << "[" << name_ << "] ticket #" << pticket->id << " wait slow: waited="
                    << waited.count() << "ms, this ticket IS the head but never called. "
                    << "Wake-logic anomaly in PickUp/Call path (head_reserved=" << head_reserved << ").";
    } else {
      LOGW(QUEUING) << "[" << name_ << "] ticket #" << pticket->id << " wait slow: waited="
                    << waited.count() << "ms, blocked behind head #" << head_id
                    << " (head_reserved=" << head_reserved << ", pending=" << pending
                    << "). Head ticket #" << head_id
                    << " not released: the task holding it exited without DeallingDone.";
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
    ticket_root.root.set_value();
  }
}

