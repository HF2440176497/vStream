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

#include <iostream>
#include "queuing_server.hpp"


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
    ticket = tickets_q_.back().root.get_future().share();
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
  ticket = tickets_q_.back().root.get_future().share();
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

void QueuingServer::WaitByTicket(QueuingTicket* pticket) { pticket->get(); }

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

