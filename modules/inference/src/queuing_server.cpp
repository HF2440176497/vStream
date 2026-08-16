/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *************************************************************************/

#include "queuing_server.hpp"

QueuingTicket QueuingServer::PickUpTicket(bool reserve, bool chain) {
  std::lock_guard<std::mutex> lk(mtx_);
  // 上一张票据仍被保留且未出队：共享之（同一 run 的多持有者）
  if (reserved_ && reserved_node_ && reserved_node_->in_queue) {
    if (reserved_node_->released) {
      // 复活：批内先到帧的任务已全部释放（holders 归零），但节点被前序未释放节点
      // 堵在队列中尚未出队。本帧共享该节点后必须撤销 released 标记，否则节点
      // 轮转到队首时会被提前出队、slot 被转交 H2D，而本帧预处理任务尚未写完该 buffer
      reserved_node_->released = false;
    }
    reserved_node_->holders++;  // 每次取票配对一次 DeallingDone
    if (!reserve) {
      // 本共享组的最后一次取票，关闭保留状态
      reserved_ = false;
    }
    return QueuingTicket(reserved_node_);
  }

  auto node = std::make_shared<Node>();
  node->chain = chain;
  node->in_queue = true;
  node->holders = 1;
  tickets_q_.push_back(node);
  if (reserve) {
    reserved_node_ = node;
    reserved_ = true;
  }
  ServeLocked();
  return QueuingTicket(node);
}

QueuingTicket QueuingServer::PickUpNewTicket(bool reserve, bool chain) {
  std::lock_guard<std::mutex> lk(mtx_);
  // 新票据终结当前共享组；旧票据的持有计数由其真实持有者的 DeallingDone 消化
  reserved_ = false;
  reserved_node_.reset();

  auto node = std::make_shared<Node>();
  node->chain = chain;
  node->in_queue = true;
  node->holders = 1;
  tickets_q_.push_back(node);
  if (reserve) {
    reserved_node_ = node;
    reserved_ = true;
  }
  ServeLocked();
  return QueuingTicket(node);
}

void QueuingServer::DeallingDone(const QueuingTicket& ticket) {
  if (!ticket.Valid()) return;
  std::lock_guard<std::mutex> lk(mtx_);
  Node* node = ticket.node_.get();
  if (!node->in_queue || node->holders == 0) return;  // 重复/无效释放
  if (--node->holders == 0) {
    node->released = true;
  }
  ServeLocked();
}

void QueuingServer::SetPoolSize(size_t n) {
  std::lock_guard<std::mutex> lk(mtx_);
  free_slots_.clear();
  for (size_t i = 0; i < n; ++i) {
    free_slots_.push_back(static_cast<int>(i));
  }
}

/**
 * @brief 出队已释放的队首节点，并按 FIFO 顺序唤醒可服务的节点
 *
 * 出队规则：队首连续 released 的节点依次弹出；弹出的 slot
 *  - 若其直接后继是 chain 票据 → 转交给后继（同 run 下一阶段继续用同一 buffer）
 *  - 否则归还空闲池，供后续新 run 使用
 *
 * 唤醒规则：从队首扫描，跳过已服务节点；
 *  - chain 节点需其前驱已释放出队并完成 slot 转交（slot >= 0）
 *  - 非 chain 节点需空闲池有 slot
 * 遇到不可服务节点即停止，保证唤醒顺序与取票顺序一致（FIFO）
 */
void QueuingServer::ServeLocked() {
  // holders == 0 双重确认：防御"复活后 released 未复位"类缺陷导致提前出队
  while (!tickets_q_.empty() && tickets_q_.front()->released && tickets_q_.front()->holders == 0) {
    std::shared_ptr<Node> node = tickets_q_.front();
    tickets_q_.pop_front();
    node->in_queue = false;
    if (node == reserved_node_) {
      reserved_ = false;
      reserved_node_.reset();
    }
    if (!tickets_q_.empty() && tickets_q_.front()->chain) {
      tickets_q_.front()->slot = node->slot;  // 同 run：slot 转交
      if (node->slot < 0) {
        // 前驱未被服务即释放（异常路径）：后继的链式期待落空，退化为独立 run
        tickets_q_.front()->chain = false;
      }
    } else if (node->slot >= 0) {
      free_slots_.push_back(node->slot);
    }
  }

  for (auto& node : tickets_q_) {
    if (node->served) continue;
    if (node->chain) {
      if (node->slot < 0) break;  // 前驱未释放，等待其出队转交
    } else {
      if (free_slots_.empty()) break;  // 空闲 slot 耗尽，等待在途 run 释放
      node->slot = free_slots_.front();
      free_slots_.pop_front();
    }
    node->served = true;
    node->served_pr.set_value();
  }
}
