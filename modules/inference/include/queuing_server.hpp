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

#ifndef MODULES_INFERENCE_SRC_QUEUING_SERVER_HPP_
#define MODULES_INFERENCE_SRC_QUEUING_SERVER_HPP_

#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>

/**
 * @brief 排队票据（值语义句柄，可拷贝、可在多持有者间共享同一底层节点）
 *
 * 票据代表"对资源池中某个 slot 的一次占用预约"：
 * - Wait()  阻塞直到票据被服务（slot 已绑定）
 * - Slot()  返回绑定的 slot 下标（Wait 返回后有效）
 * - 释放通过 QueuingServer::DeallingDone(ticket) 完成
 */
class QueuingTicket {
 public:
  QueuingTicket() = default;

  bool Valid() const { return node_ != nullptr; }
  void Wait() const {
    if (node_) node_->ready.get();
  }
  int Slot() const { return node_ ? node_->slot : -1; }

 private:
  friend class QueuingServer;
  struct Node {
    std::promise<void> served_pr;
    std::shared_future<void> ready = served_pr.get_future().share();

    uint32_t holders = 0;   // 未释放的持有者数量；归零后节点可出队
    bool in_queue = false;  // 仍在服务队列中（未被弹出）
    bool served = false;    // 已唤醒（slot 已绑定）
    bool released = false;  // holders 已归零（等待出队）
    bool chain = false;     // 链式票据：与前驱同 run，继承其 slot
    int slot = -1;          // 绑定的资源 slot 下标
  };

  explicit QueuingTicket(std::shared_ptr<Node> node) : node_(std::move(node)) {}

  std::shared_ptr<Node> node_;
};

/**
 * @brief 排队服务：单个资源的 slot 池 + FIFO 等待队列
 *
 * "run" 指同一批次在单个资源上的连续票据序列：
 * - chain=false 票据开启新 run：服务时从空闲 slot 池取一个 slot；
 *   不同 run 使用不同 slot，可同时在服务中（数量受池深限制）→ 批间重叠
 * - chain=true 票据延续前驱 run（取票时位于队列尾部的那张票据）：
 *   前驱释放出队时 slot 转交给它，保证同批次各阶段操作同一份 buffer
 * - 唤醒顺序保持 FIFO：某票据可服务的前提是队列中位于它之前的票据均已被服务
 * - 释放（DeallingDone）按票据进行：共享票据全部持有者释放后才出队
 *
 * 池深为 1 时同一时刻仅一个 run 在服务，且 chain 链上 slot 唯一，
 * 语义与单 buffer 串行设计完全一致（未启用异步的平台不受影响）
 */
class QueuingServer {
 public:
  /**
   * @brief 取一张票据
   * @param reserve true 时，若上一张票据仍处于保留状态则与其共享（同一 run 内多持有者）
   * @param chain true 时新票据延续当前队尾票据的 run（继承其 slot）
   */
  QueuingTicket PickUpTicket(bool reserve = false, bool chain = false);

  /**
   * @brief 取一张新票据，终止当前保留状态（开启新的共享组）
   */
  QueuingTicket PickUpNewTicket(bool reserve = false, bool chain = false);

  /**
   * @brief 释放票据（共享票据每个持有者各调用一次）
   * 持有者归零后票据标记 released，由 ServeLocked 出队并转交/归还 slot
   */
  void DeallingDone(const QueuingTicket& ticket);

  void WaitByTicket(QueuingTicket* pticket) { pticket->Wait(); }

  // 设置 slot 池深度（资源 Init 时调用），池内初始 slot 为 0..n-1
  void SetPoolSize(size_t n);

 private:
  using Node = QueuingTicket::Node;
  void ServeLocked();  // 出队已释放的队首节点 + FIFO 唤醒可服务节点（须持锁调用）

  std::deque<std::shared_ptr<Node>> tickets_q_;
  std::shared_ptr<Node> reserved_node_;
  bool reserved_ = false;
  std::deque<int> free_slots_ {0};  // 空闲 slot 池
  std::mutex mtx_;
};

#endif  // MODULES_INFERENCE_SRC_QUEUING_SERVER_HPP_
