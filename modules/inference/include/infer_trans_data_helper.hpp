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

#ifndef MODULES_INFERENCE_SRC_INFER_TRANS_DATA_HELPER_HPP_
#define MODULES_INFERENCE_SRC_INFER_TRANS_DATA_HELPER_HPP_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>

#include "infer_engine.hpp"

namespace cnstream {

class Inference;
class FrameInfo;

class InferTransDataHelper {
 public:
  explicit InferTransDataHelper(Inference* infer, int batchsize);
  ~InferTransDataHelper();

  void SubmitData(const std::pair<std::shared_ptr<FrameInfo>, InferEngine::ResultWaitingCard>& data);

 private:
  void Loop();
  /**
   * @brief 等待帧结果，但能被关闭打断
   * @note 直接 WaitForCall() 会无限阻塞：析构时的 join 依赖 promise 被置位，
   *       而未满批次的 promise 只由 InferEngine 持有。这里改为可中断的限时等待，
   *       保证 Loop 线程在析构时必定退出。
   * @return true 结果已就绪（可以继续下发）；false 关闭中，调用方应丢弃该帧
   */
  bool WaitResultWithExit(const InferEngine::ResultWaitingCard& card);
  std::mutex mtx_;
  std::condition_variable cond_not_full_;
  std::condition_variable cond_not_empty_;
  std::queue<std::pair<std::shared_ptr<FrameInfo>, InferEngine::ResultWaitingCard>> queue_;
  std::thread th_;
  std::atomic<bool> running_;
  Inference* infer_ = nullptr;
  int batchsize_ = 1;
};  // class InferTransDataHelper

}  // namespace cnstream

#endif  // MODULES_INFERENCE_SRC_INFER_TRANS_DATA_HELPER_HPP_
