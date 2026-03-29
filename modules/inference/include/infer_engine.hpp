/*************************************************************************
 * Copyright (C) [2024] by TensorRT Adapter. All rights reserved
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

#ifndef MODULES_INFERENCE_INFER_ENGINE_HPP_
#define MODULES_INFERENCE_INFER_ENGINE_HPP_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "batching_done_stage.hpp"
#include "batching_stage.hpp"
#include "infer_options.hpp"
#include "infer_params.hpp"
#include "infer_resource.hpp"
#include "infer_thread_pool.hpp"
#include "obj_batching_stage.hpp"
#include "postproc.hpp"
#include "preproc.hpp"
#include "model_loader.hpp"

namespace cnstream {

class InferEngine {
 public:
  class ResultWaitingCard {
   public:
    explicit ResultWaitingCard(std::shared_ptr<std::promise<void>> ret_promise) : promise_(ret_promise) {}
    void WaitForCall() {
      if (promise_) {
        promise_->get_future().share().get();
      }
    }

   private:
    std::shared_ptr<std::promise<void>> promise_;
  };

  explicit InferEngine(const InferOptions& options);

  ~InferEngine();

  ResultWaitingCard FeedData(std::shared_ptr<FrameInfo> frame_info);
  void ForceBatchingDone() { BatchingDone(); }

 private:
  void StageAssemble();
  void BatchingDone();

  ModelLoader* model_;  // InferencePrivate we own this model_loader_
  PreprocPtr preprocessor_;
  PostprocPtr postprocessor_;
  ObjPreprocPtr obj_preprocessor_;
  ObjPostprocPtr obj_postprocessor_;

  uint32_t batchsize_ = 0;
  uint32_t batching_timeout_ = 0;
  int dev_id_ = 0;
  bool batching_by_obj_ = false;  // infer_param: obj_infer_
  std::string module_name_;

  BatchingStagePtr batching_stage_ = nullptr;
  ObjBatchingStagePtr obj_batching_stage_ = nullptr;
  std::vector<BatchingDoneStagePtr> batching_done_stages_;
  std::shared_ptr<ObjPostprocessingBatchingDoneStage> obj_postproc_stage_ = nullptr;

  CpuInputResourcePtr cpu_input_res_;
  CpuOutputResourcePtr cpu_output_res_;
  NetInputResourcePtr net_input_res_;
  NetOutputResourcePtr net_output_res_;

  InferThreadPoolPtr thread_pool_;
  std::function<void(const std::string& err_msg)> error_func_;

  BatchingDoneInput batched_finfos_;
  std::vector<void*> batched_objs_;
  uint32_t cached_frame_cnt_ = 0;  // ++ on FeedData

  std::condition_variable cv_;
  std::thread timeout_thread_;
  std::atomic<bool> running_{false};
};

using InferEnginePtr = std::shared_ptr<InferEngine>;

}  // namespace cnstream

#endif  // MODULES_INFERENCE_TRT_SRC_INFER_ENGINE_HPP_
