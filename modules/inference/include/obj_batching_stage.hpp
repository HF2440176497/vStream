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

#ifndef MODULES_INFERENCE_SRC_OBJ_BATCHING_STAGE_HPP_
#define MODULES_INFERENCE_SRC_OBJ_BATCHING_STAGE_HPP_

#include <memory>


namespace cnstream {

class ModelLoader;
class FrameInfo;
class InferObject;
class InferTask;
struct IOResValue;
class IOResource;
class NetInputResource;
class CpuInputResource;
class ObjPreproc;

class ObjBatchingStage {
 public:
  ObjBatchingStage(ModelLoader* model, uint32_t batchsize)
      : model_(model), batchsize_(batchsize) {}
  virtual ~ObjBatchingStage() {}
  virtual std::shared_ptr<InferTask> Batching(std::shared_ptr<FrameInfo> finfo,
                                              std::shared_ptr<InferObject> obj) = 0;
  virtual void Reset() {}

 protected:
  ModelLoader* model_;
  uint32_t batchsize_ = 0;
};  // class ObjBatchingStage

class IOObjBatchingStage : public ObjBatchingStage {
 public:
  IOObjBatchingStage(ModelLoader* model, uint32_t batchsize,
                     std::shared_ptr<IOResource> output_res)
      : ObjBatchingStage(model, batchsize), output_res_(output_res) {}
  virtual ~IOObjBatchingStage() {}
  std::shared_ptr<InferTask> Batching(std::shared_ptr<FrameInfo> finfo, std::shared_ptr<InferObject> obj) override;
  void Reset() override { batch_idx_ = 0; }

 protected:
  virtual void ProcessOneObject(std::shared_ptr<FrameInfo> finfo, std::shared_ptr<InferObject> obj,
                                uint32_t batch_idx, const IOResValue& value) = 0;

 private:
  using ObjBatchingStage::batchsize_;
  uint32_t batch_idx_ = 0;
  std::shared_ptr<IOResource> output_res_ = nullptr;
};  // class IOObjBatchingStage

class CpuPreprocessingObjBatchingStage : public IOObjBatchingStage {
 public:
  CpuPreprocessingObjBatchingStage(ModelLoader* model, uint32_t batchsize,
                                   std::shared_ptr<ObjPreproc> preprocessor,
                                   std::shared_ptr<CpuInputResource> cpu_input_res);
  ~CpuPreprocessingObjBatchingStage();

 private:
  void ProcessOneObject(std::shared_ptr<FrameInfo> finfo, std::shared_ptr<InferObject> obj, uint32_t batch_idx,
                        const IOResValue& value) override;
  std::shared_ptr<ObjPreproc> preprocessor_ = nullptr;
};  // class CpuPreprocessingObjBatchingStage

}  // namespace cnstream

#endif  // MODULES_INFERENCE_SRC_OBJ_BATCHING_STAGE_HPP_
