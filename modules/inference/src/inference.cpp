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




#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>

#include "model_loader.hpp"
#include "infer_engine.hpp"
#include "infer_trans_data_helper.hpp"
#include "obj_filter.hpp"
#include "postproc.hpp"
#include "preproc.hpp"

#include "cnstream_frame_va.hpp"
#include "inferencer.hpp"
#include "infer_params.hpp"

#include "profiler/module_profiler.hpp"

namespace cnstream {

// 调度器 context
struct InferContext {
  std::shared_ptr<InferEngine> engine;
  std::shared_ptr<InferTransDataHelper> trans_data_helper;
  uint32_t drop_count = 0;
};  // struct InferContext

using InferContextSptr = std::shared_ptr<InferContext>;

class InferencePrivate: public NonCopyable {
 public:
  explicit InferencePrivate(Inference* q) : q_ptr_(q) {}
  InferParams params_{};
  std::unique_ptr<ModelLoader> model_loader_ = nullptr;  
  std::shared_ptr<Preproc> preproc_ = nullptr;
  std::shared_ptr<Postproc> postproc_ = nullptr;

  std::shared_ptr<ObjPreproc> obj_preproc_ = nullptr;
  std::shared_ptr<ObjPostproc> obj_postproc_ = nullptr;
  std::shared_ptr<ObjFilter> obj_filter_ = nullptr;

  uint32_t trans_data_size_ = 20;
  std::string dump_resized_image_dir_ = "";
  std::string module_name_ = "";

  std::map<std::thread::id, InferContextSptr> ctxs_ { };
  std::mutex ctx_mtx_;

  void InferEngineErrorHnadleFunc(const std::string& err_msg) {
    LOGE(INFERENCER) << err_msg;
  }

  /**
   * @brief 解析来自 params 的模型参数
   * param_set 只是用于取出路径参数，需要的参数都已在 params 中
   */
  bool InitByParams(const InferParams &params, const ModuleParamSet &param_set) {
    params_ = params;
    module_name_ = q_ptr_->GetName();
    std::string model_path = GetPathRelativeToTheJSONFile(params.model_path, param_set);

    LOGI(INFERENCER) << "[" << module_name_ << "] load model [path: " << model_path << "]";

    // TODO: 未来由 Pipeline 参数透传到此，以此为准来检验 data 中是否相同
    auto dev_type = params.device_type;
    auto dev_id = params.device_id;
    auto& factory = ModelLoaderFactory::Instance();

    if (dev_type != DevType::CPU && dev_id == -1) {
      LOGE(INFERENCER) << "[" << module_name_ << "] dev_type [" << dev_type << "] not CPU. but device_id is -1";
      return false;
    }

    // LoadEngine - ParBinding
    model_loader_ = factory.CreateModelLoader(dev_type, dev_id);
    if (!model_loader_) {
      LOGE(INFERENCER) << "[" << module_name_ << "] create model loader failed. dev_type: "
                 << dev_type << ", dev_id: " << dev_id;
      return false;
    }

    if (!model_loader_->Init(model_path)) {
      LOGE(INFERENCER) << "[" << module_name_ << "] init model failed. path: " << model_path;
      return false;
    }
    trans_data_size_ = params.trans_data_size;

    if (params.object_infer) {
      LOGI(INFERENCER) << "[" << module_name_ << "] inference mode: inference with objects.";
      if (!params.obj_filter_name.empty()) {
        obj_filter_ = std::shared_ptr<ObjFilter>(ObjFilter::Create(params.obj_filter_name));
        if (obj_filter_) {
          LOGI(INFERENCER) << "[" << module_name_ << "] Object filter set:" << params.obj_filter_name;
        } else {
          LOGE(INFERENCER) << "Can not find ObjFilter implemention by name: "
                     << params.obj_filter_name;
          return false;
        }
      }
    }

    // 前处理
    if (!params.preproc_name.empty()) {
      if (params.object_infer) {
        obj_preproc_ = std::shared_ptr<ObjPreproc>(ObjPreproc::Create(params.preproc_name));
        if (!obj_preproc_) {
          LOGE(INFERENCER) << "Can not find ObjPreproc implemention by name: " << params.preproc_name;
          return false;
        }
        if (!obj_preproc_->Init(params.custom_preproc_params)) {
          LOGE(INFERENCER) << "Preprocessor init failed.";
          return false;
        }
      } else {
        preproc_ = std::shared_ptr<Preproc>(Preproc::Create(params.preproc_name));
        if (!preproc_) {
          LOGE(INFERENCER) << "Can not find Preproc implemention by name: " << params.preproc_name;
          return false;
        }
        if (!preproc_->Init(params.custom_preproc_params)) {
          LOGE(INFERENCER) << "Preprocessor init failed.";
          return false;
        }
      }
    }

    // 后处理
    if (!params.postproc_name.empty()) {
      if (params.object_infer) {
        obj_postproc_ = std::shared_ptr<ObjPostproc>(ObjPostproc::Create(params.postproc_name));
        if (!obj_postproc_) {
          LOGE(INFERENCER) << "Can not find ObjPostproc implemention by name: " << params.postproc_name;
          return false;
        }
        if (!obj_postproc_->Init(params.custom_postproc_params)) {
          LOGE(INFERENCER) << "Postprocessor init failed.";
          return false;
        }
        obj_postproc_->SetThreshold(params.threshold);
      } else {
        postproc_ = std::shared_ptr<Postproc>(Postproc::Create(params.postproc_name));
        if (!postproc_) {
          LOGE(INFERENCER) << "Can not find Postproc implemention by name: " << params.postproc_name;
          return false;
        }
        if (!postproc_->Init(params.custom_postproc_params)) {
          LOGE(INFERENCER) << "Postprocessor init failed.";
          return false;
        }
        postproc_->SetThreshold(params.threshold);
      }
    }

    if (!params.dump_resized_image_dir.empty()) {
      dump_resized_image_dir_ = GetPathRelativeToTheJSONFile(params.dump_resized_image_dir, param_set);
    }

    return true;
  }

  // Inference::Process 调用：当前线程获取或创建InferContext
  InferContextSptr GetInferContext() {
    std::thread::id tid = std::this_thread::get_id();
    InferContextSptr ctx(nullptr);
    std::lock_guard<std::mutex> lk(ctx_mtx_);
    if (ctxs_.find(tid) != ctxs_.end()) {
      ctx = ctxs_[tid];
    } else {
      ctx = std::make_shared<InferContext>();
      std::stringstream ss;
      ss << tid;
      std::string thread_id_str = ss.str();
      thread_id_str.erase(0, thread_id_str.length() - 9);
      std::string tid_str = "th_" + thread_id_str;

      InferOptions infer_options;
      infer_options.SetDeviceId(params_.device_id)
          .SetModel(model_loader_.get())
          .SetPreprocessor(preproc_)
          .SetPostprocessor(postproc_)
          .SetBatchingTimeout(params_.batching_timeout)
          .SetErrorHandler(std::bind(&InferencePrivate::InferEngineErrorHnadleFunc, this, std::placeholders::_1))
          .SetBatchingByObj(params_.object_infer)
          .SetObjPreprocessor(obj_preproc_)
          .SetObjPostprocessor(obj_postproc_)
          .SetObjFilter(obj_filter_)
          .SetDumpResizedImageDir(dump_resized_image_dir_)
          .SetSavingInferInput(params_.saving_infer_input)
          .SetModuleName(module_name_)
          .SetProfiler(q_ptr_->GetProfiler());

      ctx->engine = std::make_shared<InferEngine>(infer_options);
      ctx->trans_data_helper = std::make_shared<InferTransDataHelper>(q_ptr_, params_.infer_interval * trans_data_size_);
      ctxs_[tid] = ctx;
    }
    return ctx;
  }

 private:
  DECLARE_PUBLIC(q_ptr_, Inference);
};  // class InferencePrivate

Inference::Inference(const std::string& name) : Module(name) {
  d_ptr_ = nullptr;
  hasTransmit_.store(true);  // transmit data by module itself
  param_register_.SetModuleDesc(
      "Inference is a module for running offline model inference,"
      " as well as preprocessing and postprocessing.");
  
  param_manager_ = std::make_shared<InferParamManager>();
  LOGF_IF(INFERENCER, !param_manager_) << "Inference::Inference new InferParams failed.";
  param_manager_->RegisterAll(&param_register_);
}

Inference::~Inference() {
  param_manager_.reset();
}

bool Inference::Open(ModuleParamSet raw_params) {
  if (d_ptr_) {
    Close();
  }
  d_ptr_ = new (std::nothrow) InferencePrivate(this);
  if (!d_ptr_) {
    LOGE(INFERENCER) << "Inference::Open() new InferencePrivate failed";
    return false;
  }

  InferParams params;
  // 调用 parser 将解析通过的参数赋值给 params
  if (!param_manager_->ParseBy(raw_params, &params)) {
    LOGE(INFERENCER) << "[" << GetName() << "] parse parameters failed.";
    return false;
  }

  // 在 InferencePrivate 中初始化 engine
  if (!d_ptr_->InitByParams(params, raw_params)) {
    LOGE(INFERENCER) << "[" << GetName() << "] init resources failed.";
    return false;
  }

  if (container_ == nullptr) {
    LOGI(INFERENCER) << name_ << " has not been added into pipeline.";
  } else {
    if (GetProfiler()) {
      if (!params.preproc_name.empty()) {
        GetProfiler()->RegisterProcessName("RUN PREPROC");
      }
      GetProfiler()->RegisterProcessName("RUN MODEL");
    }
  }

  return true;
}

/**
 * 在此清除 InferContextSptr 列表
 */
void Inference::Close() {
  if (nullptr == d_ptr_) return;

  /*destroy infer contexts*/
  d_ptr_->ctx_mtx_.lock();
  d_ptr_->ctxs_.clear();  // std::map<std::thread::id, shared_ptr<InferContext>>
  d_ptr_->ctx_mtx_.unlock();

  delete d_ptr_;
  d_ptr_ = nullptr;
}

/**
 * @brief Process the input data.
 *
 * @param data The input data.
 *
 * @return 0: success
 */
int Inference::Process(std::shared_ptr<FrameInfo> data) {

  // 获取当前 thread 的 InferContext，也就是 Pipeline 启动的 TaskLoop 线程
  // ModelLoader 仍然由 InferencePrivate 所有
  std::shared_ptr<InferContext> pctx = d_ptr_->GetInferContext();
  bool eos = data->IsEos();
  bool drop_data = d_ptr_->params_.infer_interval > 0 && pctx->drop_count++ % d_ptr_->params_.infer_interval != 0;

  if (!eos) {
    if (data->IsRemoved()) {
      // discard packets from removed-stream
      return 0;
    }
  }

  if (eos || drop_data) {
    if (eos && IsStreamRemoved(data->stream_id)) {
      // minimize batch_timeout delay
      pctx->engine->ForceBatchingDone();
    }
    // drop_ 重新从 1 计数
    if (drop_data) pctx->drop_count %= d_ptr_->params_.infer_interval;
    std::shared_ptr<std::promise<void>> promise = std::make_shared<std::promise<void>>();
    promise->set_value();
    InferEngine::ResultWaitingCard card(promise);
    pctx->trans_data_helper->SubmitData(std::make_pair(data, card));
  } else {
    InferEngine::ResultWaitingCard card = pctx->engine->FeedData(data);
    pctx->trans_data_helper->SubmitData(std::make_pair(data, card));
  }
  return 0;
}

// note: 暂时没有调用
bool Inference::CheckParamSet(const ModuleParamSet &param_set) const {
  InferParams params;
  return param_manager_->ParseBy(param_set, &params);
}

}  // namespace cnstream
