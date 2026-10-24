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
#include "input_deriver.hpp"
#include "obj_filter.hpp"
#include "frame_filter.hpp"
#include "postproc.hpp"
#include "preproc.hpp"

#include "cnstream_frame_va.hpp"
#include "inference.hpp"
#include "infer_params.hpp"

#include "profiler/module_profiler.hpp"

namespace cnstream {

// 调度器 context
//
// 成员声明顺序即决定析构顺序（逆序析构）
//
// 关闭策略 = 「尽早退出、丢弃剩余任务」，因此让回传线程先停止
//
struct InferContext {
  std::shared_ptr<InferEngine> engine;                      // 后析构
  uint32_t drop_count = 0;
  std::shared_ptr<InferTransDataHelper> trans_data_helper;  // 先析构
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
  std::shared_ptr<FrameFilter> frame_filter_ = nullptr;
  std::shared_ptr<InputDeriver> input_deriver_ = nullptr;

  uint32_t trans_data_size_ = 20;
  std::string dump_resized_image_dir_ = "";
  std::string module_name_ = "";

  // 预创建池 + tid 绑定：
  // - InitInferContextPool(depth) 先建 depth 个 ctx 放入 free_pool_
  // - 每个 TaskLoop 线程首次 Process 时按 tid 绑定一个 ctx，之后复用
  // - 预创建用尽后，为新 tid 按需补建；注意这会发生在启动锁之外，日志会告警
  std::vector<InferContextSptr> free_pool_;
  std::map<std::thread::id, InferContextSptr> tid_ctx_;
  std::mutex pool_mtx_;
  uint32_t precreated_depth_ = 0;

  void InferEngineErrorHandleFunc(const std::string& err_msg) {
    LOGE(INFER) << err_msg;
  }

  /**
   * @brief 本模块在 module mask 中的 bit（1 << module->GetId()）
   *        模块未入 pipeline（GetId 无效）时返回 0，MarkInferSkipped 对 0 不做修改。
   */
  uint64_t ModuleBitMask() const {
    size_t id = q_ptr_->GetId();
    return id < 64 ? (static_cast<uint64_t>(1) << id) : 0;
  }

  /**
   * @brief 组装模型文件路径
   * @details 路径格式为 config_dir_path[/version]/model_name.engine
   *          - 若 model_name 为绝对路径，则直接使用；
   *          - 若 config_dir_path 为相对路径，则相对于 JSON 文件目录解析；
   *          - 若未指定 version，则直接在 config_dir_path 下查找；
   *          - 自动为 model_name 追加 .engine 后缀。
   */
  std::string ComposeModelPath(const std::string& dir_path,
                               const std::string& version,
                               const std::string& model_name) const {
    if (model_name.empty()) {
      return "";
    }
    if (model_name[0] == '/') {
      return model_name;
    }
    std::string base_dir = dir_path;
    if (base_dir.empty()) {
      base_dir = "./";
    }
    // 保证目录以 '/' 结尾
    if (!base_dir.empty() && base_dir.back() != '/') {
      base_dir += "/";
    }
    // 追加 version 子目录
    if (!version.empty()) {
      base_dir += version;
      if (!base_dir.empty() && base_dir.back() != '/') {
        base_dir += "/";
      }
    }
    std::string model_file = model_name;
    const std::string kEngineExt = ".engine";
    if (model_file.size() < kEngineExt.size() ||
        model_file.compare(model_file.size() - kEngineExt.size(), kEngineExt.size(), kEngineExt) != 0) {
      model_file += kEngineExt;
    }
    return base_dir + model_file;
  }

  /**
   * @brief 解析来自 params 的模型参数
   * param_set 只是用于取出路径参数，需要的参数都已在 params 中
   */
  bool InitByParams(const InferParams &params, const ModuleParamSet &param_set) {
    params_ = params;
    module_name_ = q_ptr_->GetName();

    // 根据 config_dir_path、version 和 model_name 组装模型路径
    std::string model_path = ComposeModelPath(params.config_dir_path, params.version, params.model_name);

    LOGI(INFER) << "[" << module_name_ << "] load model [path: " << model_path << "]";

    auto device_type = params.device_type;
    auto device_id = params.device_id;
    auto& factory = ModelLoaderFactory::Instance();

    model_loader_ = factory.CreateModelLoader(device_type, device_id);
    if (!model_loader_) {
      LOGE(INFER) << "[" << module_name_ << "] create model loader failed. device_type: "
                 << DevType2Str(device_type) << ", device_id: " << device_id;
      return false;
    }

    if (!model_loader_->Init(model_path, params)) {
      LOGE(INFER) << "[" << module_name_ << "] init model failed. path: " << model_path;
      return false;
    }

    if (params.trans_data_size > 0) {
      trans_data_size_ = params.trans_data_size;
    } else {
      LOGW(INFER) << "[" << module_name_ << "] trans_data_size is 0. use default:" << trans_data_size_;
    }

    // 帧级过滤：仅用于帧级推理路径（object_infer=false）
    if (!params.frame_filter_name.empty()) {
      if (params.object_infer) {
        LOGE(INFER) << "[" << module_name_ << "] frame_filter_name is mutually exclusive with "
                    << "object_infer(true)/obj_filter_name. Use obj_filter for object-level filtering.";
        return false;
      }
      frame_filter_ = std::shared_ptr<FrameFilter>(FrameFilter::Create(params.frame_filter_name));
      if (!frame_filter_) {
        LOGE(INFER) << "[" << module_name_ << "] Can not find FrameFilter implemention by name: "
                    << params.frame_filter_name;
        return false;
      }
      if (!frame_filter_->Init(params.custom_frame_filter_params)) {
        LOGE(INFER) << "[" << module_name_ << "] frame_filter_ init failed.";
        return false;
      }
      LOGI(INFER) << "[" << module_name_ << "] Frame filter set:" << params.frame_filter_name;
    }

    if (params.object_infer) {
      LOGI(INFER) << "[" << module_name_ << "] inference mode: inference with objects.";
      if (!params.obj_filter_name.empty()) {
        obj_filter_ = std::shared_ptr<ObjFilter>(ObjFilter::Create(params.obj_filter_name));
        if (obj_filter_) {
          LOGI(INFER) << "[" << module_name_ << "] Object filter set:" << params.obj_filter_name;
        } else {
          LOGE(INFER) << "Can not find ObjFilter implemention by name: "
                     << params.obj_filter_name;
          return false;
        }
        if (!obj_filter_->Init(params.custom_obj_filter_params)) {
          LOGE(INFER) << "obj_filter_ init failed.";
          return false;
        }
      }
    }

    // 前处理
    if (!params.preproc_name.empty()) {
      if (params.object_infer) {
        obj_preproc_ = std::shared_ptr<ObjPreproc>(ObjPreproc::Create(params.preproc_name));
        if (!obj_preproc_) {
          LOGE(INFER) << "Can not find ObjPreproc implemention by name: " << params.preproc_name;
          return false;
        }
        if (!obj_preproc_->Init(params.custom_preproc_params)) {
          LOGE(INFER) << "Preprocessor init failed.";
          return false;
        }
      } else {
        preproc_ = std::shared_ptr<Preproc>(Preproc::Create(params.preproc_name));
        if (!preproc_) {
          LOGE(INFER) << "Can not find Preproc implemention by name: " << params.preproc_name;
          return false;
        }
        if (!preproc_->Init(params.custom_preproc_params)) {
          LOGE(INFER) << "Preprocessor init failed.";
          return false;
        }
      }
    }

    // 后处理
    if (!params.postproc_name.empty()) {
      if (params.object_infer) {
        obj_postproc_ = std::shared_ptr<ObjPostproc>(ObjPostproc::Create(params.postproc_name));
        if (!obj_postproc_) {
          LOGE(INFER) << "Can not find ObjPostproc implemention by name: " << params.postproc_name;
          return false;
        }
        if (!obj_postproc_->Init(params.custom_postproc_params)) {
          LOGE(INFER) << "Postprocessor init failed.";
          return false;
        }
      } else {
        postproc_ = std::shared_ptr<Postproc>(Postproc::Create(params.postproc_name));
        if (!postproc_) {
          LOGE(INFER) << "Can not find Postproc implemention by name: " << params.postproc_name;
          return false;
        }
        if (!postproc_->Init(params.custom_postproc_params)) {
          LOGE(INFER) << "Postprocessor init failed.";
          return false;
        }
      }
    }

    // 输入派生（业务级模块输入定制，如旋转）
    if (!params.input_derive_name.empty()) {
      if (params.object_infer) {
        LOGE(INFER) << "[" << module_name_ << "] input_derive_name is only supported "
                    << "when object_infer is false.";
        return false;
      }
      input_deriver_ = std::shared_ptr<InputDeriver>(InputDeriver::Create(params.input_derive_name));
      if (!input_deriver_) {
        LOGE(INFER) << "Can not find InputDeriver implemention by name: " << params.input_derive_name;
        return false;
      }
      if (!input_deriver_->Init(params.custom_input_derive_params)) {
        LOGE(INFER) << "[" << module_name_ << "] input_deriver_ init failed.";
        return false;
      }
      LOGI(INFER) << "[" << module_name_ << "] Input deriver set: " << params.input_derive_name;
    }

    if (!params.dump_resized_image_dir.empty()) {
      dump_resized_image_dir_ = GetPathRelativeToTheJSONFile(params.dump_resized_image_dir, param_set);
    }

    return true;
  }

  // 创建一个 InferContext。InferEngine 的构造会：
  // - 创建 TRT execution context（InferBatchingDoneStage 构造时 AcquireExecutionContext）
  // - 分配 NetInput/NetOutput 设备缓冲（IOResource::Init -> Allocate）
  // - 创建推理线程池
  InferContextSptr CreateInferContext() {
    auto ctx = std::make_shared<InferContext>();

    InferOptions infer_options;
    infer_options.SetDeviceId(params_.device_id)
        .SetModel(model_loader_.get())
        .SetPreprocessor(preproc_)
        .SetPostprocessor(postproc_)
        .SetBatchingTimeout(params_.batching_timeout)
        .SetErrorHandler(std::bind(&InferencePrivate::InferEngineErrorHandleFunc, this, std::placeholders::_1))
        .SetPostprocOnDevice(params_.postproc_on_device)
        .SetBatchingByObj(params_.object_infer)
        .SetObjPreprocessor(obj_preproc_)
        .SetObjPostprocessor(obj_postproc_)
        .SetObjFilter(obj_filter_)
        .SetFrameFilter(frame_filter_)
        .SetModuleBitMask(ModuleBitMask())
        .SetInputDeriver(input_deriver_)
        .SetDumpResizedImageDir(dump_resized_image_dir_)
        .SetSavingInferInput(params_.saving_infer_input)
        .SetModuleName(module_name_)
        .SetProfiler(q_ptr_->GetProfiler());

    ctx->engine = std::make_shared<InferEngine>(infer_options);
    ctx->trans_data_helper = std::make_shared<InferTransDataHelper>(
        q_ptr_, params_.infer_interval * trans_data_size_);
    return ctx;
  }

  bool InitInferContextPool(uint32_t depth) {
    std::lock_guard<std::mutex> lk(pool_mtx_);
    free_pool_.clear();
    tid_ctx_.clear();
    precreated_depth_ = 0;
    for (uint32_t i = 0; i < depth; ++i) {
      auto ctx = CreateInferContext();
      if (!ctx || !ctx->engine || !ctx->engine->IsReady()) {
        LOGE(INFER) << "[" << module_name_ << "] pre-create InferContext[" << i
                    << "] failed (engine not ready)";
        free_pool_.clear();
        return false;
      }
      free_pool_.push_back(ctx);
    }
    precreated_depth_ = depth;
    LOGI(INFER) << "[" << module_name_ << "] pre-created " << free_pool_.size()
                << " InferContext(s)";
    return true;
  }

  // 按当前线程 tid 绑定 InferContext；预创建池用尽后按需补建。
  InferContextSptr AcquireInferContext() {
    const auto tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lk(pool_mtx_);
    auto it = tid_ctx_.find(tid);
    if (it != tid_ctx_.end()) {
      return it->second;
    }

    InferContextSptr ctx;
    if (!free_pool_.empty()) {
      ctx = free_pool_.back();
      free_pool_.pop_back();
    } else {
      // 预创建 depth 已用尽：为新的 TaskLoop 线程补建
      ctx = CreateInferContext();
      if (!ctx || !ctx->engine || !ctx->engine->IsReady()) {
        LOGE(INFER) << "[" << module_name_
                    << "] Lazy-create InferContext failed for a new thread";
        return nullptr;
      }
      LOGW(INFER) << "[" << module_name_
                  << "] Lazy-created InferContext beyond pre-created depth: "
                  << precreated_depth_;
    }
    tid_ctx_.emplace(tid, ctx);
    return ctx;
  }

  void ClearInferContextPool() {
    free_pool_.clear();
    tid_ctx_.clear();
    precreated_depth_ = 0;
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
  LOGF_IF(INFER, !param_manager_) << "Inference::Inference new InferParams failed.";
  param_manager_->RegisterAll(&param_register_);
}

Inference::~Inference() {
  // 必须显式关闭：
  // InferencePrivate 持有各 pipeline 线程的 InferContext（线程池线程 + 回传线程）
  Close();
  param_manager_.reset();
}

bool Inference::Open(ModuleParamSet raw_params) {
  if (d_ptr_) {
    Close();
  }
  d_ptr_ = new (std::nothrow) InferencePrivate(this);
  if (!d_ptr_) {
    LOGE(INFER) << "Inference::Open() new InferencePrivate failed";
    return false;
  }

  InferParams params;
  // 调用 parser 将解析通过的参数赋值给 params
  if (!param_manager_->ParseBy(raw_params, &params)) {
    LOGE(INFER) << "[" << GetName() << "] parse parameters failed.";
    return false;
  }

  // 在 InferencePrivate 中初始化 engine
  if (!d_ptr_->InitByParams(params, raw_params)) {
    LOGE(INFER) << "[" << GetName() << "] init resources failed.";
    return false;
  }

  if (container_ == nullptr) {
    LOGI(INFER) << name_ << " has not been added into pipeline.";
  } else {
    if (GetProfiler()) {
      if (!params.preproc_name.empty()) {
        GetProfiler()->RegisterProcess(kPREPROC_PROFILER_NAME);
      }
      if (!params.postproc_name.empty()) {
        GetProfiler()->RegisterProcess(kPOSTPROC_PROFILER_NAME);
      }
      GetProfiler()->RegisterProcess(kINFERENCE_PROFILER_NAME);
      GetProfiler()->RegisterProcess(kMODEL_PROFILER_NAME);

    }
  }

  if (!d_ptr_->InitInferContextPool(params.infer_context_pool_depth)) {
    LOGE(INFER) << "[" << GetName() << "] pre-create InferContext pool failed.";
    return false;
  }

  return true;
}

/**
 * 在此清除 InferContextSptr 列表
 */
void Inference::Close() {
  if (nullptr == d_ptr_) return;

  /*destroy infer contexts*/
  {
    std::lock_guard<std::mutex> lk(d_ptr_->pool_mtx_);
    d_ptr_->ClearInferContextPool();
  }

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
  if (d_ptr_ == nullptr) {
    LOGE(INFER) << "Inference is not open";
    return -1;
  }

  // 按 tid 绑定：同一 TaskLoop 线程复用同一个 InferContext；
  // 预创建池用尽后，新线程会按需补建（日志会告警）。
  std::shared_ptr<InferContext> pctx = d_ptr_->AcquireInferContext();
  if (!pctx || !pctx->engine) {
    LOGE(INFER) << "[" << d_ptr_->module_name_ << "] InferContext not ready";
    return -1;
  }
  bool eos = data->IsEos();
  bool drop_data = d_ptr_->params_.infer_interval > 0 && pctx->drop_count++ % d_ptr_->params_.infer_interval != 0;

  if (!eos) {
    if (data->IsRemoved()) {
      if (GetProfiler()) {
        GetProfiler()->RecordProcessDropped(kINFERENCE_PROFILER_NAME,
            std::make_pair(data->stream_id, data->timestamp));
      }
      return 0;
    }
    if (GetProfiler()) {
      GetProfiler()->RecordProcessStart(kINFERENCE_PROFILER_NAME,
          std::make_pair(data->stream_id, data->timestamp));
    }
  }

  if (eos || drop_data) {
    if (eos && IsStreamRemoved(data->stream_id)) {
      // minimize batch_timeout delay
      pctx->engine->ForceBatchingDone();
    }
    // drop_count 重新从 1 计数
    if (drop_data) {
      pctx->drop_count %= d_ptr_->params_.infer_interval;
      data->MarkInferSkipped(d_ptr_->ModuleBitMask());
    }
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
