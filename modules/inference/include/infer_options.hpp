#ifndef MODULES_INFERENCE_INFER_OPTIONS_HPP_
#define MODULES_INFERENCE_INFER_OPTIONS_HPP_

#include <functional>
#include <memory>
#include <string>

#include "model_loader.hpp"
#include "postproc.hpp"
#include "preproc.hpp"

namespace cnstream {

class InferOptions {
 public:
  InferOptions() = default;

  InferOptions& SetDeviceId(int dev_id) {
    dev_id_ = dev_id;
    return *this;
  }

  InferOptions& SetModel(ModelLoader* model) {
    model_ = model;
    return *this;
  }

  InferOptions& SetPreprocessor(PreprocPtr preprocessor) {
    preprocessor_ = preprocessor;
    return *this;
  }

  InferOptions& SetPostprocessor(PostprocPtr postprocessor) {
    postprocessor_ = postprocessor;
    return *this;
  }

  InferOptions& SetBatchSize(uint32_t batchsize) {
    batchsize_ = batchsize;
    return *this;
  }

  InferOptions& SetBatchingTimeout(uint32_t timeout) {
    batching_timeout_ = timeout;
    return *this;
  }

  InferOptions& SetErrorHandler(std::function<void(const std::string& err_msg)> error_func) {
    error_func_ = error_func;
    return *this;
  }

  InferOptions& SetBatchingByObj(bool flag) {
    batching_by_obj_ = flag;
    return *this;
  }

  InferOptions& SetObjPreprocessor(ObjPreprocPtr obj_preprocessor) {
    obj_preprocessor_ = obj_preprocessor;
    return *this;
  }

  InferOptions& SetObjPostprocessor(ObjPostprocPtr obj_postprocessor) {
    obj_postprocessor_ = obj_postprocessor;
    return *this;
  }

  InferOptions& SetModuleName(const std::string& name) {
    module_name_ = name;
    return *this;
  }

  InferOptions& SetProfiler(ModuleProfiler* profiler) {
    profiler_ = profiler;
    return *this;
  }

  int dev_id() const { return dev_id_; }
  ModelLoader* model() const { return model_; }
  PreprocPtr preprocessor() const { return preprocessor_; }
  PostprocPtr postprocessor() const { return postprocessor_; }
  uint32_t batchsize() const { return batchsize_; }
  uint32_t batching_timeout() const { return batching_timeout_; }
  const std::function<void(const std::string& err_msg)>& error_handler() const { return error_func_; }
  bool batching_by_obj() const { return batching_by_obj_; }
  ObjPreprocPtr obj_preprocessor() const { return obj_preprocessor_; }
  ObjPostprocPtr obj_postprocessor() const { return obj_postprocessor_; }
  const std::string& module_name() const { return module_name_; }
  ModuleProfiler* profiler() const { return profiler_; }

 private:
  int dev_id_ = 0;
  ModelLoader* model_ = nullptr;
  PreprocPtr preprocessor_ = nullptr;
  PostprocPtr postprocessor_ = nullptr;
  uint32_t batchsize_ = 1;
  uint32_t batching_timeout_ = 2000;
  std::function<void(const std::string& err_msg)> error_func_;
  bool batching_by_obj_ = false;
  ObjPreprocPtr obj_preprocessor_ = nullptr;
  ObjPostprocPtr obj_postprocessor_ = nullptr;
  std::string module_name_;
  ModuleProfiler* profiler_ = nullptr;
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_INFER_OPTIONS_HPP_
