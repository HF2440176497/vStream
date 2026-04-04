#ifndef MODULES_INFERENCE_INFER_OPTIONS_HPP_
#define MODULES_INFERENCE_INFER_OPTIONS_HPP_

#include <functional>
#include <memory>
#include <string>

#include "preproc.hpp"
#include "postproc.hpp"
#include "obj_filter.hpp"

namespace cnstream {

class ModelLoader;

class InferOptions {
 public:
  InferOptions() = default;

  InferOptions& SetDeviceId(int device_id) {
    device_id_ = device_id;
    return *this;
  }

  InferOptions& SetModel(ModelLoader* model) {
    model_ = model;
    return *this;
  }

  InferOptions& SetPreprocessor(std::shared_ptr<Preproc> preprocessor) {
    preprocessor_ = preprocessor;
    return *this;
  }

  InferOptions& SetPostprocessor(std::shared_ptr<Postproc> postprocessor) {
    postprocessor_ = postprocessor;
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

  InferOptions& SetObjPreprocessor(std::shared_ptr<ObjPreproc> obj_preprocessor) {
    obj_preprocessor_ = obj_preprocessor;
    return *this;
  }

  InferOptions& SetObjPostprocessor(std::shared_ptr<ObjPostproc> obj_postprocessor) {
    obj_postprocessor_ = obj_postprocessor;
    return *this;
  }

  InferOptions& SetObjFilter(std::shared_ptr<ObjFilter> obj_filter) {
    obj_filter_ = obj_filter;
    return *this;
  }

  InferOptions& SetDumpResizedImageDir(const std::string& dump_dir) {
    dump_resized_image_dir_ = dump_dir;
    return *this;
  }
  InferOptions& SetSavingInferInput(bool saving_infer_input) {
    saving_infer_input_ = saving_infer_input;
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

  int device_id() const { return device_id_; }
  ModelLoader* model() const { return model_; }
  std::shared_ptr<Preproc> preprocessor() const { return preprocessor_; }
  std::shared_ptr<Postproc> postprocessor() const { return postprocessor_; }
  uint32_t batching_timeout() const { return batching_timeout_; }
  const std::function<void(const std::string& err_msg)>& error_handler() const { return error_func_; }
  bool batching_by_obj() const { return batching_by_obj_; }
  std::shared_ptr<ObjPreproc> obj_preprocessor() const { return obj_preprocessor_; }
  std::shared_ptr<ObjPostproc> obj_postprocessor() const { return obj_postprocessor_; }
  std::shared_ptr<ObjFilter> obj_filter() const { return obj_filter_; }
  const std::string& dump_resized_image_dir() const { return dump_resized_image_dir_; }
  bool saving_infer_input() const { return saving_infer_input_; }
  const std::string& module_name() const { return module_name_; }
  ModuleProfiler* profiler() const { return profiler_; }

 private:
  int device_id_ = 0;
  ModelLoader* model_ = nullptr;
  std::shared_ptr<Preproc> preprocessor_ = nullptr;
  std::shared_ptr<Postproc> postprocessor_ = nullptr;
  uint32_t batching_timeout_ = 3000;
  std::function<void(const std::string& err_msg)> error_func_;
  bool batching_by_obj_ = false;
  std::shared_ptr<ObjPreproc> obj_preprocessor_ = nullptr;
  std::shared_ptr<ObjPostproc> obj_postprocessor_ = nullptr;
  std::shared_ptr<ObjFilter> obj_filter_ = nullptr;
  std::string dump_resized_image_dir_;
  bool saving_infer_input_ = false;
  std::string module_name_;
  ModuleProfiler* profiler_ = nullptr;
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_INFER_OPTIONS_HPP_
