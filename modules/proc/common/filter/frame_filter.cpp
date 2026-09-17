
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "frame_filter.hpp"
#include "obj_filter_common.hpp"
#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {

static const std::string key_match_mode = "match_mode";

namespace {

// match_mode 取值：
//   "any"  （默认）：frame 上存在至少一个命中谓词的 obj 时，该帧通过（需要推理）
//   "none"         ：frame 上所有 obj 均未命中谓词时，该帧通过
// 其余取值在 Init 阶段显式报错，避免配置错误被静默忽略。
bool ParseMatchMode(const std::string& mode, bool* none_mode) {
  if (mode == "any") {
    *none_mode = false;
    return true;
  }
  if (mode == "none") {
    *none_mode = true;
    return true;
  }
  return false;
}

}  // namespace

/**
 * 借用 ObjFilterCommon 实现基于 frame 上已挂载的检出对象 进行筛选
 *
 * match_mode 控制聚合语义：
 *   - match_mode=any（默认）：任一 obj 命中 -> 帧通过
 *   - match_mode=none       ：全部 obj 未命中 -> 帧通过（例如"无目标时才推理"）
 *
 * 谓词可引用的 objs 仅限该帧到达推理模块时已挂载的结果（同路径上游模块产出）。
 */
class FrameFilterByObjs : public FrameFilter {
 public:
  bool Init(const std::map<std::string, std::string>& params) override {
    params_ = params;

    auto it = params_.find(key_match_mode);
    if (it != params_.end()) {
      if (!ParseMatchMode(it->second, &none_mode_)) {
        LOGE(FILTER) << "FrameFilterByObjs Init: invalid " << key_match_mode
                     << "=" << it->second << " (expected \"any\" or \"none\")";
        return false;
      }
      params_.erase(key_match_mode);
    }

    inner_filter_ = std::make_shared<ObjFilterCommon>();
    if (!inner_filter_->Init(params_)) {
      LOGE(FILTER) << "FrameFilterByObjs Init: inner ObjFilterCommon init failed.";
      return false;
    }

    LOGD(FILTER) << "FrameFilterByObjs Init: match_mode=" << (none_mode_ ? "none" : "any");
    return true;
  }

  /**
   * @return true = 该帧需要推理；false = 该帧跳过推理
   */
  bool Filter(const FrameInfoPtr& finfo) override {
    if (!finfo->collection.HasValue(kInferObjsTag)) {
      // 无 objs：any 语义下视为未命中（不推理），none 语义下视为全未命中（推理）
      return none_mode_;
    }

    InferObjsPtr objs_holder = finfo->collection.Get<InferObjsPtr>(kInferObjsTag);
    objs_holder->mutex_.lock();
    std::vector<std::shared_ptr<InferObject>> objs = objs_holder->objs_;
    objs_holder->mutex_.unlock();

    bool any_matched = false;
    for (const auto& obj : objs) {
      if (inner_filter_->Filter(finfo, obj)) {
        any_matched = true;
        break;
      }
    }
    return none_mode_ ? !any_matched : any_matched;
  }

 private:
  std::map<std::string, std::string> params_;
  std::shared_ptr<ObjFilterCommon> inner_filter_ = nullptr;
  bool none_mode_ = false;  // false: any（默认），true: none

 private:
  DECLARE_REFLEX_OBJECT_EX(FrameFilterByObjs, cnstream::FrameFilter);
};  // class FrameFilterByObjs

IMPLEMENT_REFLEX_OBJECT_EX(FrameFilterByObjs, cnstream::FrameFilter);

}  // namespace cnstream
