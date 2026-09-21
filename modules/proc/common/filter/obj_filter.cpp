
#include <string>

#include "common.hpp"
#include "obj_filter.hpp"
#include "obj_filter_common.hpp"

namespace cnstream {

static const std::string key_filter = "filter";

bool ObjFilterCommon::Init(const std::map<std::string, std::string>& params) {
  params_ = params;
  auto it = params_.find(key_filter);
  if (it != params_.end()) {
    std::string err;
    if (!ParseObjRules(it->second, &rules_, &err)) {
      LOGE(FILTER) << "ObjFilterCommon Init: invalid " << key_filter << ": " << err;
      rules_.clear();
      return false;
    }
  }
  LOGD(FILTER) << "ObjFilterCommon Init: " << key_filter << " rules_num=" << rules_.size();
  return true;
}

  /**
   * @return
   * 返回 false 时，说明当前 obj 被过滤，continue
   * 返回 true 时，说明当前 obj 被保留
   */
bool ObjFilterCommon::Filter(const FrameInfoPtr& finfo, const InferObjectPtr& pobj) {
  if (rules_.empty()) {
    return true;  // 未配置规则，不过滤
  }
  // 命中任一规则即保留（规则内条件取与，见 MatchObjRule）
  for (const auto& rule : rules_) {
    if (MatchObjRule(rule, pobj)) return true;
  }
  return false;
}

IMPLEMENT_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);


}  // namespace cnstream
