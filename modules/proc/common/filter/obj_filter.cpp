
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
  // 命中任一规则即保留（规则内条件取与）；条件自判断、自描述、自报值
  std::string miss_desc;
  for (size_t i = 0; i < rules_.size(); ++i) {
    const std::string reason = rules_[i].MismatchReason(pobj);
    if (reason.empty()) {
      LOGI(FILTER) << "obj keep: hit rule [" << i << "] " << rules_[i].Describe()
                   << "], obj(" << rules_[i].DescribeObjValues(pobj) << ")";
      return true;
    }
    if (!miss_desc.empty()) miss_desc += " | ";
    miss_desc += "[" + std::to_string(i) + "] " + reason;
  }
  LOGI(FILTER) << "miss match desc: " << miss_desc;
  return false;
}

IMPLEMENT_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);


}  // namespace cnstream
