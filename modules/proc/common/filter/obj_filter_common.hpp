
#ifndef MODULES_PROC_COMMON_FILTER_OBJ_FILTER_COMMON_HPP_
#define MODULES_PROC_COMMON_FILTER_OBJ_FILTER_COMMON_HPP_

#include <map>
#include <string>
#include <vector>

#include "obj_filter.hpp"
#include "cnstream_obj_rule.hpp"

namespace cnstream {

/**
 * 通用对象过滤器
 *
 * 通过 "filter" 配置项接收对象选择规则：对象命中任一规则即保留
 * 未配置 "filter" 时不过滤（全部保留）
 */
class ObjFilterCommon : public ObjFilter {
 public:
  bool Init(const std::map<std::string, std::string>& params) override;
  /**
   * @return
   * 返回 false 时，说明当前 obj 被过滤，continue
   * 返回 true 时，说明当前 obj 被保留
   */
  bool Filter(const FrameInfoPtr& finfo, const InferObjectPtr& pobj) override;

 private:
  std::map<std::string, std::string> params_;
  std::vector<ObjRule> rules_;

 private:
  DECLARE_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);
};  // class ObjFilterCommon

}  // namespace cnstream

#endif  // ifndef MODULES_PROC_COMMON_FILTER_OBJ_FILTER_COMMON_HPP_
