
#ifndef MODULES_PROC_COMMON_FILTER_OBJ_FILTER_COMMON_HPP_
#define MODULES_PROC_COMMON_FILTER_OBJ_FILTER_COMMON_HPP_

#include <map>
#include <string>
#include <vector>

#include "obj_filter.hpp"

namespace cnstream {

// 单轴坐标范围（图像坐标）。上下限均可缺省，缺省表示该侧不限制。
struct AxisRange {
  bool has_min = false;
  float min = 0.0f;
  bool has_max = false;
  float max = 0.0f;

  bool empty() const { return !has_min && !has_max; }
  bool Contains(float value) const {
    if (has_min && value < min) return false;
    if (has_max && value > max) return false;
    return true;
  }
};

/**
 * 通用对象过滤器
 *
 * 支持 f_model_name / f_obj_id / f_obj_type / f_obj_position 四类谓词配置
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
  // Parse a "1,2,3,4" style string into a list of integers.
  static std::vector<int> ParseIntList(const std::string& raw);
  // Parse a "merged,original" style string into a list of strings.
  static std::vector<std::string> ParseStringList(const std::string& raw);
  // 解析 "f_obj_position" 配置，格式如 {"x":[0.1,0.9],"y":[0.2]}
  static bool ParsePosition(const std::string& raw, AxisRange* x_range, AxisRange* y_range);
  // 解析单轴数组体，最多两个元素（min, max），超出两个元素视为非法。
  static bool ParseAxisRange(const std::string& body, AxisRange* range);
  // 返回 1：token 为有效数字；0：token 为空或 null（表示不限制）；-1：非法 token
  static int ParseBoundToken(const std::string& token, float* value);

 private:
  std::map<std::string, std::string> params_;
  std::string model_name_;
  std::vector<int> obj_ids_;
  std::vector<std::string> obj_types_;
  AxisRange x_range_;
  AxisRange y_range_;

 private:
  DECLARE_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);
};  // class ObjFilterCommon

}  // namespace cnstream

#endif  // ifndef MODULES_PROC_COMMON_FILTER_OBJ_FILTER_COMMON_HPP_
