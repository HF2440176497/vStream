
#ifndef MODULES_CNSTREAM_OBJ_RULE_HPP_
#define MODULES_CNSTREAM_OBJ_RULE_HPP_

#include <cctype>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cnstream_frame_va.hpp"

namespace cnstream {

/**
 * @brief 单轴坐标范围。上下限均可缺省，缺省表示该侧不限制。
 */
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
 * @brief 对象选择规则
 *
 * 一条规则内各已配置条件取与；多规则之间取或；集合为空 / 范围为空表示不限制。
 */
struct ObjRule {
  std::set<std::string> models;  ///< 模型名
  std::set<int> ids;             ///< 类别 id
  std::set<std::string> types;   ///< 目标类型（canonical 值："original"/"merged"）
  AxisRange x_range;             ///< bbox.x 范围
  AxisRange y_range;             ///< bbox.y 范围
};

/**
 * @brief 判断对象是否命中规则（规则内条件取与）。
 */
inline bool MatchObjRule(const ObjRule& rule, const std::shared_ptr<InferObject>& obj) {
  if (!rule.models.empty() && rule.models.count(obj->model_name) == 0) return false;
  if (!rule.ids.empty() && rule.ids.count(obj->id) == 0) return false;
  if (!rule.types.empty() && rule.types.count(GetInferObjType(obj)) == 0) return false;
  if (!rule.x_range.empty() && !rule.x_range.Contains(obj->bbox.x)) return false;
  if (!rule.y_range.empty() && !rule.y_range.Contains(obj->bbox.y)) return false;
  return true;
}

/**
 * @brief 解析规则配置（JSON 数组，或单个规则对象）。
 *
 * 规则字段：
 *   - model:    字符串或字符串数组
 *   - ids:      整数或整数数组
 *   - type:     字符串或字符串数组
 *   - position: 对象，可选 x/y 轴，各轴为数组 [min, max]，
 *               元素可为 null（该侧不限制），最多 2 个元素，空数组表示该轴不限制
 *
 * @param[in]  filter 规则 JSON 文本；空白串视为无规则
 * @param[out] rules  成功时为解析结果；失败时被清空
 * @param[out] err    失败原因；成功时为空
 * @return 解析是否成功。日志由调用方按自身模块 tag 输出 err。
 */
inline bool ParseObjRules(const std::string& filter, std::vector<ObjRule>* rules, std::string* err) {
  rules->clear();
  err->clear();

  size_t b = 0, e = filter.size();
  while (b < e && std::isspace(static_cast<unsigned char>(filter[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(filter[e - 1]))) --e;
  std::string trimmed = filter.substr(b, e - b);
  if (trimmed.empty()) {
    return true;  // 空 = 无规则，合法
  }

  nlohmann::json doc = nlohmann::json::parse(trimmed, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded()) {
    *err = "invalid JSON";
    return false;
  }
  if (doc.is_object()) {
    doc = nlohmann::json::array({doc});  // 单个规则对象 → 单规则数组
  }
  if (!doc.is_array()) {
    *err = "expect a JSON array of rule objects (or a single object)";
    return false;
  }

  std::vector<ObjRule> parsed;
  parsed.reserve(doc.size());
  for (const auto& item : doc) {
    if (!item.is_object()) {
      *err = "each rule must be a JSON object";
      return false;
    }
    ObjRule rule;
    for (auto it = item.begin(); it != item.end(); ++it) {
      const std::string& key = it.key();
      const nlohmann::json& val = it.value();
      if (key == "model") {
        if (val.is_string()) {
          rule.models.insert(val.get<std::string>());
        } else if (val.is_array()) {
          for (const auto& m : val) {
            if (!m.is_string()) {
              *err = "rule.model array must contain strings only";
              return false;
            }
            rule.models.insert(m.get<std::string>());
          }
        } else {
          *err = "rule.model must be a string or an array of strings";
          return false;
        }
      } else if (key == "ids") {
        if (val.is_number_integer()) {
          rule.ids.insert(val.get<int>());
        } else if (val.is_array()) {
          for (const auto& id_val : val) {
            if (!id_val.is_number_integer()) {
              *err = "rule.ids array must contain integers only";
              return false;
            }
            rule.ids.insert(id_val.get<int>());
          }
        } else {
          *err = "rule.ids must be an integer or an array of integers";
          return false;
        }
      } else if (key == "type") {
        auto parse_one = [&](const nlohmann::json& t) -> bool {
          if (!t.is_string()) {
            *err = "rule.type must be a string or an array of strings";
            return false;
          }
          const std::string type_str = t.get<std::string>();
          InferObjType type = InferObjTypeFromString(type_str);
          if (type == InferObjType::kUnknown) {
            *err = "unknown rule.type '" + type_str + "', expect 'original' or 'merged'";
            return false;
          }
          rule.types.insert(std::string(ToString(type)));
          return true;
        };
        if (val.is_string()) {
          if (!parse_one(val)) return false;
        } else if (val.is_array()) {
          for (const auto& t : val) {
            if (!parse_one(t)) return false;
          }
        } else {
          *err = "rule.type must be a string or an array of strings";
          return false;
        }
      } else if (key == "position") {
        if (!val.is_object()) {
          *err = "rule.position must be an object with optional x/y arrays";
          return false;
        }
        for (auto pit = val.begin(); pit != val.end(); ++pit) {
          AxisRange* range = nullptr;
          if (pit.key() == "x") {
            range = &rule.x_range;
          } else if (pit.key() == "y") {
            range = &rule.y_range;
          } else {
            *err = "unknown rule.position key '" + pit.key() + "', expect 'x'/'y'";
            return false;
          }
          const nlohmann::json& axis = pit.value();
          if (!axis.is_array()) {
            *err = "rule.position." + pit.key() + " must be an array";
            return false;
          }
          if (axis.size() > 2) {
            *err = "rule.position." + pit.key() + " must have at most 2 elements [min, max]";
            return false;
          }
          for (size_t i = 0; i < axis.size(); ++i) {
            const auto& bound = axis[i];
            if (bound.is_null()) continue;  // null = 该侧不限制
            if (!bound.is_number() || !std::isfinite(bound.get<float>())) {
              *err = "rule.position." + pit.key() + " elements must be numbers or null";
              return false;
            }
            if (i == 0) {
              range->has_min = true;
              range->min = bound.get<float>();
            } else {
              range->has_max = true;
              range->max = bound.get<float>();
            }
          }
        }
      } else {
        *err = "unknown rule key '" + key + "'";
        return false;
      }
    }
    parsed.push_back(std::move(rule));
  }

  *rules = std::move(parsed);
  return true;
}

}  // namespace cnstream
#endif  // MODULES_CNSTREAM_OBJ_RULE_HPP_
