
#ifndef MODULES_CNSTREAM_OBJ_RULE_HPP_
#define MODULES_CNSTREAM_OBJ_RULE_HPP_

#include <cctype>
#include <cmath>
#include <cstdio>
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

// ---------- 数值/集合格式化（条件自描述时复用） ----------

inline std::string FormatFloat(float v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%g", v);
  return buf;
}

inline std::string JoinStrSet(const std::set<std::string>& s) {
  std::string out;
  for (const auto& v : s) {
    if (!out.empty()) out += ",";
    out += "'" + v + "'";
  }
  return "{" + out + "}";
}

inline std::string JoinIntSet(const std::set<int>& s) {
  std::string out;
  for (int v : s) {
    if (!out.empty()) out += ",";
    out += std::to_string(v);
  }
  return "{" + out + "}";
}

inline std::string DescribeAxisRange(const AxisRange& r) {
  if (r.empty()) return "any";
  if (r.has_min && r.has_max) return "[" + FormatFloat(r.min) + "," + FormatFloat(r.max) + "]";
  if (r.has_min) return "[" + FormatFloat(r.min) + ",+inf)";
  return "(-inf," + FormatFloat(r.max) + "]";
}

/**
 * @brief 单个选择条件：自描述。
 *
 * 判断、命中/未命中说明、对象取值均由条件自己负责，
 * 新增规则类型只需新增一个 ObjCond 子类并在 ParseObjRules 中增加一个分支。
 */
struct ObjCond {
  virtual ~ObjCond() = default;
  /// 未命中返回原因（含对象实际值，如 "id=3 not in {1,2}"），命中返回空串
  virtual std::string Check(const std::shared_ptr<InferObject>& obj) const = 0;
  /// 对象在该条件上的取值，如 "id=3"（用于日志中的对象摘要）
  virtual std::string Value(const std::shared_ptr<InferObject>& obj) const = 0;
  /// 条件的配置描述，如 "ids in {1,2}"
  virtual std::string Describe() const = 0;
};

/// model 条件
struct ModelCond : ObjCond {
  std::set<std::string> models;
  std::string Check(const std::shared_ptr<InferObject>& obj) const override {
    if (models.count(obj->model_name)) return "";
    return "model '" + obj->model_name + "' not in " + JoinStrSet(models);
  }
  std::string Value(const std::shared_ptr<InferObject>& obj) const override {
    return "model='" + obj->model_name + "'";
  }
  std::string Describe() const override { return "model in " + JoinStrSet(models); }
};

/// 类别 id 条件
struct IdsCond : ObjCond {
  std::set<int> ids;
  std::string Check(const std::shared_ptr<InferObject>& obj) const override {
    if (ids.count(obj->id)) return "";
    return "id=" + std::to_string(obj->id) + " not in " + JoinIntSet(ids);
  }
  std::string Value(const std::shared_ptr<InferObject>& obj) const override {
    return "id=" + std::to_string(obj->id);
  }
  std::string Describe() const override { return "ids in " + JoinIntSet(ids); }
};

/// 目标类型条件（canonical 值："original"/"merged"）
struct TypeCond : ObjCond {
  std::set<std::string> types;
  std::string Check(const std::shared_ptr<InferObject>& obj) const override {
    const std::string type = GetInferObjType(obj);
    if (types.count(type)) return "";
    return "type '" + type + "' not in " + JoinStrSet(types);
  }
  std::string Value(const std::shared_ptr<InferObject>& obj) const override {
    return "type='" + GetInferObjType(obj) + "'";
  }
  std::string Describe() const override { return "type in " + JoinStrSet(types); }
};

/// bbox 单轴范围条件
struct AxisCond : ObjCond {
  bool is_x = true;  ///< true: bbox.x，false: bbox.y
  AxisRange range;
  float Coord(const std::shared_ptr<InferObject>& obj) const {
    return is_x ? obj->bbox.x : obj->bbox.y;
  }
  std::string Check(const std::shared_ptr<InferObject>& obj) const override {
    const float v = Coord(obj);
    if (range.Contains(v)) return "";
    return std::string("bbox.") + (is_x ? "x" : "y") + "=" + FormatFloat(v) +
           " not in " + DescribeAxisRange(range);
  }
  std::string Value(const std::shared_ptr<InferObject>& obj) const override {
    return std::string(is_x ? "x" : "y") + "=" + FormatFloat(Coord(obj));
  }
  std::string Describe() const override {
    return std::string(is_x ? "x" : "y") + " in " + DescribeAxisRange(range);
  }
};

/**
 * @brief 对象选择规则
 *
 * 一条规则内各已配置条件取与；多规则之间取或（由调用方遍历实现）；
 * 未配置任何条件 = 恒命中。
 */
struct ObjRule {
  std::vector<std::shared_ptr<ObjCond>> conds;  ///< 仅存放已配置的条件

  /// 是否命中（所有条件满足）
  bool Match(const std::shared_ptr<InferObject>& obj) const {
    for (const auto& cond : conds) {
      if (!cond->Check(obj).empty()) return false;
    }
    return true;
  }
  /// 首个不满足条件及对象实际值；全部满足返回空串
  std::string MismatchReason(const std::shared_ptr<InferObject>& obj) const {
    for (const auto& cond : conds) {
      std::string reason = cond->Check(obj);
      if (!reason.empty()) return reason;
    }
    return "";
  }
  /// 条件描述；无条件时为 "any"
  std::string Describe() const {
    std::string out;
    for (const auto& cond : conds) {
      if (!out.empty()) out += " and ";
      out += cond->Describe();
    }
    return out.empty() ? "any" : out;
  }
  /// 对象在各条件上的取值摘要（逗号连接）
  std::string DescribeObjValues(const std::shared_ptr<InferObject>& obj) const {
    std::string out;
    for (const auto& cond : conds) {
      if (!out.empty()) out += ", ";
      out += cond->Value(obj);
    }
    return out;
  }
};

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
        auto cond = std::make_shared<ModelCond>();
        if (val.is_string()) {
          cond->models.insert(val.get<std::string>());
        } else if (val.is_array()) {
          for (const auto& m : val) {
            if (!m.is_string()) {
              *err = "rule.model array must contain strings only";
              return false;
            }
            cond->models.insert(m.get<std::string>());
          }
        } else {
          *err = "rule.model must be a string or an array of strings";
          return false;
        }
        rule.conds.push_back(std::move(cond));
      } else if (key == "ids") {
        auto cond = std::make_shared<IdsCond>();
        if (val.is_number_integer()) {
          cond->ids.insert(val.get<int>());
        } else if (val.is_array()) {
          for (const auto& id_val : val) {
            if (!id_val.is_number_integer()) {
              *err = "rule.ids array must contain integers only";
              return false;
            }
            cond->ids.insert(id_val.get<int>());
          }
        } else {
          *err = "rule.ids must be an integer or an array of integers";
          return false;
        }
        rule.conds.push_back(std::move(cond));
      } else if (key == "type") {
        auto cond = std::make_shared<TypeCond>();
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
          cond->types.insert(std::string(ToString(type)));
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
        rule.conds.push_back(std::move(cond));
      } else if (key == "position") {
        if (!val.is_object()) {
          *err = "rule.position must be an object with optional x/y arrays";
          return false;
        }
        for (auto pit = val.begin(); pit != val.end(); ++pit) {
          bool is_x;
          if (pit.key() == "x") {
            is_x = true;
          } else if (pit.key() == "y") {
            is_x = false;
          } else {
            *err = "unknown rule.position key '" + pit.key() + "', expect 'x'/'y'";
            return false;
          }
          AxisRange range;
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
              range.has_min = true;
              range.min = bound.get<float>();
            } else {
              range.has_max = true;
              range.max = bound.get<float>();
            }
          }
          if (!range.empty()) {  // 空范围（如 []）= 该轴不限制，不生成条件
            auto cond = std::make_shared<AxisCond>();
            cond->is_x = is_x;
            cond->range = range;
            rule.conds.push_back(std::move(cond));
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
