
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

#include "common.hpp"
#include "obj_filter.hpp"
#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {

static const std::string key_model_name = "f_model_name";
static const std::string key_obj_id = "f_obj_id";
static const std::string key_obj_type = "f_obj_type";
static const std::string key_obj_position = "f_obj_position";

namespace {

// Trim ASCII whitespace from both ends of a string.
std::string Trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// Split a comma-separated string into non-empty trimmed tokens.
// Empty segments (caused by leading/trailing/duplicated commas) are dropped
// so that inputs like "1, , 2 ,, 3" all parse cleanly to {"1","2","3"}.
std::vector<std::string> SplitCsv(const std::string& raw) {
  std::vector<std::string> tokens;
  size_t start = 0;
  while (start <= raw.size()) {
    size_t end = raw.find(',', start);
    if (end == std::string::npos) end = raw.size();
    std::string token = Trim(raw.substr(start, end - start));
    if (!token.empty()) {
      tokens.push_back(std::move(token));
    }
    if (end == raw.size()) break;
    start = end + 1;
  }
  return tokens;
}

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

std::ostream& operator<<(std::ostream& os, const AxisRange& range) {
    if (range.empty()) {
        os << "(-∞, +∞)";
        return os;
    }
    os << (range.has_min ? "[" : "(-∞");
    if (range.has_min) os << range.min;
    os << ", ";
    if (range.has_max) os << range.max;
    os << (range.has_max ? "]" : "+∞)");
    return os;
}


}  // namespace


class ObjFilterCommon : public ObjFilter {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(key_model_name) != params_.end()) {
      model_name_ = params_[key_model_name];
    }
    if (params_.find(key_obj_id) != params_.end()) {
      std::string value = params_[key_obj_id];
      obj_ids_ = ParseIntList(value);
    }
    if (params_.find(key_obj_type) != params_.end()) {
      std::string value = params_[key_obj_type];
      obj_types_ = ParseStringList(value);
    }
    if (params_.find(key_obj_position) != params_.end()) {
      if (!ParsePosition(params_[key_obj_position], &x_range_, &y_range_)) {
        LOGE(FILTER) << "ObjFilterCommon Init: invalid " << key_obj_position
                     << "=" << params_[key_obj_position];
        return false;
      }
    }
    LOGD(FILTER) << "ObjFilterCommon Init: "
                 << key_model_name << "=" << model_name_ << " "
                 << key_obj_id << "=" << obj_ids_ << " "
                 << key_obj_type << "=" << obj_types_;
    if (!x_range_.empty() || !y_range_.empty()) {
      LOGD(FILTER) << "ObjFilterCommon position filter: "
                   << key_obj_position << "=" << params_[key_obj_position];
    }

    return true;
  }

  /**
   * @return
   * 返回 false 时，说明当前 obj 被过滤，continue
   * 返回 true 时，说明当前 obj 被保留
   */
  bool Filter(const FrameInfoPtr& finfo, const InferObjectPtr& pobj) override {
    if (!model_name_.empty() && pobj->model_name != model_name_) {
      return false;
    }
    if (!obj_ids_.empty() &&
        std::find(obj_ids_.begin(), obj_ids_.end(), pobj->id) == obj_ids_.end()) {
      return false;
    }
    if (!obj_types_.empty()) {
      std::string obj_type = GetInferObjType(pobj);
      if (std::find(obj_types_.begin(), obj_types_.end(), obj_type) == obj_types_.end()) {
        return false;
      }
    }
    // 任一轴配置了范围且坐标不在范围内时过滤该目标。
    if (!x_range_.Contains(pobj->bbox.x)) {
      return false;
    }
    if (!y_range_.Contains(pobj->bbox.y)) {
      return false;
    }
    return true;
  }

 private:
  // Parse a "1,2,3,4" style string into a list of integers.
  // Whitespace around tokens and extra/leading/trailing commas are tolerated;
  // tokens that fail to parse as integers are silently skipped.
  static std::vector<int> ParseIntList(const std::string& raw) {
    std::vector<int> result;
    for (const std::string& token : SplitCsv(raw)) {
      try {
        size_t consumed = 0;
        int value = std::stoi(token, &consumed);
        if (consumed == token.size()) {
          result.push_back(value);
        }
        // trailing garbage in token -> skip
      } catch (const std::exception&) {
        // ignore invalid integer token
      }
    }
    return result;
  }

  // Parse a "merged,original" style string into a list of strings.
  // Whitespace around tokens and extra/leading/trailing commas are tolerated.
  static std::vector<std::string> ParseStringList(const std::string& raw) {
    return SplitCsv(raw);
  }

  // 解析 "f_obj_position" 配置，格式如：
  //   - 对象  {"x":[0.1,0.9],"y":[0.2]} -> 经 dump() 为紧凑串
  //   - 字符串 "{\"x\": [0.1, 0.9], \"y\": [0.2]}"      -> 原样保留（含空格）
  // key（x/y）可缺省，缺省表示该轴不限制；数组表示 [min, max]，元素可缺省：
  //   [x1,x2]            上下限均限制
  //   [x1] / [x1,]       仅下限（>= x1）
  //   [,x2] / [null,x2]  仅上限（<= x2）
  //   []                 该轴不限制
  // 解析失败返回 false，由 Init 显式报错，避免配置错误被静默忽略。
  static bool ParsePosition(const std::string& raw, AxisRange* x_range, AxisRange* y_range) {
    std::string s = Trim(raw);
    if (s.size() < 2 || s.front() != '{' || s.back() != '}') {
      return false;
    }
    s = s.substr(1, s.size() - 2);

    size_t pos = 0;
    while (true) {
      // 跳过 key 之间的逗号和空白
      while (pos < s.size() &&
             (std::isspace(static_cast<unsigned char>(s[pos])) || s[pos] == ',')) {
        ++pos;
      }
      if (pos >= s.size()) break;

      // 解析 key（可带引号），直到 ':'
      size_t colon = s.find(':', pos);
      if (colon == std::string::npos) return false;
      std::string key = Trim(s.substr(pos, colon - pos));
      pos = colon + 1;
      if (key.size() >= 2 && key.front() == '"' && key.back() == '"') {
        key = key.substr(1, key.size() - 2);
      }
      AxisRange* range = nullptr;
      if (key == "x") {
        range = x_range;
      } else if (key == "y") {
        range = y_range;
      } else {
        return false;  // 未知 key 视为配置错误
      }

      // 解析 "[a,b]" 数组体
      size_t lb = s.find('[', pos);
      if (lb == std::string::npos || !Trim(s.substr(pos, lb - pos)).empty()) {
        return false;
      }
      size_t rb = s.find(']', lb + 1);
      if (rb == std::string::npos) return false;
      if (!ParseAxisRange(s.substr(lb + 1, rb - lb - 1), range)) return false;
      pos = rb + 1;
    }
    if (x_range) {
      LOGI(FILTER) << "x_range: " << *x_range;
    }
    if (y_range) {
      LOGI(FILTER) << "y_range: " << *y_range;
    }
    return true;
  }

  // 解析单轴数组体，最多两个元素（min, max），超出两个元素视为非法。
  static bool ParseAxisRange(const std::string& body, AxisRange* range) {
    *range = AxisRange{};
    std::string first = body;
    std::string second;
    size_t comma = body.find(',');
    if (comma != std::string::npos) {
      first = body.substr(0, comma);
      second = body.substr(comma + 1);
      if (second.find(',') != std::string::npos) {
        return false;
      }
    }
    int r = ParseBoundToken(first, &range->min);
    if (r < 0) return false;
    range->has_min = (r > 0);
    r = ParseBoundToken(second, &range->max);
    if (r < 0) return false;
    range->has_max = (r > 0);
    return true;
  }

  // 返回 1：token 为有效数字；0：token 为空或 null（表示不限制）；-1：非法 token
  static int ParseBoundToken(const std::string& token, float* value) {
    std::string t = Trim(token);
    if (t.empty() || t == "null") return 0;
    try {
      size_t consumed = 0;
      float v = std::stof(t, &consumed);
      if (consumed != t.size() || !std::isfinite(v)) return -1;
      *value = v;
    } catch (const std::exception&) {
      return -1;
    }
    return 1;
  }

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

IMPLEMENT_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);


}  // namespace cnstream