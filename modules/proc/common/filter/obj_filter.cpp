
#include <algorithm>
#include <cctype>
#include <string>

#include "common.hpp"
#include "obj_filter.hpp"
#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {

static const std::string key_model_name = "f_model_name";
static const std::string key_obj_id = "f_obj_id";
static const std::string key_obj_type = "f_obj_type";

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
    LOGD(FILTER) << "ObjFilterCommon Init: " 
                 << key_model_name << "=" << model_name_ << " "
                 << key_obj_id << "=" << obj_ids_ << " " 
                 << key_obj_type << "=" << obj_types_;

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

 private:
  std::map<std::string, std::string> params_;
  std::string model_name_;
  std::vector<int> obj_ids_;
  std::vector<std::string> obj_types_;

 private:
  DECLARE_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);
};  // class ObjFilterCommon

IMPLEMENT_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);


}  // namespace cnstream