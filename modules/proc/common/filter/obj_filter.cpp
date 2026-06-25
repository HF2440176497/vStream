
#include "obj_filter.hpp"

#include <algorithm>

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {

static const std::string key_model_name = "f_model_name";
static const std::string key_obj_id = "f_obj_id";
static const std::string key_obj_type = "f_obj_type";


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
  // Parse a "#a!b!c" style string into a list of integers.
  // The leading '#' is optional.
  static std::vector<int> ParseIntList(const std::string& raw) {
    std::vector<int> result;
    std::string value = raw;
    if (!value.empty() && value[0] == '#') {
      value.erase(0, 1);
    }
    size_t start = 0;
    while (start < value.size()) {
      size_t end = value.find('!', start);
      if (end == std::string::npos) {
        end = value.size();
      }
      std::string token = value.substr(start, end - start);
      if (!token.empty()) {
        try {
          result.push_back(std::stoi(token));
        } catch (const std::exception&) {
          // ignore invalid integer token
        }
      }
      start = end + 1;
    }
    return result;
  }

  // Parse a "#merged!original" style string into a list of strings.
  // The leading '#' is optional.
  static std::vector<std::string> ParseStringList(const std::string& raw) {
    std::vector<std::string> result;
    std::string value = raw;
    if (!value.empty() && value[0] == '#') {
      value.erase(0, 1);
    }
    size_t start = 0;
    while (start < value.size()) {
      size_t end = value.find('!', start);
      if (end == std::string::npos) {
        end = value.size();
      }
      std::string token = value.substr(start, end - start);
      if (!token.empty()) {
        result.push_back(token);
      }
      start = end + 1;
    }
    return result;
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