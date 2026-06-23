
#include "obj_filter.hpp"

#include <algorithm>

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"

namespace cnstream {

static const std::string key_model_name = "f_model_name";
static const std::string key_obj_id = "f_obj_id";


class ObjFilterCommon : public ObjFilter {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(key_model_name) != params_.end()) {
      model_name_ = params_[key_model_name];
    }
    if (params_.find(key_obj_id) != params_.end()) {
      std::string value = params_[key_obj_id];
      if (!value.empty() && value[0] == '#') {
        value.erase(0, 1);
      }
      size_t start = 0;
      while (start < value.size()) {
        size_t end = value.find('!', start);
        if (end == std::string::npos) {
          end = value.size();
        }
        std::string id_str = value.substr(start, end - start);
        if (!id_str.empty()) {
          obj_ids_.push_back(std::stoi(id_str));
        }
        start = end + 1;
      }
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
    return true;
  }

 private:
  std::map<std::string, std::string> params_;
  std::string model_name_;
  std::vector<int> obj_ids_;

 private:
  DECLARE_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);
};  // class ObjFilterCommon

IMPLEMENT_REFLEX_OBJECT_EX(ObjFilterCommon, cnstream::ObjFilter);


}  // namespace cnstream