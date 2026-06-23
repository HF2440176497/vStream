

#include "postproc.hpp"


#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace cnstream {

static const std::string key_config_file = "config_file";

static const std::string key_name = "name";

class Post_Resnet_Obj : public ObjPostproc {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {

    params_ = params;
    if (params_.find(key_config_file) != params_.end()) {
      config_file_ = params_[key_config_file];
    } else {
      LOGE(POSTPROC) << "Init config_file must be in custom_postproc_params.";
      return false;
    }
    std::string config_file_path = "./";
    if (params_.find(CNS_JSON_DIR_PARAM_NAME) != params_.end()) {
      config_file_path = params_[CNS_JSON_DIR_PARAM_NAME];
    }
    config_file_ = GetPathRelativeToTheJSONFile(config_file_, config_file_path);

    LOGI(POSTPROC) << "Post_Resnet_Obj post conf file: " << config_file_;
    std::ifstream file(config_file_);
    if (!file.is_open()) {
      LOGE(POSTPROC) << "Init Could not open file " << config_file_;
      return false;
    }
    nlohmann::ordered_json data = nlohmann::ordered_json::parse(file);
    if (!data.is_object()) {
      LOGE(POSTPROC) << "Init config file must be object type.";
      return false;
    }

    if (data.find(key_classes) != data.end()) {
      const auto& classes = data[key_classes];
      if (!classes.is_object()) {
        LOGE(POSTPROC) << "Invalid classes format in conf file.";
        return false;
      }
      int max_label = -1;
      for (auto it = classes.begin(); it != classes.end(); ++it) {
        int label = std::stoi(it.key());
        if (label > max_label) max_label = label;
      }
      if (max_label < 0) {
        LOGE(POSTPROC) << "No valid label found in config.";
        return false;
      }
      item_infos_.resize(max_label + 1);
      for (auto it = classes.begin(); it != classes.end(); ++it) {
        int label = std::stoi(it.key());
        const auto& value = it.value();
        if (!value.is_object()) {
          LOGE(POSTPROC) << "Invalid item format in conf file, key: " << it.key();
          return false;
        }
        item_infos_[label].name = value["name"].get<std::string>();
      }
    }  // find classes
    return true;
  }

  /**
   * outputs: D2H 的结果
   */
  int Execute(const std::vector<float*>& outputs, ModelLoader* model,
              const FrameInfoPtr& finfo, const std::shared_ptr<InferObject>& pobj) override {

    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }

    int output_index = 0;

    const float* output = outputs[output_index];
    TensorShape output_shape = model->OutputShape(output_index);
    int num_classes = output_shape.numel();

    auto max_it = std::max_element(output, output + num_classes);
    float max_logit = *max_it;
    int best_class = static_cast<int>(max_it - output);

    float exp_sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
      exp_sum += std::exp(output[i] - max_logit);
    }

    float best_score = std::exp(output[best_class] - max_logit) / exp_sum;

    std::string class_name;
    if (best_class < item_infos_.size()) {
      class_name = item_infos_[best_class].name;
    } 
    InferObjectInfo class_info;
    class_info.id = best_class;
    class_info.name = class_name;
    class_info.model_name = model->get_name();
    class_info.score = best_score;
    class_info.value = best_score;

    pobj->classes.push_back(class_info);

    return 0; 
  }

 private:
  // TODO: 后期可加入阈值
  struct ItemInfo {
    std::string name;
  };
  std::vector<ItemInfo> item_infos_;
  std::string model_name_;

 private:
  DECLARE_REFLEX_OBJECT_EX(Post_Resnet_Obj, cnstream::ObjPostproc);
};  // class Post_Resnet_Obj

IMPLEMENT_REFLEX_OBJECT_EX(Post_Resnet_Obj, cnstream::ObjPostproc);

}  // namespace cnstream