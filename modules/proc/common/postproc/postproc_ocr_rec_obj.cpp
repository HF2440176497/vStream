
#include "postproc.hpp"


#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"
#include "proc/common/debug_image_saver.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "crnn_process.hpp"  // from PaddleOCR

namespace cnstream {

namespace {

inline constexpr const char* key_config_file = "config_file";
inline constexpr const char* key_label_path = "label_path";
inline constexpr const char* key_enable_save = "enable_save";
inline constexpr const char* key_save_interval_ms = "save_interval_ms";

}  // namespace

class Post_PPOCRv3_rec_Obj : public ObjPostproc {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(key_config_file) != params_.end()) {
      config_file_ = params_[key_config_file];
    } else {
      LOGE(POSTPROC) << "Init config_file must be in custom_postproc_params.";
      return false;
    }
    std::string dir_path;
    if (params_.find(CNS_JSON_DIR_PARAM_NAME) != params_.end()) {
      dir_path = params_[CNS_JSON_DIR_PARAM_NAME];
    }
    config_file_ = GetPathRelativeToTheJSONFile(config_file_, dir_path);

    LOGI(POSTPROC) << "PPOCRv3 post conf file: " << config_file_;
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
    if (data.find(key_label_path) == data.end()) {
      LOGE(POSTPROC) << "Init label_path must be in config file.";
      return false;
    }
    label_path_ = data[key_label_path].get<std::string>();
    LOGI(POSTPROC) << "PPOCRv3 label_path: " << label_path_;

    if (data.find(key_enable_save) != data.end()) {
      int interval_ms = 500;
      if (data.find(key_save_interval_ms) != data.end()) {
        interval_ms = data[key_save_interval_ms].get<int>();
      }
      debug_saver_.Configure(data[key_enable_save].get<bool>(), interval_ms);
    }

    if (!label_path_.empty()) {
        label_list_ = PaddlePaddle::ReadDict(label_path_);
        label_list_.insert(this->label_list_.begin(), "#");  // blank
        label_list_.push_back(" ");  // space
    }

    return true;
  }  // Init

  int Execute(const std::vector<float*>& outputs, ModelLoader* model,
              const FrameInfoPtr& finfo, const std::shared_ptr<InferObject>& pobj) override {

    LOGD(POSTPROC) << "PPOCRv3 Execute";
    auto start_time = std::chrono::steady_clock::now();
    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }

    int output_index = 0;
    const float* output = outputs[output_index];
    TensorShape output_shape = model->OutputShape(output_index);

    if (output_shape.ndims() != 3) {
        LOGE(POSTPROC) << "PPOCRv3 output shape " << output_shape;
        return -1;
    }
    if (output_shape.shape(0) != 1) {
        LOGE(POSTPROC) << "PPOCRv3 model shape must be batch size 1, but " << output_shape;
        return -1;
    }
    
    uint32_t rows = output_shape.shape(1);  // 40
    uint32_t cols = output_shape.shape(2);  // 6625

    std::vector<float> v(rows * cols);
    memcpy(v.data(), output, rows * cols * sizeof(float));

    // 5. CTC 贪婪解码
    std::string str_res;
    float score_sum = 0.0f;
    int count = 0;
    int last_index = 0;

    for (int j = 0; j < rows; ++j) {
        int offset = j * cols;
  
        const float* row_start = output + offset;
        const float* row_end = row_start + cols;

        // 使用 std::max_element 和 std::distance 找到最大值的索引
        auto max_it = std::max_element(row_start, row_end);
        int max_idx = static_cast<int>(std::distance(row_start, max_it));
        float max_value = *max_it;

        if (max_idx > 0 && !(j > 0 && max_idx == last_index)) {
            score_sum += max_value;
            ++count;
            if (max_idx < static_cast<int>(label_list_.size())) {
                str_res += label_list_[max_idx];
            } else {
                LOGE(POSTPROC) << "max_idx out of range: " << max_idx
                               << ", label_list size: " << label_list_.size();
            }
        }
        last_index = max_idx;
    }

    if (count == 0) {
        LOGW(POSTPROC) << "no valid character decoded";
        pobj->AddAttribute(attribute_keys::key_content, InferAttr());
        return 0;
    }
    float score = score_sum / count;

    InferAttr data_cl;
    data_cl.id = 0;
    data_cl.score = score;
    data_cl.value = 0;
    data_cl.name = str_res;
    pobj->AddAttribute(attribute_keys::key_content, data_cl);

#ifdef VSTREAM_UNIT_TEST
    DataFramePtr frame = finfo->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
        LOGE(POSTPROC) << "PPOCRv3: DataFrame is null";
        return -1;
    }
  
    if (debug_saver_.enable())  {
      cv::Mat img = frame->GetImage().clone();
      cv::Rect bbox_roi(pobj->bbox.x, pobj->bbox.y, pobj->bbox.w, pobj->bbox.h);
      cv::Rect safe_roi = bbox_roi & cv::Rect(0, 0, img.cols, img.rows);
      if (safe_roi.width > 0 && safe_roi.height > 0) {
        debug_saver_.MaybeSave("post_ocr_rec_crop", img, str_res,
            [safe_roi](cv::Mat& canvas) {
              canvas = canvas(safe_roi).clone();
            });
      }
    }
#endif

    double dr_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start_time).count();
    LOGD(POSTPROC) << "PPOCRv3: " << dr_ms << " ms, result: " << str_res;

    return 0;
  }

 private:
  std::string label_path_;
  std::vector<std::string> label_list_;
  std::string model_name_;

 private:
  cnstream::DebugImageSaver debug_saver_;

  DECLARE_REFLEX_OBJECT_EX(Post_PPOCRv3_rec_Obj, cnstream::ObjPostproc);
};  // class Post_PPOCRv3_rec_Obj

IMPLEMENT_REFLEX_OBJECT_EX(Post_PPOCRv3_rec_Obj, cnstream::ObjPostproc);

}  // namespace cnstream