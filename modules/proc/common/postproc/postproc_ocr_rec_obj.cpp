
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

#include "crnn_process.hpp"  // from PaddleOCR

namespace cnstream {

namespace postproc_ocr_rec_obj {

const std::string key_config_file = "config_file";
const std::string key_label_path = "label_path";

}  // namespace postproc_ocr_rec_obj

class Post_PPOCRv3_rec_Obj : public ObjPostproc {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(postproc_ocr_rec_obj::key_config_file) != params_.end()) {
      config_file_ = params_[postproc_ocr_rec_obj::key_config_file];
    } else {
      LOGE(POSTPROC) << "Init config_file must be in custom_postproc_params.";
      return false;
    }
    std::string config_file_path = "./";
    if (params_.find(CNS_JSON_DIR_PARAM_NAME) != params_.end()) {
      config_file_path = params_[CNS_JSON_DIR_PARAM_NAME];
    }
    config_file_ = GetPathRelativeToTheJSONFile(config_file_, config_file_path);

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
    if (data.find(postproc_ocr_rec_obj::key_label_path) == data.end()) {
      LOGE(POSTPROC) << "Init label_path must be in config file.";
      return false;
    }
    label_path_ = data[postproc_ocr_rec_obj::key_label_path].get<std::string>();
    LOGI(POSTPROC) << "PPOCRv3 label_path: " << label_path_;

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
        if (offset + cols > rows * cols) {
            break;  // 防止访问越界
        }

        int max_idx = static_cast<int>(PaddlePaddle::Argmax(
            &v[offset], &v[offset + cols]));

        if (max_idx < 0 || max_idx >= static_cast<int>(label_list_.size())) {
            LOGE(POSTPROC) << "PPOCRv3: max_idx " << max_idx
                        << " out of range [0, " << label_list_.size() << ")";
            continue;
        }
        float max_value = static_cast<float>(*std::max_element(
            &v[offset], &v[offset + cols]));

        // LOGD(POSTPROC) << "Step " << j << "; max_idx=" << max_idx 
        //     << "; max_value=" << max_value 
        //     << "; last_index=" << last_index;

        // CTC 规则：跳过 blank（索引 0），跳过连续重复字符
        if (max_idx > 0 && !(j > 0 && max_idx == last_index)) {
            score_sum += max_value;
            ++count;
            str_res += label_list_[max_idx];
        }
        last_index = max_idx;
    }
    if (count == 0) {
        LOGW(POSTPROC) << "no valid character decoded";
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
  
    if (enable_save_)  {
      cv::Mat img = frame->GetImage();
      std::lock_guard<std::mutex> lock(last_save_time_mutex_);
      auto now = std::chrono::steady_clock::now();
      if (save_duration_ms_ > 0) {
        if (last_save_time_.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_save_time_).count() >= save_duration_ms_) {

              cv::rectangle(img, cv::Rect(pobj->bbox.x, pobj->bbox.y, pobj->bbox.w, pobj->bbox.h),
                            cv::Scalar(0, 255, 0), 2);
              cv::putText(img, str_res,
                          cv::Point(pobj->bbox.x, std::max(pobj->bbox.y - 5, 15.0f)),
                          cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 255, 0), 2);
            
            auto sys_now = std::chrono::system_clock::now();
            auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()).count();
            std::string filename = "save/post_ocr_rec_" +  std::to_string(timestamp_ms) + ".jpg";
            cv::imwrite(filename, img);
            last_save_time_ = now;
        }
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
  bool enable_save_ = false;
  std::mutex last_save_time_mutex_;
  std::chrono::steady_clock::time_point last_save_time_;
  uint32_t save_duration_ms_ = 500;

  DECLARE_REFLEX_OBJECT_EX(Post_PPOCRv3_rec_Obj, cnstream::ObjPostproc);
};  // class Post_PPOCRv3_rec_Obj

IMPLEMENT_REFLEX_OBJECT_EX(Post_PPOCRv3_rec_Obj, cnstream::ObjPostproc);

}  // namespace cnstream