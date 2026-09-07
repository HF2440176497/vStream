
#include "preproc_resnet.hpp"

namespace cnstream {

IMPLEMENT_REFLEX_OBJECT_EX(Pre_Resnet_Obj, cnstream::ObjPreproc);


/**
 * @brief ResNet 前处理，图像级别
 */
class Pre_Resnet : public Preproc {

 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(key_config_file) != params_.end()) {
      config_file_ = params_[key_config_file];
    } else {
      LOGE(PREPROC) << "Init config_file must be in custom_preproc_params.";
      return false;
    }
    std::string dir_path;
    if (params_.find(CNS_JSON_DIR_PARAM_NAME) != params_.end()) {
      dir_path = params_[CNS_JSON_DIR_PARAM_NAME];
    }
    config_file_ = GetPathRelativeToTheJSONFile(config_file_, dir_path);
    if (config_file_.empty()) {
      LOGW(PREPROC) << "Init config_file is empty.";
      return true;
    }
    std::ifstream file(config_file_);
    if (!file.is_open()) {
      LOGE(PREPROC) << "Init Could not open file " << config_file_;
      return false;
    }
    nlohmann::ordered_json data = nlohmann::ordered_json::parse(file);
    if (!data.is_object()) {
      LOGE(PREPROC) << "Init config file must be object type.";
      return false;
    }
    if (data.find(key_enable_save) != data.end()) {
      debug_saver_.Configure(data[key_enable_save].get<bool>(), 1000);
    }
    return true;
  }

  int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) override {

    LOGI(PREPROC) << "Execute for data: " << package->GetStreamId() << ", timestamp: " << package->GetTimestamp();
    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }

    int channel = model->get_channel();
    if (channel != 3) {
        LOGE(PREPROC) << "model input shape not supported";
        return -1;
    }
    cv::Mat img = GetModelInputImage(package);  // BGR：优先派生图，回退原图

    int input_index = model->get_input_ordered_index();  // input tensor index
    int input_h = model->get_height();
    int input_w = model->get_width();

    // Step 1: 直接 Resize
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(input_w, input_h), cv::INTER_LINEAR);

    // Step 2: BGR -> RGB
    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

    // Step 3: 归一化
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stddev[3] = {0.229f, 0.224f, 0.225f};

    // Step 4: HWC RGB -> CHW + Normalize
    std::vector<cv::Mat> channels(3);
    cv::split(resize_img, channels);  // channels item: HW

    float* cpu_output = cpu_outputs[input_index];
    for (int c = 0; c < 3; c++) {
        cv::Mat float_ch;
        float alpha = 1.0f / (255.0f * stddev[c]);
        float beta = -mean[c] / stddev[c];
        channels[c].convertTo(float_ch, CV_32FC1, alpha, beta);
        memcpy(cpu_output + c * input_h * input_w, float_ch.ptr<float>(), input_h * input_w * sizeof(float));
    }
    return 0;
  }  // Execute

 private:
  std::string model_name_;

 private:
  cnstream::DebugImageSaver debug_saver_;

  DECLARE_REFLEX_OBJECT_EX(Pre_Resnet, cnstream::Preproc);
};
IMPLEMENT_REFLEX_OBJECT_EX(Pre_Resnet, cnstream::Preproc);


}  // namespace cnstream
