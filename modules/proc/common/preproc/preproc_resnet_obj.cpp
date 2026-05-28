#include "preproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include "affine_trans.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

namespace cnstream {

static const std::string key_config_file = "config_file";

/**
 * @brief ResNet CPU 前处理
 */
class Pre_Resnet_Obj : public ObjPreproc {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(key_config_file) != params_.end()) {
      config_file_ = params_[key_config_file];
    } else {
      LOGE(PREPROC) << "Init config_file must be in custom_postproc_params.";
      return false;
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
    return true;
  }

 public:
  /**
   * @brief cpu_outputs 作为前处理的输出，作为 D2H 的输入
   */
  virtual int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
                      const FrameInfoPtr& finfo, const std::shared_ptr<InferObject>& pobj) override {

    LOGI(PREPROC) << "Execute for data: " << finfo->GetStreamId() << ", timestamp: " << finfo->GetTimestamp();

    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }

    int channel = model->get_channel();
    if (channel != 3) {
        LOGE(PREPROC) << "model input shape not supported";
        return -1;
    }
    DataFramePtr frame = finfo->collection.Get<DataFramePtr>(cnstream::kDataFrameTag);
    cv::Mat src_img = frame->GetImage();  // BGR

    if (!pobj) {
        LOGE(PREPROC) << "pobj is nullptr";
        return -1;
    }
    int src_w = src_img.cols;
    int src_h = src_img.rows;

    int bx = std::max(0, static_cast<int>(pobj->bbox.x));
    int by = std::max(0, static_cast<int>(pobj->bbox.y));
    int bw = std::min(src_w - bx, static_cast<int>(pobj->bbox.w));
    int bh = std::min(src_h - by, static_cast<int>(pobj->bbox.h));

    if (bw <= 0 || bh <= 0) {
        LOGE(PREPROC) << "invalid bbox size: " << bw << "x" << bh;
        return -1;
    }

    cv::Mat img = src_img(cv::Rect(bx, by, bw, bh)).clone();

    int img_w = img.cols;
    int img_h = img.rows;

    int input_index = model->get_input_ordered_index();
    int input_h = model->get_height();  // 224
    int input_w = model->get_width();   // 224

    // Step 1: Resize(256) -- 短边缩放到 256，保持宽高比
    const int resize_size = 256;
    float scale;
    if (img_h < img_w) {
        scale = static_cast<float>(resize_size) / img_h;
    } else {
        scale = static_cast<float>(resize_size) / img_w;
    }
    int new_w = static_cast<int>(img_w * scale);
    int new_h = static_cast<int>(img_h * scale);

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(new_w, new_h), cv::INTER_LINEAR);

    // Step 2: CenterCrop(224) -- 中心裁剪到 input_h x input_w
    cv::Mat net_input_data(input_h, input_w, CV_8UC3, cv::Scalar(0, 0, 0));
    int top = (new_h - input_h) / 2;
    int left = (new_w - input_w) / 2;
    top = std::max(0, top);
    left = std::max(0, left);
    int roi_h = std::min(input_h, new_h - top);
    int roi_w = std::min(input_w, new_w - left);

    if (roi_h > 0 && roi_w > 0) {
        resize_img(cv::Rect(left, top, roi_w, roi_h)).copyTo(
            net_input_data(cv::Rect(0, 0, roi_w, roi_h)));
    }

    // Step 3: BGR -> RGB
    cv::cvtColor(net_input_data, net_input_data, cv::COLOR_BGR2RGB);

    // Step 4: HWC RGB -> CHW RGB + ToTensor + Normalize
    //   (x/255 - mean) / std = x * (1/(255*std)) + (-mean/std)
    //   在 uint8 -> float32 转换中一步完成，避免中间 float HWC Mat 和手动逐像素循环
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stddev[3] = {0.229f, 0.224f, 0.225f};

    std::vector<cv::Mat> channels(3);
    cv::split(net_input_data, channels);  // channels item: HW 

    float* cpu_output = cpu_outputs[input_index];

    // cpu_output: CHW RGB
    for (int c = 0; c < 3; c++) {
        cv::Mat float_ch;
        float alpha = 1.0f / (255.0f * stddev[c]);
        float beta = -mean[c] / stddev[c];
        channels[c].convertTo(float_ch, CV_32FC1, alpha, beta);
        memcpy(cpu_output + c * input_h * input_w, float_ch.ptr<float>(), input_h * input_w * sizeof(float));
    }

#ifdef VSTREAM_UNIT_TEST
    if (!has_save_frame_mat_) {
        save_float_image_chw_cpu(cpu_output, input_h, input_w, save_file_, ChannelsArrange::RGB, true);
        has_save_frame_mat_ = true;
    }
#endif
    return 0;
  }
private:
  std::string model_name_;

 private:
  bool has_save_frame_mat_ = false;
  std::string save_file_ = "save/test_preproc_save.jpg";

 private:
  DECLARE_REFLEX_OBJECT_EX(Pre_Resnet_Obj, cnstream::ObjPreproc);
};  // class Pre_Resnet_Obj

IMPLEMENT_REFLEX_OBJECT_EX(Pre_Resnet_Obj, cnstream::ObjPreproc);

}  // namespace cnstream