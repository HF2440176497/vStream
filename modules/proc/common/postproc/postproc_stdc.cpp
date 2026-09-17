

#include "postproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"
#include "proc/common/debug_image_saver.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>


using json = nlohmann::json;

namespace cnstream {

namespace {

inline constexpr const char* key_config_file = "config_file";

// 分割图在 image_dict 中的 key
inline constexpr char kSegImageKey[] = "stdc_image";

}  // namespace


class Post_STDC_CPU: public Postproc {

 public:
  /**
   * @brief 解析配置 json 得到后处理参数
   * @param params 后处理参数 custom_postproc_params
   */
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

    LOGI(POSTPROC) << "Init with post conf file: " << config_file_;
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

    return true;
  }

  /**
   * @brief STDC 分割后处理：逐像素 argmax 得到 0/1/2 灰度分割图，还原到原图尺寸
   * @note 分割图通过 kCustomImagesTag 传递，最终经 ConvertFrameInfo 进入
   *       s_output_data.image_dict["stdc_image"]，供外接收
   */
  int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) override {

    LOGD(POSTPROC) << "Execute for data: " << package->GetStreamId() << ", timestamp: " << package->GetTimestamp();

    DataFramePtr frame = package->collection.Get<DataFramePtr>(cnstream::kDataFrameTag);
    const int img_w = frame->GetWidth();
    const int img_h = frame->GetHeight();

    int output_index = 0;
    const float* output = cpu_outputs[output_index];
    TensorShape output_shape = model->OutputShape(output_index);

    // ONNX 输出 NCHW：[1, num_classes, mask_h, mask_w]
    const int num_classes = output_shape.shape(1);
    const int mask_h = output_shape.shape(2);
    const int mask_w = output_shape.shape(3);
    if (num_classes <= 0 || mask_h <= 0 || mask_w <= 0) {
      LOGE(POSTPROC) << "Invalid output shape: " << output_shape;
      return -1;
    }

    // 逐像素 argmax -> CV_8UC1（灰度值 = 类别 id 0/1/2）
    const int plane = mask_h * mask_w;
    cv::Mat mask(mask_h, mask_w, CV_8UC1);
    cv::parallel_for_(cv::Range(0, mask_h), [&](const cv::Range& range) {
      for (int y = range.start; y < range.end; ++y) {
        uint8_t* dst_row = mask.ptr<uint8_t>(y);
        const int row_offset = y * mask_w;
        for (int x = 0; x < mask_w; ++x) {
          const int idx = row_offset + x;
          int best_c = 0;
          float best_v = output[idx];
          for (int c = 1; c < num_classes; ++c) {
            const float v = output[c * plane + idx];
            if (v > best_v) {
              best_v = v;
              best_c = c;
            }
          }
          dst_row[x] = static_cast<uint8_t>(best_c);
        }
      }
    });

    // 还原到原图尺寸 最近邻避免引入新灰度值
    if (mask_h != img_h || mask_w != img_w) {
      cv::resize(mask, mask, cv::Size(img_w, img_h), 0, 0, cv::INTER_NEAREST);
    }

    // 写入 kCustomImagesTag，ConvertFrameInfo 会将其拷贝进 s_output_data.image_dict
    if (!package->collection.HasValue(cnstream::kCustomImagesTag)) {
      package->collection.AddIfNotExists(cnstream::kCustomImagesTag,
          std::make_shared<std::map<std::string, cv::Mat>>());
    }
    auto custom_images = package->collection.Get<CustomImagesPtr>(cnstream::kCustomImagesTag);
    (*custom_images)[kSegImageKey] = mask;

    return 0;
  }

 private:
  DECLARE_REFLEX_OBJECT_EX(Post_STDC_CPU, cnstream::Postproc);
};  // class Post_STDC_CPU

IMPLEMENT_REFLEX_OBJECT_EX(Post_STDC_CPU, cnstream::Postproc);



}  // namespace cnstream
