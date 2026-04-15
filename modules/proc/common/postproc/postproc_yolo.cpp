

#include "postproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>


using json = nlohmann::json;

namespace cnstream {

static const std::string key_config_file = "config_file";
static const std::string key_threshold_map = "threshold";  // 配置文件中的阈值字典对应的


static float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop,
                                float bright, float bbottom) {
    float cleft = std::max(aleft, bleft);
    float ctop = std::max(atop, btop);
    float cright = std::min(aright, bright);
    float cbottom = std::min(abottom, bbottom);

    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = std::max(0.0f, aright - aleft) * std::max(0.0f, abottom - atop);
    float b_area = std::max(0.0f, bright - bleft) * std::max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

/**
 * @brief 快速非极大值抑制
 * @note 输出结果可能无法达到 max_boxes, 如果有较多置信度较高且重叠过多的框, 输出结果不稳定
 * @param threshold IOU阈值
 */
void fast_nms(ObjsVec& objs, int max_boxes, float threshold) {
  int count = std::min(static_cast<int>(objs.size()), max_boxes);
  if (count <= 0) return;

  // 按置信度降序排序
  std::partial_sort(objs.begin(), 
                    objs.begin() + count, 
                    objs.end(),
      [](const auto& a, const auto& b) { return a->score > b->score; });

  std::vector<bool> suppressed(count, false);

  for (int i = 0; i < count; ++i) {
    if (suppressed[i]) continue;
    
    const auto& cur_obj = objs[i];
    const int cur_class_id = cur_obj->id;

    for (int j = i + 1; j < count; ++j) {
      if (suppressed[j]) continue;
      
      const auto& item_obj = objs[j];
      if (item_obj->id != cur_class_id) continue;

      float iou = box_iou(
          cur_obj->bbox.x, 
          cur_obj->bbox.y,
          cur_obj->bbox.x + cur_obj->bbox.w,   // right
          cur_obj->bbox.y + cur_obj->bbox.h,   // bottom
          item_obj->bbox.x,
          item_obj->bbox.y,
          item_obj->bbox.x + item_obj->bbox.w,
          item_obj->bbox.y + item_obj->bbox.h
      );

      if (iou > threshold) {
          suppressed[j] = true;
      }
    }
  }
  // 将保留的元素移动到前面，然后截断
  size_t keep_idx = 0;
  for (int i = 0; i < count; ++i) {
      if (!suppressed[i]) {  // need to save
          objs[keep_idx++] = objs[i];
      }
  }
  objs.resize(keep_idx);
}


class Yolov8Postproc: public Postproc {

 public:
  /**
   * @brief 解析配置 json 得到后处理参数，加载阈值字典
   * @param params 后处理参数 custom_postproc_params
   */
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(key_config_file) != params_.end()) {
      config_file_ = params_[key_config_file];
    } else {
      return false;
    }
    std::ifstream file(config_file_);
    if (!file.is_open()) {
      LOGE(Postproc) << "Init Could not open file " << config_file_;
      return false;
    }
    nlohmann::ordered_json data = nlohmann::ordered_json::parse(file);
    if (!data.is_object()) {
      LOGE(Postproc) << "Init config file must be object type.";
      return false;
    }
    // 解析阈值字典
    if (!data.contains(key_threshold_map)) {
      LOGE(Postproc) << "Threshold must be in config file.";
      return false;
    } 
    if (!data[key_threshold_map].is_object()) {
      LOGE(Postproc) << "Threshold must be object type.";
      return false;
    }
    for (auto it = data[key_threshold_map].begin(); it != data[key_threshold_map].end(); ++it) {
      const std::string& key = it.key();
      threshold_map_[std::stoi(key)] = it.value().get<float>();
    }
    return true;
  }

  /**
   * @param cpu_outputs 相对于 D2H 是输出，位于 CPU 上
   * size == output tensor num
   */
  int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) {

    LOGI(Postproc) << "Postproc Execute for data: " << package->GetStreamId() << ", timestamp: " << package->GetTimestamp();
 
    DataFramePtr frame = package->collection.Get<DataFramePtr>(cnstream::kDataFrameTag);
    const int img_w = frame->GetWidth();
    const int img_h = frame->GetHeight();

    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }
    int input_index = model->get_input_ordered_index();
    int output_index = model->get_output_ordered_index();

    const int input_w = model->get_width();
    const int input_h = model->get_height();

    // 与前处理使用的缩放比例一致
    float img_scale = std::min((float)(input_w) / (float)(img_w), (float)(input_h) / (float)(img_h));
    
    // 不要超过左上角顶点
    float pad_w = std::max(0, int(input_w - img_w * img_scale) / 2);
    float pad_h = std::max(0, int(input_h - img_h * img_scale) / 2);
    
    float* output = cpu_outputs[output_index];

    InferObjsPtr objs_holder = package->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
    ObjsVec &objs = objs_holder->objs_;

    TensorShape output_shape = model->OutputShape(output_index);

    int num_bboxes = output_shape.shape(2);  // 8000
    int output_cdim = output_shape.shape(1);  // 84
    const int num_classes = output_cdim - 4;  // 80

    const int stride = num_bboxes;  // 每个属性之间的步长

    std::vector<float> bboxes;
    bboxes.reserve(num_bboxes);

    for (int position = 0; position < num_bboxes; ++position) {
        
      float cx = output[0 * stride + position];
      float cy = output[1 * stride + position];
      float width = output[2 * stride + position];
      float height = output[3 * stride + position];

      // [4, ...., 83] class scores
      // get max confidence class
      const float* class_scores = output + 4 * stride;
      float confidence = class_scores[position];  // 第0类
      int label = 0;
      for (int i = 0; i < num_classes; ++i) {
        if (class_scores[i] > confidence) {
          confidence = class_scores[i];
          label = i;
        }
      }

      // default threshold is zero
      float class_threshold = 0.0f;
      if (threshold_map_.find(label) != threshold_map_.end()) {
        class_threshold = threshold_map_[label];
      } 
      if (confidence < class_threshold) {
        continue;
      }
      float left = cx - width * 0.5f;
      float top = cy - height * 0.5f;
      float right = cx + width * 0.5f;
      float bottom = cy + height * 0.5f;

      // 相对于模型输入图的坐标，单位是 pixel
      left   = (left   - pad_w) / img_scale;
      top    = (top    - pad_h) / img_scale;
      right  = (right  - pad_w) / img_scale;
      bottom = (bottom - pad_h) / img_scale;

      left   = std::max(0.0f, std::min(left,   (float)img_w));
      top    = std::max(0.0f, std::min(top,    (float)img_h));
      right  = std::max(0.0f, std::min(right,  (float)img_w));
      bottom = std::max(0.0f, std::min(bottom, (float)img_h));

      auto obj = std::make_shared<InferObject>();
      obj->id = label;
      obj->score = confidence;

      obj->bbox.x = left;
      obj->bbox.y = top;
      obj->bbox.w = right - left;
      obj->bbox.h = bottom - top;
      obj->model_name = model_name_;

      std::lock_guard<std::mutex> objs_mutex(objs_holder->mutex_);
      objs.push_back(obj);
    }
    fast_nms(objs, max_boxes_num_, 0.5f);

#ifdef UNIT_TEST
    if (!has_save_frame_mat_) {
      cv::Mat img = frame->GetImage().clone();  // BGR
      for (auto& obj : objs) {
        float x = obj->bbox.x * img_w;
        float y = obj->bbox.y * img_h;
        float w = obj->bbox.w * img_w;
        float h = obj->bbox.h * img_h;
        cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
      }
      cv::imwrite(save_file_, img);
      has_save_frame_mat_ = true;
    }
#endif

    return 0;
  }

 private:
  const int max_boxes_num_ = 50;
  std::string model_name_;  ///< The name of the model.

#ifdef UNIT_TEST
  bool has_save_frame_mat_ = false;
  std::string save_file_ = "save_image/test_postproc_save.jpg";
#endif

 private:
  DECLARE_REFLEX_OBJECT_EX(Yolov8Postproc, cnstream::Postproc);
};  // class Yolov8Postproc

IMPLEMENT_REFLEX_OBJECT_EX(Yolov8Postproc, cnstream::Postproc);

}  // namespace cnstream
