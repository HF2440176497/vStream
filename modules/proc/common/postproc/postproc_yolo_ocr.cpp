

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

static const std::string key_classes = "classes";
static const std::string key_name = "name";
static const std::string key_threshold = "threshold";
static const std::string key_interval = "interval";

static float box_iou(float aleft, float atop, float aright, float abottom, 
                    float bleft, float btop, float bright, float bbottom) {
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
 * @brief YOLOv5 后处理类
 * @note 此后处理假设输出已进行了 NMS
 */
class Post_YOLOv5_CPU_NoNMS_OCR: public Postproc {

 public:
  struct CharBox {
      float x;   // xmin
      float y;   // ymin
      float w;   // width
      float h;   // height
      float xmax() const { return x + w; }
      float ymax() const { return y + h; }
  };

 public:
  /**
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
    std::string config_file_path = "./";
    if (params_.find(CNS_JSON_DIR_PARAM_NAME) != params_.end()) {
      config_file_path = params_[CNS_JSON_DIR_PARAM_NAME];
    }
    config_file_ = GetPathRelativeToTheJSONFile(config_file_, config_file_path);

    LOGI(POSTPROC) << "model_name: " << model_name_ << ", post conf file: " << config_file_;
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
      for (auto it = classes.begin(); it != classes.end(); ++it) {
        const std::string& key = it.key();
        const nlohmann::ordered_json& value = it.value();
        if (!value.is_object()) {
          LOGE(POSTPROC) << "Invalid item format in conf file, key: " << key;
          return false;
        }
        ItemInfo info;
        info.name = value["name"].get<std::string>();
        info.threshold = value["threshold"].get<float>();
        item_infos_[std::stoi(key)] = info;
      }
    }
    if (data.find(key_interval) != data.end()) {
      interval_ = data[key_interval].get<float>();
    }
    return true;
  }

  int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) override {

    LOGD(POSTPROC) << "Execute for data: " << package->GetStreamId() << ", timestamp: " << package->GetTimestamp();
 
    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }

    DataFramePtr frame = package->collection.Get<DataFramePtr>(cnstream::kDataFrameTag);
    const int img_w = frame->GetWidth();
    const int img_h = frame->GetHeight();

    int output_index = 0;  // output tensor index

    const int input_w = model->get_width();
    const int input_h = model->get_height();
    float img_scale = std::min((float)(input_w) / (float)(img_w), (float)(input_h) / (float)(img_h));

    float pad_w = std::max(0, int(input_w - img_w * img_scale) / 2);
    float pad_h = std::max(0, int(input_h - img_h * img_scale) / 2);
    
    const float* output = cpu_outputs[output_index];
    TensorShape output_shape = model->OutputShape(output_index);

    InferObjsPtr objs_holder = package->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
    ObjsVec &objs = objs_holder->objs_;

    LOGU(POSTPROC) << "YOLOv5 NoNMS output_shape: " << output_shape;

    int stride = 7;

    for (int i = 0; i < max_boxes_num_; ++i) {
      if (int(output[i * stride + stride - 1]) == 0) {  // flag
        break;
      }

      const float* bbox = output + i * stride;
      int detect_class =int(output[i*stride+5]);
      float score = output[i*stride+4];

      float class_threshold = 0.0f;
      if (item_infos_.find(detect_class) != item_infos_.end()) {
        class_threshold = item_infos_[detect_class].threshold;
      }

      if (score < class_threshold) {
        continue;
      }

      float left = bbox[0];
      float top = bbox[1];
      float right = bbox[2];
      float bottom = bbox[3];

      left   = (left   - pad_w) / img_scale;
      top    = (top    - pad_h) / img_scale;
      right  = (right  - pad_w) / img_scale;
      bottom = (bottom - pad_h) / img_scale;

      left   = std::max(0.0f, std::min(left,   (float)img_w));  // 先限制右边界，再左边界
      top    = std::max(0.0f, std::min(top,    (float)img_h));
      right  = std::max(0.0f, std::min(right,  (float)img_w));
      bottom = std::max(0.0f, std::min(bottom, (float)img_h));

      auto obj = std::make_shared<InferObject>();
      obj->id = detect_class;
      obj->score = score;

      obj->bbox.x = left;
      obj->bbox.y = top;
      obj->bbox.w = right - left;
      obj->bbox.h = bottom - top;
      obj->model_name = model_name_;
      cnstream::SetInferObjType(obj, cnstream::InferObjType::kOriginal);

      {
        std::lock_guard<std::mutex> objs_mutex(objs_holder->mutex_);
        objs.push_back(obj);
      }
    }  // end for 

    std::vector<CharBox> boxes;
    boxes.reserve(objs.size());
    for (const auto& obj : objs) {
      boxes.push_back({obj->bbox.x, obj->bbox.y, obj->bbox.w, obj->bbox.h});
    }

    std::vector<std::vector<float>> results_merge;
    results_merge.reserve(boxes.size());
    if (boxes.size() <= 1) {
      // 0 个或 1 个框，无需合并
      for (const auto& b : boxes) {
        results_merge.push_back({b.x, b.y, b.w, b.h});
      }
    } else {
      // 按字符框的 水平坐标 从左到右排序
      std::sort(boxes.begin(), boxes.end(),
                [](const CharBox& a, const CharBox& b) { return a.x < b.x; });

      size_t start = 0;
      while (start < boxes.size()) {
          size_t end = start + 1;
          // 向右扩展窗口，直到水平间距超过阈值
          // 小于 interval_ 间距，合并为一个字符框
          while (end < boxes.size()) {
            const CharBox& prev = boxes[end - 1];
            const CharBox& curr = boxes[end];
            float horizontal_dis = curr.x - prev.xmax();
            if (horizontal_dis > interval_) break;
            ++end;
          }

          // 4. 计算 [start, end) 区间内所有框的外接矩形
          float min_x = boxes[start].x;
          float min_y = boxes[start].y;
          float max_x = boxes[start].xmax();
          float max_y = boxes[start].ymax();

          // 这时 end 定位在第一个不合并的字符框，因此 index 到 end-1
          for (size_t n = start + 1; n < end; ++n) {
            min_x = std::min(min_x, boxes[n].x);
            min_y = std::min(min_y, boxes[n].y);
            max_x = std::max(max_x, boxes[n].xmax());
            max_y = std::max(max_y, boxes[n].ymax());
          }

          float merged_w = max_x - min_x;
          float merged_h = max_y - min_y;

          // 5. 过滤掉面积为零或负数的非法框
          if (merged_w > 0 && merged_h > 0) {
            results_merge.push_back({min_x, min_y, merged_w, merged_h});
          }
          start = end;

        }  // end while (start < boxes.size())

    }

    // 将合并后的字符行框也加入 objs，通过 type 与原始 YOLO 框区分
    for (const auto& r : results_merge) {
      auto merged_obj = std::make_shared<InferObject>();
      merged_obj->id = 0;  // 合并框类别，可按需调整
      merged_obj->score = 1.0f;
      merged_obj->bbox.x = r[0];
      merged_obj->bbox.y = r[1];
      merged_obj->bbox.w = r[2];
      merged_obj->bbox.h = r[3];
      merged_obj->model_name = model_name_;
      cnstream::SetInferObjType(merged_obj, cnstream::InferObjType::kMerged);
      {
        std::lock_guard<std::mutex> objs_mutex(objs_holder->mutex_);
        objs.push_back(merged_obj);
      }
    }

    return 0;
  }

 private:
  struct ItemInfo {
    std::string name;
    float threshold = 0.0f;
  };
  std::map<int, ItemInfo> item_infos_;
  float interval_ = 50;  // pixel
  const int max_boxes_num_ = 100;
  std::string model_name_;  ///< The name of the model.

  bool has_save_frame_mat_ = false;
  std::string save_file_ = "save/test_postproc_save.jpg";

 private:
  DECLARE_REFLEX_OBJECT_EX(Post_YOLOv5_CPU_NoNMS_OCR, cnstream::Postproc);
};  // class Post_YOLOv5_CPU_NoNMS_OCR

IMPLEMENT_REFLEX_OBJECT_EX(Post_YOLOv5_CPU_NoNMS_OCR, cnstream::Postproc);

}  // namespace cnstream