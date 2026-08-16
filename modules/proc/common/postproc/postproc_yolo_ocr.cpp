#include "postproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"
#include "proc/common/debug_image_saver.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>


using json = nlohmann::json;

namespace cnstream {

namespace {

inline constexpr const char* key_config_file = "config_file";

inline constexpr const char* key_classes = "classes";
inline constexpr const char* key_name = "name";
inline constexpr const char* key_threshold = "threshold";

inline constexpr const char* key_merge_interval = "merge_interval";
inline constexpr const char* key_max_boxes_num = "max_boxes_num";
inline constexpr const char* key_nms_iou_threshold = "nms_iou_threshold";
inline constexpr const char* key_enable_save = "enable_save";
inline constexpr const char* key_merge_direction = "merge_direction";

}  // namespace

namespace {

float box_iou(float aleft, float atop, float aright, float abottom,
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
 * @brief 快速非极大值抑制（复用自 YOLOv8 后处理逻辑）
 * @note 按类别分别进行 NMS，只保留前 max_boxes 个高分框参与计算
 * @param objs 待处理检测框（会被原地修改）
 * @param max_boxes 最多参与 NMS 的框数量
 * @param threshold IOU 阈值
 */
void fast_nms(ObjsVec& objs, int max_boxes, float threshold) {
  int count = std::min(static_cast<int>(objs.size()), max_boxes);
  if (count <= 1) return;

  // 按置信度降序排序，只取前 count 个
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
      // 仅对同类框做 NMS（与 YOLOv8 逻辑保持一致）
      if (item_obj->id != cur_class_id) continue;

      float iou = box_iou(
          cur_obj->bbox.x,
          cur_obj->bbox.y,
          cur_obj->bbox.x + cur_obj->bbox.w,
          cur_obj->bbox.y + cur_obj->bbox.h,
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
      if (!suppressed[i]) {
          objs[keep_idx++] = objs[i];
      }
  }
  objs.resize(keep_idx);
}
}  // namespace

/**
 * @brief YOLOv5 后处理类
 * @note 适用于标准 YOLOv5 ONNX 输出：shape [1, 25200, 5 + num_classes]
 */
class Post_YOLOv5_CPU_OCR: public Postproc {

 public:
  struct CharBox {
      float x;   // xmin
      float y;   // ymin
      float w;   // width
      float h;   // height
      float xmax() const { return x + w; }
      float ymax() const { return y + h; }
  };

  // 合并方向：水平（从左到右）或竖直（从上到下）
  enum class MergeDirection { Horizontal = 0, Vertical = 1 };

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
    std::string dir_path;
    if (params_.find(CNS_JSON_DIR_PARAM_NAME) != params_.end()) {
      dir_path = params_[CNS_JSON_DIR_PARAM_NAME];
    }
    config_file_ = GetPathRelativeToTheJSONFile(config_file_, dir_path);

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
    if (data.find(key_merge_interval) != data.end()) {
      interval_ = data[key_merge_interval].get<float>();
    }
    if (data.find(key_max_boxes_num) != data.end()) {
      max_boxes_num_ = data[key_max_boxes_num].get<int>();
    }
    if (data.find(key_nms_iou_threshold) != data.end()) {
      nms_iou_threshold_ = data[key_nms_iou_threshold].get<float>();
    }

    if (data.find(key_enable_save) != data.end()) {
      debug_saver_.Configure(
          data[key_enable_save].get<bool>(), 500);
    }
    if (data.find(key_merge_direction) != data.end()) {
      std::string dir_str = data[key_merge_direction].get<std::string>();
      // 大小写不敏感比较
      std::transform(dir_str.begin(), dir_str.end(), dir_str.begin(), ::tolower);
      if (dir_str == "horizontal" || dir_str == "h") {
        merge_direction_ = MergeDirection::Horizontal;
      } else if (dir_str == "vertical" || dir_str == "v") {
        merge_direction_ = MergeDirection::Vertical;
      } else {
        LOGE(POSTPROC) << "Invalid merge_direction value: " << dir_str
                       << ", expected 'horizontal' or 'vertical'. Defaulting to 'horizontal'.";
        merge_direction_ = MergeDirection::Horizontal;
      }
    }
    LOGI(POSTPROC) << "merge_direction: "
                   << (merge_direction_ == MergeDirection::Horizontal ? "horizontal" : "vertical")
                   << ", interval: " << interval_;
    LOGI(POSTPROC) << "item_infos_ size = " << item_infos_.size();
    for (const auto& kv : item_infos_) {
      LOGI(POSTPROC) << "  class " << kv.first
                    << " name=" << kv.second.name
                    << " threshold=" << kv.second.threshold;
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

    const int input_w = model->get_width();
    const int input_h = model->get_height();

    float img_scale = std::min((float)input_w / img_w, (float)input_h / img_h);
    float pad_w = std::max(0.0f, (input_w - img_w * img_scale) / 2.0f);
    float pad_h = std::max(0.0f, (input_h - img_h * img_scale) / 2.0f);

    int output_index = 0;
    const float* output = cpu_outputs[output_index];
    TensorShape output_shape = model->OutputShape(output_index);

    int num_classes = output_shape.shape(2) - 5;  // 对于 2 类模型为 2
    int stride = 5 + num_classes;                 // 对于 2 类模型为 7
    int box_num = output_shape.shape(1);          // 640x640 输入下为 25200

    // 局部容器，避免频繁加锁
    ObjsVec local_objs;
    local_objs.reserve(1024);

    for (int i = 0; i < box_num; ++i) {
      const float* row = output + i * stride;

      // 标准 YOLOv5 原始输出格式（以 2 类为例）：
      // [center_x, center_y, width, height, obj_conf, cls0_conf, cls1_conf]
      float cx = row[0];
      float cy = row[1];
      float bw = row[2];
      float bh = row[3];
      float obj_conf = row[4];

      // 找最大类别置信度及其索引
      int detect_class = 0;
      float cls_conf = row[5];
      for (int c = 1; c < num_classes; ++c) {
        if (row[5 + c] > cls_conf) {
          cls_conf = row[5 + c];
          detect_class = c;
        }
      }

      float score = obj_conf * cls_conf;

      float class_threshold = 1.0f;
      if (item_infos_.find(detect_class) != item_infos_.end()) {
        class_threshold = item_infos_[detect_class].threshold;
      }

      if (score < class_threshold) {
        continue;
      }

      float left   = cx - bw * 0.5f;
      float top    = cy - bh * 0.5f;
      float right  = cx + bw * 0.5f;
      float bottom = cy + bh * 0.5f;

      left   = (left   - pad_w) / img_scale;
      top    = (top    - pad_h) / img_scale;
      right  = (right  - pad_w) / img_scale;
      bottom = (bottom - pad_h) / img_scale;

      left   = std::clamp(left,   0.0f, (float)img_w);
      top    = std::clamp(top,    0.0f, (float)img_h);
      right  = std::clamp(right,  0.0f, (float)img_w);
      bottom = std::clamp(bottom, 0.0f, (float)img_h);

      float w = right - left;
      float h = bottom - top;
      if (w <= 0.0f || h <= 0.0f) continue;

      auto obj = std::make_shared<InferObject>();
      obj->id = detect_class;
      obj->score = score;
      obj->bbox.x = left;
      obj->bbox.y = top;
      obj->bbox.w = w;
      obj->bbox.h = h;
      obj->area = w * h;
      obj->model_name = model_name_;
      cnstream::SetInferObjType(obj, cnstream::InferObjType::kOriginal);

      local_objs.push_back(obj);
    }  // end for

    // LOGU(POSTPROC) << "YOLOv5_CPU_OCR candidates: " << local_objs.size();
    fast_nms(local_objs, max_boxes_num_, nms_iou_threshold_);

    {
      InferObjsPtr objs_holder = package->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
      std::lock_guard<std::mutex> lock(objs_holder->mutex_);
      ObjsVec& global_objs = objs_holder->objs_;
      global_objs.insert(global_objs.end(), local_objs.begin(), local_objs.end());
    }

    LOGU(POSTPROC) << "YOLOv5_CPU_OCR after NMS: " << local_objs.size();
    // for (auto& obj : local_objs) {
    //   LOGU(POSTPROC) << "obj: " << obj->id << ", score: " << obj->score << ", bbox: " << obj->bbox.x << ", " << obj->bbox.y << ", " << obj->bbox.w << ", " << obj->bbox.h;
    // }

    std::vector<CharBox> boxes;
    boxes.reserve(local_objs.size());
    for (const auto& obj : local_objs) {
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
      // 按合并方向排序：水平方向按 x（左到右），竖直方向按 y（上到下）
      const bool horizontal = (merge_direction_ == MergeDirection::Horizontal);
      std::sort(boxes.begin(), boxes.end(),
                [horizontal](const CharBox& a, const CharBox& b) {
                  return horizontal ? (a.x < b.x) : (a.y < b.y);
                });

      size_t start = 0;
      while (start < boxes.size()) {
        size_t end = start + 1;

        // 外接矩形累加器
        float min_x = boxes[start].x;
        float min_y = boxes[start].y;
        float max_x = boxes[start].xmax();
        float max_y = boxes[start].ymax();

        // 当前分组沿合并轴的最远边缘
        float group_far = horizontal ? max_x : max_y;

        // 扩展窗口，同时累加外接矩形
        while (end < boxes.size()) {
          const CharBox& curr = boxes[end];
          float gap = horizontal ? (curr.x - group_far) : (curr.y - group_far);
          if (gap > interval_) break;

          float curr_far = horizontal ? curr.xmax() : curr.ymax();
          group_far = std::max(group_far, curr_far);

          min_x = std::min(min_x, curr.x);
          min_y = std::min(min_y, curr.y);
          max_x = std::max(max_x, curr.xmax());
          max_y = std::max(max_y, curr.ymax());
          ++end;
        }

        float merged_w = max_x - min_x;
        float merged_h = max_y - min_y;
        if (merged_w > 0 && merged_h > 0) {
          results_merge.push_back({min_x, min_y, merged_w, merged_h});
        }
        start = end;
      }  // end while (start < boxes.size())
    }

    LOGU(POSTPROC) << "YOLOv5_CPU_OCR results_size: " << results_merge.size();

    // 将合并后的字符行框也加入 objs，通过 type 与原始 YOLO 框区分
    {
      InferObjsPtr objs_holder = package->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
      std::lock_guard<std::mutex> lock(objs_holder->mutex_);
      ObjsVec& global_objs = objs_holder->objs_;
      for (const auto& r : results_merge) {
        auto merged_obj = std::make_shared<InferObject>();
        merged_obj->id = 0;
        merged_obj->score = 1.0f;
        merged_obj->bbox.x = r[0];
        merged_obj->bbox.y = r[1];
        merged_obj->bbox.w = r[2];
        merged_obj->bbox.h = r[3];
        merged_obj->model_name = model_name_;
        cnstream::SetInferObjType(merged_obj, cnstream::InferObjType::kMerged);
        global_objs.push_back(merged_obj);
      }
    }

#ifdef VSTREAM_UNIT_TEST
    if (debug_saver_.enable()) {
      cv::Mat img = frame->GetImage().clone();
      debug_saver_.MaybeSave("post_yolo_ocr", img, "",
          [&boxes, &results_merge](cv::Mat& canvas) {
            // 蓝色画原始字符框 (boxes)
            for (const auto& b : boxes) {
              cv::rectangle(canvas, cv::Rect(b.x, b.y, b.w, b.h), cv::Scalar(255, 0, 0), 2);
            }
            // 绿色画合并后的文本行框 (results_merge)
            for (const auto& r : results_merge) {
              cv::rectangle(canvas, cv::Rect(r[0], r[1], r[2], r[3]), cv::Scalar(0, 255, 0), 2);
            }
          });
    }
#endif

    return 0;
  }

 private:
  struct ItemInfo {
    std::string name;
    float threshold = 0.0f;
  };
  std::map<int, ItemInfo> item_infos_;
  float interval_ = 50;  // pixel
  MergeDirection merge_direction_ = MergeDirection::Horizontal;

  int max_boxes_num_ = 200;
  float nms_iou_threshold_ = 0.45f;
  std::string model_name_;  ///< The name of the model.

 private:
  cnstream::DebugImageSaver debug_saver_;

  DECLARE_REFLEX_OBJECT_EX(Post_YOLOv5_CPU_OCR, cnstream::Postproc);
};  // class Post_YOLOv5_CPU_OCR

IMPLEMENT_REFLEX_OBJECT_EX(Post_YOLOv5_CPU_OCR, cnstream::Postproc);

}  // namespace cnstream
