

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

namespace postproc_yolo {
  
const std::string key_config_file = "config_file";

const std::string key_classes = "classes";
const std::string key_name = "name";
const std::string key_threshold = "threshold";

const std::string key_max_boxes_num = "max_boxes_num";
const std::string key_nms_iou_threshold = "nms_iou_threshold";

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
 * @brief 快速非极大值抑制
 * @note 输出结果可能无法达到 max_boxes, 如果有较多置信度较高且重叠过多的框, 输出结果不稳定
 * @param threshold IOU阈值
 */
void fast_nms(ObjsVec& objs, int max_boxes, float threshold) {
  int count = std::min(static_cast<int>(objs.size()), max_boxes);
  if (count <= 1) return;

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
}  // namespace postproc_yolo


class Post_YOLOv8_CPU: public Postproc {

 public:
  /**
   * @brief 解析配置 json 得到后处理参数，加载阈值字典
   * @param params 后处理参数 custom_postproc_params
   */
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(postproc_yolo::key_config_file) != params_.end()) {
      config_file_ = params_[postproc_yolo::key_config_file];
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

    if (data.find(postproc_yolo::key_classes) != data.end()) {
      const auto& classes = data[postproc_yolo::key_classes];
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

    if (data.find(postproc_yolo::key_max_boxes_num) != data.end()) {
      max_boxes_num_ = data[postproc_yolo::key_max_boxes_num].get<int>();
    }
    if (data.find(postproc_yolo::key_nms_iou_threshold) != data.end()) {
      nms_iou_threshold_ = data[postproc_yolo::key_nms_iou_threshold].get<float>();
    }

    return true;
  }

  /**
   * @param cpu_outputs 相对于 D2H 是输出，位于 CPU 上
   * size == output tensor num
   */
  int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) {

    LOGD(POSTPROC) << "Execute for data: " << package->GetStreamId() << ", timestamp: " << package->GetTimestamp();
 
    DataFramePtr frame = package->collection.Get<DataFramePtr>(cnstream::kDataFrameTag);
    const int img_w = frame->GetWidth();
    const int img_h = frame->GetHeight();

    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }
    int output_index = 0;

    const int input_w = model->get_width();
    const int input_h = model->get_height();

    // 与前处理使用的缩放比例一致
    float img_scale = std::min((float)(input_w) / (float)(img_w), (float)(input_h) / (float)(img_h));
    
    // 不要超过左上角顶点
    float pad_w = std::max(0, int(input_w - img_w * img_scale) / 2);
    float pad_h = std::max(0, int(input_h - img_h * img_scale) / 2);
    
    const float* output = cpu_outputs[output_index];

    InferObjsPtr objs_holder = package->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
    ObjsVec &objs = objs_holder->objs_;

    TensorShape output_shape = model->OutputShape(output_index);

    int num_bboxes = output_shape.shape(2);  // 8400
    int output_cdim = output_shape.shape(1);  // 84（classes + bbox）
    const int num_classes = output_cdim - 4;  // 80

    const int stride = num_bboxes;  // 每个属性之间的步长

    std::vector<float> bboxes;
    bboxes.reserve(num_bboxes);

    for (int position = 0; position < num_bboxes; ++position) {
        
      float cx = output[0 * stride + position];
      float cy = output[1 * stride + position];
      float width = output[2 * stride + position];
      float height = output[3 * stride + position];

      float max_conf = output[4 * stride + position];
      int label = 0;
      for (int i = 0; i < num_classes; ++i) {
        float conf = output[(4+i) * stride + position];
        if (conf > max_conf) {
          max_conf = conf;
          label = i;
        }
      }

      // default threshold is zero
      float class_threshold = 0.0f;
      if (item_infos_.find(label) != item_infos_.end()) {
        class_threshold = item_infos_[label].threshold;
      } 
      if (max_conf < class_threshold) {
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
      obj->score = max_conf;

      // note: 相对原图的实际坐标
      obj->bbox.x = left;
      obj->bbox.y = top;
      obj->bbox.w = right - left;
      obj->bbox.h = bottom - top;
      obj->area = obj->bbox.w * obj->bbox.h;
      obj->model_name = model_name_;

      std::lock_guard<std::mutex> objs_mutex(objs_holder->mutex_);
      objs.push_back(obj);
    }
    postproc_yolo::fast_nms(objs, max_boxes_num_, 0.5f);

    return 0;
  }

 private:
  struct ItemInfo {
    std::string name;
    float threshold = 0;
  };
  std::map<int, ItemInfo> item_infos_;  // class_id -> item_info
  std::string model_name_;

  int max_boxes_num_ = 100;
  float nms_iou_threshold_ = 0.45f;
  
 private:
  DECLARE_REFLEX_OBJECT_EX(Post_YOLOv8_CPU, cnstream::Postproc);
};  // class Post_YOLOv8_CPU

IMPLEMENT_REFLEX_OBJECT_EX(Post_YOLOv8_CPU, cnstream::Postproc);

namespace postproc_yolo {

float box_iou_v2(float aleft, float atop, float aright, float abottom, float a_area,
                 float bleft, float btop, float bright, float bbottom, float b_area) {
    float cleft = std::max(aleft, bleft);
    float ctop = std::max(atop, btop);
    float cright = std::min(aright, bright);
    float cbottom = std::min(abottom, bbottom);
    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area <= 0.0f) return 0.0f;
    return c_area / (a_area + b_area - c_area);
}

/**
 * @brief 首先按照类别分组，进行单向遍历
 */
void fast_nms_class(ObjsVec& objs, int max_boxes, float threshold) {
    if (objs.empty()) return;
    int count = std::min(static_cast<int>(objs.size()), max_boxes);
    if (count <= 1) return;
    std::partial_sort(objs.begin(), objs.begin() + count, objs.end(),
                      [](const auto& a, const auto& b) { return a->score > b->score; });
    objs.resize(count);

    // 按类别分组，每组单独进行 NMS
    std::unordered_map<int, std::vector<int>> class_groups;
    for (int i = 0; i < count; ++i) {
        class_groups[objs[i]->id].push_back(i);
    }

    std::vector<bool> suppressed(count, false);
    for (auto& pair : class_groups) {
        const auto& indices = pair.second;
        // 每个类别内，由于已经按置信度降序，直接顺序抑制
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx_i = indices[i];
            if (suppressed[idx_i]) continue;
            const auto& obj_i = objs[idx_i];
            for (size_t j = i + 1; j < indices.size(); ++j) {
                int idx_j = indices[j];
                if (suppressed[idx_j]) continue;
                const auto& obj_j = objs[idx_j];
                float iou = box_iou_v2(obj_i->bbox.x, obj_i->bbox.y,
                                    obj_i->bbox.x + obj_i->bbox.w, obj_i->bbox.y + obj_i->bbox.h,
                                    obj_i->area,
                                    obj_j->bbox.x, obj_j->bbox.y,
                                    obj_j->bbox.x + obj_j->bbox.w, obj_j->bbox.y + obj_j->bbox.h,
                                    obj_j->area);
                if (iou > threshold) {
                    suppressed[idx_j] = true;
                }
            }
        }
    }
    // 将保留的框移动到前面并截断
    size_t keep_idx = 0;
    for (int i = 0; i < count; ++i) {
        if (!suppressed[i]) {
            objs[keep_idx++] = objs[i];
        }
    }
    objs.resize(keep_idx);
}
}  // namespace postproc_yolo


class Post_YOLOv8_CPU_v2: public Postproc {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(postproc_yolo::key_config_file) != params_.end()) {
      config_file_ = params_[postproc_yolo::key_config_file];
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

    if (data.find(postproc_yolo::key_classes) != data.end()) {
      const auto& classes = data[postproc_yolo::key_classes];
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

    if (data.find(postproc_yolo::key_max_boxes_num) != data.end()) {
      max_boxes_num_ = data[postproc_yolo::key_max_boxes_num].get<int>();
    }
    if (data.find(postproc_yolo::key_nms_iou_threshold) != data.end()) {
      nms_iou_threshold_ = data[postproc_yolo::key_nms_iou_threshold].get<float>();
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
    int num_bboxes = output_shape.shape(2);   // 8400
    int output_cdim = output_shape.shape(1);  // 84 (bbox4 + class80)
    const int num_classes = output_cdim - 4;  // 80
    const int stride = num_bboxes;

    // 局部容器，避免加锁
    ObjsVec local_objs;
    local_objs.reserve(1024);  // 预分配

    // 指针偏移优化
    const float* cx_ptr   = output;
    const float* cy_ptr   = output + stride;
    const float* w_ptr    = output + 2 * stride;
    const float* h_ptr    = output + 3 * stride;
    const float* score_ptr= output + 4 * stride;  // 指向第一个类的分数

    for (int pos = 0; pos < num_bboxes; ++pos) {
      float cx = cx_ptr[pos];
      float cy = cy_ptr[pos];
      float width = w_ptr[pos];
      float height = h_ptr[pos];

      // 寻找最大置信度类别
      float max_conf = score_ptr[pos];  // 第0类
      int label = 0;
      for (int c = 1; c < num_classes; ++c) {
        float conf = score_ptr[c * stride + pos];
        if (conf > max_conf) {
          max_conf = conf;
          label = c;
        }
      }
      float class_threshold = (label < static_cast<int>(item_infos_.size())) ? 
                              item_infos_[label].threshold : 0.0f;
      if (max_conf < class_threshold) continue;

      // 转换到原图坐标
      float left   = (cx - width * 0.5f - pad_w) / img_scale;
      float top    = (cy - height * 0.5f - pad_h) / img_scale;
      float right  = (cx + width * 0.5f - pad_w) / img_scale;
      float bottom = (cy + height * 0.5f - pad_h) / img_scale;

      left   = std::clamp(left,   0.0f, (float)img_w);
      top    = std::clamp(top,    0.0f, (float)img_h);
      right  = std::clamp(right,  0.0f, (float)img_w);
      bottom = std::clamp(bottom, 0.0f, (float)img_h);

      float w = right - left;
      float h = bottom - top;
      if (w <= 0.0f || h <= 0.0f) continue;

      auto obj = std::make_shared<InferObject>();
      obj->id = label;
      obj->score = max_conf;
      obj->bbox.x = left;
      obj->bbox.y = top;
      obj->bbox.w = w;
      obj->bbox.h = h;
      obj->area = w * h;
      obj->model_name = model_name_;
      local_objs.push_back(obj);
    }
    postproc_yolo::fast_nms(local_objs, max_boxes_num_, nms_iou_threshold_);
    {
      InferObjsPtr objs_holder = package->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
      std::lock_guard<std::mutex> lock(objs_holder->mutex_);
      ObjsVec& global_objs = objs_holder->objs_;
      global_objs.insert(global_objs.end(), local_objs.begin(), local_objs.end());
    }
    LOGU(POSTPROC) << "After NMS, size: " << local_objs.size();

#ifdef VSTREAM_UNIT_TEST
    if (enable_save_) {
      std::lock_guard<std::mutex> lock(last_save_time_mutex_);
      auto now = std::chrono::steady_clock::now();
      if (save_duration_ms_ > 0) {
        if (last_save_time_.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_save_time_).count() >= save_duration_ms_) {
            cv::Mat img = frame->GetImage().clone();
            for (auto& obj : local_objs) {
                cv::rectangle(img, cv::Rect(obj->bbox.x, obj->bbox.y, obj->bbox.w, obj->bbox.h),
                              cv::Scalar(0, 255, 0), 2);
            }
            auto sys_now = std::chrono::system_clock::now();
            auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sys_now.time_since_epoch()).count();
            std::string filename = "save/post_yolo_" + std::to_string(timestamp_ms) + ".jpg";
            cv::imwrite(filename, img);
            last_save_time_ = now;
        }
      }
    }
#endif

    return 0;
  }

 private:
  struct ItemInfo {
    std::string name;
    float threshold = 0.0f;
  };
  std::vector<ItemInfo> item_infos_;   // 下标即类别ID，O(1)访问
  std::string model_name_;

  int max_boxes_num_ = 100;
  float nms_iou_threshold_ = 0.45f;

private:
  bool enable_save_ = false;
  std::mutex last_save_time_mutex_;
  std::chrono::steady_clock::time_point last_save_time_;
  uint32_t save_duration_ms_ = 1000;

  DECLARE_REFLEX_OBJECT_EX(Post_YOLOv8_CPU_v2, cnstream::Postproc);
};  // class Post_YOLOv8_CPU_v2

IMPLEMENT_REFLEX_OBJECT_EX(Post_YOLOv8_CPU_v2, cnstream::Postproc);


/**
 * @brief YOLOv5 后处理类
 * @note 适用于标准 YOLOv5 ONNX 输出：shape [1, 25200, 5 + num_classes]
 */
class Post_YOLOv5_CPU: public Postproc {

 public:
  /**
   * @param params 后处理参数 custom_postproc_params
   */
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    if (params_.find(postproc_yolo::key_config_file) != params_.end()) {
      config_file_ = params_[postproc_yolo::key_config_file];
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

    if (data.find(postproc_yolo::key_classes) != data.end()) {
      const auto& classes = data[postproc_yolo::key_classes];
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
    if (data.find(postproc_yolo::key_max_boxes_num) != data.end()) {
      max_boxes_num_ = data[postproc_yolo::key_max_boxes_num].get<int>();
    }
    if (data.find(postproc_yolo::key_nms_iou_threshold) != data.end()) {
      nms_iou_threshold_ = data[postproc_yolo::key_nms_iou_threshold].get<float>();
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

    int num_classes = output_shape.shape(2) - 5;  // 类别数 = 总通道数 - 5
    int stride = 5 + num_classes;                 // 单个检测框的浮点数个数
    int box_num = output_shape.shape(1);          // 640x640 输入下通常为 25200

    LOGU(POSTPROC) << "YOLOv5 output_shape: " << output_shape
                   << ", num_classes: " << num_classes
                   << ", box_num: " << box_num;

    ObjsVec local_objs;
    local_objs.reserve(1024);

    for (int i = 0; i < box_num; ++i) {
      const float* row = output + i * stride;

      // 标准 YOLOv5 原始输出格式：
      // [center_x, center_y, width, height, obj_conf, cls0_conf, cls1_conf, ...]
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

      float class_threshold = 0.0f;
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

      local_objs.push_back(obj);
    }  // end for

    postproc_yolo::fast_nms(local_objs, max_boxes_num_, nms_iou_threshold_);
    {
      InferObjsPtr objs_holder = package->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
      std::lock_guard<std::mutex> lock(objs_holder->mutex_);
      ObjsVec& global_objs = objs_holder->objs_;
      global_objs.insert(global_objs.end(), local_objs.begin(), local_objs.end());
    }
    LOGU(POSTPROC) << "After NMS, size: " << local_objs.size();
    return 0;
  }

 private:
  struct ItemInfo {
    std::string name;
    float threshold = 0.0f;
  };
  std::map<int, ItemInfo> item_infos_;
  std::string model_name_;  ///< The name of the model.

  int max_boxes_num_ = 200;
  float nms_iou_threshold_ = 0.45f;
  
 private:
  DECLARE_REFLEX_OBJECT_EX(Post_YOLOv5_CPU, cnstream::Postproc);
};  // class Post_YOLOv5_CPU

IMPLEMENT_REFLEX_OBJECT_EX(Post_YOLOv5_CPU, cnstream::Postproc);

}  // namespace cnstream
