
#ifndef MODULE_SINK_DATA_COMMON_HPP
#define MODULE_SINK_DATA_COMMON_HPP


#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace cnstream {

enum class output_result : int {
  RESULT_OK = 0,
  RESULT_UNKNOWN_ERROR = -1,
  RESULT_TIMEOUT = -2,
  RESULT_NO_INFERENCE = -3,
  RESULT_SKIPPED = -4,
  RESULT_INFER_FAILED = -5,
};

namespace output_constants {

inline const std::string key_result = "result";
inline const std::string key_timestamp = "timestamp";
inline const std::string key_objects = "objects";
inline const std::string key_objects_json = "objects_json";

inline const std::string key_image_dict = "image_dict";
inline const std::string key_original_image = "original_image";

}  // namespace output_constants

/**
 * @brief 分类信息结构体（输出层独立定义，不与框架 InferObjectInfo 耦合）
 */
struct s_class_infos {
  int id = -1;                // 分类序号
  std::string model_name;     // 模型名
  std::string name;           // 分类名称
  float score = 0;            // 得分
  float value = 0;            // 归一化得分
};

inline std::ostream& operator<<(std::ostream& os, const s_class_infos& info) {
  os << "{id=" << info.id
     << ", model_name=" << info.model_name
     << ", name=" << info.name
     << ", score=" << info.score
     << ", value=" << info.value << "}";
  return os;
}

/**
 * @brief 属性信息结构体（承接 InferAttr，输出层独立定义）
 */
struct s_attr_info {
  int id = -1;          // 属性序号
  int value = -1;       // 属性值（label value）
  float score = 0;      // 属性置信度
  std::string name;     // 属性名称/文本
};

inline std::ostream& operator<<(std::ostream& os, const s_attr_info& info) {
  os << "{id=" << info.id
     << ", value=" << info.value
     << ", score=" << info.score
     << ", name=" << info.name << "}";
  return os;
}

/**
 * @brief 对象 obj 结构体
 */
struct s_obj_in {
  int id = -1;                              // 检测 class label
  std::string track_id;                     // 追踪 id
  float score = 0;                          // 置信度
  std::vector<float> bboxs;                 // xywh（float 保留精度）
  std::vector<float> feature;               // 特征向量
  std::vector<s_class_infos> classes;       // 分类结果列表
  std::string model_name;                   // 模型名
  std::vector<std::vector<float>> key_points;  // 关键点结果列表
  std::string type;                         // 对象类型："original"
  std::vector<std::pair<std::string, s_attr_info>> attributes;  // 属性列表（key -> 属性值）
};

inline std::ostream& operator<<(std::ostream& os, const s_obj_in& obj) {
  os << "{id=" << obj.id
     << ", track_id=" << obj.track_id
     << ", score=" << obj.score
     << ", bboxs=[";
  for (size_t i = 0; i < obj.bboxs.size(); ++i) {
    if (i > 0) os << ", ";
    os << obj.bboxs[i];
  }
  os << "], feature=[";
  for (size_t i = 0; i < obj.feature.size(); ++i) {
    if (i > 0) os << ", ";
    os << obj.feature[i];
  }
  os << "], classes=[";
  for (size_t i = 0; i < obj.classes.size(); ++i) {
    if (i > 0) os << ", ";
    os << obj.classes[i];
  }
  os << "], key_points=[";
  for (size_t i = 0; i < obj.key_points.size(); ++i) {
    if (i > 0) os << ", ";
    os << "[";
    for (size_t j = 0; j < obj.key_points[i].size(); ++j) {
      if (j > 0) os << ", ";
      os << obj.key_points[i][j];
    }
    os << "]";
  }
  os << "], model_name=" << obj.model_name
     << ", type=" << obj.type
     << ", attributes=[";
  for (size_t i = 0; i < obj.attributes.size(); ++i) {
    if (i > 0) os << ", ";
    os << "(" << obj.attributes[i].first << ": " << obj.attributes[i].second << ")";
  }
  os << "]}";
  return os;
}

/**
 * @brief 单帧的输出数据结构
 */
struct s_output_data {
  output_result result = output_result::RESULT_UNKNOWN_ERROR;  // 结果码
  uint64_t timestamp = 0;                                     // 时间戳
  std::string frame_id_s;                                     // frame 标识
  std::vector<s_obj_in> objects;                              // 检测对象列表
  std::string objects_json;                                   // JSON 序列化结果
  std::unordered_map<std::string, cv::Mat> image_dict;        // 图像字典
};

inline std::ostream& operator<<(std::ostream& os, const s_output_data& data) {
  os << "{result=" << static_cast<int>(data.result)
     << ", timestamp=" << data.timestamp
     << ", frame_id_s=" << data.frame_id_s
     << ", objects=[";
  for (size_t i = 0; i < data.objects.size(); ++i) {
    if (i > 0) os << ", ";
    os << data.objects[i];
  }
  os << "], objects_json=" << data.objects_json
     << ", image_dict_size=" << data.image_dict.size() << "}";
  return os;
}

}  // namespace cnstream

#endif  // MODULE_SINK_DATA_COMMON_HPP