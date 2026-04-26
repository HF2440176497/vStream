
#ifndef MODULE_OSD_DATA_COMMON_HPP
#define MODULE_OSD_DATA_COMMON_HPP


#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#include "cnstream_collection.hpp"

namespace cnstream {

namespace output_constants {

inline const std::string key_result = "result";
inline const std::string key_timestamp = "timestamp";
inline const std::string key_objects = "objects";
inline const std::string key_objects_dict = "objects_dict";
inline const std::string key_objects_json = "objects_json";

inline const std::string key_image_dict = "image_dict";
inline const std::string key_original_image = "original_image";

}

typedef struct classInfos_ {
  int id;                             //分类序号
  std::string model_name;             //modelName  模型名
  std::string id_name;                //分类名称
  float score;                        //得分
  float value;                        //得分（0，1化）
} s_class_infos;


/**
 * @brief 对象 obj 结构体
 */
typedef struct objIn_ {
  std::string track_id;               // 追踪id
  float score;                        // 得分
  std::vector<int> bboxs;             // xywh
  std::vector<float> feature;         // 特征向量
  std::vector<s_class_infos> classes; // 分类框模型结果
  std::string str_id;                 // 分类框序号
  std::string model_name;             // 模型名
} s_obj_in;


/**
 * @brief 单帧的输出数据结构
 */
typedef struct outputData_ {
  int result = -1;      // 结果码
  uint64_t timestamp;   // 时间戳
  std::string frame_id_s;  // frame
  std::vector<s_obj_in> objects;    // 检测框
  std::vector<std::map<std::string,std::string>> objects_dict;
  std::string objects_json;         // 序列化后的字符串
  std::unordered_map<std::string, cv::Mat> image_dict;  // 原始图像字典
} s_output_data;



}  // namespace cnstream

#endif  // MODULE_OSD_DATA_COMMON_HPP
