#ifndef MODULE_SINK_DATA_CONVERTER_HPP
#define MODULE_SINK_DATA_CONVERTER_HPP

#include "cnstream_frame_va.hpp"
#include "data_common.hpp"

#include <nlohmann/json.hpp>

namespace cnstream {

/**
 * @brief 数据转换器：负责将框架内部表示（InferObject / FrameInfo）转换为输出层数据结构（s_obj_in / s_output_data）。
 *
 * 集中管理映射逻辑，未来新增字段只需修改这里的转换函数
 */

/**
 * @brief InferObjectInfo → s_class_infos
 */
inline s_class_infos ConvertClassInfo(const InferObjectInfo& src) {
  s_class_infos dst;
  dst.id         = src.id;
  dst.model_name = src.model_name;
  dst.name       = src.name;
  dst.score      = src.score;
  dst.value      = src.value;
  return dst;
}

/**
 * @brief InferAttr → s_attr_info
 */
inline s_attr_info ConvertInferAttr(const InferAttr& src) {
  s_attr_info dst;
  dst.id    = src.id;
  dst.value = src.value;
  dst.score = src.score;
  dst.name  = src.name;
  return dst;
}

/**
 * @brief InferObject → s_obj_in
 */
inline s_obj_in ConvertInferObject(const std::shared_ptr<InferObject>& src) {
  s_obj_in dst;
  dst.id         = src->id;
  dst.score      = src->score;
  dst.model_name = src->model_name;

  // classes
  {
    for (const auto& cls : src->classes) {
      dst.classes.push_back(ConvertClassInfo(cls));
    }
  }
  // feature

  dst.bboxs = {src->bbox.x, src->bbox.y, src->bbox.w, src->bbox.h};
  dst.track_id = src->track_id;

  // key_points
  {
    for (const auto& kp : src->key_points) {
      dst.key_points.push_back(kp);
    }
  }

  // attributes（支持多个 key，例如 OCR 的 Identification）
  {
    auto attrs = src->GetAttributes();
    for (const auto& [key, attr] : attrs) {
      dst.attributes.push_back({key, ConvertInferAttr(attr)});
    }
  }

  // 默认是 "original" 类型
  dst.type = cnstream::GetInferObjType(src);

  return dst;
}

// JSON 序列化
inline void to_json(nlohmann::json& j, const s_class_infos& info) {
  j = nlohmann::json{
      {"id", info.id},
      {"model_name", info.model_name},
      {"name", info.name},
      {"score", info.score},
      {"value", info.value}};
}

inline void to_json(nlohmann::json& j, const s_attr_info& info) {
  j = nlohmann::json{
      {"id", info.id},
      {"value", info.value},
      {"score", info.score},
      {"name", info.name}};
}

inline void to_json(nlohmann::json& j, const s_obj_in& obj) {
  j = nlohmann::json{
      {"id", obj.id},
      {"track_id", obj.track_id},
      {"score", obj.score},
      {"bboxs", obj.bboxs},
      {"feature", obj.feature},
      {"classes", obj.classes},
      {"model_name", obj.model_name},
      {"key_points", obj.key_points},
      {"type", obj.type},
      {"attributes", obj.attributes}};
}

/**
 * @brief 在 FrameInfo 的 collection 中保存原始图像到 kCustomImagesTag。
 *
 * 应在任何可能修改 DataFrame 原始数据的操作（如标框渲染）之前调用。
 * 通过 DataFrame::GetImage() 获取 deep copy，确保原图独立于后续修改。
 * PushHandler 和 QueueHandler 均需在 Process 入口处调用。
 */
inline void SaveOriFrame(const std::shared_ptr<FrameInfo>& frame_info) {
  if (!frame_info->collection.HasValue(kDataFrameTag)) return;
  auto frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
  if (!frame) return;

  if (!frame_info->collection.HasValue(kCustomImagesTag)) {
    frame_info->collection.AddIfNotExists(kCustomImagesTag,
        std::make_shared<std::map<std::string, cv::Mat>>());
  }
  auto custom_images = frame_info->collection.Get<CustomImagesPtr>(kCustomImagesTag);
  (*custom_images)[output_constants::key_original_image] = frame->GetImage();
}

/**
 * @brief FrameInfo → s_output_data
 *
 * 从 FrameInfo 的 collection 中提取图像、推理结果等信息，
 * 组装为对用户暴露的输出数据结构。
 */
inline s_output_data ConvertFrameInfo(const std::shared_ptr<FrameInfo>& frame_info) {
  s_output_data data;

  data.frame_id_s = frame_info->frame_id_s;
  data.timestamp  = frame_info->timestamp;

  // 跳帧标记：帧被 infer_interval 跳过，未经过推理
  if (frame_info->collection.HasValue(kSkipFrameTag)) {
    data.result = output_result::RESULT_SKIPPED;
    return data;
  }

  // 原始图像
  if (frame_info->collection.HasValue(kDataFrameTag)) {
    auto img_data = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    data.image_dict[output_constants::key_original_image] = img_data->GetImage();
  } else {
    LOGE(DATA_CONVERTER) << "ConvertFrameInfo: DataFrame not found in FrameInfo collection.";
    data.result = output_result::RESULT_UNKNOWN_ERROR;
    return data;
  }

  // 自定义图像
  if (frame_info->collection.HasValue(kCustomImagesTag)) {
    auto custom_images = frame_info->collection.Get<CustomImagesPtr>(kCustomImagesTag);
    for (const auto& [key, mat] : *custom_images) {
      data.image_dict[key] = mat;
    }
  }
  // 检测对象（原始框 + 合并框均在 kInferObjsTag 中，通过 type 区分）
  if (frame_info->collection.HasValue(kInferObjsTag)) {
    auto objs_holder = frame_info->collection.Get<InferObjsPtr>(kInferObjsTag);
    {
      std::lock_guard<std::mutex> lk(objs_holder->mutex_);
      for (const auto& obj : objs_holder->objs_) {
        data.objects.push_back(ConvertInferObject(obj));
      }
    }
  } else {
    LOGE(DATA_CONVERTER) << "ConvertFrameInfo: InferObjs not found in FrameInfo collection.";
    data.result = output_result::RESULT_UNKNOWN_ERROR;
    return data;
  }

  try {
    data.objects_json = nlohmann::json(data.objects).dump();
  } catch (const std::exception& e) {
    LOGE(DATA_CONVERTER) << "ConvertFrameInfo: failed to serialize objects to json: " << e.what();
    data.objects_json.clear();
  }

  data.result = output_result::RESULT_OK;
  return data;
}

}  // namespace cnstream

#endif  // MODULE_SINK_DATA_CONVERTER_HPP
