#ifndef MODULE_SINK_DATA_CONVERTER_HPP
#define MODULE_SINK_DATA_CONVERTER_HPP

#include "cnstream_frame_va.hpp"
#include "data_common.hpp"

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
  dst.id_name    = src.id_name;
  dst.score      = src.score;
  dst.value      = src.value;
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
  {
    for (const auto& cls : src->classes) {
      dst.classes.push_back(ConvertClassInfo(cls));
    }
  }
  dst.bboxs = {src->bbox.x, src->bbox.y, src->bbox.w, src->bbox.h};
  dst.track_id = src->track_id;
  return dst;
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

  // 原始图像
  if (frame_info->collection.HasValue(kDataFrameTag)) {
    auto img_data = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    data.image_dict[output_constants::key_original_image] = img_data->GetImage();
  }
  // 自定义图像
  if (frame_info->collection.HasValue(kCustomImagesTag)) {
    auto custom_images = frame_info->collection.Get<CustomImagesPtr>(kCustomImagesTag);
    for (const auto& [key, mat] : *custom_images) {
      data.image_dict[key] = mat;
    }
  }
  // 推理对象
  if (frame_info->collection.HasValue(kInferObjsTag)) {
    auto objs_holder = frame_info->collection.Get<InferObjsPtr>(kInferObjsTag);
    {
      std::lock_guard<std::mutex> lk(objs_holder->mutex_);
      for (const auto& obj : objs_holder->objs_) {
        data.objects.push_back(ConvertInferObject(obj));
      }
    }
    data.result = 0;
  } else {
    data.result = -1;
  }
  return data;
}

}  // namespace cnstream

#endif  // MODULE_SINK_DATA_CONVERTER_HPP