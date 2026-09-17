
#include <algorithm>
#include <mutex>
#include <string>

#include "input_deriver.hpp"

#include "cnstream_logging.hpp"

#include <opencv2/opencv.hpp>


namespace cnstream {

namespace {

// 90° 整数倍旋转（cv::rotate 语义）与坐标逆变换共用这份方向解析
int ParseRotateFlag(const std::string& direction) {
  if (direction == "clockwise" || direction.empty()) return cv::ROTATE_90_CLOCKWISE;
  if (direction == "counterclockwise") return cv::ROTATE_90_COUNTERCLOCKWISE;
  return -1;
}

}  // namespace

/**
 * @brief 将基准图旋转 90° 作为模块定制输入图，并在后处理后将 obj 坐标逆旋转还原。
 *
 * @note 坐标变换（连续坐标，忽略像素中心的 ±0.5 误差）：
 *       顺时针 90°：基准图 W0×H0 -> 派生图 W1=H0, H1=W0
 *         正变换：  x1 = H0 - y,  y1 = x
 *         逆变换：  x = y1,       y = H0 - x1   （H0 即派生图宽度 W1）
 *       框 (l1,t1,r1,b1) 逆变换后：
 *         left = t1, right = b1, top = W1 - r1, bottom = W1 - l1（宽高互换）
 *       逆时针 90°：
 *         正变换：  x1 = y,  y1 = W0 - x
 *         逆变换：  x = W0 - y1,  y = x1   （W0 即派生图高度 H1）
 *       框 (l1,t1,r1,b1) 逆变换后：
 *         left = H1 - b1, right = H1 - t1, top = l1, bottom = r1（宽高互换）
 */
class Rotate90InputDeriver: public InputDeriver {

 public:
  bool Init(const std::map<std::string, std::string>& params) override {
    params_ = params;
    std::string direction = params_.count("direction") ? params_["direction"] : "";
    rotate_flag_ = ParseRotateFlag(direction);
    if (rotate_flag_ < 0) {
      LOGE(DERIVE) << "Rotate90 Init: invalid direction [" << direction
                    << "], expect clockwise / counterclockwise";
      return false;
    }
    rotation_ = (rotate_flag_ == cv::ROTATE_90_CLOCKWISE) ? 90 : 270;
    LOGI(DERIVE) << "Rotate90 Init: direction=" << direction
                  << ", rotation: " << rotation_;
    return true;
  }

  int Derive(const FrameInfoPtr& finfo, const std::string& model_name) override {
    if (!finfo || model_name.empty()) return -1;

    // 基准图：帧级派生图（若存在）> 原图
    cv::Mat base_img = GetModelInputImage(finfo);
    if (base_img.empty()) {
      LOGE(DERIVE) << "Rotate90: base image is empty";
      return -1;
    }

    // 输出到独立 Mat
    cv::Mat derived;
    cv::rotate(base_img, derived, rotate_flag_);

    auto meta = std::make_shared<ModelInputImage>();
    meta->image = derived;
    meta->cur_offset_x = 0;
    meta->cur_offset_y = 0;
    meta->cur_width = derived.cols;
    meta->cur_height = derived.rows;
    meta->cur_scale_x = 1.0f;
    meta->cur_scale_y = 1.0f;
    meta->rotation = rotation_;

    // 覆盖写：模块级 key 归本模块独占，重处理同一帧时刷新元数据
    finfo->collection.Set<ModelInputImagePtr>(ModelInputImageTagForModel(model_name), meta);
    return 0;
  }

  /***
   * @brief 从模块派生图坐标逆变换回基准图坐标
   * @return 0 成功，-1 失败
   * 当不需要处理的时候，也是返回 0
   */
  int RestoreObjs(const FrameInfoPtr& finfo, const std::string& model_name) override {
    if (!finfo || model_name.empty()) return -1;

    const std::string module_tag = ModelInputImageTagForModel(model_name);
    if (!finfo->collection.HasValue(module_tag)) return 0;
    auto meta = finfo->collection.Get<ModelInputImagePtr>(module_tag);
    if (!meta || meta->rotation == 0) return 0;
    if (!finfo->collection.HasValue(kInferObjsTag)) return 0;

    InferObjsPtr objs_holder = finfo->collection.Get<InferObjsPtr>(kInferObjsTag);

    const cv::Mat& derived = meta->image;
    const int derived_w = derived.cols;  // 顺时针时 = 基准图高 H0
    const int derived_h = derived.rows;  // 逆时针时 = 基准图宽 W0

    LOGI(DERIVE) << "Restore for model:" << model_name
                  << ", derived size [" << derived_h << " " << derived_w << "]";
                  << ", rotation: " << rotation_;

    std::lock_guard<std::mutex> lock(objs_holder->mutex_);
    ObjsVec& objs = objs_holder->objs_;
    for (auto& obj : objs) {
      if (!obj || obj->model_name != model_name) continue;
      // AddExtraAttribute 首次成功才做还原，保证每个 obj 的坐标只被逆变换一次
      if (!obj->AddExtraAttribute(kInferObjCoordRestoredKey, "1")) continue;

      const float l1 = obj->bbox.x;
      const float t1 = obj->bbox.y;
      const float r1 = obj->bbox.x + obj->bbox.w;
      const float b1 = obj->bbox.y + obj->bbox.h;

      float left = 0.f, top = 0.f, right = 0.f, bottom = 0.f;
      if (meta->rotation == 90) {  // 顺时针
        left   = t1;
        right  = b1;
        top    = derived_w - r1;
        bottom = derived_w - l1;
      } else {  // 逆时针
        left   = derived_h - b1;
        right  = derived_h - t1;
        top    = l1;
        bottom = r1;
      }

      // derived_h 相当于原图 W0；derived_w 相当于原图 H0
      left   = std::max(0.0f, std::min(left,   static_cast<float>(derived_h)));
      right  = std::max(0.0f, std::min(right,  static_cast<float>(derived_h)));
      top    = std::max(0.0f, std::min(top,    static_cast<float>(derived_w)));
      bottom = std::max(0.0f, std::min(bottom, static_cast<float>(derived_w)));

      obj->bbox.x = left;
      obj->bbox.y = top;
      obj->bbox.w = right - left;
      obj->bbox.h = bottom - top;
      obj->area = obj->bbox.w * obj->bbox.h;
    }
    return 0;
  }

 private:
  int rotate_flag_ = cv::ROTATE_90_CLOCKWISE;
  int rotation_ = 90;

 private:
  DECLARE_REFLEX_OBJECT_EX(Rotate90InputDeriver, cnstream::InputDeriver);
};  // class Rotate90InputDeriver

IMPLEMENT_REFLEX_OBJECT_EX(Rotate90InputDeriver, cnstream::InputDeriver);

}  // namespace cnstream
