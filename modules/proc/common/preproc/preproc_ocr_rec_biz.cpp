
#include "preproc_ocr_rec_obj.hpp"

#include <opencv2/opencv.hpp>

namespace cnstream {

/**
 * @brief 业务定制：PPOCRv3 识别 CPU 前处理（字符框内容旋转 180° 后输入模型）
 *
 * 仅对裁剪出的目标图做 180° 旋转，bbox 在原图中的坐标保持不变。
 * 部署时将模型配置中的 preproc_name 指定为 "Pre_PPOCRv3_rec_Obj_Rot180"。
 */
class Pre_PPOCRv3_rec_Obj_Rot180 : public Pre_PPOCRv3_rec_Obj {
 protected:
  void OnCropped(cv::Mat& img) override {
    cv::flip(img, img, -1);  // 同时绕 x、y 轴翻转，等价于旋转 180°
  }

  DECLARE_REFLEX_OBJECT_EX(Pre_PPOCRv3_rec_Obj_Rot180, cnstream::ObjPreproc);
};  // class Pre_PPOCRv3_rec_Obj_Rot180

IMPLEMENT_REFLEX_OBJECT_EX(Pre_PPOCRv3_rec_Obj_Rot180, cnstream::ObjPreproc);

}  // namespace cnstream
