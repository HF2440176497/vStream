
#include "preproc_resnet.hpp"

#include <opencv2/opencv.hpp>

namespace cnstream {

/**
 * @brief 业务定制：ResNet CPU 前处理（分类目标图旋转 180° 后输入模型）
 *
 * 仅对裁剪出的目标图做 180° 旋转，bbox 在原图中的坐标保持不变。
 * 部署时将模型配置中的 preproc_name 指定为 "Pre_Resnet_Obj_Rot180"。
 */
class Pre_Resnet_Obj_Rot180 : public Pre_Resnet_Obj {
 protected:
  void OnCropped(cv::Mat& img) override {
    cv::flip(img, img, -1);  // 同时绕 x、y 轴翻转，等价于旋转 180°
  }

  DECLARE_REFLEX_OBJECT_EX(Pre_Resnet_Obj_Rot180, cnstream::ObjPreproc);
};  // class Pre_Resnet_Obj_Rot180

IMPLEMENT_REFLEX_OBJECT_EX(Pre_Resnet_Obj_Rot180, cnstream::ObjPreproc);

}  // namespace cnstream
