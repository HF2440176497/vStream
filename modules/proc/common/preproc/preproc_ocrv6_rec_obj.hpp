#ifndef MODULES_PROC_COMMON_PREPROC_PREPROC_OCRV6_REC_OBJ_HPP_
#define MODULES_PROC_COMMON_PREPROC_PREPROC_OCRV6_REC_OBJ_HPP_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "preproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"
#include "proc/common/debug_image_saver.hpp"

#include <opencv2/opencv.hpp>

namespace cnstream {

namespace {

inline constexpr const char* key_crop_padding = "crop_padding";

}  // namespace


/**
 * @brief PPOCRv6 识别 CPU 前处理（对象级）
 *
 * 与 PaddleOCR 框架 RecResizeImg（ppocr/data/imaug/rec_img_aug.py resize_norm_img）对齐：
 *   bbox 裁剪（可选 crop_padding 边缘外扩，限原图内）→ 等比 resize 到 h=48（宽上限 = 模型输入宽）
 *   → /255 → -0.5 → /0.5 → 右侧补 0 到模型输入宽 → HWC 转 NCHW。
 * 通道序保持 BGR
 * 输入宽高取自 model loader，静态特化 engine [1,3,48,320] 下即 48/320。
 */
class Pre_PPOCRv6_rec_Obj : public ObjPreproc {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    auto it = params.find(key_crop_padding);
    if (it != params.end()) {
      try {
        crop_padding_ = std::max(0, std::stoi(it->second));
        LOGU(PREPROC) << "crop_padding value: " << crop_padding_;
      } catch (const std::exception&) {
        LOGW(PREPROC) << "Invalid crop_padding value: " << it->second << ", using default 0";
      }
    }
    return true;
  }
  /**
   * @brief cpu_outputs 作为前处理的输出，作为 D2H 的输入
   * @detail 输入保持 BGR 排序
   */
  virtual int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
                      const FrameInfoPtr& finfo, const std::shared_ptr<InferObject>& pobj) override {

    LOGD(PREPROC) << "Pre_PPOCRv6 Execute";
    auto start_time = std::chrono::steady_clock::now();
    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }
    int input_index = model->get_input_ordered_index();

    DataFramePtr frame = finfo->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
        LOGE(PREPROC) << "Pre_PPOCRv6 Execute: DataFrame is null";
        return -1;
    }
    cv::Mat img = GetModelInputImage(finfo, model->get_name());  // 模块级派生图 > 帧级派生图 > 原图
    if (img.empty()) return -1;

    int input_h = model->get_height();  // 48
    int input_w  = model->get_width();  // 320

    // 裁剪
    int bx = std::max(0, (int)pobj->bbox.x);
    int by = std::max(0, (int)pobj->bbox.y);
    int bw = std::min((int)pobj->bbox.w, img.cols - bx);
    int bh = std::min((int)pobj->bbox.h, img.rows - by);
    if (bw <= 0 || bh <= 0) return -1;

    // 边缘外扩
    int x = std::max(0, bx - crop_padding_);
    int y = std::max(0, by - crop_padding_);
    int w = std::min(bx + bw + crop_padding_, img.cols) - x;
    int h = std::min(by + bh + crop_padding_, img.rows) - y;
    cv::Rect rect(x, y, w, h);
    cv::Mat crop_img = img(rect).clone();

    // 帧级部署侧旋转：
    // 仅旋转裁剪图，bbox 等原图信息保持不变
    if (finfo->collection.HasValue(kCropRotate180Tag)) {
      cv::rotate(crop_img, crop_img, cv::ROTATE_180);
    }

    // 业务定制点：bbox 裁剪后、Resize 前的部署侧变换（默认无操作）
    OnCropped(crop_img);

    float ratio = float(crop_img.cols) / float(crop_img.rows);
    int resize_w = std::min(int(ceilf(input_h * ratio)), input_w);
    cv::Mat resize_img;
    cv::resize(crop_img, resize_img, cv::Size(resize_w, input_h), 0, 0, cv::INTER_LINEAR);

#ifdef VSTREAM_UNIT_TEST
    if (debug_saver_.enable()) {
      debug_saver_.MaybeSave("pre_ocrv6_rec", resize_img);
    }
#endif

    resize_img.convertTo(resize_img, CV_32FC3, 1.0/255.0);

    cv::subtract(resize_img, cv::Scalar(0.5f, 0.5f, 0.5f), resize_img);
    cv::divide(resize_img, cv::Scalar(0.5f, 0.5f, 0.5f), resize_img);

    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                    input_w - resize_img.cols,
                    cv::BORDER_CONSTANT, {0, 0, 0});

    // NCHW 拷贝单个 batch 的输入
    std::vector<cv::Mat> channels(3);
    cv::split(resize_img, channels);

    float* cpu_output = cpu_outputs[input_index];
    for (int c = 0; c < 3; c++) {
        memcpy(cpu_output + c * input_h * input_w, channels[c].ptr<float>(), input_h * input_w * sizeof(float));
    }

    double dr_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start_time).count();
    LOGI(PREPROC) << " Pre_PPOCRv6 Execute " << dr_ms << " ms";
    return 0;
  }

 protected:
  /**
   * @brief 业务定制点：bbox 裁剪得到目标图后、Resize 前调用，默认不做任何处理
   * @note  子类可覆写以实现部署侧定制（如旋转、宽度压缩）；仅允许修改 img 本身
   */
  virtual void OnCropped(cv::Mat& img) {}

 private:
  std::string model_name_;
  int crop_padding_ = 0;

 private:
  cnstream::DebugImageSaver debug_saver_{false, 500};

  DECLARE_REFLEX_OBJECT_EX(Pre_PPOCRv6_rec_Obj, cnstream::ObjPreproc);
};  // class Pre_PPOCRv6_rec_Obj

}  // namespace cnstream

#endif  // MODULES_PROC_COMMON_PREPROC_PREPROC_OCRV6_REC_OBJ_HPP_
