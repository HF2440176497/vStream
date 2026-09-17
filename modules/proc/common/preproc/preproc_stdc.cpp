

#include "preproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>


namespace cnstream {

namespace {

// ImageNet 归一化参数（RGB 顺序）
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

}  // namespace

/**
 * @brief STDC 图像分割 CPU 前处理
 * @note 流程：BGR->RGB -> /255 -> (x-mean)/std -> HWC 转 CHW。
 *       整图直接 resize 到模型输入尺寸
 */
class Pre_STDC_CPU: public Preproc {

int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
            const std::shared_ptr<cnstream::FrameInfo>& package) override {

    LOGD(PREPROC) << "Execute for data: " << package->GetStreamId() << ", timestamp: " << package->GetTimestamp();

    int channel = model->get_channel();
    if (channel != 3) {
        LOGE(PREPROC) << "model input shape not supported";
        return -1;
    }

    cv::Mat img = GetModelInputImage(package, model->get_name());  // BGR：模块级派生图 > 帧级派生图 > 原图
    if (img.empty()) {
        LOGE(PREPROC) << "input image is empty";
        return -1;
    }

    int input_index = model->get_input_ordered_index();  // input tensor index
    int input_h = model->get_height();
    int input_w = model->get_width();

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

    // BGR -> RGB，uint8 -> float [0, 1]
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // HWC RGB -> CHW RGB，并按通道做 (x - mean) / std
    float* dst = cpu_outputs[input_index];
    const int plane = input_h * input_w;

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    for (int c = 0; c < 3; ++c) {
        const float* src = channels[c].ptr<float>();
        float* plane_ptr = dst + c * plane;
        const float mean = kMean[c];
        const float std_inv = 1.0f / kStd[c];
        for (int i = 0; i < plane; ++i) {
            plane_ptr[i] = (src[i] - mean) * std_inv;
        }
    }
    return 0;
}

 private:
  DECLARE_REFLEX_OBJECT_EX(Pre_STDC_CPU, cnstream::Preproc);
};  // class Pre_STDC_CPU

IMPLEMENT_REFLEX_OBJECT_EX(Pre_STDC_CPU, cnstream::Preproc);

} // namespace cnstream
