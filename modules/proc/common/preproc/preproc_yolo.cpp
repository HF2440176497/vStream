

#include "preproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include "affine_trans.hpp"
#include <opencv2/opencv.hpp>


namespace cnstream {

/**
 * @brief YOLO CPU 前处理
 */
class YoloPreproc: public Preproc {

/**
 * @param cpu_outputs 相对于 preproc 是输出，相对于 H2D 是输入，仍然位于 CPU 上
 * size == input tensor num
 */
int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
            const std::shared_ptr<cnstream::FrameInfo>& package) override {

    LOGI(PREPROC) << "Execute for data: " << package->GetStreamId() << ", timestamp: " << package->GetTimestamp();
    int channel = model->get_channel();
    if (channel != 3) {
        LOGE(PREPROC) << "model input shape not supported";
        return -1;
    }
    DataFramePtr frame = package->collection.Get<DataFramePtr>(cnstream::kDataFrameTag);
    cv::Mat img = frame->GetImage();  // BGR
    
    int img_w = img.cols;
    int img_h = img.rows;
    
    int input_index = model->get_input_ordered_index();  // input tensor index
    int input_h = model->get_height();
    int input_w = model->get_width();

    // dst / src
    float img_scale = std::min((float)(input_w) / (float)(img_w), (float)(input_h) / (float)(img_h));
    int new_w = int(img_w * img_scale);
    int new_h = int(img_h * img_scale);

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(new_w, new_h), cv::INTER_LINEAR);
            
    // background pic padding
    cv::Mat net_input_data(input_h, input_w, CV_32FC3, cv::Scalar(114.0f, 114.0f, 114.0f));
    // 理论上 input_h, input_w > new_h, new_w
    int top = (input_h - new_h) / 2;
    int left = (input_w - new_w) / 2;
    top = std::max(0, top);
    left = std::max(0, left);

    // 取 min: 不能超过 resize 后 new_w, new_h 的范围
    int roi_w = std::min(new_w, input_w - left);
    int roi_h = std::min(new_h, input_h - top);

    if (roi_w > 0 && roi_h > 0) {
        resize_img.copyTo(net_input_data(cv::Rect(left, top, roi_w, roi_h)));
    }

    // HWC BGR -> HWC RGB
    cv::cvtColor(net_input_data, net_input_data, cv::COLOR_BGR2RGB);
    net_input_data.convertTo(net_input_data, CV_32FC3);

    net_input_data /= 255.0;

    // HWC RGB -> CHW RGB
    // 按照图片本身的尺寸决定内存大小
    std::vector<cv::Mat> channels(3);
    cv::split(net_input_data, channels);

    float* cpu_output = cpu_outputs[input_index];
    for (int c = 0; c < 3; c++) {
        memcpy(cpu_output + c * input_h * input_w, channels[c].ptr<float>(), input_h * input_w * sizeof(float));
    }

#ifdef VSTREAM_UNIT_TEST
    if (!has_save_frame_mat_) {
        save_float_image_chw_cpu(cpu_output, input_h, input_w, save_file_, ChannelsArrange::RGB, true);
        has_save_frame_mat_ = true;
    }
#endif
    return 0;
}

 private:
  bool has_save_frame_mat_ = false;
  std::string save_file_ = "save/test_preproc_save.jpg";

 private:
  DECLARE_REFLEX_OBJECT_EX(YoloPreproc, cnstream::Preproc);
};  // class YoloPreproc

IMPLEMENT_REFLEX_OBJECT_EX(YoloPreproc, cnstream::Preproc);


}  // namespace cnstream