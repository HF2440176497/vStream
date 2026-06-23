

#include "preproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;


namespace cnstream {

class Pre_PPOCRv3_rec_Obj : public ObjPreproc {
 public:
  bool Init(const std::map<std::string, std::string> &params) override {
    params_ = params;
    return true;
  }
  /**
   * @brief cpu_outputs 作为前处理的输出，作为 D2H 的输入
   */
  virtual int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
                      const FrameInfoPtr& finfo, const std::shared_ptr<InferObject>& pobj) override {

    LOGI(PREPROC) << "Pre_PPOCRv3_rec_Obj Execute";
    auto start_time = std::chrono::steady_clock::now();
    if (model_name_.empty()) {
      model_name_ = model->get_name();
    }
    int input_index = model->get_input_ordered_index();

    DataFramePtr frame = finfo->collection.Get<DataFramePtr>(kDataFrameTag);
    cv::Mat img = frame->GetImage();
    if (img.empty()) return -1;

    int input_h = model->get_height();  // 48
    int input_w  = model->get_width();  // 320

    // 2. 裁剪并 clamp 到图像范围内
    int x = std::max(0, (int)pobj->bbox.x);
    int y = std::max(0, (int)pobj->bbox.y);
    int w = std::min((int)pobj->bbox.w, img.cols - x);
    int h = std::min((int)pobj->bbox.h, img.rows - y);
    if (w <= 0 || h <= 0) return -1;
    cv::Rect rect(x, y, w, h);
    cv::Mat crop_img = img(rect).clone();

    // 3. （可选）宽度压缩
    if (crop_img.cols > 3) {
        cv::resize(crop_img, crop_img,
            cv::Size((crop_img.cols/3)*2, crop_img.rows),
            0, 0, cv::INTER_LINEAR);
    }

    // 4. 按比例 resize 到 imgH
    float ratio = float(crop_img.cols) / float(crop_img.rows);
    int resize_w = std::min(int(ceilf(input_h * ratio)), input_w);
    cv::Mat resize_img;
    cv::resize(crop_img, resize_img, cv::Size(resize_w, input_h), 0, 0, cv::INTER_LINEAR);

    // 5. 右侧补边
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                       input_w - resize_img.cols,
                       cv::BORDER_CONSTANT, {127,127,127});

    // 6. BGR->RGB + float
    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
    resize_img.convertTo(resize_img, CV_32FC3, 1.0/255.0);   // 顺便 /255

    // 7. 标准化
    resize_img = (resize_img - 0.5) / 0.5;

    // 8. NCHW 拷贝单个 batch 的输入
    std::vector<cv::Mat> channels(3);
    cv::split(resize_img, channels);

    float* cpu_output = cpu_outputs[input_index];
    for (int c = 0; c < 3; c++) {
        memcpy(cpu_output + c * input_h * input_w, channels[c].ptr<float>(), input_h * input_w * sizeof(float));
    }

    double dr_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start_time).count();
    LOGI(PREPROC) << " Pre_PPOCRv3_rec_Obj Execute " << dr_ms << " ms";
    return 0;
  }

 private:
  std::string model_name_;

  DECLARE_REFLEX_OBJECT_EX(Pre_PPOCRv3_rec_Obj, cnstream::ObjPreproc);
};  // class Pre_PPOCRv3_rec_Obj

IMPLEMENT_REFLEX_OBJECT_EX(Pre_PPOCRv3_rec_Obj, cnstream::ObjPreproc);

}  // namespace cnstream