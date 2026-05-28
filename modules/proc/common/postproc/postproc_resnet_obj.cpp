

#include "postproc.hpp"


#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace cnstream {


class Post_Resnet_Obj : public ObjPostproc {
 public:
  /**
   * outputs: D2H 的结果
   */
  int Execute(const std::vector<float*>& outputs, ModelLoader* model,
              const FrameInfoPtr& finfo, const std::shared_ptr<InferObject>& pobj) override {

    int output_index = 0;

    const float* output = outputs[output_index];
    TensorShape output_shape = model->OutputShape(output_index);
    int num_classes = output_shape.numel();

    auto max_it = std::max_element(output, output + num_classes);
    float max_logit = *max_it;
    int best_class = static_cast<int>(max_it - output);

    float exp_sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
      exp_sum += std::exp(output[i] - max_logit);
    }

    float best_score = std::exp(output[best_class] - max_logit) / exp_sum;

    InferObjectInfo class_info;
    class_info.id = best_class;
    class_info.model_name = model->get_name();
    class_info.score = best_score;
    class_info.value = best_score;

    pobj->classes.push_back(class_info);

    return 0; 
  }

 private:
  DECLARE_REFLEX_OBJECT_EX(Post_Resnet_Obj, cnstream::ObjPostproc);
};  // class Post_Resnet_Obj

IMPLEMENT_REFLEX_OBJECT_EX(Post_Resnet_Obj, cnstream::ObjPostproc);

}