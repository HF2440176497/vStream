

#include "postproc.hpp"
#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include <opencv2/opencv.hpp>


namespace cnstream {

class YoloPostproc: public Postproc {

 public:
  int Execute(const std::vector<float*>& cpu_outputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) {

    LOGI(Postproc) << "YoloPostproc Execute";
    return 0;

  }
 private:
  DECLARE_REFLEX_OBJECT_EX(YoloPostproc, cnstream::Postproc);
};  // class YoloPostproc

IMPLEMENT_REFLEX_OBJECT_EX(YoloPostproc, cnstream::Postproc);

}  // namespace cnstream
