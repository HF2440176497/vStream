

#include "postproc.hpp"


#include "model_loader.hpp"
#include "reflex_object.h"

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include <opencv2/opencv.hpp>

namespace cnstream {


class Post_Resnet_Obj : public ObjPostproc {
 public:


 private:
  DECLARE_REFLEX_OBJECT_EX(Post_Resnet_Obj, cnstream::ObjPostproc);
};  // class Post_Resnet_Obj

IMPLEMENT_REFLEX_OBJECT_EX(Post_Resnet_Obj, cnstream::ObjPostproc);

}