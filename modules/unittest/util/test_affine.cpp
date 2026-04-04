
#include <opencv2/opencv.hpp>
#include <iostream>

#include "affine_trans.hpp"
#include "base.hpp"

static const std::string image_path = "test_image.png";

static const int dst_w = 640;
static const int dst_h = 640;

static const std::string output_path_resize = "save_image/test_affine_resize.jpg";
static const std::string output_path_resize_swap = "save_image/test_affine_resize_swap.jpg";
static const std::string output_path_resize_swap_norm = "save_image/test_affine_resize_swap_norm.jpg";

namespace cnstream {

TEST(AffineTrans, preprocess) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  EXPECT_TRUE(!image.empty());
  
  size_t expected = image.total() * image.elemSize();
  size_t actual = image.dataend - image.data;

  if (image.isContinuous()) {
    std::cout << "AffineTrans: image is continuous" << std::endl;
    EXPECT_EQ(expected, actual);
  } else {
    std::cout << "AffineTrans: image is not continuous" << std::endl;
    EXPECT_NE(expected, actual);
  }

  AffineTrans trans;

  int src_w = image.cols;
  int src_h = image.rows;

  std::tuple<int, int> from{src_w, src_h};
  std::tuple<int, int> to{dst_w, dst_h};
  trans.compute(from, to);

  auto norm = Norm::alpha_beta(1 / 255.0f, 0.0f);

  float* output = new float[dst_w * dst_h * 3];
  resize_cpu(image.data, src_w, src_h, image.step, output, dst_w, dst_h, 114.0f, trans.get_d2s());  // output: CHW BGR
  save_float_image_chw_cpu(output, dst_w, dst_h, output_path_resize, ChannelsArrange::BGR);

  swap_channel_cpu(output, dst_w, dst_h, dst_w * dst_h, ChannelsArrange::BGR);  // output: CHW RGB
  save_float_image_chw_cpu(output, dst_w, dst_h, output_path_resize_swap, ChannelsArrange::RGB);

  normalize_cpu(output, dst_w, dst_h, dst_w * dst_h, norm, ChannelsArrange::RGB); 
  save_float_image_chw_cpu(output, dst_w, dst_h, output_path_resize_swap_norm, ChannelsArrange::RGB, true);

  delete[] output;
}



TEST(AffineTrans, postprocess) {



}

}  // namespace cnstream


