
#include <any>
#include <map>
#include <mutex>
#include <string>
#include <utility>


#include <opencv2/opencv.hpp>
#include <libyuv/convert.h>

#include "cnstream_collection.hpp"
#include "base.hpp"


std::string key_num = "num";
std::string key_str = "str";
std::string key_obj_1 = "obj_1";
std::string key_obj_2 = "obj_2";

namespace cnstream {

TEST(Collection, ANY) {

  std::map<std::string, std::any> data;
  data[key_num] = 100;
  data[key_str] = std::string("hello");

  ASSERT_TRUE(data[key_num].has_value());
  int value = std::any_cast<int>(data[key_num]);

  ASSERT_EQ(value, 100);
  ASSERT_EQ(std::any_cast<std::string>(data[key_str]), std::string("hello"));

  std::any a0;
  ASSERT_FALSE(a0.has_value());
  std::any a1 = 100;
  ASSERT_TRUE(a1.has_value());
  ASSERT_EQ(std::any_cast<int>(a1), 100);
  a1.reset();
  ASSERT_FALSE(a1.has_value());
  auto a2 = std::make_any<std::string>("hello");
  ASSERT_TRUE(a2.has_value());

  // with smart ptr
  std::unique_ptr<std::any> a3 = std::make_unique<std::any>(100);
  ASSERT_TRUE(a3->has_value());
  std::cout << "a3 type: " << a3->type().name() << std::endl;
  ASSERT_EQ(std::any_cast<int>(*a3), 100);
}

TEST(Collection, GET) { 
  cnstream::Collection collection;
  ASSERT_FALSE(collection.HasValue(key_num));

  collection.Add(key_num, 100);
  ASSERT_TRUE(collection.HasValue(key_num));
  ASSERT_EQ(collection.Get<int>(key_num), 100);

  int num_value = 200;
  collection.Add(key_num, num_value);  // warning: 覆盖已存在数据
  ASSERT_EQ(collection.Get<int>(key_num), num_value);

  std::string str_value = "hello";
  collection.Add(key_str, str_value);
  ASSERT_EQ(collection.Get<std::string>(key_str), str_value);

  ASSERT_FALSE(collection.AddIfNotExists(key_num, 300));

}


class FrameInfoObj {
 public:
  FrameInfoObj() : frame_(cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 255))) {}
  ~FrameInfoObj() = default;

 protected:
  cv::Mat frame_;
  mutable std::mutex mtx_;
  int stride_[3];
};

TEST(Collection, Object) {
  cnstream::Collection collection;

  std::shared_ptr<FrameInfoObj> frame_info = std::make_shared<FrameInfoObj>();
  collection.Add(key_obj_1, frame_info);
  ASSERT_TRUE(collection.HasValue(key_obj_1));
  ASSERT_EQ(collection.Get<std::shared_ptr<FrameInfoObj>>(key_obj_1), frame_info);

}


}  // namespace cnstream