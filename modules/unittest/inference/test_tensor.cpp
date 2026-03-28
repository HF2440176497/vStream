
// 测试 TensorShape 和 Tensor

#include "base.hpp"
#include "tensor.hpp"

namespace cnstream {

TEST(TensorShape, DefaultConstructor) {
  TensorShape shape;
  EXPECT_EQ(shape.ndims(), 0);
  EXPECT_EQ(shape.numel(), 0);
}

TEST(TensorShape, ConstructorNCHW) {
  TensorShape shape(2, 3, 224, 224);
  EXPECT_EQ(shape.ndims(), 4);
  EXPECT_EQ(shape.N(), 2);
  EXPECT_EQ(shape.C(), 3);
  EXPECT_EQ(shape.H(), 224);
  EXPECT_EQ(shape.W(), 224);
  EXPECT_EQ(shape.numel(), 2 * 3 * 224 * 224);
}

TEST(TensorShape, ConstructorWithArray) {
  int dims[] = {1, 64, 128, 128};
  TensorShape shape(4, dims);
  EXPECT_EQ(shape.ndims(), 4);
  EXPECT_EQ(shape.size(0), 1);
  EXPECT_EQ(shape.size(1), 64);
  EXPECT_EQ(shape.size(2), 128);
  EXPECT_EQ(shape.size(3), 128);
  EXPECT_EQ(shape.numel(), 1 * 64 * 128 * 128);
}

TEST(TensorShape, ConstructorWithVector) {
  std::vector<int> dims = {4, 3, 32, 32};
  TensorShape shape(dims);
  EXPECT_EQ(shape.ndims(), 4);
  EXPECT_EQ(shape.shape(0), 4);
  EXPECT_EQ(shape.shape(1), 3);
  EXPECT_EQ(shape.shape(2), 32);
  EXPECT_EQ(shape.shape(3), 32);
}

TEST(TensorShape, CopyConstructor) {
  TensorShape shape1(1, 3, 480, 640);
  TensorShape shape2(shape1);
  EXPECT_EQ(shape2.ndims(), 4);
  EXPECT_EQ(shape2.N(), 1);
  EXPECT_EQ(shape2.C(), 3);
  EXPECT_EQ(shape2.H(), 480);
  EXPECT_EQ(shape2.W(), 640);
}

TEST(TensorShape, AssignmentOperator) {
  TensorShape shape1(2, 4, 100, 100);
  TensorShape shape2;
  shape2 = shape1;
  EXPECT_EQ(shape2.ndims(), 4);
  EXPECT_EQ(shape2.numel(), 2 * 4 * 100 * 100);
}

TEST(TensorShape, SizeAndShape) {
  TensorShape shape(1, 3, 224, 224);
  EXPECT_EQ(shape.size(0), 1);
  EXPECT_EQ(shape.size(1), 3);
  EXPECT_EQ(shape.size(2), 224);
  EXPECT_EQ(shape.size(3), 224);
  EXPECT_EQ(shape.shape(0), 1);
  EXPECT_EQ(shape.shape(1), 3);
  EXPECT_EQ(shape.shape(2), 224);
  EXPECT_EQ(shape.shape(3), 224);
}

// ----------------------------- Tensor 测试 -----------------------------


TEST(Tensor, ConstructorDtypeOnly) {
  Tensor tensor(DataType::FLOAT32);
  EXPECT_EQ(tensor.type(), DataType::FLOAT32);
  EXPECT_EQ(tensor.ndims(), 0);
  EXPECT_EQ(tensor.numel(), 0);
  EXPECT_EQ(tensor.bytes(), 0);
}

TEST(Tensor, ConstructorNCHW) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(1, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor.type(), DataType::FLOAT32);
  EXPECT_EQ(tensor.ndims(), 4);
  EXPECT_EQ(tensor.batch(), 1);
  EXPECT_EQ(tensor.channel(), 3);
  EXPECT_EQ(tensor.height(), 224);
  EXPECT_EQ(tensor.width(), 224);
  EXPECT_EQ(tensor.numel(), 1 * 3 * 224 * 224);
  EXPECT_EQ(tensor.bytes(), 1 * 3 * 224 * 224 * sizeof(float));
}

TEST(Tensor, ConstructorWithArray) {
  int dims[] = {2, 64, 56, 56};
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(4, dims, DataType::FLOAT32, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor.ndims(), 4);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 64);
  EXPECT_EQ(tensor.size(2), 56);
  EXPECT_EQ(tensor.size(3), 56);
}

TEST(Tensor, ConstructorWithVector) {
  std::vector<int> dims = {1, 128, 28, 28};
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(dims, DataType::FLOAT32, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor.ndims(), 4);
  EXPECT_EQ(tensor.dims(), dims);
}

TEST(Tensor, ResizeWithVector) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(DataType::FLOAT32, TensorFormat::LINEAR, data);
  std::vector<int> new_dims = {2, 3, 480, 640};
  tensor.resize(new_dims);
  EXPECT_EQ(tensor.ndims(), 4);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.size(2), 480);
  EXPECT_EQ(tensor.size(3), 640);
  EXPECT_EQ(tensor.numel(), 2 * 3 * 480 * 640);
}

TEST(Tensor, ResizeWithArray) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(DataType::FLOAT32, TensorFormat::LINEAR, data);
  int dims[] = {1, 256, 14, 14};
  tensor.resize(4, dims);
  EXPECT_EQ(tensor.ndims(), 4);
  EXPECT_EQ(tensor.numel(), 1 * 256 * 14 * 14);
}

TEST(Tensor, ResizeSingleDim) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(1, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  tensor.resize_single_dim(0, 4);
  EXPECT_EQ(tensor.size(0), 4);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.size(2), 224);
  EXPECT_EQ(tensor.size(3), 224);
}

TEST(Tensor, ResizeVariadic) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(DataType::FLOAT32, TensorFormat::LINEAR, data);
  tensor.resize(2, 3, 224, 224);
  EXPECT_EQ(tensor.ndims(), 4);
  EXPECT_EQ(tensor.batch(), 2);
  EXPECT_EQ(tensor.channel(), 3);
  EXPECT_EQ(tensor.height(), 224);
  EXPECT_EQ(tensor.width(), 224);
}

TEST(Tensor, CountFromAxis) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(1, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor.count(0), 1 * 3 * 224 * 224);
  EXPECT_EQ(tensor.count(1), 3 * 224 * 224);
  EXPECT_EQ(tensor.count(2), 224 * 224);
  EXPECT_EQ(tensor.count(3), 224);
}

TEST(Tensor, OffsetArray) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(1, 3, 2, 2, DataType::FLOAT32, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor.offset(0, 0, 0, 0), 0);
  EXPECT_EQ(tensor.offset(0, 0, 0, 1), 1);
  EXPECT_EQ(tensor.offset(0, 0, 1, 0), 2);
  EXPECT_EQ(tensor.offset(0, 1, 0, 0), 4);
  EXPECT_EQ(tensor.offset(0, 2, 1, 1), 11);
}

TEST(Tensor, Strides) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(1, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  const auto& strides = tensor.strides();
  EXPECT_EQ(strides.size(), 4);

  // bytes 单位
  EXPECT_EQ(strides[0], 3 * 224 * 224 * sizeof(float));
  EXPECT_EQ(strides[1], 224 * 224 * sizeof(float));
  EXPECT_EQ(strides[2], 224 * sizeof(float));
  EXPECT_EQ(strides[3], sizeof(float));
}

TEST(Tensor, ElementSize) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor_f32(1, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor_f32.element_size(), sizeof(float));

  Tensor tensor_i32(1, 3, 224, 224, DataType::INT32, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor_i32.element_size(), sizeof(int32_t));

  Tensor tensor_i8(1, 3, 224, 224, DataType::INT8, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor_i8.element_size(), sizeof(int8_t));

  Tensor tensor_ui8(1, 3, 224, 224, DataType::UINT8, TensorFormat::LINEAR, data);
  EXPECT_EQ(tensor_ui8.element_size(), sizeof(uint8_t));
}

TEST(Tensor, BytesCalculation) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(2, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  int expected_bytes = 2 * 3 * 224 * 224 * sizeof(float);
  EXPECT_EQ(tensor.bytes(), expected_bytes);
  EXPECT_EQ(tensor.bytes(1), 3 * 224 * 224 * sizeof(float));
}

TEST(Tensor, ShapeString) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(1, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  const char* shape_str = tensor.shape_string();
  ASSERT_NE(shape_str, nullptr);
  std::string str(shape_str);
  EXPECT_TRUE(str.find("1") != std::string::npos);
  EXPECT_TRUE(str.find("3") != std::string::npos);
  EXPECT_TRUE(str.find("224") != std::string::npos);
}

TEST(Tensor, Descriptor) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(1, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  const char* desc = tensor.descriptor();
  ASSERT_NE(desc, nullptr);
  std::string str(desc);
  EXPECT_TRUE(str.find("Tensor") != std::string::npos);
  EXPECT_TRUE(str.find("Float32") != std::string::npos);
}

TEST(Tensor, Release) {
  auto data = std::make_shared<void*>(nullptr);
  Tensor tensor(1, 3, 224, 224, DataType::FLOAT32, TensorFormat::LINEAR, data);
  EXPECT_GT(tensor.numel(), 0);
  tensor.release();
  EXPECT_EQ(tensor.ndims(), 0);
  EXPECT_EQ(tensor.numel(), 0);
  EXPECT_EQ(tensor.bytes(), 0);
}

}  // namespace cnstream
