
#include "data_source_param.hpp"
#include "memop_factory.hpp"
#include "memop.hpp"

#include "base.hpp"

namespace cnstream {


TEST(MemOpFactory, RegisterMemOpCreator) {
  auto& factory = MemOpFactory::Instance();
  factory.PrintRegisteredCreators();  // 此时会显示 CPU MemOp
  auto memop = factory.CreateMemOp(DevType::CPU, -1);  // CreateMemOp 查找已注册的 CPU memop
  ASSERT_TRUE(memop != nullptr);
}


TEST(MemOp, CreateMemOp) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CPU, -1);
  ASSERT_NE(memop, nullptr);
  size_t bytes = 64 * 1024;

  auto synced_mem = memop->CreateSyncedMemory(bytes);
  ASSERT_NE(synced_mem, nullptr);
  EXPECT_EQ(synced_mem->GetSize(), bytes);
  EXPECT_EQ(synced_mem->GetDevId(), -1);
  EXPECT_FALSE(synced_mem->own_dev_data_[DevType::CPU]);

  void* data = synced_mem->Allocate();
  ASSERT_NE(data, nullptr);
  ASSERT_TRUE(synced_mem->own_dev_data_[DevType::CPU]);
  ASSERT_EQ(synced_mem->GetHead(), SyncedHead::HEAD_AT_CPU);
  ASSERT_EQ(synced_mem->GetMutableCpuData(), data);
}

/**
 * @brief 测试图像格式转换
 */
TEST(MemOp, ConvertImageFormat_BGR24_RGB24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CPU, -1);
  ASSERT_TRUE(memop != nullptr);
  
  int width = 1280, height = 1280;
  DecodeFrame* src_frame = CreateTestDecodeFrame(DataFormat::PIXEL_FORMAT_BGR24, width, height);
  uint8_t* bgr_data = static_cast<uint8_t*>(src_frame->plane[0]);
  for (int i = 0; i < width * height; ++i) {
    bgr_data[i * 3] = 255;     // B
    bgr_data[i * 3 + 1] = 128; // G
    bgr_data[i * 3 + 2] = 64;  // R
  }
  
  size_t dst_size = width * height * 3;
  auto dst_mem = memop->CreateSyncedMemory(dst_size);
  ASSERT_NE(dst_mem, nullptr);
  
  int ret = memop->ConvertImageFormat(dst_mem.get(), DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  ASSERT_EQ(ret, 0);
  
  void* rgb_data = const_cast<void*>(dst_mem->GetCpuData());
  uint8_t* rgb_data_8 = static_cast<uint8_t*>(rgb_data);
  
  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(rgb_data_8[i * 3], 64);     // R (原B)
    EXPECT_EQ(rgb_data_8[i * 3 + 1], 128); // G (原G)
    EXPECT_EQ(rgb_data_8[i * 3 + 2], 255); // B (原R)
  }
  CleanupTestDecodeFrame(src_frame);
}

}