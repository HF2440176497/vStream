/**
 * @file test_syncmem.cpp
 *
 * This file contains gtest unit tests for the CNSyncedMemory class on CUDA platform.
 */

#include <vector>
#include <cstring>

#include "base.hpp"
#include "cnstream_logging.hpp"

#ifdef VSTREAM_USE_CUDA
#include "cuda/cnstream_syncmem_cuda.hpp"
#endif


class CNSyncedMemoryTest : public ::testing::Test {
 protected:
  static const int kFloatCount = 1024;
  static const int kTestSize = 1024 * sizeof(float);
};

const int CNSyncedMemoryTest::kFloatCount;
const int CNSyncedMemoryTest::kTestSize;

namespace cnstream {

#ifdef VSTREAM_USE_CUDA

/**
 * 在 CPU 上设置数据 —— 转移到 CUDA —— 转移回 CPU 并验证
 */
TEST_F(CNSyncedMemoryTest, BasicFunctionality) {
  CNSyncedMemoryCuda mem(kTestSize, 0);

  // Set data on CPU
  float* cpu_data = (float*)(mem.GetMutableCpuData());
  ASSERT_NE(cpu_data, nullptr) << "Failed to get mutable CPU data";
  ASSERT_EQ(mem.GetHead(), SyncedHead::HEAD_AT_CPU) << "Head should be HEAD_AT_CPU";
  for (int i = 0; i < kFloatCount; i++) {
    cpu_data[i] = static_cast<float>(i);
  }
  // allocate on CUDA, copy data from CPU to CUDA
  mem.ToCuda();
  ASSERT_EQ(mem.GetHead(), SyncedHead::SYNCED) << "Head should be SYNCED";

  // allocate -> own
  ASSERT_TRUE(mem.own_dev_data_[DevType::CPU]) << "CPU data should be owned";
  ASSERT_TRUE(mem.own_dev_data_[DevType::CUDA]) << "CUDA data should be owned";

  const float* cuda_data = static_cast<const float*>(mem.GetCudaData());
  ASSERT_NE(cuda_data, nullptr) << "Failed to get CUDA data";

  // Transfer back to CPU and verify
  float* cpu_data2 = (float*)(mem.GetMutableCpuData());
  ASSERT_EQ(mem.GetHead(), SyncedHead::HEAD_AT_CPU) << "Head should be HEAD_AT_CPU after GetMutableCpuData";
  ASSERT_NE(cpu_data2, nullptr) << "Failed to get mutable CPU data after CUDA transfer";

  // GetMutableCpuData 之前是同步的，所以不应该影响 own statuss
  ASSERT_TRUE(mem.own_dev_data_[DevType::CPU]) << "CPU data should be owned";
  ASSERT_TRUE(mem.own_dev_data_[DevType::CUDA]) << "CUDA data should be owned";

  for (int i = 0; i < kFloatCount; i++) {
    ASSERT_FLOAT_EQ(cpu_data2[i], static_cast<float>(i)) << "Data mismatch at index " << i;
  }
}

// Test 2: Device context
TEST_F(CNSyncedMemoryTest, DeviceContext) {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    GTEST_SKIP() << "No CUDA devices found, skipping device context test";
  }
  const int device_id = 0; // Use first CUDA device
  CNSyncedMemoryCuda mem(kTestSize, device_id);
  ASSERT_EQ(mem.GetDevId(), device_id) << "Device ID mismatch";

  float* cpu_data = (float*)(mem.GetMutableCpuData());
  ASSERT_NE(cpu_data, nullptr) << "Failed to get mutable CPU data";

  for (int i = 0; i < kFloatCount; i++) {
    cpu_data[i] = static_cast<float>(i * 2);
  }
  mem.ToCuda();
  auto cuda_data = mem.GetCudaData();
  ASSERT_NE(cuda_data, nullptr) << "Data transfer with device context failed";
}

TEST_F(CNSyncedMemoryTest, MemoryManagement) {
  CNSyncedMemoryCuda mem(kTestSize, 0);

  float* manual_cuda_ptr;
  CHECK_CUDA_RUNTIME(cudaMalloc(&manual_cuda_ptr, kTestSize));
  
  mem.SetCudaData(manual_cuda_ptr);
  ASSERT_FALSE(mem.own_dev_data_[DevType::CUDA]) << "CUDA data ownership should be false after SetCudaData";
  ASSERT_EQ(mem.GetHead(), SyncedHead::HEAD_AT_CUDA) << "Head should be HEAD_AT_CUDA after SetCudaData";

  const float* const_cuda_data = static_cast<const float*>(mem.GetCudaData());
  ASSERT_EQ(const_cast<float*>(const_cuda_data), mem.cuda_ptr_);
  ASSERT_EQ(mem.cpu_ptr_, nullptr);

  mem.ToCpu();
  ASSERT_NE(mem.cpu_ptr_, nullptr) << "CPU data pointer should not be NULL after ToCpu";
  ASSERT_TRUE(mem.own_dev_data_[DevType::CPU]) << "CPU data ownership should be true after ToCpu";
  ASSERT_FALSE(mem.own_dev_data_[DevType::CUDA]) << "CUDA data ownership should be true after ToCpu";
  ASSERT_EQ(mem.GetHead(), SyncedHead::SYNCED) << "Head should be HEAD_AT_CPU after ToCpu";

  // 1）之前是同步的，因此只改变 head 为 HEAD_AT_CUDA
  float* cuda_data = (float*)(mem.GetMutableCudaData());
  ASSERT_NE(cuda_data, nullptr) << "Failed to get CUDA data after ToCpu";
  ASSERT_EQ(mem.GetHead(), SyncedHead::HEAD_AT_CUDA);

  // 前面已经分配的 cpu 内存，这时应当还存在
  ASSERT_NE(mem.cpu_ptr_, nullptr);
  ASSERT_EQ(mem.cuda_ptr_, (void*)cuda_data);

  // 尝试改变 CUDA 数据
  void *tmp = malloc(kTestSize);
  float pattern = static_cast<float>(0xAB);
  float* tmp2 = (float*)tmp;
  for (int i = 0; i < kFloatCount; i++) {
    tmp2[i] = pattern;
  }

  CHECK_CUDA_RUNTIME(cudaMemcpy(cuda_data, tmp, kTestSize, cudaMemcpyHostToDevice));

  ASSERT_EQ(mem.GetSize(), kTestSize);

  // 2）接着 ToCpu, 会分配 cpu 内存, 接着验证 cpu 上的数据是否正确
  mem.ToCpu();
  ASSERT_EQ(mem.GetHead(), SyncedHead::SYNCED);
  ASSERT_TRUE(mem.own_dev_data_[DevType::CPU]) << "CPU data ownership should be true after ToCpu";

  const float* cpu_data = static_cast<const float*>(mem.GetCpuData());
  ASSERT_NE(cpu_data, nullptr);
  ASSERT_EQ(mem.cpu_ptr_, (void*)cpu_data);
  
  for (int i = 0; i < kFloatCount; i++) {
    ASSERT_EQ(cpu_data[i], pattern) << "Data mismatch at index " << i;
  }

  cudaFree(manual_cuda_ptr);
  free(tmp);
}

TEST_F(CNSyncedMemoryTest, AllocateMethod) {
  CNSyncedMemoryCuda mem(kTestSize, 0);

  ASSERT_EQ(mem.GetHead(), SyncedHead::UNINITIALIZED);
  
  void* ptr = mem.Allocate();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(mem.GetHead(), SyncedHead::HEAD_AT_CUDA);
  ASSERT_TRUE(mem.own_dev_data_[DevType::CUDA]);
  ASSERT_EQ(mem.cuda_ptr_, ptr);

  void* ptr2 = mem.Allocate();
  ASSERT_EQ(ptr2, ptr);
}

#endif

}  // end namespace