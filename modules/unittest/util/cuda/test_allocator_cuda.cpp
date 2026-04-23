
#include "base.hpp"
#include "cnstream_allocator.hpp"  // modules/util/include

#include "cuda/inspect_mem.hpp"
#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_allocator_cuda.hpp"

namespace cnstream {

TEST(CudaAllocator, CudaMemAlloc) {
  int device_id = -1;
  cudaGetDevice(&device_id);
  std::cout << " CudaMemAlloc use device_id: " << device_id << std::endl;
  
  auto allocator = std::make_shared<CudaAllocator>(device_id);
  std::shared_ptr<void> mem = cnMemAlloc(4000, allocator);  // align to 4096 bytes
  ASSERT_NE(mem, nullptr);
  void* ptr = mem.get();
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(allocator->device_id_, device_id);
  EXPECT_EQ(allocator->size_, 4096);

  auto mem2 = cnMemAlloc(4097, allocator);
  ASSERT_NE(mem2, nullptr);
  void* ptr2 = mem2.get();
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ(allocator->size_, 4096 * 2);
}

/**
 * 直接使用 cnCudaMemAlloc 全局函数，得到 RAII 管理的内存
 */
TEST(CudaAllocator, CudaMemAllocLoop) {
  int device_id = -1;
  cudaGetDevice(&device_id);
  std::cout << " CudaMemAllocLoop use device_id: " << device_id << std::endl;
  
  CudaMemInspect inspect(device_id);
  // 循环申请释放内存，查看内存使用
  for (int i = 0; i < 5000; ++i) {
    std::shared_ptr<void> cur_mem = cnCudaMemAlloc(4000, device_id);
    ASSERT_NE(cur_mem, nullptr);
    if (i % 500 == 0 && i > 500) {
      std::cout << "Inspect CudaMemAllocLoop: " << i << " : CUDA Info: " << inspect.GetBriefInfo() << std::endl;
    }
    void* ptr = cur_mem.get();
    ASSERT_NE(ptr, nullptr);
    cur_mem.reset();  // 手动释放，只要确保没有其他引用计数
  }
}




}  // end namespace cnstream
