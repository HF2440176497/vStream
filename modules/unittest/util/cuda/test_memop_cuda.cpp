
#include "base.hpp"

#include "cuda/cuda_check.hpp"
#include "cuda/memop_cuda.hpp"

#include "memop_factory.hpp"

namespace cnstream {


TEST(CudaMemOpFactory, RegisterCudaMemOpCreator) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  ASSERT_TRUE(memop != nullptr);
  EXPECT_EQ(memop->GetDeviceId(), 0);
}

template<typename T, typename U>
std::unique_ptr<T> memop_unique_pointer_cast(std::unique_ptr<U>&& ptr) {
    if (!ptr) return nullptr;
    
    T* result = dynamic_cast<T*>(ptr.get());
    if (!result) return nullptr;
    
    ptr.release();
    return std::unique_ptr<T>(result);
}


TEST(CudaMemOp, CreateSyncedMemoryAndAllocate) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  ASSERT_NE(memop, nullptr);

  std::shared_ptr<CudaMemOp> cuda_memop = std::dynamic_pointer_cast<CudaMemOp>(memop);
  ASSERT_NE(cuda_memop, nullptr);
  
  size_t bytes = 64 * 4096;
  auto synced_mem = cuda_memop->CreateSyncedMemory(bytes);
  ASSERT_NE(synced_mem, nullptr);
  ASSERT_EQ(synced_mem->GetSize(), bytes);
  ASSERT_EQ(synced_mem->GetDevId(), 0);

  // 2. 针对 sync_mem 进一步测试
  auto cuda_synced_mem = dynamic_cast<CNSyncedMemoryCuda*>(synced_mem.get());
  ASSERT_NE(cuda_synced_mem, nullptr);
  ASSERT_EQ(cuda_synced_mem->GetHead(), SyncedHead::UNINITIALIZED);

  void* data = cuda_synced_mem->Allocate();
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(cuda_synced_mem->GetHead(), SyncedHead::HEAD_AT_CUDA);
  ASSERT_TRUE(cuda_synced_mem->own_dev_data_[DevType::CUDA]);

  void *tmp = malloc(bytes);
  const uint8_t pattern = 0xAB;
  memset(tmp, pattern, bytes);
  CHECK_CUDA_RUNTIME(cudaMemcpy(data, tmp, bytes, cudaMemcpyHostToDevice));

  cuda_synced_mem->ToCpu();
  ASSERT_NE(cuda_synced_mem->cpu_ptr_, nullptr);
  ASSERT_EQ(cuda_synced_mem->GetHead(), SyncedHead::SYNCED);
  ASSERT_TRUE(cuda_synced_mem->own_dev_data_[DevType::CPU]);

  uint8_t* cpu_data = (uint8_t*)cuda_synced_mem->cpu_ptr_;
  for (size_t i = 0; i < bytes; ++i) {
    ASSERT_EQ(cpu_data[i], pattern);
  }
  LOGI(MEMOP_TEST) << "cuda_synced_mem status: " << cuda_synced_mem->StatusToString();
  free(tmp);
}

// ------- 以下验证图像格式转换

// 指定格式，生成 uniform 图像：填充 frame 指针指向显存
DecodeFrame* CreateTestDecodeFrameCuda(DataFormat fmt, int width, int height) {
  DecodeFrame* frame = new DecodeFrame(height, width, fmt);
  frame->fmt = fmt;
  frame->width = width;
  frame->height = height;
  frame->device_id = 0;
  frame->planeNum = 0;

  for (int i = 0; i < FRAME_MAX_PLANES; ++i) {
    frame->plane[i] = nullptr;
    frame->stride[i] = 0;
  }

  const uint8_t R = 10, G = 128, B = 242;
  const uint8_t Y = 111;
  const uint8_t U = 190;
  const uint8_t V = 72;

  size_t frame_size = 0;
  if (fmt == DataFormat::PIXEL_FORMAT_BGR24 || 
      fmt == DataFormat::PIXEL_FORMAT_RGB24) {
    frame->planeNum = 1;
    frame_size = width * height * 3;

    uint8_t* h_data = (uint8_t*)malloc(frame_size);
    for (int i = 0; i < width * height; ++i) {
      if (fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        h_data[i * 3] = B;
        h_data[i * 3 + 1] = G;
        h_data[i * 3 + 2] = R;
      } else {
        h_data[i * 3] = R;
        h_data[i * 3 + 1] = G;
        h_data[i * 3 + 2] = B;
      }
    }

    CHECK_CUDA_RUNTIME(cudaMalloc(&frame->plane[0], frame_size));
    frame->stride[0] = width * 3;
    CHECK_CUDA_RUNTIME(cudaMemcpy(frame->plane[0], h_data, frame_size, cudaMemcpyHostToDevice));

    free(h_data);

  } else if (fmt == DataFormat::PIXEL_FORMAT_YUV420_NV12 ||
             fmt == DataFormat::PIXEL_FORMAT_YUV420_NV21) {
              
    frame->planeNum = 2;
    size_t y_size = width * height;
    size_t uv_size = width * height / 2;
    CHECK_CUDA_RUNTIME(cudaMalloc(&frame->plane[0], y_size));
    CHECK_CUDA_RUNTIME(cudaMalloc(&frame->plane[1], uv_size));
    frame->stride[0] = width;
    frame->stride[1] = std::ceil(1.0 * height * width / 2);

    uint8_t* h_y = (uint8_t*)malloc(y_size);
    uint8_t* h_uv = (uint8_t*)malloc(uv_size);
    memset(h_y, Y, y_size);
    for (size_t i = 0; i < uv_size; i += 2) {
      if (fmt == DataFormat::PIXEL_FORMAT_YUV420_NV12) {
        h_uv[i] = U;
        h_uv[i + 1] = V;
      } else {
        h_uv[i] = V;
        h_uv[i + 1] = U;
      }
    }

    CHECK_CUDA_RUNTIME(cudaMemcpy(frame->plane[0], h_y, y_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_RUNTIME(cudaMemcpy(frame->plane[1], h_uv, uv_size, cudaMemcpyHostToDevice));

    free(h_y);
    free(h_uv);
  }  // end if fmt

  return frame;
}


void clear_decode_frame(DecodeFrame* src_frame) {
  if (src_frame->planeNum == 1) {
    CHECK_CUDA_RUNTIME(cudaFree(src_frame->plane[0]));
  } else if (src_frame->planeNum == 2) {
    CHECK_CUDA_RUNTIME(cudaFree(src_frame->plane[0]));
    CHECK_CUDA_RUNTIME(cudaFree(src_frame->plane[1]));
  } else {
    LOGF(FRAME) << "clear_decode_frame ERROR, fmt not supported.";
  }
  return;
}


TEST(CudaMemOp, ConvertImageFormat_BGR24_RGB24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  std::shared_ptr<CudaMemOp> cuda_memop = std::dynamic_pointer_cast<CudaMemOp>(memop);
  ASSERT_NE(cuda_memop, nullptr);

  // 1. 填充到 frame_plane 显存）uniform data
  int width = 1280, height = 1280;
  DecodeFrame* src_frame = CreateTestDecodeFrameCuda(DataFormat::PIXEL_FORMAT_BGR24, width, height);
  
  uint8_t* h_bgr = (uint8_t*)malloc(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    h_bgr[i * 3] = 255;     // B
    h_bgr[i * 3 + 1] = 128; // G
    h_bgr[i * 3 + 2] = 64;  // R
  }
  CHECK_CUDA_RUNTIME(cudaMemcpy(src_frame->plane[0], h_bgr, width * height * 3, cudaMemcpyHostToDevice));

  // 2. 转换到 dst_mem 显存
  size_t dst_size = width * height * 3;
  auto synced_mem = cuda_memop->CreateSyncedMemory(dst_size);
  ASSERT_NE(synced_mem, nullptr);

  auto cuda_synced_mem = dynamic_cast<CNSyncedMemoryCuda*>(synced_mem.get());
  ASSERT_NE(cuda_synced_mem, nullptr);
  ASSERT_EQ(cuda_synced_mem->GetHead(), SyncedHead::UNINITIALIZED);

  int ret = cuda_memop->ConvertImageFormat(synced_mem.get(), DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  ASSERT_EQ(ret, 0);

  // 3. 拷贝回来比较
  uint8_t* h_rgb = (uint8_t*)malloc(dst_size);
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_rgb, cuda_synced_mem->GetCudaData(), dst_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_rgb[i * 3], 64);      // R (原B)
    EXPECT_EQ(h_rgb[i * 3 + 1], 128); // G (原G)
    EXPECT_EQ(h_rgb[i * 3 + 2], 255); // B (原R)
  }

  free(h_bgr);
  free(h_rgb);

  clear_decode_frame(src_frame);
  delete src_frame;
}

TEST(CudaMemOp, ConvertImageFormat_RGB24_BGR24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  std::shared_ptr<CudaMemOp> cuda_memop = std::dynamic_pointer_cast<CudaMemOp>(memop);
  ASSERT_NE(cuda_memop, nullptr);

  int width = 640, height = 480;
  DecodeFrame* src_frame = CreateTestDecodeFrameCuda(DataFormat::PIXEL_FORMAT_RGB24, width, height);
  uint8_t* h_rgb = (uint8_t*)malloc(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    h_rgb[i * 3] = 100;     // R
    h_rgb[i * 3 + 1] = 150; // G
    h_rgb[i * 3 + 2] = 200; // B
  }
  CHECK_CUDA_RUNTIME(cudaMemcpy(src_frame->plane[0], h_rgb, width * height * 3, cudaMemcpyHostToDevice));

  size_t dst_size = width * height * 3;

  auto synced_mem = cuda_memop->CreateSyncedMemory(dst_size);
  ASSERT_NE(synced_mem, nullptr);

  auto cuda_synced_mem = dynamic_cast<CNSyncedMemoryCuda*>(synced_mem.get());
  ASSERT_NE(cuda_synced_mem, nullptr);
  ASSERT_EQ(cuda_synced_mem->GetHead(), SyncedHead::UNINITIALIZED);
  
  int ret = cuda_memop->ConvertImageFormat(synced_mem.get(), DataFormat::PIXEL_FORMAT_BGR24, src_frame);
  ASSERT_EQ(ret, 0);

  uint8_t* h_bgr = (uint8_t*)malloc(dst_size);
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_bgr, cuda_synced_mem->GetCudaData(), dst_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_bgr[i * 3], 200);     // B (原R)
    EXPECT_EQ(h_bgr[i * 3 + 1], 150); // G (原G)
    EXPECT_EQ(h_bgr[i * 3 + 2], 100); // R (原B)
  }

  free(h_rgb);
  free(h_bgr);

  clear_decode_frame(src_frame);
  delete src_frame;
}


static std::string save_file = "save/output_memop_nv12.jpg";

/**
 * 生成一张 NV12 图片,
 * 借助 memop 的 convert 功能，在 CUDA Synced Memory 中转换为 RGB24 格式
 */
TEST(CudaMemOp, ConvertImageFormat_NV12_RGB24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  std::shared_ptr<CudaMemOp> cuda_memop = std::dynamic_pointer_cast<CudaMemOp>(memop);
  ASSERT_NE(cuda_memop, nullptr);

  int width = 1920, height = 1080;
  DecodeFrame* src_frame = CreateTestDecodeFrameCuda(DataFormat::PIXEL_FORMAT_YUV420_NV12, width, height);

  size_t dst_size = width * height * 3;
  auto synced_mem = cuda_memop->CreateSyncedMemory(dst_size);
  ASSERT_NE(synced_mem, nullptr);

  auto cuda_synced_mem = dynamic_cast<CNSyncedMemoryCuda*>(synced_mem.get());
  ASSERT_NE(cuda_synced_mem, nullptr);
  ASSERT_EQ(cuda_synced_mem->GetHead(), SyncedHead::UNINITIALIZED);
  
  int ret = cuda_memop->ConvertImageFormat(synced_mem.get(), DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  ASSERT_EQ(ret, 0);

  uint8_t* h_rgb = (uint8_t*)malloc(dst_size);
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_rgb, cuda_synced_mem->GetCudaData(), dst_size, cudaMemcpyDeviceToHost));
  EXPECT_NE(h_rgb, nullptr);

  cv::Mat rgb = cv::Mat(height, width, CV_8UC3, h_rgb);
  cv::Mat bgr = cv::Mat(height, width, CV_8UC3);
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  cv::imwrite(save_file, bgr);

  free(h_rgb);
  clear_decode_frame(src_frame);
  delete src_frame;
}

}  // end namespace cnstream