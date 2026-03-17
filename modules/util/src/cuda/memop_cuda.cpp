// cuda_memop.cpp

#include "memop_factory.hpp"

#include "cnstream_logging.hpp"
#include "cuda/memop_cuda.hpp"
#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_syncmem_cuda.hpp"
#include "cuda/transfmt_cuda.cuh"

namespace cnstream {

static bool RegisterCudaMemOp() {
  auto& factory = MemOpFactory::Instance();
  bool result = true;
  result &= factory.RegisterMemOpCreator(DevType::CUDA,
    [](int dev_id) {
      return std::make_shared<CudaMemOp>(dev_id);
    });
  return result;
}

static bool cuda_memops_registered = RegisterCudaMemOp();

CudaMemOp::CudaMemOp(int dev_id) : device_id_(dev_id) {}

CudaMemOp::~CudaMemOp() {}

std::unique_ptr<CNSyncedMemory> CudaMemOp::CreateSyncedMemory(size_t size) {
  return std::make_unique<CNSyncedMemoryCuda>(size, device_id_);
}

std::shared_ptr<void> CudaMemOp::Allocate(size_t bytes) {
  size_ = bytes;
  CudaDeviceGuard guard(device_id_);
  return cnCudaMemAlloc(bytes, device_id_);
}

void CudaMemOp::Copy(void* dst, const void* src, size_t size) {
  CudaDeviceGuard guard(device_id_);
  CHECK_CUDA_RUNTIME(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

/**
 * @brief 此函数假设 dst_mem 分配的内存应当是“紧密排列”的，即 stride 等于 width * 3
 * 因此我们需注意: src_frame 中的对内存排列不做任何假设
 */
int CudaMemOp::ConvertImageFormat(CNSyncedMemory* dst_mem, DataFormat dst_fmt, 
                                  const DecodeFrame* src_frame) {
  if (!dst_mem) return -1;
#ifdef UNIT_TEST
  auto cuda_mem = dynamic_cast<CNSyncedMemoryCuda*>(dst_mem);
  if (!cuda_mem) {
    LOGE(CORE) << "CudaMemOp::ConvertImageFormat: dst_mem is not CNSyncedMemoryCuda";
    return -1;
  }
  void* dst = cuda_mem->Allocate();
#else
  void* dst = dst_mem->Allocate();
#endif
  if (!dst) return -1;

  int width = src_frame->width;
  int height = src_frame->height;
  DataFormat src_fmt = src_frame->fmt;

  if (dst_fmt != DataFormat::PIXEL_FORMAT_BGR24 &&
      dst_fmt != DataFormat::PIXEL_FORMAT_RGB24) {
    LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
               << static_cast<int>(dst_fmt);
    return -1;
  }
  const int dst_stride = width * 3;
  if (dst_fmt == src_fmt) {
    LOGW(CORE) << "CudaMemOp::ConvertImageFormat: Source format is same as destination format";
    int src_stride = src_frame->stride[0];  // src_fmt：RGB or BGR
    CHECK_CUDA_RUNTIME(cudaMemcpy2D(dst, dst_stride, 
                                src_frame->plane[0], src_stride, 
                                dst_stride,
                                height,
                                cudaMemcpyDeviceToDevice));
    return 0;
  }
  // size_t dst_stride = width * 3;
  int ret = 0;
  switch (src_fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = NppRGB24ToBGR24(dst, width, height, src_frame->plane[0]);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source BGR24";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_RGB24: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = NppBGR24ToRGB24(dst, width, height, src_frame->plane[0]);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source RGB24";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_YUV420_NV12: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = NppNV12ToRGB24(dst, dst_stride,
          src_frame->plane[0], src_frame->plane[1],
          src_frame->stride[0], width, height);
      } else if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = NppNV12ToBGR24(dst, dst_stride,
          src_frame->plane[0], src_frame->plane[1],
          src_frame->stride[0], width, height);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source NV12";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_YUV420_NV21: {
      LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported source format NV21";
      return -1;
    }
    default:
      LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported source format " 
                 << static_cast<int>(src_fmt);
      return -1;
  }
  if (ret != 0) {
    LOGE(CORE) << "CudaMemOp::ConvertImageFormat: libyuv conversion failed with error code: " << ret;
    return ret;
  }
  return 0;
}


}  // namespace cnstream