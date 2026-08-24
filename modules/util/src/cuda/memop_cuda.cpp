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
    [](int device_id) {
      return std::make_shared<CudaMemOp>(device_id);
    });
  return result;
}

static bool _cuda_memop_registered = RegisterCudaMemOp();

CudaMemOp::CudaMemOp(int device_id) : device_id_(device_id) {}

CudaMemOp::~CudaMemOp() {}

/**
 * note: 仅能通过 memop 来创建 synced memory
 */
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
 * host to device
 */
void CudaMemOp::CopyFromHost(void* dst, const void* src, size_t size) {
  CudaDeviceGuard guard(device_id_);
  CHECK_CUDA_RUNTIME(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

/**
 * device to host
 */
void CudaMemOp::CopyToHost(void* dst, const void* src, size_t size) {
  CudaDeviceGuard guard(device_id_);
  CHECK_CUDA_RUNTIME(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void CudaMemOp::CopyFromHostAsync(void* dst, const void* src, size_t size, void* stream) {
  CudaDeviceGuard guard(device_id_);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  CHECK_CUDA_RUNTIME(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cuda_stream));
}

void CudaMemOp::CopyToHostAsync(void* dst, const void* src, size_t size, void* stream) {
  CudaDeviceGuard guard(device_id_);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  CHECK_CUDA_RUNTIME(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, cuda_stream));
}

void CudaMemOp::SyncStream(void* stream) {
  if (stream) {
    CHECK_CUDA_RUNTIME(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }
}

/**
 * @brief dst_mem 分配的内存使用 GetStride_8U_C3 对齐后的 stride，
 *        确保满足 NPP 函数的对齐要求（4 字节对齐）。
 *        src_frame 中的 stride 不做任何假设，从 src_frame->stride 读取。
 */
int CudaMemOp::ConvertImageFormat(CNSyncedMemory* dst_mem, DataFormat dst_fmt, 
                                  const DecodeFrame* src_frame,
                                  void* stream) {
  if (!dst_mem) return -1;
  CudaDeviceGuard guard(device_id_);
  auto cuda_stream = static_cast<cudaStream_t>(stream);

  void* dst = dst_mem->Allocate();
  if (!dst) return -1;

  int width = src_frame->width;
  int height = src_frame->height;
  DataFormat src_fmt = src_frame->fmt;

  if (src_frame->device_type == DevType::CUDA && src_frame->device_id != device_id_) {
    LOGE(CORE) << "ConvertImageFormat: source frame is on CUDA device "
               << src_frame->device_id << " but CudaMemOp is bound to device " << device_id_
               << ", cross-device conversion is not supported";
    return -1;
  }

  if (dst_fmt != DataFormat::PIXEL_FORMAT_BGR24 &&
      dst_fmt != DataFormat::PIXEL_FORMAT_RGB24) {
    LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
               << static_cast<int>(dst_fmt);
    return -1;
  }
  // SourceRender::Process 中设置 DataFrame fmt 时，stride 步长设置需一致
  const int dst_stride = GetStride_8U_C3(width);

  // 对于 src_stride
  // （1）For handler_send handler_image, src_frame 是紧密排列的
  // （2）For ffmpeg, src_frame 来自 line_size
  if (dst_fmt == src_fmt) {
    LOGD(CORE) << "CudaMemOp::ConvertImageFormat: Source format is same as destination format";
    int src_stride = src_frame->stride[0];
    CHECK_CUDA_RUNTIME(cudaMemcpy2DAsync(dst, dst_stride, 
                                src_frame->plane[0], src_stride, 
                                width * 3,
                                height,
                                cudaMemcpyDeviceToDevice,
                                cuda_stream));
    return 0;
  }
  int ret = 0;

  switch (src_fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = NppRGB24ToBGR24(dst, dst_stride, width, height,
                              src_frame->plane[0], src_frame->stride[0],
                              cuda_stream);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source BGR24";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_RGB24: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = NppBGR24ToRGB24(dst, dst_stride, width, height,
                              src_frame->plane[0], src_frame->stride[0],
                              cuda_stream);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source RGB24";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_YUV420_NV12: {
      if (src_frame->stride[0] != src_frame->stride[1]) {
        LOGW(CORE) << "ConvertImageFormat: NV12 stride[0](" << src_frame->stride[0]
                   << ") != stride[1](" << src_frame->stride[1]
                   << "), NPP requires equal strides, using stride[0]";
      }
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = NppNV12ToRGB24(dst, dst_stride,
          src_frame->plane[0], src_frame->plane[1],
          src_frame->stride[0], width, height,
          cuda_stream);
      } else if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = NppNV12ToBGR24(dst, dst_stride,
          src_frame->plane[0], src_frame->plane[1],
          src_frame->stride[0], width, height,
          cuda_stream);
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
    LOGE(CORE) << "CudaMemOp::ConvertImageFormat: conversion failed with error code: " << ret;
    return ret;
  }
  return 0;
}


}  // namespace cnstream