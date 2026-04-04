
#include <cstring>
#include <opencv2/opencv.hpp>
#include "cnstream_logging.hpp"

#include "memop.hpp"
#include "memop_factory.hpp"
#include "libyuv/convert.h"
#include "libyuv/convert_argb.h"


namespace cnstream {

static bool RegisterMemOp() {
  auto& factory = MemOpFactory::Instance();
  bool result = true;
  result &= factory.RegisterMemOpCreator(DevType::CPU, 
    [](int device_id) {
      return std::make_shared<MemOp>();
    });
  return result;
}

static bool memop_registered = RegisterMemOp();

/**
 * @brief 在 Collection 中查找或者注册 buffer
 */
Buffer& MemoryBufferCollection::GetBuffer(DevType type, size_t size, int device_id = -1) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(type);
  if (it != buffers_.end()) {
    if (it->second.size >= size) {
      return it->second;
    }
    buffers_.erase(it);
  }
  Buffer empty_buffer{nullptr, size, device_id};
  auto result = buffers_.emplace(type, std::move(empty_buffer));
  return result.first->second;
}

bool MemoryBufferCollection::Has(DevType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  return buffers_.find(type) != buffers_.end();
}

Buffer* MemoryBufferCollection::Get(DevType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(type);
  if (it != buffers_.end()) {
    return &it->second;
  }
  return nullptr;
}

void MemoryBufferCollection::Clear(DevType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(type);
  if (it != buffers_.end()) {
    buffers_.erase(it);
  }
}

void MemoryBufferCollection::ClearAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  buffers_.clear();
}

size_t MemoryBufferCollection::GetDeviceCount() {
  std::lock_guard<std::mutex> lock(mutex_);
  return buffers_.size();
}

MemOp::MemOp() {}

MemOp::~MemOp() {}

std::shared_ptr<void> MemOp::Allocate(size_t bytes)  {
  size_ = bytes;
  return cnCpuMemAlloc(bytes);
}

void MemOp::Copy(void* dst, const void* src, size_t size)  {
  memcpy(dst, src, size);
}

void MemOp::CopyFromHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

void MemOp::CopyToHost(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

int MemOp::GetDeviceId() const { return -1; }

std::unique_ptr<CNSyncedMemory> MemOp::CreateSyncedMemory(size_t size) {
  return std::make_unique<CNSyncedMemory>(size);
}

/**
 * 调用时，dst_mem 的 size 需设置完成
 */
int MemOp::ConvertImageFormat(CNSyncedMemory* dst_mem, DataFormat dst_fmt,
                              const DecodeFrame* src_frame) {
  if (!dst_mem) return -1;
  void* dst = dst_mem->Allocate();  // GetMutableCpuData
  if (!dst) return -1;
  
  int width = src_frame->width;
  int height = src_frame->height;
  if (dst_fmt != DataFormat::PIXEL_FORMAT_BGR24 &&
      dst_fmt != DataFormat::PIXEL_FORMAT_RGB24) {
    LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
               << static_cast<int>(dst_fmt);
    return -1;
  }
  DataFormat src_fmt = src_frame->fmt;
  if (src_fmt == dst_fmt) {
    LOGW(CORE) << "MemOp::ConvertImageFormat: Source format is same as destination format";
    Copy(dst, src_frame->plane[0], width * height * 3);
    return 0;
  }
  size_t dst_stride = width * 3;
  int ret = 0;
  switch (src_fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        const cv::Mat bgr_mat(height, width, CV_8UC3, const_cast<void*>(src_frame->plane[0]));
        cv::Mat rgb_mat(height, width, CV_8UC3, dst);
        cv::cvtColor(bgr_mat, rgb_mat, cv::COLOR_BGR2RGB);
        ret = 0;
      } else {
        LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source BGR24";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_RGB24: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        const cv::Mat rgb_mat(height, width, CV_8UC3, const_cast<void*>(src_frame->plane[0]));
        cv::Mat bgr_mat(height, width, CV_8UC3, dst);
        cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
        ret = 0;
      } else {
        LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source RGB24";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_YUV420_NV12: {
      if (src_frame->planeNum != 2) {
        LOGE(CORE) << "MemOp::ConvertImageFormat: NV12 format requires 2 planes";
        return -1;
      }
      const uint8_t* y_plane = static_cast<const uint8_t*>(src_frame->plane[0]);
      const uint8_t* uv_plane = static_cast<const uint8_t*>(src_frame->plane[1]);
      int y_stride = width;
      int uv_stride = width;
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = libyuv::NV12ToRAW(
          y_plane, y_stride,
          uv_plane, uv_stride,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = libyuv::NV12ToRGB24(
          y_plane, y_stride,
          uv_plane, uv_stride,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else {
        LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source NV12";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_YUV420_NV21: {
      if (src_frame->planeNum != 2) {
        LOGE(CORE) << "MemOp::ConvertImageFormat: NV21 format requires 2 planes";
        return -1;
      }
      const uint8_t* y_plane = static_cast<const uint8_t*>(src_frame->plane[0]);
      const uint8_t* uv_plane = static_cast<const uint8_t*>(src_frame->plane[1]);
      int y_stride = width;
      int uv_stride = width;
      
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = libyuv::NV21ToRAW(
          y_plane, y_stride,
          uv_plane, uv_stride,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = libyuv::NV21ToRGB24(
          y_plane, y_stride,
          uv_plane, uv_stride,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else {
        LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source NV21";
        return -1;
      }
      break;
    }
    default:
      LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported source format " 
                 << static_cast<int>(src_fmt);
      return -1;
  }
  if (ret != 0) {
    LOGE(CORE) << "MemOp::ConvertImageFormat: libyuv conversion failed with error code: " << ret;
    return ret;
  }
  return 0;
}

}


