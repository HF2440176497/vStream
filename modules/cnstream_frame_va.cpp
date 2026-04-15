

#include <mutex>
#include <memory>
#include <iostream>

#include "cnstream_frame_va.hpp"
#include "memop_factory.hpp"

namespace cnstream {


static
cv::Mat BGRToBGR(const DataFrame& frame) {
  const cv::Mat bgr(frame.GetHeight(), frame.GetWidth(), CV_8UC3, const_cast<void*>(frame.data_[0]->GetCpuData()));
  return bgr(cv::Rect(0, 0, frame.GetWidth(), frame.GetHeight())).clone();
}

static
cv::Mat RGBToBGR(const DataFrame& frame) {
// #ifdef UNIT_TEST
//   LOGD(FRAME) << "RGBToBGR: width = " << frame.GetWidth() << ", height = " << frame.GetHeight() << std::endl;
//   LOGD(FRAME) << "RGBToBGR: stride = " << frame.GetStride(0) << std::endl;
//   LOGD(FRAME) << "RGBToBGR: data = " << frame.data_[0]->StatusToString() << std::endl;
//   LOGD(FRAME) << "RGBToBGR: data = " << frame.data_[0]->GetCpuData() << std::endl;
// #endif

  const cv::Mat rgb(frame.GetHeight(), frame.GetWidth(), CV_8UC3, const_cast<void*>(frame.data_[0]->GetCpuData()));
  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  return bgr(cv::Rect(0, 0, frame.GetWidth(), frame.GetHeight())).clone();
}

static
cv::Mat YUV420SPToBGR(const DataFrame& frame, bool nv21) {
  const uint8_t* y_plane = reinterpret_cast<const uint8_t*>(frame.data_[0]->GetCpuData());
  const uint8_t* uv_plane = reinterpret_cast<const uint8_t*>(frame.data_[1]->GetCpuData());
  int width = frame.GetWidth();
  int height = frame.GetHeight();
  if (width <= 0 || height <= 1) {
    LOGF(FRAME) << "Invalid width or height, width = " << width << ", height = " << height;
  }
  height = height & (~static_cast<int>(1));

  int y_stride = frame.GetStride(0);
  int uv_stride = frame.GetStride(1);
  cv::Mat bgr(height, width, CV_8UC3);
  uint8_t* dst_bgr24 = bgr.data;
  int dst_stride = width * 3;
  if (nv21)
    libyuv::NV21ToRGB24(y_plane, y_stride, uv_plane, uv_stride,
                        dst_bgr24, dst_stride, width, height);
  else
    libyuv::NV12ToRGB24(y_plane, y_stride, uv_plane, uv_stride,
                        dst_bgr24, dst_stride, width, height);
  return bgr;
}

static inline
cv::Mat NV12ToBGR(const DataFrame& frame) {
  return YUV420SPToBGR(frame, false);
}

static inline
cv::Mat NV21ToBGR(const DataFrame& frame) {
  return YUV420SPToBGR(frame, true);
}

static inline
cv::Mat FrameToImageBGR(const DataFrame& frame) {
  switch (frame.GetFmt()) {
    case DataFormat::PIXEL_FORMAT_BGR24:
      return BGRToBGR(frame);
    case DataFormat::PIXEL_FORMAT_RGB24:
      return RGBToBGR(frame);
    case DataFormat::PIXEL_FORMAT_YUV420_NV12:
      return NV12ToBGR(frame);
    case DataFormat::PIXEL_FORMAT_YUV420_NV21:
      return NV21ToBGR(frame);
    default:
      LOGF(FRAME) << "Unsupported pixel format. fmt[" << static_cast<int>(frame.GetFmt()) << "]";
  }
  return cv::Mat();
}

/**
 * @brief 转换数据到 BGR 格式
 * 在数据存在于 CPU 上，才可调用
 */
cv::Mat DataFrame::GetImage() {
  std::lock_guard<std::mutex> lk(mtx_);
  if (!mat_.empty()) {
    return mat_;
  }
  mat_ = FrameToImageBGR(*this);
  return mat_;
}

size_t DataFrame::GetPlaneBytes(int plane_idx) const {
  if (plane_idx < 0 || plane_idx >= GetPlanes()) return 0;
  switch (fmt_) {
    case DataFormat::PIXEL_FORMAT_BGR24:
    case DataFormat::PIXEL_FORMAT_RGB24:
      return height_ * stride_[0];
    case DataFormat::PIXEL_FORMAT_YUV420_NV12:
    case DataFormat::PIXEL_FORMAT_YUV420_NV21:
      if (0 == plane_idx)
        return height_ * stride_[0];
      else if (1 == plane_idx)
        return std::ceil(1.0 * height_ * stride_[1] / 2);
      else
        LOGF(FRAME) << "plane index wrong.";
        return 0;
    default:
      return 0;
  }
  return 0;
}

size_t DataFrame::GetBytes() const {
  size_t bytes = 0;
  for (int i = 0; i < GetPlanes(); ++i) {
    bytes += GetPlaneBytes(i);
  }
  return bytes;
}

/**
 * @brief 每次调用查找已注册的 MemOp 创建器，根据当前 device_type 和 device_id 创建 MemOp
 * 调用处：CopyToSyncMem(dec_frame)
 */
std::shared_ptr<MemOp> DataFrame::CreateMemOp() {
  auto device_type = this->ctx_.device_type;
  int device_id = this->ctx_.device_id;
  std::shared_ptr<MemOp> memop = MemOpFactory::Instance().CreateMemOp(device_type, device_id);  // inside mutex_ lock
  if (!memop) {
    LOGF(FRAME) << "CreateMemOp: failed to create MemOp from " << static_cast<int>(device_type) << " with device_id " << device_id;
    return nullptr;
  }
  return memop;
}

void DataFrame::CopyToSyncMem(DecodeFrame* dec_frame) {
  if (this->ctx_.device_type == DevType::INVALID) {
    LOGF(FRAME) << "CopyToSyncMem: device_type is INVALID";
    return;
  }
  if (DataFormat::PIXEL_FORMAT_RGB24 != this->fmt_ && DataFormat::PIXEL_FORMAT_BGR24 != this->fmt_) {
    LOGF(FRAME) << "CopyToSyncMem: fmt not RGB24 or BGR24, this fmt is " << static_cast<int>(this->fmt_);
    return;
  }

  std::shared_ptr<MemOp> memop = CreateMemOp();
  if (!memop) return;

  // reuse DecodeFrame plane buffer
  if (this->deAllocator_ != nullptr && dec_frame->fmt == this->fmt_) {
    for (int i = 0; i < GetPlanes(); i++) {
      const size_t plane_bytes = GetPlaneBytes(i);
      this->data_[i] = memop->CreateSyncedMemory(plane_bytes);
      this->data_[i]->SetData(dec_frame->plane[i]);
    }
    return;
  }
  const size_t bytes = GetPlaneBytes(0);
  this->data_[0] = memop->CreateSyncedMemory(bytes);
  int ret = memop->ConvertImageFormat(this->data_[0].get(), this->fmt_, dec_frame);
  if (ret != 0) {
    LOGF(FRAME) << "CopyToSyncMem: Format conversion failed with error code: " << ret;
    return;
  }

#ifdef UNIT_TEST
  LOGD(FRAME) << "CopyToSyncMem image width: " << width_ << ", height: " << height_ << ", alloca size: " << bytes;
  // LOGD(FRAME) << "------------- CopyToSyncMem sync mem status: " << this->data_[0]->StatusToString();
#endif

  this->deAllocator_.reset();
}

bool InferObject::AddAttribute(const std::string& key, const InferAttr& value) {
  std::lock_guard<std::mutex> lk(attribute_mutex_);
  if (attributes_.find(key) != attributes_.end()) return false;

  attributes_.insert(std::make_pair(key, value));
  return true;
}

bool InferObject::AddAttribute(const std::pair<std::string, InferAttr>& attribute) {
  std::lock_guard<std::mutex> lk(attribute_mutex_);
  if (attributes_.find(attribute.first) != attributes_.end()) return false;

  attributes_.insert(attribute);
  return true;
}

InferAttr InferObject::GetAttribute(const std::string& key) {
  std::lock_guard<std::mutex> lk(attribute_mutex_);
  if (attributes_.find(key) != attributes_.end()) return attributes_[key];
  return InferAttr();
}

bool InferObject::AddExtraAttribute(const std::string& key, const std::string& value) {
  std::lock_guard<std::mutex> lk(attribute_mutex_);
  if (extra_attributes_.find(key) != extra_attributes_.end()) return false;

  extra_attributes_.insert(std::make_pair(key, value));
  return true;
}

bool InferObject::AddExtraAttributes(const std::vector<std::pair<std::string, std::string>>& attributes) {
  std::lock_guard<std::mutex> lk(attribute_mutex_);
  bool ret = true;
  for (auto& attribute : attributes) {
    ret &= AddExtraAttribute(attribute.first, attribute.second);
  }
  return ret;
}

std::string InferObject::GetExtraAttribute(const std::string& key) {
  std::lock_guard<std::mutex> lk(attribute_mutex_);
  if (extra_attributes_.find(key) != extra_attributes_.end()) {
    return extra_attributes_[key];
  }
  return "";
}

bool InferObject::RemoveExtraAttribute(const std::string& key) {
  std::lock_guard<std::mutex> lk(attribute_mutex_);
  if (extra_attributes_.find(key) != extra_attributes_.end()) {
    extra_attributes_.erase(key);
  }
  return true;
}

StringPairs InferObject::GetExtraAttributes() {
  std::lock_guard<std::mutex> lk(attribute_mutex_);
  return StringPairs(extra_attributes_.begin(), extra_attributes_.end());
}


bool InferObject::AddFeature(const std::string& key, const InferFeature& feature) {
  std::lock_guard<std::mutex> lk(feature_mutex_);
  if (features_.find(key) != features_.end()) {
    return false;
  }
  features_.insert(std::make_pair(key, feature));
  return true;
}

InferFeature InferObject::GetFeature(const std::string& key) {
  std::lock_guard<std::mutex> lk(feature_mutex_);
  if (features_.find(key) != features_.end()) {
    return features_[key];
  }
  return InferFeature();
}

InferFeatures InferObject::GetFeatures() {
  std::lock_guard<std::mutex> lk(feature_mutex_);
  return InferFeatures(features_.begin(), features_.end());
}

}  // namespace cnstream
