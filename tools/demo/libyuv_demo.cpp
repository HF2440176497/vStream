#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "libyuv/convert.h"
#include "libyuv/convert_from_argb.h"

// 定义像素格式枚举 (与项目中的 DataFormat 对应)
enum PixelFormat { 
    FORMAT_BGR24, 
    FORMAT_RGB24, 
    FORMAT_YUV420_NV12, 
    FORMAT_YUV420_NV21, 
    FORMAT_UNKNOWN 
};

const char* GetFormatName(PixelFormat format) {
  switch (format) {
    case FORMAT_BGR24:
      return "BGR24";
    case FORMAT_RGB24:
      return "RGB24";
    case FORMAT_YUV420_NV12:
      return "NV12";
    case FORMAT_YUV420_NV21:
      return "NV21";
    default:
      return "UNKNOWN";
  }
}

// 获取plane数量
int GetPlanes(PixelFormat format) {
  switch (format) {
    case FORMAT_BGR24:
    case FORMAT_RGB24:
      return 1;
    case FORMAT_YUV420_NV12:
    case FORMAT_YUV420_NV21:
      return 2;
    default:
      return 0;
  }
}

struct ImageData {
  PixelFormat          fmt;
  int                  width;
  int                  height;
  int                  stride[2];  // stride[0] for Y/RGB, stride[1] for UV
  std::vector<uint8_t> plane[2];   // 最多2个平面

  ImageData() : fmt(FORMAT_UNKNOWN), width(0), height(0) { stride[0] = stride[1] = 0; }

  void PrintInfo(const char* prefix) const {
    std::cout << "[" << prefix << "] Format: " << GetFormatName(fmt) << ", Width: " << width << ", Height: " << height
              << ", Planes: " << GetPlanes(fmt);
    if (fmt == FORMAT_YUV420_NV12 || fmt == FORMAT_YUV420_NV21) {
      std::cout << ", StrideY: " << stride[0] << ", StrideUV: " << stride[1];
      std::cout << ", Y_Size: " << plane[0].size() << ", UV_Size: " << plane[1].size();
    } else {
      std::cout << ", Stride: " << stride[0];
      std::cout << ", DataSize: " << plane[0].size();
    }
    std::cout << " bytes" << std::endl;
  }
};

bool ValidateImageData(const ImageData& img) {
  if (img.width <= 0 || img.height <= 0) {
    std::cerr << "Validation failed: Invalid dimensions" << std::endl;
    return false;
  }
  int planes = GetPlanes(img.fmt);
  for (int i = 0; i < planes; ++i) {
    if (img.plane[i].empty()) {
      std::cerr << "Validation failed: Plane " << i << " is empty" << std::endl;
      return false;
    }
  }
  bool   all_zero = true;
  size_t check_size = std::min(img.plane[0].size(), static_cast<size_t>(100));
  for (size_t i = 0; i < check_size; ++i) {
    if (img.plane[0][i] != 0) {
      all_zero = false;
      break;
    }
  }
  if (all_zero) {
    std::cerr << "Warning: First 100 bytes are all zero, conversion may have failed" << std::endl;
    return false;
  }
  return true;
}

ImageData MatToBGR24(const cv::Mat& mat) {
  ImageData img;
  img.fmt = FORMAT_BGR24;
  img.width = mat.cols;
  img.height = mat.rows;
  img.stride[0] = mat.cols * 3;
  img.stride[1] = 0;

  cv::Mat bgr_mat;
  if (mat.channels() == 3) {
    bgr_mat = mat.clone();
  } else if (mat.channels() == 4) {
    cv::cvtColor(mat, bgr_mat, cv::COLOR_BGRA2BGR);
  } else if (mat.channels() == 1) {
    cv::cvtColor(mat, bgr_mat, cv::COLOR_GRAY2BGR);
  } else {
    bgr_mat = mat.clone();
  }
  img.plane[0].resize(bgr_mat.total() * bgr_mat.elemSize());
  memcpy(img.plane[0].data(), bgr_mat.data, img.plane[0].size());
  return img;
}

ImageData MatToRGB24(const cv::Mat& mat) {
  ImageData img;
  img.fmt = FORMAT_RGB24;
  img.width = mat.cols;
  img.height = mat.rows;
  img.stride[0] = mat.cols * 3;
  img.stride[1] = 0;

  cv::Mat rgb_mat;
  if (mat.channels() == 3) {
    cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
  } else if (mat.channels() == 4) {
    cv::cvtColor(mat, rgb_mat, cv::COLOR_BGRA2RGB);
  } else if (mat.channels() == 1) {
    cv::cvtColor(mat, rgb_mat, cv::COLOR_GRAY2RGB);
  } else {
    rgb_mat = mat.clone();
  }

  img.plane[0].resize(rgb_mat.total() * rgb_mat.elemSize());
  memcpy(img.plane[0].data(), rgb_mat.data, img.plane[0].size());
  return img;
}

ImageData BGR24ToRGB24(const ImageData& bgr24) {
  ImageData img;
  img.fmt = FORMAT_RGB24;
  img.width = bgr24.width;
  img.height = bgr24.height;
  img.stride[0] = bgr24.width * 3;
  img.stride[1] = 0;

  const cv::Mat bgr_mat(bgr24.height, bgr24.width, CV_8UC3, const_cast<uint8_t*>(bgr24.plane[0].data()));
  img.plane[0].resize(bgr24.height * bgr24.width * 3);
  cv::Mat rgb_mat(bgr24.height, bgr24.width, CV_8UC3, img.plane[0].data());
  cv::cvtColor(bgr_mat, rgb_mat, cv::COLOR_BGR2RGB);
  return img;
}

ImageData RGB24ToBGR24(const ImageData& rgb24) {
  ImageData img;
  img.fmt = FORMAT_BGR24;
  img.width = rgb24.width;
  img.height = rgb24.height;
  img.stride[0] = rgb24.width * 3;
  img.stride[1] = 0;
  const cv::Mat rgb_mat(rgb24.height, rgb24.width, CV_8UC3, const_cast<uint8_t*>(rgb24.plane[0].data()));
  img.plane[0].resize(rgb24.height * rgb24.width * 3);
  cv::Mat bgr_mat(rgb24.height, rgb24.width, CV_8UC3, img.plane[0].data());
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
  return img;
}

// BGR24 转 NV12 (使用 libyuv ARGB 中间格式)
ImageData BGR24ToNV12(const ImageData& bgr24) {
  ImageData img;
  img.fmt = FORMAT_YUV420_NV12;
  img.width = bgr24.width;
  img.height = bgr24.height;
  img.stride[0] = bgr24.width;
  img.stride[1] = bgr24.width;

  // 高度对齐到偶数
  int height = bgr24.height & (~1);

  size_t y_size = bgr24.width * height;
  size_t uv_size = bgr24.width * height / 2;
  img.plane[0].resize(y_size);
  img.plane[1].resize(uv_size);

  // BGR24 -> ARGB
  std::vector<uint8_t> argb_buffer(bgr24.width * height * 4);
  int                  argb_stride = bgr24.width * 4;
  libyuv::RGB24ToARGB(bgr24.plane[0].data(), bgr24.stride[0], argb_buffer.data(), argb_stride, bgr24.width, height);

  // ARGB -> NV12
  libyuv::ARGBToNV12(argb_buffer.data(), argb_stride, img.plane[0].data(), img.stride[0], img.plane[1].data(),
                     img.stride[1], bgr24.width, height);

  return img;
}

// BGR24 转 NV21 (使用 libyuv ARGB 中间格式)
ImageData BGR24ToNV21(const ImageData& bgr24) {
  ImageData img;
  img.fmt = FORMAT_YUV420_NV21;
  img.width = bgr24.width;
  img.height = bgr24.height;
  img.stride[0] = bgr24.width;
  img.stride[1] = bgr24.width;

  // 高度对齐到偶数
  int height = bgr24.height & (~1);

  size_t y_size = bgr24.width * height;
  size_t vu_size = bgr24.width * height / 2;
  img.plane[0].resize(y_size);
  img.plane[1].resize(vu_size);

  // BGR24 -> ARGB
  std::vector<uint8_t> argb_buffer(bgr24.width * height * 4);
  int                  argb_stride = bgr24.width * 4;
  libyuv::RGB24ToARGB(bgr24.plane[0].data(), bgr24.stride[0], argb_buffer.data(), argb_stride, bgr24.width, height);

  // ARGB -> NV21
  libyuv::ARGBToNV21(argb_buffer.data(), argb_stride, img.plane[0].data(), img.stride[0], img.plane[1].data(),
                     img.stride[1], bgr24.width, height);

  return img;
}

// RGB24 转 NV12 (使用 libyuv ARGB 中间格式)
ImageData RGB24ToNV12(const ImageData& rgb24) {
  ImageData img;
  img.fmt = FORMAT_YUV420_NV12;
  img.width = rgb24.width;
  img.height = rgb24.height;
  img.stride[0] = rgb24.width;
  img.stride[1] = rgb24.width;

  // 高度对齐到偶数
  int height = rgb24.height & (~1);

  size_t y_size = rgb24.width * height;
  size_t uv_size = rgb24.width * height / 2;
  img.plane[0].resize(y_size);
  img.plane[1].resize(uv_size);

  // RGB24 -> ARGB
  std::vector<uint8_t> argb_buffer(rgb24.width * height * 4);
  int                  argb_stride = rgb24.width * 4;
  libyuv::RAWToARGB(rgb24.plane[0].data(), rgb24.stride[0], argb_buffer.data(), argb_stride, rgb24.width, height);

  // ARGB -> NV12
  libyuv::ARGBToNV12(argb_buffer.data(), argb_stride, img.plane[0].data(), img.stride[0], img.plane[1].data(),
                     img.stride[1], rgb24.width, height);

  return img;
}

// RGB24 转 NV21 (使用 libyuv ARGB 中间格式)
ImageData RGB24ToNV21(const ImageData& rgb24) {
  ImageData img;
  img.fmt = FORMAT_YUV420_NV21;
  img.width = rgb24.width;
  img.height = rgb24.height;
  img.stride[0] = rgb24.width;
  img.stride[1] = rgb24.width;

  // 高度对齐到偶数
  int height = rgb24.height & (~1);

  size_t y_size = rgb24.width * height;
  size_t vu_size = rgb24.width * height / 2;
  img.plane[0].resize(y_size);
  img.plane[1].resize(vu_size);

  // RGB24 -> ARGB
  std::vector<uint8_t> argb_buffer(rgb24.width * height * 4);
  int                  argb_stride = rgb24.width * 4;
  libyuv::RAWToARGB(rgb24.plane[0].data(), rgb24.stride[0], argb_buffer.data(), argb_stride, rgb24.width, height);

  // ARGB -> NV21
  libyuv::ARGBToNV21(argb_buffer.data(), argb_stride, img.plane[0].data(), img.stride[0], img.plane[1].data(),
                     img.stride[1], rgb24.width, height);

  return img;
}

// NV12 转 RGB24 (与项目中 memop.cpp 方式一致)
ImageData NV12ToRGB24(const ImageData& nv12) {
  ImageData img;
  img.fmt = FORMAT_RGB24;
  img.width = nv12.width;
  img.height = nv12.height;
  img.stride[0] = nv12.width * 3;
  img.stride[1] = 0;

  int height = nv12.height & (~1);
  img.plane[0].resize(nv12.width * height * 3);

  const uint8_t* y_plane = nv12.plane[0].data();
  const uint8_t* uv_plane = nv12.plane[1].data();
  int            y_stride = nv12.stride[0];
  int            uv_stride = nv12.stride[1];
  size_t         dst_stride = nv12.width * 3;

  // libyuv::NV12ToRGB24(
  //     y_plane, y_stride,
  //     uv_plane, uv_stride,
  //     img.plane[0].data(), dst_stride,
  //     nv12.width, height);

  // // BGR24->RGB24
  // cv::Mat bgr_mat(height, nv12.width, CV_8UC3, img.plane[0].data());
  // cv::Mat rgb_mat(height, nv12.width, CV_8UC3, img.plane[0].data());
  // cv::cvtColor(bgr_mat, rgb_mat, cv::COLOR_BGR2RGB);
  libyuv::NV12ToRAW(y_plane, y_stride, uv_plane, uv_stride, img.plane[0].data(), dst_stride, nv12.width, height);

  return img;
}

// NV12 转 BGR24 (与项目中 memop.cpp 方式一致：先转RGB24再用OpenCV转BGR)
ImageData NV12ToBGR24(const ImageData& nv12) {
  ImageData img;
  img.fmt = FORMAT_BGR24;
  img.width = nv12.width;
  img.height = nv12.height;
  img.stride[0] = nv12.width * 3;
  img.stride[1] = 0;

  int height = nv12.height & (~1);
  img.plane[0].resize(nv12.width * height * 3);

  const uint8_t* y_plane = nv12.plane[0].data();
  const uint8_t* uv_plane = nv12.plane[1].data();
  int            y_stride = nv12.stride[0];
  int            uv_stride = nv12.stride[1];
  size_t         dst_stride = nv12.width * 3;

  libyuv::NV12ToRGB24(y_plane, y_stride, uv_plane, uv_stride, img.plane[0].data(), dst_stride, nv12.width, height);

  return img;
}

// NV21 转 RGB24 (与项目中 memop.cpp 方式一致)
ImageData NV21ToRGB24(const ImageData& nv21) {
  ImageData img;
  img.fmt = FORMAT_RGB24;
  img.width = nv21.width;
  img.height = nv21.height;
  img.stride[0] = nv21.width * 3;
  img.stride[1] = 0;

  int height = nv21.height & (~1);
  img.plane[0].resize(nv21.width * height * 3);

  const uint8_t* y_plane = nv21.plane[0].data();
  const uint8_t* vu_plane = nv21.plane[1].data();
  int            y_stride = nv21.stride[0];
  int            vu_stride = nv21.stride[1];
  size_t         dst_stride = nv21.width * 3;

  // libyuv::NV21ToRGB24(
  //     y_plane, y_stride,
  //     vu_plane, vu_stride,
  //     img.plane[0].data(), dst_stride,
  //     nv21.width, height);

  // // BGR24->RGB24
  // cv::Mat bgr_mat(height, nv21.width, CV_8UC3, img.plane[0].data());
  // cv::Mat rgb_mat(height, nv21.width, CV_8UC3, img.plane[0].data());
  // cv::cvtColor(bgr_mat, rgb_mat, cv::COLOR_BGR2RGB);

  libyuv::NV21ToRAW(y_plane, y_stride, vu_plane, vu_stride, img.plane[0].data(), dst_stride, nv21.width, height);

  return img;
}

ImageData NV21ToBGR24(const ImageData& nv21) {
  ImageData img;
  img.fmt = FORMAT_BGR24;
  img.width = nv21.width;
  img.height = nv21.height;
  img.stride[0] = nv21.width * 3;
  img.stride[1] = 0;

  int height = nv21.height & (~1);
  img.plane[0].resize(nv21.width * height * 3);

  const uint8_t* y_plane = nv21.plane[0].data();
  const uint8_t* vu_plane = nv21.plane[1].data();
  int            y_stride = nv21.stride[0];
  int            vu_stride = nv21.stride[1];
  size_t         dst_stride = nv21.width * 3;

  libyuv::NV21ToRGB24(y_plane, y_stride, vu_plane, vu_stride, img.plane[0].data(), dst_stride, nv21.width, height);

  return img;
}

// NV12 转 NV21 (UV 平面字节交换)
ImageData NV12ToNV21(const ImageData& nv12) {
  ImageData img;
  img.fmt = FORMAT_YUV420_NV21;
  img.width = nv12.width;
  img.height = nv12.height;
  img.stride[0] = nv12.stride[0];
  img.stride[1] = nv12.stride[1];

  int    height = nv12.height & (~1);
  size_t y_size = nv12.width * height;
  size_t vu_size = nv12.width * height / 2;

  img.plane[0].resize(y_size);
  img.plane[1].resize(vu_size);

  // 复制 Y plane
  memcpy(img.plane[0].data(), nv12.plane[0].data(), y_size);

  // 交换 UV -> VU
  const uint8_t* src_uv = nv12.plane[1].data();
  uint8_t*       dst_vu = img.plane[1].data();

  for (size_t i = 0; i < vu_size; i += 2) {
    dst_vu[i] = src_uv[i + 1];  // V
    dst_vu[i + 1] = src_uv[i];  // U
  }

  return img;
}

// 保存图像到文件（用于验证）
void SaveImage(const ImageData& img, const std::string& filename) {
  cv::Mat mat;
  int     height = img.height & (~1);
  if (img.fmt == FORMAT_RGB24) {
    mat = cv::Mat(height, img.width, CV_8UC3, const_cast<uint8_t*>(img.plane[0].data()));
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
  } else if (img.fmt == FORMAT_BGR24) {
    mat = cv::Mat(height, img.width, CV_8UC3, const_cast<uint8_t*>(img.plane[0].data()));
  } else {
    std::cerr << "Cannot save format " << GetFormatName(img.fmt) << " directly" << std::endl;
    return;
  }
  cv::imwrite(filename, mat);
  std::cout << "Saved to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
  std::string image_path = "image.png";
  cv::Mat     src_mat = cv::imread(image_path, cv::IMREAD_COLOR);
  if (src_mat.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return -1;
  }

  std::cout << "========================================" << std::endl;
  std::cout << "Loaded image: " << image_path << std::endl;
  std::cout << "Size: " << src_mat.cols << "x" << src_mat.rows << std::endl;
  std::cout << "Channels: " << src_mat.channels() << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  // 1. OpenCV读取的BGR24格式
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 1: Convert OpenCV Mat to BGR24" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData bgr24 = MatToBGR24(src_mat);
  bgr24.PrintInfo("BGR24");
  if (ValidateImageData(bgr24)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 2. BGR24 -> RGB24 (使用 OpenCV，与 memop.cpp 一致)
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 2: BGR24 -> RGB24 (OpenCV)" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData rgb24 = BGR24ToRGB24(bgr24);
  rgb24.PrintInfo("RGB24");
  if (ValidateImageData(rgb24)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 5. BGR24 -> NV12
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 5: BGR24 -> NV12" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData nv12_from_bgr = BGR24ToNV12(bgr24);
  nv12_from_bgr.PrintInfo("NV12(from BGR)");
  if (ValidateImageData(nv12_from_bgr)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 6. BGR24 -> NV21
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 6: BGR24 -> NV21" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData nv21_from_bgr = BGR24ToNV21(bgr24);
  nv21_from_bgr.PrintInfo("NV21(from BGR)");
  if (ValidateImageData(nv21_from_bgr)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 7. NV12 -> RGB24 (与 memop.cpp 方式一致)
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 7: NV12 -> RGB24" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData rgb24_from_nv12 = NV12ToRGB24(nv12_from_bgr);
  rgb24_from_nv12.PrintInfo("RGB24(from NV12)");
  if (ValidateImageData(rgb24_from_nv12)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 8. NV21 -> RGB24 (与 memop.cpp 方式一致)
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 8: NV21 -> RGB24" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData rgb24_from_nv21 = NV21ToRGB24(nv21_from_bgr);
  rgb24_from_nv21.PrintInfo("RGB24(from NV21)");
  if (ValidateImageData(rgb24_from_nv21)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 9. NV12 -> BGR24 (与 memop.cpp 方式一致：先转RGB24再用OpenCV)
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 9: NV12 -> BGR24" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData bgr24_from_nv12 = NV12ToBGR24(nv12_from_bgr);
  bgr24_from_nv12.PrintInfo("BGR24(from NV12)");
  if (ValidateImageData(bgr24_from_nv12)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 10. NV21 -> BGR24 (与 memop.cpp 方式一致)
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 10: NV21 -> BGR24" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData bgr24_from_nv21 = NV21ToBGR24(nv21_from_bgr);
  bgr24_from_nv21.PrintInfo("BGR24(from NV21)");
  if (ValidateImageData(bgr24_from_nv21)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 13. 循环验证：BGR24 -> NV12 -> BGR24
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Step 13: Round-trip test BGR24 -> NV12 -> BGR24" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  ImageData bgr24_roundtrip = NV12ToBGR24(nv12_from_bgr);
  bgr24_roundtrip.PrintInfo("BGR24(round-trip)");
  if (ValidateImageData(bgr24_roundtrip)) {
    std::cout << "Validation: PASSED" << std::endl;
  }
  std::cout << std::endl;

  // 保存转换后的图像用于验证
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Saving converted images for verification" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  SaveImage(rgb24, "save/output_rgb24.jpg");
  SaveImage(bgr24, "save/output_bgr24.jpg");
  SaveImage(rgb24_from_nv12, "save/output_rgb24_from_nv12.jpg");
  SaveImage(bgr24_from_nv12, "save/output_bgr24_from_nv12.jpg");
  SaveImage(rgb24_from_nv21, "save/output_rgb24_from_nv21.jpg");
  SaveImage(bgr24_from_nv21, "save/output_bgr24_from_nv21.jpg");
  SaveImage(bgr24_roundtrip, "save/output_bgr24_roundtrip.jpg");

  std::cout << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "All conversions completed!" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
