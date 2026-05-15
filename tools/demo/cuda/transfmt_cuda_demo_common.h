#pragma once

#include <cuda_runtime.h>
#include <npp.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "libyuv/convert.h"
#include "libyuv/convert_from_argb.h"

#define DEFAULT_IMAGE_PATH "image.png"

#define CHECK_CUDA_RUNTIME(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

inline bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
  if (code != cudaSuccess) {
    const char* err_name = cudaGetErrorName(code);
    const char* err_message = cudaGetErrorString(code);
    printf("check_cuda_runtime error %s:%d  %s failed. \n  code = %s, message = %s\n",
           file, line, op, err_name, err_message);
    return false;
  }
  return true;
}

#define CHECK_CUDA_KERNEL(...)                                   \
  __VA_ARGS__;                                                   \
  do {                                                           \
    cudaError_t cudaStatus = cudaPeekAtLastError();              \
    if (cudaStatus != cudaSuccess) {                             \
      printf("launch failed: %s", cudaGetErrorString(cudaStatus)); \
    }                                                            \
  } while (0);

#define CHECK_NPP(op) __check_npp((op), #op, __FILE__, __LINE__)

inline std::string nppGetStatusString(NppStatus code) {
  return "NPP error code: " + std::to_string(code);
}

inline bool __check_npp(NppStatus code, const char* op, const char* file, int line) {
  if (code != NPP_SUCCESS) {
    printf("check_npp error %s:%d  %s failed. \n  code = %d, message = %s\n",
           file, line, op, code, nppGetStatusString(code).c_str());
    return false;
  }
  return true;
}

enum class DataFormat {
  INVALID = -1,
  PIXEL_FORMAT_YUV420_NV21 = 0,
  PIXEL_FORMAT_YUV420_NV12,
  PIXEL_FORMAT_BGR24,
  PIXEL_FORMAT_RGB24,
  PIXEL_FORMAT_ARGB32,
  PIXEL_FORMAT_ABGR32,
  PIXEL_FORMAT_RGBA32,
  PIXEL_FORMAT_BGRA32
};

struct TestFrame {
  int                  width;
  int                  height;
  DataFormat           fmt;
  std::vector<uint8_t> y_plane;
  std::vector<uint8_t> uv_plane;
  std::vector<uint8_t> rgb_plane;
  std::vector<uint8_t> bgr_plane;

  void* d_y_plane = nullptr;
  void* d_uv_plane = nullptr;
  void* d_rgb_plane = nullptr;
  void* d_bgr_plane = nullptr;

  int src_pitch = 0;
  int dst_pitch = 0;

  NppiSize oSize;

  ~TestFrame() {
    if (d_y_plane) cudaFree(d_y_plane);
    if (d_uv_plane) cudaFree(d_uv_plane);
    if (d_rgb_plane) cudaFree(d_rgb_plane);
    if (d_bgr_plane) cudaFree(d_bgr_plane);
  }
};

inline bool AllocateGpuMemory(TestFrame& frame) {
  size_t y_size = frame.width * frame.height;
  size_t uv_size = frame.width * frame.height / 2;

  if (frame.d_y_plane) cudaFree(frame.d_y_plane);
  if (frame.d_uv_plane) cudaFree(frame.d_uv_plane);
  if (frame.d_rgb_plane) cudaFree(frame.d_rgb_plane);
  if (frame.d_bgr_plane) cudaFree(frame.d_bgr_plane);

  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_y_plane, y_size));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_uv_plane, uv_size));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_rgb_plane, y_size * 3));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_bgr_plane, y_size * 3));

  return true;
}

/**
 * @brief Y 平面和 UV 平面紧密排列
 */
inline bool LoadToNV12(const std::string& image_path, TestFrame& frame) {
  cv::Mat src_mat = cv::imread(image_path, cv::IMREAD_COLOR);
  if (src_mat.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return false;
  }

  frame.width = src_mat.cols;
  frame.height = src_mat.rows;

  if (frame.height % 2 != 0 || frame.width % 2 != 0) {
    frame.height = (frame.height / 2) * 2;
    frame.width = (frame.width / 2) * 2;
    src_mat = src_mat(cv::Rect(0, 0, frame.width, frame.height));
  }

  frame.oSize.width = frame.width;
  frame.oSize.height = frame.height;

  std::cout << "Image Loaded: " << frame.width << "x" << frame.height << std::endl;

  std::vector<uint8_t> bgr_buffer(src_mat.cols * src_mat.rows * 3);
  memcpy(bgr_buffer.data(), src_mat.data, bgr_buffer.size());

  frame.y_plane.resize(frame.width * frame.height);
  frame.uv_plane.resize(frame.width * frame.height / 2);

  std::vector<uint8_t> argb_buffer(frame.width * frame.height * 4);
  int                  argb_stride = frame.width * 4;
  libyuv::RGB24ToARGB(bgr_buffer.data(), frame.width * 3, argb_buffer.data(), argb_stride, frame.width, frame.height);
  libyuv::ARGBToNV12(argb_buffer.data(), argb_stride, frame.y_plane.data(), frame.width,
                      frame.uv_plane.data(), frame.width, frame.width, frame.height);

  return true;
}

inline bool TestWithLibyuvCPU(TestFrame& frame, std::string output_file_rgb, std::string output_file_bgr) {
  std::cout << "\n===  with libyuv (CPU) for comparison ===" << std::endl;

  std::vector<uint8_t> cpu_rgb(frame.width * frame.height * 3);
  std::vector<uint8_t> cpu_bgr(frame.width * frame.height * 3);

  int ret = libyuv::NV12ToRAW(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width,
                              cpu_rgb.data(), frame.width * 3, frame.width, frame.height);

  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRAW failed with error: " << ret << std::endl;
    return false;
  }

  // int NV12ToRGB24(const uint8_t* src_y,
  //                 int src_stride_y,
  //                 const uint8_t* src_uv,
  //                 int src_stride_uv,
  //                 uint8_t* dst_rgb24,
  //                 int dst_stride_rgb24,
  //                 int width,
  //                 int height)
  // 对于紧密排列 src_stride_y = frame.width, src_stride_uv = frame.width
  ret = libyuv::NV12ToRGB24(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width,
                            cpu_bgr.data(), frame.width * 3, frame.width, frame.height);

  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRGB24 failed with error: " << ret << std::endl;
    return false;
  }

  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3, cpu_rgb.data());
  cv::Mat bgr_mat;
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
  cv::imwrite(output_file_rgb, bgr_mat);
  std::cout << "NV12 -> RGB24 (libyuv) result saved to: " << output_file_rgb << std::endl;

  cv::Mat bgr_mat2(frame.height, frame.width, CV_8UC3, cpu_bgr.data());
  cv::imwrite(output_file_bgr, bgr_mat2);
  std::cout << "NV12 -> BGR24 (libyuv) result saved to: " << output_file_bgr << std::endl;

  if (!frame.rgb_plane.empty()) {
    size_t diff_rgb = 0;
    for (size_t i = 0; i < frame.rgb_plane.size(); ++i) {
      if (std::abs(static_cast<int>(frame.rgb_plane[i]) - static_cast<int>(cpu_rgb[i])) > 1) {
        diff_rgb++;
      }
    }

    size_t total_pixels = frame.width * frame.height * 3;
    double diff_ratio_rgb = 100.0 * diff_rgb / total_pixels;

    std::cout << "NV12 -> RGB24 CUDA vs libyuv: " << diff_rgb << " / " << total_pixels << " pixels different ("
              << diff_ratio_rgb << "%)" << std::endl;
  }

  if (!frame.bgr_plane.empty()) {
    size_t diff_bgr = 0;
    for (size_t i = 0; i < frame.bgr_plane.size(); ++i) {
      if (std::abs(static_cast<int>(frame.bgr_plane[i]) - static_cast<int>(cpu_bgr[i])) > 1) {
        diff_bgr++;
      }
    }

    size_t total_pixels = frame.width * frame.height * 3;
    double diff_ratio_bgr = 100.0 * diff_bgr / total_pixels;

    std::cout << "NV12 -> BGR24 CUDA vs libyuv: " << diff_bgr << " / " << total_pixels << " pixels different ("
              << diff_ratio_bgr << "%)" << std::endl;
  }

  return true;
}

inline bool CreateUniformTestImage(int width, int height, uint8_t r_val, uint8_t g_val, uint8_t b_val,
                                   TestFrame& frame, std::string output_file) {
  std::cout << "\n=== Creating uniform test image (R=" << (int)r_val
            << ", G=" << (int)g_val << ", B=" << (int)b_val << ") ===" << std::endl;

  frame.width = width;
  frame.height = height;
  frame.fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;

  frame.oSize.width = frame.width;
  frame.oSize.height = frame.height;

  if (frame.height % 2 != 0 || frame.width % 2 != 0) {
    frame.height = (frame.height / 2) * 2;
    frame.width = (frame.width / 2) * 2;
  }

  cv::Mat src_mat(frame.height, frame.width, CV_8UC3);
  for (int y = 0; y < frame.height; ++y) {
    for (int x = 0; x < frame.width; ++x) {
      src_mat.at<cv::Vec3b>(y, x) = cv::Vec3b(b_val, g_val, r_val);
    }
  }

  cv::imwrite(output_file, src_mat);
  std::cout << "Original BGR image saved to: " << output_file << std::endl;

  frame.y_plane.resize(frame.width * frame.height);
  frame.uv_plane.resize(frame.width * frame.height / 2);

  std::vector<uint8_t> bgr_buffer(frame.width * frame.height * 3);
  memcpy(bgr_buffer.data(), src_mat.data, bgr_buffer.size());

  std::vector<uint8_t> argb_buffer(frame.width * frame.height * 4);
  int                  argb_stride = frame.width * 4;
  libyuv::RGB24ToARGB(bgr_buffer.data(), frame.width * 3, argb_buffer.data(), argb_stride, frame.width, frame.height);
  libyuv::ARGBToNV12(argb_buffer.data(), argb_stride, frame.y_plane.data(), frame.width,
                      frame.uv_plane.data(), frame.width, frame.width, frame.height);

  std::cout << "Image converted to NV12 format" << std::endl;
  return true;
}

inline bool TestChannelConsistency(TestFrame& frame, uint8_t expected_r, uint8_t expected_g, uint8_t expected_b) {
  std::cout << "\n===  channel consistency (Expected: R=" << (int)expected_r
            << ", G=" << (int)expected_g << ", B=" << (int)expected_b << ") ===" << std::endl;

  if (frame.bgr_plane.empty()) {
    std::cerr << "BGR plane is empty, run TestNV12ToBGR24 first" << std::endl;
    return false;
  }

  int width = frame.width;
  int height = frame.height;

  size_t total_pixels = width * height;
  size_t b_errors = 0, g_errors = 0, r_errors = 0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      uint8_t b = frame.bgr_plane[idx + 0];
      uint8_t g = frame.bgr_plane[idx + 1];
      uint8_t r = frame.bgr_plane[idx + 2];

      if (std::abs(static_cast<int>(b) - static_cast<int>(expected_b)) > 1) b_errors++;
      if (std::abs(static_cast<int>(g) - static_cast<int>(expected_g)) > 1) g_errors++;
      if (std::abs(static_cast<int>(r) - static_cast<int>(expected_r)) > 1) r_errors++;
    }
  }

  std::cout << "Channel consistency check:" << std::endl;
  std::cout << "  B channel: " << b_errors << " / " << total_pixels << std::endl;
  std::cout << "  G channel: " << g_errors << " / " << total_pixels << std::endl;
  std::cout << "  R channel: " << r_errors << " / " << total_pixels << std::endl;

  std::cout << "\nBGR memory layout analysis:" << std::endl;
  std::cout << "  Memory[0] = B = " << (int)frame.bgr_plane[0] << " (expected: " << (int)expected_b << ")" << std::endl;
  std::cout << "  Memory[1] = G = " << (int)frame.bgr_plane[1] << " (expected: " << (int)expected_g << ")" << std::endl;
  std::cout << "  Memory[2] = R = " << (int)frame.bgr_plane[2] << " (expected: " << (int)expected_r << ")" << std::endl;

  bool b_match = (b_errors == 0);
  bool g_match = (g_errors == 0);
  bool r_match = (r_errors == 0);

  if (b_match && g_match && r_match) {
    std::cout << "\n[PASS] All channels match expected values!" << std::endl;
  } else {
    std::cout << "\n[FAIL] Channel mismatch detected!" << std::endl;
  }

  return (b_match && g_match && r_match);
}

inline bool TestChannelConsistencyLibyuvCPU(TestFrame& frame, uint8_t expected_r, uint8_t expected_g, uint8_t expected_b) {
  std::cout << "\n===  with libyuv (CPU) channel consistency for comparison ===" << std::endl;

  std::vector<uint8_t> cpu_rgb(frame.width * frame.height * 3);
  std::vector<uint8_t> cpu_bgr(frame.width * frame.height * 3);

  int ret = libyuv::NV12ToRAW(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width,
                              cpu_rgb.data(), frame.width * 3, frame.width, frame.height);

  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRAW failed with error: " << ret << std::endl;
    return false;
  }

  ret = libyuv::NV12ToRGB24(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width,
                            cpu_bgr.data(), frame.width * 3, frame.width, frame.height);

  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRGB24 failed with error: " << ret << std::endl;
    return false;
  }

  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3, cpu_rgb.data());
  cv::Mat bgr_mat;
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
  cv::Mat bgr_mat2(frame.height, frame.width, CV_8UC3, cpu_bgr.data());

  int width = frame.width;
  int height = frame.height;

  size_t total_pixels = width * height;
  size_t b_errors = 0, g_errors = 0, r_errors = 0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint8_t b = bgr_mat2.at<cv::Vec3b>(y, x)[0];
      uint8_t g = bgr_mat2.at<cv::Vec3b>(y, x)[1];
      uint8_t r = bgr_mat2.at<cv::Vec3b>(y, x)[2];
      if (std::abs(static_cast<int>(b) - static_cast<int>(expected_b)) > 1) b_errors++;
      if (std::abs(static_cast<int>(g) - static_cast<int>(expected_g)) > 1) g_errors++;
      if (std::abs(static_cast<int>(r) - static_cast<int>(expected_r)) > 1) r_errors++;
    }
  }

  std::cout << "Channel consistency check:" << std::endl;
  std::cout << "  B channel: " << b_errors << " / " << total_pixels << " pixels different ("
            << (100.0 * b_errors / total_pixels) << "%)" << std::endl;
  std::cout << "  G channel: " << g_errors << " / " << total_pixels << " pixels different ("
            << (100.0 * g_errors / total_pixels) << "%)" << std::endl;
  std::cout << "  R channel: " << r_errors << " / " << total_pixels << " pixels different ("
            << (100.0 * r_errors / total_pixels) << "%)" << std::endl;

  std::cout << "\nBGR memory layout analysis:" << std::endl;
  std::cout << "  Memory[0] = B = " << (int)bgr_mat2.at<cv::Vec3b>(0, 0)[0] << " (expected: " << (int)expected_b << ")" << std::endl;
  std::cout << "  Memory[1] = G = " << (int)bgr_mat2.at<cv::Vec3b>(0, 0)[1] << " (expected: " << (int)expected_g << ")" << std::endl;
  std::cout << "  Memory[2] = R = " << (int)bgr_mat2.at<cv::Vec3b>(0, 0)[2] << " (expected: " << (int)expected_r << ")" << std::endl;

  bool b_match = (b_errors == 0);
  bool g_match = (g_errors == 0);
  bool r_match = (r_errors == 0);

  if (b_match && g_match && r_match) {
    std::cout << "\n[PASS] All channels match expected values!" << std::endl;
  } else {
    std::cout << "\n[FAIL] Channel mismatch detected!" << std::endl;
  }
  return (b_match && g_match && r_match);
}