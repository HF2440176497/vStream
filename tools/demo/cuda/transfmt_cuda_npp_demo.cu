/**
 * @file transfmt_cuda_npp_demo.cu
 * @brief CUDA NPP 图像格式转换 Demo
 *
 * 本 Demo 演示如何使用 NVIDIA NPP (Performance Primitives) 库
 * 进行 GPU 显存上的图像格式转换
 * 支持 NV12/YUV420 -> RGB24/BGR24 的转换
 */

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

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
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
      INFO("launch failed: %s", cudaGetErrorString(cudaStatus)); \
    }                                                            \
  } while (0);


#define CHECK_NPP(op) __check_npp((op), #op, __FILE__, __LINE__)

std::string nppGetStatusString(NppStatus code) {
    return "NPP error code: " + std::to_string(code);
}

bool __check_npp(NppStatus code, const char* op, const char* file, int line) {
  if (code != NPP_SUCCESS) {
    printf("check_npp error %s:%d  %s failed. \n  code = %d, message = %s\n", 
		file, line, op, code, nppGetStatusString(code).c_str());
    return false;
  }
  return true;
}

static const int PITCH_ALIGN = 4;

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

bool AllocateGpuMemory(TestFrame& frame) {
  size_t y_size = frame.width * frame.height;
  size_t uv_size = frame.width * frame.height / 2;

  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_y_plane, y_size));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_uv_plane, uv_size));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_rgb_plane, y_size * 3));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_bgr_plane, y_size * 3));

  frame.oSize.width = frame.width;
  frame.oSize.height = frame.height;

  return true;
}

bool LoadImageAndConvertToNV12(const std::string& image_path, TestFrame& frame) {
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

  std::cout << "Image loaded: " << frame.width << "x" << frame.height << std::endl;

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

/**
 * 将 NV12 格式的 frame 数据从 CPU 拷贝到 GPU
 * 注意为了保证 NPP 的使用，我们需要保证 stride 的字节对齐
 */
bool CopyToGpu(TestFrame& frame) {
  int src_pitch = ((frame.width + PITCH_ALIGN - 1) / PITCH_ALIGN) * PITCH_ALIGN;
  int dst_pitch = ((frame.width * 3 + PITCH_ALIGN - 1) / PITCH_ALIGN) * PITCH_ALIGN;

  frame.src_pitch = src_pitch;
  frame.dst_pitch = dst_pitch;

  cudaMalloc(&frame.d_y_plane,  src_pitch * frame.height);
  cudaMalloc(&frame.d_uv_plane, src_pitch * frame.height / 2);
  cudaMalloc(&frame.d_rgb_plane, dst_pitch * frame.height);
  cudaMalloc(&frame.d_bgr_plane, dst_pitch * frame.height * 3);

  CHECK_CUDA_RUNTIME(cudaMemcpy2D(frame.d_y_plane, src_pitch,
              frame.y_plane.data(), frame.width,
              frame.width, frame.height,
              cudaMemcpyHostToDevice));

  int uv_width  = frame.width;
  int uv_height = frame.height / 2;
  CHECK_CUDA_RUNTIME(cudaMemcpy2D(frame.d_uv_plane, src_pitch,
              frame.uv_plane.data(), uv_width,
              uv_width, uv_height,
              cudaMemcpyHostToDevice));
  return true;
}

bool TestNV12ToRGB24_NPP(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  NV12 -> RGB24 (NPP) ===" << std::endl;

  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);

  const Npp8u* aSrc[2] = {
    static_cast<const Npp8u*>(frame.d_y_plane),
    static_cast<const Npp8u*>(frame.d_uv_plane),
  };
  int aSrcStep = frame.src_pitch;

  Npp8u* pDst = static_cast<Npp8u*>(frame.d_rgb_plane);
  int nDstStep = frame.dst_pitch;  // 对齐后的步长

  NppiSize oSizeROI;
  oSizeROI.width  = frame.width;
  oSizeROI.height = frame.height;

  status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
    aSrc, aSrcStep,
    pDst, nDstStep,
    oSizeROI,
    npp_stream_ctx
  );

  // status = nppiNV12ToRGB_8u_P2C3R_Ctx(
  //   aSrc, aSrcStep,
  //   pDst, nDstStep,
  //   oSizeROI,
  //   npp_stream_ctx
  // );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.rgb_plane.resize(frame.width * frame.height * 3);  // 主机端准备紧密布局缓冲区（无填充）
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.rgb_plane.data(), frame.width * 3,
                  frame.d_rgb_plane, frame.dst_pitch,
                  frame.width * 3, frame.height,
                  cudaMemcpyDeviceToHost));

  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3, frame.rgb_plane.data());

  cv::Mat bgr_mat;
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
  frame.bgr_plane.resize(frame.width * frame.height * 3);
  memcpy(frame.bgr_plane.data(), bgr_mat.data, frame.bgr_plane.size());

  cv::imwrite(output_file, bgr_mat);
  std::cout << "NV12 -> RGB24 (NPP) result saved to: " << output_file << std::endl;

  return true;
}

bool TestNV12ToBGR24_NPP(TestFrame& frame, std::string output_file) {

  std::cout << "\n===  NV12 -> BGR24 (NPP) ===" << std::endl;

  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);

  const Npp8u* aSrc[2] = {
    static_cast<const Npp8u*>(frame.d_y_plane),
    static_cast<const Npp8u*>(frame.d_uv_plane)
  };
  int aSrcStep = frame.src_pitch;

  Npp8u* pDst = static_cast<Npp8u*>(frame.d_bgr_plane);
  int nDstStep = frame.dst_pitch;

  NppiSize oSizeROI;
  oSizeROI.width  = frame.width;
  oSizeROI.height = frame.height;

  status = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(
    aSrc, aSrcStep,
    pDst, nDstStep,
    oSizeROI,
    npp_stream_ctx
  );

  // status = nppiNV12ToBGR_8u_P2C3R_Ctx(
  //   aSrc, aSrcStep, 
  //   pDst, nDstStep,
  //   oSizeROI, 
  //   npp_stream_ctx
  // );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.bgr_plane.data(), frame.width * 3,
                   frame.d_bgr_plane, frame.dst_pitch,
                   frame.width * 3, frame.height,
                   cudaMemcpyDeviceToHost));

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, frame.bgr_plane.data());
  cv::imwrite(output_file, bgr_mat);
  std::cout << "NV12 -> BGR24 (NPP) result saved to: " << output_file << std::endl;

  return true;
}

static const Npp32f MATRIX_RGB_TO_YUV709[3][4] = {
  { 0.183f,  0.614f,  0.062f,  16.0f },
  {-0.101f, -0.339f,  0.439f, 128.0f },
  { 0.439f, -0.399f, -0.040f, 128.0f }
};

static const Npp32f MATRIX_BGR_TO_YUV709[3][4] = {
  { 0.062f,  0.614f,  0.183f,  16.0f },
  { 0.439f, -0.339f, -0.101f, 128.0f },
  {-0.040f, -0.399f,  0.439f, 128.0f }
};

/**
 * 需要前面用过 NV12ToRGB24_NPP 保存有 rgb_plane
 * RGB24 -> NV12 -> BGR24 (NPP)
 */
bool TestRGB24ToNV12ToBGR_NPP(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  RGB24 -> NV12 -> BGR24 (NPP) ===" << std::endl;

  if (frame.rgb_plane.empty()) {
    std::cerr << "RGB plane is empty, run NV12ToRGB24_NPP first" << std::endl;
    return false;
  }
  // 上传紧密 RGB 到设备对齐内存
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.d_rgb_plane, frame.dst_pitch,
                   frame.rgb_plane.data(), frame.width * 3,
                   frame.width * 3, frame.height,
                   cudaMemcpyHostToDevice));

  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);

  NppiSize oSizeROI;
  oSizeROI.width  = frame.width;
  oSizeROI.height = frame.height;

  const Npp8u* pSrc = static_cast<const Npp8u*>(frame.d_rgb_plane);
  Npp8u* pDst[2] = { (Npp8u*)frame.d_y_plane, (Npp8u*)frame.d_uv_plane };
  int DstStep[2] = { frame.src_pitch, frame.src_pitch };   // 使用对齐步长

  status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
    pSrc, frame.dst_pitch,
    pDst, DstStep,
    oSizeROI,
    MATRIX_RGB_TO_YUV709,
    npp_stream_ctx);
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.y_plane.resize(frame.width * frame.height);
  frame.uv_plane.resize(frame.width * frame.height / 2);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.y_plane.data(), frame.width,
                   frame.d_y_plane, frame.src_pitch,
                   frame.width, frame.height,
                   cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.uv_plane.data(), frame.width,
                   frame.d_uv_plane, frame.src_pitch,
                   frame.width, frame.height / 2,
                   cudaMemcpyDeviceToHost));

  std::vector<uint8_t> cpu_bgr(frame.width * frame.height * 3);
  int ret = libyuv::NV12ToRGB24(frame.y_plane.data(), frame.width,
                                frame.uv_plane.data(), frame.width,
                                cpu_bgr.data(), frame.width * 3,
                                frame.width, frame.height);
  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRGB24 failed with error: " << ret << std::endl;
    return false;
  }

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, cpu_bgr.data());
  cv::imwrite(output_file, bgr_mat);
  std::cout << "RGB24 -> NV12 -> BGR24 result saved to: " << output_file << std::endl;
  return true;
}


bool TestBGR24ToNV12ToBGR_NPP(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  BGR24 -> NV12 -> BGR24 (NPP) ===" << std::endl;

  if (frame.bgr_plane.empty()) {
    std::cerr << "BGR plane is empty, run NV12ToBGR24_NPP first" << std::endl;
    return false;
  }
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.d_bgr_plane, frame.dst_pitch,
                   frame.bgr_plane.data(), frame.width * 3,
                   frame.width * 3, frame.height,
                   cudaMemcpyHostToDevice));

  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);

  NppiSize oSizeROI;
  oSizeROI.width  = frame.width;
  oSizeROI.height = frame.height;

  const Npp8u* pSrc = static_cast<const Npp8u*>(frame.d_bgr_plane);
  Npp8u* pDst[2] = { (Npp8u*)frame.d_y_plane, (Npp8u*)frame.d_uv_plane };
  int DstStep[2] = { frame.src_pitch, frame.src_pitch };

  status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
    pSrc, frame.dst_pitch,
    pDst, DstStep,
    oSizeROI,
    MATRIX_BGR_TO_YUV709,
    npp_stream_ctx);
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.y_plane.resize(frame.width * frame.height);
  frame.uv_plane.resize(frame.width * frame.height / 2);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.y_plane.data(), frame.width,
                   frame.d_y_plane, frame.src_pitch,
                   frame.width, frame.height,
                   cudaMemcpyDeviceToHost));
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.uv_plane.data(), frame.width,
                   frame.d_uv_plane, frame.src_pitch,
                   frame.width, frame.height / 2,
                   cudaMemcpyDeviceToHost));

  std::vector<uint8_t> cpu_bgr(frame.width * frame.height * 3);
  int ret = libyuv::NV12ToRGB24(frame.y_plane.data(), frame.width,
                                frame.uv_plane.data(), frame.width,
                                cpu_bgr.data(), frame.width * 3,
                                frame.width, frame.height);
  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRGB24 failed with error: " << ret << std::endl;
    return false;
  }

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, cpu_bgr.data());
  cv::imwrite(output_file, bgr_mat);
  std::cout << "BGR24 -> NV12 -> BGR24 result saved to: " << output_file << std::endl;
  return true;
}

bool TestRGB24ToBGR24_NPP(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  RGB24 -> BGR24 (NPP) ===" << std::endl;
  if (frame.rgb_plane.empty()) {
    std::cerr << "RGB plane is empty, run NV12ToRGB24_NPP first" << std::endl;
    return false;
  }
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.d_rgb_plane, frame.dst_pitch,
                   frame.rgb_plane.data(), frame.width * 3,
                   frame.width * 3, frame.height,
                   cudaMemcpyHostToDevice));

  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);

  NppiSize oSizeROI;
  oSizeROI.width  = frame.width;
  oSizeROI.height = frame.height;

  int aDstOrder[3] = { 2, 1, 0 };
  status = nppiSwapChannels_8u_C3R_Ctx(
      static_cast<const Npp8u*>(frame.d_rgb_plane), frame.dst_pitch,
      static_cast<Npp8u*>(frame.d_bgr_plane), frame.dst_pitch,
      oSizeROI, aDstOrder, npp_stream_ctx);
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy2D(frame.bgr_plane.data(), frame.width * 3,
                   frame.d_bgr_plane, frame.dst_pitch,
                   frame.width * 3, frame.height,
                   cudaMemcpyDeviceToHost));

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, frame.bgr_plane.data());
  cv::imwrite(output_file, bgr_mat);
  std::cout << "RGB24 -> BGR24 (NPP) result saved to: " << output_file << std::endl;
  return true;
}

/**
 * 使用 libyuv 将 frame 的 y_plane 和 uv_plane 转换为 BGR24 / RGB24 并保存到 output_file
 */
bool TestWithLibyuvCPU(TestFrame& frame, std::string output_file_rgb, std::string output_file_bgr) {
  std::cout << "\n===  with libyuv (CPU) for comparison ===" << std::endl;

  std::vector<uint8_t> cpu_rgb(frame.width * frame.height * 3);
  std::vector<uint8_t> cpu_bgr(frame.width * frame.height * 3);

  // actual: To RGB24
  int ret = libyuv::NV12ToRAW(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width, 
                            cpu_rgb.data(), frame.width * 3, frame.width, frame.height);

  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRAW failed with error: " << ret << std::endl;
    return false;
  }

  // actual: To BGR24
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

  // note: bgr_plane 是前面 CUDA 转换得到的数据，与 libyuv 转换得到的进行对比
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

bool CreateUniformTestImage(int width, int height, uint8_t r_val, uint8_t g_val, uint8_t b_val, 
                            TestFrame& frame, std::string output_file) {
  std::cout << "\n=== Creating uniform test image (R=" << (int)r_val 
            << ", G=" << (int)g_val << ", B=" << (int)b_val << ") ===" << std::endl;

  frame.width = width;
  frame.height = height;
  frame.fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;

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

/**
 * 检查经过核函数转换之后，每个通道的内存排列
 */
bool TestChannelConsistency(TestFrame& frame, uint8_t expected_r, uint8_t expected_g, uint8_t expected_b) {
  std::cout << "\n===  channel consistency (Expected: R=" << (int)expected_r 
            << ", G=" << (int)expected_g << ", B=" << (int)expected_b << ") ===" << std::endl;

  if (frame.bgr_plane.empty()) {
    std::cerr << "BGR plane is empty, run TestNV12ToBGR24_CUDA first" << std::endl;
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
  std::cout << "  B channel: " << b_errors << " / " << total_pixels << " pixels different ("
            << (100.0 * b_errors / total_pixels) << "%)" << std::endl;
  std::cout << "  G channel: " << g_errors << " / " << total_pixels << " pixels different ("
            << (100.0 * g_errors / total_pixels) << "%)" << std::endl;
  std::cout << "  R channel: " << r_errors << " / " << total_pixels << " pixels different ("
            << (100.0 * r_errors / total_pixels) << "%)" << std::endl;

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


/**
 * 将 y_plane uv_plane 转换为 BGR24 图像, 对比 expected value
 */
bool TestChannelConsistencyLibyuvCPU(TestFrame& frame, uint8_t expected_r, uint8_t expected_g, uint8_t expected_b) {
  std::cout << "\n===  with libyuv (CPU) channel consistency for comparison ===" << std::endl;

  std::vector<uint8_t> cpu_rgb(frame.width * frame.height * 3);
  std::vector<uint8_t> cpu_bgr(frame.width * frame.height * 3);

  // actual: To RGB24
  int ret = libyuv::NV12ToRAW(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width, 
                            cpu_rgb.data(), frame.width * 3, frame.width, frame.height);

  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRAW failed with error: " << ret << std::endl;
    return false;
  }

  // actual: To BGR24
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

  // 得到 libyuv 的 BGR 图像之后，对比期望的 BGR 值
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

/**
 * 通过 libyuv 将 NV12 转换为 BGR24，并逐通道验证是否一致
 */
bool TestOpenCVConversionConsistency(TestFrame& frame, uint8_t expected_r, uint8_t expected_g, uint8_t expected_b,
                                    std::string output_file) {
  std::cout << "\n===  libyuv NV12 -> BGR24 conversion ===" << std::endl;

  std::vector<uint8_t> opencv_bgr(frame.width * frame.height * 3);
  
  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3);
  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3);

  libyuv::NV12ToRGB24(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width,
                      rgb_mat.data, frame.width * 3, frame.width, frame.height);
                      
  // cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
  // memcpy(opencv_bgr.data(), bgr_mat.data, opencv_bgr.size());
  // cv::imwrite("nv12_to_bgr24_opencv.jpg", bgr_mat);

  // note: 根据之前 libyuv_demo 的分析结果，不再需要 cvtColor
  memcpy(opencv_bgr.data(), rgb_mat.data, opencv_bgr.size());
  cv::imwrite(output_file, rgb_mat);
  
  std::cout << "OpenCV conversion result saved to: " << output_file << std::endl;

  int width = frame.width;
  int height = frame.height;
  // size_t total_pixels = width * height;
  size_t b_errors = 0, g_errors = 0, r_errors = 0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      uint8_t b = opencv_bgr[idx + 0];
      uint8_t g = opencv_bgr[idx + 1];
      uint8_t r = opencv_bgr[idx + 2];

      if (std::abs(static_cast<int>(b) - static_cast<int>(expected_b)) > 1) b_errors++;
      if (std::abs(static_cast<int>(g) - static_cast<int>(expected_g)) > 1) g_errors++;
      if (std::abs(static_cast<int>(r) - static_cast<int>(expected_r)) > 1) r_errors++;
    }
  }

  // std::cout << "OpenCV channel consistency check:" << std::endl;
  // std::cout << "  B channel: " << b_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * b_errors / total_pixels) << "%)" << std::endl;
  // std::cout << "  G channel: " << g_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * g_errors / total_pixels) << "%)" << std::endl;
  // std::cout << "  R channel: " << r_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * r_errors / total_pixels) << "%)" << std::endl;

  std::cout << "\nOpenCV BGR memory layout:" << std::endl;
  std::cout << "  Memory[0] = B = " << (int)opencv_bgr[0] << " (expected: " << (int)expected_b << ")" << std::endl;
  std::cout << "  Memory[1] = G = " << (int)opencv_bgr[1] << " (expected: " << (int)expected_g << ")" << std::endl;
  std::cout << "  Memory[2] = R = " << (int)opencv_bgr[2] << " (expected: " << (int)expected_r << ")" << std::endl;

  return (b_errors == 0 && g_errors == 0 && r_errors == 0);
}


int main(int argc, char** argv) {
  std::string image_path = (argc > 1) ? argv[1] : DEFAULT_IMAGE_PATH;
  std::cout << "  CUDA NPP Image Format Conversion     " << std::endl;

  int         deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cerr << "No CUDA devices found" << std::endl;
    return -1;
  }

  std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using device 0: " << prop.name << std::endl;

  TestFrame frame;
  frame.fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;

  std::cout << "\nLoading image: " << image_path << std::endl;
  if (!LoadImageAndConvertToNV12(image_path, frame)) {
    std::cerr << "Failed to load image and convert to NV12" << std::endl;
    return -1;
  }

  std::cout << "Allocating GPU memory..." << std::endl;
  if (!AllocateGpuMemory(frame)) {
    std::cerr << "Failed to allocate GPU memory" << std::endl;
    return -1;
  }

  std::cout << "Copying data to GPU..." << std::endl;
  CopyToGpu(frame);

  TestNV12ToRGB24_NPP(frame, "save/nv12_to_rgb24_npp.jpg");
  TestNV12ToBGR24_NPP(frame, "save/nv12_to_bgr24_npp.jpg");
  TestRGB24ToBGR24_NPP(frame, "save/rgb24_to_bgr24_npp.jpg");
  TestRGB24ToNV12ToBGR_NPP(frame, "save/rgb24_to_nv12_bgr24_npp.jpg");
  TestBGR24ToNV12ToBGR_NPP(frame, "save/bgr24_to_nv12_bgr24_npp.jpg");
  TestWithLibyuvCPU(frame, "save/nv12_to_rgb24_libyuv.jpg", "save/nv12_to_bgr24_libyuv.jpg");

  std::cout << "\n\n";

  TestFrame uniform_frame;
  const int test_width = 640;
  const int test_height = 480;
  const uint8_t test_r = 10;
  const uint8_t test_g = 128;
  const uint8_t test_b = 242;

  if (!CreateUniformTestImage(test_width, test_height, test_r, test_g, test_b, 
                              uniform_frame, "save/uniform_test.jpg")) {
    std::cerr << "Failed to create uniform test image" << std::endl;
    return -1;
  } 

  if (!AllocateGpuMemory(uniform_frame)) {
    std::cerr << "Failed to allocate GPU memory for uniform frame" << std::endl;
    return -1;
  }
  CopyToGpu(uniform_frame);
  
  std::cout << "\n--- NPP Direct BGR ---" << std::endl;
  TestNV12ToBGR24_NPP(uniform_frame, "save/nv12_to_bgr24_npp_uniform.jpg");
  TestChannelConsistency(uniform_frame, test_r, test_g, test_b);

  std::cout << "\n--- NPP with Swap ---" << std::endl;
  TestNV12ToRGB24_NPP(uniform_frame, "save/nv12_to_rgb24_npp_uniform.jpg");
  TestChannelConsistency(uniform_frame, test_r, test_g, test_b);

  TestChannelConsistencyLibyuvCPU(uniform_frame, test_r, test_g, test_b);
  std::cout << " NPP Demo completed " << std::endl;

  return 0;
}
