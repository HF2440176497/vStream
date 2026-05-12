/**
 * @file transfmt_cuda_npp_demo.cu
 * @brief CUDA NPP 图像格式转换 Demo
 *
 * 本 Demo 演示如何使用 NVIDIA NPP (Performance Primitives) 库
 * 进行 GPU 显存上的图像格式转换
 * 支持 NV12/YUV420 -> RGB24/BGR24 的转换
 */

#include "transfmt_cuda_demo_common.h"

static const int PITCH_ALIGN = 4;

/**
 * 将 NV12 格式的 frame 数据从 CPU 拷贝到 GPU
 * 注意为了保证 NPP 的使用，我们需要保证 stride 的字节对齐
 */
static bool CopyToGpu(TestFrame& frame) {
  int src_pitch = ((frame.width + PITCH_ALIGN - 1) / PITCH_ALIGN) * PITCH_ALIGN;
  int dst_pitch = ((frame.width * 3 + PITCH_ALIGN - 1) / PITCH_ALIGN) * PITCH_ALIGN;

  // device buf stride
  frame.src_pitch = src_pitch;  // Y/UV 平面的对齐步长
  frame.dst_pitch = dst_pitch;  // RGB/BGR 的对齐步长

  cudaMalloc(&frame.d_y_plane,  src_pitch * frame.height);
  cudaMalloc(&frame.d_uv_plane, src_pitch * frame.height / 2);
  cudaMalloc(&frame.d_rgb_plane, dst_pitch * frame.height);
  cudaMalloc(&frame.d_bgr_plane, dst_pitch * frame.height);

  CHECK_CUDA_RUNTIME(cudaMemcpy2D(frame.d_y_plane, frame.src_pitch,
              frame.y_plane.data(), frame.width,
              frame.width, frame.height,
              cudaMemcpyHostToDevice));

  int uv_width  = frame.width;
  int uv_height = frame.height / 2;
  CHECK_CUDA_RUNTIME(cudaMemcpy2D(frame.d_uv_plane, frame.src_pitch,
              frame.uv_plane.data(), frame.width,
              uv_width, uv_height,
              cudaMemcpyHostToDevice));
  return true;
}

static bool TestNV12ToRGB24_NPP(TestFrame& frame, std::string output_file) {
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
  int nDstStep = frame.dst_pitch;

  // For npp func, 使用对齐的 
  status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
    aSrc, aSrcStep,
    pDst, nDstStep,
    frame.oSize,
    npp_stream_ctx
  );

  // status = nppiNV12ToRGB_8u_P2C3R_Ctx(
  //   aSrc, aSrcStep,
  //   pDst, nDstStep,
  //   frame.oSize,
  //   npp_stream_ctx
  // );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.rgb_plane.resize(frame.width * frame.height * 3);

  // 拷贝回 host,需要使用紧密步长 frame.width * 3
  // frame.width * 3: column in bytes
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

static bool TestNV12ToBGR24_NPP(TestFrame& frame, std::string output_file) {

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

  status = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(
    aSrc, aSrcStep,
    pDst, nDstStep,
    frame.oSize,
    npp_stream_ctx
  );

  // status = nppiNV12ToBGR_8u_P2C3R_Ctx(
  //   aSrc, aSrcStep, 
  //   pDst, nDstStep,
  //   frame.oSize, 
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
static bool TestRGB24ToNV12ToBGR_NPP(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  RGB24 -> NV12 -> BGR24 (NPP) ===" << std::endl;

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

  const Npp8u* pSrc = static_cast<const Npp8u*>(frame.d_rgb_plane);
  Npp8u* pDst[2] = { (Npp8u*)frame.d_y_plane, (Npp8u*)frame.d_uv_plane };
  int DstStep[2] = { frame.src_pitch, frame.src_pitch };   // 使用对齐步长

  status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
    pSrc, frame.dst_pitch,
    pDst, DstStep,
    frame.oSize,
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

static bool TestBGR24ToNV12ToBGR_NPP(TestFrame& frame, std::string output_file) {
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

  const Npp8u* pSrc = static_cast<const Npp8u*>(frame.d_bgr_plane);
  Npp8u* pDst[2] = { (Npp8u*)frame.d_y_plane, (Npp8u*)frame.d_uv_plane };
  int DstStep[2] = { frame.src_pitch, frame.src_pitch };

  status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
    pSrc, frame.dst_pitch,
    pDst, DstStep,
    frame.oSize,
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

static bool TestRGB24ToBGR24_NPP(TestFrame& frame, std::string output_file) {
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

  int aDstOrder[3] = { 2, 1, 0 };
  status = nppiSwapChannels_8u_C3R_Ctx(
      static_cast<const Npp8u*>(frame.d_rgb_plane), frame.dst_pitch,
      static_cast<Npp8u*>(frame.d_bgr_plane), frame.dst_pitch,
      frame.oSize, aDstOrder, npp_stream_ctx);
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

static bool TestOpenCVConversionConsistency(TestFrame& frame, uint8_t expected_r, uint8_t expected_g, uint8_t expected_b,
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

  // size_t total_pixels = width * height;
  // size_t b_errors = 0, g_errors = 0, r_errors = 0;
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
  // 首先验证 LIBYUV 转换结果，是否存在色彩偏移的情况
  TestWithLibyuvCPU(frame, "save/nv12_to_rgb24_libyuv_pre.jpg", "save/nv12_to_bgr24_libyuv_pre.jpg");

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