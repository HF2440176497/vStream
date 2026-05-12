/**
 * @file transfmt_cuda_kernel_demo.cu
 * @brief CUDA Kernel 图像格式转换 Demo
 *
 * 本 Demo 演示如何使用手写 CUDA Kernel
 * 进行 GPU 显存上的图像格式转换
 * 支持 NV12/YUV420 -> RGB24/BGR24 的转换 (BT.601标准)
 * 目前采用此方案进行 convert，因为 NPP 库的转换为 BT.709 标准
 */

#include "transfmt_cuda_demo_common.h"

static bool LoadImageAndConvertToNV21(const std::string& image_path, TestFrame& frame) {
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
  libyuv::ARGBToNV21(argb_buffer.data(), argb_stride, frame.y_plane.data(), frame.width,
                      frame.uv_plane.data(), frame.width, frame.width, frame.height);

  return true;
}

/**
 * 对于 kernel 转换都统一采用紧密排列内存
 */
static bool CopyToGpu(TestFrame& frame) {
  size_t y_size = frame.width * frame.height;
  size_t uv_size = frame.width * frame.height / 2;

  CHECK_CUDA_RUNTIME(cudaMemcpy(frame.d_y_plane, frame.y_plane.data(), y_size, cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaMemcpy(frame.d_uv_plane, frame.uv_plane.data(), uv_size, cudaMemcpyHostToDevice));

  return true;
}

__global__ void nv21ToRGBKernel(const uint8_t* yPlane, const uint8_t* vuPlane,
                                uint8_t* bgrOutput, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint8_t Y = yPlane[idx];
    int uvX = x / 2;
    int uvY = y / 2;
    int uvIdx = uvY * (width / 2) + uvX;

    uint8_t V = vuPlane[uvIdx * 2];
    uint8_t U = vuPlane[uvIdx * 2 + 1];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int R = (298 * C + 459 * E + 128) >> 8;
    int G = (298 * C - 55 * D - 137 * E + 128) >> 8;
    int B = (298 * C + 541 * D + 128) >> 8;

    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));

    int outIdx = idx * 3;
    bgrOutput[outIdx] = R;
    bgrOutput[outIdx + 1] = G;
    bgrOutput[outIdx + 2] = B;
}

/**
 * 此 kernel 假设输入的 yPlane, vuPlane 均为紧密排列内存
 * uvIdx 为 UV pair index, *2 得到字节偏移
 */
__global__ void nv21ToBGRKernel(const uint8_t* yPlane, const uint8_t* vuPlane,
                                uint8_t* bgrOutput, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint8_t Y = yPlane[idx];

    int uvX = x / 2;
    int uvY = y / 2;
    int uvIdx = uvY * (width / 2) + uvX;

    uint8_t V = vuPlane[uvIdx * 2];
    uint8_t U = vuPlane[uvIdx * 2 + 1];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int R = (298 * C + 459 * E + 128) >> 8;
    int G = (298 * C - 55 * D - 137 * E + 128) >> 8;
    int B = (298 * C + 541 * D + 128) >> 8;

    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));

    int outIdx = idx * 3;
    bgrOutput[outIdx] = B;
    bgrOutput[outIdx + 1] = G;
    bgrOutput[outIdx + 2] = R;
}

__global__ void nv12ToRGB24Kernel(const uint8_t* yPlane, const uint8_t* uvPlane,
                                  uint8_t* rgbOutput, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint8_t Y = yPlane[idx];

    int uvX = x / 2;
    int uvY = y / 2;
    int uvIdx = uvY * (width / 2) + uvX;

    uint8_t U = uvPlane[uvIdx * 2];
    uint8_t V = uvPlane[uvIdx * 2 + 1];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int R = (298 * C + 459 * E + 128) >> 8;
    int G = (298 * C - 55 * D - 137 * E + 128) >> 8;
    int B = (298 * C + 541 * D + 128) >> 8;

    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));

    int outIdx = idx * 3;
    rgbOutput[outIdx + 0] = R;
    rgbOutput[outIdx + 1] = G;
    rgbOutput[outIdx + 2] = B;
}

__global__ void nv12ToBGR24Kernel(const uint8_t* yPlane, const uint8_t* uvPlane,
                                  uint8_t* bgrOutput, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint8_t Y = yPlane[idx];

    int uvX = x / 2;
    int uvY = y / 2;
    int uvIdx = uvY * (width / 2) + uvX;

    uint8_t U = uvPlane[uvIdx * 2];
    uint8_t V = uvPlane[uvIdx * 2 + 1];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;

    int R = (298 * C + 459 * E + 128) >> 8;
    int G = (298 * C - 55 * D - 137 * E + 128) >> 8;
    int B = (298 * C + 541 * D + 128) >> 8;

    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));

    int outIdx = idx * 3;
    bgrOutput[outIdx + 0] = B;
    bgrOutput[outIdx + 1] = G;
    bgrOutput[outIdx + 2] = R;
}

static bool TestNV12ToRGB24_CUDA(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  NV12 -> RGB24 (CUDA Kernel) ===" << std::endl;

  dim3 block(16, 16);
  dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

  nv12ToRGB24Kernel<<<grid, block>>>(
      static_cast<const uint8_t*>(frame.d_y_plane),
      static_cast<const uint8_t*>(frame.d_uv_plane),
      static_cast<uint8_t*>(frame.d_rgb_plane),
      frame.width, frame.height);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.rgb_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.rgb_plane.data(), frame.d_rgb_plane, frame.width * frame.height * 3, cudaMemcpyDeviceToHost));

  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3, frame.rgb_plane.data());
  cv::Mat bgr_mat;
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  memcpy(frame.bgr_plane.data(), bgr_mat.data, frame.width * frame.height * 3);

  cv::imwrite(output_file, bgr_mat);
  std::cout << "NV12 -> RGB24 (CUDA Kernel) result saved to: " << output_file << std::endl;

  return true;
}

static bool TestNV12ToBGR24_CUDA(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  NV12 -> BGR24 (CUDA Kernel) ===" << std::endl;

  dim3 block(16, 16);
  dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

  nv12ToBGR24Kernel<<<grid, block>>>(
      static_cast<const uint8_t*>(frame.d_y_plane),
      static_cast<const uint8_t*>(frame.d_uv_plane),
      static_cast<uint8_t*>(frame.d_bgr_plane),
      frame.width, frame.height);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.bgr_plane.data(), frame.d_bgr_plane, frame.width * frame.height * 3, cudaMemcpyDeviceToHost));

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, frame.bgr_plane.data());
  cv::imwrite(output_file, bgr_mat);
  std::cout << "NV12 -> BGR24 (CUDA Kernel) result saved to: " << output_file << std::endl;

  return true;
}

static bool TestNV21ToBGR24_CUDA(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  NV21 -> BGR24 (Kernel) ===" << std::endl;

  dim3 block(16, 16);
  dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

  nv21ToBGRKernel<<<grid, block>>>(static_cast<const uint8_t*>(frame.d_y_plane), static_cast<const uint8_t*>(frame.d_uv_plane),
                    static_cast<uint8_t*>(frame.d_bgr_plane), frame.width, frame.height);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.bgr_plane.data(), frame.d_bgr_plane, frame.width * frame.height * 3, cudaMemcpyDeviceToHost));

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, frame.bgr_plane.data());
  cv::imwrite(output_file, bgr_mat);
  std::cout << "NV21 -> BGR24 (Kernel) result saved to: " << output_file << std::endl;

  return true;
}

static bool TestNV21ToRGB24_CUDA(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  NV21 -> RGB24 (Kernel) ===" << std::endl;

  dim3 block(16, 16);
  dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

  nv21ToRGBKernel<<<grid, block>>>(static_cast<const uint8_t*>(frame.d_y_plane), static_cast<const uint8_t*>(frame.d_uv_plane),
                    static_cast<uint8_t*>(frame.d_rgb_plane), frame.width, frame.height);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.rgb_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.rgb_plane.data(), frame.d_rgb_plane, frame.width * frame.height * 3, cudaMemcpyDeviceToHost));

  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3, frame.rgb_plane.data());
  cv::Mat bgr_mat;
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  memcpy(frame.bgr_plane.data(), bgr_mat.data, frame.width * frame.height * 3);

  cv::imwrite(output_file, bgr_mat);
  std::cout << "NV21 -> RGB24 (Kernel) result saved to: " << output_file << std::endl;

  return true;
}

__global__ void RGB24ToBGR24Kernel(const uint8_t* __restrict__ rgb_in, uint8_t* __restrict__ bgr_out, int width,
                                   int height, int stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = y * stride + x * 3;
  bgr_out[idx + 0] = rgb_in[idx + 2];
  bgr_out[idx + 1] = rgb_in[idx + 1];
  bgr_out[idx + 2] = rgb_in[idx + 0];
}

static bool TestRGB24ToBGR24_CUDA(TestFrame& frame, std::string output_file) {
  std::cout << "\n===  RGB24 -> BGR24 (CUDA) ===" << std::endl;

  if (frame.rgb_plane.empty()) {
    std::cerr << "RGB plane is empty, run NV12ToRGB24_CUDA first" << std::endl;
    return false;
  }

  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.d_rgb_plane, frame.rgb_plane.data(), frame.width * frame.height * 3, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

  RGB24ToBGR24Kernel<<<grid, block>>>(static_cast<const uint8_t*>(frame.d_rgb_plane),
                                      static_cast<uint8_t*>(frame.d_bgr_plane), frame.width, frame.height,
                                      frame.width * 3);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.bgr_plane.data(), frame.d_bgr_plane, frame.width * frame.height * 3, cudaMemcpyDeviceToHost));

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, frame.bgr_plane.data());
  cv::imwrite(output_file, bgr_mat);
  std::cout << "RGB24 -> BGR24 (CUDA) result saved to: " << output_file << std::endl;

  return true;
}

int main(int argc, char** argv) {
  std::string image_path = (argc > 1) ? argv[1] : DEFAULT_IMAGE_PATH;
  std::cout << "  CUDA Kernel Image Format Conversion     " << std::endl;

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

  LoadImageAndConvertToNV12(image_path, frame);
  AllocateGpuMemory(frame);
  CopyToGpu(frame);

  TestNV12ToBGR24_CUDA(frame, "save/nv12_to_bgr24_cuda.jpg");
  TestNV12ToRGB24_CUDA(frame, "save/nv12_to_rgb24_cuda.jpg");
  TestRGB24ToBGR24_CUDA(frame, "save/rgb24_to_bgr24_cuda.jpg");
  TestWithLibyuvCPU(frame, "save/nv12_to_rgb24_libyuv.jpg", "save/rgb24_to_bgr24_libyuv.jpg");

  LoadImageAndConvertToNV21(image_path, frame);
  AllocateGpuMemory(frame);
  CopyToGpu(frame);
  TestNV21ToBGR24_CUDA(frame, "save/nv21_to_bgr24_cuda.jpg");
  TestNV21ToRGB24_CUDA(frame, "save/nv21_to_rgb24_cuda.jpg");

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
  AllocateGpuMemory(uniform_frame);
  CopyToGpu(uniform_frame);

  TestNV12ToBGR24_CUDA(uniform_frame, "save/nv12_to_bgr24_cuda_uniform.jpg");
  TestChannelConsistency(uniform_frame, test_r, test_g, test_b);

  TestNV12ToRGB24_CUDA(uniform_frame, "save/nv12_to_rgb24_cuda_uniform.jpg");
  TestChannelConsistency(uniform_frame, test_r, test_g, test_b);

  TestChannelConsistencyLibyuvCPU(uniform_frame, test_r, test_g, test_b);

  std::cout << " Demo completed " << std::endl;

  return 0;
}