

#include "cuda/transfmt_cuda.cuh"

#include <cstring>


namespace cnstream {

#define CHECK_NPP(op) __check_npp((op), #op, __FILE__, __LINE__)

static std::string nppGetStatusString(NppStatus code) {
    return "NPP error code: " + std::to_string(code);
}

static bool __check_npp(NppStatus code, const char* op, const char* file, int line) {
  if (code != NPP_SUCCESS) {
    printf("check_npp error %s:%d  %s failed. \n  code = %d, message = %s\n",
           file, line, op, code, nppGetStatusString(code).c_str());
    return false;
  }
  return true;
}

int NppNV12ToRGB24(void* dst, int dst_stride,
                  const void* y_plane,
                  const void* uv_plane,
                  int src_stride,
                  int width, 
                  int height, 
                  cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  const Npp8u* aSrc[2] = {
    static_cast<const Npp8u*>(y_plane),
    static_cast<const Npp8u*>(uv_plane),
  };

  Npp8u* pDst = static_cast<Npp8u*>(dst);

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
    aSrc, src_stride,
    pDst, dst_stride,
    oSizeROI,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

int NppNV12ToBGR24(void* dst, int dst_stride,
                  const void* y_plane,
                  const void* uv_plane,
                  int src_stride,
                  int width,
                  int height, 
                  cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  const Npp8u* aSrc[2] = {
    static_cast<const Npp8u*>(y_plane),
    static_cast<const Npp8u*>(uv_plane),
  };

  Npp8u* pDst = static_cast<Npp8u*>(dst);

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  status = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(
    aSrc, src_stride,
    pDst, dst_stride,
    oSizeROI,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

/**
 * BT.709 HDTV: 转换 NV21 到 RGB24
 */
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
 * BT.709 HDTV: 转换 NV21 到 BGR24
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


int NppNV21ToRGB24(void* dst, int width, int height, 
  const void* y_plane, const void* uv_plane, cudaStream_t stream) {

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  
  nv21ToRGBKernel<<<grid, block, 0, stream>>>(static_cast<const uint8_t*>(y_plane), static_cast<const uint8_t*>(uv_plane), 
                    static_cast<uint8_t*>(dst), width, height);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}


int NppNV21ToBGR24(void* dst, int width, int height, 
  const void* y_plane, const void* uv_plane, cudaStream_t stream) {

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  
  nv21ToBGRKernel<<<grid, block, 0, stream>>>(static_cast<const uint8_t*>(y_plane), static_cast<const uint8_t*>(uv_plane), 
                    static_cast<uint8_t*>(dst), width, height);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

static int NppSwapChannels_8u_C3R(void* dst, int width, int height, const void* src, cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  int nStep = width * 3;

  int aDstOrder[3] = { 2, 1, 0 };
  status = nppiSwapChannels_8u_C3R_Ctx(
    static_cast<const Npp8u*>(src),
    nStep,
    static_cast<Npp8u*>(dst),
    nStep,
    oSizeROI,
    aDstOrder,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

int NppRGB24ToBGR24(void* dst, int width, int height, const void* src, cudaStream_t stream) {
  return NppSwapChannels_8u_C3R(dst, width, height, src, stream);
}

int NppBGR24ToRGB24(void* dst, int width, int height, const void* src, cudaStream_t stream) {
  return NppSwapChannels_8u_C3R(dst, width, height, src, stream);
}

}  // namespace cnstream
