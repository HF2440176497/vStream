
#include "cuda/transfmt_cuda.cuh"
#include <cstring>

namespace cnstream {

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


int NppRGB24ToNV12(void* dst_y, void* dst_uv,
                  int y_stride, 
                  int uv_stride,
                  const void* src,
                  int src_stride,
                  int width, 
                  int height, 
                  cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  const Npp8u* pSrc = static_cast<const Npp8u*>(src);

  Npp8u* pDst[2] = {(Npp8u*)dst_y, (Npp8u*)dst_uv};
  int DstStep[2] = {y_stride, uv_stride};

  status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
    pSrc, src_stride,
    pDst, DstStep,
    oSizeROI,
    MATRIX_RGB_TO_YUV709,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

/**
 * 仅改变矩阵，复用 NppRGB24ToNV12 的逻辑
 */
int NppBGR24ToNV12(void* dst_y, void* dst_uv,
                  int y_stride, 
                  int uv_stride,
                  const void* src,
                  int src_stride,
                  int width, 
                  int height, 
                  cudaStream_t stream) {

  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  const Npp8u* pSrc = static_cast<const Npp8u*>(src);

  Npp8u* pDst[2] = {(Npp8u*)dst_y, (Npp8u*)dst_uv};
  int DstStep[2] = {y_stride, uv_stride};

  status = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
    pSrc, src_stride,
    pDst, DstStep,
    oSizeROI,
    MATRIX_BGR_TO_YUV709,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}


/**
 * BT.709 HDTV: 转换 NV21 到 RGB24
 * @note: stride 以字节为单位
 */
__global__ void nv21ToRGBKernel(const uint8_t* yPlane, int yStride,
                                const uint8_t* vuPlane, int vuStride,
                                uint8_t* rgbOutput, int dstStride,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint8_t Y = yPlane[y * yStride + x];
    
    int uvX = x / 2;
    int uvY = y / 2;
    int uvIdx = uvY * vuStride + uvX * 2;  // 字节偏移量

    uint8_t V = vuPlane[uvIdx];
    uint8_t U = vuPlane[uvIdx + 1];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;
    
    int R = (298 * C + 459 * E + 128) >> 8;
    int G = (298 * C - 55 * D - 137 * E + 128) >> 8;
    int B = (298 * C + 541 * D + 128) >> 8;
    
    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));
    
    int outIdx = y * dstStride + x * 3;
    rgbOutput[outIdx] = R;
    rgbOutput[outIdx + 1] = G;
    rgbOutput[outIdx + 2] = B;
}

/**
 * BT.709 HDTV: 转换 NV21 到 BGR24
 */
__global__ void nv21ToBGRKernel(const uint8_t* yPlane, int yStride,
                                const uint8_t* vuPlane, int vuStride,
                                uint8_t* bgrOutput, int dstStride,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint8_t Y = yPlane[y * yStride + x];
    
    int uvX = x / 2;
    int uvY = y / 2;
    int uvIdx = uvY * vuStride + uvX * 2;  // 字节偏移量

    uint8_t V = vuPlane[uvIdx];
    uint8_t U = vuPlane[uvIdx + 1];

    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;
    
    int R = (298 * C + 459 * E + 128) >> 8;
    int G = (298 * C - 55 * D - 137 * E + 128) >> 8;
    int B = (298 * C + 541 * D + 128) >> 8;
    
    R = max(0, min(255, R));
    G = max(0, min(255, G));
    B = max(0, min(255, B));
    
    int outIdx = y * dstStride + x * 3;
    bgrOutput[outIdx] = B;
    bgrOutput[outIdx + 1] = G;
    bgrOutput[outIdx + 2] = R;
}


int NppNV21ToRGB24(void* dst, int dst_stride,
                int width, int height, 
                const void* y_plane, int y_stride,
                const void* uv_plane, int uv_stride,
                cudaStream_t stream) {

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  
  nv21ToRGBKernel<<<grid, block, 0, stream>>>(
    static_cast<const uint8_t*>(y_plane), y_stride,
    static_cast<const uint8_t*>(uv_plane), uv_stride,
    static_cast<uint8_t*>(dst), dst_stride,
    width, height);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}


int NppNV21ToBGR24(void* dst, int dst_stride,
                int width, int height, 
                const void* y_plane, int y_stride,
                const void* uv_plane, int uv_stride,
                cudaStream_t stream) {

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  
  nv21ToBGRKernel<<<grid, block, 0, stream>>>(
    static_cast<const uint8_t*>(y_plane), y_stride,
    static_cast<const uint8_t*>(uv_plane), uv_stride,
    static_cast<uint8_t*>(dst), dst_stride,
    width, height);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

static int NppSwapChannels_8u_C3R(void* dst, int dst_stride,
                                   int width, int height,
                                   const void* src, int src_stride,
                                   cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  int aDstOrder[3] = { 2, 1, 0 };
  status = nppiSwapChannels_8u_C3R_Ctx(
    static_cast<const Npp8u*>(src),
    src_stride,
    static_cast<Npp8u*>(dst),
    dst_stride,
    oSizeROI,
    aDstOrder,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

int NppRGB24ToBGR24(void* dst, int dst_stride,
                int width, int height,
                const void* src, int src_stride,
                cudaStream_t stream) {
  return NppSwapChannels_8u_C3R(dst, dst_stride, width, height, src, src_stride, stream);
}

int NppBGR24ToRGB24(void* dst, int dst_stride,
                int width, int height,
                const void* src, int src_stride,
                cudaStream_t stream) {
  return NppSwapChannels_8u_C3R(dst, dst_stride, width, height, src, src_stride, stream);
}

}  // namespace cnstream
