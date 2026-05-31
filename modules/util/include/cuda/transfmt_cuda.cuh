#ifndef TRANSFMT_CUDA_CUH_
#define TRANSFMT_CUDA_CUH_

#include <cuda_runtime.h>
#include <npp.h>

#include "data_source_param.hpp"
#include "cuda/cuda_check.hpp"

namespace cnstream {


int NppNV12ToRGB24(void* dst, int dst_stride,
                  const void* y_plane,
                  const void* uv_plane,
                  int src_stride,
                  int width, 
                  int height, 
                  cudaStream_t stream = nullptr);

int NppNV12ToBGR24(void* dst, int dst_stride,
                  const void* y_plane,
                  const void* uv_plane,
                  int src_stride,
                  int width,
                  int height, 
                  cudaStream_t stream = nullptr);

int NppRGB24ToNV12(void* dst_y, void* dst_uv,
                  int y_stride, 
                  int uv_stride,
                  const void* src,
                  int src_stride,
                  int width, 
                  int height, 
                  cudaStream_t stream = nullptr);

int NppBGR24ToNV12(void* dst_y, void* dst_uv,
                  int y_stride, 
                  int uv_stride,
                  const void* src,
                  int src_stride,
                  int width, 
                  int height, 
                  cudaStream_t stream = nullptr);

int NppNV21ToRGB24(void* dst, int dst_stride,
                int width, 
                int height, 
                const void* y_plane, int y_stride,
                const void* uv_plane, int uv_stride,
                cudaStream_t stream = nullptr);

int NppNV21ToBGR24(void* dst, int dst_stride,
                int width, 
                int height, 
                const void* y_plane, int y_stride,
                const void* uv_plane, int uv_stride,
                cudaStream_t stream = nullptr);

int NppRGB24ToBGR24(void* dst, int dst_stride,
                int width, 
                int height, 
                const void* src, int src_stride,
                cudaStream_t stream = nullptr);

int NppBGR24ToRGB24(void* dst, int dst_stride,
                int width,
                int height, 
                const void* src, int src_stride,
                cudaStream_t stream = nullptr);

int ConvertRGB24ToNV12_Resize(void* dst_y, int y_stride, void* dst_uv, int uv_stride,
                            int dst_width, int dst_height,
                            const void* src, int src_stride, int src_width, int src_height,
                            cudaStream_t stream = nullptr);

int ConvertBGR24ToNV12_Resize(void* dst_y, int y_stride, void* dst_uv, int uv_stride,
                            int dst_width, int dst_height,
                            const void* src, int src_stride, int src_width, int src_height,
                            cudaStream_t stream = nullptr);

}  // namespace cnstream

#endif  // TRANSFMT_CUDA_CUH_
