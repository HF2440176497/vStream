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


int NppNV21ToRGB24(void* dst, 
                int width, 
                int height, 
                const void* y_plane, 
                const void* uv_plane,
                cudaStream_t stream = nullptr);

int NppNV21ToBGR24(void* dst, 
                int width, 
                int height, 
                const void* y_plane, 
                const void* uv_plane, 
                cudaStream_t stream = nullptr);

int NppRGB24ToBGR24(void* dst, 
                int width, 
                int height, 
                const void* src, 
                cudaStream_t stream = nullptr);

int NppBGR24ToRGB24(void* dst, 
                int width,
                int height, 
                const void* src, 
                cudaStream_t stream = nullptr);

}  // namespace cnstream

#endif  // TRANSFMT_CUDA_CUH_
