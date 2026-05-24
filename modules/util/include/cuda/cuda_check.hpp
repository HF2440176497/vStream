
#ifndef CUDA_CHECK_HPP_
#define CUDA_CHECK_HPP_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <npp.h>
#include <nppi.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

namespace cnstream {

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

#ifdef VSTREAM_UNIT_TEST
#define CHECK_CUDA_RUNTIME(op) __check_cuda_runtime_debug((op), #op, __FILE__, __LINE__)
#else
#define CHECK_CUDA_RUNTIME(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)
#endif

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

inline bool __check_cuda_runtime_debug(cudaError_t code, const char* op, const char* file, int line) {
  if (code != cudaSuccess) {
    const char* err_name = cudaGetErrorName(code);
    const char* err_message = cudaGetErrorString(code);
    printf("check_cuda_runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", 
		file, line, op, err_name, err_message);
    printf("Aborting for debugging. Use 'bt' in GDB to see the stack trace.\n");
    abort();
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

}  // namespace cnstream

#endif  // CUDA_CHECK_HPP_