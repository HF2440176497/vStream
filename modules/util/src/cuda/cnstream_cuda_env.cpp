#include "cuda/cnstream_cuda_env.hpp"

#include <cuda_runtime.h>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
}

#include <mutex>
#include <string>
#include <unordered_map>

#include "cnstream_logging.hpp"
#include "cuda/cnstream_cuda_init.hpp"

namespace cnstream {

namespace {

std::mutex g_hwdevice_cache_mtx;
std::unordered_map<int, AVBufferRef*> g_hwdevice_cache;

}  // namespace

bool ProbeCudaDevice(int device_id) {
  int device_count = 0;
  CudaInitResult ret = RunCudaInit("ProbeCudaDevice",
                                   [&device_count]() -> int {
                                     return static_cast<int>(cudaGetDeviceCount(&device_count));
                                   },
                                   ClassifyCudaRuntimeError);
  if (!ret.ok) {
    LOGE(CUDA_ENV) << "CUDA probe failed: cudaGetDeviceCount returned "
                   << CudaInitErrorName(ret.last_error) << " [" << ret.attempt
                   << " attempt(s), " << ret.elapsed_ms << " ms]";
    return false;
  }
  if (device_count <= 0 || device_id < 0 || device_id >= device_count) {
    LOGE(CUDA_ENV) << "CUDA probe failed: device_id=" << device_id
                   << " invalid, visible device count=" << device_count
                   << " (note: CUDA_VISIBLE_DEVICES may restrict visibility in container).";
    return false;
  }

  ret = RunCudaInit("ProbeCudaDeviceSet",
                    [device_id]() -> int { return static_cast<int>(cudaSetDevice(device_id)); },
                    ClassifyCudaRuntimeError);
  if (!ret.ok) {
    LOGE(CUDA_ENV) << "CUDA probe failed: cudaSetDevice(" << device_id << ") returned "
                   << CudaInitErrorName(ret.last_error) << " [" << ret.attempt
                   << " attempt(s), " << ret.elapsed_ms << " ms]";
    return false;
  }

  // 真实分配一次：强制建立 primary context，并验证当前确实可分配
  ret = RunCudaInit("ProbeCudaDeviceMalloc",
                    []() -> int {
                      void* probe_ptr = nullptr;
                      cudaError_t err = cudaMalloc(&probe_ptr, 1024);
                      if (err != cudaSuccess) {
                        return static_cast<int>(err);
                      }
                      cudaFree(probe_ptr);
                      return 0;
                    },
                    ClassifyCudaRuntimeError);
  if (!ret.ok) {
    LOGE(CUDA_ENV) << "CUDA probe failed: context creation / memory allocation on device "
                   << device_id << " returned " << CudaInitErrorName(ret.last_error)
                   << " [" << ret.attempt
                   << " attempt(s), " << ret.elapsed_ms << " ms]";
    return false;
  }
  return true;
}

AVBufferRef* AcquireSharedCudaHwDeviceCtx(int device_id) {
  {
    std::lock_guard<std::mutex> lk(g_hwdevice_cache_mtx);
    auto it = g_hwdevice_cache.find(device_id);
    if (it != g_hwdevice_cache.end()) {
      return av_buffer_ref(it->second);
    }
  }

  // 未命中缓存的创建放在锁外（带闸门 + 有界重试）：
  // 避免退避等待期间持有缓存锁，阻塞其它线程查缓存
  AVBufferRef* ref = nullptr;
  CudaInitResult ret = RunCudaInit("SharedCudaHwDeviceCtx",
                                   [&ref, device_id]() -> int {
                                     int err = av_hwdevice_ctx_create(
                                         &ref, AV_HWDEVICE_TYPE_CUDA,
                                         std::to_string(device_id).c_str(), nullptr, 0);
                                     if (err < 0 || ref == nullptr) {
                                       // 失败路径若仍返回了部分初始化的引用，就地释放后再上报
                                       if (ref != nullptr) {
                                         av_buffer_unref(&ref);
                                       }
                                       return err < 0 ? err : AVERROR_EXTERNAL;
                                     }
                                     return 0;
                                   },
                                   ClassifyFFmpegError);
  if (!ret.ok) {
    char errbuf[128] = {0};
    av_strerror(ret.last_error, errbuf, sizeof(errbuf));
    LOGE(CUDA_ENV) << "av_hwdevice_ctx_create(CUDA, device " << device_id << ") failed: "
                   << ret.last_error << " (" << errbuf << ")"
                   << " after " << ret.attempt << " attempt(s), " << ret.elapsed_ms << " ms";
    return nullptr;
  }

  // 并发创建时保留先入缓存的那个，本次多余的引用释放掉
  std::lock_guard<std::mutex> lk(g_hwdevice_cache_mtx);
  auto it = g_hwdevice_cache.find(device_id);
  if (it != g_hwdevice_cache.end()) {
    AVBufferRef* existing = av_buffer_ref(it->second);
    av_buffer_unref(&ref);
    return existing;
  }

  // 缓存持有的引用有意不释放（进程级 keep-alive），与缓存生命周期一致
  g_hwdevice_cache.emplace(device_id, ref);
  return av_buffer_ref(ref);
}

}  // namespace cnstream
