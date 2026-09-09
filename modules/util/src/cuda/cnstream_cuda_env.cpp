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

namespace cnstream {

namespace {

std::mutex g_hwdevice_cache_mtx;
std::unordered_map<int, AVBufferRef*> g_hwdevice_cache;

}  // namespace

bool ProbeCudaDevice(int device_id) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    LOGE(CUDA_ENV) << "CUDA probe failed: cudaGetDeviceCount returned "
                   << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ")."
                   << " GPU stack unavailable: check NVIDIA driver/kernel module state"
                   << " (nvidia-smi, dmesg | grep -i -E 'nvrm|xid') and"
                   << " libcuda vs host driver version mismatch inside container.";
    return false;
  }
  if (device_count <= 0 || device_id < 0 || device_id >= device_count) {
    LOGE(CUDA_ENV) << "CUDA probe failed: device_id=" << device_id
                   << " invalid, visible device count=" << device_count
                   << " (note: CUDA_VISIBLE_DEVICES may restrict visibility in container).";
    return false;
  }

  err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    LOGE(CUDA_ENV) << "CUDA probe failed: cudaSetDevice(" << device_id << ") returned "
                   << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ").";
    return false;
  }

  // 真实分配一次：强制建立 primary context，并验证当前确实可分配
  void* probe_ptr = nullptr;
  err = cudaMalloc(&probe_ptr, 1024);
  if (err != cudaSuccess) {
    LOGE(CUDA_ENV) << "CUDA probe failed: context creation / memory allocation on device "
                   << device_id << " returned " << cudaGetErrorName(err)
                   << " (" << cudaGetErrorString(err) << ")."
                   << " If this is an out-of-memory error while nvidia-smi shows free memory,"
                   << " it usually indicates kernel-side driver resource exhaustion"
                   << " (try: rmmod/modprobe nvidia_uvm on host, or reboot).";
    return false;
  }
  cudaFree(probe_ptr);
  return true;
}

AVBufferRef* AcquireSharedCudaHwDeviceCtx(int device_id) {
  std::lock_guard<std::mutex> lk(g_hwdevice_cache_mtx);

  auto it = g_hwdevice_cache.find(device_id);
  if (it != g_hwdevice_cache.end()) {
    return av_buffer_ref(it->second);
  }

  AVBufferRef* ref = nullptr;
  int ret = av_hwdevice_ctx_create(&ref, AV_HWDEVICE_TYPE_CUDA,
                                   std::to_string(device_id).c_str(), nullptr, 0);
  if (ret < 0 || ref == nullptr) {
    char errbuf[128] = {0};
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGE(CUDA_ENV) << "av_hwdevice_ctx_create(CUDA, device " << device_id << ") failed: "
                   << ret << " (" << errbuf << ")";
    return nullptr;
  }

  // 缓存持有的引用有意不释放（进程级 keep-alive），与缓存生命周期一致
  g_hwdevice_cache.emplace(device_id, ref);
  return av_buffer_ref(ref);
}

}  // namespace cnstream
