#include "cuda/cnstream_cuda_env.hpp"

#include <cuda_runtime.h>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <mutex>
#include <string>
#include <unordered_map>

#include "cnstream_logging.hpp"

namespace cnstream {

namespace {

struct SharedHwDeviceEntry {
  AVBufferRef* ref = nullptr;  // 缓存持有的引用：进程存活期内不释放
  std::recursive_mutex ctx_lock;  // FFmpeg 组件共享 CUcontext 的互斥保护
};

std::mutex g_hwdevice_cache_mtx;
std::unordered_map<int, SharedHwDeviceEntry*> g_hwdevice_cache;

void CudaHwctxLock(void* lock_ctx) {
  static_cast<std::recursive_mutex*>(lock_ctx)->lock();
}

void CudaHwctxUnlock(void* lock_ctx) {
  static_cast<std::recursive_mutex*>(lock_ctx)->unlock();
}

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
    return av_buffer_ref(it->second->ref);
  }

  auto* entry = new SharedHwDeviceEntry();
  int ret = av_hwdevice_ctx_create(&entry->ref, AV_HWDEVICE_TYPE_CUDA,
                                   std::to_string(device_id).c_str(), nullptr, 0);
  if (ret < 0 || entry->ref == nullptr) {
    char errbuf[128] = {0};
    av_strerror(ret, errbuf, sizeof(errbuf));
    LOGE(CUDA_ENV) << "av_hwdevice_ctx_create(CUDA, device " << device_id << ") failed: "
                   << ret << " (" << errbuf << ")";
    delete entry;
    return nullptr;
  }

  // 多个 nvenc/cuvid 组件共用同一 CUcontext：通过 FFmpeg 提供的 lock/unlock
  // 回调串行化其驱动 API 调用（与 ffmpeg CLI 共享 -init_hw_device 的做法一致）
  auto* hwdev = reinterpret_cast<AVHWDeviceContext*>(entry->ref->data);
  auto* hwctx = reinterpret_cast<AVCUDADeviceContext*>(hwdev->data);
  hwctx->lock = &CudaHwctxLock;
  hwctx->unlock = &CudaHwctxUnlock;
  hwctx->lock_ctx = &entry->ctx_lock;

  // entry 有意不释放（进程级 keep-alive），与缓存生命周期一致
  g_hwdevice_cache.emplace(device_id, entry);
  return av_buffer_ref(entry->ref);
}

}  // namespace cnstream
