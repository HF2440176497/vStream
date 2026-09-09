#ifndef MODULES_UTIL_CUDA_CNSTREAM_CUDA_ENV_HPP_
#define MODULES_UTIL_CUDA_CNSTREAM_CUDA_ENV_HPP_

struct AVBufferRef;

namespace cnstream {

/**
 * CUDA 可用性探针（fail-fast）。
 *
 * 在加载模型 / 初始化推拉流之前调用：依次执行 cudaGetDeviceCount →
 * cudaSetDevice → 一次真实 cudaMalloc/cudaFree，强制完成 cuInit 与
 * primary context 建立。把“初始化阶段的驱动级失败”（如 cuInit/cuCtxCreate
 * 返回 out of memory、驱动状态损坏）从 TensorRT/FFmpeg 深处提前到调用点，
 * 并给出明确错误码与排查方向。
 *
 * @param device_id 待探测的 CUDA 设备号（进程内可见编号，受
 *                 CUDA_VISIBLE_DEVICES 影响）。
 * @return 环境可用返回 true；失败返回 false（已打印错误详情）。
 */
bool ProbeCudaDevice(int device_id);

/**
 * 进程级共享的 CUDA AVHWDeviceContext（按 device_id 缓存）。
 *
 * 语义：同一 device_id 的所有推流/拉流句柄共用一个 FFmpeg CUDA 设备上下文
 * （即一个 CUcontext），替代“每条流独立 av_hwdevice_ctx_create”的旧模式，
 * 把异常退出时驱动需要回收的 context 数量从 O(流数) 降为 O(设备数)。
 *
 * 生命周期：返回的 AVBufferRef* 引用计数已 +1，调用方负责 av_buffer_unref；
 * 缓存自身持有的引用存活到进程退出（刻意的 keep-alive，避免流级重建context 的开销与生命周期耦合）
 * 并发安全由 FFmpeg 保证
 *
 * @param device_id CUDA 设备号。
 * @return 可用的设备上下文引用；失败返回 nullptr（已打印日志）。
 */
AVBufferRef* AcquireSharedCudaHwDeviceCtx(int device_id);

}  // namespace cnstream

#endif  // MODULES_UTIL_CUDA_CNSTREAM_CUDA_ENV_HPP_
