#ifndef MODULES_UTIL_INCLUDE_DIAGNOSTICS_HPP_
#define MODULES_UTIL_INCLUDE_DIAGNOSTICS_HPP_

namespace cnstream {

/**
 * 平台无关的诊断快照。
 *
 * 每个进程按 stage+kind 只写一次，追加到：
 *   $VSTREAM_CUDA_DIAG_DIR/vstream_cuda_diag_<mode>_<pid>.log
 * 其中 mode = normal / failure，正常与异常文件分离，便于对比。
 *
 * env:
 *   VSTREAM_CUDA_DIAG=0          关闭全部采集
 *   VSTREAM_CUDA_DIAG_NORMAL=0   只关闭 normal 采集（默认开启）
 *   VSTREAM_CUDA_DIAG_NVML=0     关闭设备侧采集（默认通过 dlopen 读 NVML）
 *   VSTREAM_CUDA_DIAG_DIR=dir    输出目录（默认 /tmp，需已存在）
 *
 * @param stage   阶段（load / probe / infer_pool / sink_stream / CUDA 初始化 stage）
 * @param kind    init（初始化失败）/ stale（初始化前发现陈旧错误）/ normal
 * @param code    原始错误码（CUDA 正值或 AVERROR 负值）
 * @param attempt 第几次尝试（0 表示非重试路径）
 */
void DumpDiagSnapshotOnError(const char* stage, const char* kind, int code,
                             int attempt);

/**
 * 采集一次正常路径快照（每个 stage 每进程只写一次）。
 * 包含 VmSize/VmData/smaps_rollup/overcommit/dev/shm/
 * 进程内 nvidia fd 持有者及父进程链（可用于发现旧进程/forkserver 父进程残留）、
 * 最大 VMA 列表（定位 VmSize 的固定 VA 预留来源）、
 * NVML 设备状态（显存/BAR1/compute 进程数）。
 */
void DumpDiagSnapshotAtStage(const char* stage);

}  // namespace cnstream

#endif  // MODULES_UTIL_INCLUDE_DIAGNOSTICS_HPP_
