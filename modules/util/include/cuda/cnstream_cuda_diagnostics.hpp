#ifndef MODULES_UTIL_INCLUDE_CUDA_CNSTREAM_CUDA_DIAGNOSTICS_HPP_
#define MODULES_UTIL_INCLUDE_CUDA_CNSTREAM_CUDA_DIAGNOSTICS_HPP_

namespace cnstream {

/**
 * 在 CUDA 初始化失败现场采集进程内诊断快照。
 *
 * 只在失败路径调用；采集全部来自 /proc 和静态文件。
 * 每个进程按 stage+kind 只写一次，追加到 $VSTREAM_CUDA_DIAG_DIR/vstream_cuda_diag_<pid>.log。
 *
 * env:
 *   VSTREAM_CUDA_DIAG=0        关闭采集（默认开启：失败时自动采集）
 *   VSTREAM_CUDA_DIAG_DIR=dir  输出目录（默认 /tmp，需已存在）
 *
 * @param stage   失败阶段
 * @param kind    init（初始化失败）/ stale（初始化前发现陈旧错误）
 * @param code    原始错误码（CUDA 正值或 AVERROR 负值）
 * @param attempt 第几次尝试（0 表示非重试路径）
 */
void DumpCudaFailureDiagnostics(const char* stage, const char* kind, int code,
                                int attempt);

}  // namespace cnstream

#endif  // MODULES_UTIL_INCLUDE_CUDA_CNSTREAM_CUDA_DIAGNOSTICS_HPP_
