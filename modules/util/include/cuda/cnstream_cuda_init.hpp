#ifndef MODULES_UTIL_CUDA_CNSTREAM_CUDA_INIT_HPP_
#define MODULES_UTIL_CUDA_CNSTREAM_CUDA_INIT_HPP_

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace cnstream {

/**
 * 启动期 CUDA / NVENC 资源初始化保护：进程内串行 + 有界退避重试 + 结构化日志。
 *
 *   1) CudaInitGate —— 进程级闸门，串行化设备资源的创建路径
 *   2) RunCudaInit —— 对单次创建做有界退避重试，吸收跨进程的瞬时争用；
 *      重试之间不持锁睡眠，失败按错误分类决定是否继续重试。
 *
 * 环境变量（进程级，只读取一次；缺省值即推荐值）：
 *   VSTREAM_CUDA_INIT_SERIALIZE=0          关闭闸门（回滚开关）
 *   VSTREAM_CUDA_INIT_RETRY_ATTEMPTS=n     单点最大尝试次数（默认 3）
 *   VSTREAM_CUDA_INIT_RETRY_BACKOFF_MS=n   首次退避毫秒数（默认 1000，指数增长）
 *   VSTREAM_CUDA_INIT_RETRY_BUDGET_MS=n    单点重试总预算毫秒数（默认 30000）
 */

// 错误分类
enum class CudaInitErrorClass {
  kNone = 0,   // 成功
  kTransient,  // 瞬时资源争用（分配被拒 / 设备暂不可用 / 尚未就绪）—— 在预算内重试
  kPermanent,  // 配置或环境问题（无设备、非法设备号、驱动不匹配等）—— 立即失败
};

/**
 * 按 CUDA runtime 错误码分类
 * 未列出的错误码按 kPermanent 处理，"非预期错误快速失败"
 */
CudaInitErrorClass ClassifyCudaRuntimeError(int code);

/**
 * 按 FFmpeg 错误码（AVERROR）分类
 */
CudaInitErrorClass ClassifyFFmpegError(int averror);

/** 把错误码转成可读文本（负值按 AVERROR 解析，非负按 cudaError_t 解析）。 */
std::string CudaInitErrorName(int code);

/**
 * 进程级初始化闸门。可重入：同一线程内的嵌套创建（如 probe 内再取闸门）不会自锁。
 */
class CudaInitGate {
 public:
  static CudaInitGate& Instance();

  void Enter(const char* stage);
  void Leave();
  bool enabled() const { return enabled_; }

 private:
  CudaInitGate();

  std::recursive_mutex mtx_;
  bool enabled_;
};

/** 闸门 RAII 守卫；构造时取闸门，析构时释放。 */
class CudaInitGateGuard {
 public:
  explicit CudaInitGateGuard(const char* stage);
  ~CudaInitGateGuard();

  CudaInitGateGuard(const CudaInitGateGuard&) = delete;
  CudaInitGateGuard& operator=(const CudaInitGateGuard&) = delete;

 private:
  bool held_;
};

struct CudaInitRetryPolicy {
  int attempts = 3;
  int backoff_ms = 1000;
  int max_backoff_ms = 8000;
  int budget_ms = 30000;  // 总预算耗时上限
};

const CudaInitRetryPolicy& GetCudaInitRetryPolicy();

struct CudaInitResult {
  bool ok = false;
  int attempt = 0;                                    // 实际尝试次数
  int64_t elapsed_ms = 0;                             // 含退避等待的总耗时
  int last_error = 0;                                 // 最后一次的原始错误码
  CudaInitErrorClass last_class = CudaInitErrorClass::kNone;
};

using CudaInitCall = std::function<int()>;
using CudaInitClassifier = CudaInitErrorClass (*)(int);

/**
 * 「失败但没有错误码可上报」的占位返回值（例如 TensorRT 只返回 nullptr）。
 * 取值刻意避开 CUDA 错误码（非负小整数）与 AVERROR（小负 errno 或 FFERRTAG 编码），
 * 使 CudaInitErrorName() 能明确显示为 no-code，而不是被 av_strerror 解成无关的
 * "Operation not permitted"。
 */
constexpr int kCudaInitNoCode = INT32_MIN;

/**
 * 执行一次"设备资源创建"，内部实施编排：闸门串行 + 有界退避重试 + 结构化日志。
 *
 * @param stage    阶段名，用于日志定位（如 "probe.malloc"、"model.trt_exec_context"）
 * @param call     返回 0 表示成功的可调用对象；非 0 视为原始错误码
 * @param classify 错误码分类函数；传 nullptr 表示"没有错误码可分类"
 *                （如 TensorRT 只返回 nullptr），此时按瞬时争用处理（有界重试）
 * @return 含成功标志、尝试次数、耗时与最后错误码的结果；调用方据此决定是否继续
 */
CudaInitResult RunCudaInit(const char* stage, const CudaInitCall& call,
                           CudaInitClassifier classify = nullptr);

}  // namespace cnstream

#endif  // MODULES_UTIL_CUDA_CNSTREAM_CUDA_INIT_HPP_
