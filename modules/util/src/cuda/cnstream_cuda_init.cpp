#include "cuda/cnstream_cuda_init.hpp"

#include <cuda_runtime.h>

extern "C" {
#include <libavutil/error.h>
}

#include <chrono>
#include <cstdlib>
#include <thread>

#include "cnstream_logging.hpp"

namespace cnstream {

namespace {

using Clock = std::chrono::steady_clock;

int64_t ElapsedMs(Clock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - begin).count();
}

int ReadEnvInt(const char* name, int defval, int minval) {
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0') {
    return defval;
  }
  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (end == text || (end != nullptr && *end != '\0') || parsed < minval) {
    LOGW(CUDA_ENV) << "[CUDA_INIT] invalid " << name << "=\"" << text
                   << "\", fallback to default " << defval;
    return defval;
  }
  return static_cast<int>(parsed);
}

// 退避时长
int BackoffMsForAttempt(const CudaInitRetryPolicy& policy, int attempt) {
  int64_t backoff = static_cast<int64_t>(policy.backoff_ms) << (attempt - 1);
  if (backoff > policy.max_backoff_ms) {
    backoff = policy.max_backoff_ms;
  }
  backoff += (attempt * 137) % 250;
  return static_cast<int>(backoff);
}

}  // namespace

CudaInitErrorClass ClassifyCudaRuntimeError(int code) {
  switch (code) {
    case cudaSuccess:
      return CudaInitErrorClass::kNone;
    // 瞬时争用
    case cudaErrorMemoryAllocation:
    case cudaErrorDevicesUnavailable:
    case cudaErrorNotReady:
      return CudaInitErrorClass::kTransient;
    default:
      return CudaInitErrorClass::kPermanent;
  }
}

CudaInitErrorClass ClassifyFFmpegError(int averror) {
  switch (averror) {
    case 0:
      return CudaInitErrorClass::kNone;
    // AVERROR_EXTERNAL：FFmpeg hwcontext 对驱动错误的包装（含 cuCtxCreate OOM）
    case AVERROR(ENOMEM):
    case AVERROR(EAGAIN):
    case AVERROR_EXTERNAL:
      return CudaInitErrorClass::kTransient;
    default:
      return CudaInitErrorClass::kPermanent;
  }
}

std::string CudaInitErrorName(int code) {
  if (code == kCudaInitNoCode) {
    return "no-code";
  }
  if (code < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(code, errbuf, sizeof(errbuf));
    return std::string(errbuf);
  }
  const char* name = cudaGetErrorName(static_cast<cudaError_t>(code));
  return name != nullptr ? std::string(name) : ("code " + std::to_string(code));
}

CudaInitGate& CudaInitGate::Instance() {
  static CudaInitGate instance;
  return instance;
}

CudaInitGate::CudaInitGate() : enabled_(true) {
  const char* text = std::getenv("VSTREAM_CUDA_INIT_SERIALIZE");
  if (text != nullptr && text[0] == '0' && text[1] == '\0') {
    enabled_ = false;
  }
}

// CudaInitGateGuard RAII 构造时调用，加锁耗时超过 500ms 时警告
void CudaInitGate::Enter(const char* stage) {
  auto begin = Clock::now();
  mtx_.lock();
  int64_t waited_ms = ElapsedMs(begin);
  if (waited_ms >= 500) {
    LOGW(CUDA_ENV) << "CudaInitGate wait " << waited_ms << " ms ["
                   << (stage != nullptr ? stage : "?") << "]";
  }
}

void CudaInitGate::Leave() {
  mtx_.unlock();
}

CudaInitGateGuard::CudaInitGateGuard(const char* stage)
    : held_(CudaInitGate::Instance().enabled()) {
  if (held_) {
    CudaInitGate::Instance().Enter(stage);
  }
}

CudaInitGateGuard::~CudaInitGateGuard() {
  if (held_) {
    CudaInitGate::Instance().Leave();
  }
}

const CudaInitRetryPolicy& GetCudaInitRetryPolicy() {
  static const CudaInitRetryPolicy policy = []() {
    CudaInitRetryPolicy p;
    p.attempts = ReadEnvInt("VSTREAM_CUDA_INIT_RETRY_ATTEMPTS", p.attempts, 1);
    p.backoff_ms = ReadEnvInt("VSTREAM_CUDA_INIT_RETRY_BACKOFF_MS", p.backoff_ms, 0);
    p.budget_ms = ReadEnvInt("VSTREAM_CUDA_INIT_RETRY_BUDGET_MS", p.budget_ms, 0);
    return p;
  }();
  return policy;
}

CudaInitResult RunCudaInit(const char* stage, const CudaInitCall& call,
                           CudaInitClassifier classify) {
  const CudaInitRetryPolicy& policy = GetCudaInitRetryPolicy();
  const char* name = stage != nullptr ? stage : "?";
  const auto begin = Clock::now();

  CudaInitResult result;
  for (int attempt = 1; attempt <= policy.attempts; ++attempt) {
    result.attempt = attempt;

    int code = 0;
    {
      // 闸门只覆盖创建动作本身，退避等待在闸门外进行
      CudaInitGateGuard guard(name);
      code = call();
    }
    result.last_error = code;

    // NOTE: 需要保证 call() 的成功语义是返回 0
    if (code == 0) {
      result.ok = true;
      result.last_class = CudaInitErrorClass::kNone;
      result.elapsed_ms = ElapsedMs(begin);
      if (attempt > 1) {
        LOGW(CUDA_ENV) << "[" << name << "] recovered " << attempt
                       << " attempts, cost " << result.elapsed_ms << " ms";
      } else {
        LOGI(CUDA_ENV) << "[" << name << "] ok cost " << result.elapsed_ms << " ms";
      }
      return result;
    }

    // 无错误码可分类
    result.last_class = classify != nullptr ? classify(code) : CudaInitErrorClass::kTransient;
    if (result.last_class != CudaInitErrorClass::kTransient) {
      LOGW(CUDA_ENV) << "[" << name << "] failed [permanent] rc=" << code
                     << " (" << CudaInitErrorName(code) << ") cost " << ElapsedMs(begin) << " ms";
      break;
    }

    const int64_t elapsed_ms = ElapsedMs(begin);
    const int wait_ms = BackoffMsForAttempt(policy, attempt);
    int total_ms = elapsed_ms + wait_ms;
    int budget_ms = policy.budget_ms;

    if (attempt >= policy.attempts || (budget_ms > 0 && total_ms > budget_ms)) {
      LOGE(CUDA_ENV) << "[" << name << "] " << attempt
                     << " attempt(s), cost " << elapsed_ms << " ms rc=" << code
                     << " (" << CudaInitErrorName(code) << ")";
      break;
    }

    LOGW(CUDA_ENV) << "[" << name << "] attempt=" << attempt << "/"
                   << policy.attempts << " rc=" << code << " (" << CudaInitErrorName(code)
                   << ") [transient] retry_in_ms=" << wait_ms;
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
  }

  result.elapsed_ms = ElapsedMs(begin);
  return result;
}

}  // namespace cnstream
