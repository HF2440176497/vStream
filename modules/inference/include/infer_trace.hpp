
#ifndef MODULES_INFERENCE_DBG_TRACE_HPP_
#define MODULES_INFERENCE_DBG_TRACE_HPP_

#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>

#include <glog/logging.h>

/**
 * @brief 异步流水线追踪日志
 *
 * 开关：运行前设置环境变量 VSTREAM_INFERTRACE=1
 * 未开启时 INFERTRACE 行为空语句，无格式化开销；开启后每行带统一前缀 [INFERTRACE]
 *
 * 行格式：
 *   [INFERTRACE][<阶段>][tid=<线程短标签>] bt=<批次首帧ts> ... slot=<n> buf=0x...
 *
 * 验证方法：
 * - 批间并发：不同 bt 的 H2D/INFER/D2H 行在时间上交错（tid 不同）
 * - buffer 一致性：同一 bt 的 PREPROC cpu_buf == H2D cpu_buf、
 *   H2D net_buf == INFER in_buf、INFER out_buf == D2H net_buf、
 *   D2H cpu_buf == POST cpu_buf；不同 slot 的同资源地址应不同
 */
namespace cnstream {
namespace trace {

inline bool Enabled() {
  static const bool on = [] {
    const char* v = getenv("VSTREAM_INFERTRACE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return on;
}

inline std::string TidTag() {
  std::ostringstream os;
  os << std::hex << std::this_thread::get_id();
  return os.str();
}

inline int64_t NowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace trace
}  // namespace cnstream

#define INFERTRACE(stage)                                                                   \
  LOG_IF(INFO, cnstream::trace::Enabled())                                                  \
      << "[INFERTRACE][" << (stage) << "][TID: " << cnstream::trace::TidTag() << "] "

#endif  // MODULES_INFERENCE_INFER_TRACE_HPP_
