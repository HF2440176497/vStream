#include "cnstream_signal_stop.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"

namespace cnstream {

namespace {

std::mutex& RegistryMutex() {
  static std::mutex m;
  return m;
}

// 全局且线程安全的注册表
std::set<Pipeline*>& Registry() {
  static std::set<Pipeline*> s;
  return s;
}

int g_sig_pipe[2] = {-1, -1};
std::atomic<bool> g_stop_requested{false};

// 信号处理器：只做 async-signal-safe 的 write，真正的工作交给 watchdog 线程
void SignalStopHandler(int sig) {
  const int saved_errno = errno;
  const char c = static_cast<char>(sig);
  const ssize_t unused = write(g_sig_pipe[1], &c, 1);
  (void)unused;
  errno = saved_errno;
}

// 优雅停止完成后按信号默认语义退出（docker stop 期望的 143/130 退出码）
void ReraiseSignal(int sig) {
  struct sigaction dfl;
  std::memset(&dfl, 0, sizeof(dfl));
  dfl.sa_handler = SIG_DFL;
  std::sigemptyset(&dfl.sa_mask);
  if (std::sigaction(sig, &dfl, nullptr) == 0) {
    kill(getpid(), sig);
    // 默认处理应当立即终止；兜底等待，覆盖被第三方处理器拦截的极端情况
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  _exit(128 + sig);
}

void WatchdogLoop() {
  while (true) {
    char c = 0;
    const ssize_t n = read(g_sig_pipe[0], &c, 1);
    if (n != 1) {
      if (n < 0 && errno == EINTR) continue;
      break;  // 管道异常关闭，退出看门狗（进程生命周期内不应发生）
    }
    const int sig = static_cast<unsigned char>(c);
    g_stop_requested.store(true);

    // 持锁遍历：与 ~Pipeline 的反注册互斥，保证 Stop() 期间对象存活
    {
      std::lock_guard<std::mutex> lk(RegistryMutex());
      for (Pipeline* p : Registry()) {
        if (p == nullptr) continue;
        LOGI(CORE) << "Signal " << sig << " received, stopping pipeline ["
                   << p->GetName() << "]";
        p->Stop();
      }
    }
    ReraiseSignal(sig);
  }
}

bool InstallSignalStop() {
  if (pipe(g_sig_pipe) != 0) {
    LOGE(CORE) << "signal-stop: pipe() failed: " << std::strerror(errno);
    return false;
  }
  // 避免管道泄漏给子进程
  fcntl(g_sig_pipe[0], F_SETFD, FD_CLOEXEC);
  fcntl(g_sig_pipe[1], F_SETFD, FD_CLOEXEC);

  struct sigaction sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sa_handler = &SignalStopHandler;
  std::sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  if (std::sigaction(SIGTERM, &sa, nullptr) != 0) {
    LOGE(CORE) << "signal-stop: sigaction(SIGTERM) failed: " << std::strerror(errno);
    close(g_sig_pipe[0]);
    close(g_sig_pipe[1]);
    g_sig_pipe[0] = g_sig_pipe[1] = -1;
    return false;
  }
  if (std::sigaction(SIGINT, &sa, nullptr) != 0) {
    LOGE(CORE) << "signal-stop: sigaction(SIGINT) failed: " << std::strerror(errno);
    // 回滚已安装的 SIGTERM，避免出现“有处理器但无看门狗”的半安装状态
    struct sigaction dfl;
    std::memset(&dfl, 0, sizeof(dfl));
    dfl.sa_handler = SIG_DFL;
    std::sigemptyset(&dfl.sa_mask);
    std::sigaction(SIGTERM, &dfl, nullptr);
    close(g_sig_pipe[0]);
    close(g_sig_pipe[1]);
    g_sig_pipe[0] = g_sig_pipe[1] = -1;
    return false;
  }

  std::thread(&WatchdogLoop).detach();
  return true;
}

}  // namespace

bool EnableSignalStop(Pipeline* pipeline) {
  if (pipeline == nullptr) return false;

  static std::once_flag install_once;
  static bool install_ok = false;
  std::call_once(install_once, []() { install_ok = InstallSignalStop(); });
  if (!install_ok) {
    LOGW(CORE) << "signal-stop: handler installation failed, graceful signal stop disabled";
    return false;
  }

  std::lock_guard<std::mutex> lk(RegistryMutex());
  Registry().insert(pipeline);
  return true;
}

/**
 * @brief 析构触发调用，避免信号处理造成的 Pipeline 资源竞态
 */
void DisableSignalStop(Pipeline* pipeline) {
  if (pipeline == nullptr) return;
  std::lock_guard<std::mutex> lk(RegistryMutex());
  Registry().erase(pipeline);
}

bool SignalStopRequested() {
  return g_stop_requested.load();
}

}  // namespace cnstream
