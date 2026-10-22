// Minimal multi-process CUDA init stress/repro.
//
// Each child mirrors the process-level GPU init sequence used by vStream:
//   cudaGetDeviceCount -> cudaSetDevice -> cudaStreamCreate
//   -> cudaMalloc(1KiB)/cudaFree (ProbeCudaDevice)
//   -> av_hwdevice_ctx_create (independent cuCtxCreate by default)
//   -> AVHWFramesContext + av_hwframe_get_buffer (sink-style surfaces)
//   -> optional cudaMalloc (simulated engine/activation memory)
//   -> optional cudaHostAlloc (pinned memory/memlock).
//
// It does not deserialize a TensorRT engine or open NVDEC/NVENC codecs; use
// --alloc-mb/--frames to approximate the footprint measured with nvidia-smi.

#include <cuda_runtime.h>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/log.h>
#include <libavutil/pixfmt.h>
}

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <poll.h>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kMaxProcesses = 64;

struct Options {
  int processes = 5;
  int rounds = 1;
  int stagger_ms = 0;
  int hold_ms = 1000;
  int round_gap_ms = 0;
  int init_timeout_ms = 30000;
  bool kill_after_hold = false;
  int alloc_mb = 0;
  int pinned_mb = 0;
  int width = 1280;
  int height = 720;
  int frames = 2;
  bool use_ffmpeg = true;
  bool primary_ctx = false;
  int device = 0;
  bool verbose = false;
  bool child = false;
};

struct ChildResources {
  cudaStream_t stream = nullptr;
  cudaStream_t sink_stream = nullptr;
  void* device_ptr = nullptr;
  void* host_ptr = nullptr;
  AVBufferRef* hw_device_ctx = nullptr;
  AVBufferRef* hw_frames_ctx = nullptr;
  std::vector<AVFrame*> frames;
};

struct ChildProc {
  pid_t pid = -1;
  int fd = -1;
  int slot = -1;
  bool ready = false;
};

pid_t g_children[kMaxProcesses];
volatile sig_atomic_t g_child_count = 0;

void Log(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::fprintf(stderr, "[%d] ", static_cast<int>(getpid()));
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
  va_end(args);
  std::fflush(stderr);
}

void SleepMs(int ms) {
  if (ms <= 0) {
    return;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool ParseInt(const char* name, const char* text, int min_value, int max_value,
              int* out) {
  if (text == nullptr || text[0] == '\0') {
    Log("invalid --%s: empty value", name);
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || (end != nullptr && *end != '\0') ||
      value < min_value || value > max_value) {
    Log("invalid --%s=%s (expected %d..%d)", name, text, min_value, max_value);
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

bool CheckCuda(const char* stage, cudaError_t err, bool verbose) {
  if (err == cudaSuccess) {
    if (verbose) {
      Log("stage=%s ok", stage);
    }
    return true;
  }
  Log("stage=%s FAIL cuda=%d (%s) (%s)", stage, static_cast<int>(err),
      cudaGetErrorName(err), cudaGetErrorString(err));
  return false;
}

bool CheckFfmpeg(const char* stage, int err, bool verbose) {
  if (err >= 0) {
    if (verbose) {
      Log("stage=%s ok", stage);
    }
    return true;
  }
  char message[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(err, message, sizeof(message));
  Log("stage=%s FAIL ffmpeg=%d (%s)", stage, err, message);
  return false;
}

void CleanupChild(ChildResources* res) {
  for (AVFrame*& frame : res->frames) {
    if (frame != nullptr) {
      av_frame_free(&frame);
    }
  }
  res->frames.clear();
  if (res->hw_frames_ctx != nullptr) {
    av_buffer_unref(&res->hw_frames_ctx);
  }
  if (res->hw_device_ctx != nullptr) {
    av_buffer_unref(&res->hw_device_ctx);
  }
  if (res->sink_stream != nullptr) {
    const cudaError_t err = cudaStreamDestroy(res->sink_stream);
    if (err != cudaSuccess) {
      Log("stage=cleanup.sink_stream_destroy FAIL cuda=%d (%s)",
          static_cast<int>(err), cudaGetErrorName(err));
    }
    res->sink_stream = nullptr;
  }
  if (res->stream != nullptr) {
    const cudaError_t err = cudaStreamDestroy(res->stream);
    if (err != cudaSuccess) {
      Log("stage=cleanup.cudaStreamDestroy FAIL cuda=%d (%s)", static_cast<int>(err),
          cudaGetErrorName(err));
    }
    res->stream = nullptr;
  }
  if (res->device_ptr != nullptr) {
    cudaFree(res->device_ptr);
    res->device_ptr = nullptr;
  }
  if (res->host_ptr != nullptr) {
    cudaFreeHost(res->host_ptr);
    res->host_ptr = nullptr;
  }
}

bool InitChild(const Options& opt, ChildResources* res) {
  if (opt.verbose) {
    Log("child init: device=%d ffmpeg=%d frames=%d size=%dx%d alloc_mb=%d "
        "pinned_mb=%d primary_ctx=%d",
        opt.device, opt.use_ffmpeg ? 1 : 0, opt.frames, opt.width, opt.height,
        opt.alloc_mb, opt.pinned_mb, opt.primary_ctx ? 1 : 0);
  }

  int device_count = 0;
  if (!CheckCuda("cudaGetDeviceCount", cudaGetDeviceCount(&device_count),
                 opt.verbose)) {
    return false;
  }
  if (device_count <= 0 || opt.device < 0 || opt.device >= device_count) {
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    Log("stage=device_check FAIL device=%d visible_count=%d "
        "CUDA_VISIBLE_DEVICES=%s",
        opt.device, device_count, visible != nullptr ? visible : "<unset>");
    return false;
  }

  if (!CheckCuda("cudaSetDevice", cudaSetDevice(opt.device), opt.verbose)) {
    return false;
  }
  if (!CheckCuda("cudaStreamCreate", cudaStreamCreate(&res->stream),
                 opt.verbose)) {
    return false;
  }

  // ProbeCudaDevice: one real 1KiB allocation after the primary context exists.
  void* probe = nullptr;
  if (!CheckCuda("probe.cudaMalloc", cudaMalloc(&probe, 1024), opt.verbose)) {
    return false;
  }
  if (!CheckCuda("probe.cudaFree", cudaFree(probe), opt.verbose)) {
    return false;
  }

  if (opt.use_ffmpeg) {
    AVDictionary* options = nullptr;
    if (opt.primary_ctx &&
        av_dict_set(&options, "primary_ctx", "1", 0) < 0) {
      Log("stage=av_dict_set(primary_ctx) FAIL");
      av_dict_free(&options);
      return false;
    }

    const std::string device_text = std::to_string(opt.device);
    const int hw_ret = av_hwdevice_ctx_create(
        &res->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, device_text.c_str(),
        options, 0);
    av_dict_free(&options);
    if (hw_ret < 0 || res->hw_device_ctx == nullptr) {
      if (res->hw_device_ctx != nullptr) {
        av_buffer_unref(&res->hw_device_ctx);
      }
      CheckFfmpeg("av_hwdevice_ctx_create",
                  hw_ret < 0 ? hw_ret : AVERROR_EXTERNAL, opt.verbose);
      return false;
    }
    if (opt.verbose) {
      Log("stage=av_hwdevice_ctx_create ok primary_ctx=%d",
          opt.primary_ctx ? 1 : 0);
    }

    // Sink-style surfaces: AVHWFramesContext + one buffer per --frames count.
    if (opt.frames > 0) {
      res->hw_frames_ctx = av_hwframe_ctx_alloc(res->hw_device_ctx);
      if (res->hw_frames_ctx == nullptr) {
        Log("stage=av_hwframe_ctx_alloc FAIL");
        return false;
      }
      auto* frames =
          reinterpret_cast<AVHWFramesContext*>(res->hw_frames_ctx->data);
      frames->format = AV_PIX_FMT_CUDA;
      frames->sw_format = AV_PIX_FMT_NV12;
      frames->width = opt.width;
      frames->height = opt.height;
      frames->initial_pool_size = opt.frames;
      if (!CheckFfmpeg("av_hwframe_ctx_init",
                       av_hwframe_ctx_init(res->hw_frames_ctx), opt.verbose)) {
        return false;
      }
      for (int i = 0; i < opt.frames; ++i) {
        AVFrame* frame = av_frame_alloc();
        if (frame == nullptr) {
          Log("stage=av_frame_alloc FAIL index=%d", i);
          return false;
        }
        const int ret = av_hwframe_get_buffer(res->hw_frames_ctx, frame, 0);
        if (!CheckFfmpeg("av_hwframe_get_buffer", ret, opt.verbose)) {
          av_frame_free(&frame);
          return false;
        }
        res->frames.push_back(frame);
      }
    }
  }

  // Sink InitDeviceCtx also creates its own stream after the hwframes setup.
  if (!CheckCuda("sink.cudaStreamCreate", cudaStreamCreate(&res->sink_stream),
                 opt.verbose)) {
    return false;
  }

  if (opt.alloc_mb > 0) {
    const size_t bytes =
        static_cast<size_t>(opt.alloc_mb) * 1024ULL * 1024ULL;
    if (!CheckCuda("cudaMalloc(alloc)",
                   cudaMalloc(&res->device_ptr, bytes), opt.verbose)) {
      return false;
    }
    if (!CheckCuda("cudaMemset(alloc)",
                   cudaMemset(res->device_ptr, 0, bytes), opt.verbose)) {
      return false;
    }
  }

  if (opt.pinned_mb > 0) {
    const size_t bytes =
        static_cast<size_t>(opt.pinned_mb) * 1024ULL * 1024ULL;
    if (!CheckCuda("cudaHostAlloc",
                   cudaHostAlloc(&res->host_ptr, bytes, cudaHostAllocDefault),
                   opt.verbose)) {
      return false;
    }
    std::memset(res->host_ptr, 0, bytes);
  }

  Log("child READY device=%d ffmpeg=%d frames=%d size=%dx%d alloc_mb=%d "
      "pinned_mb=%d primary_ctx=%d",
      opt.device, opt.use_ffmpeg ? 1 : 0, opt.frames, opt.width, opt.height,
      opt.alloc_mb, opt.pinned_mb, opt.primary_ctx ? 1 : 0);
  return true;
}

void WriteByte(int fd, char value) {
  for (;;) {
    const ssize_t written = write(fd, &value, 1);
    if (written == 1) {
      return;
    }
    if (written < 0 && errno == EINTR) {
      continue;
    }
    return;
  }
}

ssize_t ReadByte(int fd, char* value) {
  for (;;) {
    const ssize_t read_bytes = read(fd, value, 1);
    if (read_bytes < 0 && errno == EINTR) {
      continue;
    }
    return read_bytes;
  }
}

bool WaitChild(pid_t pid, int* status, int timeout_ms) {
  const int64_t deadline = NowMs() + timeout_ms;
  for (;;) {
    const pid_t reaped = waitpid(pid, status, WNOHANG);
    if (reaped == pid) {
      return true;
    }
    if (reaped < 0 && errno != EINTR) {
      return false;
    }
    if (NowMs() >= deadline) {
      return false;
    }
    SleepMs(10);
  }
}

void HandleSignal(int sig) {
  for (int i = 0; i < g_child_count; ++i) {
    if (g_children[i] > 0) {
      kill(g_children[i], SIGKILL);
    }
  }
  _exit(128 + sig);
}

void RunChildProcess(const Options& opt, int pipe_fd) {
  std::signal(SIGINT, SIG_DFL);
  std::signal(SIGTERM, SIG_DFL);
  std::signal(SIGPIPE, SIG_IGN);

  ChildResources res;
  const bool ok = InitChild(opt, &res);
  WriteByte(pipe_fd, ok ? '1' : '0');
  close(pipe_fd);

  if (!ok) {
    CleanupChild(&res);
    _exit(1);
  }

  const int wait_ms =
      opt.kill_after_hold ? std::max(opt.hold_ms + 120000, 120000)
                          : opt.hold_ms;
  SleepMs(wait_ms);
  CleanupChild(&res);
  _exit(0);
}

int RunSingle(const Options& opt) {
  ChildResources res;
  if (!InitChild(opt, &res)) {
    CleanupChild(&res);
    return 1;
  }
  Log("single-process mode: holding %d ms", opt.hold_ms);
  SleepMs(opt.hold_ms);
  CleanupChild(&res);
  return 0;
}

/**
 * 父进程调用
 */
bool RunRound(const Options& opt, int round) {
  Log("[parent] round=%d processes=%d stagger_ms=%d hold_ms=%d kill=%d "
      "alloc_mb=%d pinned_mb=%d frames=%d size=%dx%d ffmpeg=%d primary_ctx=%d",
      round, opt.processes, opt.stagger_ms, opt.hold_ms,
      opt.kill_after_hold ? 1 : 0, opt.alloc_mb, opt.pinned_mb, opt.frames,
      opt.width, opt.height, opt.use_ffmpeg ? 1 : 0,
      opt.primary_ctx ? 1 : 0);

  std::vector<ChildProc> children;
  children.reserve(static_cast<size_t>(opt.processes));
  g_child_count = 0;

  for (int i = 0; i < opt.processes; ++i) {
    int fds[2] = {-1, -1};
    if (pipe(fds) != 0) {
      Log("[parent] pipe failed errno=%d (%s), starting %d/%d children", errno,
          std::strerror(errno), i, opt.processes);
      break;
    }
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
      close(fds[0]);
      RunChildProcess(opt, fds[1]);
      _exit(1);
    }
    if (pid < 0) {
      close(fds[0]);
      close(fds[1]);
      Log("[parent] fork failed errno=%d (%s), starting %d/%d children", errno,
          std::strerror(errno), i, opt.processes);
      break;
    }
    close(fds[1]);
    ChildProc child;
    child.pid = pid;
    child.fd = fds[0];
    if (g_child_count < kMaxProcesses) {
      child.slot = static_cast<int>(g_child_count);
      g_children[g_child_count++] = pid;
    }
    children.push_back(child);
    if (opt.stagger_ms > 0 && i + 1 < opt.processes) {
      SleepMs(opt.stagger_ms);
    }
  }

  int failed = opt.processes - static_cast<int>(children.size());
  if (failed > 0) {
    Log("[parent] %d child process(es) could not be started", failed);
  }

  size_t remaining = children.size();
  const int64_t deadline = NowMs() + opt.init_timeout_ms;
  while (remaining > 0 && NowMs() < deadline) {
    bool progressed = false;
    for (auto& child : children) {
      if (child.fd < 0) {
        continue;
      }
      struct pollfd pfd {};
      pfd.fd = child.fd;
      pfd.events = POLLIN;
      const int poll_ret = poll(&pfd, 1, 0);
      if (poll_ret <= 0) {
        continue;
      }

      char value = 0;
      const ssize_t read_bytes = ReadByte(child.fd, &value);
      if (read_bytes == 1 && value == '1') {
        child.ready = true;
        if (opt.verbose) {
          Log("[parent] child pid=%d ready", child.pid);
        }
      } else {
        Log("[parent] child pid=%d init %s", child.pid,
            read_bytes == 1 ? "failed" : "exited before ready");
      }
      close(child.fd);
      child.fd = -1;
      --remaining;
      progressed = true;
    }
    if (remaining > 0 && !progressed) {
      SleepMs(10);
    }
  }

  for (auto& child : children) {
    if (child.fd < 0) {
      continue;
    }
    Log("[parent] child pid=%d init timeout after %d ms", child.pid,
        opt.init_timeout_ms);
    close(child.fd);
    child.fd = -1;
    kill(child.pid, SIGKILL);
  }

  if (opt.kill_after_hold) {
    bool any_ready = false;
    for (const auto& child : children) {
      any_ready = any_ready || child.ready;
    }
    if (any_ready) {
      SleepMs(opt.hold_ms);
      for (const auto& child : children) {
        if (child.ready) {
          kill(child.pid, SIGKILL);
        }
      }
    }
  }

  int success = 0;
  for (auto& child : children) {
    int status = 0;
    const int wait_ms = opt.kill_after_hold
                            ? 30000
                            : std::max(30000, opt.hold_ms + 30000);
    bool reaped = WaitChild(child.pid, &status, wait_ms);
    if (!reaped) {
      Log("[parent] child pid=%d wait timeout, killing", child.pid);
      kill(child.pid, SIGKILL);
      reaped = WaitChild(child.pid, &status, 5000);
    }
    if (child.slot >= 0 && child.slot < kMaxProcesses) {
      g_children[child.slot] = -1;
    }

    bool ok = child.ready && reaped;
    if (ok) {
      if (opt.kill_after_hold) {
        ok = (WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL) ||
             (WIFEXITED(status) && WEXITSTATUS(status) == 0);
      } else {
        ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
      }
    }

    if (ok) {
      ++success;
    } else {
      ++failed;
      if (reaped) {
        Log("[parent] child pid=%d FAIL ready=%d status=0x%x", child.pid,
            child.ready ? 1 : 0, status);
      } else {
        Log("[parent] child pid=%d FAIL (not reaped)", child.pid);
      }
    }
  }

  g_child_count = 0;
  Log("[parent] round=%d success=%d failed=%d", round, success, failed);
  return failed == 0;
}

void PrintUsage(const char* prog) {
  std::printf(
      "Usage: %s [options]\n"
      "\n"
      "Multi-process CUDA init stress/repro. Each child mirrors the process-level\n"
      "vStream GPU init sequence:\n"
      "  cudaGetDeviceCount/setDevice -> cudaStreamCreate -> cudaMalloc(1KiB)\n"
      "  -> FFmpeg CUDA hwdevice (independent cuCtxCreate by default)\n"
      "  -> AVHWFramesContext + av_hwframe_get_buffer\n"
      "  -> optional cudaMalloc (simulated engine memory) / cudaHostAlloc.\n"
      "\n"
      "Options:\n"
      "  --processes N        child processes per round (default 5, max %d)\n"
      "  --rounds N           rounds (default 1)\n"
      "  --stagger-ms N       delay between child starts (default 0)\n"
      "  --hold-ms N          hold resources after init (default 1000)\n"
      "  --round-gap-ms N     delay between rounds (default 0)\n"
      "  --init-timeout-ms N  readiness timeout per round (default 30000)\n"
      "  --kill               SIGKILL ready children after hold (async reclaim)\n"
      "  --alloc-mb N         cudaMalloc per child (simulate engine/activation)\n"
      "  --pinned-mb N        cudaHostAlloc per child (memlock/pinned test)\n"
      "  --width N            hwframe width (default 1280, must be even)\n"
      "  --height N           hwframe height (default 720, must be even)\n"
      "  --frames N           hwframe pool size; 0 disables (default 2; sink 20)\n"
      "  --no-ffmpeg          skip FFmpeg CUDA hwdevice (isolate FFmpeg context)\n"
      "  --primary-ctx        FFmpeg reuses primary context (A/B; FFmpeg >= 4.0)\n"
      "  --device N           CUDA device (default 0)\n"
      "  --verbose            print each stage\n"
      "  --child              run one flow in the current process (debug)\n"
      "  --help               this help\n"
      "\n"
      "Examples:\n"
      "  %s --processes 5 --rounds 1 --frames 20 --alloc-mb 512\n"
      "  %s --processes 5 --rounds 50 --kill --hold-ms 1000 "
      "--round-gap-ms 0 --frames 20 --alloc-mb 512\n"
      "  %s --processes 5 --rounds 1 --stagger-ms 3000 --alloc-mb 512\n",
      prog, kMaxProcesses, prog, prog, prog);
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;

  static const struct option long_options[] = {
      {"processes", required_argument, nullptr, 'p'},
      {"rounds", required_argument, nullptr, 'r'},
      {"stagger-ms", required_argument, nullptr, 's'},
      {"hold-ms", required_argument, nullptr, 'o'},
      {"round-gap-ms", required_argument, nullptr, 'g'},
      {"init-timeout-ms", required_argument, nullptr, 'i'},
      {"kill", no_argument, nullptr, 'k'},
      {"alloc-mb", required_argument, nullptr, 'a'},
      {"pinned-mb", required_argument, nullptr, 'P'},
      {"width", required_argument, nullptr, 'w'},
      {"height", required_argument, nullptr, 'e'},
      {"frames", required_argument, nullptr, 'f'},
      {"no-ffmpeg", no_argument, nullptr, 'n'},
      {"primary-ctx", no_argument, nullptr, 'c'},
      {"device", required_argument, nullptr, 'd'},
      {"verbose", no_argument, nullptr, 'v'},
      {"child", no_argument, nullptr, 'C'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  bool help = false;
  int ch = 0;
  while ((ch = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
    switch (ch) {
      case 'p':
        if (!ParseInt("processes", optarg, 1, kMaxProcesses, &opt.processes)) {
          return 2;
        }
        break;
      case 'r':
        if (!ParseInt("rounds", optarg, 1, 100000, &opt.rounds)) {
          return 2;
        }
        break;
      case 's':
        if (!ParseInt("stagger-ms", optarg, 0, 600000, &opt.stagger_ms)) {
          return 2;
        }
        break;
      case 'o':
        if (!ParseInt("hold-ms", optarg, 0, 3600000, &opt.hold_ms)) {
          return 2;
        }
        break;
      case 'g':
        if (!ParseInt("round-gap-ms", optarg, 0, 600000, &opt.round_gap_ms)) {
          return 2;
        }
        break;
      case 'i':
        if (!ParseInt("init-timeout-ms", optarg, 1000, 600000,
                      &opt.init_timeout_ms)) {
          return 2;
        }
        break;
      case 'a':
        if (!ParseInt("alloc-mb", optarg, 0, 65536, &opt.alloc_mb)) {
          return 2;
        }
        break;
      case 'P':
        if (!ParseInt("pinned-mb", optarg, 0, 65536, &opt.pinned_mb)) {
          return 2;
        }
        break;
      case 'w':
        if (!ParseInt("width", optarg, 2, 16384, &opt.width)) {
          return 2;
        }
        break;
      case 'e':
        if (!ParseInt("height", optarg, 2, 16384, &opt.height)) {
          return 2;
        }
        break;
      case 'f':
        if (!ParseInt("frames", optarg, 0, 128, &opt.frames)) {
          return 2;
        }
        break;
      case 'd':
        if (!ParseInt("device", optarg, 0, 1024, &opt.device)) {
          return 2;
        }
        break;
      case 'k':
        opt.kill_after_hold = true;
        break;
      case 'n':
        opt.use_ffmpeg = false;
        break;
      case 'c':
        opt.primary_ctx = true;
        break;
      case 'v':
        opt.verbose = true;
        break;
      case 'C':
        opt.child = true;
        break;
      case 'h':
        help = true;
        break;
      default:
        PrintUsage(argv[0]);
        return 2;
    }
  }

  if (help) {
    PrintUsage(argv[0]);
    return 0;
  }
  if (opt.width % 2 != 0 || opt.height % 2 != 0) {
    Log("width/height must be even for NV12 (got %dx%d)", opt.width,
        opt.height);
    return 2;
  }

  av_log_set_level(AV_LOG_ERROR);

  if (opt.child) {
    return RunSingle(opt);
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  bool all_ok = true;
  for (int round = 1; round <= opt.rounds; ++round) {
    if (!RunRound(opt, round)) {
      all_ok = false;
    }
    if (round < opt.rounds && opt.round_gap_ms > 0) {
      SleepMs(opt.round_gap_ms);
    }
  }
  return all_ok ? 0 : 1;
}
