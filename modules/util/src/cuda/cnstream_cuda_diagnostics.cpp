#include "cuda/cnstream_cuda_diagnostics.hpp"

#if defined(__linux__)

#include <dirent.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cnstream_logging.hpp"
#include "cuda/cnstream_cuda_init.hpp"

namespace cnstream {
namespace {

constexpr size_t kMaxSmallFile = 64 * 1024;
constexpr size_t kMaxMapsLines = 200000;

struct Summary {
  std::string vm_pin_kb = "?";
  std::string threads = "?";
  std::string mem_available_kb = "?";
  std::string page_tables_kb = "?";
  std::string nvidia_fds = "?";
  std::string maps_lines = "?";
  std::string max_map_count = "?";
  std::string libcuda;
};

std::mutex g_diag_mutex;
std::set<std::string> g_dumped;

std::string Trim(std::string s) {
  const size_t begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

bool ReadFile(const std::string& path, size_t max_bytes, std::string* out) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return false;
  }
  out->clear();
  char buf[4096];
  size_t total = 0;
  while (in && total < max_bytes) {
    const size_t want = std::min(sizeof(buf), max_bytes - total);
    in.read(buf, static_cast<std::streamsize>(want));
    const std::streamsize got = in.gcount();
    if (got <= 0) {
      break;
    }
    out->append(buf, static_cast<size_t>(got));
    total += static_cast<size_t>(got);
  }
  return true;
}

std::string ReadFirstLine(const std::string& path) {
  std::string content;
  if (!ReadFile(path, 4096, &content)) {
    return "?";
  }
  const size_t pos = content.find('\n');
  return Trim(content.substr(0, pos));
}

std::string FieldValue(const std::string& content, const std::string& key) {
  std::istringstream lines(content);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.compare(0, key.size(), key) != 0) {
      continue;
    }
    std::istringstream values(line.substr(key.size()));
    std::string token;
    values >> token;
    return token.empty() ? "?" : token;
  }
  return "?";
}

void WriteIndented(std::ostream& os, const std::string& content, size_t max_lines) {
  std::istringstream lines(content);
  std::string line;
  size_t count = 0;
  while (std::getline(lines, line) && count < max_lines) {
    os << "    " << line << "\n";
    ++count;
  }
  if (count == max_lines) {
    os << "    ...\n";
  }
}

void WriteFiltered(std::ostream& os, const std::string& content,
                   const std::vector<std::string>& keys) {
  std::istringstream lines(content);
  std::string line;
  while (std::getline(lines, line)) {
    for (const auto& key : keys) {
      if (line.find(key) != std::string::npos) {
        os << line << "\n";
        break;
      }
    }
  }
}

void WriteSection(std::ostream& os, const char* title) {
  os << "\n--- " << title << " ---\n";
}

void WriteFileSection(std::ostream& os, const char* title, const std::string& path,
                      size_t max_lines) {
  WriteSection(os, title);
  std::string content;
  if (!ReadFile(path, kMaxSmallFile, &content)) {
    os << "  " << path << " <unreadable errno=" << errno << ">\n";
    return;
  }
  os << "  " << path << ":\n";
  WriteIndented(os, content, max_lines);
}

void CollectHeader(std::ostream& os, const char* stage, const char* kind, int code,
                   int attempt) {
  std::time_t now = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::now());
  std::tm tm_now {};
  localtime_r(&now, &tm_now);
  char timestamp[64] = {0};
  std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S%z", &tm_now);
  char hostname[256] = {0};
  gethostname(hostname, sizeof(hostname) - 1);

  os << "===== vStream CUDA failure snapshot =====\n";
  os << "time=" << timestamp << "\n";
  os << "host=" << hostname << "\n";
  os << "pid=" << getpid() << " ppid=" << getppid()
     << " tid=" << static_cast<long>(syscall(SYS_gettid)) << "\n";
  os << "stage=" << (stage != nullptr ? stage : "?")
     << " kind=" << (kind != nullptr ? kind : "?")
     << " attempt=" << attempt << "\n";
  os << "code=" << code << " name=" << CudaInitErrorName(code) << "\n";
  os << "errno=" << errno << " (" << std::strerror(errno) << ")\n";

  std::string cmdline;
  if (ReadFile("/proc/self/cmdline", 4096, &cmdline)) {
    std::replace(cmdline.begin(), cmdline.end(), '\0', ' ');
    os << "cmdline=" << Trim(cmdline) << "\n";
  }
  os << "\n";
}

void CollectStatus(std::ostream& os, Summary* summary) {
  WriteSection(os, "process status");
  std::string content;
  if (!ReadFile("/proc/self/status", kMaxSmallFile, &content)) {
    os << "  /proc/self/status <unreadable errno=" << errno << ">\n";
    return;
  }
  WriteFiltered(os, content,
                {"Name:", "State:", "Tgid:", "Pid:", "PPid:", "Threads:",
                 "VmPeak:", "VmSize:", "VmLck:", "VmPin:", "VmHWM:", "VmRSS:",
                 "VmData:", "VmStk:", "VmExe:", "VmLib:", "VmPTE:", "VmSwap:",
                 "Cpus_allowed_list:", "Mems_allowed_list:",
                 "voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:"});
  summary->vm_pin_kb = FieldValue(content, "VmPin:");
  summary->threads = FieldValue(content, "Threads:");
}

void CollectLimits(std::ostream& os) {
  WriteFileSection(os, "process limits", "/proc/self/limits", 40);
}

void CollectMeminfo(std::ostream& os, Summary* summary) {
  WriteSection(os, "system memory");
  std::string content;
  if (!ReadFile("/proc/meminfo", kMaxSmallFile, &content)) {
    os << "  /proc/meminfo <unreadable>\n";
    return;
  }
  WriteFiltered(os, content,
                {"MemTotal:", "MemFree:", "MemAvailable:", "SwapTotal:", "SwapFree:",
                 "CommitLimit:", "Committed_AS:", "PageTables:", "Slab:",
                 "SUnreclaim:", "Mlocked:", "VmallocUsed:"});
  summary->mem_available_kb = FieldValue(content, "MemAvailable:");
  summary->page_tables_kb = FieldValue(content, "PageTables:");
}

void CollectCgroup(std::ostream& os) {
  WriteSection(os, "cgroup memory");
  const char* files[] = {
      "/proc/self/cgroup",
      "/sys/fs/cgroup/memory.max",
      "/sys/fs/cgroup/memory.current",
      "/sys/fs/cgroup/memory.peak",
      "/sys/fs/cgroup/memory.events",
      "/sys/fs/cgroup/memory/memory.limit_in_bytes",
      "/sys/fs/cgroup/memory/memory.usage_in_bytes",
      "/sys/fs/cgroup/memory/memory.failcnt",
  };
  for (const char* path : files) {
    std::string content;
    if (!ReadFile(path, kMaxSmallFile, &content)) {
      continue;
    }
    os << "  " << path << ":\n";
    WriteIndented(os, content, 12);
  }
}

void CollectMaps(std::ostream& os, Summary* summary) {
  WriteSection(os, "maps / cuda libraries");
  std::ifstream in("/proc/self/maps");
  if (!in) {
    os << "  /proc/self/maps <unreadable errno=" << errno << ">\n";
    return;
  }

  std::set<std::string> libs;
  size_t lines = 0;
  bool truncated = false;
  bool compat = false;
  bool stubs = false;
  std::string line;
  while (std::getline(in, line)) {
    if (lines >= kMaxMapsLines) {
      truncated = true;
      break;
    }
    ++lines;
    if (line.find("libcuda") == std::string::npos &&
        line.find("libcudart") == std::string::npos &&
        line.find("libnvinfer") == std::string::npos &&
        line.find("libnvonnxparser") == std::string::npos &&
        line.find("libnvcuvid") == std::string::npos &&
        line.find("libnvidia") == std::string::npos &&
        line.find("libavcodec") == std::string::npos &&
        line.find("libavutil") == std::string::npos &&
        line.find("libavformat") == std::string::npos &&
        line.find("libavdevice") == std::string::npos &&
        line.find("libswscale") == std::string::npos) {
      continue;
    }
    std::istringstream fields(line);
    std::string addr;
    std::string perms;
    std::string offset;
    std::string dev;
    std::string inode;
    std::string path;
    fields >> addr >> perms >> offset >> dev >> inode;
    std::getline(fields, path);
    path = Trim(path);
    if (path.empty()) {
      continue;
    }
    libs.insert(path);
    if (path.find("/compat/") != std::string::npos) {
      compat = true;
    }
    if (path.find("/stubs/") != std::string::npos) {
      stubs = true;
    }
  }

  os << "  total_lines=" << lines << " (vm.max_map_count="
     << ReadFirstLine("/proc/sys/vm/max_map_count") << ")\n";
  if (truncated) {
    os << "  truncated_at=" << kMaxMapsLines << "\n";
  }
  for (const auto& lib : libs) {
    os << "  lib=" << lib << "\n";
    if (lib.find("libcuda.so") != std::string::npos && summary->libcuda.empty()) {
      summary->libcuda = lib;
    }
  }
  if (libs.empty()) {
    os << "  lib=?\n";
  }
  if (compat) {
    os << "  WARN libcuda loaded from compat path\n";
  }
  if (stubs) {
    os << "  WARN libcuda loaded from stubs path\n";
  }
  summary->maps_lines = std::to_string(lines);
  summary->max_map_count = ReadFirstLine("/proc/sys/vm/max_map_count");
}

void CollectFds(std::ostream& os, Summary* summary) {
  WriteSection(os, "nvidia device fds");
  DIR* dir = opendir("/proc/self/fd");
  if (dir == nullptr) {
    os << "  /proc/self/fd <unreadable errno=" << errno << ">\n";
    return;
  }

  int total = 0;
  std::map<std::string, int> nvidia;
  struct dirent* ent = nullptr;
  while ((ent = readdir(dir)) != nullptr) {
    if (ent->d_name[0] == '.') {
      continue;
    }
    ++total;
    char target[4096] = {0};
    const std::string link = std::string("/proc/self/fd/") + ent->d_name;
    const ssize_t len = readlink(link.c_str(), target, sizeof(target) - 1);
    if (len <= 0) {
      continue;
    }
    target[len] = '\0';
    const std::string path(target);
    if (path.find("/dev/nvidia") != std::string::npos) {
      nvidia[path]++;
    }
  }
  closedir(dir);

  os << "  total_fds=" << total << "\n";
  int nvidia_total = 0;
  for (const auto& item : nvidia) {
    os << "  " << item.first << " x" << item.second << "\n";
    nvidia_total += item.second;
  }
  if (nvidia_total == 0) {
    os << "  no /dev/nvidia* fd\n";
  }
  summary->nvidia_fds = std::to_string(nvidia_total);
}

void CollectThread(std::ostream& os) {
  WriteSection(os, "failing thread");
  const long tid = static_cast<long>(syscall(SYS_gettid));
  const std::string base = "/proc/self/task/" + std::to_string(tid);
  os << "  tid=" << tid << "\n";

  std::string content;
  if (ReadFile(base + "/status", kMaxSmallFile, &content)) {
    WriteFiltered(os, content,
                  {"State:", "voluntary_ctxt_switches:",
                   "nonvoluntary_ctxt_switches:"});
  }
  if (ReadFile(base + "/wchan", 4096, &content)) {
    os << "  wchan=" << Trim(content) << "\n";
  }
  if (ReadFile(base + "/stack", 8192, &content)) {
    os << "  kernel_stack:\n";
    WriteIndented(os, content, 20);
  }
}

void CollectEnv(std::ostream& os) {
  WriteSection(os, "environment (cuda/nvidia/ld/vstream)");
  std::string content;
  if (!ReadFile("/proc/self/environ", 128 * 1024, &content)) {
    os << "  <unreadable>\n";
    return;
  }
  size_t start = 0;
  while (start < content.size()) {
    size_t end = content.find('\0', start);
    if (end == std::string::npos) {
      end = content.size();
    }
    const std::string entry = content.substr(start, end - start);
    if (!entry.empty() &&
        (entry.find("CUDA_") == 0 || entry.find("NVIDIA_") == 0 ||
         entry.find("LD_") == 0 || entry.find("VSTREAM_") == 0 ||
         entry.find("MPS") != std::string::npos)) {
      os << "  " << entry << "\n";
    }
    start = end + 1;
  }
}

void CollectGpuProc(std::ostream& os) {
  WriteSection(os, "gpu procfs");
  std::string content;
  if (ReadFile("/proc/driver/nvidia/version", kMaxSmallFile, &content)) {
    os << "  /proc/driver/nvidia/version:\n";
    WriteIndented(os, content, 8);
  } else {
    os << "  /proc/driver/nvidia/version <unreadable errno=" << errno << ">\n";
  }

  DIR* dir = opendir("/proc/driver/nvidia/gpus");
  if (dir == nullptr) {
    os << "  /proc/driver/nvidia/gpus <unreadable errno=" << errno << ">\n";
  } else {
    struct dirent* ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
      if (ent->d_name[0] == '.') {
        continue;
      }
      const std::string base =
          std::string("/proc/driver/nvidia/gpus/") + ent->d_name;
      for (const char* name : {"information", "power"}) {
        std::string body;
        if (!ReadFile(base + "/" + name, kMaxSmallFile, &body)) {
          continue;
        }
        os << "  " << base << "/" << name << ":\n";
        WriteIndented(os, body, 20);
      }
    }
    closedir(dir);
  }

  DIR* warnings = opendir("/proc/driver/nvidia/warnings");
  if (warnings != nullptr) {
    struct dirent* ent = nullptr;
    while ((ent = readdir(warnings)) != nullptr) {
      if (ent->d_name[0] == '.') {
        continue;
      }
      const std::string path =
          std::string("/proc/driver/nvidia/warnings/") + ent->d_name;
      std::string body;
      if (!ReadFile(path, 4096, &body)) {
        continue;
      }
      os << "  " << path << ":\n";
      WriteIndented(os, body, 10);
    }
    closedir(warnings);
  }
}

void CollectDevices(std::ostream& os) {
  WriteSection(os, "device nodes");
  const char* nodes[] = {"/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm",
                         "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"};
  for (const char* node : nodes) {
    struct stat st {};
    if (stat(node, &st) == 0) {
      os << "  " << node << " exists mode=" << std::oct << (st.st_mode & 0777)
         << std::dec << " rdev=" << st.st_rdev << "\n";
    } else {
      os << "  " << node << " missing errno=" << errno << " ("
         << std::strerror(errno) << ")\n";
    }
  }
}

void CollectMountInfo(std::ostream& os) {
  WriteSection(os, "mounts (nvidia/cuda)");
  std::ifstream in("/proc/self/mountinfo");
  if (!in) {
    os << "  /proc/self/mountinfo <unreadable>\n";
    return;
  }
  std::string line;
  size_t count = 0;
  while (std::getline(in, line) && count < 50) {
    if (line.find("nvidia") == std::string::npos &&
        line.find("cuda") == std::string::npos) {
      continue;
    }
    os << "  " << line << "\n";
    ++count;
  }
}

void CollectCudaToolkit(std::ostream& os) {
  WriteSection(os, "cuda toolkit");
  const char* json_path = "/usr/local/cuda/version.json";
  std::string content;
  if (ReadFile(json_path, 8192, &content)) {
    std::string version;
    const size_t pos = content.find("\"version\"");
    if (pos != std::string::npos) {
      const size_t colon = content.find(':', pos);
      const size_t quote1 =
          colon != std::string::npos ? content.find('"', colon) : std::string::npos;
      const size_t quote2 = quote1 != std::string::npos
                                ? content.find('"', quote1 + 1)
                                : std::string::npos;
      if (quote1 != std::string::npos && quote2 != std::string::npos) {
        version = content.substr(quote1 + 1, quote2 - quote1 - 1);
      }
    }
    os << "  " << json_path << ": " << (version.empty() ? "?" : version) << "\n";
    return;
  }

  const char* txt_path = "/usr/local/cuda/version.txt";
  if (ReadFile(txt_path, 4096, &content)) {
    os << "  " << txt_path << ":\n";
    WriteIndented(os, content, 4);
    return;
  }
  os << "  cuda toolkit version <unreadable>\n";
}

void DumpImpl(const char* stage, const char* kind, int code, int attempt) {
  static const bool enabled = []() {
    const char* value = std::getenv("VSTREAM_CUDA_DIAG");
    return !(value != nullptr && value[0] == '0' && value[1] == '\0');
  }();
  if (!enabled) {
    return;
  }

  const char* dir_env = std::getenv("VSTREAM_CUDA_DIAG_DIR");
  std::string dir = (dir_env != nullptr && dir_env[0] != '\0') ? dir_env : "/tmp";
  while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) {
    dir.pop_back();
  }
  if (dir.empty()) {
    dir = "/tmp";
  }

  std::lock_guard<std::mutex> lock(g_diag_mutex);
  const std::string key = std::string(kind != nullptr ? kind : "?") + ":" +
                          (stage != nullptr ? stage : "?");
  if (g_dumped.find(key) != g_dumped.end()) {
    return;
  }

  std::string path =
      dir + "/vstream_cuda_diag_" + std::to_string(getpid()) + ".log";
  Summary summary;
  bool written = false;
  try {
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out && dir != "/tmp") {
      path = "/tmp/vstream_cuda_diag_" + std::to_string(getpid()) + ".log";
      out.open(path, std::ios::out | std::ios::app);
    }
    if (!out) {
      LOGE(CUDA_ENV) << "[CUDA_DIAG] cannot open diagnostics file dir=" << dir
                     << " errno=" << errno << " (" << std::strerror(errno) << ")";
      return;
    }

    CollectHeader(out, stage, kind, code, attempt);
    CollectStatus(out, &summary);
    CollectLimits(out);
    CollectMeminfo(out, &summary);
    CollectCgroup(out);
    CollectMaps(out, &summary);
    CollectFds(out, &summary);
    CollectThread(out);
    CollectEnv(out);
    CollectGpuProc(out);
    CollectDevices(out);
    CollectMountInfo(out);
    CollectCudaToolkit(out);

    out << "\n--- notes ---\n"
        << "  CUcontext/stream handles cannot be enumerated from userspace;"
        << " VmPin, nvidia fd count and maps size are proxies.\n"
        << "  GPU memory/process state is intentionally not queried in-process"
        << " (avoid blocking in a failed driver).\n"
        << "  Run on host: nvidia-smi;"
        << " nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv;"
        << " dmesg -T | grep -iE 'nvrm|xid|gsp|nvidia'\n";
    out << "===== end =====\n\n";
    out.flush();
    written = static_cast<bool>(out);
  } catch (const std::exception& e) {
    LOGE(CUDA_ENV) << "[CUDA_DIAG] exception while collecting: " << e.what();
    return;
  } catch (...) {
    LOGE(CUDA_ENV) << "[CUDA_DIAG] unknown exception while collecting";
    return;
  }

  if (!written) {
    LOGE(CUDA_ENV) << "[CUDA_DIAG] write failed path=" << path
                   << " errno=" << errno;
    return;
  }

  g_dumped.insert(key);
  LOGW(CUDA_ENV) << "[CUDA_DIAG] stage=" << (stage != nullptr ? stage : "?")
                 << " kind=" << (kind != nullptr ? kind : "?")
                 << " attempt=" << attempt << " code=" << code
                 << " pid=" << getpid()
                 << " vm_pin_kb=" << summary.vm_pin_kb
                 << " threads=" << summary.threads
                 << " nvidia_fds=" << summary.nvidia_fds
                 << " maps_lines=" << summary.maps_lines << "/"
                 << summary.max_map_count
                 << " mem_available_kb=" << summary.mem_available_kb
                 << " page_tables_kb=" << summary.page_tables_kb
                 << " libcuda=" << (summary.libcuda.empty() ? "?" : summary.libcuda)
                 << " file=" << path;
}

}  // namespace

void DumpCudaFailureDiagnostics(const char* stage, const char* kind, int code,
                                int attempt) {
  DumpImpl(stage, kind, code, attempt);
}

}  // namespace cnstream

#else

namespace cnstream {

void DumpCudaFailureDiagnostics(const char*, const char*, int, int) {}

}  // namespace cnstream

#endif  // defined(__linux__)
