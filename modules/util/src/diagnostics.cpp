#include "diagnostics.hpp"

// 本文件位于公共 src/ 目录，所有平台都会编译，保证接口符号始终存在；
// 平台相关的采集实现只在 CUDA 平台生效，其余平台退化为空操作。
#if defined(VSTREAM_USE_CUDA) && defined(__linux__)

#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
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
  std::string vm_data_kb = "?";
  std::string threads = "?";
  std::string mem_available_kb = "?";
  std::string page_tables_kb = "?";
  std::string commit_limit_kb = "?";
  std::string committed_as_kb = "?";
  std::string shmem_kb = "?";
  std::string shm_used_kb = "?";
  std::string overcommit = "?";
  std::string smaps_anon_kb = "?";
  std::string private_dirty_kb = "?";
  std::string largest_kb = "?";
  std::string nvidia_fds = "?";
  std::string nvidia_holders = "?";
  std::string maps_lines = "?";
  std::string max_map_count = "?";
  std::string libcuda;
  // 设备侧（NVML，只读）。容器内 memory.* 是 GPU 全局值，和宿主机一致。
  std::string gpu_total_mib = "?";
  std::string gpu_used_mib = "?";
  std::string gpu_free_mib = "?";
  std::string gpu_reserved_mib = "?";
  std::string bar1_used_mib = "?";
  std::string gpu_clients = "?";
  std::string gpu_clients_in_proc = "?";
  // 地址空间布局 / ASLR：当前首要假设是固定地址预留与继承布局冲突。
  std::string aslr = "?";
  std::string no_randomize = "?";
  std::string low_va_kb = "?";
  std::string low_vmas = "?";
  std::string low_top_range = "?";
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

// 进程已运行秒数：启动/失败时刻的时间对齐用（与宿主采样按秒对照）。
std::string SelfElapsedSeconds() {
  std::string stat;
  if (!ReadFile("/proc/self/stat", 4096, &stat)) {
    return "?";
  }
  const size_t rparen = stat.rfind(')');
  if (rparen == std::string::npos) {
    return "?";
  }
  std::istringstream rest(stat.substr(rparen + 1));
  std::string field;
  if (!(rest >> field)) {  // state
    return "?";
  }
  for (int f = 4; f <= 22; ++f) {  // 22 = starttime (clock ticks since boot)
    if (!(rest >> field)) {
      return "?";
    }
  }
  unsigned long long start_ticks = 0;
  try {
    start_ticks = std::stoull(field);
  } catch (...) {
    return "?";
  }
  std::string uptime_text;
  double uptime = 0.0;
  if (!ReadFile("/proc/uptime", 256, &uptime_text)) {
    return "?";
  }
  try {
    uptime = std::stod(uptime_text);
  } catch (...) {
    return "?";
  }
  const long hz = sysconf(_SC_CLK_TCK);
  if (hz <= 0) {
    return "?";
  }
  const double elapsed = uptime - static_cast<double>(start_ticks) / hz;
  return elapsed >= 0.0 ? std::to_string(static_cast<long long>(elapsed)) : "?";
}

void CollectHeader(std::ostream& os, const char* mode, const char* stage,
                   const char* kind, int code, int attempt, int call_errno) {
  std::time_t now = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::now());
  std::tm tm_now {};
  localtime_r(&now, &tm_now);
  char timestamp[64] = {0};
  std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S%z", &tm_now);
  char hostname[256] = {0};
  gethostname(hostname, sizeof(hostname) - 1);

  os << "===== vStream CUDA " << (mode != nullptr ? mode : "?")
     << " snapshot =====\n";
  os << "time=" << timestamp << "\n";
  os << "host=" << hostname << "\n";
  os << "pid=" << getpid() << " ppid=" << getppid()
     << " tid=" << static_cast<long>(syscall(SYS_gettid))
     << " etimes_s=" << SelfElapsedSeconds() << "\n";
  os << "mode=" << (mode != nullptr ? mode : "?")
     << " stage=" << (stage != nullptr ? stage : "?")
     << " kind=" << (kind != nullptr ? kind : "?")
     << " attempt=" << attempt << "\n";
  os << "code=" << code << " name=" << CudaInitErrorName(code) << "\n";
  // call_errno 是失败调用返回后立即捕获的值，才是可信证据；
  // snapshot_errno 是采集时刻的残留值，仅供参考（正常路径常见 2/115）。
  if (call_errno >= 0) {
    os << "call_errno=" << call_errno << " (" << std::strerror(call_errno) << ")\n";
  } else {
    os << "call_errno=not_captured\n";
  }
  os << "snapshot_errno=" << errno << " (" << std::strerror(errno) << ")\n";

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
                {"Name:", "State:", "Tgid:", "Pid:", "NSpid:", "PPid:", "Threads:",
                 "VmPeak:", "VmSize:", "VmLck:", "VmPin:", "VmHWM:", "VmRSS:",
                 "VmData:", "VmStk:", "VmExe:", "VmLib:", "VmPTE:", "VmSwap:",
                 "Cpus_allowed_list:", "Mems_allowed_list:",
                 "voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:"});
  summary->vm_pin_kb = FieldValue(content, "VmPin:");
  summary->vm_data_kb = FieldValue(content, "VmData:");
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
                 "CommitLimit:", "Committed_AS:", "Shmem:", "SReclaimable:",
                 "SUnreclaim:", "KernelStack:", "PageTables:", "Slab:",
                 "Mlocked:", "VmallocUsed:"});
  summary->mem_available_kb = FieldValue(content, "MemAvailable:");
  summary->page_tables_kb = FieldValue(content, "PageTables:");
  summary->commit_limit_kb = FieldValue(content, "CommitLimit:");
  summary->committed_as_kb = FieldValue(content, "Committed_AS:");
  summary->shmem_kb = FieldValue(content, "Shmem:");
}

void CollectCgroup(std::ostream& os) {
  WriteSection(os, "cgroup memory / pids");
  const char* files[] = {
      "/proc/self/cgroup",
      "/sys/fs/cgroup/memory.max",
      "/sys/fs/cgroup/memory.current",
      "/sys/fs/cgroup/memory.peak",
      "/sys/fs/cgroup/memory.events",
      "/sys/fs/cgroup/memory/memory.limit_in_bytes",
      "/sys/fs/cgroup/memory/memory.usage_in_bytes",
      "/sys/fs/cgroup/memory/memory.failcnt",
      // clone/pthread_create 触顶会返回 EAGAIN，与 cuInit 失败时的 errno 一致
      "/sys/fs/cgroup/pids.max",
      "/sys/fs/cgroup/pids.current",
      "/sys/fs/cgroup/pids.events",
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

// 内核侧分配能力。显存/BAR1 都空闲却 cuInit 失败时，剩下的嫌疑是宿主内核：
// 高阶连续页枯竭、水位/原子分配失败、线程与 pid 计数触顶。这里一次采全判据。
void CollectKernelAlloc(std::ostream& os) {
  WriteSection(os, "kernel alloc / fragmentation");
  const char* sysctls[] = {
      "/proc/sys/vm/min_free_kbytes",
      "/proc/sys/vm/extfrag_threshold",
      "/proc/sys/vm/zone_reclaim_mode",
      "/proc/sys/kernel/threads-max",
      "/proc/sys/kernel/pid_max",
      "/proc/sys/kernel/dmesg_restrict",
  };
  for (const char* path : sysctls) {
    const std::string value = ReadFirstLine(path);
    os << "  " << path << "=" << (value.empty() ? "?" : value) << "\n";
  }

  // buddyinfo：每行是 Node/zone 后跟 order 0..10 的空闲块数。
  // 高 order（>=3）长期为 0 => 连续页枯竭，驱动的 DMA/连续分配会失败。
  std::string buddy;
  if (ReadFile("/proc/buddyinfo", kMaxSmallFile, &buddy)) {
    os << "  /proc/buddyinfo (order 0..10):\n";
    WriteIndented(os, buddy, 16);
  } else {
    os << "  /proc/buddyinfo <unreadable errno=" << errno << ">\n";
  }

  // zoneinfo 只保留水位与空闲页，避免整份文件（几百 KB）进入快照。
  std::string zone;
  if (ReadFile("/proc/zoneinfo", 512 * 1024, &zone)) {
    os << "  /proc/zoneinfo (filtered):\n";
    std::istringstream in(zone);
    std::string line;
    while (std::getline(in, line)) {
      const std::string trimmed = Trim(line);
      if (trimmed.rfind("Node ", 0) == 0) {
        os << "    " << trimmed << "\n";
      } else if (trimmed.rfind("pages free", 0) == 0 ||
                 trimmed.rfind("min ", 0) == 0 ||
                 trimmed.rfind("low ", 0) == 0 ||
                 trimmed.rfind("high ", 0) == 0 ||
                 trimmed.rfind("managed ", 0) == 0 ||
                 trimmed.rfind("nr_free_pages", 0) == 0) {
        os << "      " << trimmed << "\n";
      }
    }
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

void CollectLargestMappings(std::ostream& os, Summary* summary) {
  WriteSection(os, "largest mappings");
  std::ifstream in("/proc/self/maps");
  if (!in) {
    os << "  /proc/self/maps <unreadable errno=" << errno << ">\n";
    return;
  }

  struct Vma {
    unsigned long long size_kb = 0;
    std::string range;
    std::string perms;
    std::string path;
  };
  constexpr size_t kTopN = 15;
  std::vector<Vma> top;
  unsigned long long total_kb = 0;
  unsigned long long nvidia_kb = 0;
  unsigned long long nvidia0_kb = 0;   // /dev/nvidia0（compute client 的映射）
  unsigned long long ctl_kb = 0;       // /dev/nvidiactl
  unsigned long long uvm_kb = 0;       // /dev/nvidia-uvm（UVM per-process VA）
  unsigned long long modeset_kb = 0;   // /dev/nvidia-modeset
  unsigned long long anon_kb = 0;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string range;
    std::string perms;
    std::string offset;
    std::string dev;
    std::string inode;
    std::string path;
    fields >> range >> perms >> offset >> dev >> inode;
    std::getline(fields, path);
    path = Trim(path);

    const size_t dash = range.find('-');
    if (dash == std::string::npos) {
      continue;
    }
    unsigned long long start = 0;
    unsigned long long end = 0;
    try {
      start = std::stoull(range.substr(0, dash), nullptr, 16);
      end = std::stoull(range.substr(dash + 1), nullptr, 16);
    } catch (...) {
      continue;
    }
    if (end <= start) {
      continue;
    }
    const unsigned long long size_kb = (end - start) / 1024ULL;
    total_kb += size_kb;
    if (path.rfind("/dev/nvidia", 0) == 0) {
      nvidia_kb += size_kb;
      if (path.rfind("/dev/nvidia-uvm", 0) == 0) {
        uvm_kb += size_kb;
      } else if (path.rfind("/dev/nvidiactl", 0) == 0) {
        ctl_kb += size_kb;
      } else if (path.rfind("/dev/nvidia-modeset", 0) == 0) {
        modeset_kb += size_kb;
      } else {
        nvidia0_kb += size_kb;
      }
    } else if (path.empty()) {
      anon_kb += size_kb;
    }

    Vma vma;
    vma.size_kb = size_kb;
    vma.range = range;
    vma.perms = perms;
    vma.path = path;
    top.push_back(vma);
    std::sort(top.begin(), top.end(),
              [](const Vma& a, const Vma& b) { return a.size_kb > b.size_kb; });
    if (top.size() > kTopN) {
      top.pop_back();
    }
  }

  os << "  total_va_kb=" << total_kb << " nvidia_va_kb=" << nvidia_kb
     << " anon_va_kb=" << anon_kb << "\n";
  // 按设备节点拆分：成功建立 context 时 uvm/nvidia0 会各有一块固定 VA；
  // 失败时两者都接近 0，可直接看出卡在 context 建立之前。
  os << "  nvidia0_va_kb=" << nvidia0_kb << " ctl_va_kb=" << ctl_kb
     << " uvm_va_kb=" << uvm_kb << " modeset_va_kb=" << modeset_kb << "\n";
  for (const auto& vma : top) {
    os << "  " << std::setw(10) << vma.size_kb << " kB " << vma.range << " "
       << vma.perms << " " << (vma.path.empty() ? "[anon]" : vma.path) << "\n";
  }
  summary->largest_kb = top.empty() ? "?" : std::to_string(top.front().size_kb);
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

void CollectSmapsRollup(std::ostream& os, Summary* summary) {
  WriteSection(os, "smaps_rollup");
  std::string content;
  if (!ReadFile("/proc/self/smaps_rollup", kMaxSmallFile, &content)) {
    os << "  /proc/self/smaps_rollup <unreadable errno=" << errno << ">\n";
    return;
  }
  WriteFiltered(os, content,
                {"Rss:", "Pss:", "Shared_Clean:", "Shared_Dirty:", "Private_Clean:",
                 "Private_Dirty:", "Anonymous:", "KSM:", "LazyFree:",
                 "AnonHugePages:", "ShmemPmdMapped:", "FilePmdMapped:",
                 "Shared_Hugetlb:", "Private_Hugetlb:", "Swap:", "SwapPss:",
                 "Locked:"});
  summary->smaps_anon_kb = FieldValue(content, "Anonymous:");
  summary->private_dirty_kb = FieldValue(content, "Private_Dirty:");
}

void CollectOvercommit(std::ostream& os, Summary* summary) {
  WriteSection(os, "overcommit / vm limits");
  const char* files[] = {
      "/proc/sys/vm/overcommit_memory",
      "/proc/sys/vm/overcommit_ratio",
      "/proc/sys/vm/overcommit_kbytes",
      "/proc/sys/vm/max_map_count",
      "/proc/sys/kernel/pid_max",
      "/proc/sys/fs/file-max",
  };
  for (const char* path : files) {
    const std::string value = ReadFirstLine(path);
    os << "  " << path << "=" << (value.empty() ? "?" : value) << "\n";
  }
  summary->overcommit = ReadFirstLine("/proc/sys/vm/overcommit_memory");
}

void CollectShm(std::ostream& os, Summary* summary) {
  WriteSection(os, "/dev/shm");
  struct statvfs vfs {};
  if (statvfs("/dev/shm", &vfs) == 0) {
    const unsigned long long unit = vfs.f_frsize != 0 ? vfs.f_frsize : 4096ULL;
    const unsigned long long total_kb =
        static_cast<unsigned long long>(vfs.f_blocks) * unit / 1024ULL;
    const unsigned long long free_kb =
        static_cast<unsigned long long>(vfs.f_bfree) * unit / 1024ULL;
    const unsigned long long used_kb = total_kb >= free_kb ? total_kb - free_kb : 0ULL;
    os << "  total_kb=" << total_kb << " used_kb=" << used_kb
       << " free_kb=" << free_kb << "\n";
    summary->shm_used_kb = std::to_string(used_kb);
  } else {
    os << "  statvfs(/dev/shm) failed errno=" << errno << "\n";
  }

  DIR* dir = opendir("/dev/shm");
  if (dir == nullptr) {
    os << "  /dev/shm <unreadable errno=" << errno << ">\n";
    return;
  }
  int entries = 0;
  int frame_entries = 0;
  int shown = 0;
  struct dirent* ent = nullptr;
  while ((ent = readdir(dir)) != nullptr) {
    if (ent->d_name[0] == '.') {
      continue;
    }
    ++entries;
    const std::string name(ent->d_name);
    if (name.find("frame_shm") != std::string::npos ||
        name.find("process_lock") != std::string::npos) {
      ++frame_entries;
    }
    if (shown < 20) {
      os << "  entry=" << name << "\n";
      ++shown;
    }
  }
  closedir(dir);
  os << "  entries=" << entries << " frame_shm_or_lock_entries=" << frame_entries
     << "\n";
}

int NvidiaFdCountForPid(int pid) {
  const std::string dir_path = "/proc/" + std::to_string(pid) + "/fd";
  DIR* dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    return -1;
  }
  int count = 0;
  struct dirent* ent = nullptr;
  while ((ent = readdir(dir)) != nullptr) {
    if (ent->d_name[0] == '.') {
      continue;
    }
    char target[4096] = {0};
    const std::string link = dir_path + "/" + ent->d_name;
    const ssize_t len = readlink(link.c_str(), target, sizeof(target) - 1);
    if (len <= 0) {
      continue;
    }
    target[len] = '\0';
    if (std::string(target).find("/dev/nvidia") != std::string::npos) {
      ++count;
    }
  }
  closedir(dir);
  return count;
}

std::string ReadPidStatusField(int pid, const char* key) {
  std::string content;
  const std::string path = "/proc/" + std::to_string(pid) + "/status";
  if (!ReadFile(path, kMaxSmallFile, &content)) {
    return "?";
  }
  const std::string value = FieldValue(content, key);
  return value.empty() ? "?" : value;
}

std::string ReadPidCmdline(int pid) {
  std::string cmdline;
  const std::string path = "/proc/" + std::to_string(pid) + "/cmdline";
  if (!ReadFile(path, 4096, &cmdline)) {
    return "?";
  }
  std::replace(cmdline.begin(), cmdline.end(), '\0', ' ');
  cmdline = Trim(cmdline);
  if (cmdline.size() > 160) {
    cmdline = cmdline.substr(0, 160) + "...";
  }
  return cmdline.empty() ? "?" : cmdline;
}

void CollectNvidiaHolders(std::ostream& os, Summary* summary) {
  WriteSection(os, "nvidia fd holders (PID namespace)");
  DIR* proc = opendir("/proc");
  if (proc == nullptr) {
    os << "  /proc <unreadable errno=" << errno << ">\n";
    return;
  }
  int holders = 0;
  struct dirent* ent = nullptr;
  while ((ent = readdir(proc)) != nullptr) {
    if (ent->d_name[0] < '0' || ent->d_name[0] > '9') {
      continue;
    }
    const int pid = std::atoi(ent->d_name);
    if (pid <= 0) {
      continue;
    }
    const int nvidia_fds = NvidiaFdCountForPid(pid);
    if (nvidia_fds <= 0) {
      continue;
    }
    ++holders;
    os << "  pid=" << pid
       << " ppid=" << ReadPidStatusField(pid, "PPid:")
       << " state=" << ReadPidStatusField(pid, "State:")
       << " threads=" << ReadPidStatusField(pid, "Threads:")
       << " vm_size=" << ReadPidStatusField(pid, "VmSize:")
       << " vm_data=" << ReadPidStatusField(pid, "VmData:")
       << " vm_pin=" << ReadPidStatusField(pid, "VmPin:")
       << " rss=" << ReadPidStatusField(pid, "VmRSS:")
       << " nvidia_fds=" << nvidia_fds << "\n";
    os << "    cmd=" << ReadPidCmdline(pid) << "\n";
  }
  closedir(proc);
  os << "  holders=" << holders << "\n";
  summary->nvidia_holders = std::to_string(holders);

  os << "  self parent chain:\n";
  int pid = getpid();
  for (int depth = 0; depth < 15 && pid > 0; ++depth) {
    const std::string ppid_text = ReadPidStatusField(pid, "PPid:");
    os << "    pid=" << pid
       << " ppid=" << ppid_text
       << " nvidia_fds=" << NvidiaFdCountForPid(pid)
       << " cmd=" << ReadPidCmdline(pid) << "\n";
    int ppid = 0;
    try {
      ppid = std::stoi(ppid_text);
    } catch (...) {
      break;
    }
    if (ppid <= 0 || ppid == pid) {
      break;
    }
    pid = ppid;
  }
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
  // 不采 /proc/<tid>/stack：多为读取路径自身（proc_pid_stack），且需 root，
  // 对定位 context 创建失败无帮助，去掉 20 行噪声。
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
      // NVIDIA_REQUIRE_CUDA 是镜像里几百个 brand/driver 组合的长串，
      // 对定位无帮助，截断避免每个快照都占几 KB。
      std::string shown = entry;
      if (shown.find("NVIDIA_REQUIRE_CUDA=") == 0 && shown.size() > 160) {
        shown = shown.substr(0, 160) + "...";
      }
      os << "  " << shown << "\n";
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

// ---------------------------------------------------------------------------
// 设备侧只读采集（NVML）。用 dlopen 动态加载，避免给 util 增加链接依赖；
// 只取 memory / BAR1 / compute 进程三项，都是轻量查询。
// 注意：容器内 memory.* 是 GPU 全局值；compute 进程的 PID 可能是宿主 PID，
// 因此只按"能否在 /proc 解析"标注，不做强弱判断。
// 若担心驱动异常时 NVML 调用卡住，可设 VSTREAM_CUDA_DIAG_NVML=0 关闭。
// ---------------------------------------------------------------------------
using NvmlReturn = int;
using NvmlDevice = void*;

struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

// v2 的 used 不含 reserved；v1 在本驱动上 used 含 reserved（会把 597 报成 used，
// 而 nvidia-smi 显示 used=153 + reserved=446）。优先用 v2。
struct NvmlMemoryV2 {
  unsigned int version;
  unsigned long long total;
  unsigned long long reserved;
  unsigned long long free;
  unsigned long long used;
};
constexpr unsigned int kNvmlMemoryV2Version = 0x02000028;

struct NvmlBar1Memory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct NvmlProcessInfo {
  unsigned int pid;
  unsigned long long used_gpu_memory;
};

constexpr NvmlReturn kNvmlSuccess = 0;
constexpr unsigned long long kNvmlValueNotAvailable = ~0ULL;

struct NvmlApi {
  void* handle = nullptr;
  bool ready = false;
  NvmlReturn (*init)() = nullptr;
  NvmlReturn (*shutdown)() = nullptr;
  NvmlReturn (*get_count)(unsigned int*) = nullptr;
  NvmlReturn (*get_handle)(unsigned int, NvmlDevice*) = nullptr;
  NvmlReturn (*get_name)(NvmlDevice, char*, unsigned int) = nullptr;
  NvmlReturn (*get_memory)(NvmlDevice, NvmlMemory*) = nullptr;
  NvmlReturn (*get_memory_v2)(NvmlDevice, NvmlMemoryV2*) = nullptr;
  NvmlReturn (*get_bar1)(NvmlDevice, NvmlBar1Memory*) = nullptr;
  NvmlReturn (*get_procs)(NvmlDevice, unsigned int*, NvmlProcessInfo*) = nullptr;
};

NvmlApi& Nvml() {
  static NvmlApi api = []() {
    NvmlApi result;
    const char* disable = std::getenv("VSTREAM_CUDA_DIAG_NVML");
    if (disable != nullptr && disable[0] == '0' && disable[1] == '\0') {
      return result;
    }
    result.handle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (result.handle == nullptr) {
      return result;
    }
    auto sym = [&result](const char* name) {
      return dlsym(result.handle, name);
    };
    result.init = reinterpret_cast<NvmlReturn (*)()>(
        sym("nvmlInit_v2") != nullptr ? sym("nvmlInit_v2") : sym("nvmlInit"));
    result.shutdown =
        reinterpret_cast<NvmlReturn (*)()>(sym("nvmlShutdown"));
    result.get_count = reinterpret_cast<NvmlReturn (*)(unsigned int*)>(
        sym("nvmlDeviceGetCount_v2") != nullptr ? sym("nvmlDeviceGetCount_v2")
                                                : sym("nvmlDeviceGetCount"));
    result.get_handle = reinterpret_cast<NvmlReturn (*)(unsigned int, NvmlDevice*)>(
        sym("nvmlDeviceGetHandleByIndex_v2") != nullptr
            ? sym("nvmlDeviceGetHandleByIndex_v2")
            : sym("nvmlDeviceGetHandleByIndex"));
    result.get_name = reinterpret_cast<NvmlReturn (*)(NvmlDevice, char*, unsigned int)>(
        sym("nvmlDeviceGetName"));
    result.get_memory = reinterpret_cast<NvmlReturn (*)(NvmlDevice, NvmlMemory*)>(
        sym("nvmlDeviceGetMemoryInfo"));
    result.get_memory_v2 =
        reinterpret_cast<NvmlReturn (*)(NvmlDevice, NvmlMemoryV2*)>(
            sym("nvmlDeviceGetMemoryInfo_v2"));
    result.get_bar1 = reinterpret_cast<NvmlReturn (*)(NvmlDevice, NvmlBar1Memory*)>(
        sym("nvmlDeviceGetBAR1MemoryInfo"));
    // 只用 v1：结构体只有 {pid, usedGpuMemory}，ABI 最稳定。
    result.get_procs = reinterpret_cast<NvmlReturn (*)(NvmlDevice, unsigned int*,
                                                       NvmlProcessInfo*)>(
        sym("nvmlDeviceGetComputeRunningProcesses"));
    if (result.init == nullptr || result.get_count == nullptr ||
        result.get_handle == nullptr || result.get_memory == nullptr) {
      return result;
    }
    // 符号已就绪；init/shutdown 由每次采集自己成对完成（见 CollectDeviceState）。
    result.ready = true;
    return result;
  }();
  return api;
}

// NVML init 会打开若干个 /dev/nvidia* fd 并映射少量 VA；为了让诊断本身不改变
// 进程的 fd/VA 指纹，采集结束后立即 shutdown（每次快照 init/shutdown 一次）。
struct NvmlShutdownGuard {
  ~NvmlShutdownGuard() {
    NvmlApi& api = Nvml();
    if (api.ready && api.shutdown != nullptr) {
      api.shutdown();
    }
  }
};

void CollectDeviceState(std::ostream& os, Summary* summary) {
  WriteSection(os, "device state (nvml)");
  NvmlApi& api = Nvml();
  if (!api.ready) {
    os << "  nvml unavailable (libnvidia-ml.so.1 load failed or symbols missing, "
          "VSTREAM_CUDA_DIAG_NVML may be 0)\n";
    return;
  }
  if (api.init() != kNvmlSuccess) {
    os << "  nvml init failed\n";
    return;
  }
  NvmlShutdownGuard shutdown_guard;

  unsigned int count = 0;
  if (api.get_count(&count) != kNvmlSuccess || count == 0) {
    os << "  nvml get_device_count failed count=" << count << "\n";
    return;
  }
  // 部署约定所有通道都在 device 0 上。
  const unsigned int index = 0;
  NvmlDevice device = nullptr;
  if (api.get_handle(index, &device) != kNvmlSuccess || device == nullptr) {
    os << "  nvml get_handle(" << index << ") failed\n";
    return;
  }

  char name[96] = {0};
  if (api.get_name != nullptr &&
      api.get_name(device, name, sizeof(name) - 1) == kNvmlSuccess) {
    os << "  name=" << name << "\n";
  }

  const unsigned long long mib = 1024ULL * 1024ULL;
  NvmlMemoryV2 memory_v2 {};
  memory_v2.version = kNvmlMemoryV2Version;
  if (api.get_memory_v2 != nullptr &&
      api.get_memory_v2(device, &memory_v2) == kNvmlSuccess) {
    os << "  memory_total_mib=" << memory_v2.total / mib
       << " used_mib=" << memory_v2.used / mib
       << " free_mib=" << memory_v2.free / mib
       << " reserved_mib=" << memory_v2.reserved / mib << "\n";
    summary->gpu_total_mib = std::to_string(memory_v2.total / mib);
    summary->gpu_used_mib = std::to_string(memory_v2.used / mib);
    summary->gpu_free_mib = std::to_string(memory_v2.free / mib);
    summary->gpu_reserved_mib = std::to_string(memory_v2.reserved / mib);
  } else {
    NvmlMemory memory {};
    if (api.get_memory(device, &memory) == kNvmlSuccess) {
      // v1 的 used 含 reserved，单独标注避免误读。
      os << "  memory_total_mib=" << memory.total / mib
         << " used_incl_reserved_mib=" << memory.used / mib
         << " free_mib=" << memory.free / mib << "\n";
      summary->gpu_total_mib = std::to_string(memory.total / mib);
      summary->gpu_used_mib = std::to_string(memory.used / mib);
      summary->gpu_free_mib = std::to_string(memory.free / mib);
    } else {
      os << "  nvml get_memory failed\n";
    }
  }

  if (api.get_bar1 != nullptr) {
    NvmlBar1Memory bar1 {};
    if (api.get_bar1(device, &bar1) == kNvmlSuccess) {
      const unsigned long long mib = 1024ULL * 1024ULL;
      os << "  bar1_total_mib=" << bar1.total / mib
         << " bar1_used_mib=" << bar1.used / mib
         << " bar1_free_mib=" << bar1.free / mib << "\n";
      summary->bar1_used_mib = std::to_string(bar1.used / mib);
    } else {
      os << "  nvml get_bar1 not supported/available\n";
    }
  }

  if (api.get_procs == nullptr) {
    return;
  }
  // NVML 在 info=nullptr 且存在进程时返回 NVML_ERROR_INSUFFICIENT_SIZE(7)，
  // 同时把 count 填成实际数量；不能当失败处理（否则正常快照永远看不到 client 列表）。
  constexpr NvmlReturn kNvmlInsufficientSize = 7;
  unsigned int proc_count = 0;
  NvmlReturn proc_rc = api.get_procs(device, &proc_count, nullptr);
  if (proc_rc != kNvmlSuccess && proc_rc != kNvmlInsufficientSize) {
    os << "  nvml get_compute_processes(count) failed rc=" << proc_rc << "\n";
    return;
  }
  os << "  compute_processes=" << proc_count << "\n";
  summary->gpu_clients = std::to_string(proc_count);
  if (proc_count == 0) {
    summary->gpu_clients_in_proc = "0";
    return;
  }

  std::vector<NvmlProcessInfo> infos(proc_count);
  unsigned int filled = proc_count;
  proc_rc = api.get_procs(device, &filled, infos.data());
  if (proc_rc == kNvmlInsufficientSize && filled > proc_count) {
    // 查询期间又有新进程：按返回的容量再取一次
    infos.resize(filled);
    proc_rc = api.get_procs(device, &filled, infos.data());
  }
  if (proc_rc != kNvmlSuccess) {
    os << "  nvml get_compute_processes(list) failed rc=" << proc_rc << "\n";
    return;
  }
  int in_proc = 0;
  for (unsigned int i = 0; i < filled; ++i) {
    const NvmlProcessInfo& info = infos[i];
    const bool resolvable =
        access(("/proc/" + std::to_string(info.pid)).c_str(), F_OK) == 0;
    if (resolvable) {
      ++in_proc;
    }
    os << "    pid=" << info.pid
       << " used_mib=" << (info.used_gpu_memory == kNvmlValueNotAvailable
                               ? 0ULL
                               : info.used_gpu_memory / (1024ULL * 1024ULL))
       << " in_my_pid_ns=" << (resolvable ? "yes" : "no");
    if (resolvable) {
      os << " cmd=" << ReadPidCmdline(static_cast<int>(info.pid));
    }
    os << "\n";
  }
  summary->gpu_clients_in_proc = std::to_string(in_proc);
}

// 地址空间布局 / ASLR 指纹。
// 背景：失败代与正常代的 worker 低位预留区间"代内一致、代间不同"，而父链不加载
// CUDA 库、forkserver 是 fork+exec 的单线程新解释器，继承类假设已排除。剩下的
// 首要假设是固定地址预留与布局冲突，因此这里把可判据的布局量一次采全：
//   - randomize_va_space / personality：验证 setarch -R 是否真的生效（跨 fork/exec 继承）
//   - 关键库基址：关 ASLR 后各代应当完全一致
//   - 低地址（<16GiB）VMA 全量清单：固定地址预留的冲突只可能出现在这里，
//     而 largest mappings 只按大小取 top15，会漏掉小的冲突映射。
void CollectLayout(std::ostream& os, Summary* summary) {
  WriteSection(os, "layout / aslr");
  summary->aslr = ReadFirstLine("/proc/sys/kernel/randomize_va_space");
  os << "  randomize_va_space=" << summary->aslr << "\n";
  os << "  mmap_min_addr=" << ReadFirstLine("/proc/sys/vm/mmap_min_addr") << "\n";

  const std::string personality_text = ReadFirstLine("/proc/self/personality");
  unsigned long personality = 0;
  try {
    personality = std::stoul(personality_text, nullptr, 16);
  } catch (...) {
    personality = 0;
  }
  // ADDR_NO_RANDOMIZE = 0x00040000（setarch -R 设置，跨 fork/exec 继承）
  const bool no_randomize = (personality & 0x00040000UL) != 0;
  summary->no_randomize = no_randomize ? "yes" : "no";
  os << "  personality=0x" << std::hex << personality << std::dec
     << " no_randomize=" << summary->no_randomize << "\n";

  constexpr unsigned long long kLowLimit = 0x400000000ULL;  // 16 GiB
  struct LowVma {
    unsigned long long start = 0;
    unsigned long long size_kb = 0;
    std::string range;
    std::string perms;
    std::string path;
  };
  std::vector<LowVma> low;
  unsigned long long low_va_kb = 0;
  std::string libcuda_base = "?";
  std::string libnvinfer_base = "?";
  std::string libc_base = "?";

  std::ifstream in("/proc/self/maps");
  if (!in) {
    os << "  /proc/self/maps <unreadable errno=" << errno << ">\n";
    return;
  }
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string range;
    std::string perms;
    std::string offset;
    std::string dev;
    std::string inode;
    std::string path;
    fields >> range >> perms >> offset >> dev >> inode;
    std::getline(fields, path);
    path = Trim(path);
    const size_t dash = range.find('-');
    if (dash == std::string::npos) {
      continue;
    }
    unsigned long long start = 0;
    unsigned long long end = 0;
    try {
      start = std::stoull(range.substr(0, dash), nullptr, 16);
      end = std::stoull(range.substr(dash + 1), nullptr, 16);
    } catch (...) {
      continue;
    }
    if (end <= start) {
      continue;
    }

    if (path.find("libcuda.so") != std::string::npos && libcuda_base == "?") {
      libcuda_base = range.substr(0, dash);
    } else if (path.find("libnvinfer.so") != std::string::npos &&
               libnvinfer_base == "?") {
      libnvinfer_base = range.substr(0, dash);
    } else if (path.find("/libc.so") != std::string::npos && libc_base == "?") {
      libc_base = range.substr(0, dash);
    }

    if (start < kLowLimit) {
      LowVma vma;
      vma.start = start;
      vma.size_kb = (end - start) / 1024ULL;
      vma.range = range;
      vma.perms = perms;
      vma.path = path;
      low.push_back(vma);
      low_va_kb += vma.size_kb;
    }
  }

  os << "  lib_base libcuda=0x" << libcuda_base << " libnvinfer=0x" << libnvinfer_base
     << " libc=0x" << libc_base << "\n";
  summary->low_va_kb = std::to_string(low_va_kb);
  summary->low_vmas = std::to_string(low.size());
  os << "  low_va_kb=" << low_va_kb << " low_vmas=" << low.size()
     << " (start < 16GiB)\n";

  std::sort(low.begin(), low.end(),
            [](const LowVma& a, const LowVma& b) { return a.size_kb > b.size_kb; });
  if (!low.empty()) {
    summary->low_top_range = low.front().range;
  }
  constexpr size_t kMaxLowPrint = 60;
  for (size_t i = 0; i < low.size() && i < kMaxLowPrint; ++i) {
    os << "    " << std::setw(10) << low[i].size_kb << " kB " << low[i].range
       << " " << low[i].perms << " "
       << (low[i].path.empty() ? "[anon]" : low[i].path) << "\n";
  }
  if (low.size() > kMaxLowPrint) {
    os << "    ... truncated, total=" << low.size() << "\n";
  }
}

size_t CountMapsMatches(int pid, const char* needle) {
  const std::string path = "/proc/" + std::to_string(pid) + "/maps";
  std::ifstream in(path);
  if (!in) {
    return 0;
  }
  size_t count = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find(needle) != std::string::npos) {
      ++count;
    }
  }
  return count;
}

// 父链是否已加载 CUDA 库 / 是否多线程。
// 之前只能靠手工命令回答"动态加载是否与启动方式无关"，这里固化进每份快照：
// 若任一祖先 libnvinfer>0，说明 CUDA 库在 fork 前已加载，继承假设需要重新评估。
void CollectAncestors(std::ostream& os) {
  WriteSection(os, "ancestor chain (cuda libs / threads)");
  int pid = getppid();
  for (int depth = 0; depth < 15 && pid > 1; ++depth) {
    const std::string cmd = ReadPidCmdline(pid);
    const bool is_forkserver = cmd.find("forkserver") != std::string::npos;
    os << "  pid=" << pid
       << " ppid=" << ReadPidStatusField(pid, "PPid:")
       << " threads=" << ReadPidStatusField(pid, "Threads:")
       << " libcuda=" << CountMapsMatches(pid, "libcuda")
       << " libnvinfer=" << CountMapsMatches(pid, "libnvinfer")
       << " nvidia_fds=" << NvidiaFdCountForPid(pid)
       << " forkserver=" << (is_forkserver ? "yes" : "no") << "\n";
    os << "    cmd=" << cmd << "\n";
    int ppid = 0;
    try {
      ppid = std::stoi(ReadPidStatusField(pid, "PPid:"));
    } catch (...) {
      break;
    }
    if (ppid <= 0 || ppid == pid) {
      break;
    }
    pid = ppid;
  }
}

void DumpImpl(const char* mode, const char* stage, const char* kind, int code,
              int attempt, int call_errno) {
  static const bool enabled_all = []() {
    const char* value = std::getenv("VSTREAM_CUDA_DIAG");
    return !(value != nullptr && value[0] == '0' && value[1] == '\0');
  }();
  static const bool enabled_normal = []() {
    const char* value = std::getenv("VSTREAM_CUDA_DIAG_NORMAL");
    return !(value != nullptr && value[0] == '0' && value[1] == '\0');
  }();
  const std::string mode_str =
      (mode != nullptr && mode[0] != '\0') ? mode : "failure";
  if (!enabled_all) {
    return;
  }
  if (mode_str == "normal" && !enabled_normal) {
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
  const std::string key = mode_str + ":" +
                          std::string(kind != nullptr ? kind : "?") + ":" +
                          (stage != nullptr ? stage : "?");
  if (g_dumped.find(key) != g_dumped.end()) {
    return;
  }

  std::string path = dir + "/vstream_cuda_diag_" + mode_str + "_" +
                     std::to_string(getpid()) + ".log";
  Summary summary;
  bool written = false;
  try {
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out && dir != "/tmp") {
      path = "/tmp/vstream_cuda_diag_" + mode_str + "_" +
             std::to_string(getpid()) + ".log";
      out.open(path, std::ios::out | std::ios::app);
    }
    if (!out) {
      LOGE(CUDA_ENV) << "[CUDA_DIAG] cannot open diagnostics file dir=" << dir
                     << " errno=" << errno << " (" << std::strerror(errno) << ")";
      return;
    }

    CollectHeader(out, mode_str.c_str(), stage, kind, code, attempt, call_errno);
    CollectStatus(out, &summary);
    CollectLimits(out);
    CollectSmapsRollup(out, &summary);
    CollectMeminfo(out, &summary);
    CollectOvercommit(out, &summary);
    CollectCgroup(out);
    CollectShm(out, &summary);
    CollectMaps(out, &summary);
    CollectLargestMappings(out, &summary);
    CollectLayout(out, &summary);
    CollectFds(out, &summary);
    CollectNvidiaHolders(out, &summary);
    CollectAncestors(out);
    CollectThread(out);
    CollectEnv(out);
    CollectGpuProc(out);
    CollectDevices(out);
    CollectMountInfo(out);
    CollectCudaToolkit(out);
    CollectKernelAlloc(out);
    // NVML 放在最后：nvmlInit 会打开 /dev/nvidia* 句柄，提前调用会污染
    // 上面 fds / holders 的计数（11-12 现场就出现过 7 → 13 的假增长）。
    CollectDeviceState(out, &summary);

    out << "\n--- notes ---\n"
        << "  normal/failure snapshots are written to separate files;"
        << " compare VmData/smaps_rollup/overcommit/dev_shm/nvidia holders.\n"
        << "  CUcontext/stream handles cannot be enumerated from userspace;"
        << " VmPin, nvidia fd count and maps size are proxies.\n"
        << "  Device state comes from NVML (read-only, dlopen); memory.* is"
        << " GPU-global, compute PIDs may belong to another PID namespace.\n"
        << "  Disable with VSTREAM_CUDA_DIAG_NVML=0 if NVML blocks on a wedged driver.\n"
        << "  Run on host: nvidia-smi;"
        << " nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv;"
        << " nvidia-smi -q -d MEMORY (BAR1);"
        << " nvidia-smi -q -d ECC,ROW_REMAPPER,PAGE_RETIREMENT;"
        << " dmesg -T | grep -iE 'nvrm|xid|gsp|nvidia|uvm'.\n"
        << "  Residual/old GPU clients: ps -eo pid,ppid,etimes,rss,vsz,args;"
        << " fuser -v /dev/nvidia*; check the forkserver parent chain above.\n";
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
  LOGW(CUDA_ENV) << "[CUDA_DIAG] mode=" << mode_str
                 << " stage=" << (stage != nullptr ? stage : "?")
                 << " kind=" << (kind != nullptr ? kind : "?")
                 << " attempt=" << attempt << " code=" << code
                 << " pid=" << getpid()
                 << " vm_data_kb=" << summary.vm_data_kb
                 << " largest_vma_kb=" << summary.largest_kb
                 << " vm_pin_kb=" << summary.vm_pin_kb
                 << " smaps_anon_kb=" << summary.smaps_anon_kb
                 << " private_dirty_kb=" << summary.private_dirty_kb
                 << " threads=" << summary.threads
                 << " nvidia_fds=" << summary.nvidia_fds
                 << " nvidia_holders=" << summary.nvidia_holders
                 << " gpu_used_mib=" << summary.gpu_used_mib
                 << " gpu_free_mib=" << summary.gpu_free_mib
                 << " gpu_reserved_mib=" << summary.gpu_reserved_mib
                 << " bar1_used_mib=" << summary.bar1_used_mib
                 << " gpu_clients=" << summary.gpu_clients
                 << " gpu_clients_in_pid_ns=" << summary.gpu_clients_in_proc
                 << " maps_lines=" << summary.maps_lines << "/"
                 << summary.max_map_count
                 << " mem_available_kb=" << summary.mem_available_kb
                 << " commit_limit_kb=" << summary.commit_limit_kb
                 << " committed_as_kb=" << summary.committed_as_kb
                 << " shmem_kb=" << summary.shmem_kb
                 << " shm_used_kb=" << summary.shm_used_kb
                 << " overcommit=" << summary.overcommit
                 << " aslr=" << summary.aslr
                 << " no_randomize=" << summary.no_randomize
                 << " low_va_kb=" << summary.low_va_kb
                 << " low_vmas=" << summary.low_vmas
                 << " low_top=" << summary.low_top_range
                 << " call_errno=" << call_errno
                 << " page_tables_kb=" << summary.page_tables_kb
                 << " libcuda=" << (summary.libcuda.empty() ? "?" : summary.libcuda)
                 << " file=" << path;
}

}  // namespace

void DumpDiagSnapshotOnError(const char* stage, const char* kind, int code,
                             int attempt, int call_errno) {
  DumpImpl("failure", stage, kind, code, attempt, call_errno);
}

void DumpDiagSnapshotAtStage(const char* stage) {
  DumpImpl("normal", stage, "normal", 0, 0, -1);
}

}  // namespace cnstream

#else

namespace cnstream {

void DumpDiagSnapshotOnError(const char*, const char*, int, int, int) {}

void DumpDiagSnapshotAtStage(const char*) {}

}  // namespace cnstream

#endif  // defined(VSTREAM_USE_CUDA) && defined(__linux__)
