#include "cnstream_logging.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>

namespace cnstream {
namespace logging {

namespace {
// 格式化为 YYYYMMDD HH:MM:SS.uuuuuu
std::string FormatTime(const struct ::tm* tm_time, int64_t usec) {
  std::ostringstream oss;
  oss << std::setfill('0')
      << std::setw(4) << 1900 + tm_time->tm_year
      << std::setw(2) << 1 + tm_time->tm_mon
      << std::setw(2) << tm_time->tm_mday
      << ' '
      << std::setw(2) << tm_time->tm_hour
      << ':' << std::setw(2) << tm_time->tm_min
      << ':' << std::setw(2) << tm_time->tm_sec
      << '.' << std::setw(6) << usec;
  return oss.str();
}

char SeverityLetter(google::LogSeverity severity) {
  switch (severity) {
    case google::GLOG_INFO:    return 'I';
    case google::GLOG_WARNING: return 'W';
    case google::GLOG_ERROR:   return 'E';
    case google::GLOG_FATAL:   return 'F';
    default:                   return '?';
  }
}

int64_t GetCurrentMicroseconds() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto usec = std::chrono::duration_cast<std::chrono::microseconds>(duration);
  return usec.count() % 1000000;
}

// Runtime flags. Seeded from the compile-time VSTREAM_LOG_TO_STDERR /
// VSTREAM_LOG_TO_FILE macros, then overridden by the VSTREAM_LOG_STDERR env
// var. Unittest builds (VSTREAM_UNIT_TEST defined) always force stderr on so
// gtest output is visible. The library default is stderr OFF, so importing
// the pybind .so does not contaminate Python's logging output.
bool g_log_to_stderr = (VSTREAM_LOG_TO_STDERR != 0);
bool g_log_to_file   = (VSTREAM_LOG_TO_FILE != 0);

bool ParseBoolEnv(const char* value) {
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

void InitLogFlags() {
  if (const char* env = std::getenv("VSTREAM_LOG_STDERR")) {
    g_log_to_stderr = ParseBoolEnv(env);
  }
#ifdef VSTREAM_UNIT_TEST
  g_log_to_stderr = true;
#endif
}

// Map a textual level name to glog's FLAGS_v and FLAGS_minloglevel pair.
// TRACE  -> VLOG(0..2) + INFO
// DEBUG  -> VLOG(0..1) + INFO
// INFO   -> no VLOG    + INFO
// WARN   -> no VLOG    + WARNING
// ERROR  -> no VLOG    + ERROR
// FATAL  -> no VLOG    + FATAL
// Returns true on recognized level.
bool ParseLogLevel(const char* value, int& v, int& min_level) {
  if (value == nullptr || value[0] == '\0') return false;
  std::string s(value);
  for (auto& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  if (s == "TRACE") {
    v = 2; min_level = google::GLOG_INFO; return true;
  } else if (s == "DEBUG") {
    v = 1; min_level = google::GLOG_INFO; return true;
  } else if (s == "INFO") {
    v = 0; min_level = google::GLOG_INFO; return true;
  } else if (s == "WARNING" || s == "WARN") {
    v = 0; min_level = google::GLOG_WARNING; return true;
  } else if (s == "ERROR") {
    v = 0; min_level = google::GLOG_ERROR; return true;
  } else if (s == "FATAL") {
    v = 0; min_level = google::GLOG_FATAL; return true;
  }
  return false;
}

// 判断字符串是否为 YYYY-MM-DD 格式的日期目录名
bool IsDateDir(const std::string& name) {
  if (name.size() != 10) return false;
  for (int i : {0, 1, 2, 3, 5, 6, 8, 9}) {
    if (!std::isdigit(static_cast<unsigned char>(name[i]))) return false;
  }
  return name[4] == '-' && name[7] == '-';
}

// 将 YYYY-MM-DD 目录名转换为 time_t（当天 00:00:00），失败返回 false
bool ParseDateDir(const std::string& name, std::time_t& out) {
  if (!IsDateDir(name)) return false;
  std::tm tm = {};
  std::istringstream iss(name);
  iss >> std::get_time(&tm, "%Y-%m-%d");
  if (iss.fail()) return false;
  out = std::mktime(&tm);
  return out != static_cast<std::time_t>(-1);
}

// 清理过期的日志日期目录：删除早于 retention_days 天的 YYYY-MM-DD 子目录
void CleanupExpiredLogs(const std::string& log_dir, int retention_days) {
  if (retention_days <= 0) return;

  std::error_code ec;
  if (!std::filesystem::exists(log_dir, ec)) return;

  std::time_t now = std::time(nullptr);
  std::time_t cutoff = now - static_cast<std::time_t>(retention_days) * 24 * 60 * 60;

  for (auto it = std::filesystem::directory_iterator(log_dir, ec);
       it != std::filesystem::directory_iterator(); ++it) {
    if (ec) {
      std::cerr << "Failed to open dir " << log_dir << ": " << ec.message() << std::endl;
      break;
    }
    if (!it->is_directory()) continue;

    std::string dirname = it->path().filename().string();
    std::time_t dir_time;
    if (!ParseDateDir(dirname, dir_time)) continue;

    if (dir_time < cutoff) {
      std::filesystem::remove_all(it->path(), ec);
      if (ec) {
        std::cerr << "Failed to remove dir " << it->path() << ": " << ec.message() << std::endl;
        ec.clear();
      }
    }
  }
}

}  // anonymous namespace

CustomLogSink::CustomLogSink() {
  InitLogFlags();

  if (VSTREAM_LOG_ROLLING_SIZE_MB > 0) {
    max_size_ = static_cast<size_t>(VSTREAM_LOG_ROLLING_SIZE_MB) * 1024 * 1024;
  }
  if (g_log_to_file) {
    std::string dir = VSTREAM_LOG_FILE_DIR;

    // 按日期创建子目录: <log_dir>/YYYY-MM-DD/
    std::time_t now = std::time(nullptr);
    std::tm tm_time;
    localtime_r(&now, &tm_time);
    std::ostringstream date_dir;
    date_dir << dir << "/" << std::put_time(&tm_time, "%Y-%m-%d");
    std::string full_dir = date_dir.str();

    if (!std::filesystem::exists(full_dir)) {
      std::filesystem::create_directories(full_dir);
    }

    // 文件名: vstream_<Unix时间戳>_<PID>.out
    std::ostringstream fname;
    fname << full_dir << "/vstream_"
          << now << "_"
          << getpid() << ".out";
    file_path_ = fname.str();
    file_stream_.open(file_path_, std::ios::app);
    if (!file_stream_.is_open()) {
      std::cerr << "Failed to open log file: " << file_path_ << std::endl;
    }
  }
}

void CustomLogSink::send(google::LogSeverity severity,
                         const char* /*full_filename*/,
                         const char* base_filename,
                         int line,
                         const struct ::tm* tm_time,
                         const char* message,
                         size_t message_len) {
  int64_t usec = GetCurrentMicroseconds();

  std::ostringstream tid_stream;
  tid_stream << std::this_thread::get_id();

  std::ostringstream line_stream;
  line_stream << SeverityLetter(severity)
              << FormatTime(tm_time, usec) << ' '
              << tid_stream.str() << ' '
              << '[' << base_filename << ':' << line << "] "
              << std::string(message, message_len)
              << '\n';
  std::string log_line = line_stream.str();

  if (g_log_to_stderr) {
    EmitToStderr(log_line);
  }
  if (g_log_to_file) {
    EmitToFile(log_line);
  }
}

void CustomLogSink::EmitToStderr(const std::string& line) {
  std::lock_guard<std::mutex> lock(write_mutex_);
  std::cerr << line;
  std::cerr.flush();
}

void CustomLogSink::EmitToFile(const std::string& line) {
  std::lock_guard<std::mutex> lock(write_mutex_);
  if (!file_stream_.is_open()) return;

  if (max_size_ > 0 && current_size_ > 0 &&
      current_size_ + line.size() > max_size_) {
    RollFileIfNeeded();
  }

  file_stream_ << line;
  file_stream_.flush();
  current_size_ += line.size();
}

void CustomLogSink::RollFileIfNeeded() {
  file_stream_.close();
  std::string backup = file_path_ + "." + std::to_string(time(nullptr));
  std::filesystem::rename(file_path_, backup);
  file_stream_.open(file_path_, std::ios::trunc);
  current_size_ = 0;
}

GlogLevelInitializer::GlogLevelInitializer() {
  // Defaults: show INFO and DEBUG
  FLAGS_v = 1;
  FLAGS_minloglevel = google::GLOG_INFO;
#ifdef VSTREAM_UNIT_TEST
  FLAGS_v = 1;
#endif

  // VSTREAM_LOG_LEVEL=TRACE|DEBUG|INFO|WARN|ERROR|FATAL overrides the default.
  if (const char* env = std::getenv("VSTREAM_LOG_LEVEL")) {
    int v = 1, min_level = google::GLOG_INFO;
    if (ParseLogLevel(env, v, min_level)) {
      FLAGS_v = v;
      FLAGS_minloglevel = min_level;
    }
  }

  // glog's LogMessage::Flush() writes to stderr whenever
  // IsGoogleLoggingInitialized() is false, regardless of FLAGS_logtostderr.
  // Mark glog as initialized so the flag below is actually honored.
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging("vstream");
  }

  // Suppress glog's built-in stderr / logfile output. Only the custom sink runs.
  FLAGS_logtostderr = false;
  FLAGS_alsologtostderr = false;
  FLAGS_stderrthreshold = google::GLOG_FATAL + 1;

  // 清理过期日志：在创建 sink 之前执行，删除超过保留期的日期目录
  int retention_days = VSTREAM_LOG_RETENTION_DAYS;
  if (const char* env = std::getenv("VSTREAM_LOG_RETENTION_DAYS")) {
    char* end = nullptr;
    long val = std::strtol(env, &end, 10);
    if (end != env && val >= 0) {
      retention_days = static_cast<int>(val);
    }
  }
  if (retention_days > 0) {
    std::string log_dir = VSTREAM_LOG_FILE_DIR;
    // 确保日志根目录存在（锁文件需要在此目录创建）
    if (!std::filesystem::exists(log_dir)) {
      std::filesystem::create_directories(log_dir);
    }
    // 使用文件锁避免多进程同时启动时的并发清理冲突
    std::string lock_path = log_dir + "/.cleanup.lock";
    int fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd >= 0) {
      if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
        CleanupExpiredLogs(log_dir, retention_days);
        flock(fd, LOCK_UN);
      }
      close(fd);
    }
  }

  static CustomLogSink sink;
  google::AddLogSink(&sink);
  for (int severity = google::GLOG_INFO; severity <= google::GLOG_FATAL; ++severity) {
    google::SetLogDestination(static_cast<google::LogSeverity>(severity), "");
  }
}

GlogLevelInitializer g_glog_level_init;

}  // namespace logging
}  // namespace cnstream