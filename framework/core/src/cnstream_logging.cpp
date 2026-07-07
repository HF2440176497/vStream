#include "cnstream_logging.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>

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

}  // anonymous namespace

CustomLogSink::CustomLogSink() {
  InitLogFlags();

  if (VSTREAM_LOG_ROLLING_SIZE_MB > 0) {
    max_size_ = static_cast<size_t>(VSTREAM_LOG_ROLLING_SIZE_MB) * 1024 * 1024;
  }
  if (g_log_to_file) {
    std::string dir = VSTREAM_LOG_FILE_DIR;
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    std::ostringstream fname;
    fname << dir << "/vstream_"
          << getpid() << "_"
          << time(nullptr) << ".out";
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
#ifdef VSTREAM_UNIT_TEST
  FLAGS_v = 1;
  FLAGS_minloglevel = 0;
#else
  FLAGS_minloglevel = 0;
#endif

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

  static CustomLogSink sink;
  google::AddLogSink(&sink);
  for (int severity = google::GLOG_INFO; severity <= google::GLOG_FATAL; ++severity) {
    google::SetLogDestination(static_cast<google::LogSeverity>(severity), "");
  }
}

GlogLevelInitializer g_glog_level_init;

}  // namespace logging
}  // namespace cnstream