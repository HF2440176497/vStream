/*************************************************************************
* Copyright (C) [2019-2022] by Cambricon, Inc. All rights reserved
*
* This source code is licensed under the Apache-2.0 license found in the
* LICENSE file in the root directory of this source tree.
*
* A part of this source code is referenced from glog project.
* https://github.com/google/glog/blob/master/src/logging.cc
*
* Copyright (c) 1999, Google Inc.
*
* This source code is licensed under the BSD 3-Clause license found in the
* LICENSE file in the root directory of this source tree.
*
*************************************************************************/

#ifndef CNSTREAM_LOGGING_HPP_
#define CNSTREAM_LOGGING_HPP_
#include <glog/logging.h>

#define LOGF(tag) LOG(FATAL) << "[" << (#tag) << " FATAL] "
#define LOGE(tag) LOG(ERROR) << "[" << (#tag) << " ERROR] "
#define LOGW(tag) LOG(WARNING) << "[" << (#tag) << " WARN] "
#define LOGI(tag) LOG(INFO) << "[" << (#tag) << " INFO] "
#define LOGD(tag) VLOG(1) << "[" << (#tag) << " DEBUG] "
#define LOGT(tag) VLOG(2) << "[" << (#tag) << " TRACE] "

#define LOGU(tag) LOGD(tag)

#define LOGF_IF(tag, condition) LOG_IF(FATAL, condition) << "[" << (#tag) << " FATAL] "
#define LOGE_IF(tag, condition) LOG_IF(ERROR, condition) << "[" << (#tag) << " ERROR] "
#define LOGW_IF(tag, condition) LOG_IF(WARNING, condition) << "[" << (#tag) << " WARN] "
#define LOGI_IF(tag, condition) LOG_IF(INFO, condition) << "[" << (#tag) << " INFO] "
#define LOGD_IF(tag, condition) VLOG_IF(1, condition) << "[" << (#tag) << " DEBUG] "
#define LOGT_IF(tag, condition) VLOG_IF(2, condition) << "[" << (#tag) << " TRACE] "


#ifndef VSTREAM_LOG_TO_STDERR
// Default OFF so the C++ library does not flood stderr and contaminate
// Python's logging output. Unittest builds force-enable stderr via the
// VSTREAM_UNIT_TEST define (see cnstream_logging.cpp). Power users can
// also enable it at runtime by setting the VSTREAM_LOG_STDERR env var.
#  define VSTREAM_LOG_TO_STDERR  0
#endif
#ifndef VSTREAM_LOG_TO_FILE
#  define VSTREAM_LOG_TO_FILE    1
#endif
#ifndef VSTREAM_LOG_FILE_DIR
#  define VSTREAM_LOG_FILE_DIR   "./log"  // 文件存放目录
#endif
#ifndef VSTREAM_LOG_ROLLING_SIZE_MB
#  define VSTREAM_LOG_ROLLING_SIZE_MB  100  // 按大小滚动，单位MB，0表示不滚动
#endif

#include <fstream>
#include <mutex>
#include <string>

namespace cnstream {
namespace logging {

class CustomLogSink : public google::LogSink {
 public:
  CustomLogSink();
  ~CustomLogSink() override = default;

  void send(google::LogSeverity severity,
            const char* full_filename,
            const char* base_filename,
            int line,
            const struct ::tm* tm_time,
            const char* message,
            size_t message_len) override;

  CustomLogSink(const CustomLogSink&) = delete;
  CustomLogSink& operator=(const CustomLogSink&) = delete;

 private:
  void RollFileIfNeeded();
  void EmitToStderr(const std::string& line);
  void EmitToFile(const std::string& line);

  std::mutex      write_mutex_;
  std::ofstream   file_stream_;
  std::string     file_path_;
  size_t          current_size_ = 0;
  size_t          max_size_ = 0;
};

struct GlogLevelInitializer {
  GlogLevelInitializer();
};
extern GlogLevelInitializer g_glog_level_init;

}  // namespace logging
}  // namespace cnstream

#endif  // CNSTREAM_LOGGING_HPP_