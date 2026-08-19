#include "cnstream_ffmpeg_logging.hpp"

extern "C" {
#include <libavutil/log.h>
}

namespace cnstream {

namespace {

__attribute__((constructor)) void InitFFmpegLogging() {
  av_log_set_level(AV_LOG_WARNING);
}

}  // namespace

void SetFFmpegLogLevel(int level) {
  av_log_set_level(level);
}

}  // namespace cnstream