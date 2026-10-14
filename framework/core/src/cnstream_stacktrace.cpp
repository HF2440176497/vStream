#include "util/cnstream_stacktrace.hpp"

#include <backward.hpp>

#include <cstdio>
#include <ostream>

namespace cnstream {
namespace {

constexpr int kDefaultMaxFrames = 32;

size_t ClampFrames(int max_frames) {
  return static_cast<size_t>(max_frames > 0 ? max_frames : kDefaultMaxFrames);
}

}  // namespace

void DumpStackTrace(int max_frames) {
  backward::StackTrace trace;
  trace.load_here(ClampFrames(max_frames));
  backward::Printer printer;
  printer.address = true;  // 同时打印地址，便于无符号时用 addr2line 兜底
  printer.print(trace, stderr);
}

void DumpStackTrace(std::ostream& os, int max_frames) {
  backward::StackTrace trace;
  trace.load_here(ClampFrames(max_frames));
  backward::Printer printer;
  printer.address = true;
  printer.print(trace, os);
}

}  // namespace cnstream
