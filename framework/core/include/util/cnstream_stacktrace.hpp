//
// 基于 backward-cpp 的调用栈打印封装
//
// 除手动调用 DumpStackTrace 外，libvstream_core 加载时 backward-cpp 已通过
// backward.cpp 中的全局对象自动注册 SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL 等信号处理器
// 程序崩溃时会自动向 stderr 打印堆栈，之后按信号默认行为终止。
//

#ifndef CNSTREAM_STACKTRACE_HPP_
#define CNSTREAM_STACKTRACE_HPP_

#include <iosfwd>

namespace cnstream {

/**
 * 打印当前线程的调用栈到 stderr。
 *
 * 堆栈中的函数名、源文件、行号依赖调试符号（-g）
 *
 * @param max_frames 最大帧数，默认 32；<=0 时按默认值处理。
 */
void DumpStackTrace(int max_frames = 32);

/**
 * 打印当前线程的调用栈到指定流
 *
 * @param os 输出流
 * @param max_frames 最大帧数，默认 32；<=0 时按默认值处理。
 */
void DumpStackTrace(std::ostream& os, int max_frames = 32);

}  // namespace cnstream

#endif  // CNSTREAM_STACKTRACE_HPP_
