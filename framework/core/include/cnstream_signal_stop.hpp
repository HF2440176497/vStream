#ifndef CNSTREAM_SIGNAL_STOP_HPP_
#define CNSTREAM_SIGNAL_STOP_HPP_

namespace cnstream {

class Pipeline;

/**
 * 注册 pipeline 进入“信号优雅停止”列表（建议在 Start() 成功后立即调用）。
 *
 * 首次调用时安装 SIGTERM/SIGINT 处理器并启动内部 watchdog 线程。信号到达后：
 *   1. 处理器只做 async-signal-safe 的 write（写自管道）；
 *   2. watchdog 线程依次调用所有已注册 pipeline 的 Stop()
 *   3. 恢复信号默认处理并重新发出信号，进程以 128+sig 退出
 *
 * 注意：
 * - 停止耗时应小于容器的 stop_grace_period，否则宽限期后仍会被 SIGKILL；
 * - 若调用方（如 Python 宿主）随后自行安装同名信号处理器，将覆盖本机制；
 * - Pipeline 析构时会自动反注册，无需手动 DisableSignalStop。
 *
 * @param pipeline 待注册的 pipeline，不能为空。
 * @return 注册成功返回 true；信号处理器安装失败（机制未启用）返回 false，
 *         调用方按无信号模式继续即可。
 */
bool EnableSignalStop(Pipeline* pipeline);

/**
 * 反注册 pipeline。Pipeline 析构时会自动调用，通常无需手动调用。
 */
void DisableSignalStop(Pipeline* pipeline);

/**
 * SIGTERM/SIGINT 是否已到达且优雅停止流程已启动。
 * 可用于主循环轮询（如 Python 侧 while 循环）。
 */
bool SignalStopRequested();

}  // namespace cnstream

#endif  // CNSTREAM_SIGNAL_STOP_HPP_
