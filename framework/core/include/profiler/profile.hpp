
#ifndef CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROFILE_HPP_
#define CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROFILE_HPP_

#include <cstdint>
#include <string>
#include <vector>
#include <ostream>

namespace cnstream {

struct ProcessProfile {
  std::string process_name;      /*!< The process name. */
  uint64_t    counter = 0;       /*!< 窗口内帧计数，等于 completed + dropped */
  uint64_t    completed = 0;     /*!< 窗口内完成帧数 */
  int64_t     dropped = 0;       /*!< 窗口内丢弃帧数 */
  double      avg_latency = 0.0; /*!< 窗口内平均延迟。(单位:ms) */
  double      max_latency = 0.0; /*!< 窗口内最大延迟。(单位:ms) */
  double      min_latency = 0.0; /*!< 窗口内最小延迟。(单位:ms) */
  double      fps = 0.0;         /*!< 窗口吞吐率：completed / (window_ms/1000) */
  uint64_t    window_ms = 0;     /*!< 本次统计的滑动窗口长度。(单位:ms) */
  uint64_t    inflight = 0;      /*!< 当前尚未结束的记录数，>0 且 completed==0 说明正在阻塞 */
  double      oldest_inflight_ms = 0.0; /*!< 最老未结束记录已等待时长。(单位:ms) */
  uint64_t    purged = 0;        /*!< 生命周期内因超过窗口上限被清理的残留记录数 */

  ProcessProfile() = default;
  ProcessProfile(const ProcessProfile& it) = default;
  ProcessProfile& operator=(const ProcessProfile& it) = default;
};

struct ModuleProfile {
  std::string                 module_name;      /*!< The module name. */
  uint64_t                    counter = 0;      /*!< The frame counter, it is equal to completed plus dropped frames. */
  uint64_t                    completed = 0;    /*!< The completed frame counter. */
  int64_t                     dropped = 0;      /*!< The dropped frame counter. */
  std::vector<ProcessProfile> process_profiles; /*!< The process profiles. */

  ModuleProfile() = default;
  ModuleProfile(const ModuleProfile& it) = default;
  ModuleProfile& operator=(const ModuleProfile& it) = default;
};

std::string ProcessProfileToString(const ProcessProfile& profile, const std::string& indent);
std::string ModuleProfileToString(const ModuleProfile& profile);

std::ostream& operator<<(std::ostream& os, const ProcessProfile& profile);
std::ostream& operator<<(std::ostream& os, const ModuleProfile& profile);

}  // namespace cnstream

#endif  // CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROFILE_HPP_