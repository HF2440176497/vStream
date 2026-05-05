
#ifndef CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROFILE_HPP_
#define CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_PROFILE_HPP_

#include <cstdint>
#include <string>
#include <vector>
#include <ostream>

namespace cnstream {

struct ProcessProfile {
  std::string process_name;      /*!< The process name. */
  uint64_t    counter = 0;       /*!< The frame counter, it is equal to completed plus dropped frames. */
  uint64_t    completed = 0;     /*!< The completed frame counter. */
  int64_t     dropped = 0;       /*!< The dropped frame counter. */
  double      avg_latency = 0.0; /*!< The average latency. (unit:ms) */
  double      max_latency = 0.0; /*!< The maximum latency. (unit:ms) */
  double      min_latency = 0.0; /*!< The minimum latency. (unit:ms) */
  double      fps = 0.0;         /*!< The throughput. */

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