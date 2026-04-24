/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#ifndef CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_MODULE_PROFILER_HPP_
#define CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_MODULE_PROFILER_HPP_

#include <string>
#include <map>
#include <mutex>
#include <memory>

#include "cnstream_config.hpp"
#include "profiler/profile.hpp"
#include "profiler/process_profiler.hpp"

namespace cnstream {

/*!
 * @class ModuleProfiler
 *
 * @brief ModuleProfiler is the profiler for a module.
 *
 * @note This class is thread safe. 
 */
class ModuleProfiler: private NonCopyable {

 public:
  /*!
   * @brief Constructs a ModuleProfiler object.
   *
   * @param[in] config The configuration of the profiler.
   * @param[in] module_name The name of the module.
   *
   * @return No return value. 
  */
  explicit ModuleProfiler(const ProfilerConfig& config, const std::string& module_name);
  
  /*!
   * @brief Destructs a ModuleProfiler object.
   *
   * @return No return value.
   */
  ~ModuleProfiler() = default;

  /*!
   * @brief Registers a process to be profiled.
   *
   * @param[in] process_name The name of the process.
   *
   * @return No return value. 
   */
  void RegisterProcess(const std::string& process_name);

  /*!
   * @brief Records the start of a process.
   *
   * @param[in] process_name The name of the process.
   * @param[in] key The unique identifier of a FrameInfo instance.
   *
   * @return No return value.  
   */
  void RecordProcessStart(const std::string& process_name, const RecordKey& key);

  /*!
   * @brief Records the end of a process.
   *
   * @param[in] process_name The name of the process.
   * @param[in] key The unique identifier of a FrameInfo instance.
   *
   * @return No return value. 
   */
  void RecordProcessEnd(const std::string& process_name, const RecordKey& key);

  /*!
   * @brief Records a dropped frame.
   *
   * @param[in] process_name The name of the process.
   * @param[in] key The unique identifier of a FrameInfo instance.
   *
   * @return No return value. 
   */
  void RecordProcessDropped(const std::string& process_name, const RecordKey& key);

  /*!
   * @brief Gets the name of the module.
   *
   * @return The name of the module. 
   */
  std::string GetName() const;

  /*!
   * @brief Gets profiling results of the module during the execution of the program.
   *
   * @return Returns the profiling results. 
   */
  ModuleProfile GetProfile();

  /**
   * @brief 获得指定 process 的 profile
   */
  ProcessProfile GetProcessProfile(const std::string& process_name);

  /*!
   * @brief Clears profiling data of the stream named by ``stream_name``, as the end of the stream is reached.
   *
   * @param[in] stream_name The name of the stream, usually the ``FrameInfo::stream_id``.
   */
  void OnStreamEos(const std::string& stream_name);

 private:
  ProfilerConfig                         config_;            /*!< The configuration of the profiler. */
  std::mutex                             mutex_;             /*!< The mutex for thread safety. */
  std::string                            module_name_;       /*!< The name of the module. */
  std::map<std::string, std::unique_ptr<ProcessProfiler>> process_profilers_; /*!< The map of process profilers. */
};

inline constexpr char kINPUT_PROFILER_NAME[] = "INPUT";
inline constexpr char kPROCESS_PROFILER_NAME[] = "PROCESS";

inline std::string ModuleProfiler::GetName() const {
  return module_name_;
}

}  // namespace cnstream

#endif  // CNSTREAM_FRAMEWORK_CORE_INCLUDE_PROFILER_MODULE_PROFILER_HPP_