/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
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

#ifndef CNSTREAM_CONFIG_HPP_
#define CNSTREAM_CONFIG_HPP_

/**
 * @file cnstream_config.hpp
 *
 * This file contains a declaration of the CNModuleConfig class.
 */
#include <list>
#include <memory>
#include <set>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include "cnstream_common.hpp"

namespace cnstream {

/*!
 * Defines an alias for std::map<std::string, std::string>.
 * ModuleParamSet now denotes an unordered map which contains the pairs of parameter name and parameter value.
 */
using ModuleParamSet = std::map<std::string, std::string>;

/**
 * @brief Gets the complete path of a file.
 *
 * If the path you set is an absolute path, returns the absolute path.
 * If the path you set is a relative path, retuns the path that appends the relative path
 * to the specified JSON file path.
 *
 * @param[in] path The path relative to the JSON file or an absolute path.
 * @param[in] param_set The module parameters. The JSON file path is one of the parameters.
 *
 * @return Returns the complete path of a file.
 */
std::string GetPathRelativeToTheJSONFile(const std::string &path, const ModuleParamSet &param_set);

/**
 * @struct CNConfigBase
 *
 * @brief CNConfigBase is a base structure for configurations.
 */
struct CNConfigBase {
  std::string config_root_dir;   ///< The directory where a configuration file is stored.
  /**
   * @brief Parses members from a JSON file.
   *
   * @param[in] jfname JSON configuration file path.
   *
   * @return Returns true if the JSON file has been parsed successfully. Otherwise, returns false.
   */
  bool ParseByJSONFile(const std::string &jfname);

  /**
   * @brief Parses members from JSON string.
   *
   * @param[in] jstr JSON string of a configuration.
   *
   * @return Returns true if the JSON string has been parsed successfully. Otherwise, returns false.
   */
  virtual bool ParseByJSONStr(const std::string &jstr) = 0;

  /**
   * @brief Destructor to destruct config base.
   *
   * @return No return value.
   */
  virtual ~CNConfigBase() {}
};  // struct CNConfigBase

/**
 * @struct ProfilerConfig
 *
 * @brief ProfilerConfig is a structure for profiler configuration.
 *
 * The profiler configuration can be a JSON file.
 *
 * @code {.json}
 * {
 *   "profiler_config" : {
 *     "enable_profile" : true,
 *     "enable_tracing" : true
 *   }
 * }
 * @endcode
 *
 **/
struct ProfilerConfig : public CNConfigBase {
  bool enable_profile = false;           ///< Whether to enable profiling.
  bool ParseByJSONStr(const std::string &jstr) override;
 private:
  std::string key_enable_profile = "enable_profile";  ///< The key of enable_profiling.
};  // struct ProfilerConfig

/**
 * @struct CNModuleConfig
 *
 * CNModuleConfig is a structure for module configuration.
 * The module configuration can be a JSON file.
 *
 * @code {.json}
 * {
 *   "name": {
 *     "parallelism": 3,
 *     "max_input_queue_size": 20,
 *     "class_name": "cnstream::Inference",
 *     "next_modules": ["module_name", ...],
 *     "custom_params" : {
 *       "param_name" : "param_value",
 *       "param_name" : "param_value",
 *       ...
 *     }
 *   }
 * }
 * @endcode
 */
struct CNModuleConfig : public CNConfigBase {
  std::string                        name;        ///< The name of the module.
  std::map<std::string, std::string> parameters;  ///< The key-value pairs. custom parameters of the module.
  int parallelism = 1;  ///< Module parallelism. It is equal to module thread number or the data queue of input data.
  int maxInputQueueSize = 20;       ///< The maximum size of the input data queues.
  std::string           className;  ///< The class name of the module.
  std::set<std::string> next;       ///< The name of the downstream modules.
  bool                  ParseByJSONStr(const std::string &jstr) override;

 private:
  std::string key_name = "name";                                  ///< The key of name.
  std::string key_parallelism = "parallelism";                    ///< The key of parallelism.
  std::string key_max_input_queue_size = "max_input_queue_size";  ///< The key of max_input_queue_size.
  std::string key_class_name = "class_name";                      ///< The key of class_name.
  std::string key_next_modules = "next_modules";                  ///< The key of next_modules.
  std::string key_custom_params = "custom_params";                ///< The key of custom_params.
};

/**
 * @struct CNGraphConfig
 *
 * @brief CNGraphConfig is a structure for graph configuration.
 *
 * You can use ``CNGraphConfig`` to initialize a CNGraph instance.
 * The graph configuration can be a JSON file.
 *
 * @code {.json}
 * {
 *   "profiler_config" : {
 *     "enable_profile" : true,
 *     "enable_tracing" : true
 *   },
 *   "module1": {
 *     "parallelism": 3,
 *     "max_input_queue_size": 20,
 *     "class_name": "cnstream::DataSource",
 *     "next_modules": ["module2"],
 *     "custom_params" : {
 *       "param_name" : "param_value",
 *       "param_name" : "param_value",
 *       ...
 *     }
 *   }
 * }
 * @endcode
 */
struct CNGraphConfig : public CNConfigBase {
  std::string name = "";                            ///< Graph name.
  ProfilerConfig profiler_config;                   ///< Configuration of profiler.
  std::vector<CNModuleConfig> module_configs;       ///< Configurations of modules.
  /**
   * @brief Parses members except ``CNGraphConfig::name`` from the JSON file.
   *
   * @param[in] jstr: Json configuration string.
   *
   * @return Returns true if the JSON string has been parsed successfully. Otherwise, returns false.
   */
  bool ParseByJSONStr(const std::string &jstr) override;
};  // struct GraphConfig

/**
 * @class ParamRegister
 *
 * @brief ParamRegister is a class for module parameter registration.
 *
 * Each module registers its own parameters and descriptions.
 * This is used in CNStream Inspect tool to detect parameters of each module.
 *
 */
class ParamRegister {
 private:
  std::vector<std::pair<std::string /*key*/, std::string /*desc*/>> module_params_ {};
  std::string module_desc_;

 public:
  /**
   * @brief Registers a paramter and its description.
   *
   * This is used in CNStream Inspect tool.
   *
   * @param[in] key The parameter name.
   * @param[in] desc The description of the paramter.
   *
   * @return Void.
   */
  void Register(const std::string &key, const std::string &desc) {
    module_params_.push_back(std::make_pair(key, desc));
  }
  /**
   * @brief Gets the registered paramters and the parameter descriptions.
   *
   * This is used in CNStream Inspect tool.
   *
   * @return Returns the registered paramters and the parameter descriptions.
   */
  std::vector<std::pair<std::string, std::string>> GetParams() { return module_params_; }
  /**
   * @brief Checks if the paramter is registered.
   *
   * This is used in CNStream Inspect tool.
   *
   * @param[in] key The parameter name.
   *
   * @return Returns true if the parameter has been registered. Otherwise, returns false.
   */
  bool IsRegisted(const std::string &key) const {
    if (key == CNS_JSON_DIR_PARAM_NAME) {
      return true;
    }
    for (auto &it : module_params_) {
      if (key == it.first) {
        return true;
      }
    }
    return false;
  }
  /**
   * @brief Sets the description of the module.
   *
   * This is used in CNStream Inspect tool.
   *
   * @param[in] desc The description of the module.
   *
   * @return Void.
   */
  void SetModuleDesc(const std::string &desc) { module_desc_ = desc; }
  /**
   * @brief Gets the description of the module.
   *
   * This is used in CNStream Inspect tool.
   *
   * @return Returns the description of the module.
   */
  std::string GetModuleDesc() { return module_desc_; }
};

/**
 * @class ParametersChecker
 *
 * @brief ParameterChecker is a class used to check module parameters.
 */
class ParametersChecker {
 public:
  /**
   * @brief Checks if a path exists.
   *
   * @param[in] path The path relative to JSON file or an absolute path.
   * @param[in] paramSet The module parameters. The JSON file path is one of the parameters.
   *
   * @return Returns true if the path exists. Otherwise, returns false.
   */
  bool CheckPath(const std::string &path, const ModuleParamSet &paramSet) {
    std::string relative_path = GetPathRelativeToTheJSONFile(path, paramSet);
    // From posix <unistd.h>
    if ((access(relative_path.c_str(), R_OK)) == -1) {
      return false;
    }
    return true;
  }
  /**
   * @brief Checks if the parameters are number, and the value is specified in the correct range.
   *
   * @param[in] check_list A list of parameter names.
   * @param[in] paramSet The module parameters.
   * @param[out] err_msg The error message.
   * @param[in] allow_negative If this parameter is set to ``true`` (default), negative numbers are allowed.
   *                           If set to ``false``, the parameter must be non-negative (>= 0).
   *
   * @return Returns true if the parameters are number and the value is in the correct range. Otherwise, returns false.
   */
  bool IsNum(const std::list<std::string> &check_list, const ModuleParamSet &paramSet, std::string &err_msg,
             bool allow_negative = true) {
    for (auto &it : check_list) {
      if (paramSet.find(it) != paramSet.end()) {
        std::stringstream sin(paramSet.find(it)->second);
        double d;
        char c;
        if (!(sin >> d)) {
          err_msg = "[" + it + "] : " + paramSet.find(it)->second + " is not a number.";
          return false;
        }
        if (sin >> c) {
          err_msg = "[" + it + "] : " + paramSet.find(it)->second + " is not a number.";
          return false;
        }
        if (!allow_negative && d < 0) {
          err_msg = "[" + it + "] : " + paramSet.find(it)->second + " must be non-negative (>= 0).";
          return false;
        }
      }
    }
    return true;
  }
};  // class ParametersChecker

}  // namespace cnstream

#endif  // CNSTREAM_CONFIG_HPP_
