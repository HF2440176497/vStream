/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved
 *
 * This source code is licensed under the Apache-2.0 license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * A part of this source code is referenced from Nebula project.
 * https://github.com/Bwar/Nebula/blob/master/src/actor/DynamicCreator.hpp
 * https://github.com/Bwar/Nebula/blob/master/src/actor/ActorFactory.hpp
 *
 * Copyright (C) Bwar.
 *
 * This source code is licensed under the Apache-2.0 license found in the
 * LICENSE file in the root directory of this source tree.
 *
 *************************************************************************/

#ifndef CNSTREAM_MODULE_PRI_HPP_
#define CNSTREAM_MODULE_PRI_HPP_

/**
 * @file cnstream_module.hpp
 *
 * This file contains a declaration of the Module class and the ModuleFactory class.
 */
#include <cxxabi.h>
#include <unistd.h>

#include <atomic>
#include <bitset>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <typeinfo>
#include <map>
#include <utility>
#include <vector>
#include <mutex>
#include <iostream>

#include "private/cnstream_constants_pri.hpp"


namespace cnstream {

class Module;

// Group:Framework Function
/*!
 * @brief Gets the number of modules that a pipeline is able to hold.
 *
 * @return The maximum modules of a pipeline can own.
 */
uint32_t GetMaxModuleNumber();

// Group:Framework Function
/*!
 * @brief Gets the number of streams that a pipeline can hold, regardless of the limitation of hardware resources.
 *
 * @return Returns the value of `MAX_STREAM_NUM`.
 *
 * @note The factual stream number that a pipeline can process is always subject to hardware resources, no more than
 * `MAX_STREAM_NUM`.
 */
uint32_t GetMaxStreamNumber();

/**
 * @class ModuleFactory
 *
 * @brief Provides functions to create instances with the ``ModuleClassName``and ``moduleName`` parameters.
 *
 * @note  ModuleCreator, ModuleFactory, and ModuleCreatorWorker:
 *        Implements reflection mechanism to create a module instance dynamically with the ``ModuleClassName`` and
 *        ``moduleName`` parameters. See ActorFactory&DynamicCreator in https://github.com/Bwar/Nebula.
 */
class ModuleFactory {
 public:
  /**
   * @brief Creates or gets the instance of the ModuleFactory class.
   *
   * @param None.
   *
   * @return Returns the instance of the ModuleFactory class.
   */
  static ModuleFactory *Instance() {
    if (nullptr == factory_) {
      factory_ = new (std::nothrow) ModuleFactory();
      LOGF_IF(CORE, nullptr == factory_) << "ModuleFactory::Instance() new ModuleFactory failed.";
    }
    return (factory_);
  }
  /**
   * @brief Destructor. A destructor to destruct ModuleFactory.
   *
   * @param None.
   *
   * @return None.
   */
  virtual ~ModuleFactory() {}

  /**
   * @brief Registers the pair of ``ModuleClassName`` and ``CreateFunction`` to module factory.
   *
   * @param[in] strTypeName The module class name.
   * @param[in] pFunc The ``CreateFunction`` of a Module object that has a parameter ``moduleName``.
   *
   * @return Returns true if this function has run successfully.
   */
  bool Regist(const std::string &strTypeName, std::function<Module *(const std::string &)> pFunc) {
    if (nullptr == pFunc) {
      return (false);
    }
#ifdef VSTREAM_UNIT_TEST
    std::cout << "=== FACTORY REGISTRATION ===" << std::endl;
    std::cout << "Registering type: " << strTypeName << std::endl;
    std::cout << "=== REGISTRATION COMPLETE ===" << std::endl;
#endif
    bool ret = map_.insert(std::make_pair(strTypeName, pFunc)).second;
    return ret;
  }

  /**
   * @brief Creates a module instance with ``ModuleClassName`` and ``moduleName``.
   *
   * @param[in] strTypeName The module class name.
   * @param[in] name The module name which is passed to ``CreateFunction`` to identify a module.
   *
   * @return Returns the module instance if this function has run successfully. Otherwise, returns nullptr if failed.
   */
  Module *Create(const std::string &strTypeName, const std::string &name) {
    auto iter = map_.find(strTypeName);
    if (iter == map_.end()) {
      return (nullptr);
    } else {
      return (iter->second(name));
    }
  }
  // 说明 Module 的构造函数只能是 Module(name) 形式的

  /**
   * @brief Gets all registered modules.
   *
   * @param None.
   *
   * @return All registered module class names.
   */
  std::vector<std::string> GetRegisted() {
    std::vector<std::string> registed_modules;
    for (auto &it : map_) {
      registed_modules.push_back(it.first);
    }
    return registed_modules;
  }

  bool IsRegist(const std::string &strTypeName) {
    return (map_.find(strTypeName) != map_.end());
  }

#ifdef VSTREAM_UNIT_TEST
  void PrintRegistedModules() {
    std::vector<std::string> registed_modules = GetRegisted();
    std::cout << "------- registed_modules: ";
    for (auto &it : registed_modules) {
      std::cout << it << "; ";
    }
    std::cout << std::endl;
  }
#endif

 private:
  ModuleFactory() {}
  static ModuleFactory *factory_;
  std::map<std::string, std::function<Module *(const std::string &)>> map_;
};

/**
 * @class ModuleCreatorWorker
 *
 * @brief ModuleCreatorWorker is class as a dynamic-creator helper.
 *
 * @note  ModuleCreator, ModuleFactory, and ModuleCreatorWorker:
 *        Implements reflection mechanism to create a module instance dynamically with the ``ModuleClassName`` and
 *        ``moduleName`` parameters. See ActorFactory&DynamicCreator in https://github.com/Bwar/Nebula.
 */
class ModuleCreatorWorker {
 public:
  /**
   * @brief Creates a module instance with ``ModuleClassName`` and ``moduleName``.
   *
   * @param[in] strTypeName The module class name.
   * @param[in] name The module name.
   *
   * @return Returns the module instance if the module instance is created successfully. Returns nullptr if failed.
   * @see ModuleFactory::Create
   */
  Module *Create(const std::string &strTypeName, const std::string &name) {
    Module *p = ModuleFactory::Instance()->Create(strTypeName, name);
    return (p);
  }
};

/**
 * @class ModuleCreator
 *
 * @brief A concrete ModuleClass needs to inherit ModuleCreator to enable reflection mechanism.
 *        ModuleCreator provides ``CreateFunction``, and registers ``ModuleClassName`` and ``CreateFunction`` to
 *        ModuleFactory().
 *
 * @note  ModuleCreator, ModuleFactory, and ModuleCreatorWorker:
 *        Implements reflection mechanism to create a module instance dynamically with the ``ModuleClassName`` and
 *        ``moduleName`` parameters. See ActorFactory&DynamicCreator in https://github.com/Bwar/Nebula.
 */
template <typename T>
class ModuleCreator {
 public:
  struct Register {
    Register() {
      char *szDemangleName = nullptr;
      std::string strTypeName;
#ifdef __GNUC__
      szDemangleName = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
#else
      // in this format?:     szDemangleName =  typeid(T).name();
      szDemangleName = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
#endif
      if (nullptr != szDemangleName) {
        strTypeName = szDemangleName;
        free(szDemangleName);
      }
      ModuleFactory::Instance()->Regist(strTypeName, CreateObject);
    }
    inline void do_nothing() const {}
  };

  static bool registered_;

  /**
   * @brief Constructor. A constructor to construct module creator.
   *
   * @param None.
   *
   * @return None.
   */
  ModuleCreator() {}

  /**
   * @brief Destructor. A destructor to destruct module creator.
   *
   * @param None.
   *
   * @return None.
   */
  virtual ~ModuleCreator() {}
  /**
   * @brief Creates an instance of template (T) with specified instance name. This is a template function.
   *
   * @param[in] name The name of the instance.
   *
   * @return Returns the instance of template (T).
   */
  static T *CreateObject(const std::string &name) { return new (std::nothrow) T(name); }
};

/**
 * @brief 模板类的静态成员变量初始化，通过 lambada 表达式自动执行，从而初始化 Register
 */
template <typename T>
bool ModuleCreator<T>::registered_ = []() -> bool {
    static Register register_instance;  // C++11 保证了线程安全
    return true;
}();

/**
 * 注册宏
 * ModuleClass 实现的源文件中使用
 */ 
#define REGISTER_MODULE(ModuleClass) \
  template class ModuleCreator<ModuleClass>


/**
 * @brief ModuleId&StreamIdx manager for pipeline. Allocates and deallocates id for Pipeline modules & streams.
 */
class IdxManager {
 public:
  IdxManager() = default;
  IdxManager(const IdxManager&) = delete;
  IdxManager& operator=(const IdxManager&) = delete;
  uint32_t GetStreamIndex(const std::string& stream_id);
  void ReturnStreamIndex(const std::string& stream_id);
  size_t GetModuleIdx();
  void ReturnModuleIdx(size_t id_);

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  std::mutex id_lock;
  std::map<std::string, uint32_t> stream_idx_map;
  std::bitset<MAX_STREAM_NUM> stream_bitset;
  uint64_t module_id_mask_ = 0;
};  // class IdxManager

}  // namespace cnstream

#endif  // CNSTREAM_MODULE_PRI_HPP_
