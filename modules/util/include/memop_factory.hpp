#ifndef MEMOP_FACTORY_HPP_
#define MEMOP_FACTORY_HPP_

#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>

#include "cnstream_logging.hpp"
#include "data_source_param.hpp"

namespace cnstream {

class MemOp;

/**
 * @class MemOpFactory
 * @brief 用于创建MemOp实例的工厂类，支持不同硬件平台的MemOp扩展
 * 
 * MemOpFactory采用单例模式，允许不同硬件平台的MemOp实现通过注册机制加入，而不需要修改通用代码。
 */
class MemOpFactory {
 public:
  /**
   * @brief 获取MemOpFactory的单例实例
   * @return 返回MemOpFactory的唯一实例
   */
  static MemOpFactory& Instance();

  /**
   * @brief 注册MemOp创建函数
   * @param src_device_type 源设备类型
   * @param dst_device_type 目标设备类型
   * @param creator 创建MemOp实例的函数
   * @return 注册是否成功
   */
  bool RegisterMemOpCreator(DevType device_type,
                           std::function<std::shared_ptr<MemOp>(int device_id)> creator);

  /**
   * @brief 根据设备类型创建MemOp实例
   * @param device_type 设备类型
   * @param device_id 设备ID，默认值为-1
   * @return 返回创建的MemOp实例，如果不支持该设备类型则返回nullptr
   */
  std::shared_ptr<MemOp> CreateMemOp(DevType device_type, int device_id);

 private:
  MemOpFactory();
  ~MemOpFactory();
  MemOpFactory(const MemOpFactory&) = delete;
  MemOpFactory& operator=(const MemOpFactory&) = delete;

 public:
  void PrintRegisteredCreators() {
    LOGI(MEMOP_FACTORY) << "PrintRegisteredCreators size: " << creators_.size();
    for (const auto& pair : creators_) {
      LOGI(MEMOP_FACTORY) << "DevType: " << DevType2Str(pair.first) << " -> Creator Func Address: " << &pair.second;
    }
  }

 private:
  struct DevTypeHash {
    template <typename T>
    std::size_t operator()(const T& device_type) const {
      return static_cast<std::size_t>(device_type);
    }
  };

  std::unordered_map<DevType, std::function<std::shared_ptr<MemOp>(int device_id)>, DevTypeHash> creators_ {};
  std::mutex mutex_;
};

}  // namespace cnstream

#endif  // MEMOP_FACTORY_HPP_