#ifndef MODULE_OSD_OUTPUT_MODULE_HPP
#define MODULE_OSD_OUTPUT_MODULE_HPP

#include <string>
#include <memory>

#include "cnstream_module.hpp"
#include "data_common.hpp"

namespace cnstream {

/**
 * @brief OutputModule 是输出模块的基类
 * @details 继承自 Module，定义了 GetData 接口，用于从模块中获取处理后的输出数据
 */
class OutputModule : public Module {
 public:
  explicit OutputModule(const std::string& name) : Module(name) {}
  virtual ~OutputModule() = default;

  /**
   * @brief 获取输出数据（引用版本）
   * @param[out] data 输出数据
   * @param[in] wait_ms 等待时间（毫秒），-1 表示阻塞等待，0 表示非阻塞等待，>0 表示等待指定毫秒
   * @return true 成功获取数据，false 获取失败
   */
  virtual bool GetData(s_output_data& data, int wait_ms = 0) = 0;

  /**
   * @brief 获取输出数据（返回值版本）
   * @param[in] wait_ms 等待时间（毫秒），-1 表示阻塞等待，0 表示非阻塞等待，>0 表示等待指定毫秒
   * @return 获取到的输出数据，如果失败则返回默认构造的 s_output_data
   */
  virtual s_output_data GetData(int wait_ms = 0) {
    s_output_data data;
    return GetData(data, wait_ms) ? data : s_output_data();
  };
};

}  // namespace cnstream

#endif  // MODULE_OSD_OUTPUT_MODULE_HPP
