
#ifndef MODULES_INFERENCE_RUN_SHAPE_HPP_
#define MODULES_INFERENCE_RUN_SHAPE_HPP_

/**
 * 运行期形状契约
 */

#include <vector>

#include "tensor.hpp"

namespace cnstream {

/**
 * @brief 单个张量的形状契约
 *
 * - min / max ：由后端在加载期填好，整个生命周期内不变。
 *   Static 模式下二者都等于定形形状；Dynamic 模式下取自 profile 的 kMIN / kMAX。
 * - actual：input 由前处理每批回填，output 由后端推理后回填。
 *
 * @note actual 必须始终被初始化：FillRunShapeCaps 中初始化为 max
 */
struct RunShape {
  TensorShape min;     // 合法下界
  TensorShape max;     // 缓冲区形状 / 可写入上限
  TensorShape actual;  // 本次实际使用
};

/**
 * @brief 单次推理的形状契约
 * 随批在流水线各 stage 之间流动
 * 因为强制 batch_size == 1，这里是 per-batch 单例
 */
struct RunShapes {
  bool dynamic = false;              // 本次流水线是否启用运行期动态形状
  std::vector<RunShape> inputs;      // size == ModelLoader::InputNum()
  std::vector<RunShape> outputs;     // size == ModelLoader::OutputNum()
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_RUN_SHAPE_HPP_
