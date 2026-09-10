
#ifndef MODULES_CONTRIB_ORIENTATION_CHECK_HPP_
#define MODULES_CONTRIB_ORIENTATION_CHECK_HPP_

/**
 * @file orientation_check.hpp
 * @brief 正反判断模块：
 *  - 结论得出前（WARMUP）：标记每帧跳过下游推理模块（skip_module），
 *    帧经框架"虚拟通过"
 *  - 结论得出后（DECIDED）：若结论为背面（需旋转），在帧上标记 kCropRotate180Tag
 *  - 结论 per-stream 保留，同一 stream_id 重连不重新判断；
 */

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "cnstream_frame.hpp"
#include "cnstream_module.hpp"

namespace cnstream {

class OrientationCheck : public Module, public ModuleCreator<OrientationCheck> {
 public:
  explicit OrientationCheck(const std::string& name) : Module(name) {}
  ~OrientationCheck() override = default;

  bool Open(ModuleParamSet param_set) override;
  void Close() override;
  void OnEos(const std::string& stream_id) override;
  int Process(std::shared_ptr<FrameInfo> data) override;
  bool CheckParamSet(const ModuleParamSet& param_set) const override;

 private:
  /// 正反结论：kUnknown 表示尚未判定
  enum class Orientation { kUnknown, kFront, kBack };

  struct StreamState {
    Orientation orientation = Orientation::kUnknown;
    size_t sample_count = 0;  // WARMUP 期间已积累的样本帧数
  };

  /**
   * @brief 正反判断算法。
   *
   * @param[in,out] state 该流的状态（可积累样本）。
   * @param[in] data 当前帧。
   * @param[out] result 判定结果（仅在返回 true 时有效）。
   * @return 返回 true 表示结论已得出。
   */
  bool JudgeOrientation(StreamState* state, const std::shared_ptr<FrameInfo>& data, Orientation* result);

  std::string skip_module_;                  // WARMUP 期间需跳过的下游模块名
  std::atomic<Module*> skip_module_ptr_{nullptr};  // Open 时按名解析，惰性兜底
  size_t warmup_frames_ = 10;                // WARMUP 需积累的帧数（占位算法使用）

  std::mutex mtx_;
  std::map<std::string, StreamState> states_;  // stream_id -> state（结论跨重连保留，不清理）
};

REGISTER_MODULE(OrientationCheck);

}  // namespace cnstream

#endif  // MODULES_CONTRIB_ORIENTATION_CHECK_HPP_
