
#ifndef MODULES_CONTRIB_FRAME_STITCHER_HPP_
#define MODULES_CONTRIB_FRAME_STITCHER_HPP_

/**
 * @file frame_stitcher.hpp
 * @brief 帧拼接模块：缓存上一帧、裁剪后与当前帧拼接
 * 写入 FrameInfo::collection 的 kModelInputImageTag
 */

#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <opencv2/opencv.hpp>

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_module.hpp"

namespace cnstream {

class FrameStitcher : public Module, public ModuleCreator<FrameStitcher> {
 public:
  explicit FrameStitcher(const std::string& name) : Module(name) {}
  ~FrameStitcher() override = default;

  bool Open(ModuleParamSet param_set) override;
  void Close() override;
  void OnEos(const std::string& stream_id) override;
  int Process(std::shared_ptr<FrameInfo> data) override;
  bool CheckParamSet(const ModuleParamSet& param_set) const override;

 private:
  bool enable_ = true;
  enum class Direction { kHorizontal = 0, kVertical = 1 };
  Direction direction_ = Direction::kHorizontal;
  float crop_ratio_ = 0.5f;  // 上一帧裁剪比例 (0, 1)，例如 0.5 表示取上一帧右半（水平）或下半（垂直）

  std::mutex cache_mtx_;
  std::map<std::string, cv::Mat> frame_cache_;  // stream_id -> frame
};

REGISTER_MODULE(FrameStitcher);

}  // namespace cnstream

#endif  // MODULES_CONTRIB_FRAME_STITCHER_HPP_
