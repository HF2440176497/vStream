
#ifndef MODULES_CONTRIB_IMAGE_SAVER_HPP_
#define MODULES_CONTRIB_IMAGE_SAVER_HPP_

/**
 * @file image_saver.hpp
 * @brief 图像保存模块（用于调试）：将输入帧落盘到指定目录
 *  - 支持可配置：保存目录 / 保存间隔 / 最大保存数量 / 命名前缀与后缀 / 扩展名
 *  - 文件命名格式：<prefix>_<stream_id>_<timestamp_ms>[_<suffix>].<ext>
 *  - 当某一路已保存文件数超过 max_count 时，删除最早的文件（FIFO）
 */

#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <opencv2/opencv.hpp>

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_module.hpp"

namespace cnstream {

class ImageSaver : public Module, public ModuleCreator<ImageSaver> {
 public:
  explicit ImageSaver(const std::string& name) : Module(name) {}
  ~ImageSaver() override = default;

  bool Open(ModuleParamSet param_set) override;
  void Close() override;
  void OnEos(const std::string& stream_id) override;
  int Process(std::shared_ptr<FrameInfo> data) override;
  bool CheckParamSet(const ModuleParamSet& param_set) const override;

 private:
  struct StreamState {
    std::chrono::steady_clock::time_point last_save_time{};
    bool has_saved = false;
    std::deque<std::string> saved_files;
  };

  static std::string NormalizeDir(std::string dir);
  void TrimOldestFiles(StreamState& state);

  bool enable_ = true;
  std::string save_dir_ = "save";   // 保存目录
  int interval_ms_ = 0;             // 保存间隔（ms），0 表示每帧都保存
  int max_count_ = 0;               // 单路最大保存数量，0 表示不限制
  std::string prefix_ = "img";      // 文件名前缀
  std::string suffix_ = "";         // 文件名后缀（可空）
  std::string ext_ = "jpg";         // 文件扩展名（不含 '.'）

  std::mutex mtx_;
  std::map<std::string, StreamState> states_;  // stream_id -> state
};

REGISTER_MODULE(ImageSaver);

}  // namespace cnstream

#endif  // MODULES_CONTRIB_IMAGE_SAVER_HPP_
