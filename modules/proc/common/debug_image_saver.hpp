// Copyright (c) 2026 vStream Authors. All Rights Reserved.
//
// DebugImageSaver: VSTREAM_UNIT_TEST 下专用的线程安全调试图像保存器。
//
// 使用方式：
//   cnstream::DebugImageSaver saver_(enable, interval_ms);                       // 默认存到 save/
//   cnstream::DebugImageSaver saver_(enable, interval_ms, "/tmp/debug/save");  // 自定义目录
//   saver_.MaybeSave("post_yolo", img);                          // 仅落盘
//   saver_.MaybeSave("post_yolo", img, /*suffix=*/"", [&](cv::Mat& c) {
//     for (auto& o : objs) cv::rectangle(c, ..., green, 2);
//   });
//
// 行为约定：
//   - enable_=false 时，MaybeSave 直接 no-op，零开销。
//   - interval_ms=0 时不做节流，每次都落盘。
//   - 节流判断基于 steady_clock（不受系统时间跳变影响）。
//   - 文件名时间戳基于 system_clock（用于命名）。
//   - 整个类的状态（mutex / last_save_time）由自身持有，调用方零负担。
//

#ifndef CNSTREAM_MODULES_PROC_COMMON_DEBUG_IMAGE_SAVER_HPP_
#define CNSTREAM_MODULES_PROC_COMMON_DEBUG_IMAGE_SAVER_HPP_

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <string>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace cnstream {

// 调试落盘开关在 custom_preproc_params / custom_postproc_params 中的键名约定。
// 职责划分：调试开关属于 pipeline 级配置
inline constexpr const char* kDebugParamEnableSave = "enable_save";
inline constexpr const char* kDebugParamSaveIntervalMs = "save_interval_ms";

class DebugImageSaver {
 public:
  DebugImageSaver() = default;
  DebugImageSaver(bool enable, int interval_ms, std::string save_dir = "save")
      : enable_(enable), interval_(interval_ms), save_dir_(std::move(save_dir)) {
    NormalizeSaveDir();
  }

  // 重新配置：允许在 Init() 加载配置后更新参数；并重置节流状态
  void Configure(bool enable, int interval_ms, std::string save_dir = "save") {
    std::lock_guard<std::mutex> lock(mutex_);
    enable_ = enable;
    interval_ = std::chrono::milliseconds(interval_ms);
    save_dir_ = std::move(save_dir);
    NormalizeSaveDir();
    has_saved_ = false;
  }

  // 解析调试落盘开关
  //   enable_save: "true"/"1" 开，"false"/"0" 关；未配置时保持当前配置（默认关闭）。
  //   save_interval_ms: 节流间隔，缺省用 default_interval_ms。
  // 非法值视为保持关闭并告警
  void ConfigureFromParams(const std::map<std::string, std::string>& params,
                           int default_interval_ms = 500) {
    auto it = params.find(kDebugParamEnableSave);
    if (it == params.end()) return;
    bool enable = (it->second == "true" || it->second == "True" || it->second == "1");
    bool disable = (it->second == "false" || it->second == "False" || it->second == "0");
    if (!enable && !disable) {
      LOGW(SAVE) << "invalid enable_save value: "
                   << it->second << ", keep disabled";
      return;
    }
    int interval_ms = default_interval_ms;
    auto it_interval = params.find(kDebugParamSaveIntervalMs);
    if (it_interval != params.end()) {
      try {
        interval_ms = std::stoi(it_interval->second);
      } catch (const std::exception&) {
        LOGW(SAVE) << "invalid save_interval_ms value: "
                     << it_interval->second << ", using default " << default_interval_ms;
      }
    }
    Configure(enable, interval_ms);
  }

  // 尝试保存图像到 <save_dir>/<prefix>_<timestamp_ms>[_<suffix>].jpg
  //   - 返回 true：实际落盘了一次
  //   - 返回 false：未启用 / 空图 / 仍在节流窗口内
  // draw 回调如果提供，传入的是 img 的可变副本，可在写入前叠加可视化。
  bool MaybeSave(const std::string& prefix, const cv::Mat& img,
                 const std::string& suffix = "",
                 const std::function<void(cv::Mat&)>& draw = nullptr) {
    if (!enable_ || img.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    if (interval_.count() > 0 && has_saved_ && (now - last_save_time_) < interval_) {
      return false;
    }
    cv::Mat canvas = img.clone();
    if (draw) draw(canvas);
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::string filename = save_dir_ + prefix + "_" + std::to_string(ts);
    if (!suffix.empty()) filename += "_" + suffix;
    filename += ".jpg";
    cv::imwrite(filename, canvas);
    last_save_time_ = now;
    has_saved_ = true;
    return true;
  }

  bool enable() const { return enable_; }
  int interval_ms() const { return static_cast<int>(interval_.count()); }
  const std::string& save_dir() const { return save_dir_; }

 private:
  // 保证目录路径以 '/' 结尾，避免与文件名拼接时漏分隔符
  void NormalizeSaveDir() {
    if (save_dir_.empty()) {
      save_dir_ = "save/";
      return;
    }
    if (save_dir_.back() != '/') save_dir_.push_back('/');
  }

  bool enable_ = false;
  std::chrono::milliseconds interval_{0};
  std::string save_dir_{"save/"};
  std::mutex mutex_;
  std::chrono::steady_clock::time_point last_save_time_;
  bool has_saved_ = false;
};

}  // namespace cnstream

#endif  // CNSTREAM_MODULES_PROC_COMMON_DEBUG_IMAGE_SAVER_HPP_
