
#include "frame_stitcher.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

#include "cnstream_logging.hpp"

namespace cnstream {

namespace {

constexpr const char* kParamEnable = "enable";
constexpr const char* kParamDirection = "direction";   // "horizontal" / "vertical"
constexpr const char* kParamCropRatio = "crop_ratio";  // (0, 1)
constexpr const char* kParamTargetModels = "target_models";  // 逗号分隔的模型名列表

constexpr const char* kDirHorizontal = "horizontal";
constexpr const char* kDirVertical = "vertical";

std::string Trim(const std::string& s) {
  size_t begin = 0;
  size_t end = s.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
  return s.substr(begin, end - begin);
}

/**
 * @brief 解析 target_models。
 *
 * 支持两种写法：
 * - 逗号分隔字符串："model_a,model_b"
 * - JSON 数组：["model_a","model_b"]（配置系统 ModuleConfig 会把数组 dump 成字符串）
 *
 * 空项会被忽略，重复项会被去重。
 */
std::vector<std::string> ParseTargetModels(const std::string& raw) {
  std::string s = Trim(raw);
  if (s.size() >= 2 && s.front() == '[' && s.back() == ']') {
    s = s.substr(1, s.size() - 2);
  }

  std::vector<std::string> models;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = Trim(item);
    // JSON 数组元素可能带引号
    if (item.size() >= 2 && (item.front() == '"' || item.front() == '\'') &&
        item.back() == item.front()) {
      item = Trim(item.substr(1, item.size() - 2));
    }
    if (item.empty()) continue;
    if (std::find(models.begin(), models.end(), item) == models.end()) {
      models.push_back(item);
    }
  }
  return models;
}

}  // namespace

/**
 * @param param_set 参数集 custom_params
 */
bool FrameStitcher::CheckParamSet(const ModuleParamSet& param_set) const {
  if (param_set.find(kParamCropRatio) != param_set.end()) {
    try {
      float r = std::stof(param_set.at(kParamCropRatio));
      if (r <= 0.f || r >= 1.f) {
        LOGE(STITCHER) << "crop_ratio must be in (0, 1), got " << r;
        return false;
      }
    } catch (const std::exception&) {
      LOGE(STITCHER) << "crop_ratio is not a valid float: "
                     << param_set.at(kParamCropRatio);
      return false;
    }
  }
  if (param_set.find(kParamDirection) != param_set.end()) {
    const std::string& d = param_set.at(kParamDirection);
    if (d != kDirHorizontal && d != kDirVertical) {
      LOGE(STITCHER) << "direction must be '" << kDirHorizontal << "' or '"
                     << kDirVertical << "', got " << d;
      return false;
    }
  }
  return true;
}

bool FrameStitcher::Open(ModuleParamSet param_set) {
  if (param_set.find(kParamEnable) != param_set.end()) {
    const std::string& v = param_set.at(kParamEnable);
    enable_ = (v == "1" || v == "true" || v == "True" || v == "TRUE");
  }
  if (param_set.find(kParamDirection) != param_set.end()) {
    const std::string& d = param_set.at(kParamDirection);
    direction_ = (d == kDirVertical) ? Direction::kVertical : Direction::kHorizontal;
  }
  if (param_set.find(kParamCropRatio) != param_set.end()) {
    try {
      crop_ratio_ = std::clamp(std::stof(param_set.at(kParamCropRatio)), 0.01f, 1.0f);
    } catch (const std::exception&) {
      LOGE(STITCHER) << "invalid crop_ratio, fallback to " << crop_ratio_;
    }
  }
  if (param_set.find(kParamTargetModels) != param_set.end()) {
    target_models_ = ParseTargetModels(param_set.at(kParamTargetModels));
  }
  // 开启拼接时必须明确作用目标，否则拼出的图没有模型消费
  if (enable_ && target_models_.empty()) {
    LOGE(STITCHER) << "enable=true requires at least one target model in ["
                   << kParamTargetModels << "]";
    return false;
  }

  std::string targets;
  for (const auto& model_name : target_models_) {
    if (!targets.empty()) targets += ",";
    targets += model_name;
  }
  LOGI(STITCHER) << "Open: enable=" << (enable_ ? "true" : "false")
                 << ", direction=" << (direction_ == Direction::kVertical ? kDirVertical : kDirHorizontal)
                 << ", crop_ratio=" << crop_ratio_
                 << ", target_models=[" << targets << "]";
  return true;
}

void FrameStitcher::Close() {
  std::lock_guard<std::mutex> lk(cache_mtx_);
  frame_cache_.clear();
  LOGI(STITCHER) << "Close: cache cleared";
}

void FrameStitcher::OnEos(const std::string& stream_id) {
  std::lock_guard<std::mutex> lk(cache_mtx_);
  frame_cache_.erase(stream_id);
  LOGI(STITCHER) << "OnEos: clear cache for stream " << stream_id;
}

int FrameStitcher::Process(std::shared_ptr<FrameInfo> data) {
  if (!data) return -1;
  if (data->IsEos()) return 0;

  if (!data->collection.HasValue(kDataFrameTag)) {
    LOGE(STITCHER) << "DataFrame not found in collection, stream=" << data->GetStreamId();
    return -1;
  }
  DataFramePtr frame = data->collection.Get<DataFramePtr>(kDataFrameTag);
  if (!frame) {
    LOGE(STITCHER) << "DataFrame is null, stream=" << data->GetStreamId();
    return -1;
  }

  cv::Mat cur = frame->GetImage();  // BGR, shallow ref（只读使用）
  if (cur.empty()) {
    LOGE(STITCHER) << "current frame image is empty, stream=" << data->GetStreamId();
    return -1;
  }
  const std::string& stream_id = data->GetStreamId();

  if (!enable_) {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    frame_cache_[stream_id] = cur.clone();
    return 0;
  }

  cv::Mat prev;
  {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    auto it = frame_cache_.find(stream_id);
    if (it != frame_cache_.end()) {
      prev = it->second;  // shallow ref 即可，本帧内不会被并发改写
    }
  }

  // 首帧：无上一帧，无法拼接，仅缓存当前帧。
  if (prev.empty()) {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    frame_cache_[stream_id] = cur.clone();
    LOGD(STITCHER) << "first frame, no prev to stitch, stream=" << stream_id;
    return 0;
  }

  auto derived = std::make_shared<ModelInputImage>();
  derived->image = cur;

  int cur_w = cur.cols;
  int cur_h = cur.rows;

  if (direction_ == Direction::kHorizontal) {
    // 取上一帧右侧 crop_ratio_ 宽度的裁剪
    int crop_w = std::max(1, static_cast<int>(prev.cols * crop_ratio_));
    crop_w = std::min(crop_w, prev.cols);
    cv::Rect crop_roi(prev.cols - crop_w, 0, crop_w, prev.rows);
    cv::Mat _crop = prev(crop_roi);

    if (_crop.rows != cur_h) {  // 检查是否等高
      cv::Mat resized;
      cv::resize(_crop, resized, cv::Size(crop_w, cur_h), 0, 0, cv::INTER_LINEAR);
      _crop = resized;
    }
    // 水平拼接：左 = 上一帧裁剪，右 = 当前帧
    cv::hconcat(_crop, cur, derived->image);

    // 当前帧在派生图中的位置（用于后处理坐标还原）
    derived->cur_offset_x = _crop.cols;  // resize 后的实际宽度
    derived->cur_offset_y = 0;
    derived->cur_width = cur_w;
    derived->cur_height = cur_h;
    derived->cur_scale_x = 1.0f;
    derived->cur_scale_y = 1.0f;

  } else {
    // 垂直方向：取上一帧底部 crop_ratio_ 高度的裁剪
    int crop_h = std::max(1, static_cast<int>(prev.rows * crop_ratio_));
    crop_h = std::min(crop_h, prev.rows);
    cv::Rect crop_roi(0, prev.rows - crop_h, prev.cols, crop_h);
    cv::Mat _crop = prev(crop_roi);

    if (_crop.cols != cur_w) {  // 检查是否等宽
      cv::Mat resized;
      cv::resize(_crop, resized, cv::Size(cur_w, crop_h), 0, 0, cv::INTER_LINEAR);
      _crop = resized;
    }

    cv::vconcat(_crop, cur, derived->image);

    derived->cur_offset_x = 0;
    derived->cur_offset_y = _crop.rows;
    derived->cur_width = cur_w;
    derived->cur_height = cur_h;
    derived->cur_scale_x = 1.0f;
    derived->cur_scale_y = 1.0f;
  }

  // 按 target_models 写入模块级派生图槽位；Set 覆盖已有值时给出告警
  for (const auto& model_name : target_models_) {
    const std::string tag = ModelInputImageTagForModel(model_name);
    if (data->collection.HasValue(tag)) {
      LOGW(STITCHER) << "overwrite existing module-level derived image, model=" << model_name
                     << ", tag=" << tag << ", stream=" << stream_id;
    }
    data->collection.Set<ModelInputImagePtr>(tag, derived);
  }

  {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    frame_cache_[stream_id] = cur.clone();
  }

  LOGD(STITCHER) << "stitched derived image " << derived->image.cols << "x" << derived->image.rows
                 << ", cur_offset=(" << derived->cur_offset_x << "," << derived->cur_offset_y
                 << "), stream=" << stream_id;
  return 0;
}

}  // namespace cnstream
