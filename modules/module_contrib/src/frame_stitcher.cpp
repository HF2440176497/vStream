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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include "frame_stitcher.hpp"

#include "cnstream_logging.hpp"

namespace cnstream {

namespace {

constexpr const char* kParamEnable = "enable";
constexpr const char* kParamDirection = "direction";   // "horizontal" / "vertical"
constexpr const char* kParamCropRatio = "crop_ratio";  // (0, 1)

constexpr const char* kDirHorizontal = "horizontal";
constexpr const char* kDirVertical = "vertical";

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
      crop_ratio_ = std::clamp(std::stof(param_set.at(kParamCropRatio)), 0.01f, 0.99f);
    } catch (const std::exception&) {
      LOGE(STITCHER) << "invalid crop_ratio, fallback to " << crop_ratio_;
    }
  }
  LOGI(STITCHER) << "Open: enable=" << (enable_ ? "true" : "false")
                 << ", direction=" << (direction_ == Direction::kVertical ? kDirVertical : kDirHorizontal)
                 << ", crop_ratio=" << crop_ratio_;
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

  // 未启用：不写派生图，Preproc 自动回退原图；但仍缓存上一帧以便随时启用
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
  derived->image = cur;  // 先占位，下面覆盖；保证失败时也有可读值

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

  data->collection.AddIfNotExists(kModelInputImageTag, derived);

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
