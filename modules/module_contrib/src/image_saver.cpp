
#include "image_saver.hpp"

#include <chrono>
#include <filesystem>
#include <system_error>

#include "cnstream_logging.hpp"

namespace cnstream {

namespace {

constexpr const char* kParamEnable = "enable";
constexpr const char* kParamSaveDir = "save_dir";
constexpr const char* kParamIntervalMs = "interval_ms";
constexpr const char* kParamMaxCount = "max_count";
constexpr const char* kParamPrefix = "prefix";
constexpr const char* kParamSuffix = "suffix";
constexpr const char* kParamExt = "ext";

}  // namespace

std::string ImageSaver::NormalizeDir(std::string dir) {
  if (dir.empty()) return "save/";
  if (dir.back() != '/') dir.push_back('/');
  return dir;
}

void ImageSaver::TrimOldestFiles(StreamState& state) {
  if (max_count_ <= 0) return;
  while (static_cast<int>(state.saved_files.size()) > max_count_) {
    const std::string& oldest = state.saved_files.front();
    std::error_code ec;
    std::filesystem::remove(oldest, ec);
    if (ec) {
      LOGW(IMAGE_SAVER) << "failed to remove oldest file: " << oldest
                        << ", err=" << ec.message();
    }
    state.saved_files.pop_front();
  }
}

/**
 * @param param_set 参数集 custom_params
 *   - enable:       "1"/"true" 启用，"0"/"false" 关闭（默认 true）
 *   - save_dir:     保存目录（默认 "save"）
 *   - interval_ms:  保存节流间隔 ms（默认 0，每帧保存）
 *   - max_count:    单路最大文件数（默认 0，不限制；>0 时超出按 FIFO 删除）
 *   - prefix:       文件名前缀（默认 "img"）
 *   - suffix:       文件名后缀，可空（默认 ""）
 *   - ext:          文件扩展名，不含 '.'（默认 "jpg"）
 */
bool ImageSaver::CheckParamSet(const ModuleParamSet& param_set) const {
  if (param_set.find(kParamIntervalMs) != param_set.end()) {
    try {
      int v = std::stoi(param_set.at(kParamIntervalMs));
      if (v < 0) {
        LOGE(IMAGE_SAVER) << "interval_ms must be >= 0, got " << v;
        return false;
      }
    } catch (const std::exception&) {
      LOGE(IMAGE_SAVER) << "interval_ms is not a valid int: "
                        << param_set.at(kParamIntervalMs);
      return false;
    }
  }
  if (param_set.find(kParamMaxCount) != param_set.end()) {
    try {
      int v = std::stoi(param_set.at(kParamMaxCount));
      if (v < 0) {
        LOGE(IMAGE_SAVER) << "max_count must be >= 0, got " << v;
        return false;
      }
    } catch (const std::exception&) {
      LOGE(IMAGE_SAVER) << "max_count is not a valid int: "
                        << param_set.at(kParamMaxCount);
      return false;
    }
  }
  return true;
}

bool ImageSaver::Open(ModuleParamSet param_set) {
  if (param_set.find(kParamEnable) != param_set.end()) {
    const std::string& v = param_set.at(kParamEnable);
    enable_ = (v == "1" || v == "true" || v == "True" || v == "TRUE");
  }
  if (param_set.find(kParamSaveDir) != param_set.end()) {
    save_dir_ = param_set.at(kParamSaveDir);
  }
  if (param_set.find(kParamIntervalMs) != param_set.end()) {
    try {
      interval_ms_ = std::max(0, std::stoi(param_set.at(kParamIntervalMs)));
    } catch (const std::exception&) {
      LOGE(IMAGE_SAVER) << "invalid interval_ms, fallback to " << interval_ms_;
    }
  }
  if (param_set.find(kParamMaxCount) != param_set.end()) {
    try {
      max_count_ = std::max(0, std::stoi(param_set.at(kParamMaxCount)));
    } catch (const std::exception&) {
      LOGE(IMAGE_SAVER) << "invalid max_count, fallback to " << max_count_;
    }
  }
  if (param_set.find(kParamPrefix) != param_set.end()) {
    prefix_ = param_set.at(kParamPrefix);
  }
  if (param_set.find(kParamSuffix) != param_set.end()) {
    suffix_ = param_set.at(kParamSuffix);
  }
  if (param_set.find(kParamExt) != param_set.end()) {
    ext_ = param_set.at(kParamExt);
  }

  save_dir_ = NormalizeDir(save_dir_);

  if (enable_) {
    std::error_code ec;
    std::filesystem::create_directories(save_dir_, ec);
    if (ec) {
      LOGE(IMAGE_SAVER) << "failed to create save_dir: " << save_dir_
                        << ", err=" << ec.message();
      return false;
    }
  }

  LOGI(IMAGE_SAVER) << "Open: enable=" << (enable_ ? "true" : "false")
                    << ", save_dir=" << save_dir_
                    << ", interval_ms=" << interval_ms_
                    << ", max_count=" << max_count_
                    << ", prefix=" << prefix_
                    << ", suffix=" << suffix_
                    << ", ext=" << ext_;
  return true;
}

void ImageSaver::Close() {
  std::lock_guard<std::mutex> lk(mtx_);
  for (auto& kv : states_) {
    kv.second.saved_files.clear();
  }
  states_.clear();
  LOGI(IMAGE_SAVER) << "Close: state cleared";
}

void ImageSaver::OnEos(const std::string& stream_id) {
  std::lock_guard<std::mutex> lk(mtx_);
  states_.erase(stream_id);
  LOGD(IMAGE_SAVER) << "OnEos: clear state for stream " << stream_id;
}

int ImageSaver::Process(std::shared_ptr<FrameInfo> data) {
  if (!data) return -1;
  if (data->IsEos()) return 0;
  if (!enable_) return 0;

  if (!data->collection.HasValue(kDataFrameTag)) {
    LOGE(IMAGE_SAVER) << "DataFrame not found in collection, stream="
                      << data->GetStreamId();
    return -1;
  }
  DataFramePtr frame = data->collection.Get<DataFramePtr>(kDataFrameTag);
  if (!frame) {
    LOGE(IMAGE_SAVER) << "DataFrame is null, stream=" << data->GetStreamId();
    return -1;
  }

  cv::Mat img = frame->GetImage();  // BGR, shallow ref
  if (img.empty()) {
    LOGE(IMAGE_SAVER) << "frame image is empty, stream=" << data->GetStreamId();
    return -1;
  }

  const std::string& stream_id = data->GetStreamId();

  // 节流判断：必须持锁，避免与 Close/OnEos 出现状态竞态
  auto now = std::chrono::steady_clock::now();
  std::string filename;
  StreamState* state = nullptr;

  {
    std::lock_guard<std::mutex> lk(mtx_);
    auto& s = states_[stream_id];
    if (interval_ms_ > 0 && s.has_saved &&
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - s.last_save_time).count() < interval_ms_) {
      return 0;
    }

    // 文件名：<prefix>_<stream_id>_<timestamp_ms>[_<suffix>].<ext>
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    filename = save_dir_ + prefix_ + "_" + stream_id + "_" +
               std::to_string(ts);
    if (!suffix_.empty()) filename += "_" + suffix_;
    filename += "." + ext_;

    // 写入磁盘（clone 一份避免外层修改影响磁盘内容）
    cv::Mat clone = img.clone();
    if (!cv::imwrite(filename, clone)) {
      LOGE(IMAGE_SAVER) << "imwrite failed: " << filename
                        << ", stream=" << stream_id;
      return -1;
    }

    s.last_save_time = now;
    s.has_saved = true;
    if (max_count_ > 0) {
      s.saved_files.push_back(filename);
      TrimOldestFiles(s);
    }
    state = &s;
  }

  LOGD(IMAGE_SAVER) << "saved " << filename
                    << ", stream=" << stream_id
                    << ", kept=" << (state && max_count_ > 0
                                         ? static_cast<int>(state->saved_files.size())
                                         : -1);
  return 0;
}

}  // namespace cnstream
