#include "orientation_check.hpp"

#include <cstdlib>

#include "cnstream_logging.hpp"
#include "cnstream_pipeline.hpp"

namespace cnstream {

namespace {

constexpr const char* kParamSkipModule = "skip_module";
constexpr const char* kParamWarmupFrames = "warmup_frames";

}  // namespace

bool OrientationCheck::CheckParamSet(const ModuleParamSet& param_set) const {
  auto it = param_set.find(kParamSkipModule);
  if (it == param_set.end() || it->second.empty()) {
    LOGE(ORIENTATION) << "param [" << kParamSkipModule << "] is required, e.g. \"ocr_rec\"";
    return false;
  }
  if (param_set.find(kParamWarmupFrames) != param_set.end()) {
    try {
      if (std::stoi(param_set.at(kParamWarmupFrames)) <= 0) {
        LOGE(ORIENTATION) << kParamWarmupFrames << " must be > 0";
        return false;
      }
    } catch (const std::exception& e) {
      LOGE(ORIENTATION) << "invalid " << kParamWarmupFrames << ": " << e.what();
      return false;
    }
  }
  return true;
}

bool OrientationCheck::Open(ModuleParamSet param_set) {
  skip_module_ = param_set.at(kParamSkipModule);
  if (param_set.find(kParamWarmupFrames) != param_set.end()) {
    warmup_frames_ = static_cast<size_t>(std::stoul(param_set.at(kParamWarmupFrames)));
  }

  // Open 在全部模块创建之后调用，可直接按名解析需跳过的下游模块
  Module* skip_module = GetContainer()->GetModule(skip_module_);
  if (!skip_module) {
    LOGE(ORIENTATION) << "[" << GetName() << "] skip module [" << skip_module_ << "] not found in pipeline";
    return false;
  }
  skip_module_ptr_.store(skip_module);

  LOGI(ORIENTATION) << "[" << GetName() << "] opened, skip_module=" << skip_module_
                << ", warmup_frames=" << warmup_frames_;
  return true;
}

void OrientationCheck::Close() {
  std::lock_guard<std::mutex> lk(mtx_);
  states_.clear();
}

void OrientationCheck::OnEos(const std::string& stream_id) {
  // 结论 per-stream 保留（同一 stream_id 重连不重新判断）；
  // 仅重置未决期的样本积累，重连后重新积累
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = states_.find(stream_id);
  if (it != states_.end() && it->second.orientation == Orientation::kUnknown) {
    it->second.sample_count = 0;
  }
}

int OrientationCheck::Process(std::shared_ptr<FrameInfo> data) {
  Module* skip_module = skip_module_ptr_.load();
  if (!skip_module) {
    skip_module = GetContainer()->GetModule(skip_module_);  // 惰性兜底
    if (!skip_module) {
      LOGE(ORIENTATION) << "[" << GetName() << "] skip module [" << skip_module_ << "] not found";
      return -1;
    }
    skip_module_ptr_.store(skip_module);
  }

  Orientation result = Orientation::kUnknown;
  bool decided = false;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    StreamState& state = states_[data->stream_id];
    if (state.orientation != Orientation::kUnknown) {
      result = state.orientation;
      decided = true;
    } else {
      decided = JudgeOrientation(&state, data, &result);
      if (decided) state.orientation = result;
    }
  }

  if (!decided) {
    // 结论未出：本帧跳过指定推理模块
    data->MarkSkipModule(skip_module);
    LOGI(ORIENTATION) << " Mark skip module [" << skip_module->GetName() << "]";
    return 0;
  }

  if (result == Orientation::kBack) {
    data->collection.AddIfNotExists(kCropRotate180Tag, true);
  }
  return 0;
}

bool OrientationCheck::JudgeOrientation(StreamState* state, const std::shared_ptr<FrameInfo>& data,
                                        Orientation* result) {
  ++state->sample_count;
  if (state->sample_count >= warmup_frames_) {
    *result = Orientation::kFront;  // 占位结论：默认正面（不旋转）
    LOGI(ORIENTATION) << "[" << GetName() << "] stream [" << data->stream_id
                  << "] orientation decided (placeholder): front, frames=" << state->sample_count;
    return true;
  }
  return false;
}

}  // namespace cnstream
