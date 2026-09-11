#include "mark_render.hpp"

#include <cctype>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

#include "cnstream_logging.hpp"

#ifdef VSTREAM_USE_CUDA
#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_syncmem_cuda.hpp"
#endif

namespace cnstream {

std::unique_ptr<MarkRender> MarkRender::Create(DevType device_type) {
#ifdef VSTREAM_USE_CUDA
  if (device_type == DevType::CUDA) {
    return std::make_unique<CudaMarkRender>();
  }
#endif
  return std::make_unique<CpuMarkRender>();
}

namespace {

std::string Trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

bool ShouldDraw(const MarkConfig& config, const std::shared_ptr<InferObject>& obj) {
  if (config.rules.empty()) {
    return true;  // 没有配置规则，默认绘制
  }
  // 每个规则（对应一个列表）内的条件都必须满足
  // 只需要满足一个规则的条件，即可绘制
  for (const auto& rule : config.rules) {
    if (!rule.model.empty() && rule.model != obj->model_name) continue;
    if (!rule.ids.empty() && rule.ids.count(obj->id) == 0) continue;
    if (rule.has_type && GetInferObjType(obj) != ToString(rule.type)) continue;
    return true;
  }
  return false;
}

}  // namespace

bool MarkConfig::ParseMarkFilter(const std::string& filter) {
  rules.clear();
  std::string trimmed = Trim(filter);
  if (trimmed.empty()) {
    return true;  // empty filter is a valid no-op
  }

  nlohmann::json doc = nlohmann::json::parse(trimmed, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_array()) {
    LOGE(SINK) << "Mark filter: expect a JSON array of rule objects";
    return false;
  }

  std::vector<MarkRule> parsed;
  parsed.reserve(doc.size());
  for (const auto& item : doc) {
    if (!item.is_object()) {
      LOGE(SINK) << "Mark filter: each rule must be a JSON object";
      return false;
    }
    MarkRule rule;
    for (auto it = item.begin(); it != item.end(); ++it) {
      const std::string& key = it.key();
      if (key == "model") {
        if (!it.value().is_string()) {
          LOGE(SINK) << "Mark filter: rule.model must be a string";
          return false;
        }
        rule.model = it.value().get<std::string>();
      } else if (key == "ids") {
        if (!it.value().is_array()) {
          LOGE(SINK) << "Mark filter: rule.ids must be an array of integers";
          return false;
        }
        for (const auto& id_val : it.value()) {
          if (!id_val.is_number_integer()) {
            LOGE(SINK) << "Mark filter: rule.ids must contain integers only";
            return false;
          }
          rule.ids.insert(id_val.get<int>());
        }
      } else if (key == "type") {
        if (!it.value().is_string()) {
          LOGE(SINK) << "Mark filter: rule.type must be a string";
          return false;
        }
        const std::string type_str = it.value().get<std::string>();
        rule.type = InferObjTypeFromString(type_str);
        if (rule.type == InferObjType::kUnknown) {
          LOGE(SINK) << "Mark filter: unknown rule.type '" << type_str
                     << "', expect 'original' or 'merged'";
          return false;
        }
        rule.has_type = true;
      } else {
        LOGW(SINK) << "Mark filter: unknown rule key '" << key << "', ignored";
      }
    }
    parsed.push_back(std::move(rule));
  }

  rules = std::move(parsed);
  return true;
}

bool CpuMarkRender::Render(DataFramePtr frame, const InferObjsPtr& objs,
                                const MarkConfig& config) {
  if (!frame || !objs || objs->objs_.empty()) return false;
  if (frame->GetFmt() != DataFormat::PIXEL_FORMAT_RGB24 &&
      frame->GetFmt() != DataFormat::PIXEL_FORMAT_BGR24) {
    LOGW(SINK) << "Mark: unsupported pixel format "
               << static_cast<int>(frame->GetFmt()) << ", skip render";
    return false;
  }

  int img_w = frame->GetWidth();
  int img_h = frame->GetHeight();
  int stride = frame->GetStride(0);

  void* mutable_data = frame->data_[0]->GetMutableCpuData();
  if (!mutable_data) {
    LOGW(SINK) << "Mark: GetMutableCpuData failed";
    return false;
  }

  cv::Mat img(img_h, img_w, CV_8UC3, mutable_data, stride);

  std::lock_guard<std::mutex> lk(objs->mutex_);
  for (const auto& obj : objs->objs_) {
    if (!obj) continue;
    if (!ShouldDraw(config, obj)) continue;

    float x = obj->bbox.x;
    float y = obj->bbox.y;
    float w = obj->bbox.w;
    float h = obj->bbox.h;

    int left   = std::max(0, static_cast<int>(x));
    int top    = std::max(0, static_cast<int>(y));
    int right  = std::min(img_w, static_cast<int>(x + w));
    int bottom = std::min(img_h, static_cast<int>(y + h));

    if (right <= left || bottom <= top) continue;

    cv::rectangle(img, cv::Rect(left, top, right - left, bottom - top),
                  config.color, config.thickness);

    if (config.draw_label || config.draw_score) {
      std::string text;
      if (config.draw_label) text += std::to_string(obj->id);
      if (config.draw_score) {
        if (!text.empty()) text += ":";
        char buf[16];
        snprintf(buf, sizeof(buf), "%.2f", obj->score);
        text += buf;
      }
      if (!text.empty()) {
        int baseline;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                      config.font_scale, 1, &baseline);
        cv::rectangle(img,
                      cv::Point(left, top - ts.height - 4),
                      cv::Point(left + ts.width, top),
                      config.color, cv::FILLED);
        cv::putText(img, text, cv::Point(left, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX, config.font_scale,
                    cv::Scalar(255, 255, 255), 1);
      }
    }
  }

  return true;
}

#ifdef VSTREAM_USE_CUDA
bool CudaMarkRender::Render(DataFramePtr frame, const InferObjsPtr& objs,
                            const MarkConfig& config) {
  if (!frame || !objs || objs->objs_.empty()) return false;

  CpuMarkRender cpu_render;
  return cpu_render.Render(frame, objs, config);
}
#endif

}  // namespace cnstream