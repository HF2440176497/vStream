#include "mark_render.hpp"

#include <cctype>
#include <mutex>
#include <sstream>
#include <string>

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

bool ShouldDraw(const MarkConfig& config, const InferObject& obj) {
  if (config.filter_model_ids.empty()) {
    return true;  // no filter configured, draw everything
  }
  auto it = config.filter_model_ids.find(obj.model_name);
  if (it == config.filter_model_ids.end()) {
    return false;
  }
  // Empty id-set for a model = wildcard (any id under this model_name is allowed).
  if (it->second.empty()) {
    return true;
  }
  return it->second.count(obj.id) > 0;
}

}  // namespace

bool MarkConfig::ParseMarkFilter(const std::string& filter) {
  filter_model_ids.clear();
  std::string trimmed = Trim(filter);
  if (trimmed.empty()) {
    return true;  // empty filter is a valid no-op
  }

  std::unordered_map<std::string, std::set<int>> parsed;
  std::stringstream entries(trimmed);
  std::string entry;
  while (std::getline(entries, entry, ';')) {
    entry = Trim(entry);
    if (entry.empty()) continue;

    size_t sep = entry.find(':');
    if (sep == std::string::npos) {
      LOGE(SINK) << "Mark filter: missing ':' in entry '" << entry << "'";
      filter_model_ids.clear();
      return false;
    }
    std::string model = Trim(entry.substr(0, sep));
    std::string ids   = Trim(entry.substr(sep + 1));
    if (model.empty()) {
      LOGE(SINK) << "Mark filter: empty model name in entry '" << entry << "'";
      filter_model_ids.clear();
      return false;
    }

    std::set<int> id_set;
    if (!ids.empty()) {
      std::stringstream id_stream(ids);
      std::string id_str;
      while (std::getline(id_stream, id_str, ',')) {
        id_str = Trim(id_str);
        if (id_str.empty()) continue;
        try {
          size_t consumed = 0;
          int id = std::stoi(id_str, &consumed);
          if (consumed != id_str.size()) {
            LOGE(SINK) << "Mark filter: trailing chars in id '" << id_str << "'";
            filter_model_ids.clear();
            return false;
          }
          id_set.insert(id);
        } catch (const std::exception&) {
          LOGE(SINK) << "Mark filter: invalid id '" << id_str << "' in entry '" << entry << "'";
          filter_model_ids.clear();
          return false;
        }
      }
    }
    parsed[model] = std::move(id_set);
  }

  filter_model_ids = std::move(parsed);
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
    if (!ShouldDraw(config, *obj)) continue;

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