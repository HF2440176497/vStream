#ifndef MODULES_SINK_MARK_RENDER_HPP_
#define MODULES_SINK_MARK_RENDER_HPP_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#include "cnstream_frame_va.hpp"

namespace cnstream {

struct MarkConfig {
  bool draw_bbox = true;  // 开启绘制时 默认只开启标框
  bool draw_label = false;
  bool draw_score = false;
  float font_scale = 0.5f;
  int thickness = 2;
  cv::Scalar color{0, 255, 0};

  /**
   * Optional whitelist filter applied before drawing.
   *
   * When `filter_model_ids` is non-empty, an object is drawn only if its
   * (model_name, id) pair is contained in the map. A model_name entry with
   * an empty id-set acts as a wildcard: it accepts any id for that model.
   *
   * Format expected from configuration (parsed by ParseMarkFilter):
   *   "model_a:0,1,2;model_b:5"
   * i.e. entries separated by ';', model name and id list separated by ':',
   * ids separated by ','. Whitespace around tokens is trimmed.
   */
  std::unordered_map<std::string, std::set<int>> filter_model_ids;

  /**
   * Parse a filter string into `filter_model_ids`.
   * @param filter Filter spec, see format above. Empty string clears the map.
   * @return true on success, false if the string is malformed (in which case
   *         the map is left empty and no filter is applied).
   */
  bool ParseMarkFilter(const std::string& filter);
};

class MarkRender {
 public:
  MarkRender() = default;
  virtual ~MarkRender() = default;

  MarkRender(const MarkRender&) = delete;
  MarkRender& operator=(const MarkRender&) = delete;

  virtual bool Render(DataFramePtr frame, const InferObjsPtr& objs,
                      const MarkConfig& config) = 0;

  static std::unique_ptr<MarkRender> Create(DevType device_type);
};

class CpuMarkRender : public MarkRender {
 public:
  bool Render(DataFramePtr frame, const InferObjsPtr& objs,
              const MarkConfig& config) override;
};

#ifdef VSTREAM_USE_CUDA
class CudaMarkRender : public MarkRender {
 public:
  bool Render(DataFramePtr frame, const InferObjsPtr& objs,
              const MarkConfig& config) override;
};
#endif

}  // namespace cnstream

#endif  // MODULES_SINK_MARK_RENDER_HPP_
