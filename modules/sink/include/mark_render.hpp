#ifndef MODULES_SINK_MARK_RENDER_HPP_
#define MODULES_SINK_MARK_RENDER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "cnstream_obj_rule.hpp"

namespace cnstream {

struct MarkConfig {
  bool draw_bbox = true;  // 开启绘制时 默认只开启标框
  bool draw_label = false;
  bool draw_score = false;
  float font_scale = 0.5f;
  int thickness = 2;
  cv::Scalar color{0, 255, 0};

  /**
   * Whitelist rules applied before drawing. An object is drawn only if it
   * matches at least one rule; within a rule all set conditions must hold.
   * Empty `rules` means no filtering (draw everything).
   *
   * mark_filter is configured as a JSON array of rule objects; 
   * a single rule object is also accepted (see ParseObjRules):
   *   "mark_filter": {"type": "merged"}
   *   "mark_filter": [{"model": "yolo_ocr", "ids": [0, 1], "type": "merged"},
   *                   {"model": ["a", "b"], "position": {"x": [0.1, 0.9]}}]
   * - model:    string or string array; empty/omitted = any model
   * - ids:      integer or integer array; empty/omitted = any class id
   * - type:     "original"/"merged", string or string array; omitted = any type
   * - position: optional per-rule bbox filter {"x": [min, max], "y": [min, max]}
   */
  std::vector<ObjRule> rules;

  /**
   * Parse a JSON filter string into `rules`.
   * @param filter Filter spec, see format above. Empty string clears rules.
   * @return true on success, false if the string is malformed (in which case
   *         the rules are left empty and no filter is applied).
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
