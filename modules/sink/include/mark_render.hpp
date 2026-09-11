#ifndef MODULES_SINK_MARK_RENDER_HPP_
#define MODULES_SINK_MARK_RENDER_HPP_

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "cnstream_frame_va.hpp"

namespace cnstream {

/**
 * @brief 单条标框白名单规则：对象命中任一规则即绘制，规则内部各条件取与。
 */
struct MarkRule {
  std::string model;      ///< 模型名，空表示不限制
  std::set<int> ids;      ///< 类别 id 集合，空表示不限制
  bool has_type = false;  ///< 是否限定目标类型
  InferObjType type = InferObjType::kUnknown;
};

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
   * mark_filter is configured as a JSON array of rule objects:
   *   "mark_filter": [
   *     {"model": "yolo_ocr", "ids": [0, 1], "type": "merged"},
   *     {"type": "original"}
   *   ]
   * - model: empty/omitted = any model
   * - ids:   empty/omitted = any class id
   * - type:  "original" or "merged" (InferObjType); omitted = any type
   */
  std::vector<MarkRule> rules;

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
