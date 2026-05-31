#ifndef MODULES_SINK_MARK_RENDER_HPP_
#define MODULES_SINK_MARK_RENDER_HPP_

#include <memory>
#include <opencv2/opencv.hpp>

#include "cnstream_frame_va.hpp"

namespace cnstream {

struct MarkConfig {
  bool draw_bbox = true;
  bool draw_label = false;
  bool draw_score = false;
  float font_scale = 0.5f;
  int thickness = 2;
  cv::Scalar color{0, 255, 0};
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
