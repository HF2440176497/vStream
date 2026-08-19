#ifndef MODULES_INFERENCE_MODEL_VALIDATOR_HPP_
#define MODULES_INFERENCE_MODEL_VALIDATOR_HPP_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "model_loader.hpp"
#include "infer_params.hpp"
#include "preproc.hpp"
#include "postproc.hpp"
#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "data_source_param.hpp"
#include "memop.hpp"
#include "cnstream_logging.hpp"

namespace cnstream {

#define MODEL_VALIDATOR "ModelValidator"

/// @brief Single tensor descriptor (name, shape, dtype)
struct ModelTensorInfo {
  std::string name;
  std::vector<int> shape;
  std::string dtype;
};

/// @brief Summary of loaded model metadata
struct ModelInfo {
  std::vector<ModelTensorInfo> inputs;
  std::vector<ModelTensorInfo> outputs;
  int batch_size = 0;
  int channel = 0;
  int height = 0;
  int width = 0;
  std::string device_type;
  int device_id = 0;
  std::string model_path;
};

/// @brief One detection result (normalized bbox)
struct ValidatorDetection {
  int class_id = -1;
  std::string class_name;
  float score = 0.0f;
  float x = 0.0f;  ///< normalized center-x
  float y = 0.0f;  ///< normalized center-y
  float w = 0.0f;  ///< normalized width
  float h = 0.0f;  ///< normalized height
};

/// @brief End-to-end validation result
struct E2EResult {
  std::vector<ValidatorDetection> detections;
  double latency_ms = 0.0;
  std::string error;  ///< non-empty if failed
};

/// @brief Single batch-size benchmark entry
struct BenchmarkResult {
  int batch_size = 1;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double p99_ms = 0.0;
  double fps = 0.0;
  int error_count = 0;
};

/**
 * @class ModelValidator
 * @brief Standalone model validation tool
 * 但是暂不支持对象级 obj 的模型验证
 *
 * Usage flow:
 *   1. Construct with model path + device
 *   2. Load() — loads the model via ModelLoaderFactory
 *   3. GetModelInfo() — inspect tensor shapes/dtypes
 *   4. Infer() — raw tensor in/out (no preproc/postproc)
 *   5. RunE2E() — image -> preproc -> infer -> postproc -> detections
 *   6. Benchmark() — measure latency over many runs
 *
 * Does NOT depend on Pipeline, Connector, EventBus, or InferEngine.
 */
class ModelValidator {
 public:
  /**
   * @param model_path Full path to engine file (.engine / .rknn / .onnx)
   * @param device_type "cpu" | "cuda" | "rockchip"
   * @param device_id   Device ordinal (GPU id / NPU core)
   * @param input_ordered_index Index of the primary image input tensor
   */
  ModelValidator(const std::string& model_path,
                 const std::string& device_type = "cuda",
                 int device_id = 0,
                 int input_ordered_index = 0);
  ~ModelValidator();

  /// @brief Load model and allocate buffers. Returns false on failure.
  bool Load();

  /// @brief Whether model is loaded and valid.
  bool IsLoaded() const;

  /// @brief Get model metadata (shapes, dtypes, names).
  ModelInfo GetModelInfo() const;

  /// @brief Raw tensor inference — no preproc/postproc.
  /// @param inputs One flat float vector per input tensor. Each vector's size
  ///               must equal InputShape(i).DataCount().
  /// @return One flat float vector per output tensor.
  std::vector<std::vector<float>> Infer(
      const std::vector<std::vector<float>>& inputs);

  /// @brief End-to-end: image -> preproc -> infer -> postproc -> detections.
  /// @param image BGR image (CV_8UC3)
  /// @param preproc_name Registered preproc class name (e.g. "Pre_YOLO_CPU_v2")
  /// @param postproc_name Registered postproc class name (e.g. "Post_YOLOv8_CPU_v2")
  /// @param preproc_params Custom params passed to Preproc::Init
  /// @param postproc_params Custom params passed to Postproc::Init
  ///        (e.g. {"config_file": "yolo_coco.json"})
  E2EResult RunE2E(
      const cv::Mat& image,
      const std::string& preproc_name,
      const std::string& postproc_name,
      const std::map<std::string, std::string>& preproc_params = {},
      const std::map<std::string, std::string>& postproc_params = {});

  /// @brief Benchmark E2E latency over multiple runs.
  /// @param warmup_runs Number of warmup iterations (not measured)
  /// @param test_runs Number of measured iterations
  /// @param batch_sizes Currently only batch_size=1 is supported per-image
  std::vector<BenchmarkResult> Benchmark(
      const cv::Mat& image,
      const std::string& preproc_name,
      const std::string& postproc_name,
      const std::map<std::string, std::string>& preproc_params,
      const std::map<std::string, std::string>& postproc_params,
      int warmup_runs = 10,
      int test_runs = 100,
      const std::vector<int>& batch_sizes = {1});

 private:
  DevType device_type_ = DevType::CUDA;
  int device_id_ = 0;
  std::string model_path_;
  uint32_t input_ordered_index_ = 0;

  std::unique_ptr<ModelLoader> model_loader_;
  void* exec_ctx_ = nullptr;  ///< loader 提供的独占执行上下文（Load 获取，析构归还）
  std::shared_ptr<MemOp> memop_;  ///< device-aware memory operations

  // RAII 内存
  // CPU-side float buffers (preproc writes / postproc reads)
  std::vector<std::vector<float>> cpu_input_bufs_;
  std::vector<std::vector<float>> cpu_output_bufs_;

  // Device-side buffers (passed to RunSync)
  std::vector<std::shared_ptr<void>> dev_input_bufs_;
  std::vector<std::shared_ptr<void>> dev_output_bufs_;

  void AllocateBuffers();

  /// @brief Create a FrameInfo with DataFrame(image) + InferObjs + ModelInputImage
  FrameInfoPtr CreateFrameInfo(const cv::Mat& image,
                               const std::string& stream_id = "validator");

  /// @brief Run preproc: fills cpu_input_bufs_ from image in frame_info
  bool RunPreproc(Preproc* preproc, const FrameInfoPtr& frame_info);

  /// @brief Copy cpu_input_bufs_ -> device, RunSync, copy -> cpu_output_bufs_
  bool RunInference();

  /// @brief Run postproc: reads cpu_output_bufs_, writes detections to frame_info
  bool RunPostproc(Postproc* postproc, const FrameInfoPtr& frame_info);

  /// @brief Extract ValidatorDetection list from frame_info's InferObjs
  std::vector<ValidatorDetection> ExtractDetections(const FrameInfoPtr& frame_info);

  static std::string DataTypeToString(DataType dt);
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_MODEL_VALIDATOR_HPP_
