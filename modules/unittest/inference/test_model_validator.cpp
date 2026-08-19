
#include <cstdlib>
#include <string>
#include <algorithm>

#include "base.hpp"
#include "model_validator.hpp"
#include "cnstream_frame_va.hpp"

using namespace cnstream;


static std::string model_path = "./model/20260625/yolov8s_tracing_static_b1_pre.engine";
static std::string device_type = "cuda";
static std::string image_path = "./image.png";

// Test: DataFrame::SetImage functionality
TEST(ModelValidator, DataFrameSetImage) {
  DataFramePtr frame = std::make_shared<DataFrame>();

  cv::Mat img(480, 640, CV_8UC3, cv::Scalar(1, 2, 3));
  frame->SetImage(img);

  EXPECT_EQ(frame->GetWidth(), 640);
  EXPECT_EQ(frame->GetHeight(), 480);
  EXPECT_TRUE(frame->HasImage());

  cv::Mat recovered = frame->GetImage();
  EXPECT_FALSE(recovered.empty());
  EXPECT_EQ(recovered.cols, 640);
  EXPECT_EQ(recovered.rows, 480);
}

TEST(ModelValidator, LoadNonExistentModel) {
  ModelValidator v(model_path, device_type, 0);
  bool ok = v.Load();
  ASSERT_TRUE(ok);
  ASSERT_TRUE(v.IsLoaded());
}


/* ============================================================ *
 *  Tests that REQUIRE a real model — skipped if env var not set.
 *
 *  Set environment variable to enable:
 *    VSTREAM_TEST_MODEL_PATH=/path/to/yolov8s.engine
 *    VSTREAM_TEST_DEVICE=cuda            (optional, default: cuda)
 *    VSTREAM_TEST_POSTPROC_CONFIG=/path/to/yolo_coco.json  (optional)
 *    VSTREAM_TEST_IMAGE=/path/to/test.jpg (optional, default: synthetic)
 * ============================================================ */

static std::string GetEnv(const char* name, const std::string& default_val = "") {
  const char* val = std::getenv(name);
  return val ? std::string(val) : default_val;
}

static bool HasTestModel() {
  return !GetEnv("VSTREAM_TEST_MODEL_PATH").empty();
}

// Test: Load real model and inspect info
TEST(ModelValidator, RealModelLoadAndGetInfo) {
  if (!HasTestModel()) GTEST_SKIP() << "Set VSTREAM_TEST_MODEL_PATH to run this test";

  std::string model_path = GetEnv("VSTREAM_TEST_MODEL_PATH");
  std::string device = GetEnv("VSTREAM_TEST_DEVICE", "cuda");

  ModelValidator v(model_path, device, 0);
  ASSERT_TRUE(v.Load()) << "Failed to load model: " << model_path;
  EXPECT_TRUE(v.IsLoaded());

  ModelInfo info = v.GetModelInfo();
  EXPECT_GT(info.inputs.size(), 0u);
  EXPECT_GT(info.outputs.size(), 0u);
  EXPECT_GT(info.batch_size, 0);
  EXPECT_GT(info.width, 0);
  EXPECT_GT(info.height, 0);
  EXPECT_EQ(info.channel, 3);

  // Print model info for debugging
  printf("  Model: %s\n", info.model_path.c_str());
  printf("  Device: %s, id=%d\n", info.device_type.c_str(), info.device_id);
  printf("  Batch: %d, Input: %dx%dx%d\n", info.batch_size, info.width, info.height, info.channel);
  for (size_t i = 0; i < info.inputs.size(); ++i) {
    printf("  Input[%zu]: name=%s, shape=[%d,%d,%d,%d], dtype=%s\n",
           i, info.inputs[i].name.c_str(),
           info.inputs[i].shape[0], info.inputs[i].shape[1],
           info.inputs[i].shape[2], info.inputs[i].shape[3],
           info.inputs[i].dtype.c_str());
  }
  for (size_t i = 0; i < info.outputs.size(); ++i) {
    printf("  Output[%zu]: name=%s, shape=[%d,%d,%d,%d], dtype=%s\n",
           i, info.outputs[i].name.c_str(),
           info.outputs[i].shape[0], info.outputs[i].shape[1],
           info.outputs[i].shape[2], info.outputs[i].shape[3],
           info.outputs[i].dtype.c_str());
  }
}

/**
 * @brief 采用随机数据进行模型推理（不包含前处理和后处理）
 */
TEST(ModelValidator, RealModelRawInfer) {
  if (!HasTestModel()) GTEST_SKIP() << "Set VSTREAM_TEST_MODEL_PATH to run this test";

  std::string model_path = GetEnv("VSTREAM_TEST_MODEL_PATH", model_path);
  std::string device = GetEnv("VSTREAM_TEST_DEVICE", "cuda");

  ModelValidator v(model_path, device, 0);
  ASSERT_TRUE(v.Load());

  ModelInfo info = v.GetModelInfo();

  // Build random inputs matching tensor shapes
  std::vector<std::vector<float>> inputs;
  for (size_t i = 0; i < info.inputs.size(); ++i) {
    size_t count = 1;
    for (int d : info.inputs[i].shape) count *= d;
    std::vector<float> data(count);
    for (size_t j = 0; j < count; ++j) {
      data[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    inputs.push_back(std::move(data));
  }

  auto outputs = v.Infer(inputs);
  EXPECT_EQ(outputs.size(), info.outputs.size());

  // Check output sizes
  for (size_t i = 0; i < outputs.size(); ++i) {
    size_t expected = 1;
    for (int d : info.outputs[i].shape) expected *= d;
    EXPECT_EQ(outputs[i].size(), expected);

    for (float val : outputs[i]) {
      EXPECT_FALSE(std::isnan(val)) << "NaN in output[" << i << "]";
      EXPECT_FALSE(std::isinf(val)) << "Inf in output[" << i << "]";
    }

    if (!outputs[i].empty()) {
      float min_v = *std::min_element(outputs[i].begin(), outputs[i].end());
      float max_v = *std::max_element(outputs[i].begin(), outputs[i].end());
      double sum = std::accumulate(outputs[i].begin(), outputs[i].end(), 0.0);
      double mean = sum / outputs[i].size();
      printf("  Output[%zu]: min=%.4f, max=%.4f, mean=%.4f, size=%zu\n",
             i, min_v, max_v, mean, outputs[i].size());
    }
  }
}


TEST(ModelValidator, RealModelRunE2E) {
  if (!HasTestModel()) GTEST_SKIP() << "Set VSTREAM_TEST_MODEL_PATH to run this test";

  std::string model_path = GetEnv("VSTREAM_TEST_MODEL_PATH", model_path);
  std::string device = GetEnv("VSTREAM_TEST_DEVICE", "cuda");
  std::string postproc_config = GetEnv("VSTREAM_TEST_POSTPROC_CONFIG");
  std::string image_path = GetEnv("VSTREAM_TEST_IMAGE", image_path);

  ModelValidator v(model_path, device, 0);
  ASSERT_TRUE(v.Load());

  // Prepare test image
  cv::Mat img;
  if (!image_path.empty()) {
    img = cv::imread(image_path);
    ASSERT_FALSE(img.empty()) << "Failed to read image: " << image_path;
  } else {
    // Synthetic image: 1280x720 BGR
    img = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(50, 100, 150));
    cv::rectangle(img, cv::Rect(100, 100, 200, 200), cv::Scalar(0, 255, 0), 3);
  }

  // Prepare postproc params
  std::map<std::string, std::string> postproc_params;
  if (!postproc_config.empty()) {
    std::string key_config_file = "config_file";
    postproc_params[key_config_file] = postproc_config;
  }

  E2EResult r = v.RunE2E(img, "Pre_YOLO_CPU_v2", "Post_YOLOv8_CPU_v2", {}, postproc_params);

  if (!r.error.empty()) {
    printf("  E2E error: %s\n", r.error.c_str());
    if (postproc_config.empty()) {
      printf("  (postproc config not provided, skipping detection check)\n");
      return;
    }
  }

  printf("  E2E: %zu detections, %.2f ms\n", r.detections.size(), r.latency_ms);
  for (size_t i = 0; i < r.detections.size() && i < 10; ++i) {
    const auto& d = r.detections[i];
    printf("    det[%zu]: class=%d, score=%.4f, bbox=[%.3f, %.3f, %.3f, %.3f]\n",
           i, d.class_id, d.score, d.x, d.y, d.w, d.h);
  }

  EXPECT_GE(r.latency_ms, 0.0);
}

// Test: Benchmark performance
TEST(ModelValidator, RealModelBenchmark) {
  if (!HasTestModel()) GTEST_SKIP() << "Set VSTREAM_TEST_MODEL_PATH to run this test";

  std::string model_path = GetEnv("VSTREAM_TEST_MODEL_PATH", model_path);
  std::string device = GetEnv("VSTREAM_TEST_DEVICE", "cuda");
  std::string postproc_config = GetEnv("VSTREAM_TEST_POSTPROC_CONFIG");

  ModelValidator v(model_path, device, 0);
  ASSERT_TRUE(v.Load());

  cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(50, 100, 150));

  std::map<std::string, std::string> postproc_params;
  if (!postproc_config.empty()) {
    std::string key_config_file = "config_file";
    postproc_params[key_config_file] = postproc_config;
  }

  auto results = v.Benchmark(img, "Pre_YOLO_CPU_v2", "Post_YOLOv8_CPU_v2",
                              {}, postproc_params,
                              3, 10, {1});

  if (!results.empty()) {
    const auto& r = results[0];
    printf("  Benchmark bs=%d: avg=%.2fms, min=%.2fms, max=%.2fms, p99=%.2fms, fps=%.1f, errors=%d\n",
           r.batch_size, r.avg_ms, r.min_ms, r.max_ms, r.p99_ms, r.fps, r.error_count);
    EXPECT_GT(r.avg_ms, 0.0);
    EXPECT_GE(r.max_ms, r.min_ms);
  }
}


