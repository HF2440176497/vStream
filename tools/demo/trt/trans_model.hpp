

#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace TRT {

enum class Mode { FP32, FP16, INT8 };

enum class ModelSourceType { ONNX, ONNXDATA };

enum class CompileOutputType { File, Memory };


struct ProfileShape {
  nvinfer1::Dims min;
  nvinfer1::Dims opt;
  nvinfer1::Dims max;
};


struct CompileConfig {
  size_t max_workspace_size = 2ULL << 30;

  bool dynamic_batch = true;  // 仅支持 batch_size 是动态的
  int  max_batch_size = 8;
  int  opt_batch_size = 4;
  int  min_batch_size = 1;

  // specific profile shapes
  std::map<std::string, ProfileShape> profile_shapes;

  bool strict_qdq = true;
  int  min_compute_capability = 75;

  std::string calibration_cache_file;
  std::vector<std::vector<uint8_t>> calibration_data;
};

class ModelSource {
 public:
  ModelSource(const char* onnxmodel);
  ModelSource(const std::string& onnxmodel);
  ModelSource(const void* data, size_t size);  // 内存中的 ONNX

  ModelSourceType type() const;
  std::string     descript() const;
  std::string     onnxmodel() const;
  const void*     onnx_data() const;
  size_t          onnx_data_size() const;

 private:
  ModelSourceType type_;
  std::string     onnxmodel_;
  const void*     onnx_data_ = nullptr;
  size_t          onnx_data_size_ = 0;
};

class CompileOutput {
 public:
  CompileOutput(CompileOutputType type);
  CompileOutput(const std::string& file);
  CompileOutput(const char* file);

  void set_data(const std::vector<uint8_t>& data);
  void set_data(std::vector<uint8_t>&& data);

  CompileOutputType    type_;
  std::string          file_;
  std::vector<uint8_t> data_;
};


bool compile(Mode mode, const ModelSource& source, const CompileOutput& saveto,
             const CompileConfig& config = CompileConfig{});


}  // namespace TRT