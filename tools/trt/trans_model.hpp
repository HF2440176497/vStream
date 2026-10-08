

#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace TRT {

enum class ModelSourceType { ONNX, ONNXDATA };

enum class CompileOutputType { File, Memory };


struct ProfileShape {
  nvinfer1::Dims min;
  nvinfer1::Dims opt;
  nvinfer1::Dims max;
};


struct CompileConfig {
  size_t max_workspace_size = 2ULL << 30;

  // TensorRT 日志等级：只打印不高于该等级的日志（数值越小越严重），默认只保留 WARNING/ERROR
  nvinfer1::ILogger::Severity log_severity = nvinfer1::ILogger::Severity::kWARNING;

  // 动态输入的 optimization profile，按输入名配置 min/opt/max。
  // 全静态模型会忽略此配置
  std::map<std::string, ProfileShape> profile_shapes;

  // 是否允许 TensorRT 对 FP32 层使用 TF32（TensorFloat-32）。
  // TensorRT 自身默认开启，关闭后 FP32 层走 FP32，数值更精确但可能更慢。
  // 仅对 compute capability >= 8.0 的设备（Ampere/Ada/Hopper/Blackwell）有意义。
  // 默认 true = 与 TensorRT 原生默认一致，保持既有行为不变。
  bool tf32 = true;

  // 是否允许使用 FP16 kernel。默认关闭。"许可"而非"强制"。
  bool fp16 = false;

  // 是否以 strongly-typed 网络构建（对应 TensorRT 的 kSTRONGLY_TYPED）。
  // 主要用于 QDQ / 量化模型：要求所有张量显式声明精度，builder 不做隐式降精度。
  // 注意：TensorRT 11.0 起只支持 strongly-typed 网络，该开关届时将成为默认行为。
  bool strict_qdq = true;
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

  CompileOutputType    type_;
  std::string          file_;
};


/**
 * @param source 模型源 (ONNX 文件路径或内存数据)
 * @param saveto 输出配置 (File: 写入磁盘; Memory: 不写盘, 由返回值返回)
 * @param config 编译配置
 * @return 序列化后的 engine 字节序列; 失败时返回空 vector
 */
std::vector<uint8_t> compile(const ModelSource& source, const CompileOutput& saveto,
                             const CompileConfig& config = CompileConfig{});


}  // namespace TRT