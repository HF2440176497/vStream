// =============================================================================
// ov_hello.cpp
//
// OpenVINO 2026.x C++ 最小示例：环境自检 -> IR + model cache 加载
// -> 同步/异步推理。只链接 openvino::runtime
//
// 部署侧只读 IR（.xml + .bin）：ONNX -> IR 由 ov_trans_model 离线完成。
// 加载采用官方主推的 IR + ov::cache_dir：
//   默认：先查 EXPORT_IMPORT 能力，支持才开 cache_dir（默认目录 = IR 同目录/ov_cache）
//   --cache-dir <dir>  指定缓存目录（与离线预热保持一致）
//   --no-cache         关闭缓存，用于对比首次编译耗时
//
// 用法：
//   ov_hello <model.xml> [device=CPU] [iters=10]
//            [--cache-dir dir] [--no-cache] [--threads N] [--throughput]
//
// 例：
//   ./ov_hello bin/model/openvino/yolov5.xml CPU 10 --cache-dir bin/model/openvino/cache
//
// =============================================================================

#include <openvino/openvino.hpp>

#include "ov_model_load.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string ShapeStr(const ov::Shape& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

size_t ShapeSize(const ov::Shape& s) {
  return std::accumulate(s.begin(), s.end(), size_t{1}, std::multiplies<size_t>{});
}

// 端口信息统一从 CompiledModel 取（官方按路径 compile_model 也返回它）
// note: CompiledModel 类似 Trt::ICudaEngine
void PrintPorts(const ov::CompiledModel& compiled) {
  std::cout << "  inputs:\n";
  for (size_t i = 0; i < compiled.inputs().size(); ++i) {
    const auto& p = compiled.input(i);
    std::cout << "    [" << i << "] name=" << p.get_any_name()
              << "  type=" << p.get_element_type()
              << "  shape=" << ShapeStr(p.get_shape()) << "\n";
  }
  std::cout << "  outputs:\n";
  for (size_t i = 0; i < compiled.outputs().size(); ++i) {
    const auto& p = compiled.output(i);
    std::cout << "    [" << i << "] name=" << p.get_any_name()
              << "  type=" << p.get_element_type()
              << "  shape=" << ShapeStr(p.get_shape()) << "\n";
  }
}

void PrintUsage(const char* exe) {
  std::cout
      << "Usage: " << exe
      << " <model.xml> [device=CPU] [iters=10] "
         "[--cache-dir dir] [--no-cache] [--threads N] [--throughput]\n"
      << "  只接受离线转换得到的 IR (.xml)，同名 .bin 必须同目录\n"
      << "  先用 ov_trans_model <model.onnx> <out.xml> --cache-dir <dir> 转换并预热缓存\n"
      << "  默认检测 EXPORT_IMPORT 能力，支持则启用 model cache（IR 同目录/ov_cache）\n"
      << "  --no-cache : 关闭缓存，用于对比首次 compile_model 耗时\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      PrintUsage(argv[0]);
      return 1;
    }

    // ---------------- 参数解析（保持简单，位置参数 + 少量 flag）----------------
    const std::string model_path = argv[1];  // 必须是离线转换好的 .xml
    std::string device = "CPU";
    int iters = 10;
    std::string cache_dir;
    bool enable_cache = true;
    int threads = 0;
    bool throughput = false;
    int positional = 0;

    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--cache-dir" && i + 1 < argc) {
        cache_dir = argv[++i];
      } else if (a == "--no-cache") {
        enable_cache = false;
      } else if (a == "--threads" && i + 1 < argc) {
        threads = std::stoi(argv[++i]);
      } else if (a == "--throughput") {
        throughput = true;
      } else if (a.rfind("--", 0) == 0) {
        std::cerr << "unknown option: " << a << "\n";
        return 1;
      } else {
        if (positional == 0) device = a;
        if (positional == 1) iters = std::max(1, std::stoi(a));
        ++positional;
      }
    }

    // [0] 环境自检：版本 + 可用设备 + 插件版本
    //     TRT 对照：cudaSetDevice(device_id) + 检查 driver/runtime 版本
    //     注：若 ov::get_openvino_version() 在你的版本里编译不过，删掉这一行即可，
    //     其余流程不依赖它。
    ov::Core core;
    std::cout << "[0] OpenVINO " << ov::get_openvino_version() << "\n";
    auto devices = core.get_available_devices();
    for (const auto& d : devices) {
      std::cout << "    device=" << d;
      for (const auto& kv : core.get_versions(d)) {
        std::cout << "  [" << kv.first << ": " << kv.second << "]";
      }
      std::cout << "\n";
    }
    if (device.find(':') == std::string::npos && device != "AUTO" &&
        std::find(devices.begin(), devices.end(), device) == devices.end()) {
      std::cerr << "[WARN] device '" << device
                << "' 不在 get_available_devices() 列表里，"
                   "请检查 libopenvino-intel-cpu-plugin 是否安装\n";
    }

    // [1]-[4] 部署侧加载：只读 IR，官方主推的 IR + model cache 路径
    //     TRT 对照：deserializeCudaEngine(.engine) + 建 context
    //     ov_model_load.hpp 先查 EXPORT_IMPORT 能力，支持才 set_property(cache_dir)，
    //     再用 compile_model(ir_path, device, config) 直接按路径编译（跳过 read_model）。
    //     缓存由 runtime 自动管理：首次生成 .blob/.cl_cache，之后命中。
    ov::CompiledModel compiled = ov_demo::LoadCompiledModel(
        core, model_path, device, cache_dir, enable_cache, throughput, threads);
    PrintPorts(compiled);
    std::cout << "[2] ONNX -> IR 及输入形状固定已在离线阶段由 ov_trans_model 完成\n"
              << "[3] 部署侧不再解析 ONNX；设备侧优化由 compile_model 完成，"
                 "缓存命中时跳过大部分耗时步骤\n";

    // [5] 创建推理请求 + 用"外部 host 内存"零拷贝包装输入
    //     TRT 对照：createExecutionContext() + setInputTensorAddress(name, ptr)
    //     这一招是集成到 ModelLoader 的关键：RunSync 拿到的 inputs[i].get()
    //     本来就是 host 指针，直接包成 ov::Tensor，不需要任何 H2D 拷贝。
    ov::InferRequest req = compiled.create_infer_request();
    const auto in_port = compiled.input(0);
    const ov::Shape in_shape = in_port.get_shape();  // 动态 shape 会抛异常
    const ov::element::Type in_type = in_port.get_element_type();

    ov::Tensor probe(in_type, in_shape);
    std::vector<uint8_t> host_in(probe.get_byte_size(), 0);
    ov::Tensor input_tensor(in_type, in_shape, host_in.data());
    if (in_type == ov::element::f32) {
      float* p = input_tensor.data<float>();
      std::mt19937 gen(42);
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (size_t i = 0; i < ShapeSize(in_shape); ++i) p[i] = dist(gen);
    }
    req.set_input_tensor(0, input_tensor);
    std::cout << "[5] input tensor " << ShapeStr(in_shape) << " " << in_type
              << ", wrapped external buffer " << host_in.size()
              << " bytes (zero-copy)\n";

    // [6] 同步推理 + 计时 + 打印输出
    //     TRT 对照：enqueueV3(stream) + cudaStreamSynchronize(stream)
    //     OV 的 infer() 内部已包含"入队 + 等待"，没有显式 stream 概念。
    double first_ms = 0.0;
    double sum_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
      auto a = Clock::now();
      req.infer();
      auto b = Clock::now();
      const double ms = ElapsedMs(a, b);
      if (i == 0) first_ms = ms;
      else sum_ms += ms;
    }
    std::cout << "[6] infer x" << iters << ": first=" << std::fixed
              << std::setprecision(2) << first_ms << " ms, avg(rest)="
              << (iters > 1 ? sum_ms / (iters - 1) : 0.0) << " ms\n";

    for (size_t i = 0; i < compiled.outputs().size(); ++i) {
      const ov::Tensor out = req.get_output_tensor(i);
      std::cout << "    output[" << i << "] " << ShapeStr(out.get_shape())
                << " " << out.get_element_type();
      if (out.get_element_type() == ov::element::f32) {
        const float* p = out.data<float>();
        double sum = 0.0;
        for (size_t k = 0; k < out.get_size(); ++k) sum += p[k];
        std::cout << "  first5=[";
        for (size_t k = 0; k < std::min<size_t>(5, out.get_size()); ++k) {
          if (k) std::cout << ",";
          std::cout << p[k];
        }
        std::cout << "]  sum=" << sum;
      }
      std::cout << "\n";
    }

    // [7] 演示输出也写进外部 buffer（集成时可直接把框架 outputs[i] 传进来）
    //     注意：部分插件/设备可能忽略外部输出内存，生产代码建议
    //     "优先 set_output_tensor，失败则 get_output_tensor + memcpy" 兜底。
    {
      const auto out_port = compiled.output(0);
      const ov::Shape out_shape = out_port.get_shape();
      const ov::element::Type out_type = out_port.get_element_type();
      ov::Tensor probe2(out_type, out_shape);
      std::vector<uint8_t> host_out(probe2.get_byte_size(), 0);
      ov::Tensor out_ext(out_type, out_shape, host_out.data());

      ov::InferRequest r2 = compiled.create_infer_request();
      ov::Tensor in_ext(in_type, in_shape, host_in.data());
      r2.set_input_tensor(0, in_ext);
      r2.set_output_tensor(0, out_ext);
      r2.infer();

      float max_diff = -1.0f;
      const ov::Tensor out0 = req.get_output_tensor(0);
      if (out_type == ov::element::f32 && out0.get_size() == out_ext.get_size()) {
        const float* a = out0.data<float>();
        const float* b = out_ext.data<float>();
        max_diff = 0.0f;
        for (size_t k = 0; k < out0.get_size(); ++k) {
          max_diff = std::max(max_diff, std::fabs(a[k] - b[k]));
        }
      }
      std::cout << "[7] set_output_tensor OK, external buffer " << host_out.size()
                << " bytes, max|diff| vs [6] = " << max_diff
                << " (确定性模型应为 0)\n";
    }

    // [8] 异步推理 + 多 request 并发
    //     TRT 对照：多 cudaStream + 多 IExecutionContext（vStream 曾设计后回退）
    //     OV 用多个 InferRequest 表达并发；THROUGHPUT 模式下插件内部自动分流。
    {
      ov::InferRequest r1 = compiled.create_infer_request();
      ov::InferRequest r2 = compiled.create_infer_request();
      std::vector<uint8_t> buf1(host_in), buf2(host_in);
      ov::Tensor t1(in_type, in_shape, buf1.data());
      ov::Tensor t2(in_type, in_shape, buf2.data());
      r1.set_input_tensor(0, t1);
      r2.set_input_tensor(0, t2);
      auto a = Clock::now();
      r1.start_async();
      r2.start_async();
      r1.wait();
      r2.wait();
      std::cout << "[8] 2 concurrent async requests: "
                << ElapsedMs(a, Clock::now()) << " ms  "
                   "(单个 request 未 wait 前不能复用/改写它的输入 buffer)\n";
    }

    std::cout << "\n环境链路 OK。用 --no-cache 再跑一次，对比 [load] compile_model "
                 "耗时，就能看到 model cache 的收益。\n"
                 "下一步用 ov_yolo_demo 跑真实图片，或对照 "
                 "openvino_integration_guide.md 集成到 ModelLoader。\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n"
              << "  排查顺序：\n"
              << "   1) apt list --installed | grep openvino  (dev / cpu-plugin / frontend)\n"
              << "   2) 只接受 .xml (IR)；先用 ov_trans_model 把 .onnx 离线转换\n"
              << "   3) .xml 必须与同名 .bin 同目录；cache_dir 需可写\n"
              << "   4) ldd ./ov_hello | grep 'not found'\n";
    return 1;
  }
}
