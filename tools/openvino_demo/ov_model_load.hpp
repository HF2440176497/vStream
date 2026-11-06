// =============================================================================
// ov_model_load.hpp
//
// 采用 IR + Model Caching 路径：
//   ov::Core core;
//   core.set_property(ov::cache_dir("/path/to/cache/dir"));   // Step 1b
//   auto compiled = core.compile_model(modelPath, device, config);  // Step 2
//
// 关键点：
//   1. 先用 ov::device::capabilities 检查设备是否支持 EXPORT_IMPORT；
//      支持才开启 cache_dir（官方：不支持时开启缓存无效果、不报错）。
//   2. 直接按 IR 文件路径 compile_model，跳过 read_model —— 官方 "Use optimized
//      methods" 明确指出启用缓存时这样总加载时间更短。
//   3. 缓存由 runtime 自动管理：首次 compile 稍慢并落 .blob(CPU)/.cl_cache(GPU)，
//      之后命中。应用侧不需要、也不应该手动 export_model/import_model。
//
// ONNX -> IR 由 ov_trans_model 离线完成，部署侧只读 .xml（同名 .bin 必须同目录）。
// =============================================================================

#ifndef VSTREAM_TOOLS_OPENVINO_DEMO_OV_MODEL_LOAD_HPP_
#define VSTREAM_TOOLS_OPENVINO_DEMO_OV_MODEL_LOAD_HPP_

#include <openvino/openvino.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ov_demo {

inline bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline double ElapsedMs(std::chrono::steady_clock::time_point a,
                        std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// 未显式指定 --cache-dir 时，默认放到 IR 同目录下的 ov_cache/
inline std::string DefaultCacheDir(const std::string& ir_path) {
  const size_t slash = ir_path.find_last_of('/');
  const std::string dir = (slash == std::string::npos) ? "." : ir_path.substr(0, slash);
  return dir + "/ov_cache";
}

// 官方 Advanced Examples：
//   std::vector<std::string> caps = core.get_property(deviceName, ov::device::capabilities);
//   bool cachingSupported = std::find(caps.begin(), caps.end(), EXPORT_IMPORT) != caps.end();
// AUTO / MULTI 这类复合设备可能不支持该查询，异常时按"不支持"处理。
inline bool IsModelCachingSupported(ov::Core& core, const std::string& device) {
  try {
    const auto caps = core.get_property(device, ov::device::capabilities);
    return std::find(caps.begin(), caps.end(), std::string("EXPORT_IMPORT")) != caps.end();
  } catch (const std::exception& e) {
    std::cerr << "[load] capability query failed for '" << device
              << "': " << e.what() << " -> treat caching as unsupported\n";
    return false;
  }
}

// ir_path       : 离线转换得到的 .xml
// cache_dir     : 空表示用 DefaultCacheDir(ir_path)
// enable_cache  : false 对应 --no-cache，用于对比性能
// throughput    : 作为 compile_model 的 config 属性传入
inline ov::CompiledModel LoadCompiledModel(ov::Core& core,
                                           const std::string& ir_path,
                                           const std::string& device,
                                           const std::string& cache_dir,
                                           bool enable_cache,
                                           bool throughput,
                                           int threads) {
  using Clock = std::chrono::steady_clock;

  if (!EndsWith(ir_path, ".xml")) {
    throw std::runtime_error(
        "部署侧只读 IR (.xml)。请先用 ov_trans_model 离线转换：\n"
        "  ov_trans_model <model.onnx> <model.xml> --shape 1,3,640,640 "
        "--cache-dir <cache_dir>\n"
        "当前传入: " + ir_path);
  }

  if (threads > 0 && device == "CPU") {
    core.set_property("CPU", ov::inference_num_threads(threads));
  }

  bool caching_on = false;
  const std::string dir = cache_dir.empty() ? DefaultCacheDir(ir_path) : cache_dir;
  if (enable_cache) {
    if (IsModelCachingSupported(core, device)) {
      try {
        core.set_property(ov::cache_dir(dir));
        caching_on = true;
        std::cout << "[load] model cache ENABLED (device '" << device
                  << "' supports EXPORT_IMPORT), dir=" << dir << "\n";
      } catch (const std::exception& e) {
        std::cerr << "[WARN] set_property(ov::cache_dir) 失败: " << e.what()
                  << "\n       目录不可写？禁用缓存继续加载\n";
      }
    } else {
      std::cout << "[load] device '" << device
                << "' 不支持 EXPORT_IMPORT，cache_dir 无效果，跳过（官方：不会报错）\n";
    }
  } else {
    std::cout << "[load] model cache DISABLED (--no-cache)\n";
  }

  // 官方 "Use optimized methods"：直接按路径编译，跳过 read_model。
  // config 与官方示例一致；属性通过 AnyMap::insert 加入。
  ov::AnyMap config;
  if (throughput) {
    config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  }

  const auto t = Clock::now();
  ov::CompiledModel compiled = core.compile_model(ir_path, device, config);
  std::cout << "[load] compile_model('" << ir_path << "', '" << device
            << "') took " << ElapsedMs(t, Clock::now()) << " ms"
            << (caching_on
                    ? "  <- 首次稍慢(生成缓存)，之后命中 .blob/.cl_cache\n"
                    : "  <- 未启用缓存，每次进程启动都要付这笔\n");
  return compiled;
}

}  // namespace ov_demo

#endif  // VSTREAM_TOOLS_OPENVINO_DEMO_OV_MODEL_LOAD_HPP_
