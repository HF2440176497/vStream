// =============================================================================
// ov_trans_model.cpp
//
// 离线模型转换工具
// 编译成可执行文件 ov_trans_model
//
// 采用 OpenVINO 官方主推的 IR + Model Caching 方式：
//   core.set_property(ov::cache_dir(dir));
//   core.compile_model(ir_path, device, config);
//
// 本工具做两件事：
//   1) ONNX -> IR (.xml/.bin)：read_model + 可选 reshape + ov::serialize
//   2) 可选：预热 model cache —— 用同一个 IR 路径做一次 compile_model，
//      让插件把 .blob(CPU)/.cl_cache(GPU) 落到 --cache-dir。
//
// --shape 的三种写法（可重复指定，支持多输入模型）：
//   --shape 1,3,640,640             单输入快捷写法（等价于 reshape(PartialShape)）
//   --shape 0=1,3,640,640           按输入 index 指定
//   --shape image=1,3,640,640       按 tensor name 指定
//   多个不带 key 的 --shape 按出现顺序对应 input 0,1,2...
// reshape 会更新输入形状并沿所有中间层做形状传播
//
// 例：
//   ov_trans_model bin/onnx/yolov5_sheath_detection.onnx \
//                  bin/model/openvino/yolov5.xml --shape 1,3,640,640 \
//                  --device CPU --cache-dir bin/model/openvino/cache
//
//   多输入（例如 OCR 的 data + seq_len）：
//   ov_trans_model ocr.onnx ocr.xml \
//                  --shape data=3,150,200,1 --shape seq_len=3
// =============================================================================

#include <openvino/openvino.hpp>
#include <openvino/pass/serialize.hpp>  // ov::serialize

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct ShapeSpec {
  bool has_key = false;         // true: key 是 index 或 tensor name
  std::string key;
  ov::PartialShape shape;
};

struct Options {
  std::string input;
  std::string out_xml;
  std::vector<ShapeSpec> shapes;
  std::string device = "CPU";
  std::string cache_dir;
  int threads = 0;
  bool verify = true;
};

double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// "1,3,640,640" / "3" -> ov::PartialShape（当前只接受静态整数维度）
ov::PartialShape ParseShape(const std::string& text) {
  std::vector<int64_t> dims;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t end = text.find(',', start);
    const std::string token = text.substr(
        start, end == std::string::npos ? std::string::npos : end - start);
    if (!token.empty()) dims.push_back(std::stoll(token));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (dims.empty()) throw std::runtime_error("--shape 为空: " + text);
  ov::PartialShape ps = ov::PartialShape::dynamic(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) ps[i] = dims[i];
  return ps;
}

std::string ShapeStr(const ov::PartialShape& ps) {
  return ps.to_string();
}

std::string ShapeStr(const ov::Shape& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

std::string NamesStr(const ov::Output<ov::Node>& port) {
  std::string out;
  for (const auto& n : port.get_names()) {
    if (!out.empty()) out += ",";
    out += n;
  }
  return out.empty() ? "<unnamed>" : out;
}

// 打印输入清单：index / 全部 tensor name / dtype / partial shape
void PrintInputs(const std::vector<ov::Output<ov::Node>>& inputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& p = inputs[i];
    std::cout << "    input [" << i << "] name=" << NamesStr(p)
              << "  " << p.get_element_type()
              << "  " << ShapeStr(p.get_partial_shape()) << "\n";
  }
}

void PrintPorts(const std::shared_ptr<ov::Model>& model) {
  PrintInputs(model->inputs());
  for (size_t i = 0; i < model->outputs().size(); ++i) {
    const auto& p = model->output(i);
    std::cout << "    output[" << i << "] name=" << NamesStr(p)
              << "  " << p.get_element_type()
              << "  " << ShapeStr(p.get_partial_shape()) << "\n";
  }
}

// key 为纯数字时按 index，否则按 tensor name 查找（get_names() 覆盖多别名）
size_t ResolveInputIndex(const std::vector<ov::Output<ov::Node>>& inputs,
                         const std::string& key) {
  const bool digits = !key.empty() &&
      std::all_of(key.begin(), key.end(),
                  [](unsigned char c) { return std::isdigit(c) != 0; });
  if (digits) return static_cast<size_t>(std::stoull(key));

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto names = inputs[i].get_names();
    if (names.find(key) != names.end()) return i;
  }
  std::cerr << "可用输入列表（用 index 或 name 指定 --shape）：\n";
  PrintInputs(inputs);
  throw std::runtime_error("找不到输入 '" + key + "'");
}

// reshape 不能改变 rank（维度个数），提前给出比底层异常更清楚的错误
void CheckRank(const ov::Output<ov::Node>& port,
               const ov::PartialShape& new_shape, size_t idx) {
  const auto old = port.get_partial_shape();
  if (old.rank().is_static() && new_shape.rank().is_static() &&
      old.rank().get_length() != new_shape.rank().get_length()) {
    throw std::runtime_error(
        "input [" + std::to_string(idx) + "] rank 不一致：原 "
        + ShapeStr(old) + " -> 新 " + ShapeStr(new_shape)
        + "。reshape 只能改各维的值，不能改 rank。");
  }
}

// 官方三种多输入写法里选了 std::map<size_t, ov::PartialShape>（按 index）：
// 名字在 ResolveInputIndex 里先解析成 index，这样即使某个输入没有名字也能工作。
void ApplyReshape(const std::shared_ptr<ov::Model>& model,
                  const std::vector<ShapeSpec>& specs) {
  const auto inputs = model->inputs();

  if (specs.empty()) {
    if (model->is_dynamic()) {
      std::cerr << "[WARN] 模型仍含动态维度。CPU 插件在静态形状下优化更充分，"
                   "建议用 --shape 固定（多输入可多次指定）。\n";
    }
    return;
  }

  if (specs.size() == 1 && !specs[0].has_key) {
    // 单输入快捷重载 model->reshape(PartialShape)：要求恰好 1 个输入
    if (inputs.size() != 1) {
      std::cerr << "当前模型有 " << inputs.size() << " 个输入：\n";
      PrintInputs(inputs);
      throw std::runtime_error(
          "不带 key 的 --shape 是【单输入】快捷写法；多输入请用 "
          "--shape <index>=dims 或 --shape <tensor_name>=dims（可重复指定）");
    }
    CheckRank(inputs[0], specs[0].shape, 0);
    model->reshape(specs[0].shape);
  } else {
    std::map<size_t, ov::PartialShape> idx_to_shape;
    size_t auto_index = 0;
    for (const auto& spec : specs) {
      const size_t idx = spec.has_key ? ResolveInputIndex(inputs, spec.key)
                                     : auto_index++;
      if (idx >= inputs.size()) {
        throw std::runtime_error(
            "输入 index " + std::to_string(idx) + " 越界，模型只有 "
            + std::to_string(inputs.size()) + " 个输入");
      }
      CheckRank(inputs[idx], spec.shape, idx);
      idx_to_shape[idx] = spec.shape;  // 未指定的输入保持原状
    }
    model->reshape(idx_to_shape);
  }

  std::cout << "[2] reshape 完成，形状传播后的输入：\n";
  PrintInputs(model->inputs());
  for (size_t i = 0; i < model->inputs().size(); ++i) {
    if (model->input(i).get_partial_shape().is_dynamic()) {
      std::cerr << "[WARN] input [" << i << "] 仍含动态维度：本次 --shape 未覆盖它；"
                   "如推理间形状固定，建议补一条 --shape " << i << "=...\n";
    }
  }
}

// 官方 Advanced Examples：只有支持 EXPORT_IMPORT 的设备，cache_dir 才有效
bool IsModelCachingSupported(ov::Core& core, const std::string& device) {
  try {
    const auto caps = core.get_property(device, ov::device::capabilities);
    return std::find(caps.begin(), caps.end(), std::string("EXPORT_IMPORT")) != caps.end();
  } catch (const std::exception& e) {
    std::cerr << "[WARN] capability query failed for '" << device
              << "': " << e.what() << " -> treat caching as unsupported\n";
    return false;
  }
}

size_t FileSize(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return 0;
  return static_cast<size_t>(f.tellg());
}

void PrintUsage(const char* exe) {
  std::cout
      << "Usage: " << exe << " <model.onnx|model.xml> <out.xml> [options]\n"
      << "  --shape <dims>           单输入快捷：1,3,640,640\n"
      << "  --shape <idx>=<dims>     按输入 index：0=1,3,640,640\n"
      << "  --shape <name>=<dims>    按 tensor name：image=1,3,640,640\n"
      << "                           （可重复；多个不带 key 的按顺序对应 input 0,1,...）\n"
      << "  --device CPU             目标设备\n"
      << "  --threads N              离线编译线程数\n"
      << "  --cache-dir <dir>        预热 model cache（runtime 自动落 .blob/.cl_cache）\n"
      << "  --no-verify              跳过转换后的一次 warmup 推理\n"
      << "先不带 --shape 跑一次，看 [1] 打印的 index/name/shape 再决定怎么写。\n";
}

bool ParseArgs(int argc, char** argv, Options* o) {
  if (argc < 3) return false;
  o->input = argv[1];
  o->out_xml = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--shape" && i + 1 < argc) {
      const std::string v = argv[++i];
      ShapeSpec spec;
      const size_t eq = v.find('=');
      if (eq != std::string::npos) {
        spec.has_key = true;
        spec.key = v.substr(0, eq);
        spec.shape = ParseShape(v.substr(eq + 1));
      } else {
        spec.shape = ParseShape(v);
      }
      o->shapes.push_back(spec);
    } else if (a == "--device" && i + 1 < argc) {
      o->device = argv[++i];
    } else if (a == "--threads" && i + 1 < argc) {
      o->threads = std::stoi(argv[++i]);
    } else if (a == "--cache-dir" && i + 1 < argc) {
      o->cache_dir = argv[++i];
    } else if (a == "--no-verify") {
      o->verify = false;
    } else {
      std::cerr << "unknown option: " << a << "\n";
      return false;
    }
  }
  if (!EndsWith(o->out_xml, ".xml")) o->out_xml += ".xml";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opt;
    if (!ParseArgs(argc, argv, &opt)) {
      PrintUsage(argv[0]);
      return 1;
    }

    const std::string out_bin = opt.out_xml.substr(0, opt.out_xml.size() - 4) + ".bin";

    ov::Core core;
    std::cout << "[0] OpenVINO " << ov::get_openvino_version()
              << "  target device=" << opt.device << "\n";

    // ---- 1. 读源模型（ONNX 走 ONNX frontend，IR 走 IR frontend）----------
    auto t0 = Clock::now();
    std::shared_ptr<ov::Model> model = core.read_model(opt.input);
    std::cout << "[1] read_model('" << opt.input << "') " << Ms(t0, Clock::now())
              << " ms\n";
    PrintPorts(model);

    // ---- 2. 离线固定形状：单输入用快捷重载，多输入用 index map ----------
    ApplyReshape(model, opt.shapes);

    // ---- 3. 序列化 IR：部署侧唯一必须带的模型文件 -----------------------
    const auto t3 = Clock::now();
    ov::serialize(model, opt.out_xml, out_bin);
    std::cout << "[3] IR saved: " << opt.out_xml << " (" << FileSize(opt.out_xml)
              << " B) + " << out_bin << " (" << FileSize(out_bin) << " B), took "
              << Ms(t3, Clock::now()) << " ms\n";

    const bool need_compile = !opt.cache_dir.empty() || opt.verify;
    if (!need_compile) {
      std::cout << "\n只产出 IR。部署侧首次 compile_model 会自行生成缓存；"
                   "建议加 --cache-dir 在构建阶段预热。\n";
      return 0;
    }

    // ---- 4. 先检查设备是否支持 model cache，支持才开启 -------------------
    bool caching_on = false;
    if (!opt.cache_dir.empty()) {
      if (IsModelCachingSupported(core, opt.device)) {
        core.set_property(ov::cache_dir(opt.cache_dir));
        caching_on = true;
        std::cout << "[4] model cache ENABLED, cache_dir = " << opt.cache_dir
                  << " (CPU 自动落 .blob / GPU 落 .cl_cache)\n";
      } else {
        std::cerr << "[WARN] 设备 '" << opt.device
                  << "' 不支持 EXPORT_IMPORT，cache_dir 无效果（官方：不报错）；"
                     "仍然继续编译与校验\n";
      }
    }
    if (opt.threads > 0 && opt.device == "CPU") {
      core.set_property("CPU", ov::inference_num_threads(opt.threads));
    }

    // ---- 5. 用 IR 路径编译一次：预热缓存 + 校验 -------------------------
    // 必须用 IR 路径（与部署侧完全相同的调用），缓存 key 才会一致。
    ov::AnyMap config;
    const auto t5 = Clock::now();
    ov::CompiledModel compiled = core.compile_model(opt.out_xml, opt.device, config);
    std::cout << "[5] compile_model('" << opt.out_xml << "', '" << opt.device
              << "') " << Ms(t5, Clock::now()) << " ms"
              << (caching_on ? "  <- 首次生成缓存稍慢，部署侧同目录会命中\n"
                             : "  <- 未启用缓存\n");

    // ---- 6. warmup 校验：所有输入都填一遍，避免多输入模型漏填 -----------
    if (opt.verify) {
      try {
        ov::InferRequest req = compiled.create_infer_request();
        for (size_t i = 0; i < compiled.inputs().size(); ++i) {
          ov::Tensor t = req.get_input_tensor(i);
          if (t.get_element_type() == ov::element::f32) {
            float* p = t.data<float>();
            std::fill_n(p, t.get_size(), 0.5f);
          } else {
            std::memset(t.data(), 0, t.get_byte_size());
          }
        }
        const auto t6 = Clock::now();
        req.infer();
        std::cout << "[6] warmup infer OK, " << Ms(t6, Clock::now()) << " ms";
        const ov::Tensor out = req.get_output_tensor(0);
        std::cout << ", output[0] shape=" << ShapeStr(out.get_shape())
                  << " dtype=" << out.get_element_type() << "\n";
      } catch (const std::exception& e) {
        // 仍有动态输入 / 特殊算子时 warmup 可能失败，但 IR 已经产出，不阻断转换
        std::cerr << "[WARN] warmup infer 失败: " << e.what()
                  << "\n       若因未固定的动态输入，请补全 --shape；"
                     "或用 --no-verify 跳过校验（IR 已生成）\n";
      }
    }

    const std::string deploy_cache =
        opt.cache_dir.empty() ? "<与 IR 同目录的 ov_cache/>" : opt.cache_dir;
    std::cout << "\n部署侧（只读 IR + model cache）：\n"
              << "  ./ov_hello " << opt.out_xml << " " << opt.device
              << " 10 --cache-dir " << deploy_cache << "\n"
              << "  ./ov_yolo_demo " << opt.out_xml << " <image> " << opt.device
              << " 0.25 0.45 --cache-dir " << deploy_cache << "\n"
              << "缓存目录需可写且与离线预热一致；OV 版本/设备/编译配置变化时 "
                 "runtime 会自动重新生成缓存，无需手工管理。\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n"
              << "  排查：\n"
              << "   1) .onnx 需要 libopenvino-onnx-frontend-*；.xml 需要同名 .bin\n"
              << "   2) 先不带 --shape 跑一次，看 [1] 打印的 index/name/shape\n"
              << "   3) 多输入用 --shape <index>=dims 或 <name>=dims，可重复指定\n"
              << "   4) rank 不能改；硬编码 Reshape/Global Pooling 等结构可能不支持 reshape\n"
              << "   5) --cache-dir 目录需存在且可写\n";
    return 1;
  }
}
