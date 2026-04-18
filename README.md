# vStream

> 高性能视频结构化分析框架，采用模块化流水线设计。

## ✨ 特性

| 特性 | 描述 |
|:---:|:---|
| 🧩 模块化架构 | 基于流水线的插件式设计 |
| 📝 JSON 配置 | 动态构建处理流水线 |
| ⚡ 多线程并行 | 提高处理效率 |
| 🚀 CUDA 加速 | 可选的 GPU 加速支持 |

---

## 📦 依赖

### 必需依赖

`gflags` · `glog` · `opencv`

### 已包含的第三方库

| 库 | 版本 |
|:---|:---:|
| nlohmann/json | 3.11.3 |
| googletest | 1.15.2 |
| pybind11 | 3.0.3 |

---

## 🔨 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_MODULES=ON -DBUILD_TESTS=ON
make -j$(nproc)
```

---

## 📖 使用

### 1. JSON 配置

```json
{
  "profiler_config": { "enable_profile": true },
  "datasource": {
    "class_name": "cnstream::DataSource",
    "next_modules": ["inference"],
    "custom_params": { 
      "image_path": "test.png" 
    }
  },
  "inference": {
    "class_name": "cnstream::Inference",
    "next_modules": [],
    "custom_params": { 
      "model_path": "./models/detect.mlu"
    }
  }
}
```

### 2. C++ 示例

```cpp
#include "cnstream_pipeline.hpp"

int main() {
    cnstream::Pipeline pipeline("my_pipeline");
    pipeline.BuildPipelineByJSONFile("config.json");
    pipeline.Start();
    // ...
    pipeline.Stop();
    return 0;
}
```

### 3. 自定义模块

```cpp
#include "cnstream_module.hpp"

class MyModule : public cnstream::Module, public cnstream::ModuleCreator<MyModule> {
 public:
    explicit MyModule(const std::string& name) : cnstream::Module(name) {}
    bool Open(cnstream::ModuleParamSet params) override { return true; }
    void Close() override {}
    int Process(std::shared_ptr<cnstream::FrameInfo> data) override { return 0; }
};

REGISTER_MODULE(MyModule);
```

---

## 📁 项目结构

```
vStream/
├── framework/core/       # 核心框架 (Pipeline, Module, FrameInfo, Connector)
├── modules/              # 功能模块 (source, util, unittest)
├── 3rdparty/             # 第三方库 (libyuv, googletest, json)
├── tools/                # 工具程序
├── samples/              # 示例程序
└── CMakeLists.txt
```

---

## ✅ 已完成

- [x] Pipeline 流水线管理
- [x] JSON 配置解析
- [x] Module 模块基类
- [x] Connector 模块连接器
- [x] EventBus 事件总线
- [x] SourceModule 数据源模块
- [x] Inference 推理模块

## 🚧 待开发
- [ ] Python 绑定
- [ ] Encoder/Render 模块

## 📌 适配模型

### CUDA 平台
- [x] YOLOv5 检测
- [x] YOLOv8 检测



---

## 📚 参考

[CNStream](https://github.com/Cambricon/CNStream)
