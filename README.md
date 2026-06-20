
**高性能视频结构化分析框架** — 模块化流水线设计，支持动态模块注册与加载

![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)
![CMake](https://img.shields.io/badge/CMake-%3E%3D3.13-064F8C?logo=cmake)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)

---

## 目录

- [特性](#特性)
- [依赖](#依赖)
  - [系统依赖](#系统依赖)
  - [已集成的第三方库](#已集成的第三方库)
- [编译](#编译)
  - [0. 安装系统依赖](#0-安装系统依赖)
  - [1. 使用 build.sh 脚本（推荐）](#1-使用-buildsh-脚本推荐)
  - [2. 手动 CMake 构建](#2-手动-cmake-构建)
  - [3. CMake 构建选项](#3-cmake-构建选项)
  - [4. 运行单元测试](#4-运行单元测试)
- [项目结构](#项目结构)
- [核心概念](#核心概念)
  - [Pipeline（流水线）](#pipeline流水线)
  - [Module（模块）](#module模块)
  - [FrameInfo（数据载体）](#frameinfo数据载体)
  - [Connector（连接器）](#connector连接器)
  - [EventBus（事件总线）](#eventbus事件总线)
- [使用](#使用)
  - [1. JSON 配置流水线](#1-json-配置流水线)
  - [2. C++ 示例](#2-c-示例)
  - [3. Python 示例](#3-python-示例)
- [工具](#工具)
- [适配模型](#适配模型)
- [开发计划](#开发计划)
- [参考](#参考)

---

## 特性

- **模块化架构**  
  基于流水线的插件式设计，支持动态模块注册与加载

- **JSON 配置**  
  通过 JSON 声明式构建处理流水线，无需重新编译

- **多线程并行**  
  模块级并行 + 数据并行，充分利用多核 CPU

- **CUDA 加速**  
  可选的 GPU 加速支持，含 TensorRT 推理后端

- **Python 绑定**  
  完整的 Python API，支持用 Python 编写自定义模块

- **推理引擎**  
  内置推理服务，支持批量推理、超时控制、多种前后处理

- **数据源支持**  
  图片源、视频源、流式推送等多种数据输入方式

- **性能剖析**  
  内置 Profiler，支持模块级耗时统计与链路追踪

---

## 依赖

### 系统依赖

| 依赖 | 说明 |
|:---|:---:|
| CMake | >= 3.13 |
| GCC | 支持 C++17 |
| FFmpeg | 视频编解码 |
| OpenCV | 图像处理 |
| Python 3.12 | Python API（可选） |
| CUDA / TensorRT | GPU 加速（可选） |

### 已集成的第三方库

| 库 | 版本 | 用途 |
|:---|:---:|:---|
| gflags | 2.3.0 | 命令行参数解析 |
| glog | 0.7.1 | 日志系统 |
| nlohmann/json | 3.11.3 | JSON 解析 |
| googletest | 1.15.2 | 单元测试框架 |
| pybind11 | 3.0.4 | Python 绑定 |
| libyuv | — | YUV 图像格式转换 |
| backward-cpp | — | 异常栈回溯 |

### 镜像

为方便测试，笔者提供了开发环境镜像，需要可联系邮件获取：wanghf_cust@163.com

---

## 编译

### 0. 安装系统依赖

```bash
# 编译安装 gflags
cd 3rdparty/gflags
cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
      -DINSTALL_HEADERS=ON -DINSTALL_SHARED_LIBS=ON -DINSTALL_STATIC_LIBS=ON ..
make -j$(nproc) && make install

# 编译安装 glog
cd 3rdparty/glog
cmake -DBUILD_SHARED_LIBS=ON ..
make -j$(nproc) && make install
```

### 1. 使用 build.sh 脚本（推荐）

```bash
# Debug 构建（默认）
./build.sh

# Release 构建
./build.sh -t Release

# Release 构建，禁用测试和 Python API
./build.sh -t Release --no-tests --no-python

# 指定构建目录和并行任务数
./build.sh -t Release -b build_release -j 8

# 清理后重新构建
./build.sh --clean -t Release
```

脚本支持的参数：

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `-t, --build-type` | 构建类型：`Debug` / `Release` | `Debug` |
| `-b, --build-dir` | CMake 构建目录 | `./build` |
| `-j, --jobs` | 并行编译任务数 | `$(nproc)` |
| `--clean` | 构建前清理构建目录 | 否 |
| `--cuda / --no-cuda` | 启用/禁用 CUDA 支持 | 启用 |
| `--tests / --no-tests` | 启用/禁用单元测试 | 启用 |
| `--python / --no-python` | 启用/禁用 Python API | 启用 |
| `--tools / --no-tools` | 启用/禁用工具构建 | 启用 |

### 2. 手动 CMake 构建

```bash
mkdir build && cd build

# Debug 构建（默认）
cmake ..

# Release 构建，禁用 CUDA 和 Python API
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DVSTREAM_USE_CUDA=OFF \
         -DVSTREAM_BUILD_PYTHON_API=OFF

make -j$(nproc)
```

### 3. CMake 构建选项

| 选项 | 说明 | 默认值 |
|:---|:---|:---:|
| `CMAKE_BUILD_TYPE` | 构建类型：`Debug` / `Release` | `Debug` |
| `VSTREAM_USE_CUDA` | 启用 NVIDIA CUDA 平台支持 | `ON` |
| `VSTREAM_BUILD_TESTS` | 构建单元测试 | `ON` |
| `VSTREAM_BUILD_MODULES` | 构建功能模块 | `ON` |
| `VSTREAM_BUILD_PYTHON_API` | 构建 Python API | `ON` |
| `VSTREAM_BUILD_TOOLS` | 构建工具程序 | `ON` |
| `VSTREAM_BUILD_LIBYUV` | 构建 libyuv | `ON` |
| `VSTREAM_BUILD_PYBIND11` | 构建 pybind11 | `ON` |

> **注意**：`Release` 模式下会自动为 GCC 启用 `-O2 -DNDEBUG` 优化，CUDA 编译器同样启用 `-O2 -DNDEBUG`。

### 4. 运行单元测试

```bash
cd build
ctest --output-on-failure
```

---

## 项目结构

```
vStream/
├── framework/              # 核心框架
│   ├── core/               # 核心框架代码
│   │   ├── include/        # 头文件
│   │   │   ├── util/
│   │   │   ├── profiler/
│   │   │   └── private/
│   │   └── src/            # 源文件
│   └── unittest/           # 框架单元测试
├── modules/                # 功能模块
│   ├── inference/          # 推理模块
│   ├── source/             # 数据源模块（图片、视频、推送）
│   ├── sink/               # 数据输出模块（推送、队列）
│   ├── proc/common/        # 通用前后处理（YOLO、ResNet、STDC）
│   ├── util/               # 工具库（CUDA 内存、仿射变换、过滤器）
│   └── unittest/           # 模块单元测试
├── python/                 # Python API
│   ├── src/                # C++ 绑定源码
│   ├── test/               # Python 测试脚本
│   └── doc/                # 文档
├── tools/
│   └── trt/                # TensorRT 模型转换工具
├── cmake/                  # CMake 模块
├── 3rdparty/               # 第三方库源码
├── build.sh                # 一键构建脚本
├── CMakeLists.txt          # 顶层 CMake 配置
└── README.md               # 项目说明
```

---

## 核心概念

### Pipeline（流水线）

Pipeline 是框架的核心，负责管理模块的生命周期和数据流转。通过 JSON 配置文件声明模块及其连接关系，支持动态构建 DAG 拓扑。

### Module（模块）

模块是基本处理单元。内置模块类型：

| 模块 | 说明 |
|:---|:---:|
| `DataSource` | 数据源模块，支持图片/视频输入和流式推送 |
| `Inference` | 推理模块，支持批量推理、前后处理管线 |
| `SinkModule` | 数据输出模块，支持推流和队列两种输出方式 |

### FrameInfo（数据载体）

`FrameInfo` 是模块间传递数据的标准载体，包含图像/视频帧、时间戳、流 ID 等元信息。`FrameVa` 扩展了视频分析结果（检测框、跟踪 ID 等）。

### Connector（连接器）

Connector 负责模块间的数据传递，支持线程安全的多生产者-多消费者队列。

### EventBus（事件总线）

EventBus 提供模块间和模块-Pipeline 间的异步消息通信，用于传递 EOS、错误等控制消息。

---

## 使用

### 1. JSON 配置流水线

```json
{
  "profiler_config": {
    "enable_profile": true
  },
  "source": {
    "class_name": "cnstream::DataSource",
    "max_input_queue_size": 20,
    "next_modules": ["inference"],
    "custom_params": {
      "config_file": "data_source.json"
    }
  },
  "inference": {
    "class_name": "cnstream::Inference",
    "next_modules": ["sink"],
    "parallelism": 2,
    "max_input_queue_size": 20,
    "custom_params": {
      "object_infer": "false",
      "model_path": "yolov8s_tracing_static_b1_pre.engine",
      "device_type": "cuda",
      "device_id": "0",
      "input_ordered_index": "0",
      "batching_timeout": "3000",
      "preproc_name": "Pre_YOLO_CPU_v2",
      "postproc_name": "Post_YOLOv8_CPU_v2",
      "custom_postproc_params": {
        "config_file": "yolo_coco.json"
      }
    }
  },
  "sink": {
    "class_name": "cnstream::DataSink",
    "parallelism": 2,
    "max_input_queue_size": 20,
    "next_modules": [],
    "custom_params": {
      "config_file": "data_sink.json"
    }
  }
}
```

data_source.json 示例
```json
{
    "channel-1": {
        "output_type": "cpu",
        "device_id": "-1",
        "interval": "1",
        "file_path": "image.png",
        "frame_rate": "600"
    },
    "channel-2": {
        "output_type": "cuda",
        "device_id": "0",
        "interval": "1",
        "url": "rtmp://localhost:1935/in/channel-2",
        "frame_rate": "30"
    },
    "channel-3": {
        "output_type": "cpu",
        "device_id": "-1",
        "interval": "1",
        "file_path": "image.png",
        "frame_rate": "300"
    },
    "channel-4": {
        "output_type": "cpu",
        "device_id": "-1",
        "interval": "1"
    }
}
```

data_sink.json 示例
```json
{
    "channel-1": {
        "device_id": "0",
        "fps": "30",
        "width": "960",
        "height": "640",
        "url": "rtmp://localhost:1935/out/channel-1"
    },
    "channel-2": {
        "device_id": "0",
        "fps": "30",
        "width": "960",
        "height": "640",
        "url": "rtmp://localhost:1935/out/channel-2"
    },
    "channel-3": {
        "queue_size": "40"
    },
    "channel-4": {
        "queue_size": "40"
    }
}
```

### 2. C++ 示例

#### 基本流水线

```cpp
#include "cnstream_pipeline.hpp"

int main() {
    cnstream::Pipeline pipeline("my_pipeline");
    if (!pipeline.BuildPipelineByJSONFile("pipeline.json")) {
      return -1;
    }
    pipeline.Start();

    std::string stream_id = "channel-1";
    std::string source_module_name = "source";
    std::string sink_module_name = "sink";

    DataSource *source = dynamic_cast<DataSource*>(pipeline_->GetModule(source_module_name));
    auto source_handler_ptr = ImageHandler::Create(source, stream_id);
    auto handler = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
    source->AddSource(handler)；

    DataSink *sink = dynamic_cast<DataSink*>(pipeline_->GetModule(sink_module_name));
    auto sink_handler = PushHandler::Create(sink, stream_id);
    auto push_handler = std::dynamic_pointer_cast<PushHandler>(sink_handler);
    sink->AddSink(push_handler)；

    // 等待处理完成...

    pipeline.Stop();
    return 0;
}
```

#### 自定义模块

```cpp
#include "cnstream_module.hpp"

class MyModule : public cnstream::Module,
                 public cnstream::ModuleCreator<MyModule> {
 public:
    explicit MyModule(const std::string& name) : cnstream::Module(name) {}

    bool CheckParamSet(const cnstream::ModuleParamSet& params) const override {
        return true;
    }

    bool Open(cnstream::ModuleParamSet params) override {
        // 初始化资源
        return true;
    }

    void Close() override {
        // 释放资源
    }

    int Process(std::shared_ptr<cnstream::FrameInfo> data) override {
        // 处理每一帧数据
        return 0;
    }
};

// 注册模块（使其可被 JSON 配置引用）
REGISTER_MODULE(MyModule);
```

### 3. Python 示例

#### 环境准备

```bash
# 安装 Python 依赖
pip install -r python/requirements.txt

# 设置环境变量
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)/lib:$PYTHONPATH
```

#### 使用流水线

```python
import vstream

# 创建流水线
stream_id = "channel-1"
pipeline = vstream.Pipeline("my_pipeline")

# 从 JSON 文件构建
pipeline.build_pipeline_by_json_file("config.json")

source = pipeline.get_data_source("source")
sink = pipeline.get_data_sink("sink")

send_handler = vstream.SendHandler(source, stream_id)
queue_handler = vstream.QueueHandler(sink, stream_id)

# 启动
pipeline.start()
print(f"Pipeline running: {pipeline.is_running()}")

ret = source.add_source(send_handler)
ret = sink.add_sink(queue_handler)

if source:
    import cv2
    img = cv2.imread("test.png")
    frame = vstream.FrameInfo()
    source.send_image(stream_id, img, frame)

# 停止
pipeline.stop()
```

#### 自定义 Python 模块

```python
import vstream

class MyPythonModule(vstream.Module):
    def __init__(self, name):
        super().__init__(name)

    def open(self, params):
        print(f"[{self.get_name()}] opened")
        return True

    def close(self):
        print(f"[{self.get_name()}] closed")

    def process(self, frame_info):
        print(f"[{self.get_name()}] process frame {frame_info.frame_id}")
        return 0

# 运行内置测试
# cd python/test && python test_pipeline.py
```

---

## 工具

| 工具 | 路径 | 说明 |
|:---|:---:|:---|
| TRT 模型转换 | `tools/demo/trt/` | ONNX → TensorRT Engine 模型转换工具 |

---

## 适配模型

### CUDA 平台（TensorRT 后端）

| 模型 | 状态 |
|:---|:---:|
| YOLOv5 检测 | 已完成 |
| YOLOv8 检测 | 已完成 |
| ResNet 分类 | 已完成 |

### 前后处理

| 处理类型 | 模型 |
|:---:|:---|
| 预处理 | YOLOv5/v8、ResNet |
| 后处理 | YOLOv5/v8、ResNet|

---

## 开发计划

- [x] Pipeline 流水线管理
- [x] JSON 配置解析与动态构建
- [x] Module 模块基类与注册机制
- [x] Connector 模块连接器
- [x] EventBus 事件总线
- [x] SourceModule 数据源模块（图片/视频/流式）
- [x] Inference 推理模块（批量推理/超时控制）
- [x] SinkModule 数据输出模块
- [x] CUDA / TensorRT 推理后端
- [x] Python 绑定（Pipeline / Module / Sink / Source）
- [x] Profiler 性能剖析
- [x] Encoder 推流模块

---

## 参考

[CNStream](https://github.com/Cambricon/CNStream)
