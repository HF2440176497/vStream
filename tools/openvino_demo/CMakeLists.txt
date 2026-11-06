# =============================================================================
# tools/openvino_demo/CMakeLists.txt
#
# 独立的小工程：不接入 vStream 根构建（tools/CMakeLists.txt 只会
# add_subdirectory(平台 TOOL_SUBDIRS)，本目录不在其中），专门用来在容器里
# 验证"apt 装的 OpenVINO 能不能被 find_package 找到并链接成功"。
#
#   cmake -S tools/openvino_demo -B build/ov_demo -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/ov_demo -j$(nproc)
#   ./build/ov_demo/ov_trans_model <model.onnx> <out.xml> --shape 1,3,640,640 \
#       --device CPU --cache-dir <cache_dir>
#
# 部署侧 demo 只读 IR (.xml)：先查 EXPORT_IMPORT 能力，支持才开 ov::cache_dir，
# 再用 compile_model(ir_path, device, config) 加载（官方主推路径）。
# 若 find_package 找不到，显式指定 -DOpenVINO_DIR=<OpenVINOConfig.cmake 所在目录>：
#   cmake -S tools/openvino_demo -B build/ov_demo \
#         -DOpenVINO_DIR=/usr/lib/x86_64-linux-gnu/cmake/openvino
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(vstream_openvino_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# apt 安装会提供 OpenVINOConfig.cmake + imported target openvino::runtime。
# 注意：apt 方式不需要 source setupvars.sh（那是 archive/tarball 安装才需要）。
find_package(OpenVINO REQUIRED)
message(STATUS "OpenVINO_DIR = ${OpenVINO_DIR}")

# ---- target 0：离线转换工具（ONNX -> IR + 预热 model cache），对标 trans_model.cpp ----
add_executable(ov_trans_model ov_trans_model.cpp)
target_link_libraries(ov_trans_model PRIVATE openvino::runtime)

# ---- target 1：纯 OpenVINO，零外部依赖，先验证环境链路 ----
add_executable(ov_hello ov_hello.cpp)
target_link_libraries(ov_hello PRIVATE openvino::runtime)

