# ==============================================
# Try to find TensorRT libraries:
# - nvinfer          (Core library)
# - nvinfer_plugin   (Plugin library)
# - nvonnxparser     (ONNX parser)
#
# 命名约定（重要）：
#   输出变量的前缀必须与本文件名中 "Find" 之后的部分完全一致，即 `TensorRT`。
#   cmake/platforms/<id>.cmake 的 FIND_MODULES 填的也是 "TensorRT"，
#   cmake/PlatformBase.cmake 会按 <X>_FOUND / <X>_INCLUDE_DIRS / <X>_LIBRARIES 聚合。
#   CMake 变量大小写敏感，写成 TENSORRT_* 会导致聚合时全部取到空值。
#
# Output:
#   TensorRT_FOUND          - system has TensorRT
#   TensorRT_INCLUDE_DIR    - the TensorRT include directory
#   TensorRT_INCLUDE_DIRS   - 复数形式（契约要求）
#   TensorRT_LIBRARY        - 核心库 nvinfer（供打包期推断依赖搜索目录）
#   TensorRT_LIBRARIES      - Link these to use TensorRT
#
# Input:
#   TensorRT_ROOT_DIR       - 自定义安装根（-DTensorRT_ROOT_DIR=... 或 CMAKE_PREFIX_PATH）
# ==============================================
# Notice: this script is for Linux.

# Support custom TensorRT installation path
if(NOT DEFINED TensorRT_ROOT_DIR)
    set(TensorRT_ROOT_DIR "/usr/local/tensorrt" CACHE PATH "Default TensorRT root directory")
endif()

if (TensorRT_LIBRARIES AND TensorRT_INCLUDE_DIR)
    # in cache already
    set(TensorRT_FOUND TRUE)
else ()
    # Find include directory (looking for NvInfer.h)
    find_path(
        TensorRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATHS ${TensorRT_ROOT_DIR}/include
              /usr/include/x86_64-linux-gnu
              /usr/include/aarch64-linux-gnu
              /usr/local/include
              /usr/include
        DOC "TensorRT include directory"
    )

    # Find nvinfer (Core library, required)
    find_library(
        TensorRT_LIB_NVINFER
        NAMES nvinfer
        PATHS ${TensorRT_ROOT_DIR}/lib
              ${TensorRT_ROOT_DIR}/lib64
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu
              /usr/lib64
              /usr/local/lib
    )

    # Find nvinfer_plugin
    find_library(
        TensorRT_LIB_NVINFER_PLUGIN
        NAMES nvinfer_plugin
        PATHS ${TensorRT_ROOT_DIR}/lib
              ${TensorRT_ROOT_DIR}/lib64
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu
              /usr/lib64
              /usr/local/lib
    )

    # Find nvonnxparser
    find_library(
        TensorRT_LIB_NVONNXPARSER
        NAMES nvonnxparser
        PATHS ${TensorRT_ROOT_DIR}/lib
              ${TensorRT_ROOT_DIR}/lib64
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu
              /usr/lib64
              /usr/local/lib
    )

    # Check required components
    if (NOT TensorRT_LIB_NVINFER)
        message(FATAL_ERROR "Cannot find TensorRT core library (nvinfer)")
    endif()

    if (NOT TensorRT_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find TensorRT headers (NvInfer.h)")
    endif()

    # Set found flag and libraries
    if (TensorRT_LIB_NVINFER AND TensorRT_INCLUDE_DIR)
        set(TensorRT_FOUND TRUE)

        # ---- 公开契约变量（前缀必须与 FindTensorRT.cmake 文件名一致）----
        set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
        set(TensorRT_LIBRARY     ${TensorRT_LIB_NVINFER})

        # Collect all found libraries
        set(TensorRT_LIBRARIES ${TensorRT_LIB_NVINFER})

        if (TensorRT_LIB_NVINFER_PLUGIN)
            list(APPEND TensorRT_LIBRARIES ${TensorRT_LIB_NVINFER_PLUGIN})
        else()
            message(WARNING "TensorRT plugin library (nvinfer_plugin) not found")
        endif()

        if (TensorRT_LIB_NVONNXPARSER)
            list(APPEND TensorRT_LIBRARIES ${TensorRT_LIB_NVONNXPARSER})
        else()
            message(WARNING "TensorRT ONNX parser (nvonnxparser) not found")
        endif()

        message(STATUS "Found TensorRT: ${TensorRT_LIB_NVINFER}")
        message(STATUS "TensorRT Include: ${TensorRT_INCLUDE_DIR}")
    else()
        set(TensorRT_FOUND FALSE)
        message(FATAL_ERROR "Could not find TensorRT libraries!")
    endif()

    # Hide advanced variables in CMake GUI
    mark_as_advanced(
        TensorRT_INCLUDE_DIR
        TensorRT_LIB_NVINFER
        TensorRT_LIB_NVINFER_PLUGIN
        TensorRT_LIB_NVONNXPARSER
    )
endif()
