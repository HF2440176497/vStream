# ==============================================
# Try to find TensorRT libraries:
# - nvinfer          (Core library)
# - nvinfer_plugin   (Plugin library)
# - nvonnxparser     (ONNX parser)
#
# TENSORRT_FOUND - system has TensorRT
# TENSORRT_INCLUDE_DIR - the TensorRT include directory
# TENSORRT_LIBRARIES - Link these to use TensorRT
# ==============================================
# Notice: this script is for Linux.

# Support custom TensorRT installation path
if(NOT DEFINED TensorRT_ROOT_DIR)
    set(TensorRT_ROOT_DIR "/usr/local/tensorrt" CACHE PATH "Default TensorRT root directory")
endif()

if (TENSORRT_LIBRARIES AND TENSORRT_INCLUDE_DIR)
    # in cache already
    set(TENSORRT_FOUND TRUE)
else ()
    # Find include directory (looking for NvInfer.h)
    find_path(
        TENSORRT_INCLUDE_DIR
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
        TENSORRT_LIB_NVINFER
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
        TENSORRT_LIB_NVINFER_PLUGIN
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
        TENSORRT_LIB_NVONNXPARSER
        NAMES nvonnxparser
        PATHS ${TensorRT_ROOT_DIR}/lib
              ${TensorRT_ROOT_DIR}/lib64
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu
              /usr/lib64
              /usr/local/lib
    )

    # Check required components
    if (NOT TENSORRT_LIB_NVINFER)
        message(FATAL_ERROR "Cannot find TensorRT core library (nvinfer)")
    endif()

    if (NOT TENSORRT_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find TensorRT headers (NvInfer.h)")
    endif()

    # Set found flag and libraries
    if (TENSORRT_LIB_NVINFER AND TENSORRT_INCLUDE_DIR)
        set(TENSORRT_FOUND TRUE)
        
        # Collect all found libraries
        set(TENSORRT_LIBRARIES ${TENSORRT_LIB_NVINFER})
        
        if (TENSORRT_LIB_NVINFER_PLUGIN)
            list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIB_NVINFER_PLUGIN})
        else()
            message(WARNING "TensorRT plugin library (nvinfer_plugin) not found")
        endif()
        
        if (TENSORRT_LIB_NVONNXPARSER)
            list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIB_NVONNXPARSER})
        else()
            message(WARNING "TensorRT ONNX parser (nvonnxparser) not found")
        endif()

        message(STATUS "Found TensorRT: ${TENSORRT_LIB_NVINFER}")
        message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
    else()
        set(TENSORRT_FOUND FALSE)
        message(FATAL_ERROR "Could not find TensorRT libraries!")
    endif()

    # Hide advanced variables in CMake GUI
    mark_as_advanced(
        TENSORRT_INCLUDE_DIR
        TENSORRT_LIB_NVINFER
        TENSORRT_LIB_NVINFER_PLUGIN
        TENSORRT_LIB_NVONNXPARSER
    )
endif()