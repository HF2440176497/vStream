/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#ifndef MODULES_DATA_SOURCE_PARAM_HPP_
#define MODULES_DATA_SOURCE_PARAM_HPP_

#include <memory>
#include <string>
#include <unordered_map>

#include "cnstream_config.hpp"

namespace cnstream {

inline constexpr uint32_t CN_MAX_PLANES = 3;

/**
 * @enum DataFormat
 *
 * @brief Enumeration variables describling the pixel format of the data in DataFrame.
 */
enum class DataFormat {
  INVALID = -1,                 /*!< This frame is invalid. */
  PIXEL_FORMAT_YUV420_NV21 = 0, /*!< This frame is in the YUV420SP(NV21) format. */
  PIXEL_FORMAT_YUV420_NV12,     /*!< This frame is in the YUV420sp(NV12) format. */
  PIXEL_FORMAT_BGR24,           /*!< This frame is in the BGR24 format. */
  PIXEL_FORMAT_RGB24,           /*!< This frame is in the RGB24 format. */
  PIXEL_FORMAT_ARGB32,          /*!< This frame is in the ARGB32 format. */
  PIXEL_FORMAT_ABGR32,          /*!< This frame is in the ABGR32 format. */
  PIXEL_FORMAT_RGBA32,          /*!< This frame is in the RGBA32 format. */
  PIXEL_FORMAT_BGRA32           /*!< This frame is in the BGRA32 format. */
};

inline std::string DataFormat2Str(DataFormat fmt) {
  switch (fmt) {
    case DataFormat::PIXEL_FORMAT_YUV420_NV21: return "YUV420_NV21";
    case DataFormat::PIXEL_FORMAT_YUV420_NV12: return "YUV420_NV12";
    case DataFormat::PIXEL_FORMAT_BGR24: return "BGR24";
    case DataFormat::PIXEL_FORMAT_RGB24: return "RGB24";
    case DataFormat::PIXEL_FORMAT_ARGB32: return "ARGB32";
    case DataFormat::PIXEL_FORMAT_ABGR32: return "ABGR32";
    case DataFormat::PIXEL_FORMAT_RGBA32: return "RGBA32";
    case DataFormat::PIXEL_FORMAT_BGRA32: return "BGRA32";
    default: return "INVALID";
  }
}

enum class DevType {
  INVALID = -1,                /*!< Invalid device type. */
  CPU = 0,                     /*!< The data is allocated by CPU. */
  CUDA = 1,                    /*!< The data is allocated by CUDA. */
};

/**
 * @struct DevContext
 *
 * @brief DevContext is a structure holding the information that CNDataFrame data is allocated by CPU or MLU.
 */
struct DevContext {
  DevContext() = default;
  DevContext(DevType type, int id) : device_type(type), device_id(id) {}
  DevType device_type = DevType::INVALID; /*!< Device type. The default value is ``INVALID``.*/
  int device_id = -1;                /*!< Ordinal device ID. */
};

inline std::string DevType2Str(DevType type) {
  switch (type) {
    case DevType::CPU: return "CPU";
    case DevType::CUDA: return "CUDA";
    default: return "INVALID";
  }
}

inline std::unordered_map<std::string, DevType> device_type_map = {
  {"cpu", DevType::CPU},
  {"CPU", DevType::CPU},
  {"cuda", DevType::CUDA},
  {"CUDA", DevType::CUDA}
};

/**
 * @brief Gets image plane number by a specified image format.
 * 表示数量，范围为自然数
 * @retval 0: Unsupported image format.
 * @retval >0: Image plane number.
 */
inline int FormatPlanes(DataFormat fmt) {
  switch (fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24:
    case DataFormat::PIXEL_FORMAT_RGB24:
      return 1;
    case DataFormat::PIXEL_FORMAT_YUV420_NV12:
    case DataFormat::PIXEL_FORMAT_YUV420_NV21:
      return 2;
    default:
      return 0;
  }
  return 0;
}


/*!
 * @enum OutputType
 * @brief Enumeration variables describing the storage type of the output frame data of a module.
 */
enum class OutputType {
  OUTPUT_CPU,  /*!< CPU is the used storage type. */
  OUTPUT_CUDA,  /*!< CUDA is the used storage type. */
  OUTPUT_NPU   /*!< NPU is the used storage type. */
};

/*!
 * @enum DecoderType
 * @brief Enumeration variables describing the decoder type used in source module.
 */
enum class DecoderType {
  DECODER_CPU,  /*!< CPU decoder is used. */
  DECODER_CUDA,  /*!< Video decoder is used. */
  DECODER_NPU   /*!< NPU decoder is used. */
};

class IDecBufRef {
public:
  virtual ~IDecBufRef() {}
};

/**
 * @class IDataDeallocator
 *
 * @brief IDataDeallocator is an abstract class of deallocator for the CNDecoder buffer.
 */
class IDataDeallocator {
 public:
  /*!
   * @brief Destructs the base object.
   *
   * @return No return value.
   */
  virtual ~IDataDeallocator() {}
};

/**
 * @brief 零拷贝内存引用器 作为 DecodeFrame 的 buf_ref
 */
class Deallocator : public IDataDeallocator {
 public:
  explicit Deallocator(IDecBufRef *ptr) {
    ptr_.reset(ptr);
  }
  virtual ~Deallocator() {
    ptr_.reset();
    ptr_ = nullptr;
  }
 private:
  std::unique_ptr<IDecBufRef> ptr_;
};

struct DecodeFrame {
  DecodeFrame(int height, int width, DataFormat fmt = DataFormat::PIXEL_FORMAT_BGR24) : height(height), width(width), fmt(fmt) {
    valid = true;
    pts = 0;
    for (int i = 0; i < CN_MAX_PLANES; ++i) {
      plane[i] = nullptr;
      stride[i] = 0;
    }
  }
  bool valid;
  int64_t pts;
  std::string frame_id_s;
  int32_t height;
  int32_t width; 
  
  DevType device_type = DevType::INVALID;
  int32_t device_id = -1;
  
  DataFormat fmt;
  int32_t planeNum;
  void *plane[CN_MAX_PLANES];
  int stride[CN_MAX_PLANES];
  std::unique_ptr<IDecBufRef> buf_ref = nullptr;

 public:
  ~DecodeFrame() {
    // note: DecodeFrame 不负责 plane 的内存管理
    for (int i = 0; i < CN_MAX_PLANES; ++i) {
      plane[i] = nullptr;
    }
    buf_ref.reset();
  }
};


inline const std::unordered_map<std::string, OutputType> param_output_map_ = {
  {"cpu", OutputType::OUTPUT_CPU},
  {"cuda", OutputType::OUTPUT_CUDA},
  {"npu", OutputType::OUTPUT_NPU}
};

inline const std::unordered_map<std::string, DecoderType> param_decoder_map_ = {
  {"cpu", DecoderType::DECODER_CPU},
  {"cuda", DecoderType::DECODER_CUDA},
  {"npu", DecoderType::DECODER_NPU}
};

inline const std::string key_output_type = "output_type";
inline const std::string key_device_id = "device_id";
inline const std::string key_interval = "interval";
inline const std::string key_decoder_type = "decoder_type";
inline const std::string key_only_key_frame = "only_key_frame";

inline const std::string key_file_path = "file_path";
inline const std::string key_frame_rate = "frame_rate";
inline const std::string key_stream_url = "stream_url";

/*!
 * @brief DataSourceParam is a structure for private usage.
 */
struct DataSourceParam {
  int  device_id_ = -1;                                 /*! DataFrame 的 device_id 直接来自 decode_frame  */
  size_t  interval_ = 1;                                /*!< The interval of outputting one frame. It outputs one frame every n (interval_) frames. */
  OutputType output_type_ = OutputType::OUTPUT_CPU;     /*!< The output type */
  DecoderType decoder_type_ = DecoderType::DECODER_CPU; /*!< The decoder type. */
  bool only_key_frame_ = false;                         /*!< Whether only to decode key frames. */
  ModuleParamSet param_set_ {};
};
}  // namespace cnstream

#endif  // MODULES_DATA_SOURCE_PARAM_HPP_
