/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
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

#ifndef CNSTREAM_FRAME_HPP_
#define CNSTREAM_FRAME_HPP_

#include <memory>
#include <string>
#include <map>
#include <vector>

#include "cnstream_collection.hpp"
#include "cnstream_common.hpp"

/**
 *  @file cnstream_frame.hpp
 *
 *  This file contains a declaration of the FrameInfo class.
 */
namespace cnstream {

// Sasha: 前置声明代替头文件包含
// 因为 Module 和 Pipeline 会 include cnstream_frame，避免造成循环引用

class Module;
class Pipeline;

/**
 * @enum FrameFlag
 *
 * @brief Enumeration variables describing the mask of DataFrame.
 */
enum class DataFrameFlag {
  FRAME_FLAG_EOS = 1 << 0,     /*!< This enumeration indicates the end of data stream. */
  FRAME_FLAG_INVALID = 1 << 1, /*!< This enumeration indicates an invalid frame. */
  FRAME_FLAG_REMOVED = 1 << 2  /*!< This enumeration indicates that the stream has been removed. */
};

/**
 * @class FrameInfo
 *
 * @brief FrameInfo is a class holding the information of a frame.
 *
 */
class FrameInfo : private NonCopyable {
 public:
  /**
   * @brief Creates a FrameInfo instance.
   *
   * @param[in] stream_id The data stream alias. Identifies which data stream the frame data comes from.
   * @param[in] eos  Whether this is the end of the stream. This parameter is set to false by default to
   *                 create a FrameInfo instance. If you set this parameter to true,
   *                 DataFrame::flags will be set to ``CN_FRAME_FLAG_EOS``. Then, the modules
   *                 do not have permission to process this frame. This frame should be handed over to
   *                 the pipeline for processing.
   *
   * @return Returns ``shared_ptr`` of ``FrameInfo`` if this function has run successfully. Otherwise, returns NULL.
   */
  static std::shared_ptr<FrameInfo> Create(const std::string& stream_id, bool eos = false);

 private:
  FrameInfo() = default;

 public:
  /**
   * @brief Destructs FrameInfo object.
   *
   * @return No return value.
   */
  ~FrameInfo();

  /**
   * @brief Checks whether DataFrame is end of stream (EOS) or not.
   *
   * @return Returns true if the frame is EOS. Returns false if the frame is not EOS.
   */
  bool IsEos() { return (flags & static_cast<size_t>(DataFrameFlag::FRAME_FLAG_EOS)) ? true : false; }

  /**
   * @brief Checks whether DataFrame is removed or not.
   *
   * @return Returns true if the frame is removed. Returns false if the frame is not removed.
   */
  bool IsRemoved() {
    return (flags & static_cast<size_t>(DataFrameFlag::FRAME_FLAG_REMOVED)) ? true : false;
  }

  /**
   * @brief Checks if DataFrame is valid or not.
   *
   * @return Returns true if frame is invalid, otherwise returns false.
   */
  bool IsInvalid() {
    return (flags & static_cast<size_t>(DataFrameFlag::FRAME_FLAG_INVALID)) ? true : false;
  }

  /**
   * @brief Sets index (usually the index is a number) to identify stream.
   *
   * @param[in] index Number to identify stream.
   *
   * @return No return value.
   *
   * @note This is only used for distributing each stream data to the appropriate thread.
   * We do not recommend SDK users to use this API because it will be removed later.
   */
  void SetStreamIndex(uint32_t index) { channel_idx = index; }

  /**
   * @brief Gets index number which identifies stream.
   * 与创建时的 SourceModule 的 stream_id 保持一致
   *
   * @return Index number.
   *
   * @note This is only used for distributing each stream data to the appropriate thread.
   * We do not recommend SDK users to use this API because it will be removed later.
   */
  uint32_t GetStreamIndex() const { return channel_idx; }

  std::string GetStreamId() const { return stream_id; }
  int64_t GetTimestamp() const { return timestamp; }

  std::string stream_id;  /*!< The data stream aliases where this frame is located to. */
  std::string frame_id_s;  // for send handler, 用来额外标记 frame
  int64_t timestamp = -1; /*!< The time stamp of this frame. */
  size_t flags = 0;       /*!< The mask for this frame, ``DataFrameFlag``. */

  Collection collection;

#ifdef UNIT_TEST
 public:
  uint32_t test_idx = 0;
#else
 private:
#endif
  /**
   * The below methods and members are used by the framework.
   */
  friend class Pipeline;
  mutable uint32_t channel_idx = INVALID_STREAM_IDX;        ///< The index of the channel, stream_index
  void SetModulesMask(uint64_t mask);
  uint64_t GetModulesMask();
  uint64_t MarkPassed(Module* current);  // return changed mask

  mutable std::mutex mask_lock_;
  /* Identifies which modules have processed this data */
  uint64_t modules_mask_ = 0;

};  // end class FrameInfo

/*!
 * Defines an alias for the std::shared_ptr<FrameInfo>. FrameInfoPtr now denotes a shared pointer of frame
 * information.
 */
using FrameInfoPtr = std::shared_ptr<FrameInfo>;

}  // namespace cnstream

#endif  // CNSTREAM_FRAME_HPP_
