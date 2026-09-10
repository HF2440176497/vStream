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

inline const std::string kSkipFrameTag = "skip_frame";

/// 帧级裁剪旋转标记：被标记的帧，其对象级处理在 bbox 裁剪后将裁剪图旋转 180°
inline const std::string kCropRotate180Tag = "crop_rotate_180";

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
  uint64_t GetTimestamp() const { return timestamp; }

  std::string stream_id;  /*!< The data stream aliases where this frame is located to. */
  std::string frame_id_s;  // for send handler, 用来额外标记 frame
  uint64_t timestamp; /*!< The time stamp of this frame. */
  size_t flags = 0;       /*!< The mask for this frame, ``DataFrameFlag``. */

  Collection collection;

  /**
   * @brief 标记本帧跳过指定的下游模块。
   *
   * 路由时框架对该模块"虚拟通过"（置位 modules_mask_ 但不入队），数据直接沿其
   * 下游继续传播；EOS 帧由框架豁免，始终流经所有模块。
   *
   * @param[in] module 需要跳过的下游模块（通常通过 Pipeline::GetModule 获取）。
   *
   * @return No return value.
   */
  void MarkSkipModule(Module* module);

#ifdef VSTREAM_UNIT_TEST
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
  /**
   * @brief 锁内原子的 test-and-set 置位：仅当 module 的 bit 原本未置位时置位并返回 true
   *        （本次调用完成 0→1 翻转），否则不产生任何修改并返回 false。
   *
   * 仅供 Pipeline 旁路路由（虚拟通过）使用：将"谁翻转 bit 谁负责其下游传播"的判定
   *
   * @param[in] module 目标模块。
   * @param[out] new_mask 翻转成功时输出置位后的完整 modules_mask_ 快照（供路由继续判定）。
   * @return 返回 true 表示本次调用翻转了该 bit，调用方负责其下游传播。
   */
  bool MarkPassedOnce(Module* module, uint64_t* new_mask);

  mutable std::mutex mask_lock_;
  /* Identifies which modules have processed this data */
  uint64_t modules_mask_ = 0;
  /* Identifies which downstream modules should be skipped for this data（仅供框架路由使用） */
  uint64_t skip_mask_ = 0;

  /**
   * @brief 查询本帧是否跳过指定模块。仅供 Pipeline 路由（friend Pipeline）使用。
   */
  bool IsModuleSkipped(Module* module);

};  // end class FrameInfo

/*!
 * Defines an alias for the std::shared_ptr<FrameInfo>. FrameInfoPtr now denotes a shared pointer of frame
 * information.
 */
using FrameInfoPtr = std::shared_ptr<FrameInfo>;

}  // namespace cnstream

#endif  // CNSTREAM_FRAME_HPP_
