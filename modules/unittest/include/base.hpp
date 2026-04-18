

#ifndef MODULES_UNITEST_INCLUDE_TEST_BASE_HPP_
#define MODULES_UNITEST_INCLUDE_TEST_BASE_HPP_

#include <string>
#include <string.h>
#include <utility>
#include <cstdlib>
#include <unistd.h>
#include <cerrno>
#include <utility>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "cnstream_module.hpp"
#include "data_source_param.hpp"
#include "cnstream_frame_va.hpp"

#define PATH_MAX_LENGTH 1024

/**
 * @brief 获取当前执行程序的路径
 * @return 返回当前执行程序的路径
 * @example /usr/bin/app return /usr/bin/
 **/
std::string GetExePath();
void CheckExePath(const std::string& path);

/**
 * @brief Creates a temp file.
 * @return Returns temp file informations.
 * Return value is a std::pair object, the first value stored temp file fd,
 * and the second value stored temp file name.
 * @note close(fd), unlink(filename) must be called when the temp file is uesd up.
 **/
std::pair<int, std::string> CreateTempFile(const std::string& filename_prefix);

/**
 * @brief 截取自 CNConfigBase::ParseByJSONFile
 * 读取 json 文件内容为 json 字符串
 */
std::string readFile(const char* filename);

inline uint64_t get_timestamp_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

namespace cnstream {

/**
 * @brief 创建一个测试的 DecodeFrame
 * @param fmt 图像格式
 * @param width 图像宽度
 * @param height 图像高度
 */
DecodeFrame* CreateTestDecodeFrame(DataFormat fmt, int width, int height);

/**
 * @brief 配合 CreateTestDecodeFrame 使用，清理测试的 DecodeFrame
 */
void CleanupTestDecodeFrame(DecodeFrame* frame);

inline std::string process_total_name = "process_total";
inline std::string process_one_name = "process_one";
inline std::string process_two_name = "process_two";

struct FrameCountData {
  uint64_t process_count = 0;
  std::mutex mtx;
};

/**
 * @brief 创建测试 pipeline 用到的 Module
 * 测试并发性
 （1）对于配置为 next_nods: [ProcessOne, ProcessTwo] 的情况， ProcessOne 和 ProcessTwo 是并发执行的
 （2）验证 ProcessThree 接收的 data 一定是 ProcessOne 和 ProcessTwo 处理后的 data
 */
class ProcessOne: public Module, public ModuleCreator<ProcessOne> {
 public:
  ProcessOne(const std::string &name) : Module(name) {}
  ~ProcessOne() {}
  bool Open(ModuleParamSet params) override {
    return true;
  }
  void Close() override {
    LOGI(ProcessOne) << "Close";
  }
  void OnEos(const std::string& stream_id) override {
    LOGI(ProcessOne) << "OnEos: " << stream_id;
  }
  int Process(std::shared_ptr<FrameInfo> frame_info) override {
    DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
      LOGE(ProcessOne) << "frame is empty";
      return -1;
    }
    frame_count_++;

    if (!frame_info->collection.HasValue(process_one_name)) {
      frame_info->collection.Add(process_one_name, std::make_shared<FrameCountData>());
    }
    if (!frame_info->collection.HasValue(process_total_name)) {
      frame_info->collection.AddIfNotExists(process_total_name, std::make_shared<FrameCountData>());
    }
    // total_count_ 相当于是全局计数
    // process_xxx_name 是模块内自己的 用于验证是否经过
    // 1. 获取 process_total_name 对应的 FrameCountData
    auto total_count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_total_name);
    {
      std::lock_guard<std::mutex> lock(total_count_data->mtx);
      total_count_data->process_count++;
      LOGD(ProcessOne) << "frame ts: " << frame_info->timestamp << " process_total_count: " << total_count_data->process_count;
    }
    // 2. 获取 当前 module 对应的 FrameCountData, 自定义赋值
    auto count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_one_name);
    {
      std::lock_guard<std::mutex> lock(count_data->mtx);
      count_data->process_count = frame_count_;
      LOGD(ProcessOne) << "frame ts: " << frame_info->timestamp << " process_one_count: " << count_data->process_count;
    }
    return 0;
  }
 private:
  uint64_t frame_count_ = 0;
};
REGISTER_MODULE(ProcessOne);


class ProcessTwo: public Module, public ModuleCreator<ProcessTwo> {
 public:
  ProcessTwo(const std::string &name) : Module(name) {}
  ~ProcessTwo() {}
  bool Open(ModuleParamSet params) override {
    return true;
  }
  void Close() override {
    LOGI(ProcessTwo) << "Close";
  }
  void OnEos(const std::string& stream_id) override {
    LOGI(ProcessTwo) << "OnEos: " << stream_id;
  }

  int Process(std::shared_ptr<FrameInfo> frame_info) override {
    DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
      LOGE(ProcessOne) << "frame is empty";
      return -1;
    }
    frame_count_++;

    if (!frame_info->collection.HasValue(process_two_name)) {
      frame_info->collection.Add(process_two_name, std::make_shared<FrameCountData>());
    }
    if (!frame_info->collection.HasValue(process_total_name)) {
      frame_info->collection.AddIfNotExists(process_total_name, std::make_shared<FrameCountData>());
    }
    // 1. 获取 process_total_name 对应的 FrameCountData
    auto total_count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_total_name);
    {
      std::lock_guard<std::mutex> lock(total_count_data->mtx);
      total_count_data->process_count += frame_count_;
      LOGD(ProcessTwo) << "frame ts: " << frame_info->timestamp << " process_total_count: " << total_count_data->process_count;
    }
    // 2. 获取 当前 module 对应的 FrameCountData, 自定义赋值
    auto count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_two_name);
    {
      std::lock_guard<std::mutex> lock(count_data->mtx);
      count_data->process_count = 2 * frame_count_;
      LOGD(ProcessTwo) << "frame ts: " << frame_info->timestamp << " process_two_count: " << count_data->process_count;
    }
    return 0;
  }
 private:
  uint64_t frame_count_ = 0;  // 表示当前 module 处理的 frame 数量
};
REGISTER_MODULE(ProcessTwo);


class ProcessThree: public Module, public ModuleCreator<ProcessThree> {
 public:
  ProcessThree(const std::string &name) : Module(name) {}
  ~ProcessThree() {}
  bool Open(ModuleParamSet params) override {
    return true;
  }
  void Close() override {
    LOGI(ProcessThree) << "Close";
  }
  void OnEos(const std::string& stream_id) override {
    LOGI(ProcessThree) << "OnEos: " << stream_id;
  }
  int Process(std::shared_ptr<FrameInfo> frame_info) override {
    DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
      LOGE(ProcessThree) << "frame is empty";
      return -1;
    }
    frame_count_++;

    // 经过前两个 module, 才会到达 ProcessThree 因此 total 一定存在
    if (!frame_info->collection.HasValue(process_total_name)) {
      LOGE(ProcessThree) << "process_total not found";
      return -1;
    }
    // total_count_: 前两个模块 count 的加和; should == ProcessThree frame_count_ + 1
    auto total_count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_total_name);
    EXPECT_EQ(total_count_data->process_count, frame_count_ + 1);
    LOGD(ProcessThree) << "frame ts: " << frame_info->timestamp << " total_count_data: " << total_count_data->process_count << "; while module_three count: " << frame_count_;

    // process_two_count should == 2 * process_one_count
    if (frame_info->collection.HasValue(process_one_name)) {
      auto count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_one_name);
      LOGD(ProcessThree) << "frame ts: " << frame_info->timestamp << " process_one_count: " << count_data->process_count;
    } else {
      LOGE(ProcessThree) << "process_one not found";
      return -1;
    }
    // 对于 process_two, 其内部进行了两倍的处理
    if (frame_info->collection.HasValue(process_two_name)) {
      auto one_count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_one_name);
      auto two_count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_two_name);
      EXPECT_EQ(two_count_data->process_count, 2 * one_count_data->process_count);
      LOGD(ProcessThree) << "frame ts: " << frame_info->timestamp << " process_two_count: " << two_count_data->process_count;
    } else {
      LOGE(ProcessThree) << "process_two not found";
      return -1;
    }
    return 0;
  }
 private:
  uint64_t frame_count_ = 0;
  std::unordered_map<std::string, std::mutex> mutex_map_;
  std::mutex mtx_;
};
REGISTER_MODULE(ProcessThree);


/**
 * 提取 frame_info 中的 frame_id_s 得到数字，验证是否连续
 * 配合 test_send 单元测试
 */
class ProcessCount: public Module, public ModuleCreator<ProcessCount> {
 public:
  ProcessCount(const std::string &name) : Module(name) {}
  ~ProcessCount() {}
  bool Open(ModuleParamSet params) override {
    return true;
  }
  void Close() override {
    LOGI(ProcessCount) << "Close";
  }
  void OnEos(const std::string& stream_id) override {
    LOGI(ProcessCount) << "OnEos: " << stream_id;
  }
  int Process(std::shared_ptr<FrameInfo> frame_info) override {
    DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
      LOGE(ProcessCount) << "frame is empty";
      return -1;
    }
    frame_count_++;
    int current_frame_id = stoi(frame_info->frame_id_s);
    if (last_frame_id_ != -1) {
      EXPECT_EQ(current_frame_id, last_frame_id_ + 1);
    }
    last_frame_id_ = current_frame_id;
    return 0;
  }  // Process

 private:
  int frame_count_ = 0;
  int last_frame_id_ = -1;
};
REGISTER_MODULE(ProcessCount);


}  // namespace cnstream

#endif
