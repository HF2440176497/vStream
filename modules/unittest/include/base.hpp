

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
#include <atomic>

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
 （1）对于配置为 next_nods: [CountOne, CountTwo] 的情况， CountOne 和 CountTwo 是并发执行的
 （2）验证 CountThree 接收的 data 一定是 CountOne 和 CountTwo 处理后的 data
 */
class CountOne: public Module, public ModuleCreator<CountOne> {
 public:
  CountOne(const std::string &name) : Module(name) {}
  ~CountOne() {}
  bool Open(ModuleParamSet params) override {
    return true;
  }
  void Close() override {
    LOGI(CountOne) << "Close";
  }
  void OnEos(const std::string& stream_id) override {
    LOGI(CountOne) << "OnEos: " << stream_id;
  }
  int Process(std::shared_ptr<FrameInfo> frame_info) override {
    DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
      LOGE(CountOne) << "frame is empty";
      return -1;
    }

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
    }
    // 2. 获取 当前 module 对应的 FrameCountData, 自定义赋值
    auto count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_one_name);
    {
      std::lock_guard<std::mutex> lock(count_data->mtx);
      count_data->process_count++;
    }
    return 0;
  }
};
REGISTER_MODULE(CountOne);


class CountTwo: public Module, public ModuleCreator<CountTwo> {
 public:
  CountTwo(const std::string &name) : Module(name) {}
  ~CountTwo() {}
  bool Open(ModuleParamSet params) override {
    return true;
  }
  void Close() override {
    LOGI(CountTwo) << "Close";
  }
  void OnEos(const std::string& stream_id) override {
    LOGI(CountTwo) << "OnEos: " << stream_id;
  }

  int Process(std::shared_ptr<FrameInfo> frame_info) override {
    DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
      LOGE(CountTwo) << "frame is empty";
      return -1;
    }

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
      total_count_data->process_count++;
      // LOGD(CountTwo) << "frame ts: " << frame_info->timestamp << " process_total_count: " << total_count_data->process_count;
    }
    // 2. 获取 当前 module 对应的 FrameCountData, 自定义赋值
    auto count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_two_name);
    {
      std::lock_guard<std::mutex> lock(count_data->mtx);
      count_data->process_count++;
      // LOGD(CountTwo) << "frame ts: " << frame_info->timestamp << " process_two_count: " << count_data->process_count;
    }
    return 0;
  }
};
REGISTER_MODULE(CountTwo);


class CountThree: public Module, public ModuleCreator<CountThree> {
 public:
  CountThree(const std::string &name) : Module(name) {}
  ~CountThree() {}
  bool Open(ModuleParamSet params) override {
    return true;
  }
  void Close() override {
    LOGI(CountThree) << "Close";
  }
  void OnEos(const std::string& stream_id) override {
    LOGI(CountThree) << "OnEos: " << stream_id;
  }
  int Process(std::shared_ptr<FrameInfo> frame_info) override {
    DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
      LOGE(CountThree) << "frame is empty";
      return -1;
    }
    // 经过前两个 module, 才会到达 CountThree 
    if (!frame_info->collection.HasValue(process_total_name)) {
      LOGE(CountThree) << "process_total not found";
      return -1;
    }
    if (!frame_info->collection.HasValue(process_one_name)) {
      LOGE(CountThree) << "process_one not found";
      return -1;
    }
    auto total_count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_total_name);
    if (frame_info->collection.HasValue(process_two_name)) {
      auto one_count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_one_name);
      auto two_count_data = frame_info->collection.Get<std::shared_ptr<FrameCountData>>(process_two_name);
      EXPECT_EQ(one_count_data->process_count, two_count_data->process_count);
      EXPECT_EQ(total_count_data->process_count, 1 + one_count_data->process_count);
      // LOGD(CountThree) << "frame ts: " << frame_info->timestamp << " process_two_count: " << two_count_data->process_count;
    } else {
      LOGE(CountThree) << "process_two not found";
      return -1;
    }
    return 0;
  }
 private:
  std::unordered_map<std::string, std::mutex> mutex_map_;
  std::mutex mtx_;
};
REGISTER_MODULE(CountThree);


/**
 * 提取 frame_info 中的 frame_id_s 得到数字，验证是否连续
 * 配合 test_send 单元测试
 */
class CountModule: public Module, public ModuleCreator<CountModule> {
 public:
  CountModule(const std::string &name) : Module(name) {}
  ~CountModule() {}
  bool Open(ModuleParamSet params) override {
    return true;
  }
  void Close() override {
    LOGI(CountModule) << "Close";
  }
  void OnEos(const std::string& stream_id) override {
    LOGI(CountModule) << "OnEos: " << stream_id;
  }
  int Process(std::shared_ptr<FrameInfo> frame_info) override {
    DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
    if (!frame) {
      LOGE(CountModule) << "frame is empty";
      return -1;
    }
    int current_frame_id = stoi(frame_info->frame_id_s);
    if (last_frame_id_ != -1) {
      EXPECT_EQ(current_frame_id, last_frame_id_ + 1);
    }
    last_frame_id_ = current_frame_id;
    return 0;
  }  // Process

 private:
  int last_frame_id_ = -1;
};
REGISTER_MODULE(CountModule);


}  // namespace cnstream

#endif
