
#include "base.hpp"
#include "profiler/profile.hpp"
#include "profiler/process_profiler.hpp"
#include "profiler/module_profiler.hpp"
#include "profiler/trace_serialize_helper.hpp"


using namespace std;
using namespace std::chrono;

namespace cnstream {

TEST(ProcessProfilerTest, BasicFunctionality) {
  ProfilerConfig config;
  config.enable_profile = true;
  
  ProcessProfiler profiler(config, "test_process");
  EXPECT_EQ(profiler.GetName(), "test_process");
  
  // 每一个 key 代表一个 frame
  RecordKey key1("stream1", 1000);
  RecordKey key2("stream1", 2000);
  RecordKey key3("stream2", 1500);
  
  // Record start times
  profiler.RecordStart(key1);
  profiler.RecordStart(key2);
  profiler.RecordStart(key3);
  
  // Simulate processing delay
  std::this_thread::sleep_for(10ms);
  
  // Record end times
  profiler.RecordEnd(key1);
  profiler.RecordEnd(key2);
  profiler.RecordEnd(key3);
  
  RecordKey key4("stream1", 3000);
  profiler.RecordStart(key4);
  profiler.RecordDropped(key4);
  
  ProcessProfile profile = profiler.GetProfile();
  EXPECT_EQ(profile.process_name, "test_process");
  EXPECT_EQ(profile.counter, 4);
  EXPECT_EQ(profile.completed, 3);
  EXPECT_EQ(profile.dropped, 1);
  EXPECT_GT(profile.avg_latency, 0.0);
  EXPECT_GT(profile.max_latency, 0.0);
  EXPECT_GT(profile.min_latency, 0.0);
  EXPECT_GT(profile.fps, 0.0);
}

// Test ModuleProfiler class
TEST(ModuleProfilerTest, BasicFunctionality) {
  ProfilerConfig config;
  config.enable_profile = true;
  
  ModuleProfiler module_profiler(config, "test_module");
  
  EXPECT_EQ(module_profiler.GetName(), "test_module");

  module_profiler.RegisterProcess("process1");
  module_profiler.RegisterProcess("process2");
  
  RecordKey key1("stream1", 1000);  // timestamp
  RecordKey key2("stream1", 2000);
  
  // Module 中注册两个处理 Processs
  module_profiler.RecordProcessStart("process1", key1);
  std::this_thread::sleep_for(5ms);
  module_profiler.RecordProcessEnd("process1", key1);
  
  // key 代表的是起始记录到 Profiler 的时间戳
  module_profiler.RecordProcessStart("process2", key1);
  std::this_thread::sleep_for(8ms);
  module_profiler.RecordProcessEnd("process2", key1);
  
  module_profiler.RecordProcessStart("process1", key2);
  module_profiler.RecordProcessDropped("process1", key2);
  
  ModuleProfile module_profile = module_profiler.GetProfile();
  
  // Verify module profile data
  EXPECT_EQ(module_profile.module_name, "test_module");
  EXPECT_EQ(module_profile.process_profiles.size(), 2);
  
  // Find process1 profile
  bool found_process1 = false;
  bool found_process2 = false;
  
  for (const auto& process_profile : module_profile.process_profiles) {
    if (process_profile.process_name == "process1") {
      found_process1 = true;
      EXPECT_EQ(process_profile.counter, 2);
      EXPECT_EQ(process_profile.completed, 1);
      EXPECT_EQ(process_profile.dropped, 1);
      EXPECT_GT(process_profile.avg_latency, 0.0);
    } else if (process_profile.process_name == "process2") {
      found_process2 = true;
      EXPECT_EQ(process_profile.counter, 1);
      EXPECT_EQ(process_profile.completed, 1);
      EXPECT_EQ(process_profile.dropped, 0);
      EXPECT_GT(process_profile.avg_latency, 0.0);
    }
  }
  EXPECT_TRUE(found_process1);
  EXPECT_TRUE(found_process2);
}

// Test ModuleProfiler with multiple streams
TEST(ModuleProfilerTest, MultiStream) {
  // Create a profiler configuration
  ProfilerConfig config;
  config.enable_profile = true;
  
  // Create a ModuleProfiler instance
  ModuleProfiler module_profiler(config, "multi_stream_module");
  
  // Register a process
  module_profiler.RegisterProcess("main_process");
  
  // Record frame processing for multiple streams
  std::vector<std::string> streams = {"stream1", "stream2", "stream3"};
  
  for (int i = 0; i < 5; ++i) {
    for (const auto& stream : streams) {
      RecordKey key(stream, i * 1000);
      module_profiler.RecordProcessStart("main_process", key);
      
      // Simulate varying processing times
      std::this_thread::sleep_for(std::chrono::milliseconds(2 + i));
      module_profiler.RecordProcessEnd("main_process", key);
    }
  }
  ModuleProfile module_profile = module_profiler.GetProfile();
  EXPECT_EQ(module_profile.module_name, "multi_stream_module");
  EXPECT_EQ(module_profile.process_profiles.size(), 1);
  
  // 表示：当前这个 Process 会处理来自三个 stream 的，总共 15 个 frame
  const auto& process_profile = module_profile.process_profiles[0];
  EXPECT_EQ(process_profile.process_name, "main_process");
  EXPECT_EQ(process_profile.counter, 15);
  EXPECT_EQ(process_profile.completed, 15);
  EXPECT_EQ(process_profile.dropped, 0);
  EXPECT_GT(process_profile.avg_latency, 0.0);
  EXPECT_GT(process_profile.fps, 0.0);
}

// Test TraceSerializeHelper class
TEST(TraceSerializeHelperTest, Serialization) {
  ModuleProfile module_profile;
  module_profile.module_name = "test_module";
  
  ProcessProfile process_profile1;
  process_profile1.process_name = "process1";
  process_profile1.counter = 10;
  process_profile1.completed = 8;
  process_profile1.dropped = 2;
  process_profile1.avg_latency = 15.5;
  process_profile1.max_latency = 25.0;
  process_profile1.min_latency = 10.0;
  process_profile1.fps = 60.0;
  
  ProcessProfile process_profile2;
  process_profile2.process_name = "process2";
  process_profile2.counter = 10;
  process_profile2.completed = 10;
  process_profile2.dropped = 0;
  process_profile2.avg_latency = 10.2;
  process_profile2.max_latency = 18.0;
  process_profile2.min_latency = 5.0;
  process_profile2.fps = 95.5;
  
  module_profile.process_profiles.push_back(process_profile1);
  module_profile.process_profiles.push_back(process_profile2);

  TraceSerializeHelper serializer;
  serializer.Serialize(module_profile);
  std::string json_str = serializer.ToJsonStr();
  std::cout << "JSON String: " << json_str << std::endl;

  EXPECT_TRUE(json_str.find("test_module") != std::string::npos);
  EXPECT_TRUE(json_str.find("process1") != std::string::npos);
  EXPECT_TRUE(json_str.find("process2") != std::string::npos);
  EXPECT_TRUE(json_str.find("15.5") != std::string::npos);
  EXPECT_TRUE(json_str.find("10.2") != std::string::npos);
  
  serializer.Reset();
  std::string reset_json = serializer.ToJsonStr();
  EXPECT_TRUE(reset_json.find("test_module") == std::string::npos);
}

// 验证格式化打印
TEST(ProcessProfilerTest, PrintString) {
  ModuleProfile module_profile;
  module_profile.module_name = "test_module";
  
  ProcessProfile process_profile1;
  process_profile1.process_name = "process1";
  process_profile1.counter = 10;
  process_profile1.completed = 8;
  process_profile1.dropped = 2;
  process_profile1.avg_latency = 15.5;
  process_profile1.max_latency = 25.0;
  process_profile1.min_latency = 10.0;
  process_profile1.fps = 60.0;
  
  ProcessProfile process_profile2;
  process_profile2.process_name = "process2";
  process_profile2.counter = 10;
  process_profile2.completed = 10;
  process_profile2.dropped = 0;
  process_profile2.avg_latency = 10.2;
  process_profile2.max_latency = 18.0;
  process_profile2.min_latency = 5.0;
  process_profile2.fps = 95.5;
  
  module_profile.process_profiles.push_back(process_profile1);
  module_profile.process_profiles.push_back(process_profile2);

  std::cout << "Module Profile String: " << ModuleProfileToString(module_profile) << std::endl;
}

/**
 * @brief 启动 NumThreads 线程，
 * 每个线程都对 process_profiler 进行操作 
 */
TEST(ProcessProfilerTest, ThreadSafety) {
  ProfilerConfig config;
  config.enable_profile = true;
  
  ProcessProfiler profiler(config, "thread_safe_process");
  const int kNumThreads = 4;
  const int kIterationsPerThread = 100;
  
  std::vector<std::thread> threads;
  
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&profiler, i]() {
      for (int j = 0; j < kIterationsPerThread; ++j) {
        RecordKey key(std::string("stream") + std::to_string(i), j * 1000 + i);
        profiler.RecordStart(key);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        if ((j % 10) == 0) {
          profiler.RecordDropped(key);
        } else {
          profiler.RecordEnd(key);
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  ProcessProfile profile = profiler.GetProfile();
  EXPECT_EQ(profile.process_name, "thread_safe_process");

  // 每个线程都 kIterationsPerThread 次操作，因此 (j % 10) == 0 会触发  kIterationsPerThread / 10 次 dropped
  EXPECT_EQ(profile.counter, int(kNumThreads * kIterationsPerThread));
  EXPECT_EQ(profile.completed, int(kNumThreads * kIterationsPerThread * 9 / 10));
  EXPECT_EQ(profile.dropped, int(kNumThreads * kIterationsPerThread / 10));
  EXPECT_GT(profile.avg_latency, 0.0);
}

/**
 * @brief 滑动窗口：滑出窗口的记录不再计入统计
 */
TEST(ProcessProfilerTest, SlidingWindowEviction) {
  ProfilerConfig config;
  config.enable_profile = true;
  config.window_ms = 200;
  config.bucket_ms = 100;

  ProcessProfiler profiler(config, "window_process");
  RecordKey key1("stream1", 1000);
  RecordKey key2("stream1", 2000);
  profiler.RecordStart(key1);
  profiler.RecordStart(key2);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  profiler.RecordEnd(key1);
  profiler.RecordEnd(key2);

  ProcessProfile profile = profiler.GetProfile();
  EXPECT_EQ(profile.window_ms, 200u);
  EXPECT_EQ(profile.completed, 2u);
  EXPECT_EQ(profile.counter, 2u);
  EXPECT_GT(profile.fps, 0.0);

  // 等待超过 1 个窗口，旧记录应被滑出
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  profile = profiler.GetProfile();
  EXPECT_EQ(profile.completed, 0u);
  EXPECT_EQ(profile.counter, 0u);
  EXPECT_DOUBLE_EQ(profile.fps, 0.0);
  EXPECT_EQ(profile.inflight, 0u);
}

/**
 * @brief 未结束记录：inflight 与 oldest_inflight_ms 能反映“正在阻塞”
 */
TEST(ProcessProfilerTest, InflightMetrics) {
  ProfilerConfig config;
  config.enable_profile = true;
  config.window_ms = 1000;
  config.bucket_ms = 200;

  ProcessProfiler profiler(config, "inflight_process");
  RecordKey key("stream1", 1000);
  profiler.RecordStart(key);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  ProcessProfile profile = profiler.GetProfile();
  EXPECT_EQ(profile.completed, 0u);
  EXPECT_EQ(profile.inflight, 1u);
  EXPECT_GE(profile.oldest_inflight_ms, 40.0);

  profiler.RecordEnd(key);
  profile = profiler.GetProfile();
  EXPECT_EQ(profile.completed, 1u);
  EXPECT_EQ(profile.inflight, 0u);
}

/**
 * @brief 残留记录：超过 3 个窗口仍未结束的记录会被清理并计入 purged
 */
TEST(ProcessProfilerTest, PurgeStaleRecords) {
  ProfilerConfig config;
  config.enable_profile = true;
  config.window_ms = 200;
  config.bucket_ms = 100;

  ProcessProfiler profiler(config, "purge_process");
  RecordKey stale("stream1", 1000);
  profiler.RecordStart(stale);
  // 超过 3 * window_ms，且跨过 1s 的清理节流
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));

  RecordKey fresh("stream1", 2000);
  profiler.RecordStart(fresh);

  ProcessProfile profile = profiler.GetProfile();
  EXPECT_EQ(profile.purged, 1u);
  EXPECT_EQ(profile.inflight, 1u);
  profiler.RecordEnd(fresh);
}

}