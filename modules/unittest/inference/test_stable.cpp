
/**
 * 测试稳定性:
 * Infer 模块并行度 = 2, 检验在 stream 间的是否存在干扰
 * 检查高帧率情况下的队列, 检查性能统计, 队列长度情况
 * 检查频繁启停, 内存使用情况
 */

#include "base.hpp"

#include "data_source_param.hpp"
#include "cnstream_frame_va.hpp"
#include "data_source.hpp"
#include "data_handler_image.hpp"
#include "cnstream_pipeline.hpp"

#include "cuda/inspect_mem.hpp"

#include "common.hpp"
#include "inference.hpp"
#include "infer_params.hpp"

#include <csignal>
#include <opencv2/opencv.hpp>


namespace cnstream {

static std::string test_pipeline_json = "pipeline_stable.json";

class StableTest : public testing::Test {
 protected:
  virtual void SetUp() {
    pipeline_ = std::make_shared<Pipeline>("pipeline");
    EXPECT_NE(pipeline_, nullptr);
    EXPECT_TRUE(pipeline_->BuildPipelineByJSONFile(test_pipeline_json));
  }

  virtual void TearDown() {  // 当前用例结束
    LOGI(StableTest) << "TearDown";
    if (pipeline_) {
      pipeline_->Stop();
    }
    image_handler_.reset();
  }

 protected:
  const std::string             stream_id_1_ = "channel-1";
  const std::string             stream_id_2_ = "channel-2";
  const std::string             stream_id_3_ = "channel-3";
  std::vector<std::string>      stream_ids_ = {stream_id_1_, stream_id_2_, stream_id_3_};

  std::shared_ptr<ImageHandler> image_handler_ = nullptr;
  std::shared_ptr<DataSource>   module_ = nullptr;
  std::shared_ptr<Pipeline>     pipeline_ = nullptr;
};


TEST_F(StableTest, MultiStream) {

  int device_id = 0;
  CudaMemInspect inspect(device_id);
  bool force_exit = false;

  EXPECT_TRUE(pipeline_->Start());

  DataSource *source = dynamic_cast<DataSource*>(pipeline_->GetModule("decoder"));
  EXPECT_NE(source, nullptr);

  for (auto stream_id : stream_ids_) {
    std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_id);
    auto handler = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
    EXPECT_NE(handler, nullptr);
    EXPECT_EQ(source->AddSource(handler), 0);
    EXPECT_TRUE(handler->impl_->running_);
  }

  auto inference_module = pipeline_->GetModule("Inference");
  EXPECT_NE(inference_module, nullptr);

  std::this_thread::sleep_for(std::chrono::seconds(10));

  auto profiler = inference_module->GetProfiler();
  if (profiler) {
    auto profile = profiler->GetProcessProfile(key_profile_inference);
    LOGI(StableTest) << "Inference profile: " << profile;
  }

  std::this_thread::sleep_for(std::chrono::seconds(10));
  LOGI(StableTest) << "Inspect brief info: " << inspect.GetBriefInfo();;

  std::this_thread::sleep_for(std::chrono::seconds(10));
  if (!force_exit) {
    pipeline_->Stop();
  } else {
    system("pause");
  }

}

namespace {
    std::atomic<bool> g_sigint_received{false};
    extern "C" void SafeSigintHandler(int) {
        g_sigint_received.store(true, std::memory_order_release);
    }
}

class StableTestWithSigint : public testing::Test {
protected:
    static constexpr int kStopTimeoutSec = 3;   // Stop() 最大容忍时间
    static constexpr int kCaseTimeoutSec = 10;   // 单条用例总超时

    void SetUp() override {
        prev_sigint_ = std::signal(SIGINT, SafeSigintHandler);
        g_sigint_received.store(false);

        pipeline_ = std::make_shared<Pipeline>("pipeline");
        ASSERT_NE(pipeline_, nullptr);
        ASSERT_TRUE(pipeline_->BuildPipelineByJSONFile(test_pipeline_json));
        stop_done_ = false;
    }

    void TearDown() override {
        StopWatchdog();

        // 异步 Stop()，确保在测试结束时能够及时清理
        if (pipeline_ && !stop_done_) {
            LOGI(StableTestWithSigint) << "TearDown: forcing Stop()";
            auto future = std::async(std::launch::async, [this]() {
                pipeline_->Stop();
            });
            if (future.wait_for(std::chrono::seconds(10)) == std::future_status::timeout) {
                LOGI(StableTestWithSigint) << "TearDown: Stop() timeout, possible deadlock!";
            }
        }

        image_handler_.reset();
        module_.reset();
        pipeline_.reset();
        std::signal(SIGINT, prev_sigint_);
    }

    // 带超时的 Stop 包装
    bool StopWithTimeout(int timeout_sec) {
        auto future = std::async(std::launch::async, [this]() {
            pipeline_->Stop();
            stop_done_ = true;
        });

        if (future.wait_for(std::chrono::seconds(timeout_sec)) == std::future_status::timeout) {
            LOGI(StableTestWithSigint) << "Stop() blocked over " << timeout_sec << "s, possible deadlock!";
            return false;
        }
        return true;
    }


    // 独立线程监控整个测试用例的执行时间。如果超过阈值，
    // 直接调用 _Exit(1) 终止进程（不调用析构函数，模拟异常终止）。
    void StartWatchdog(int timeout_sec) {
        watchdog_exit_.store(false);
        watchdog_ = std::thread([this, timeout_sec]() {
            auto start = std::chrono::steady_clock::now();
            while (!watchdog_exit_.load(std::memory_order_acquire)) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start).count();
                if (elapsed >= timeout_sec) {
                    LOGI(StableTestWithSigint) << "WATCHDOG: case hung for " << elapsed << "s, abort!";
                    _Exit(1);  // 不触发析构，用于外部检测残留
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    void StopWatchdog() {
        watchdog_exit_.store(true, std::memory_order_release);
        if (watchdog_.joinable()) {
            watchdog_.join();
        }
    }

    int GetThreadCount() {
        std::ifstream fs("/proc/self/status");
        std::string line;
        while (std::getline(fs, line)) {
            if (line.compare(0, 8, "Threads:") == 0) {
                return std::stoi(line.substr(8));
            }
        }
        return -1;
    }

protected:
    std::shared_ptr<ImageHandler> image_handler_ = nullptr;
    std::shared_ptr<DataSource>   module_ = nullptr;
    std::shared_ptr<Pipeline>     pipeline_ = nullptr;
    std::atomic<bool>             stop_done_{false};

private:
    std::thread       watchdog_;
    std::atomic<bool> watchdog_exit_{false};
    void (*prev_sigint_)(int) = nullptr;
};

// 用例 1：停止
TEST_F(StableTestWithSigint, GracefulStop) {
    ASSERT_TRUE(pipeline_->Start());
    std::this_thread::sleep_for(std::chrono::seconds(2));

    EXPECT_TRUE(StopWithTimeout(kStopTimeoutSec))
        << "Stop() 卡死超过 " << kStopTimeoutSec << " 秒";
    EXPECT_TRUE(stop_done_);
}

// 用例 2：运行中收到 SIGINT
TEST_F(StableTestWithSigint, SigintWhileRunning) {
    ASSERT_TRUE(pipeline_->Start());
    StartWatchdog(kCaseTimeoutSec);

    // 后台线程延迟几秒后发送 SIGINT
    std::thread([]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::raise(SIGINT);
    }).detach();

    // 轮询等待信号到达（最多等 10 秒）
    bool caught = false;
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start).count() < 10) {
        if (g_sigint_received.load(std::memory_order_acquire)) {
            caught = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_TRUE(caught) << "未成功捕获 SIGINT";

    if (caught) {
        LOGI(StableTestWithSigint) << "SIGINT caught, now calling Stop()";
        EXPECT_TRUE(StopWithTimeout(kStopTimeoutSec))
            << "收到 SIGINT 后 Stop() 卡死";
    }
    StopWatchdog();
}

// 用例 3：Stop 执行过程中收到 SIGINT
TEST_F(StableTestWithSigint, SigintDuringStop) {
    ASSERT_TRUE(pipeline_->Start());
    std::this_thread::sleep_for(std::chrono::seconds(3));
    StartWatchdog(kCaseTimeoutSec);

    // 独立线程执行 Stop
    std::atomic<bool> stop_started{false};
    std::thread stop_thread([this, &stop_started]() {
        stop_started = true;
        pipeline_->Stop();
        stop_done_ = true;
    });
    // 很短时间内发送 SIGINT
    while (!stop_started.load()) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::raise(SIGINT);

    stop_thread.join();
    EXPECT_TRUE(stop_done_) << "Stop() 在 SIGINT 干扰下未能完成，疑似死锁";

    StopWatchdog();
}

// 用例 4：Stop 死锁/超时检测
TEST_F(StableTestWithSigint, StopHangDetection) {
    ASSERT_TRUE(pipeline_->Start());
    std::this_thread::sleep_for(std::chrono::seconds(2));

    EXPECT_TRUE(StopWithTimeout(kStopTimeoutSec))
        << "Stop() 执行超过 " << kStopTimeoutSec << " 秒，疑似死锁";
    EXPECT_TRUE(stop_done_);
}


}  // namespace cnstream