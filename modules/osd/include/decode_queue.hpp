

#ifndef MODULE_OSD_CONSUMER_QUEUE_HPP
#define MODULE_OSD_CONSUMER_QUEUE_HPP


#include <queue>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>

#include "cnstream_common.hpp"
#include "cnstream_config.hpp"
#include "cnstream_module.hpp"
#include "util/cnstream_queue.hpp"
#include "profiler/module_profiler.hpp"

#include "cnstream_frame_va.hpp"
#include "cnstream_logging.hpp"

#include "data_common.hpp"


namespace cnstream {

inline const std::string key_decode_queue_size = "queue_size";

/**
 * @brief DecodeQueue class
 * @details DecodeQueue is a module that is used to decode the results
 */
class DecodeQueue : public Module, public ModuleCreator<DecodeQueue> {
 public:
  explicit DecodeQueue(const std::string& name);
  ~DecodeQueue();

  bool Open(cnstream::ModuleParamSet paramSet) override;
  void Close() override;
  bool CheckParamSet(const ModuleParamSet& paramSet) const override;
  int  Process(std::shared_ptr<FrameInfo> data) override;
  bool GetData(s_output_data& data, int wait_ms = 0);  // 默认为非阻塞等待
  s_output_data GetData(int wait_ms = 0);

private:
  void OnFrame(std::shared_ptr<FrameInfo> frame_info);
  bool Push(const s_output_data& data);

private:
  std::unique_ptr<ThreadSafeQueue<s_output_data>> queue_;
};

REGISTER_MODULE(DecodeQueue);

}  // namespace cnstream


#endif
