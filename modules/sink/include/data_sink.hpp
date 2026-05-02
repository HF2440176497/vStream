
#ifndef MODULES_DATA_SINK_HPP_
#define MODULES_DATA_SINK_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cnstream_config.hpp"
#include "cnstream_sink.hpp"
#include "data_common.hpp"

namespace cnstream {

inline const std::string key_output_url = "url";
inline const std::string key_output_format = "format";
inline const std::string key_queue_size = "queue_size";

inline const std::set<std::string> key_supported_formats = {"mp4", "flv"};

/*!
 * @class DataSink
 *
 * @brief DataSink is a class to handle output data.
 *
 * @note It is always the last module in a pipeline.
 */
class DataSink : public SinkModule, public ModuleCreator<DataSink> {
 public:
  explicit DataSink(const std::string &moduleName);
  ~DataSink();

  bool Open(ModuleParamSet paramSet) override;
  void Close() override;
  bool CheckParamSet(const ModuleParamSet &paramSet) const override;

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
};  // class DataSink

REGISTER_MODULE(DataSink);


class PushHandlerImpl;

class PushHandler : public SinkHandler {
 public:
  static std::shared_ptr<SinkHandler> Create(DataSink *module, const std::string &stream_id);
  ~PushHandler();

  bool Open() override;
  void Stop() override;
  void Close() override;
  int Process(const std::shared_ptr<FrameInfo> data) override;

  void RegisterHandlerParams() override;
  bool CheckHandlerParams(const ModuleParamSet& params) override;
  bool SetHandlerParams(const ModuleParamSet& params) override;

 private:
  explicit PushHandler(DataSink *module, const std::string &stream_id);

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  PushHandlerImpl* impl_ = nullptr;
};  // class PushHandler


class QueueHandlerImpl;

class QueueHandler : public SinkHandler {
 public:
  static std::shared_ptr<SinkHandler> Create(DataSink *module, const std::string &stream_id);
  ~QueueHandler();

  bool Open() override;
  void Stop() override;
  void Close() override;
  int Process(const std::shared_ptr<FrameInfo> data) override;

  void RegisterHandlerParams() override;
  bool CheckHandlerParams(const ModuleParamSet& params) override;
  bool SetHandlerParams(const ModuleParamSet& params) override;

  /**
   * @brief Gets output data from the internal queue (reference version).
   *
   * @param[out] data The output data.
   * @param[in] wait_ms Wait time in milliseconds. -1 means block, 0 means non-block, >0 means wait specified ms.
   * @return true if data is retrieved successfully, false otherwise.
   */
  bool GetData(s_output_data& data, int wait_ms = 0);

  /**
   * @brief Gets output data from the internal queue (return value version).
   *
   * @param[in] wait_ms Wait time in milliseconds. -1 means block, 0 means non-block, >0 means wait specified ms.
   * @return The output data. Returns default-constructed s_output_data on failure.
   */
  s_output_data GetData(int wait_ms = 0);

 private:
  explicit QueueHandler(DataSink *module, const std::string &stream_id);

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  QueueHandlerImpl* impl_ = nullptr;
};  // class QueueHandler

}  // namespace cnstream

#endif  // MODULES_DATA_SINK_HPP_
