
#ifndef CNSTREAM_SINK_HPP_
#define CNSTREAM_SINK_HPP_

/**
 * @file cnstream_sink.hpp
 *
 * This file contains a declaration of the Sink Module class.
 */

#include <atomic>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include "cnstream_common.hpp"
#include "cnstream_config.hpp"
#include "cnstream_module.hpp"
#include "cnstream_frame.hpp"

namespace cnstream {

class SinkHandler;

/*!
 * @class SinkModule
 *
 * @brief SinkModule is the base class of sink modules.
 */
class SinkModule : public Module {
 public:
  /**
   * @brief Constructs a sink module.
   *
   * @param[in] name The name of the sink module.
   *
   * @return No return value.
   */
  explicit SinkModule(const std::string &name) : Module(name) {}
  /**
   * @brief Destructs a sink module.
   *
   * @return No return value.
   */
  virtual ~SinkModule() { RemoveSinks(); }
  /**
   * @brief Adds one stream to SinkModule. This function should be called after pipeline starts.
   *
   * @param[in] handler The sink handler
   *
   * @retval Returns 0 for success, otherwise returns -1.
   */
  int AddSink(std::shared_ptr<SinkHandler> handler);
  /**
   * @brief Gets the handler of the stream.
   *
   * @param[in] stream_id The stream identifier.
   *
   * @return Returns the handler of the stream.
   */
  std::shared_ptr<SinkHandler> GetSinkHandler(const std::string &stream_id);
  /**
   * @brief Removes one stream from SinkModule with given handler.
   *
   * @param[in] handler The handler of one stream.
   * @param[in] force The flag describing the removing behaviour.
   *
   * @retval 0: success.
   */
  int RemoveSink(std::shared_ptr<SinkHandler> handler, bool force = false);
  /**
   * @brief Removes one stream from SinkModule with given the stream identification.
   *
   * @param[in] stream_id The stream identification.
   * @param[in] force The flag describing the removing behaviour.
   *
   * @retval 0: success.
   */
  int RemoveSink(const std::string &stream_id, bool force = false);
  /**
   * @brief Removes all streams from SinkModule.
   *
   * @param[in] force The flag describing the removing behaviour.
   *
   * @retval 0: success.
   */
  int RemoveSinks(bool force = false);

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 protected:
#endif
  /**
   * @brief Dispatches data to the corresponding sink handler by stream_id.
   */
  int DispatchData(const std::shared_ptr<FrameInfo> data);

  friend class SinkHandler;

 protected:
  ModuleParamSet param_set_;

 private:
  int Process(std::shared_ptr<FrameInfo> data) override {
    if (!data) {
      LOGE(SINK) << "data is null";
      return -1;
    }
    return DispatchData(data);
  }
  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<SinkHandler>> sink_map_;
};  // class SinkModule

/**
 * @class SinkHandler
 *
 * @brief SinkHandler is a class that handles various sinks, such as push stream and file output.
 */
class SinkHandler : private NonCopyable {
 public:
  /**
   * @brief Constructs a sink handler.
   *
   * @param[in] module The sink module this handler belongs to.
   * @param[in] stream_id The name of the stream.
   *
   * @return No return value.
   */
  explicit SinkHandler(SinkModule *module, const std::string &stream_id)
      : module_(module), stream_id_(stream_id) {}
  /**
   * @brief Destructs a sink handler.
   *
   * @return No return value.
   */
  virtual ~SinkHandler() = default;

  std::string GetStreamId() const { return stream_id_; }
  /**
   * @brief Opens a sink handler.
   *
   * @return Returns true if a sink handler is opened successfully, otherwise returns false.
   */
  virtual bool Open() = 0;
  /**
   * @brief Closes a sink handler.
   *
   * @return No return value.
   */
  virtual void Close() = 0;
  /**
   * @brief Stops a sink handler. The Close() function should be called afterwards.
   *
   * @return No return value.
   */
  virtual void Stop() {}
  /**
   * @brief Processes data.
   *
   * @param[in] data The data need to be processed by sink handler.
   *
   * @return Returns true if process data successfully, otherwise returns false.
   */
  virtual int Process(const std::shared_ptr<FrameInfo> data) {
    return 0;
  };

  // note: params of handler itself
  virtual void RegisterHandlerParams() {}
  virtual bool CheckHandlerParams(const ModuleParamSet& params) { return true; }
  virtual bool SetHandlerParams(const ModuleParamSet& params) { return true; }

 protected:
  SinkModule *module_ = nullptr;
  mutable std::string stream_id_;
  ParamRegister param_register_;
};

}  // namespace cnstream

#endif  // CNSTREAM_SINK_HPP_
