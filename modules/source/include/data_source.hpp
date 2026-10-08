

#ifndef MODULES_DATA_SOURCE_HPP_
#define MODULES_DATA_SOURCE_HPP_


#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cnstream_config.hpp"
#include "cnstream_source.hpp"
#include "data_source_param.hpp"

#include <opencv2/opencv.hpp>

namespace cnstream {

/*!
 * @class DataSource
 *
 * @brief DataSource is a class to handle encoded input data.
 *
 * @note It is always the first module in a pipeline.
 */
class DataSource : public SourceModule, public ModuleCreator<DataSource> {
 public:
  /*!
   * @brief Constructs a DataSource object.
   *
   * @param[in] moduleName The name of this module.
   *
   * @return No return value.
   */
  explicit DataSource(const std::string &moduleName);

  /*!
   * @brief Destructs a DataSource object.
   *
   * @return No return value.
   */
  ~DataSource();

  /*!
   * @brief Initializes the configuration of the DataSource module.
   *
   * This function will be called by the pipeline when the pipeline starts.
   *
   * @param[in] paramSet The module's parameter set to configure a DataSource module.
   *
   * @return Returns true if the parammeter set is supported and valid, othersize returns false.
   */
  bool Open(ModuleParamSet paramSet) override;
  // override Module's virtual function

  /*!
   * @brief Frees the resources that the object may have acquired.
   *
   * This function will be called by the pipeline when the pipeline stops.
   *
   * @return No return value.
   */
  void Close() override;

  /*!
   * @brief Checks the parameter set for the DataSource module.
   *
   * @param[in] paramSet Parameters for this module.
   * 
   * @return Returns true if all parameters are valid. Otherwise, returns false.
   * 
   * @note DataSource::Open 调用
   */
  bool CheckParamSet(const ModuleParamSet &paramSet) const override;

  /**
   * override Module::Process
   */
  int Process(std::shared_ptr<FrameInfo> data) override;

  /*!
   * @brief Gets the parameters of the DataSource module.
   *
   * @return Returns the parameters of this module.
   *
   * @note This function should be called after ``Open`` function.
   */
  DataSourceParam GetSourceParam() const;

  ModuleParamSet GetStreamParams(const std::string& stream_id) const;
  bool LoadStreamConf(const std::string& config_dir_path);

#ifdef VSTREAM_UNIT_TEST
  public:
#else
  private:
#endif
   DataSourceParam param_;
   std::map<std::string, ModuleParamSet> stream_configs_;
};  // class DataSource

REGISTER_MODULE(DataSource);

// 派生关系: Module SourceModule DataSource
// SourceModule 并没有提供虚函数接口, DataSource 主要重写 Module 的相关 virtual func

class ImageHandlerImpl;

class ImageHandler : public SourceHandler {
 public:
  static std::shared_ptr<SourceHandler> Create(DataSource *module, const std::string &stream_id);
  ~ImageHandler();

  bool Open() override;
  void Stop() override;
  void Close() override;

  void RegisterHandlerParams() override;
  bool CheckHandlerParams(const ModuleParamSet& params) override;
  bool SetHandlerParams(const ModuleParamSet& params) override;

 private:
  explicit ImageHandler(DataSource *module, const std::string &stream_id);

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  ImageHandlerImpl* impl_ = nullptr;
};  // class ImageHandler

class PullHandlerIm;

class PullHandler : public SourceHandler {
 public:
  static std::shared_ptr<SourceHandler> Create(DataSource *module, const std::string &stream_id);
  ~PullHandler();

  bool Open() override;
  void Stop() override;
  void Close() override;

  void RegisterHandlerParams() override;
  bool CheckHandlerParams(const ModuleParamSet& params) override;
  bool SetHandlerParams(const ModuleParamSet& params) override;

 private:
  explicit PullHandler(DataSource *module, const std::string &stream_id, DecoderType decoder_type);

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  PullHandlerIm* impl_ = nullptr;
};  // class PullHandler

class SendHandlerImpl;

/**
 * @brief Send 返回码，用于区分失败原因
 */
enum class SendRet {
  OK = 0,             // 入队成功
  ERR_PARAM = -1,     // 参数非法
  ERR_TIMEOUT = -2,   // 队列满且等待超时，可重试（背压）
  ERR_STOPPED = -3,   // 不可重试
};

class SendHandler : public SourceHandler {
 public:
  static std::shared_ptr<SourceHandler> Create(DataSource *module, const std::string &stream_id);
  ~SendHandler();

  bool Open() override;
  void Stop() override;
  void Close() override;

  bool SetHandlerParams(const ModuleParamSet& params) override;

  /**
   * @brief 发送一帧数据
   * @param wait_ms 队列满时的等待策略：
   *                < 0 一直阻塞；
   *                == 0 非阻塞（丢弃式，默认）；
   *                > 0 最多等待 wait_ms 毫秒
   * @return SendRet::OK 成功；ERR_PARAM 参数非法；ERR_TIMEOUT 超时；ERR_STOPPED 已停止
   */
  int Send(const SendFrame& send_frame, int wait_ms = 0);
  int Send(uint64_t pts, std::string frame_id_s, const cv::Mat &image, int wait_ms = 0);

 private:
  explicit SendHandler(DataSource *module, const std::string &stream_id);

#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  SendHandlerImpl* impl_ = nullptr;
};  // class SendHandler


}  // namespace cnstream

#endif