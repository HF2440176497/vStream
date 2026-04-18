

#ifndef MODULES_DATA_SOURCE_HPP_
#define MODULES_DATA_SOURCE_HPP_


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

#ifdef UNIT_TEST
  public:
#else
  private:
#endif
   DataSourceParam param_;
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

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  ImageHandlerImpl* impl_ = nullptr;
};  // class ImageHandler

class VideoHandlerImpl;

class VideoHandler : public SourceHandler {
 public:
  static std::shared_ptr<SourceHandler> Create(DataSource *module, const std::string &stream_id);
  ~VideoHandler();

  bool Open() override;
  void Stop() override;
  void Close() override;

  void RegisterHandlerParams() override;
  bool CheckHandlerParams(const ModuleParamSet& params) override;
  bool SetHandlerParams(const ModuleParamSet& params) override;

 private:
  explicit VideoHandler(DataSource *module, const std::string &stream_id);

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  VideoHandlerImpl* impl_ = nullptr;
};  // class VideoHandler

class SendHandlerImpl;

class SendHandler : public SourceHandler {
 public:
  static std::shared_ptr<SourceHandler> Create(DataSource *module, const std::string &stream_id);
  ~SendHandler();

  bool Open() override;
  void Stop() override;
  void Close() override;

  bool SetHandlerParams(const ModuleParamSet& params) override;
  int Send(const SendFrame& send_frame);
  int Send(uint64_t pts, std::string frame_id_s, const cv::Mat &image);

 private:
  explicit SendHandler(DataSource *module, const std::string &stream_id);

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  SendHandlerImpl* impl_ = nullptr;
};  // class SendHandler


}  // namespace cnstream

#endif