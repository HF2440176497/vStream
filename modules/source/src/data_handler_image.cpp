
#include "cnstream_source.hpp"  // DataSource
#include "data_handler_image.hpp"

namespace cnstream {

std::shared_ptr<SourceHandler> ImageHandler::Create(DataSource *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SOURCE) << "[" << stream_id << "]: module_ null";
    return nullptr;
  }
  return std::shared_ptr<ImageHandler>(new ImageHandler(module, stream_id));
}

ImageHandler::ImageHandler(DataSource *module, const std::string &stream_id)
    : SourceHandler(module, stream_id) {
  impl_ = new ImageHandlerImpl(module, this);
}

ImageHandler::~ImageHandler() {
  Close();
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

bool ImageHandler::Open() {
  if (!module_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: module_ null";
    return false;
  }
  if (!impl_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: File handler open failed, no memory left";
    return false;
  }
  if (stream_index_ == INVALID_STREAM_IDX) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Invalid stream_idx";
    return false;
  }
  return impl_->Open();
}

void ImageHandler::Close() {
  if (impl_) {
    impl_->Close();  // for image_impl: close consumer thread
  }
}

void ImageHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

// Note: not use，handler may carry additional params
void ImageHandler::RegisterHandlerParams() {
  param_register_.Register(KEY_FILE_PATH, "Path to the image file.");
  param_register_.Register(KEY_FRAME_RATE, "Framerate for image display. Default is 5.");
}

bool ImageHandler::CheckHandlerParams(const ModuleParamSet& params) {
  if (params.find(KEY_FILE_PATH) == params.end()) {
    LOGE(SOURCE) << "[ImageHandler] file_path is required";
    return false;
  }
  if (access(params.at(KEY_FILE_PATH).c_str(), F_OK) == -1) {
    LOGE(SOURCE) << "[ImageHandler] file not found: " << params.at(KEY_FILE_PATH);
    return false;
  }
  if (params.find(KEY_FRAME_RATE) == params.end()) {
    LOGE(SOURCE) << "[ImageHandler] frame_rate is required";
    return false;
  }
  return true;
}

/**
 * @brief 保留来自 module 的 params_set_
 */
bool ImageHandler::SetHandlerParams(const ModuleParamSet& params) {
  if (impl_) {
    impl_->param_set_ = params;  // SourceModule param_set_
  }
  return true;
}

bool ImageHandlerImpl::Open() {
  // if you need something, just get it
  image_path_ = param_set_.at(KEY_FILE_PATH);
  frame_rate_ = std::stoi(param_set_.at(KEY_FRAME_RATE));
  if (image_path_.empty() || access(image_path_.c_str(), F_OK) == -1) {
    LOGE(SOURCE) << "ImageHandlerImpl: Image path not found: " << image_path_;
    return false;
  }
  image_ = cv::imread(image_path_);
  if (image_.empty()) {
    LOGE(SOURCE) << "ImageHandlerImpl: Failed to load image: " << image_path_;
    return false;
  }
  if (image_.type() != CV_8UC3 || image_.elemSize() != 3) {
    LOGE(SOURCE) << "ImageHandlerImpl: Image format is not BGR24!";
    return false;
  }
  running_.store(true);
  thread_ = std::thread(&ImageHandlerImpl::Loop, this);
  return true;
}

void ImageHandlerImpl::Stop() {
  if (running_.load()) {
    running_.store(false);
  }
}

void ImageHandlerImpl::Close() {
  Stop();
  if (thread_.joinable()) {
    thread_.join();
  }
}

/**
 * @brief 循环读取图片，模拟 decode 生成 DecodeFrame
 * 调用处：ImageHandlerImpl::Open
 */
void ImageHandlerImpl::Loop() {
  if (image_.empty()) {
    LOGE(SOURCE) << "ImageHandlerImpl: Failed to load image: " << image_path_;
    return;
  }
  FrController controller(frame_rate_);
  if (frame_rate_ > 0) controller.Start();

  while (running_.load()) {
    // 每次循环创建新的 DecodeFrame
    DecodeFrame frame(image_.rows, image_.cols, DataFormat::PIXEL_FORMAT_BGR24);
    frame.device_type = DevType::CPU;
    frame.device_id = -1;
    frame.planeNum = 1;  // BGR格式使用1个平面

    size_t data_size = image_.total() * image_.elemSize();
    // size_t data_size = get_plane_bytes(frame.fmt, 0, frame.height, frame.stride);

    uint8_t* buffer = new (std::nothrow) uint8_t[data_size];
    if (!buffer) {
      LOGE(SOURCE) << "ImageHandlerImpl: Failed to allocate memory for image data";
      return;
    }
    if (image_.isContinuous()) {
      memcpy(buffer, image_.data, image_.total() * image_.elemSize());
    } else {
      for (int i = 0; i < image_.rows; ++i) {
        memcpy(buffer + i * image_.cols * image_.elemSize(), 
              image_.ptr(i), 
              image_.cols * image_.elemSize());
      }
    }
#ifdef UNIT_TEST
    LOGI(SOURCE) << "ImageHandlerImpl: Loop; image width: " << image_.cols << ", height: " << image_.rows << ", alloca data_size: " << data_size;
#endif

    frame.stride[0] = frame.width * image_.elemSize();  // BGR格式每个像素3字节
    frame.plane[0] = buffer;
    frame.buf_ref = std::make_unique<MatBufRef>(buffer);  // 交给 MatBufRef 管理释放

    controller.Control();
    frame.pts += 1000 / frame_rate_;
    std::shared_ptr<FrameInfo> data = OnDecodeFrame(&frame);
    if (!module_ || !handler_) {
      LOGE(SOURCE) << "ImageHandler: [" << stream_id_ << "]: module_ or handler_ is null";
      break;
    }
    handler_->SendData(data);
  }
  OnEndFrame();
}

/**
 * 定义如何处理来自数据源图像
 * 调用处：Loop 线程
 */
std::shared_ptr<FrameInfo> ImageHandlerImpl::OnDecodeFrame(DecodeFrame* frame) {
  if (!frame) {
    LOGE(SOURCE) << "[ImageHandlerImpl] OnDecodeFrame function frame is nullptr.";
    return nullptr;
  }
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo();
  if (!data) {
    LOGE(SOURCE) << "[ImageHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return nullptr;
  }
  data->timestamp = frame->pts;
  if (!frame->valid) {
    data->flags = static_cast<size_t>(DataFrameFlag::FRAME_FLAG_INVALID);
    this->SendFrameInfo(data);
    return nullptr;
  }
  int ret = SourceRender::Process(data, frame, frame_id_++);
  if (ret < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: SetupDataFrame function, failed to setup data frame.";
    return nullptr;
  }
  return data;
}

/**
 * Handler 线程循环结束时, 发送结束帧
 */
void ImageHandlerImpl::OnEndFrame() {
  // 调用 SourceRender::OnEndFrame 发送 EOS 帧
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo(true);
  if (!data) {
    LOGW(SOURCE) << "[FileHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return;
  }
  this->SendFrameInfo(data);
  LOGI(SOURCE) << "[ImageHandlerImpl] OnEndFrame function, send end frame.";
}


}  // namespace cnstream
