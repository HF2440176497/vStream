
#include "cnstream_source.hpp"  // DataSource
#include "data_handler_send.hpp"


namespace cnstream {

std::shared_ptr<SourceHandler> SendHandler::Create(DataSource *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SOURCE) << "[" << stream_id << "]: module_ null";
    return nullptr;
  }
  return std::shared_ptr<SendHandler>(new SendHandler(module, stream_id));
}

SendHandler::SendHandler(DataSource *module, const std::string &stream_id)
    : SourceHandler(module, stream_id) {
  impl_ = new SendHandlerImpl(module, this);
}

SendHandler::~SendHandler() {
  Close();
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

int SendHandler::Send(const SendFrame& send_frame) {
  if (send_frame.image.empty()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: image is empty";
    return -1;
  }
  if (impl_->Push(send_frame)) {
    return 0;
  }
  LOGW(SOURCE) << "[" << stream_id_ << "]: send frame failed";
  return -1;
}

int SendHandler::Send(uint64_t pts, std::string frame_id_s, const cv::Mat &image) {
  if (!impl_) {
    LOGE(SOURCE) << "[" << stream_id_ << "] handler is not valid";
    return -1;
  }
  if (image.empty()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: image is not valid";
    return -1;
  }
  if (impl_->Push(SendFrame{pts, frame_id_s, image})) {
    return 0;
  }
  LOGW(SOURCE) << "[" << stream_id_ << "]: send frame failed";
  return -1;
}


void SendHandler::Close() {
  if (impl_) {
    impl_->Close();  // for image_impl: close consumer thread
  }
}

void SendHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

bool SendHandler::Open() {
  if (!module_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: module_ null";
    return false;
  }
  if (!impl_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Video handler open failed, no memory left";
    return false;
  }
  if (stream_index_ == INVALID_STREAM_IDX) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Invalid stream_idx";
    return false;
  }
  return impl_->Open();
}

bool SendHandler::SetHandlerParams(const ModuleParamSet& params) {
  if (impl_) {
    impl_->param_set_ = params;  // SourceModule param_set_
  }
  return true;
}

bool SendHandlerImpl::Push(const SendFrame& send_frame) {
  return image_queue_.Push(send_frame);
}

bool SendHandlerImpl::Open() {
  running_.store(true);
  thread_ = std::thread(&SendHandlerImpl::Loop, this);
  return true;
}

void SendHandlerImpl::Stop() {
  image_queue_.Stop();
  if (running_.load()) {
    running_.store(false);
  }
}

void SendHandlerImpl::Close() {
  Stop();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void SendHandlerImpl::Loop() {

  while (running_.load()) {
    SendFrame send_frame;
    if (!image_queue_.TryPop(send_frame)) {  // Non block pop
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    
    DecodeFrame frame(send_frame.image.rows, send_frame.image.cols, DataFormat::PIXEL_FORMAT_BGR24);
    frame.device_type = DevType::CPU;
    frame.device_id = -1;
    frame.planeNum = 1;  // BGR格式使用1个平面

    size_t data_size = send_frame.image.total() * send_frame.image.elemSize();
    // size_t data_size = get_plane_bytes(frame.fmt, 0, frame.height, frame.stride);

    uint8_t* buffer = new (std::nothrow) uint8_t[data_size];
    if (!buffer) {
      LOGE(SOURCE) << "SendHandlerImpl: Failed to allocate memory for image data";
      return;
    }
    if (send_frame.image.isContinuous()) {
      memcpy(buffer, send_frame.image.data, send_frame.image.total() * send_frame.image.elemSize());
    } else {
      for (int i = 0; i < send_frame.image.rows; ++i) {
        memcpy(buffer + i * send_frame.image.cols * send_frame.image.elemSize(), 
              send_frame.image.ptr(i), 
              send_frame.image.cols * send_frame.image.elemSize());
      }
    }
#ifdef VSTREAM_UNIT_TEST
    LOGD(SOURCE) << "SendHandlerImpl: Loop; image width: " << send_frame.image.cols << ", height: " << send_frame.image.rows << ", alloca data_size: " << data_size;
#endif

    frame.stride[0] = frame.width * send_frame.image.elemSize();  // BGR格式每个像素3字节
    frame.plane[0] = buffer;
    frame.buf_ref = std::make_unique<MatBufRef>(buffer);

    frame.pts = send_frame.pts;
    frame.frame_id_s = send_frame.frame_id_s;
    std::shared_ptr<FrameInfo> data = OnDecodeFrame(&frame);
    if (!module_ || !handler_) {
      LOGE(SOURCE) << "SendHandlerImpl: [" << stream_id_ << "]: module_ or handler_ is null";
      break;
    }
    handler_->SendData(data);
  }
  OnEndFrame();
}


std::shared_ptr<FrameInfo> SendHandlerImpl::OnDecodeFrame(DecodeFrame* frame) {
  if (!frame) {
    LOGE(SOURCE) << "[SendHandlerImpl] OnDecodeFrame function frame is nullptr.";
    return nullptr;
  }
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo();
  if (!data) {
    LOGE(SOURCE) << "[SendHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return nullptr;
  }
  data->timestamp = frame->pts;
  data->frame_id_s = frame->frame_id_s;
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

void SendHandlerImpl::OnEndFrame() {
  std::shared_ptr<FrameInfo> data = this->CreateFrameInfo(true);
  if (!data) {
    LOGW(SOURCE) << "[SendHandlerImpl] OnEndFrame function, failed to create FrameInfo.";
    return;
  }
  this->SendFrameInfo(data);
  LOGI(SOURCE) << "[SendHandlerImpl] OnEndFrame function, send end frame.";
}

}  // namespace cnstream
