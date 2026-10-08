
#include "cnstream_source.hpp"  // DataSource
#include "data_handler_send.hpp"

#include <stdexcept>


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

int SendHandler::Send(const SendFrame& send_frame, int wait_ms) {
  if (!impl_) {
    LOGE(SOURCE) << "[" << stream_id_ << "] handler is not valid";
    return static_cast<int>(SendRet::ERR_PARAM);
  }
  if (send_frame.image.empty()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: image is empty";
    return static_cast<int>(SendRet::ERR_PARAM);
  }
  SendRet ret = impl_->Push(send_frame, wait_ms);
  if (ret != SendRet::OK) {
    LOGW(SOURCE) << "[" << stream_id_ << "]: send frame failed, ret=" << static_cast<int>(ret);
  }
  return static_cast<int>(ret);
}

int SendHandler::Send(uint64_t pts, std::string frame_id_s, const cv::Mat &image, int wait_ms) {
  if (!impl_) {
    LOGE(SOURCE) << "[" << stream_id_ << "] handler is not valid";
    return static_cast<int>(SendRet::ERR_PARAM);
  }
  if (image.empty()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: image is not valid";
    return static_cast<int>(SendRet::ERR_PARAM);
  }
  SendRet ret = impl_->Push(SendFrame{pts, frame_id_s, image}, wait_ms);
  if (ret != SendRet::OK) {
    LOGW(SOURCE) << "[" << stream_id_ << "]: send frame failed, ret=" << static_cast<int>(ret);
  }
  return static_cast<int>(ret);
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
    LOGE(SOURCE) << "[" << stream_id_ << "]: Send handler open failed, impl_ is null";
    return false;
  }
  if (stream_index_ == INVALID_STREAM_IDX) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Invalid stream_idx";
    return false;
  }
  return impl_->Open();
}

/**
 * note: For send handler, can be empty
 */
bool SendHandler::SetHandlerParams(const ModuleParamSet& params) {
  if (!impl_) {
    return false;
  }
  DataSource* ds = dynamic_cast<DataSource*>(module_);
  if (ds) {
    ModuleParamSet stream_params = ds->GetStreamParams(stream_id_);
    if (!stream_params.empty()) {
      impl_->param_set_ = stream_params;
      impl_->SetupQueue();
      return true;
    }
  }
  impl_->param_set_ = params;
  impl_->SetupQueue();
  return true;
}

void SendHandlerImpl::SetupQueue() {
  auto it = param_set_.find(key_send_queue_size);
  if (it != param_set_.end()) {
    try {
      int size = std::stoi(it->second);
      if (size > 0) {
        queue_size_ = static_cast<uint32_t>(size);
      } else {
        LOGW(SOURCE) << "[" << stream_id_ << "]: queue_size must be positive, use default "
                     << queue_size_;
      }
    } catch (const std::exception&) {
      LOGW(SOURCE) << "[" << stream_id_ << "]: invalid queue_size '" << it->second
                   << "', use default " << queue_size_;
    }
  }
  // SetHandlerParams 在 Open 之前调用，此时队列必为空
  image_queue_ = std::make_unique<ThreadSafeQueue<SendFrame>>(queue_size_);
}

SendRet SendHandlerImpl::Push(const SendFrame& send_frame, int wait_ms) {
  if (!image_queue_) {
    return SendRet::ERR_STOPPED;
  }
  if (image_queue_->WaitAndTryPush(send_frame, std::chrono::milliseconds(wait_ms))) {
    return SendRet::OK;
  }
  if (!running_.load() || !image_queue_) {
    return SendRet::ERR_STOPPED;
  }
  return SendRet::ERR_TIMEOUT;
}

bool SendHandlerImpl::Open() {
  running_.store(true);
  thread_ = std::thread(&SendHandlerImpl::Loop, this);
  return true;
}

void SendHandlerImpl::Stop() {
  // 先置 running_ 为 false 再停队列：保证 Push 失败时能正确判定为 ERR_STOPPED
  running_.store(false);
  if (image_queue_) {
    image_queue_->Stop();  // 唤醒所有阻塞在 WaitAndTryPush 上的生产者
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
    if (!image_queue_->TryPop(send_frame)) {  // Non block pop
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    
    DecodeFrame frame(send_frame.image.rows, send_frame.image.cols, DataFormat::PIXEL_FORMAT_BGR24);
    frame.device_type = DevType::CPU;
    frame.device_id = -1;
    frame.planeNum = 1;  // BGR格式使用1个平面

    const int stride = GetStride_8U_C3(frame.width);
    size_t data_size = frame.height * stride;

    uint8_t* buffer = new (std::nothrow) uint8_t[data_size];
    if (!buffer) {
      LOGE(SOURCE) << "SendHandlerImpl: Failed to allocate for image data, skip this frame";
      continue;
    }
    for (int i = 0; i < send_frame.image.rows; ++i) {
      memcpy(buffer + i * stride,
             send_frame.image.ptr(i),
             frame.width * 3);
    }
    LOGU(SOURCE) << "SendHandlerImpl: Loop; image width: " << send_frame.image.cols << ", height: " << send_frame.image.rows << ", alloca data_size: " << data_size;

    frame.stride[0] = stride;
    frame.plane[0] = buffer;
    frame.buf_ref = std::make_unique<MatBufRef>(buffer);

    frame.pts = send_frame.pts;
    frame.frame_id_s = send_frame.frame_id_s;
    std::shared_ptr<FrameInfo> data = OnDecodeFrame(&frame);
    if (!module_ || !handler_) {
      LOGE(SOURCE) << "SendHandlerImpl: [" << stream_id_ << "]: module_ or handler_ is null";
      break;
    }
    if (running_.load()) {
      handler_->SendData(data);
    }
  }
  OnEndFrame();
}


std::shared_ptr<FrameInfo> SendHandlerImpl::OnDecodeFrame(DecodeFrame* frame) {
  if (!frame) {
    LOGE(SOURCE) << "[SendHandlerImpl] OnDecodeFrame function frame is nullptr.";
    return nullptr;
  }
  std::shared_ptr<FrameInfo> data = CreateFrameInfo();
  if (!data) {
    LOGE(SOURCE) << "[SendHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return nullptr;
  }
  data->timestamp = frame->pts;
  data->frame_id_s = frame->frame_id_s;
  if (!frame->valid) {
    data->flags = static_cast<size_t>(DataFrameFlag::FRAME_FLAG_INVALID);
    SendFrameInfo(data);
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
    LOGW(SOURCE) << "[" << stream_id_ << "]: SendHandlerImpl OnEndFrame function, failed to create FrameInfo.";
    return;
  }
  SendFrameInfo(data);
  LOGI(SOURCE) << "[" << stream_id_ << "]: [SendHandlerImpl] OnEndFrame function, send end frame.";
}

}  // namespace cnstream
