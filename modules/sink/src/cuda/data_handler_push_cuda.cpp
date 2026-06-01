
#include "data_handler_push.hpp"
#include "cuda/data_handler_push_cuda.hpp"


#include "cnstream_logging.hpp"
#include "cnstream_frame_va.hpp"
#include "data_common.hpp"
#include "memop.hpp"
#include "memop_factory.hpp"

#include "cuda/cuda_check.hpp"
#include "cuda/transfmt_cuda.cuh"

#include <memory>
#include <opencv2/opencv.hpp>

namespace cnstream {

bool PushHandlerImplCUDA::InitDeviceCtx() {
  int ret = av_hwdevice_ctx_create(&ctx_.hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                                   std::to_string(device_id_).c_str(), nullptr, 0);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwdevice_ctx_create (CUDA) failed: " << ret;
    return false;
  }

  ctx_.hw_frames_ctx = av_hwframe_ctx_alloc(ctx_.hw_device_ctx);
  if (!ctx_.hw_frames_ctx) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwframe_ctx_alloc failed";
    return false;
  }

  AVHWFramesContext* hw_frames = reinterpret_cast<AVHWFramesContext*>(ctx_.hw_frames_ctx->data);
  hw_frames->format            = AV_PIX_FMT_CUDA;
  hw_frames->sw_format         = AV_PIX_FMT_NV12;
  hw_frames->width             = width_;
  hw_frames->height            = height_;
  hw_frames->initial_pool_size = 20;

  ret = av_hwframe_ctx_init(ctx_.hw_frames_ctx);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwframe_ctx_init failed: " << ret;
    return false;
  }

  ctx_.codec_ctx->hw_device_ctx = av_buffer_ref(ctx_.hw_device_ctx);
  ctx_.codec_ctx->hw_frames_ctx = av_buffer_ref(ctx_.hw_frames_ctx);

  ctx_.hw_frame = av_frame_alloc();
  ret = av_hwframe_get_buffer(ctx_.hw_frames_ctx, ctx_.hw_frame, 0);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_hwframe_get_buffer failed: " << ret;
    return false;
  }

  if (!sink_stream_) {
    CHECK_CUDA_RUNTIME(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&sink_stream_)));
  }

  return true;
}

void PushHandlerImplCUDA::CleanDeviceCtx() {
  if (ctx_.hw_frame)      { av_frame_free(&ctx_.hw_frame); }
  if (ctx_.hw_frames_ctx) { av_buffer_unref(&ctx_.hw_frames_ctx); }
  if (ctx_.hw_device_ctx) { av_buffer_unref(&ctx_.hw_device_ctx); }
  if (sink_stream_)       { CHECK_CUDA_RUNTIME(cudaStreamDestroy(sink_stream_)); sink_stream_ = nullptr; }
}

bool PushHandlerImplCUDA::SendDataFrame(const DataFramePtr& frame, AVPixelFormat src_pix_fmt) {
  auto dev_type = frame->GetCtx().device_type;
  if (dev_type == DevType::CUDA) {
    int actual_device = frame->GetCtx().device_id;
    if (!hw_ctx_initialized_.load()) {
      if (actual_device >= 0 && actual_device != device_id_) {
        LOGI(SINK) << "Reinitializing stream for device " << actual_device;
        if (!ReinitStream(actual_device)) {
            return false;
        }
        device_id_ = actual_device;
      }
      hw_ctx_initialized_.store(true);
    }
    return SendFrameCuda(frame, src_pix_fmt);
  } else if (dev_type == DevType::CPU) {
    return SendFrameToCuda(frame, src_pix_fmt);
  } else {
    return SendFrame(frame, src_pix_fmt);
  }
}

bool PushHandlerImplCUDA::SendFrameCuda(const DataFramePtr& frame, AVPixelFormat src_pix_fmt) {
  const int src_width  = frame->GetWidth();
  const int src_height = frame->GetHeight();
  const int src_stride = frame->GetStride(0);

#ifdef VSTREAM_UNIT_TEST
  if (src_stride != GetStride_8U_C3(src_width)) {
    LOGE(SINK) << "[" << stream_id_ << "]: src_stride != GetStride_8U_C3(src_width)";
    return false;
  }
#endif

  DataFormat src_fmt;
  if (src_pix_fmt == AV_PIX_FMT_RGB24) {
    src_fmt = DataFormat::PIXEL_FORMAT_RGB24;
  } else if (src_pix_fmt == AV_PIX_FMT_BGR24) {
    src_fmt = DataFormat::PIXEL_FORMAT_BGR24;
  } else {
    LOGW(SINK) << "[" << stream_id_ << "]: unsupported GPU src format, fallback to CPU";
    return SendFrameCpuFallback(frame, src_pix_fmt);
  }

  const void* cuda_data = frame->data_[0]->GetDevData();

  int ret = av_frame_make_writable(ctx_.hw_frame);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_make_writable (hw_frame) failed";
    return false;
  }

  if (src_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
    ret = ConvertRGB24ToNV12_Resize(
        ctx_.hw_frame->data[0], ctx_.hw_frame->linesize[0],
        ctx_.hw_frame->data[1], ctx_.hw_frame->linesize[1],
        width_, height_,
        cuda_data, src_stride, src_width, src_height,
        sink_stream_);
  } else if (src_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
    ret = ConvertBGR24ToNV12_Resize(
        ctx_.hw_frame->data[0], ctx_.hw_frame->linesize[0],
        ctx_.hw_frame->data[1], ctx_.hw_frame->linesize[1],
        width_, height_,
        cuda_data, src_stride, src_width, src_height,
        sink_stream_);
  } else {
    return false;
  }

  if (ret != 0) {
    LOGW(SINK) << "[" << stream_id_ << "]: GPU RGB to NV12 conversion failed, fallback to CPU";
    return SendFrameCpuFallback(frame, src_pix_fmt);
  }

  CHECK_CUDA_RUNTIME(cudaStreamSynchronize(sink_stream_));

  ctx_.hw_frame->pts = ComputePts();
  return EncodeFrame(ctx_.hw_frame);
}

bool PushHandlerImplCUDA::SendFrameToCuda(const DataFramePtr& frame, AVPixelFormat src_pix_fmt) {
  if (!hw_ctx_initialized_.load()) {
    hw_ctx_initialized_.store(true);
  }
  const int src_width  = frame->GetWidth();
  const int src_height = frame->GetHeight();
  const int src_stride = frame->GetStride(0);

#ifdef VSTREAM_UNIT_TEST
  if (src_stride != GetStride_8U_C3(src_width)) {
    LOGE(SINK) << "[" << stream_id_ << "]: src_stride != GetStride_8U_C3(src_width)";
    return false;
  }
#endif

  DataFormat src_fmt;
  if (src_pix_fmt == AV_PIX_FMT_RGB24) {
    src_fmt = DataFormat::PIXEL_FORMAT_RGB24;
  } else if (src_pix_fmt == AV_PIX_FMT_BGR24) {
    src_fmt = DataFormat::PIXEL_FORMAT_BGR24;
  } else {
    LOGW(SINK) << "[" << stream_id_ << "]: unsupported CPU src format for CUDA path, fallback";
    return SendFrameCpuFallback(frame, src_pix_fmt);
  }

  const uint8_t* cpu_data = static_cast<const uint8_t*>(frame->data_[0]->GetCpuData());
  size_t src_size = frame->GetPlaneBytes(0);

  auto memop = MemOpFactory::Instance().CreateMemOp(DevType::CUDA, device_id_);
  auto cuda_buf = memop->Allocate(src_size);
  if (!cuda_buf) {
    LOGE(SINK) << "[" << stream_id_ << "]: failed to allocate GPU buffer for H2D copy";
    return false;
  }
  memop->CopyFromHostAsync(cuda_buf.get(), cpu_data, src_size, sink_stream_);

  int ret = av_frame_make_writable(ctx_.hw_frame);
  if (ret < 0) {
    LOGE(SINK) << "[" << stream_id_ << "]: av_frame_make_writable (hw_frame) failed";
    return false;
  }

  if (src_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
    ret = ConvertRGB24ToNV12_Resize(
      ctx_.hw_frame->data[0], ctx_.hw_frame->linesize[0],
      ctx_.hw_frame->data[1], ctx_.hw_frame->linesize[1],
      width_, height_,
      cuda_buf.get(), src_stride, src_width, src_height,
      sink_stream_);
  } else if (src_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
    ret = ConvertBGR24ToNV12_Resize(
      ctx_.hw_frame->data[0], ctx_.hw_frame->linesize[0],
      ctx_.hw_frame->data[1], ctx_.hw_frame->linesize[1],
      width_, height_,
      cuda_buf.get(), src_stride, src_width, src_height,
      sink_stream_);
  } else {
    return false;
  }

  if (ret != 0) {
    LOGW(SINK) << "[" << stream_id_ << "]: CPU RGB to NV12 conversion failed, fallback to CPU";
    return SendFrameCpuFallback(frame, src_pix_fmt);
  }

  CHECK_CUDA_RUNTIME(cudaStreamSynchronize(sink_stream_));

  ctx_.hw_frame->pts = ComputePts();
  return EncodeFrame(ctx_.hw_frame);
}

}  // namespace cnstream