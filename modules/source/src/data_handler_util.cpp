
#include "cnstream_logging.hpp"

#include "data_source_param.hpp"  // DevContext, DataFormat
#include "data_handler_util.hpp"

namespace cnstream {

/**
 * OnDecodeFrame 同步调用
 */
int SourceRender::Process(std::shared_ptr<FrameInfo> frame_info, DecodeFrame *dec_frame, uint64_t frame_id) {
  DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
  if (!frame || !dec_frame) {
    LOGF(SOURCE) << "SourceRender::Process: frame or dec_frame is NULL";
    return -1;
  }
  if (!dec_frame->valid) return -1;
  frame->frame_id_ = frame_id;
  frame->width_ = dec_frame->width;
  frame->height_ = dec_frame->height;
  if (dec_frame->buf_ref) {
    frame->deAllocator_ = std::make_unique<Deallocator>(dec_frame->buf_ref.release());
    dec_frame->buf_ref = nullptr;
  }
  frame->ctx_ = DevContext(dec_frame->device_type, dec_frame->device_id);
  // TODO: 支持配置 RGB24 或 BGR24
  frame->fmt_ = DataFormat::PIXEL_FORMAT_RGB24;  // dst fmt
  for (int i = 0; i < frame->GetPlanes(); ++i) {
    if (i == 0) {
      frame->stride_[i] = frame->width_ * 3;
    }
  }
  frame->CopyToSyncMem(dec_frame);
  return 0;
}

}