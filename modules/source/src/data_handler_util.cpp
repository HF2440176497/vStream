
#include "cnstream_logging.hpp"

#include "data_source_param.hpp"  // DevContext, DataFormat
#include "data_handler_util.hpp"

namespace cnstream {

/**
 * OnDecodeFrame 同步调用
 */
int SourceRender::Process(std::shared_ptr<FrameInfo> frame_info, DecodeFrame *dec_frame, uint64_t frame_id, void* stream) {
  DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
  if (!frame || !dec_frame) {
    LOGF(SOURCE) << "SourceRender::Process: frame or dec_frame is NULL";
    return -1;
  }
  if (!dec_frame->valid) return -1;

  DataFrame::Meta meta;
  meta.frame_id  = frame_id;
  meta.width     = dec_frame->width;
  meta.height    = dec_frame->height;
  meta.fmt       = DataFormat::PIXEL_FORMAT_RGB24;  // dst fmt
  meta.ctx       = DevContext(dec_frame->device_type, dec_frame->device_id);
  // RGB24 只需要对齐 plane 0 的步长
  meta.stride[0] = GetStride_8U_C3(meta.width);
  frame->SetMeta(std::move(meta));

  if (dec_frame->buf_ref) {
    frame->deAllocator_ = std::make_unique<Deallocator>(dec_frame->buf_ref.release());
    dec_frame->buf_ref = nullptr;
  }
  frame->CopyToSyncMem(dec_frame, stream);
  return 0;
}

}