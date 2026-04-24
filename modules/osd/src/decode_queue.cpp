
#include "decode_queue.hpp"
#include "cnstream_collection.hpp"
#include "cnstream_logging.hpp"

#include <nlohmann/json.hpp>

namespace cnstream {

/**
 * @brief DecodeQueue 不允许自己传输 frame_info 仍然需要借助 Pipeline 传输
 */
DecodeQueue::DecodeQueue(const std::string& name) : Module(name) {
  param_register_.SetModuleDesc("DecodeQueue is a module for decoding the results.");
  param_register_.Register(key_decode_queue_size, "Size of the decode queue.");
}

DecodeQueue::~DecodeQueue() {
}

int DecodeQueue::Process(std::shared_ptr<FrameInfo> data) {
  OnFrame(data);
  return 0;
}

/**
 * @brief 从 frame_info 中得到 output_data 放入队列
 */
void DecodeQueue::OnFrame(std::shared_ptr<FrameInfo> frame_info) {
    
    if (frame_info->IsInvalid() || frame_info->IsRemoved()){
        LOGW(DECODE_QUEUE) << "OnFrame::frame_info has problems";
        return;
    }
    LOGI(DECODE_QUEUE) << "OnFrame:" << frame_info->timestamp;
    s_output_data data;

    if (frame_info->collection.HasValue(cnstream::kDataFrameTag)) {
        auto img_data = frame_info->collection.Get<cnstream::DataFramePtr>(cnstream::kDataFrameTag);
        data.image_dict[output_constants::key_original_image] = img_data->GetImage();
    }

    data.frame_id_s = frame_info->frame_id_s;
    data.timestamp = frame_info->timestamp;
    InferObjsPtr objs_holder = nullptr;
    if (frame_info->collection.HasValue(cnstream::kInferObjsTag)) {
      objs_holder = frame_info->collection.Get<InferObjsPtr>(cnstream::kInferObjsTag);
    } else {
      LOGW(DECODE_QUEUE) << "OnFrame::frame_info has no infer objs";
      data.result = -1;
      return;
    }

    std::lock_guard<std::mutex> lk(objs_holder->mutex_);
    for(auto re : objs_holder->objs_) {
        s_obj_in data_obj;
        data_obj.str_id = re->id;
        data_obj.track_id = re->track_id;
        data_obj.score = re->score;
        data_obj.model_name = re->model_name;

        std::vector<int> bboxs;
        {
            bboxs.push_back(re->bbox.x);
            bboxs.push_back(re->bbox.y);
            bboxs.push_back(re->bbox.w);
            bboxs.push_back(re->bbox.h);
        }
        data_obj.bboxs = bboxs;
        data.objects.push_back(data_obj);
    }
    data.result = 0;  // note: 放入就表示成功
    Push(data);
    return;
}

bool DecodeQueue::Push(const s_output_data& data) {
    if (!queue_) {
        return false;
    }
    return queue_->Push(data);
}

/**
 * @param data
 * @param wait_ms 等待时间，-1 表示阻塞等待，0 表示非阻塞等待, >0 表示等待 wait_ms 毫秒
 * @return true 成功获取数据，false 失败获取数据
 */
bool DecodeQueue::GetData(s_output_data& data, int wait_ms) {
    if (!queue_) {
        return false;
    }
    if (wait_ms < 0) {
        queue_->WaitAndPop(data);
        if (data.result != 0) {  // 说明此时队列可能停止
            return false;
        }
        return true;
    } else if (wait_ms == 0) {
        return queue_->TryPop(data);
    } else {
      std::chrono::milliseconds timeout(wait_ms);
      return queue_->WaitAndTryPop(data, timeout);
    }
    return false;
}

s_output_data DecodeQueue::GetData(int wait_ms) {
    s_output_data data;
    data.result = -1;
    if (GetData(data, wait_ms)) {
        return data;
    }
    return s_output_data();
}
    
bool DecodeQueue::Open(ModuleParamSet paramSet) {
  if (!CheckParamSet(paramSet)) {
    return false;
  }
  int queue_size = 20;  // 默认值
  if (paramSet.find(key_decode_queue_size) != paramSet.end()) {
    queue_size = std::stoi(paramSet[key_decode_queue_size]);
  }
  queue_ = std::make_unique<ThreadSafeQueue<s_output_data>>(queue_size);
  return true;
}

/**
 * @brief 检查以下参数：
 * （1） queue_size
 */
bool DecodeQueue::CheckParamSet(const ModuleParamSet& paramSet) const {
  bool ret = true;
  ParametersChecker checker;
  for (auto &it : paramSet) {
    if (!param_register_.IsRegisted(it.first)) {
      LOGW(DECODE_QUEUE) << "unknown param: " << it.first;
    }
  }
  // 如果存在配置项，则进行对应检查
  std::string err_msg;
  if (!checker.IsNum({key_decode_queue_size}, paramSet, err_msg, true)) {
    LOGE(DECODE_QUEUE) << "queue_size check failed: " << err_msg;
    return false;
  }
  return ret;
}

void DecodeQueue::Close() {
    if (queue_) {
        queue_->Stop();
        queue_.reset();
    }
}

}  // namespace cnstream
