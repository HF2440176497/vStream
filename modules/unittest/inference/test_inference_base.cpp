

#include "base.hpp"
#include "memop.hpp"
#include "memop_factory.hpp"
#include "data_source_param.hpp"
#include "cnstream_frame_va.hpp"

#include "common.hpp"
#include "tensor.hpp"
#include "infer_params.hpp"
#include "infer_resource.hpp"
#include "model_loader.hpp"
#include "inference.hpp"

#include "affine_trans.hpp"
#include "cnstream_queue.hpp"
#include "cnstream_logging.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>


static const std::string trt_yolov8_engine_path = "yolov8s_tracing_static_b4_quant.engine";

static const int device_id = 0;

#ifdef VSTREAM_USE_CUDA

#include "cuda/cuda_check.hpp"
#include "cuda/memop_cuda.hpp"
#include "cuda/model_loader_trt.hpp"

#else

#endif

namespace cnstream {


class ModelLoaderTest : public testing::Test {
 protected:
  virtual void SetUp() {
    input_image_ = cv::imread(image_file_, cv::IMREAD_COLOR);
    if (input_image_.empty()) {
      LOGF(ModelLoaderTest) << "Failed to load image file: " << image_file_;
      return;
    }

    auto& factory = ModelLoaderFactory::Instance();
    factory.PrintRegisteredCreators();

#ifdef VSTREAM_USE_CUDA
    memop_ = MemOpFactory::Instance().CreateMemOp(DevType::CUDA, device_id);
    ASSERT_NE(memop_, nullptr);

    model_loader_owner_ = factory.CreateModelLoader(DevType::CUDA, device_id);
    ASSERT_NE(model_loader_owner_, nullptr);
    model_loader_ = model_loader_owner_.get();

#else



#endif

  }

  virtual void TearDown() {
    LOGI(ModelLoaderTest) << "ModelLoaderTest TearDown";
    model_loader_ = nullptr;
    model_loader_owner_.reset();
  }

 protected:
  int          batch_size_ = 4;
  std::unique_ptr<ModelLoader> model_loader_owner_ = nullptr;
  ModelLoader* model_loader_ = nullptr;
  cv::Mat      input_image_;
  std::string  image_file_ = "test_image.png";
  std::shared_ptr<MemOp> memop_ = nullptr;

  std::shared_ptr<void> input_mem_ = nullptr;  /** net_one_input */
  std::shared_ptr<void> output_mem_ = nullptr; /** net_one_output */
};



TEST_F(ModelLoaderTest, Create) {

#ifdef VSTREAM_USE_CUDA

  ModelLoaderTrt* trt_model_loader = dynamic_cast<ModelLoaderTrt*>(model_loader_);
  ASSERT_NE(trt_model_loader, nullptr);

  ASSERT_EQ(trt_model_loader->GetDeviceId(), device_id);
  ASSERT_EQ(trt_model_loader->GetDeviceType(), DevType::CUDA);

  std::string model_path = GetExePath() + trt_yolov8_engine_path;
  InferParams params;

  params.device_type = DevType::CUDA;
  params.device_id = 0;
  params.model_path = model_path;

  ASSERT_TRUE(model_loader_->Init(model_path, params));

#else

#endif
  
  if(model_loader_ == nullptr) {
    std::cout << "model_loader_ is nullptr" << std::endl;
    return;
  }

  for (int i = 0; i < model_loader_->InputNum(); ++i) {
    std::string input_name = model_loader_->InputName(i);
    TensorShape input_shape = model_loader_->InputShape(i);
    std::cout << "Input name [" << i << "] = " << input_name << "; shape = " << input_shape << std::endl;
  }
  for (int i = 0; i < model_loader_->OutputNum(); ++i) {
    std::string output_name = model_loader_->OutputName(i);
    TensorShape output_shape = model_loader_->OutputShape(i);
    std::cout << "Output name [" << i << "] = " << output_name << "; shape = " << output_shape << std::endl;
  }


}  // ModelLoader Create


/**
 * 验证 YOLOv8 模型输入输出加载
 */
TEST_F(ModelLoaderTest, Run) {
  if(model_loader_ == nullptr) {
    return;
  }

  uint32_t batch_size = model_loader_->get_batch_size();
  uint32_t channel_size = model_loader_->get_channel();
  uint32_t height = model_loader_->get_height();
  uint32_t width = model_loader_->get_width();

  auto input_index = model_loader_->get_input_ordered_index();
  auto output_index = model_loader_->get_output_ordered_index();
  
  // 1）构造输入 IOResValue
  IOResValue i_value;
  int input_num = model_loader_->InputNum();
  i_value.datas.resize(input_num);
  i_value.ptrs.resize(input_num);

  for (int i = 0; i < input_num; ++i) {
    auto input_data_type = model_loader_->InputDataType(i);
    ASSERT_EQ(input_data_type, DataType::FLOAT32);

    size_t input_size = model_loader_->GetInputDataBatchAlignSize(i);
    std::shared_ptr<void> input_mem = memop_->Allocate(input_size);

    auto input_shape = model_loader_->InputShape(i);
    ASSERT_NE(input_mem, nullptr);
    i_value.ptrs[i] = input_mem;
    i_value.datas[i].ptr = input_mem.get();
    i_value.datas[i].shape = input_shape;
    i_value.datas[i].batch_offset = input_size / input_shape.N();;
    i_value.datas[i].batchsize = input_shape.N();
  }

  if (input_mem_ == nullptr) {
    input_mem_ = i_value.ptrs[input_index];
  }

  // 2）构造输出 IOResValue
  IOResValue o_value;
  int output_num = model_loader_->OutputNum();
  o_value.datas.resize(output_num);
  o_value.ptrs.resize(output_num);

  for (int i = 0; i < output_num; ++i) {
    auto output_data_type = model_loader_->OutputDataType(i);
    ASSERT_EQ(output_data_type, DataType::FLOAT32);

    size_t output_size = model_loader_->GetOutputDataBatchAlignSize(i);
    std::shared_ptr<void> output_mem = memop_->Allocate(output_size);
    ASSERT_NE(output_mem, nullptr);

    auto output_shape = model_loader_->OutputShape(i);
    o_value.ptrs[i] = output_mem;
    o_value.datas[i].ptr = output_mem.get();
    o_value.datas[i].shape = output_shape;
    o_value.datas[i].batch_offset = output_size / output_shape.N();;
    o_value.datas[i].batchsize = output_shape.N();
  }

  if (output_mem_ == nullptr) {
    output_mem_ = o_value.ptrs[output_index];
  }

  // 3）pre process
  int src_w = input_image_.cols;
  int src_h = input_image_.rows;

  int dst_w = 640;
  int dst_h = 640;
  
  AffineTrans trans;
  std::tuple<int, int> from{src_w, src_h};
  std::tuple<int, int> to{dst_w, dst_h};
  trans.compute(from, to);

  auto norm = Norm::alpha_beta(1 / 255.0f, 0.0f);   // 缩放为 1/255 ，并非最大最小归一化

  /* 输出内存是紧密排列的 */
  float* one_input_data = (float*)i_value.datas[input_index].ptr;  // raw input data
  resize_cpu(input_image_.data, src_w, src_h, input_image_.step, one_input_data, dst_w, dst_h, 114.0f, trans.get_d2s());
  swap_channel_cpu(one_input_data, dst_w, dst_h, dst_w * dst_h, ChannelsArrange::BGR);
  normalize_cpu(one_input_data, dst_w, dst_h, dst_w * dst_h, norm, ChannelsArrange::RGB); 
  
  size_t one_input_size = model_loader_->GetInputDataBatchAlignSize(input_index);
  auto one_input_shape = model_loader_->InputShape(input_index);
  
  int pixes_num = model_loader_->InputShape(input_index).DataCount() / one_input_shape.N();
  ASSERT_EQ(pixes_num, dst_w * dst_h * 3);
  ASSERT_EQ(pixes_num * data_type_size(DataType::FLOAT32), one_input_size);

  // 拷贝到 batch 每个 idx 
  size_t one_batch_offset = one_input_size / one_input_shape.N();
  for (int i = 0; i < batch_size; ++i) {
    memop_->CopyFromHost(one_input_data + i * one_batch_offset, one_input_data, one_batch_offset);
  }
  model_loader_->RunSync(i_value.ptrs, o_value.ptrs);

  // one output: post process
  void* one_output_data = o_value.datas[output_index].ptr;
  TensorShape one_output_shape = model_loader_->OutputShape(output_index);

  LOGI(InferenceTest) << "one_output_shape = " << one_output_shape;

  int num_bboxes = one_output_shape.shape(2);  // 8000
  int output_cdim = one_output_shape.shape(1);  // 84

  int num_classes = output_cdim - 4;  // 84 - 4

  std::cout << "num_bboxes = " << num_bboxes << "; num_classes = " << num_classes << std::endl;
}


bool InferParamsEQ(const InferParams &p1, const InferParams &p2) {
  return p1.device_type == p2.device_type &&
    p1.device_id == p2.device_id &&
    p1.object_infer == p2.object_infer &&
    p1.threshold == p2.threshold &&
    p1.infer_interval == p2.infer_interval &&
    p1.batching_timeout == p2.batching_timeout &&
    p1.trans_data_size == p2.trans_data_size &&
    p1.model_path == p2.model_path &&
    p1.preproc_name == p2.preproc_name &&
    p1.postproc_name == p2.postproc_name &&
    p1.obj_filter_name == p2.obj_filter_name &&
    p1.dump_resized_image_dir == p2.dump_resized_image_dir &&
    p1.saving_infer_input == p2.saving_infer_input &&
    p1.custom_preproc_params == p2.custom_preproc_params &&
    p1.custom_postproc_params == p2.custom_postproc_params;
}



class InferObserver : public IModuleObserver {
 public:
  void notify(std::shared_ptr<FrameInfo> data) override {
    output_frame_queue_.Push(data);
  }

  std::shared_ptr<FrameInfo> GetOutputFrame() {
    std::shared_ptr<FrameInfo> output_frame = nullptr;
    output_frame_queue_.WaitAndTryPop(output_frame, std::chrono::milliseconds(100));
    return output_frame;
  }

 private:
  ThreadSafeQueue<std::shared_ptr<FrameInfo>> output_frame_queue_;
};

void GetResult(std::shared_ptr<InferObserver> observer) {
  uint32_t i = 0;
  while (true) {
    auto data = observer->GetOutputFrame();
    if (data != nullptr) {
      if (!data->IsEos()) {
        DataFramePtr frame = data->collection.Get<DataFramePtr>(kDataFrameTag);

        // 测试时，需要按顺序放入 frame_id = 0, ... framedata
        EXPECT_EQ(frame->frame_id_, i);
        i++;
        std::cout << "Got data, frame id = " << frame->frame_id_ << std::endl;
      } else {
        std::cout << "Got EOS, break" << std::endl;
        break;
      }
    }
  }
}


/**
 * 测试推理模块读取参数
 */
TEST(Inference, Param) {

  InferParamManager manager;
  ParamRegister param_register;
  manager.RegisterAll(&param_register);

  std::vector<std::string> infer_param_list = {
    "device_id",
    "object_infer",
    "threshold",
    "infer_interval",
    "batching_timeout",
    "trans_data_size",
    "model_path",
    "preproc_name",
    "postproc_name",
    "obj_filter_name",
    "dump_resized_image_dir",
    "saving_infer_input",
    "custom_preproc_params",
    "custom_postproc_params"
  };

  // 验证 infer_param_list 中的参数项，是否已经通过 RegisterAll 中注册到 param_register 内部
  for (const auto &it : infer_param_list) {
    EXPECT_TRUE(param_register.IsRegisted(it));
  }

  // check parse params right
  InferParams expect_ret;
  expect_ret.device_id = 1;
  expect_ret.object_infer = true;
  expect_ret.threshold = 0.5;
  expect_ret.infer_interval = 1;
  expect_ret.batching_timeout = 3;
  expect_ret.trans_data_size = 20;
  expect_ret.model_path = "fake_path";
  expect_ret.preproc_name = "fake_name";
  expect_ret.postproc_name = "fake_name";
  expect_ret.obj_filter_name = "filter_name";
  expect_ret.dump_resized_image_dir = "dir";
  expect_ret.saving_infer_input = true;
  expect_ret.custom_preproc_params = {
    std::make_pair(std::string("param"), std::string("value"))};
  expect_ret.custom_postproc_params = {
    std::make_pair(std::string("param"), std::string("value"))};

  ModuleParamSet raw_params;
  raw_params["device_id"] = std::to_string(expect_ret.device_id);
  raw_params["object_infer"] = std::to_string(expect_ret.object_infer);
  raw_params["threshold"] = std::to_string(expect_ret.threshold);
  raw_params["infer_interval"] = std::to_string(expect_ret.infer_interval);
  raw_params["batching_timeout"] = std::to_string(expect_ret.batching_timeout);
  raw_params["trans_data_size"] = std::to_string(expect_ret.trans_data_size);
  raw_params["model_path"] = expect_ret.model_path;
  raw_params["preproc_name"] = expect_ret.preproc_name;
  raw_params["postproc_name"] = expect_ret.postproc_name;
  raw_params["obj_filter_name"] = expect_ret.obj_filter_name;
  raw_params["dump_resized_image_dir"] = expect_ret.dump_resized_image_dir;
  raw_params["saving_infer_input"] = std::to_string(expect_ret.saving_infer_input);
  raw_params["custom_preproc_params"] = "{\"param\" : \"value\"}";
  raw_params["custom_postproc_params"] = "{\"param\" : \"value\"}";

  // ParseBy: 遍历当时 InferParamManager 注册的参数，存在的参数会调用 parser 进行校验
  {
    InferParams ret;
    EXPECT_TRUE(manager.ParseBy(raw_params, &ret));
    EXPECT_TRUE(InferParamsEQ(expect_ret, ret));
  }

  raw_params.clear();
  {
    InferParams ret;
    raw_params["device_id"] = "wrong";
    EXPECT_FALSE(manager.ParseBy(raw_params, &ret));
  }

  raw_params.clear();
  {
    InferParams ret;
    raw_params["object_infer"] = "wrong";
    EXPECT_FALSE(manager.ParseBy(raw_params, &ret));
  }

  raw_params.clear();
  {
    InferParams ret;
    raw_params["threshold"] = "wrong";
    EXPECT_FALSE(manager.ParseBy(raw_params, &ret));
  }

  raw_params.clear();
  {
    InferParams ret;
    raw_params["infer_interval"] = "wrong";
    EXPECT_FALSE(manager.ParseBy(raw_params, &ret));
  }

  raw_params.clear();
  {
    InferParams ret;
    raw_params["batching_timeout"] = "wrong";
    EXPECT_FALSE(manager.ParseBy(raw_params, &ret));
  }

}

TEST(Inference, custom_preproc_params_parse) {
  InferParamManager manager;
  ParamRegister param_register;
  manager.RegisterAll(&param_register);
  ModuleParamSet raw_params;
  raw_params["custom_preproc_params"] = "{wrong_json_format,}";
  InferParams ret;
  EXPECT_FALSE(manager.ParseBy(raw_params, &ret));
}

TEST(Inference, custom_postproc_params_parse) {
  InferParamManager manager;
  ParamRegister param_register;
  manager.RegisterAll(&param_register);
  ModuleParamSet raw_params;
  raw_params["custom_postproc_params"] = "{wrong_json_format,}";
  InferParams ret;
  EXPECT_FALSE(manager.ParseBy(raw_params, &ret));
}

static const char *g_preproc_name = "PreprocClassification";
static const char *g_postproc_name = "PostprocClassification";

static constexpr int g_device_id = 0;
static const std::string g_channel_id = "channel_1";

/**
 * 单独创建一个 Inference 模块，每次处理单个 FrameInfo
 */
TEST(Inference, Demo) {
  std::string model_path = GetExePath() + trt_yolov8_engine_path;

  std::shared_ptr<Module> infer = std::make_shared<Inference>("test_inference");
  std::shared_ptr<InferObserver> observer = std::make_shared<InferObserver>();
  infer->SetObserver(reinterpret_cast<IModuleObserver *>(observer.get()));
  std::thread th = std::thread(&GetResult, observer);
  ModuleParamSet param;
  param["model_path"] = model_path;
  param["preproc_name"] = g_preproc_name;
  param["postproc_name"] = g_postproc_name;
  param["device_id"] = std::to_string(g_device_id);
  param["batching_timeout"] = "3000";

  ASSERT_TRUE(infer->Open(param));

  /**
   * 模拟在 CPU 上创建 dataframe
   * 然后
   */
  auto memop = MemOpFactory::Instance().CreateMemOp(DevType::CPU, -1);

  const int width = 1280, height = 720;
  auto dec_frame = CreateTestDecodeFrame(DataFormat::PIXEL_FORMAT_RGB24, width, height);

  size_t nbytes = width * height * sizeof(uint8_t) * 3;
  size_t boundary = 1 << 16;
  nbytes = (nbytes + boundary - 1) & ~(boundary - 1);  // align to 64kb

  std::vector<std::shared_ptr<void>> frame_data_vec;
  for (uint32_t i = 0; i < 32; i++) {
    // fake data
    std::shared_ptr<void> frame_data = memop->Allocate(nbytes);
    frame_data_vec.push_back(frame_data);
    void *planes[3] = {nullptr, nullptr, nullptr};

    uint8_t *frame_data_uint8 = (uint8_t *)frame_data.get();
    planes[0] = (void *)frame_data_uint8;                   // R plane
    planes[1] = (void *)(frame_data_uint8 + width * height);  // G plane
    planes[2] = (void *)(frame_data_uint8 + width * height * 2);  // B plane   

    auto data = cnstream::FrameInfo::Create(g_channel_id);

    std::shared_ptr<DataFrame> frame(new (std::nothrow) DataFrame());
    data->collection.Add(kDataFrameTag, frame);  // add DataFrame member
    data->collection.Add(kInferObjsTag, std::make_shared<InferObjs>());
    data->timestamp = i;

    // set DataFrame
    frame->width_ = width;
    frame->height_ = height;
    void *ptr_cpu[3] = {planes[0], planes[1], planes[2]};
    frame->stride_[0] = frame->stride_[1] = frame->stride_[2] = width;
    frame->ctx_.device_id = -1;
    frame->ctx_.device_type = DevType::CPU;
    frame->fmt_ = DataFormat::PIXEL_FORMAT_RGB24;
    frame->CopyToSyncMem(dec_frame);

    int ret = infer->Process(data);
    EXPECT_EQ(ret, 0);
  }  // end for
  
  // eos frame
  auto data = cnstream::FrameInfo::Create(g_channel_id, true);
  int ret = infer->Process(data);
  EXPECT_EQ(ret, 0);

  ASSERT_NO_THROW(infer->Close());

  if (th.joinable()) {
    th.join();
  }
  CleanupTestDecodeFrame(dec_frame);

}

}  // namespace cnstream