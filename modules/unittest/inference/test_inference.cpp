

// 测试：（1）ModelLoder 创建,
//
//
//

#include "base.hpp"
#include "data_source_param.hpp"
#include "model_loader_base.hpp"

static const std::string engine_path = "test.engine";

#ifdef NVIDIA
#include "cuda/model_loader.hpp"

#else

#endif

namespace cnstream {

TEST(ModelLoader, Create) {
  auto& factory = ModelLoaderFactory::Instance();
  factory.PrintRegisteredCreators();
  auto model_loader_unique = factory.CreateModelLoader(DevType::CUDA, 0);
  ASSERT_NE(model_loader_unique, nullptr);
  auto model_loader_raw = model_loader_unique.get();

#ifdef NVIDIA

  ModelLoader* model_loader = dynamic_cast<TrtModelLoader*>(model_loader_raw);
  ASSERT_NE(model_loader, nullptr);

  ASSERT_EQ(model_loader->GetDeviceId(), 0);
  ASSERT_EQ(model_loader->GetDeviceType(), DevType::CUDA);

#else

#endif
  ASSERT_TRUE(model_loader_raw->Init(engine_path));
  for (int i = 0; i < model_loader_raw->input_names_.size(); ++i) {
    std::cout << "Input name [" << i << "] = " << model_loader_raw->input_names_[i] << std::endl;
  }
  for (int i = 0; i < model_loader_raw->output_names_.size(); ++i) {
    std::cout << "Output name [" << i << "] = " << model_loader_raw->output_names_[i] << std::endl;
  }

}  // ModelLoader Create




}  // namespace cnstream