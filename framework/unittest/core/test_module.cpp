

/**
 * Unit test for Module
 */

#include "base.hpp"
#include "cnstream_module.hpp"
#include "cnstream_config.hpp"


namespace cnstream {

class TestModuleOne : public Module, public ModuleCreator<TestModuleOne> {
 public:
  explicit TestModuleOne(const std::string& name = "ModuleOne") : Module(name) {}
  bool Open(ModuleParamSet params) override {return true;}
  void Close() override {}
  int Process(std::shared_ptr<FrameInfo> frame_info) override {return 0;}
};

class TestModuleTwo : public Module, public ModuleCreator<TestModuleTwo> {
 public:
  explicit TestModuleTwo(const std::string& name = "ModuleTwo") : Module(name) {}
  bool Open(ModuleParamSet params) override {return true;}
  void Close() override {}
  int Process(std::shared_ptr<FrameInfo> frame_info) override {return 0;}
};

REGISTER_MODULE(TestModuleOne);
REGISTER_MODULE(TestModuleTwo);

/**
 * 测试单例模式
 */
TEST(FactoryTest, VerifySingleton) {
  auto* factory1 = ModuleFactory::Instance();
  auto* factory2 = ModuleFactory::Instance();
  EXPECT_EQ(factory1, factory2);  // 应该是同一个实例
  EXPECT_NE(factory1, nullptr);   // 不应该为null
}

/**
 * 自动注册机制
 */
TEST(ModuleCreatorTest, AutoRegistration) {
  std::vector<std::string> registed_modules = ModuleFactory::Instance()->GetRegisted();
  for (int i = 0; i < registed_modules.size(); i++) {
    std::cout << "[" << i << "] = " << registed_modules[i] << " ";
  }
  std::cout << std::endl;
  LOGI(ModuleCreatorTest) << "registed_modules.size() = " << registed_modules.size();
  // EXPECT_TRUE(ModuleFactory::Instance()->IsRegist("cnstream::TestModuleOne"));
  // EXPECT_TRUE(ModuleFactory::Instance()->IsRegist("cnstream::TestModuleTwo"));
}

TEST(ModuleCreatorTest, CreateObject) {
  std::string cur_class_name = "cnstream::TestModuleOne";
  std::string module_create_name = "module_create_name";
  Module* module = ModuleFactory::Instance()->Create(cur_class_name, module_create_name);
  EXPECT_TRUE(module != nullptr);
  EXPECT_EQ(module->GetName(), module_create_name);  // 构造函数参数
  delete module;  // 释放创建的模块对象
}

}
