
// Test reflection

#include "base.hpp"

#include "cnstream_logging.hpp"
#include "preproc.hpp"
#include "postproc.hpp"

#include "cnstream_frame.hpp"
#include "model_loader.hpp"

#include "reflex_object.h"

namespace cnstream {

class PreprocTest: public Preproc {

 public:
  int Execute(const std::vector<float*>& net_inputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) {
    std::cout << "PreprocTest::Execute" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return 0;
  }

 private:
  DECLARE_REFLEX_OBJECT_EX(PreprocTest, cnstream::Preproc);
};  // class PreprocTest

IMPLEMENT_REFLEX_OBJECT_EX(PreprocTest, cnstream::Preproc);


class PostprocTest: public Postproc {

 public:
  int Execute(const std::vector<float*>& net_outputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) {
    std::cout << "PostprocTest::Execute" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return 0;
  }

 private:
  DECLARE_REFLEX_OBJECT_EX(PostprocTest, cnstream::Postproc);
};  // class PostprocTest

IMPLEMENT_REFLEX_OBJECT_EX(PostprocTest, cnstream::Postproc);


TEST(REFLEX, Test) {
  std::map<std::string, ClassInfo<ReflexObject>>& obj_map = CheckGlobalObjMap();

  for (auto it = obj_map.begin(); it != obj_map.end(); it++) {
    std::string name = it->first;
    LOGI(TEST_REFLEX)  << "TEST_REFLEX: obj_map name = " << name;
  }

}

/**
 * @brief Test create object
 * 
 */
TEST(REFLEX, CreateObject) {

  // 验证两种层次的创建过程（基类，向下类型转换的）
  ASSERT_NE(ReflexObject::CreateObject("PreprocTest"), nullptr);
  ASSERT_NE(ReflexObject::CreateObject("PostprocTest"), nullptr);

  ASSERT_NE(ReflexObjectEx<Preproc>::CreateObject("PreprocTest"), nullptr);
  ASSERT_NE(ReflexObjectEx<Postproc>::CreateObject("PostprocTest"), nullptr);
  
  // 手动创建 class_info（T: ReflexObject） 验证注册
  ClassInfo<ReflexObject> info2(std::string(), ObjectConstructor<ReflexObject>([]() {
    return reinterpret_cast<ReflexObject*>(new PostprocTest());
  }));

  ObjectConstructor<ReflexObject> base_constructor;
  base_constructor = [info2]() { return reinterpret_cast<ReflexObject*>(info2.constructor()()); };
  ClassInfo<ReflexObject> base_info(info2.name(), base_constructor);
  ASSERT_EQ(ReflexObject::Register(base_info), true);
}

}