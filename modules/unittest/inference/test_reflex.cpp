
// Test reflection

#include <iostream>

#include "base.hpp"

#include "cnstream_logging.hpp"
#include "preproc.hpp"
#include "postproc.hpp"

#include "cnstream_frame.hpp"
#include "model_loader.hpp"

#include "reflex_object.h"

namespace cnstream {

class PreprocYolo: public Preproc {

 public:
  int Execute(const std::vector<float*>& net_inputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) {
    std::cout << "PreprocYolo::Execute" << std::endl;
    return 0;
  }

 private:
  DECLARE_REFLEX_OBJECT_EX(PreprocYolo, cnstream::Preproc);
};  // class PreprocYolo

IMPLEMENT_REFLEX_OBJECT_EX(PreprocYolo, cnstream::Preproc);


class PostprocYolo: public Postproc {

 public:
  int Execute(const std::vector<float*>& net_outputs, ModelLoader* model,
              const std::shared_ptr<cnstream::FrameInfo>& package) {
    std::cout << "PostprocYolo::Execute" << std::endl;
    return 0;
  }

 private:
  DECLARE_REFLEX_OBJECT_EX(PostprocYolo, cnstream::Postproc);
};  // class PostprocYolo

IMPLEMENT_REFLEX_OBJECT_EX(PostprocYolo, cnstream::Postproc);


TEST(REFLEX, Yolo) {
  std::map<std::string, ClassInfo<ReflexObject>>& obj_map = CheckGlobalObjMap();
  EXPECT_EQ(obj_map.size(), 2);

  for (auto it = obj_map.begin(); it != obj_map.end(); it++) {
    std::string name = it->first;
    std::cout << "REFLEX: obj_map name = " << name << std::endl;
  }

}

/**
 * @brief Test create object
 * 
 */
TEST(REFLEX, CreateObject) {

  // 验证两种层次的创建过程（基类，向下类型转换的）
  ASSERT_NE(ReflexObject::CreateObject("PreprocYolo"), nullptr);
  ASSERT_NE(ReflexObject::CreateObject("PostprocYolo"), nullptr);

  ASSERT_NE(ReflexObjectEx<Preproc>::CreateObject("PreprocYolo"), nullptr);
  ASSERT_NE(ReflexObjectEx<Postproc>::CreateObject("PostprocYolo"), nullptr);
  
  // 手动创建 class_info（T: ReflexObject） 验证注册
  ClassInfo<ReflexObject> info2(std::string(), ObjectConstructor<ReflexObject>([]() {
    return reinterpret_cast<ReflexObject*>(new PostprocYolo());
  }));

  ObjectConstructor<ReflexObject> base_constructor;
  base_constructor = [info2]() { return reinterpret_cast<ReflexObject*>(info2.constructor()()); };
  ClassInfo<ReflexObject> base_info(info2.name(), base_constructor);
  ASSERT_EQ(ReflexObject::Register(base_info), true);
}

}