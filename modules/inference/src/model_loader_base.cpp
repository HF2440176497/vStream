
#include "model_loader.hpp"
#include "cnstream_logging.hpp"

#include <fstream>
#include <iostream>
#include <cstring>


namespace cnstream {

ModelLoaderFactory::ModelLoaderFactory() { }

ModelLoaderFactory::~ModelLoaderFactory() { }

ModelLoaderFactory& ModelLoaderFactory::Instance() {
  static ModelLoaderFactory instance;
  return instance;
}

ModelLoaderFactory::RegisterModelLoaderCreator(DevType dev_type, 
    std::function<std::unique_ptr<ModelLoader>(int dev_id)> creator) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = creators_.insert({dev_type, creator});
  return inserted;
}

std::unique_ptr<ModelLoader> ModelLoaderFactory::CreateModelLoader(DevType dev_type, int dev_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = creators_.find(dev_type);
  if (it == creators_.end()) {
    LOGI(MODEL_LOADER) << "Not found creator for dev_type: " << dev_type;
    return nullptr;
  }
  return it->second(dev_id);
}


}  // namespace cnstream
