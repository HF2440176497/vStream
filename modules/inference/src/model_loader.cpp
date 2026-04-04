
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

bool ModelLoaderFactory::RegisterModelLoaderCreator(DevType device_type, 
    std::function<std::unique_ptr<ModelLoader>(int device_id)> creator) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = creators_.insert({device_type, creator});
  return inserted;
}

std::unique_ptr<ModelLoader> ModelLoaderFactory::CreateModelLoader(DevType device_type, int device_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = creators_.find(device_type);
  if (it == creators_.end()) {
    LOGI(MODEL_LOADER) << "Not found creator for device_type: " << DevType2Str(device_type);
    return nullptr;
  }
  return it->second(device_id);
}


}  // namespace cnstream
