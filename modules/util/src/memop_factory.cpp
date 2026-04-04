
#include "memop_factory.hpp"
#include "memop.hpp"

namespace cnstream {

MemOpFactory::MemOpFactory() {}

MemOpFactory::~MemOpFactory() {}

MemOpFactory& MemOpFactory::Instance() {
  static MemOpFactory instance;
  return instance;
}

bool MemOpFactory::RegisterMemOpCreator(DevType device_type,
                                        std::function<std::shared_ptr<MemOp>(int device_id)> creator) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = creators_.insert({device_type, creator});
  return inserted;
}

std::shared_ptr<MemOp> MemOpFactory::CreateMemOp(DevType device_type = DevType::CPU, int device_id = -1) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = creators_.find(device_type);
  if (it != creators_.end()) {
    return it->second(device_id);
  }
  return nullptr;
}

}  // namespace cnstream