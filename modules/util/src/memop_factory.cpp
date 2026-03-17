
#include "memop_factory.hpp"
#include "memop.hpp"

namespace cnstream {

MemOpFactory::MemOpFactory() {}

MemOpFactory::~MemOpFactory() {}

MemOpFactory& MemOpFactory::Instance() {
  static MemOpFactory instance;
  return instance;
}

bool MemOpFactory::RegisterMemOpCreator(DevType dev_type,
                                        std::function<std::shared_ptr<MemOp>(int dev_id)> creator) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = creators_.insert({dev_type, creator});
  return inserted;
}

std::shared_ptr<MemOp> MemOpFactory::CreateMemOp(DevType dev_type = DevType::CPU, int dev_id = -1) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = creators_.find(dev_type);
  if (it != creators_.end()) {
    return it->second(dev_id);
  }
  return nullptr;
}

}  // namespace cnstream