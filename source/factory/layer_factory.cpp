#include "factory/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
// 加入注册表
void LayerRegisterer::RegisterCreator(OpType op_type, const Creator &creator) {
  CHECK(creator != nullptr) << "Layer creator is empty";
  CreateRegistry &registry = Registry();
  CHECK_EQ(registry.count(op_type), 0) << "Layer type: " << int(op_type) << " has already registered!";
  registry.insert({op_type, creator});
}

// 生成实例
std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op) {
  CreateRegistry &registry = Registry();
  const OpType op_type = op->op_type_;

  LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the layer type: " << int(op_type);
  const auto &creator = registry.find(op_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  std::shared_ptr<Layer> layer = creator(op);
  LOG_IF(FATAL, !layer) << "Layer init failed!";
  return layer;
}

// 初始化单例注册表
LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {
  // static表示只初始化一次
  static CreateRegistry *kRegistry = new CreateRegistry();
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return *kRegistry;
}
}
