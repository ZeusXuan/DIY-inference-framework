#include "factory/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
void LayerRegisterer::RegisterCreator(OpType op_type, const Creator &creator) {
  CHECK(creator != nullptr) << "Layer creator is empty";
  CreateRegistry &registry = Registry(); //实现单例的关键
  // 根据operator type
  CHECK_EQ(registry.count(op_type), 0) << "Layer type: " << int(op_type) << " has already registered!";
  // ReluLayer::CreateInstance 没有被注册过,就塞入到注册表当中
  registry.insert({op_type, creator});
}

std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op) {
  CreateRegistry &registry = Registry();
  const OpType op_type = op->op_type_;

  LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the layer type: " << int(op_type);
  // 根据传入的op_type(relu type)得到CreateInstance creator
  const auto &creator = registry.find(op_type)->second;

  LOG_IF(FATAL, !creator) << "Layer creator is empty!";
  std::shared_ptr<Layer> layer = creator(op);
  LOG_IF(FATAL, !layer) << "Layer init failed!";
  return layer;
}

LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {
  static  CreateRegistry *kRegistry = new CreateRegistry();
  // 没有static 那就是调用一次初始化一次
  // 不构成单例
  CHECK(kRegistry != nullptr) << "Global layer register init failed!";
  return *kRegistry;
}
}
