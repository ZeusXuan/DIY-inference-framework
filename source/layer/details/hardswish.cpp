#include "hardswish.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace kuiper_infer {
HardSwishLayer::HardSwishLayer() : NonParamLayer("HardSwish") {}

StatusCode HardSwishLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationForward(ActivationType::kActivationHardSwish, inputs, outputs);
}

StatusCode HardSwishLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                          std::shared_ptr<Layer<float>>& hardswish_layer) {
  if (!op) {
    LOG(ERROR) << "The hardswish operator parameter in the layer is null pointer.";
    return StatusCode::kParseOperatorNullParam;
  }
  hardswish_layer = std::make_shared<HardSwishLayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kHardSwishCreateInstance(HardSwishLayer::CreateInstance, "nn.Hardswish");

}  // namespace kuiper_infer