
#include "hardsigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace kuiper_infer {
HardSigmoid::HardSigmoid() : NonParamLayer("HardSigmoid") {}

StatusCode HardSigmoid::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationForward(ActivationType::kActivationHardSigmoid, inputs, outputs);
}

StatusCode HardSigmoid::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                       std::shared_ptr<Layer<float>>& hardsigmoid_layer) {
  if (!op) {
    LOG(ERROR) << "The hardsigmoid operator parameter in the layer is null pointer.";
    return StatusCode::kParseOperatorNullParam;
  }
  hardsigmoid_layer = std::make_shared<HardSigmoid>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kHardSigmoidCreateInstance(HardSigmoid::CreateInstance, "nn.Hardsigmoid");

}  // namespace kuiper_infer
