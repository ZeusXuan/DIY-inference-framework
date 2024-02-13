#include "relu.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace kuiper_infer {
StatusCode ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationForward(ActivationType::kActivationRelu, inputs, outputs);
}

StatusCode ReluLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& relu_layer) {
  if (!op) {
    LOG(ERROR) << "The relu operator parameter in the layer is null pointer.";
    return StatusCode::kParseOperatorNullParam;
  }

  relu_layer = std::make_shared<ReluLayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kReluCreateInstance(ReluLayer::CreateInstance, "nn.ReLU");
}  // namespace kuiper_infer
