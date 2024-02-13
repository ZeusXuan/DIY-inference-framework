
#include "relu6.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace kuiper_infer {
StatusCode Relu6Layer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                               std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationForward(ActivationType::kActivationRelu6, inputs, outputs);
}
StatusCode Relu6Layer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                      std::shared_ptr<Layer<float>>& relu_layer) {
  if (!op) {
    LOG(ERROR) << "The relu6 operator parameter in the layer is null pointer.";
    return StatusCode::kParseOperatorNullParam;
  }

  relu_layer = std::make_shared<Relu6Layer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kRelu6CreateInstance(Relu6Layer::CreateInstance, "nn.ReLU6");
}  // namespace kuiper_infer
