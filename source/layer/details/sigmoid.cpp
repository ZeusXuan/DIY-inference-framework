
#include "sigmoid.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"

namespace kuiper_infer {

StatusCode SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationForward(ActivationType::kActivationSigmoid, inputs, outputs);
}

StatusCode SigmoidLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                        std::shared_ptr<Layer<float>>& sigmoid_layer) {
  if (!op) {
    LOG(ERROR) << "The sigmoid operator parameter in the layer is null pointer.";
    return StatusCode::kParseOperatorNullParam;
  }
  sigmoid_layer = std::make_shared<SigmoidLayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kSigmoidCreateInstance(SigmoidLayer::CreateInstance, "nn.Sigmoid");
}  // namespace kuiper_infer