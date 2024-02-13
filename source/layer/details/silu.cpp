

#include "silu.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "simd.hpp"
#include "tick.hpp"

namespace kuiper_infer {

SiLULayer::SiLULayer() : NonParamLayer("SiLU") {}

StatusCode SiLULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  using namespace activation;
  return ActivationForward(ActivationType::kActivationSilu, inputs, outputs);
}

StatusCode SiLULayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& silu_layer) {
  if (!op) {
    LOG(ERROR) << "The SiLU operator parameter in the layer is null pointer.";
    return StatusCode::kParseOperatorNullParam;
  }
  silu_layer = std::make_shared<SiLULayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kSiluCreateInstance(SiLULayer::CreateInstance, "nn.SiLU");

}  // namespace kuiper_infer
