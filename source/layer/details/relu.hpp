

#ifndef KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#define KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class ReluLayer : public NonParamLayer {
 public:
  ReluLayer() : NonParamLayer("Relu") {}
  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& relu_layer);
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_BINOCULAR_RELU_HPP_
