

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class HardSigmoid : public NonParamLayer {
 public:
  explicit HardSigmoid();

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& hardsigmoid_layer);
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSIGMOID_HPP_
