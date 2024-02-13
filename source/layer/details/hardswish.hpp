

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
#include "layer/abstract/non_param_layer.hpp"

#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_HARDSWISH_HPP_
namespace kuiper_infer {
class HardSwishLayer : public NonParamLayer {
 public:
  explicit HardSwishLayer();

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& hardswish_layer);
};
}  // namespace kuiper_infer