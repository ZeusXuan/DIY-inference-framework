#ifndef KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#define KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {
class SoftmaxLayer : public NonParamLayer {
 public:
  explicit SoftmaxLayer(int32_t dim = -1);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& softmax_layer);

 private:
  int32_t softmax_dim_ = -1;
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_SOFTMAX_HPP_
