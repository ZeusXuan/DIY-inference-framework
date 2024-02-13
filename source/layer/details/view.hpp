#ifndef KUIPER_INFER_SOURCE_LAYER_FLATTEN_HPP_
#define KUIPER_INFER_SOURCE_LAYER_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class ViewLayer : public NonParamLayer {
 public:
  explicit ViewLayer(std::vector<int32_t> shapes);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& view_layer);

 private:
  std::vector<int32_t> shapes_;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_FLATTEN_HPP_
