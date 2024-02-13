

#ifndef KUIPER_INFER_SOURCE_LAYER_MAX_POOLING_
#define KUIPER_INFER_SOURCE_LAYER_MAX_POOLING_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class MaxPoolingLayer : public NonParamLayer {
 public:
  explicit MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h,
                           uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& max_layer);

 private:
  uint32_t padding_h_ = 0;
  uint32_t padding_w_ = 0;
  uint32_t pooling_size_h_ = 0;
  uint32_t pooling_size_w_ = 0;
  uint32_t stride_h_ = 1;
  uint32_t stride_w_ = 1;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_MAX_POOLING_
