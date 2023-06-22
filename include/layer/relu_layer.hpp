#ifndef KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
#include "layer.hpp"
#include "ops/relu_op.hpp"

namespace kuiper_infer {
class ReluLayer : public Layer {
 public:
  ~ReluLayer() = default;

  // 构造函数把relu_op中的thresh告知给relu layer
  explicit ReluLayer(const std::shared_ptr<Operator> &op);

  // 执行relu操作的具体函数
  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

 private:
  std::unique_ptr<ReluOperator> op_;
};
}
#endif //KUIPER_COURSE_INCLUDE_LAYER_RELU_LAYER_HPP_
