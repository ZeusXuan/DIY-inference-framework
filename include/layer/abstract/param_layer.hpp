#ifndef KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#define KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#include "layer.hpp"

namespace kuiper_infer {
class ParamLayer : public Layer<float> {
 public:
  explicit ParamLayer(const std::string& layer_name);

  /**
   * @brief Initializes weight tensors
   *
   * Initializes the shapes of the weight tensors based on provided dimensions.
   * Does not initialize values.
   *
   * @param param_count Number of filters/kernels
   * @param param_channel Channels of filters
   * @param param_height Height of kernels
   * @param param_width Width of kernels
   */
  void InitWeightParam(uint32_t param_count, uint32_t param_channel, uint32_t param_height,
                       uint32_t param_width);

  /**
   * @brief Initializes bias tensors
   *
   * Initializes the shapes of the bias tensors based on dimensions.
   * Values are not initialized.
   *
   * @param param_count Number of bias terms
   * @param param_channel Channels of biases
   * @param param_height Height
   * @param param_width Width
   */
  void InitBiasParam(uint32_t param_count, uint32_t param_channel, uint32_t param_height,
                     uint32_t param_width);

  /**
   * @brief Gets the layer weight tensors
   *
   * Overrides the base class weights() method.
   *
   * Returns a reference to the weight tensors for this layer.
   *
   * @return Constant reference to the weight tensors
   */
  const std::vector<std::shared_ptr<Tensor<float>>>& weights() const override;

  /**
   * @brief Gets the layer bias tensors
   *
   * Overrides the base class bias() method.
   *
   * Returns a constant reference to the bias tensors for this layer.
   *
   * @return Constant reference to the bias tensors
   */
  const std::vector<std::shared_ptr<Tensor<float>>>& bias() const override;

  /**
   * @brief Sets the weight values
   *
   * Overrides the base class method.
   *
   * Sets the weight tensor values from a vector of floats.
   * The vector length should match the total number of values.
   *
   * @param weights Vector of weight values
   */
  void set_weights(const std::vector<float>& weights) override;

  /**
   * @brief Sets the bias values
   *
   * Overrides the base class method.
   *
   * Sets the bias tensor values from a vector of floats.
   * The vector length should match the total number of values.
   *
   * @param bias Vector of bias values
   */
  void set_bias(const std::vector<float>& bias) override;

  /**
   * @brief Sets the weight tensors
   *
   * Overrides the base class method.
   *
   * @param weights Weight tensors to set
   */
  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;

  /**
   * @brief Sets the bias tensors
   *
   * Overrides the base class method.
   *
   * @param bias Bias tensors to set
   */
  void set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) override;

 protected:
  std::vector<std::shared_ptr<Tensor<float>>> weights_;
  std::vector<std::shared_ptr<Tensor<float>>> bias_;
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
