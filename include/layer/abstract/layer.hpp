#ifndef KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
#include <glog/logging.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "data/tensor.hpp"
#include "runtime/runtime_op.hpp"
#include "status_code.hpp"

namespace kuiper_infer {
template <typename T>
class Layer;

template <>
class Layer<int8_t> {};

/**
 * @brief Base layer class
 *
 * Abstract base class for neural network layers.
 * Defines the common layer interface and methods.
 *
 * @tparam T Data type (float, int8 etc)
 */
template <>
class Layer<float> {
 public:
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}

  /**
   * @brief Performs forward inference
   *
   * This method implements the forward inference for this layer.
   * It takes the input tensors, performs the required computations
   * (convolution, matrix multiply etc.), and produces the output tensors.
   *
   * The inputs and outputs are based on the connections in the runtime
   * graph. This method implements the core inference logic of the layer.
   *
   * @return Status code indicating success or failure
   */
  virtual StatusCode Forward();

  /**
   * @brief Performs forward inference
   *
   * This method implements the forward inference for this layer.
   * It takes the input tensors, performs the required computations
   * (convolution, matrix multiply etc.), and produces the output tensors.
   *
   * Takes the input tensors, executes the required computations
   * (conv, matrix multiply etc.), and produces the output tensors.
   *
   * @param inputs Input tensors
   * @param outputs Output tensors
   * @return Status code
   */
  virtual StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                             std::vector<std::shared_ptr<Tensor<float>>>& outputs);

  /**
   * @brief Gets layer weights
   *
   * @return Vector of weight tensors
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>>& weights() const;

  /**
   * @brief Gets layer biases
   *
   * @return Vector of bias tensors
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>>& bias() const;

  /**
   * @brief Sets the layer weights
   *
   * @param weights Vector of weight tensors
   */
  virtual void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights);

  /**
   * @brief Sets the layer biases
   *
   * @param bias Vector of bias tensors
   */
  virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias);

  /**
   * @brief Sets weights from float vector
   *
   * Sets the weight values from a vector of floats.
   *
   * The length of the vector should match the total number of weight values.
   *
   * @param weights Vector of weight values
   */
  virtual void set_weights(const std::vector<float>& weights);

  /**
   * @brief Sets biases from float vector
   *
   * Sets the bias values from a vector of floats.
   *
   * The length of the vector should match the total number of bias values.
   *
   * @param bias Vector of bias values
   */
  virtual void set_bias(const std::vector<float>& bias);

  /**
   * @brief Gets layer name
   *
   * @return Layer name
   */
  virtual const std::string& layer_name() const { return this->layer_name_; }

  /**
   * @brief Sets corresponding runtime operator
   *
   * Associates this layer instance with the given runtime operator.
   *
   * @param runtime_operator The RuntimeOperator for this layer
   */
  void set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator);

 protected:
  std::string layer_name_;
  std::weak_ptr<RuntimeOperator> runtime_operator_;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
