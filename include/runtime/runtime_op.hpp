#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "runtime/pnnx/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace kuiper_infer {
template <typename T>
class Layer;

/**
 * @brief Base for runtime graph operator
 *
 * Template base class representing an operator node in a runtime graph.
 * Contains node execution order, name, type, layer, inputs, outputs,
 * parameters, attributes etc.
 *
 * @tparam T Operator data type (float, int8, etc.)
 */
template <typename T>
struct RuntimeOperatorBase {
  /// Execution order index of this operator
  int32_t forward_index = -1;

  /// Whether this operator has run in current execution
  bool has_forward = false;

  /// Name of the operator
  std::string name;

  /// Type of the operator
  std::string type;

  /// Layer for this operator
  std::shared_ptr<Layer<T>> layer;

  /// Names of output operators
  std::vector<std::string> output_names;

  /// Output operand
  std::shared_ptr<RuntimeOperandBase<T>> output_operands;

  /// Input operands mapped by provider name
  std::map<std::string, std::shared_ptr<RuntimeOperandBase<T>>> input_operands;

  /// Input operands in sequence
  std::vector<std::shared_ptr<RuntimeOperandBase<T>>> input_operands_seq;

  /// Output operators mapped by output name
  std::map<std::string, std::shared_ptr<RuntimeOperatorBase<T>>> output_operators;

  /// Operator parameters
  std::map<std::string, std::shared_ptr<RuntimeParameter>> params;

  /// Operator attributes like weights
  std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
};

using RuntimeOperator = RuntimeOperatorBase<float>;

using RuntimeOperatorQuantized = RuntimeOperatorBase<int8_t>;

template <typename T>
class RuntimeOperatorUtils;

/**
 * @brief Float runtime operator utilities
 *
 * Static utilities for float runtime operators.
 * Initializes operator inputs and outputs.
 */
template <>
class RuntimeOperatorUtils<float> {
 public:
  /**
   * @brief Initializes float operator inputs
   *
   * If first run, initializes input tensors based on shapes.
   * On later runs, checks shape match.
   *
   * @param operators Vector of runtime operators
   */
  static void InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

  /**
   * @brief Initializes float operator outputs
   *
   * If first run, initializes output tensors based on shapes.
   * On later runs, checks shape match.
   *
   * @param pnnx_operators Vector of PNNX operators
   * @param operators Vector of runtime operators
   */
  static void InitOperatorOutput(const std::vector<pnnx::Operator*>& pnnx_operators,
                                 const std::vector<std::shared_ptr<RuntimeOperator>>& operators);
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
