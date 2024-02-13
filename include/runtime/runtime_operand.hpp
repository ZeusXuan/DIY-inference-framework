#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#include <memory>
#include <string>
#include <vector>
#include "data/tensor.hpp"
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace kuiper_infer {
/**
 * @brief Base for runtime graph operand
 *
 * Template base class representing an operand (input/output) in a
 * graph. Contains operand name, shape, data vector, and data type.
 *
 * @tparam T Operand data type (float, int, etc.)
 */
template <typename T>
struct RuntimeOperandBase {
  RuntimeOperandBase() = default;

  RuntimeOperandBase(std::string name, std::vector<int32_t> shapes,
                     std::vector<std::shared_ptr<Tensor<T>>> datas, RuntimeDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), datas(std::move(datas)), type(type) {}

  /// Name of the operand
  std::string name;

  /// Shape of the operand
  std::vector<int32_t> shapes;

  /// Vector containing operand data
  std::vector<std::shared_ptr<Tensor<T>>> datas;

  /// Data type of the operand
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;
};

using RuntimeOperand = RuntimeOperandBase<float>;

using RuntimeOperandQuantized = RuntimeOperandBase<int8_t>;

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
