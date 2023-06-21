#include "data/tensor.hpp"
#include <gtest/gtest.h>
#include <armadillo>
#include <glog/logging.h>
#include "data/tensor.hpp"

TEST(test_tensor, create) {
  using namespace kuiper_infer;
  Tensor<float> tensor(3, 32, 32);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 32);
  ASSERT_EQ(tensor.cols(), 32);
  ASSERT_EQ(tensor.empty(), false);
}