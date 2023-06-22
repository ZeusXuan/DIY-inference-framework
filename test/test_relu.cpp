#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"
// op list
// conv 1
// conv 2
// relu
// sigmoid
// linear
// conv 3

//  有注册机制后的理论调用
/*
 * ops:[] = {conv 1,conv 2,relu,sigmod,linear,conv 3}
 * layers = []
 * for op in ops:
 *    layers.append(LayerRegisterer::CreateLayer(op))
 */

// 如果没有注册机制呢?
/*  ops:[] = {conv 1,conv 2,relu , sigmod,linear,conv 3}
 *  ConvLayer conv1 = std::make_shared(conv1_op);
 *  ConvLayer conv2 = std::make_shared(conv1_op);
 *  ReluLayer relu1 = std::make_shared(relu_op);
 *  SigmoidLayer sig = std::make_shared(sigmod_op);
 *  SigmoidLayer sig1 = std::make_shared(sigmod_op1);
    layers.append(conv1)
    layers.append(conv2)
    layers.append(relu1)
    layers.append(sig)
    layers.append(sig1)

 */

TEST(test_layer, forward_relu1) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f; 
  input->index(1) = -2.f; 
  input->index(2) = 3.f; 

  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  ReluLayer layer(relu_op);
  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}

// 有了注册机制后的框架是如何init layer
TEST(test_layer, forward_relu2) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);
  std::shared_ptr<Layer> relu_layer = LayerRegisterer::CreateLayer(relu_op);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  relu_layer->Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}