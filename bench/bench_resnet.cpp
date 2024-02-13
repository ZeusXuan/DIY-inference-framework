#include <benchmark/benchmark.h>
#include "runtime/runtime_ir.hpp"

const static int kIterationNum = 5;

static void BM_Resnet18_Batch8_224x224(benchmark::State& state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/resnet18_batch8.pnnx.param",
                     "tmp/resnet/resnet18_batch8.pnnx.bin");

  graph.Build();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  const uint32_t batch_size = 8;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(1.);
    inputs.push_back(input1);
  }
  graph.set_inputs("pnnx_input_0", inputs);
  for (auto _ : state) {
    graph.Forward(false);
  }
}

static void BM_Resnet18_Batch16_224x224(benchmark::State& state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/resnet18_batch16.pnnx.param",
                     "tmp/resnet/resnet18_batch16.pnnx.bin");

  graph.Build();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t batch_size = 16;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(1.);
    inputs.push_back(input1);
  }
  graph.set_inputs("pnnx_input_0", inputs);
  for (auto _ : state) {
    graph.Forward(false);
  }
}

BENCHMARK(BM_Resnet18_Batch8_224x224)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Resnet18_Batch16_224x224)->Unit(benchmark::kMillisecond);