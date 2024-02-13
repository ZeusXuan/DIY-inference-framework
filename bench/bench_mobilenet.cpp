#include <benchmark/benchmark.h>
#include "runtime/runtime_ir.hpp"

static void BM_MobilenetV3_Batch8_224x224(benchmark::State& state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/mobilenet/mobile_batch8.pnnx.param", "tmp/mobilenet/mobile_batch8.bin");

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

BENCHMARK(BM_MobilenetV3_Batch8_224x224)->Unit(benchmark::kMillisecond);
