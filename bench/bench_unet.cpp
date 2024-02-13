#include <benchmark/benchmark.h>
#include "runtime/runtime_ir.hpp"

const static int kIterationNum = 4;

static void BM_Unet_Batch1_512x512(benchmark::State& state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/unet/unet_demo.pnnx.param", "tmp/unet/unet_demo.pnnx.bin");
  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 512, 512);
    input->RandN();
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  for (auto _ : state) {
    graph.Forward(false);
  }
}

BENCHMARK(BM_Unet_Batch1_512x512)->Unit(benchmark::kMillisecond)->Iterations(kIterationNum);