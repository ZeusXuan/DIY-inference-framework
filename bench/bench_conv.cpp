#include <benchmark/benchmark.h>
#include "../source/layer/details/backup/winograd.hpp"
#include "../source/layer/details/convolution.hpp"
#include "../source/layer/details/deconvolution.hpp"
#include "runtime/runtime_ir.hpp"

static void BM_Convolution(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t kernel_count = state.range(0);
  uint32_t channels = state.range(1);
  uint32_t rows = state.range(2);
  uint32_t cols = state.range(3);

  uint32_t kernel_h = state.range(4);
  uint32_t kernel_w = state.range(5);
  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(channels, kernel_h, kernel_w);
    weight->RandN();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, channels, kernel_h, kernel_w, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Convolution)->Args({32, 3, 320, 320, 3, 3})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({64, 32, 160, 160, 3, 3})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({128, 64, 80, 80, 3, 3})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({256, 128, 40, 40, 3, 3})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({512, 256, 20, 20, 3, 3})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({32, 3, 320, 320, 1, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({64, 32, 160, 160, 1, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({128, 64, 80, 80, 1, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({256, 128, 40, 40, 1, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolution)->Args({512, 256, 20, 20, 1, 1})->Unit(benchmark::kMillisecond);

static void BM_DeConvolutionk2x2s2x2(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t kernel_count = state.range(0);
  uint32_t channels = state.range(1);
  uint32_t rows = state.range(2);
  uint32_t cols = state.range(3);
  uint32_t batch = state.range(4);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<float> weight_values;
  for (uint32_t k = 0; k < kernel_count * channels * 3 * 3; ++k) {
    weight_values.push_back(float(k % 31));
  }

  std::vector<sftensor> outputs(batch);
  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch; ++i) {
    inputs.push_back(input);
  }
  DeconvolutionLayer conv_layer(kernel_count, channels, 3, 3, 0, 0, 2, 2, 1, true);
  conv_layer.set_weights(weight_values);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_DeConvolutionk2x2s2x2g4(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t kernel_count = state.range(0);
  uint32_t channels = state.range(1);
  uint32_t rows = state.range(2);
  uint32_t cols = state.range(3);
  uint32_t batch = state.range(4);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<float> weight_values;
  for (uint32_t k = 0; k < kernel_count * channels / 4 * 3 * 3; ++k) {
    weight_values.push_back(float(k % 31));
  }

  std::vector<sftensor> outputs(batch);
  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch; ++i) {
    inputs.push_back(input);
  }
  DeconvolutionLayer conv_layer(kernel_count, channels, 3, 3, 0, 0, 2, 2, 4, true);
  conv_layer.set_weights(weight_values);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_DeConvolutionk2x2s2x2)->Args({64, 128, 192, 192, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_DeConvolutionk2x2s2x2)->Args({128, 256, 96, 96, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_DeConvolutionk2x2s2x2)->Args({256, 512, 48, 48, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_DeConvolutionk2x2s2x2)->Args({512, 1024, 24, 24, 1})->Unit(benchmark::kMillisecond);

/////////////////////////////////////////
BENCHMARK(BM_DeConvolutionk2x2s2x2g4)->Args({64, 128, 192, 192, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_DeConvolutionk2x2s2x2g4)->Args({128, 256, 96, 96, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_DeConvolutionk2x2s2x2g4)->Args({256, 512, 48, 48, 1})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_DeConvolutionk2x2s2x2g4)->Args({512, 1024, 24, 24, 1})->Unit(benchmark::kMillisecond);