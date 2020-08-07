// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "benchmark/benchmark.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "experimental/ModelBuilder/test/DepthwiseConvBuilder.h"

using namespace mlir;

void BM_DepthwiseConvVectors(benchmark::State &state) {
  constexpr unsigned N = 1;
  constexpr unsigned H = 224;
  constexpr unsigned W = 224;
  constexpr unsigned Ci = 4;
  constexpr unsigned Co = 32;
  constexpr unsigned Kh = 4;
  constexpr unsigned Kw = 4;
  constexpr unsigned Ho = H - Kh + 1;
  constexpr unsigned Wo = W - Kw + 1;

  ModelBuilder modelBuilder =
      buildDepthwiseConvVector(N, Ho, Wo, Co, H, W, Ci, Kh, Kw);

  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(CompilationOptions());

  auto oneInit = [](unsigned idx, float *ptr) { ptr[idx] = 1.0 + idx % 4; };
  auto zeroInit = [](unsigned idx, float *ptr) { ptr[idx] = 0.0; };
  auto input =
      makeInitializedStridedMemRefDescriptor<float, 4>({N, H, W, Ci}, oneInit);
  auto filter = makeInitializedStridedMemRefDescriptor<float, 4>(
      {Kh, Kw, Ci, Co}, oneInit);

  auto output = makeInitializedStridedMemRefDescriptor<float, 4>(
      {N, Ho, Wo, Co * Ci}, zeroInit);

  auto funcToRun = "depthwise-conv-vectorized";
  auto err = runner.invoke(funcToRun, input, filter, output);
  if (err) llvm_unreachable("Error running function.");
  for (auto _ : state) {
    auto err = runner.invoke(funcToRun, input, filter, output);
    if (err) llvm_unreachable("Error running function.");
  }
}

void BM_DepthwiseConvLoops(benchmark::State &state) {
  constexpr unsigned N = 1;
  constexpr unsigned H = 224;
  constexpr unsigned W = 224;
  constexpr unsigned Ci = 4;
  constexpr unsigned Co = 32;
  constexpr unsigned Kh = 4;
  constexpr unsigned Kw = 4;
  constexpr unsigned Ho = H - Kh + 1;
  constexpr unsigned Wo = W - Kw + 1;

  ModelBuilder modelBuilder =
      buildDepthwiseConvLoops(N, Ho, Wo, Co, H, W, Ci, Kh, Kw);

  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(CompilationOptions());

  auto oneInit = [](unsigned idx, float *ptr) { ptr[idx] = 1.0 + idx % 4; };
  auto zeroInit = [](unsigned idx, float *ptr) { ptr[idx] = 0.0; };
  auto input =
      makeInitializedStridedMemRefDescriptor<float, 4>({N, H, W, Ci}, oneInit);
  auto filter = makeInitializedStridedMemRefDescriptor<float, 4>(
      {Kh, Kw, Ci, Co}, oneInit);

  auto output = makeInitializedStridedMemRefDescriptor<float, 4>(
      {N, Ho, Wo, Co * Ci}, zeroInit);

  auto funcToRun = "depthwise-conv-loopnest";
  auto err = runner.invoke(funcToRun, input, filter, output);
  if (err) llvm_unreachable("Error running function.");
  for (auto _ : state) {
    auto err = runner.invoke(funcToRun, input, filter, output);
    if (err) llvm_unreachable("Error running function.");
  }
}

int main(int argc, char **argv) {
  mlir::ModelBuilder::registerAllDialects();
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
}

BENCHMARK(BM_DepthwiseConvVectors)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_DepthwiseConvLoops)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
