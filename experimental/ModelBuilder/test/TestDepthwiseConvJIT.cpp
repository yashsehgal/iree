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

// clang-format off

// NOLINTNEXTLINE
// RUN: test-depthwise-conv -runtime-support=$(dirname %s)/runtime-support.so 2>&1 | IreeFileCheck %s

// clang-format on

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "experimental/ModelBuilder/test/DepthwiseConvBuilder.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

using namespace mlir;                    // NOLINT
using namespace mlir::edsc;              // NOLINT
using namespace mlir::edsc::intrinsics;  // NOLINT

static llvm::cl::opt<std::string> runtimeSupport(
    "runtime-support", llvm::cl::desc("Runtime support library filename"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

template <typename BuilderFunc>
void runDepthwiseConv(BuilderFunc builderFunc, std::string funcToRun) {
  // N,  H,  W, Ci
  // Kh, Kw, Ci, Co
  // N, Ho, Wo, Ci * Co
  constexpr unsigned N = 1;
  constexpr unsigned H = 32;
  constexpr unsigned W = 32;
  constexpr unsigned Ci = 4;
  constexpr unsigned Co = 8;
  constexpr unsigned Kh = 4;
  constexpr unsigned Kw = 4;
  constexpr unsigned Ho = H - Kh + 1;
  constexpr unsigned Wo = W - Kw + 1;

  ModelBuilder modelBuilder = builderFunc(N, Ho, Wo, Co, H, W, Ci, Kh, Kw);

  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(CompilationOptions(), runtimeSupport);

  auto oneInit = [](unsigned idx, float *ptr) { ptr[idx] = 1.0 + idx % 4; };
  auto zeroInit = [](unsigned idx, float *ptr) { ptr[idx] = 0.0; };
  auto input =
      makeInitializedStridedMemRefDescriptor<float, 4>({N, H, W, Ci}, oneInit);
  auto filter = makeInitializedStridedMemRefDescriptor<float, 4>(
      {Kh, Kw, Ci, Co}, oneInit);

  auto output = makeInitializedStridedMemRefDescriptor<float, 4>(
      {N, Ho, Wo, Co * Ci}, zeroInit);

  auto err = runner.invoke(funcToRun, input, filter, output);
  if (err) llvm_unreachable("Error running function.");

  ::impl::printMemRef(*output);
}

int main(int argc, char **argv) {
  ModelBuilder::registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TestDepthwiseConv\n");

  runDepthwiseConv(buildDepthwiseConvLoops, "depthwise-conv-loopnest");

  runDepthwiseConv(buildDepthwiseConvVector, "depthwise-conv-vectorized");
}
// CHECK: 16
