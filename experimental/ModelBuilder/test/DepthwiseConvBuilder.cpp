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

#include "experimental/ModelBuilder/test/DepthwiseConvBuilder.h"

using namespace mlir;                    // NOLINT
using namespace mlir::edsc;              // NOLINT
using namespace mlir::edsc::intrinsics;  // NOLINT

namespace mlir {

template <bool VECTORIZE>
ModelBuilder buildDepthwiseConv(unsigned N, unsigned Ho, unsigned Wo,
                                unsigned Co, unsigned H, unsigned W,
                                unsigned Ci, unsigned Kh, unsigned Kw) {
  ModelBuilder modelBuilder;
  const std::string funcName =
      VECTORIZE ? "depthwise-conv-vectorized" : "depthwise-conv-loopnest";
  auto f32 = modelBuilder.f32;
  auto inputType = modelBuilder.getMemRefType({N, H, W, Ci}, f32);
  auto filterType = modelBuilder.getMemRefType({Kh, Kw, Ci, Co}, f32);
  auto outputType = modelBuilder.getMemRefType({N, Ho, Wo, Ci * Co}, f32);

  constexpr int vecSize = 16;

  SmallVector<AffineExpr, 4> inputMaps, outputMaps;
  inputMaps.push_back(modelBuilder.getAffineDimExpr(0) +
                      modelBuilder.getAffineDimExpr(1));
  outputMaps.push_back(modelBuilder.getAffineDimExpr(0) * Co +
                       modelBuilder.getAffineDimExpr(1));
  auto mapReader = AffineMap::get(2, 0, inputMaps, modelBuilder.getContext());
  auto mapOutput = AffineMap::get(2, 0, outputMaps, modelBuilder.getContext());
  {
    auto func = modelBuilder.makeFunction(
        funcName, {}, {inputType, filterType, outputType},
        MLIRFuncOpConfig().setEmitCInterface(true));
    OpBuilder b(&func.getBody());
    ScopedContext scope(b, func.getLoc());
    Value zero = std_constant_index(0);
    Value step = std_constant_index(1);
    Value vStep = std_constant_index(vecSize);
    Value boundN = std_constant_index(N);
    Value boundHo = std_constant_index(Ho);
    Value boundWo = std_constant_index(Wo);
    Value boundCi = std_constant_index(Ci);
    Value boundCo = std_constant_index(Co);
    Value boundKh = std_constant_index(Kh);
    Value boundKw = std_constant_index(Kw);
    auto vType = modelBuilder.getVectorType({vecSize}, f32);
    loopNestBuilder(zero, boundN, step, [&](Value n) {
      loopNestBuilder(zero, boundHo, step, [&](Value ho) {
        loopNestBuilder(zero, boundWo, step, [&](Value wo) {
          loopNestBuilder(zero, boundKh, step, [&](Value kh) {
            loopNestBuilder(zero, boundKw, step, [&](Value kw) {
              loopNestBuilder(zero, boundCi, step, [&](Value ci) {
                Value input = func.getArgument(0);
                Value hIndex = affine_apply(mapReader, ArrayRef<Value>{ho, kh});
                Value wIndex = affine_apply(mapReader, ArrayRef<Value>{wo, kw});
                Value x =
                    std_load(input, ArrayRef<Value>{n, hIndex, wIndex, ci});
                if (VECTORIZE) x = vector_broadcast(vType, x);
                loopNestBuilder(
                    zero, boundCo, VECTORIZE ? vStep : step, [&](Value co) {
                      Value filter = func.getArgument(1);
                      Value output = func.getArgument(2);
                      Value coIndex =
                          affine_apply(mapOutput, ArrayRef<Value>{ci, co});
                      Value w =
                          VECTORIZE
                              ? (Value)vector_transfer_read(
                                    vType, filter,
                                    ArrayRef<Value>{kh, kw, ci, co})
                              : (Value)std_load(
                                    filter, ArrayRef<Value>{kh, kw, ci, co});
                      Value y = VECTORIZE
                                    ? (Value)vector_transfer_read(
                                          vType, output,
                                          ArrayRef<Value>{n, ho, wo, coIndex})
                                    : (Value)std_load(
                                          output,
                                          ArrayRef<Value>{n, ho, wo, coIndex});
                      Value result = VECTORIZE
                                         ? (Value)vector_fma(x, w, y)
                                         : (Value)std_addf(std_mulf(x, w), y);
                      if (VECTORIZE) {
                        vector_transfer_write(
                            result, output,
                            ArrayRef<Value>{n, ho, wo, coIndex});
                      } else {
                        std_store(result, output,
                                  ArrayRef<Value>{n, ho, wo, coIndex});
                      }
                    });
              });
            });
          });
        });
      });
    });
    std_ret();
  }
  return modelBuilder;
}

ModelBuilder buildDepthwiseConvLoops(unsigned N, unsigned Ho, unsigned Wo,
                                     unsigned Co, unsigned H, unsigned W,
                                     unsigned Ci, unsigned Kh, unsigned Kw) {
  return buildDepthwiseConv<false>(N, Ho, Wo, Co, H, W, Ci, Kh, Kw);
}
ModelBuilder buildDepthwiseConvVector(unsigned N, unsigned Ho, unsigned Wo,
                                      unsigned Co, unsigned H, unsigned W,
                                      unsigned Ci, unsigned Kh, unsigned Kw) {
  return buildDepthwiseConv<true>(N, Ho, Wo, Co, H, W, Ci, Kh, Kw);
}

}  // namespace mlir
