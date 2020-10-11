// Copyright 2019 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_FLOWTOHAL_CONVERTFLOWTOHAL_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_FLOWTOHAL_CONVERTFLOWTOHAL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Adds op legality rules to |conversionTarget| to ensure all incoming flow ops
// are removed during Flow->HAL lowering.
void setupFlowToHALLegality(MLIRContext *context,
                            ConversionTarget &conversionTarget,
                            TypeConverter &typeConverter);

// Populates conversion patterns for Flow->HAL.
void populateFlowToHALPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_CONVERSION_FLOWTOHAL_CONVERTFLOWTOHAL_H_
