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

//===- IREEToSPIRVPass.h ---------------------------------------*- C++//-*-===//
//
// Pass to translate iree executables for vulkan-spirv.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_IREETOSPIRVPASS_H_
#define IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_IREETOSPIRVPASS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

/// Generates a spirv::ModuleOp from the module within an IREE Executable with
/// target-config vulkan-spirv.
std::unique_ptr<OperationPass<ModuleOp>> createIREEToSPIRVPass();

/// Adds all the passes needed to lower dispatch function to SPIR-V
void addIREEToSPIRVPasses(OpPassManager &conversionPassManager);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_IREETOSPIRVPASS_H_