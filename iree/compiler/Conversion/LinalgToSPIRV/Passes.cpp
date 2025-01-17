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

//===- Passes.cpp - Pipeline from HLO to Linalg to SPIR-V -----------------===//
//
// Implementation of conversion from XLA-HLO to Linalg to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"

#include "iree/compiler/Conversion/HLOToHLO/Passes.h"
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Conversion/LinalgToVector/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Linalg to SPIR-V pass pipeline options. In theory this is a superset of all
/// options in all passes in the pipeline. Adding those based on need, and they
/// should be needed only for testing.
struct LinalgToSPIRVPassPipelineOptions
    : public PassPipelineOptions<LinalgToSPIRVPassPipelineOptions> {
  Option<bool> useVectorization{
      *this, "use-vectorization",
      llvm::cl::desc(
          "Enable use of vectorization in SPIR-V code generation pipeline"),
      llvm::cl::init(false)};
  Option<bool> useVectorPass{
      *this, "use-vector-pass",
      llvm::cl::desc("Enable use of Linalg vectorization in SPIR-V code "
                     "generation pipeline"),
      llvm::cl::init(false)};
};

static void addLinalgToSPIRVPasses(OpPassManager &pm,
                                   const SPIRVCodegenOptions &options) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  //===--------------------------------------------------------------------===//
  // Tile Linalg on buffers.
  //
  // Pre-conditions:
  //   - All Linalg ops have buffer semantics.
  //
  // Post-conditions:
  //   - If the input Linalg ops are tilable:
  //     - loop.parallel ops are generated for mapping to workgroups.
  //     - Linalg ops are nested inside loop.parallel ops and ready for mapping
  //       to workitems.
  //   - Otherwise:
  //     - The Linalg op is kept untouched.
  //   - Dispatch functions might be split into multiple ones.
  //
  // Note:
  //   We first try to tile and fuse the dispatch function as a whole. If there
  //   are multiple Linalg ops inside, they may not share any number of common
  //   outer parallel iterators. Then the first tile and fuse pass will do
  //   nothing. We split all the Linalg ops into their own dispatch functions
  //   afterwards. This gives each Linalg op a second chance to be tiled,
  //   with the second tile and fuse pass.
  //===--------------------------------------------------------------------===//
  pm.addPass(createSplitDispatchFunctionPass());
  pm.addPass(createLinalgTileAndFusePass(options));
  if (options.useVectorPass) {
    pm.addPass(createLoadStoreVectorizationPass());
  }
  pm.addPass(createCanonicalizerPass());

  //===--------------------------------------------------------------------===//
  // Map to GPU processor IDs.
  //
  // Post-conditions:
  //   - loop.parallel ops are converted to loop.for ops and mapped to
  //     workgroups.
  //   - Linalg ops are converted to loop.for ops and mapped to workitems.
  //===--------------------------------------------------------------------===//
  pm.addPass(createConvertToGPUPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  //===--------------------------------------------------------------------===//
  // Legalize the function that computes the number of workgroups to be runnable
  // on the host.
  //
  // Post-conditions:
  //   - The shape of the values created from `iree.placeholder` operations are
  //     tied to the arguments of the function.
  //===--------------------------------------------------------------------===//
  pm.addPass(createLegalizeNumWorkgroupsFnPass());

  //===--------------------------------------------------------------------===//
  // Resolve shape related ops.
  //
  // Pre-conditions:
  //   - All dynamic tensors bridge through a shapex.tie_shape op with the
  //     appropriate shape.
  //   - No shapex.get_ranked_shape ops exist.
  //   - Shape folding and canonicalization has been done.
  // Post-conditions:
  //   - shapex.tie_shape and other shapex ops are all converted away.
  //   - std.dim ops are traced back and replaced by the corresponding
  //     hal.inteface.load.constant op. There are no std.dim ops left
  //     in the IR.
  //===--------------------------------------------------------------------===//
  pm.addPass(createResolveShapeOpsPass());

  //===--------------------------------------------------------------------===//
  // Legalize the function that computes the number of workgroups to be runnable
  // on the host.
  //
  // Post-conditions:
  //   - The dead `iree.placeholder` operations are removed after shape
  //     resolution.
  //===--------------------------------------------------------------------===//
  pm.addPass(createLegalizeNumWorkgroupsFnPass());

  //===--------------------------------------------------------------------===//
  // Prepare stdandard ops for SPIR-V conversion.
  //
  // Post-conditions:
  //   - Load/store on std.subview ops are converted into load/store on the
  //     original buffers.
  //===--------------------------------------------------------------------===//
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  //===--------------------------------------------------------------------===//
  // Final conversion to SPIR-V dialect.
  //
  // Post-conditions:
  //   - All ops are converted to SPIR-V counterparts.
  //   - spv.module ops are formed to hold all SPIR-V ops.
  //===--------------------------------------------------------------------===//
  pm.addPass(createConvertToSPIRVPass());

  //===--------------------------------------------------------------------===//
  // SPIR-V dialect level conversions.
  //
  // Post-conditions:
  //   - SPIR-V Entry point ops are inserted.
  //   - Required version/extension/capability are deduced.
  //===--------------------------------------------------------------------===//
  OpPassManager &spirvModulePM = pm.nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(createCanonicalizerPass());
  spirvModulePM.addPass(createCSEPass());
  spirvModulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
}
}  // namespace

void buildSPIRVTransformPassPipeline(OpPassManager &pm,
                                     const SPIRVCodegenOptions &options) {
  //===--------------------------------------------------------------------===//
  // The entry point functions call an _impl function that captures the ABI that
  // the host side uses for the dispatch region. This ABI is needed when
  // generating the function that computes the number of workgroups. Declare the
  // function that returns the number of workgroups needed for an entry point
  // function.
  //
  // Post-conditions

  //   - An empty, private function is defined for each entry point function
  //     that returns the number of workgroups.
  //   - The entry point function gets an attribute `vkspv.num_workgroups_fn` to
  //     record which function in the module returns the number of workgroups.
  pm.addPass(createDeclareNumWorkgroupsFnPass());

  //===--------------------------------------------------------------------===//
  // Inline the impl dispatch function into the wrapper dispatch function.
  //
  // TODO(antiagainst): re-evaluate the inlining timing.
  //===--------------------------------------------------------------------===//
  pm.addPass(createInlinerPass());

  //===--------------------------------------------------------------------===//
  // Inject shape calculation for output buffers.
  //
  // Pre-conditions:
  //   - All transformations altering the tensor-level shapes have been done.
  //   - "Root" dynamic tensors all pass through a single shapex.tie_shape
  //     use which associates them to their shape.
  //   - Loose, non-associated shapex.get_ranked_shape ops can exist anywhere
  //     and will be resolved.
  // Post-conditions:
  //   - All dynamic tensors bridge through a shapex.tie_shape op with the
  //     appropriate shape.
  //   - No shapex.get_ranked_shape ops exist.
  //   - Shape folding and canonicalization has been done.
  //===--------------------------------------------------------------------===//
  pm.addNestedPass<FuncOp>(Shape::createTieDynamicShapesPass());
  pm.addNestedPass<FuncOp>(Shape::createMaterializeShapeCalculationsPass());
  pm.addNestedPass<FuncOp>(Shape::createHoistShapeCalculationsPass());

  //===--------------------------------------------------------------------===//
  // Convert XLA HLO ops to Linalg ops with buffer semantics.
  //
  // Post-conditions:
  //   - All XLA HLO ops are converted.
  //   - All Linalg ops are operating on buffers.
  //===--------------------------------------------------------------------===//
  pm.addPass(createDecomposeHLOClampPass());
  addHLOToLinalgOnBuffersPasses(pm);

  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to SPIR-V ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final spv.module ready for serialization.
  //===--------------------------------------------------------------------===//
  addLinalgToSPIRVPasses(pm, options);
}

static SPIRVCodegenOptions initializeCodegenOptions(
    const LinalgToSPIRVPassPipelineOptions &options) {
  SPIRVCodegenOptions codegenOptions;
  codegenOptions.useVectorization = options.useVectorization;
  return codegenOptions;
}

static PassPipelineRegistration<LinalgToSPIRVPassPipelineOptions>
    linalgToSPIRVPipeline(
        "iree-codegen-linalg-to-spirv-pipeline",
        "Runs the progressive lowering pipeline from Linalg to SPIR-V",
        [](OpPassManager &passManager,
           const LinalgToSPIRVPassPipelineOptions &options) {
          addLinalgToSPIRVPasses(passManager,
                                 initializeCodegenOptions(options));
        });

static PassPipelineRegistration<LinalgToSPIRVPassPipelineOptions>
    hloToLinalgSPIRVPipeline(
        "iree-codegen-hlo-to-spirv-pipeline",
        "Runs the progressive lowering pipeline from XLA HLO to Linalg to "
        "SPIR-V",
        [](OpPassManager &passManager,
           const LinalgToSPIRVPassPipelineOptions &options) {
          buildSPIRVTransformPassPipeline(passManager,
                                          initializeCodegenOptions(options));
        });

}  // namespace iree_compiler
}  // namespace mlir
