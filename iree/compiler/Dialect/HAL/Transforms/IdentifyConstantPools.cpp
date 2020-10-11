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

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class IdentifyConstantPoolsPass
    : public PassWrapper<IdentifyConstantPoolsPass, OperationPass<ModuleOp>> {
 public:
  IdentifyConstantPoolsPass() : targetOptions_(getTargetOptionsFromFlags()) {}
  explicit IdentifyConstantPoolsPass(TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Derive buffer constraints based on target backends.
    auto bufferConstraints = computeConservativeBufferConstraints(
        targetOptions_, moduleOp.getContext());

    // Gather constant variables. We assume that prior passes/pipelines have
    // hoisted anything worth pooling to flow.variables at the module scope.
    // We expect that immutable variables have already been de-duped and that
    // mutable variables that remain may have identical initializers.
    SmallVector<IREE::Flow::VariableOp, 16> mutableOps;
    SmallVector<IREE::Flow::VariableOp, 16> immutableOps;
    for (auto variableOp : moduleOp.getOps<IREE::Flow::VariableOp>()) {
      if (!variableOp.initial_value().hasValue()) continue;
      auto variableType = variableOp.type().dyn_cast<RankedTensorType>();
      if (!variableType) continue;
      if (variableOp.is_mutable()) {
        mutableOps.push_back(variableOp);
      } else {
        immutableOps.push_back(variableOp);
      }
    }

    SymbolTable moduleSymbolTable(moduleOp);
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

    // Process the mutable ops where each constant is only used as an
    // initializer. The lifetime of these is short as we only use them to
    // populate the initial variable buffer contents.
    makeConstantPool("_const_pool_init", mutableOps, bufferConstraints,
                     moduleSymbolTable, moduleBuilder);

    // Process the immutable ops where the same buffer will be used for the
    // lifetime of the module.
    makeConstantPool("_const_pool", immutableOps, bufferConstraints,
                     moduleSymbolTable, moduleBuilder);

    // NOTE: pools now contain the values but they are in an undefined order.
    // We should have following passes that reorder the values to cluster them
    // by usage time locality so that there's a better chance of them landing
    // in the same runtime buffers and prefetched mapped storage pages.
  }

 private:
  // Tries to find the min/max constraints on buffers across all target
  // backends. This should really be done per pool based on the usage of the
  // constants (if pool 0 is used by device A and pool 1 is used by device B
  // then they should not need to have matching constraints).
  BufferConstraintsAttr computeConservativeBufferConstraints(
      const TargetOptions &targetOptions, MLIRContext *context) {
    auto targetBackends = matchTargetBackends(targetOptions.targets);
    assert(!targetBackends.empty());
    BufferConstraintsAttr attr = {};
    for (auto &targetBackend : targetBackends) {
      if (attr) {
        attr = intersectBufferConstraints(
            attr, targetBackend->queryBufferConstraints(context));
      } else {
        attr = targetBackend->queryBufferConstraints(context);
      }
    }
    return attr;
  }

  // Makes a new hal.constant_pool containing the values of the given
  // variable ops. The variables will be erased and all variable loads will be
  // replaced with constant loads. Returns the constant pool, if it was created.
  Optional<ConstantPoolOp> makeConstantPool(
      StringRef poolName, ArrayRef<IREE::Flow::VariableOp> variableOps,
      BufferConstraintsAttr bufferConstraints, SymbolTable &moduleSymbolTable,
      OpBuilder &moduleBuilder) {
    // Create the pool to be filled with constant values.
    auto poolOp = OpBuilder(moduleBuilder.getContext())
                      .create<ConstantPoolOp>(moduleBuilder.getUnknownLoc(),
                                              poolName, bufferConstraints);
    moduleSymbolTable.insert(poolOp, moduleBuilder.getInsertionPoint());
    SymbolTable::setSymbolVisibility(poolOp, SymbolTable::Visibility::Private);

    auto poolBuilder = OpBuilder::atBlockBegin(poolOp.getBody());
    for (auto variableOp : variableOps) {
      // Grab the constant value from the variable that we'll be pooling.
      auto value = variableOp.initial_value()
                       .getValue()
                       .dyn_cast_or_null<ElementsAttr>();
      assert(value && "value precondition not met: must be elements attr");

      // Create the constant in the pool.
      auto valueOp = poolBuilder.create<ConstantPoolValueOp>(
          variableOp.getLoc(), variableOp.getName(), value);
      SymbolTable::setSymbolVisibility(valueOp,
                                       SymbolTable::Visibility::Nested);

      // Find all variable loads in the module.
      // May fail if the variable is used in ways not supported by constant
      // pools.
      auto loadOps = findVariableLoadOps(variableOp);

      // If the variable is an immutable constant and used in compatible ways
      // we can turn them into constant loads instead. These will avoid the
      // additional runtime overhead of variable lifetime tracking and allow
      // further optimizations at use sites where we know the values come from
      // constant memory.
      if (loadOps.hasValue() && !variableOp.is_mutable()) {
        // Replace all loads of the variable with loads of the constant.
        for (auto loadOp : loadOps.getValue()) {
          replaceVariableLoadWithConstantLoad(loadOp, valueOp);
        }
        variableOp.erase();
      } else {
        // Build an initializer function to populate the variable with the
        // constant value on startup.
        changeToVariableInitializerFunc(variableOp, valueOp);
      }
    }

    // Remove the pool if it didn't end up with any constants.
    if (poolOp.getBody()->front().isKnownTerminator()) {
      poolOp.erase();
      return None;
    }
    return poolOp;
  }

  // Constructs a function that can be used as an initializer for a variable
  // and inserts it by the variable op in the module.
  FuncOp changeToVariableInitializerFunc(
      IREE::Flow::VariableOp variableOp,
      IREE::HAL::ConstantPoolValueOp valueOp) {
    // Create the function and make the variable point to it for init.
    OpBuilder moduleBuilder(variableOp.getContext());
    moduleBuilder.setInsertionPointAfter(variableOp);
    auto initializerName = (variableOp.getName() + "_initializer").str();
    auto initializerFunc = moduleBuilder.create<FuncOp>(
        variableOp.getLoc(), initializerName,
        moduleBuilder.getFunctionType({}, {variableOp.type()}));
    SymbolTable::setSymbolVisibility(initializerFunc,
                                     SymbolTable::Visibility::Private);
    variableOp.removeAttr("initial_value");
    variableOp.setAttr("initializer",
                       moduleBuilder.getSymbolRefAttr(initializerFunc));

    // Emit a constant load that will later on be turned into a runtime buffer
    // reference.
    auto funcBuilder = OpBuilder::atBlockBegin(initializerFunc.addEntryBlock());
    auto constValue = funcBuilder.createOrFold<IREE::HAL::ConstantPoolLoadOp>(
        variableOp.getLoc(), variableOp.type(),
        funcBuilder.getSymbolRefAttr(
            valueOp.getParentOfType<ConstantPoolOp>().getName(),
            {funcBuilder.getSymbolRefAttr(valueOp)}));
    funcBuilder.create<mlir::ReturnOp>(variableOp.getLoc(), constValue);

    return initializerFunc;
  }

  // Returns a list of all load ops referencing |variableOp| or None if the
  // variable has undefined/unsupported uses that prevent it from being pooled.
  Optional<SmallVector<IREE::Flow::VariableLoadOp, 8>> findVariableLoadOps(
      IREE::Flow::VariableOp variableOp) {
    // Find all uses of the variable within the module.
    auto variableUses =
        SymbolTable::getSymbolUses(variableOp, variableOp.getParentRegion());
    if (!variableUses.hasValue()) {
      // Cannot turn this variable into a constant as its used in undefined
      // places.
      variableOp.emitWarning()
          << "symbol may be used in an undefined op; ignoring";
      return None;
    }

    SmallVector<IREE::Flow::VariableLoadOp, 8> loadOps;
    for (auto variableUse : variableUses.getValue()) {
      if (isa<IREE::Flow::VariableAddressOp>(variableUse.getUser())) {
        variableOp.emitWarning() << "variable is used indirectly; currently "
                                    "unsupported for constant pooling";
        return None;
      } else if (auto loadOp = dyn_cast<IREE::Flow::VariableLoadOp>(
                     variableUse.getUser())) {
        loadOps.push_back(loadOp);
      }
    }
    return loadOps;
  }

  // Replaces a flow.variable.load with a hal.constant_pool.load of a pooled
  // value.
  void replaceVariableLoadWithConstantLoad(
      IREE::Flow::VariableLoadOp variableLoadOp, ConstantPoolValueOp valueOp) {
    OpBuilder builder(variableLoadOp);
    auto loadOp = builder.create<ConstantPoolLoadOp>(
        variableLoadOp.getLoc(), variableLoadOp.getType(),
        builder.getSymbolRefAttr(
            valueOp.getParentOfType<ConstantPoolOp>().getName(),
            {builder.getSymbolRefAttr(valueOp)}));
    variableLoadOp.replaceAllUsesWith(loadOp.result());
    variableLoadOp.erase();
  }

  TargetOptions targetOptions_;
};

std::unique_ptr<OperationPass<ModuleOp>> createIdentifyConstantPoolsPass(
    TargetOptions targetOptions) {
  return std::make_unique<IdentifyConstantPoolsPass>(targetOptions);
}

static PassRegistration<IdentifyConstantPoolsPass> pass(
    "iree-hal-identify-constant-pools",
    "Combines constant variables into one or more hal.constant_pools based on "
    "usage semantics.");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
