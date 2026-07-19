//===- VerifyPlan.cpp - Verify RAIR Plan correctness reachability --------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPasses.h"
#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPlanAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static rair::PlanOp findAssociatedPlan(rair::ScopeOp scope) {
  for (Operation &operation : scope.getBody().front())
    if (auto plan = dyn_cast<rair::PlanOp>(operation))
      return plan;
  return {};
}

static LogicalResult
emitPlanValidationIssue(const rair::PlanValidationIssue &issue) {
  if (auto mismatch = std::get_if<rair::PlanTaskCountMismatch>(&issue))
    return rair::PlanOp(mismatch->plan).emitOpError()
           << "has " << mismatch->actualTaskCount
           << " tasks, but its source Core graph has "
           << mismatch->expectedActionCount << " actions";
  if (auto mismatch = std::get_if<rair::PlanActionKindMismatch>(&issue))
    return rair::TaskOp(mismatch->task).emitOpError()
           << "kind " << rair::TaskOp(mismatch->task).getKindAttr()
           << " does not match source_action " << mismatch->sourceAction
           << " kind " << rair::stringifyTaskKind(mismatch->expectedKind);
  const auto &missing = std::get<rair::PlanMissingCorrectnessPath>(issue);
  return rair::PlanOp(missing.plan).emitOpError()
         << "does not preserve required Core correctness path from "
            "source_action "
         << missing.from << " to source_action " << missing.to;
}

class VerifyPlanPass
    : public PassWrapper<VerifyPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyPlanPass)

  StringRef getArgument() const final { return "rair-verify-plan"; }
  StringRef getDescription() const final {
    return "Verify RAIR Plan action mapping and correctness reachability";
  }

  void runOnOperation() final {
    bool foundFailure = false;
    getOperation().walk([&](rair::ScopeOp scope) {
      FailureOr<rair::RAIRStaticEffectGraph> coreGraph =
          rair::RAIRStaticEffectGraph::build(scope);
      if (failed(coreGraph)) {
        scope.emitError("could not build a static effect graph from verified "
                        "Core IR");
        foundFailure = true;
        return;
      }

      rair::PlanOp plan = findAssociatedPlan(scope);
      if (!plan) {
        scope.emitOpError("has no associated rair.plan to verify");
        foundFailure = true;
        return;
      }

      FailureOr<rair::RAIRPlanGraph> planGraph =
          rair::RAIRPlanGraph::build(plan);
      if (failed(planGraph)) {
        plan.emitOpError(
            "could not build a Plan graph from structurally verified IR");
        foundFailure = true;
        return;
      }

      std::optional<rair::PlanValidationIssue> issue =
          planGraph->validateAgainst(*coreGraph);
      if (issue) {
        (void)emitPlanValidationIssue(*issue);
        foundFailure = true;
      }
    });

    if (foundFailure) {
      signalPassFailure();
      return;
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> rair::createVerifyPlanPass() {
  return std::make_unique<VerifyPlanPass>();
}
