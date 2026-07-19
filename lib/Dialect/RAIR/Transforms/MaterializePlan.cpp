//===- MaterializePlan.cpp - Build a RAIR Plan task DAG ------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPasses.h"
#include "torch-mlir/Dialect/RAIR/Transforms/RAIRPlanAnalysis.h"

#include "torch-mlir/Dialect/RAIR/IR/RAIROps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

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

static LogicalResult
validateExistingPlan(const rair::RAIRStaticEffectGraph &coreGraph,
                     rair::PlanOp plan) {
  FailureOr<rair::RAIRPlanGraph> planGraph = rair::RAIRPlanGraph::build(plan);
  if (failed(planGraph))
    return plan.emitOpError(
        "could not build a Plan graph from structurally verified IR");
  std::optional<rair::PlanValidationIssue> issue =
      planGraph->validateAgainst(coreGraph);
  if (issue)
    return emitPlanValidationIssue(*issue);
  return success();
}

struct PlanTaskSpec {
  Location location;
  rair::TaskKind kind;
  SmallVector<unsigned> dependencies;
};

struct PendingPlan {
  rair::ScopeOp scope;
  SmallVector<PlanTaskSpec> tasks;
};

static PendingPlan createPendingPlan(const rair::RAIRStaticEffectGraph &graph) {
  PendingPlan pending{graph.getScope(), {}};
  rair::StaticPlanDependencies dependencies =
      rair::computeStaticTransitiveReduction(graph);
  for (auto [actionId, action] : llvm::enumerate(graph.getActions())) {
    PlanTaskSpec task{action.operation->getLoc(),
                      rair::getTaskKindForStaticAction(action.kind),
                      dependencies[actionId]};
    pending.tasks.push_back(std::move(task));
  }
  return pending;
}

static void materializePlan(const PendingPlan &pending) {
  rair::ScopeOp scope = pending.scope;
  OpBuilder builder(scope.getContext());
  builder.setInsertionPoint(scope.getBody().front().getTerminator());
  auto plan =
      builder.create<rair::PlanOp>(scope.getLoc(), scope.getTargetAttr());
  builder.createBlock(&plan.getBody());
  builder.setInsertionPointToEnd(&plan.getBody().front());

  auto eventType =
      rair::EventType::get(scope.getContext(), scope.getTargetAttr());
  SmallVector<Value> events;
  events.reserve(pending.tasks.size());
  for (auto [actionId, task] : llvm::enumerate(pending.tasks)) {
    SmallVector<Value> dependencies;
    dependencies.reserve(task.dependencies.size());
    for (unsigned dependency : task.dependencies)
      dependencies.push_back(events[dependency]);
    auto materializedTask = builder.create<rair::TaskOp>(
        task.location, eventType, dependencies, task.kind, actionId);
    events.push_back(materializedTask.getEvent());
  }
  builder.create<rair::PlanTerminatorOp>(scope.getLoc());
}

class MaterializePlanPass
    : public PassWrapper<MaterializePlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializePlanPass)

  StringRef getArgument() const final { return "rair-materialize-plan"; }
  StringRef getDescription() const final {
    return "Materialize raw correctness Plans for RAIR Core scopes";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<rair::RAIRDialect>();
  }

  void runOnOperation() final {
    SmallVector<rair::ScopeOp> scopes;
    getOperation().walk([&](rair::ScopeOp scope) { scopes.push_back(scope); });

    SmallVector<PendingPlan> pendingPlans;
    for (rair::ScopeOp scope : scopes) {
      FailureOr<rair::RAIRStaticEffectGraph> graph =
          rair::RAIRStaticEffectGraph::build(scope);
      if (failed(graph)) {
        scope.emitError("could not build a static effect graph from verified "
                        "Core IR");
        signalPassFailure();
        return;
      }

      if (rair::PlanOp plan = findAssociatedPlan(scope)) {
        if (failed(validateExistingPlan(*graph, plan))) {
          signalPassFailure();
          return;
        }
        continue;
      }
      pendingPlans.push_back(createPendingPlan(*graph));
    }

    if (pendingPlans.empty()) {
      markAllAnalysesPreserved();
      return;
    }
    for (const PendingPlan &pending : pendingPlans)
      materializePlan(pending);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> rair::createMaterializePlanPass() {
  return std::make_unique<MaterializePlanPass>();
}
