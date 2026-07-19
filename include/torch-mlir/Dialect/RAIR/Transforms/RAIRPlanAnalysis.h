//===- RAIRPlanAnalysis.h - Typed Plan DAG correctness ----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPLANANALYSIS_H
#define TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPLANANALYSIS_H

#include "torch-mlir/Dialect/RAIR/Transforms/RAIRStaticEffectGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <variant>

namespace rair {

TaskKind getTaskKindForStaticAction(StaticActionKind kind);

struct PlanTaskCountMismatch {
  PlanOp plan;
  unsigned expectedActionCount;
  unsigned actualTaskCount;
};

struct PlanActionKindMismatch {
  TaskOp task;
  unsigned sourceAction;
  TaskKind expectedKind;
};

struct PlanMissingCorrectnessPath {
  PlanOp plan;
  unsigned from;
  unsigned to;
};

using PlanValidationIssue =
    std::variant<PlanTaskCountMismatch, PlanActionKindMismatch,
                 PlanMissingCorrectnessPath>;

/// Typed graph view of one structurally verified rair.plan. Source action IDs
/// index tasks and adjacency. Build fails only when the Plan violates the
/// structural ODS/verifier contract.
class RAIRPlanGraph {
public:
  static mlir::FailureOr<RAIRPlanGraph> build(PlanOp plan);

  PlanOp getPlan() const { return plan; }
  llvm::ArrayRef<TaskOp> getTasksBySourceAction() const {
    return tasksBySourceAction;
  }
  llvm::ArrayRef<llvm::SmallVector<unsigned>> getPredecessors() const {
    return predecessors;
  }
  bool isReachable(unsigned from, unsigned to) const;

  /// Returns the first deterministic semantic mismatch with the source Core
  /// graph. Extra Plan edges are legal; every raw Core edge must be reachable.
  std::optional<PlanValidationIssue>
  validateAgainst(const RAIRStaticEffectGraph &coreGraph) const;

private:
  explicit RAIRPlanGraph(PlanOp plan) : plan(plan) {}

  PlanOp plan;
  llvm::SmallVector<TaskOp> tasksBySourceAction;
  llvm::SmallVector<llvm::SmallVector<unsigned>> predecessors;
  llvm::SmallVector<llvm::BitVector> reachability;
};

using StaticPlanDependencies = llvm::SmallVector<llvm::SmallVector<unsigned>>;

/// Computes the unique deterministic transitive reduction of the static Core
/// DAG. Returned entries are incoming source action IDs in ascending order.
StaticPlanDependencies
computeStaticTransitiveReduction(const RAIRStaticEffectGraph &graph);

} // namespace rair

#endif // TORCHMLIR_DIALECT_RAIR_TRANSFORMS_RAIRPLANANALYSIS_H
